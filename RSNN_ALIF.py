import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.encode    import SpikeLatencyLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell, LIParameters

class RSNN_ALIF(nn.Module):

    def __init__(self, n_in, n_rec, n_out, n_t, thr, tau_m, tau_o, b_o, gamma, dt, w_init_gain, rho, beta, lr_layer, device):

        super(RSNN_ALIF, self).__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.n_t = n_t
        self.thr = thr
        self.dt = dt
        self.alpha = 0.8
        self.kappa = 0.8
        self.gamma = gamma
        self.b_o = b_o
        self.beta = beta
        self.rho = rho
        self.refrac = 2
        self.ch = 0.025
        # bias critic
        self.b_co = 0.0
        self.lr_layer = lr_layer
        self.device = device
        self.n_b = 1
        self.encoder = ConstantCurrentLIFEncoder(n_t)
        # experiment with cv mnight give better results
        self.cv = 1
        self.rl_gamma = 0.998
        # Parameters
        self.critic = LILinearCell(self.n_rec, 1, p=LIParameters()).requires_grad_(False).to(self.device)
        self.w_in = nn.Parameter(torch.Tensor(n_rec, n_in))
        self.w_rec = nn.Parameter(torch.Tensor(n_rec, n_rec))
        self.w_out = nn.Parameter(torch.Tensor(n_out, n_rec))
        self.w_critic = nn.Parameter(torch.Tensor(1, n_rec))
        self.reg_term = torch.zeros(self.n_rec).to(self.device)
        self.B_out = torch.Tensor(n_out, n_rec).to(self.device)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.reset_parameters(w_init_gain)
        # utils
        self.Ls = []
        self.elig_in = []
        self.elig_rec = []
        self.elig_out = []
        self.elig_critic = []
        self.LH = []
        self.H = []
        self.action_tensors_for_wout = []
        self.probs = []
        self.zs = []
        self.voltages = []
        self.state_values = []
        self.action_taken = []
        self.saved_log_probs = []
        self.rewards = []
        self.saved_actions = []

    def init_net(self, n_b, n_t, n_in, n_rec, n_out):
        # Hidden state
        self.v = torch.zeros(n_t, n_b, n_rec).to(self.device)
        self.vo = torch.zeros(n_t, n_b, n_out).to(self.device)
        self.co = torch.zeros(n_t, n_b, 1).to(self.device)
        # Visible state
        self.z = torch.zeros(n_t, n_b, n_rec).to(self.device)
        # adaptability variable
        self.adaptability = torch.zeros(n_t, n_b, n_rec).to(self.device)

    def reset_parameters(self, gain):
        torch.nn.init.kaiming_normal_(self.w_in)
        self.w_in.data = gain[0] * self.w_in.data
        torch.nn.init.kaiming_normal_(self.w_rec)
        self.w_rec.data = gain[1] * self.w_rec.data
        torch.nn.init.kaiming_normal_(self.w_out)
        self.w_out.data = gain[2] * self.w_out.data
        torch.nn.init.kaiming_normal_(self.w_critic)
        self.w_critic.data = gain[2] * self.w_critic.data

    def init_grad(self):
        self.w_in.grad = torch.zeros_like(self.w_in)
        self.w_rec.grad = torch.zeros_like(self.w_rec)
        self.w_out.grad = torch.zeros_like(self.w_out)
        self.w_critic.grad = torch.zeros_like(self.w_critic)
        self.critic.input_weights.grad = torch.zeros_like(self.w_critic)


    def get_pos(self, x):
        scale = 50
        x_pos = self.encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)
        return x

    def forward(self, x, do_training, plot):
        x = self.get_pos(x)
        self.n_b = x.shape[1]  # Extracting batch size
        self.init_net(self.n_b, self.n_t, self.n_in, self.n_rec, self.n_out)  # Network reset
        self.w_rec *= (1 - torch.eye(self.n_rec, self.n_rec, device=self.device))  # Making sure recurrent self excitation/inhibition is cancelled
        counter = torch.zeros(self.n_rec).to(self.device)
        voltage_critic = torch.zeros(
            self.n_t, self.n_b, 1, device=self.device
        )
        sv = None
        # include refractory period
        for t in range(self.n_t - 1):
            self.v[t + 1] = (self.alpha * self.v[t] + torch.mm(self.z[t], self.w_rec.t()) + torch.mm(x[t], self.w_in.t())) - self.z[t] * self.thr
            self.adaptability[t + 1] = self.rho * self.adaptability[t] + self.z[t]
            self.z[t + 1] = (self.v[t + 1] > self.thr + self.beta * self.adaptability[t + 1]).float()

            # TOO SLOW
            """for i, z in enumerate(self.z[t + 1][0]):
                if z.item() == 1.0 and counter[i].item() >= 1:
                    self.z[t + 1] = 0
                    counter[i] += 1
                elif z.item() == 1.0 and counter[i].item() == 0:
                    self.z[t + 1] = 1
                    counter[i] += 1
                elif z.item() == 0.0 and counter[i].item() >= 1:
                    counter[i] += 1
            for i, c in enumerate(counter):
                if c.item() == self.refrac:
                    counter[i] = 0"""

            self.z[t + 1] = self.dropout(self.z[t + 1])
            self.vo[t + 1] = self.kappa * self.vo[t] + torch.mm(self.z[t + 1], self.w_out.t())
            readout_val, sv = self.critic(self.z[t + 1], sv)
            voltage_critic[t + 1, :, :] = readout_val
        m, _ = torch.max(self.vo, 0)
        plot_yo = F.softmax(self.vo, dim=-1)
        yo = F.softmax(m, dim=1)
        critic_value, _ = torch.max(voltage_critic, 0)
        # tracking data for plotting
        if plot:
          return yo, critic_value, self.z, self.v, plot_yo
        return yo, critic_value
      
    def update_grad(self, returns):
      td_error = None
      for t, ((log_prob, value), R) in enumerate(zip(self.saved_actions, returns)):
        td_error = R - value.item()
        self.w_in.grad += self.lr_layer[0] * td_error * self.elig_in[t] + self.lr_layer[0] * self.LH[t] * self.elig_in[t]
        self.w_rec.grad += self.lr_layer[1] * td_error * self.elig_rec[t] + self.lr_layer[1] * self.LH[t] * self.elig_rec[t]
        self.critic.input_weights.grad -= self.lr_layer[2] * self.cv * td_error * torch.sum(self.elig_critic[t], dim=(0, -1))
        self.w_out.grad += self.lr_layer[2] * td_error * torch.sum(self.elig_out[t], dim=-1) * self.action_tensors_for_wout[t].unsqueeze(-1) + self.probs[t].unsqueeze(-1) * (torch.log(self.probs[t].unsqueeze(-1)) + self.H[t]) * torch.sum(self.elig_out[t], dim=-1)
      # delete util variables used for weight update steps
      del self.elig_in[:]
      del self.elig_rec[:]
      del self.elig_out[:]
      del self.elig_critic[:]
      del self.LH[:]
      del self.H[:]
      del self.action_tensors_for_wout[:]
      del self.probs[:]

    def eligibility_traces(self, x, use_entropy_reg):
      x = self.get_pos(x)
      h = self.gamma * torch.max(torch.zeros_like(self.v), 1 - torch.abs((self.v - self.thr - self.beta * self.adaptability) / self.thr))
      alpha_conv = torch.tensor([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1,-1).to(self.device)
      rl_gamma_conv = torch.tensor([self.rl_gamma ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1,-1).to(self.device)

      # trace in and trace out are eligibility vectors for the hidden state v

      etrace_in = F.conv1d(x.permute(1, 2, 0), alpha_conv.expand(self.n_in, -1, -1), padding=self.n_t,
                          groups=self.n_in)[:, :, 1:self.n_t + 1].unsqueeze(1).expand(-1, self.n_rec, -1,
                                                                                      -1)  # n_b, n_rec, n_in , n_t
      etrace_in = torch.einsum('tbr,brit->brit', h, etrace_in)  # n_b, n_rec, n_in , n_t

      trace_rec = F.conv1d(self.z.permute(1, 2, 0), alpha_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                          groups=self.n_rec)[:, :, :self.n_t].unsqueeze(1).expand(-1, self.n_rec, -1,
                                                                                  -1)  # n_b, n_rec, n_rec, n_t
      trace_rec = torch.einsum('tbr,brit->brit', h, trace_rec)  # n_b, n_rec, n_rec, n_t

      # now compute eligibility vectors for adaptive threshold
      # elig_a_in
      
      elig_vec_a_in = torch.zeros_like(etrace_in)
      for t in range(self.n_t - 1):
          elig_vec_a_in[:, :, :, t + 1] = torch.einsum('br,bri->bri', h[t], etrace_in[:, :, :, t]) + torch.einsum('br,bri->bri', (self.rho - h[t] * self.beta), elig_vec_a_in[:, :, :, t])
      etrace_in = torch.einsum('tbr,brit->brit', h, (etrace_in - self.beta * elig_vec_a_in))

      # elig_a_rec
      elig_vec_a_rec = torch.zeros_like(trace_rec)
      for t in range(self.n_t - 1):
          elig_vec_a_rec[:, :, :, t + 1] = torch.einsum('br,bri->bri', h[t], trace_rec[:, :, :, t]) + torch.einsum('br,bri->bri', (self.rho - h[t] * self.beta), elig_vec_a_rec[:, :, :, t])
      etrace_rec = torch.einsum('tbr,brit->brit', h, (trace_rec - self.beta * elig_vec_a_rec))

      # trace out
      kappa_conv = torch.tensor([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)]).float().view(1, 1,-1).to(self.device)
      trace_out = F.conv1d(self.z.permute(1, 2, 0), kappa_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                          groups=self.n_rec)[:, :, 1:self.n_t + 1]  # n_b, n_rec, n_t

      action_taken = self.action_taken.pop()
      action_tensor = torch.zeros((self.n_out)).to(self.device)
      action_for_w_out = torch.zeros((self.n_out)).to(self.device)
      action_probs = action_taken[0]
      for k in range(self.n_out):
        action_tensor[k] = action_taken[0].probs[0][k]
      for k in range(self.n_out):
        if k == action_taken[1]:
            action_for_w_out[k] = action_taken[0].probs[0][k] - 1
        else:
            action_for_w_out[k] = action_taken[0].probs[0][k]
      L = -self.cv * self.w_critic + torch.sum(self.w_out * action_tensor.unsqueeze(1), dim=0)
      H = None
      if use_entropy_reg:
        H = action_probs.entropy()
        H = H.unsqueeze(-1).to(self.device)
        L_H = self.ch * torch.sum(self.w_out * action_tensor.unsqueeze(1) * (torch.log(action_tensor.unsqueeze(1)) + H), dim=0)
        L_H = L_H.unsqueeze(-1).to(self.device)
      
      etrace_rec = torch.squeeze(etrace_rec, 0)
      L_rec = etrace_rec * L.unsqueeze(-1).expand(1, self.n_rec, self.n_t)
      L_rec_filter = F.conv1d(L_rec, rl_gamma_conv.expand(self.n_rec, -1, -1), padding=self.n_t, groups=self.n_rec)[:,
                    :, :self.n_t]
      Le_rec = torch.sum(etrace_rec * L_rec_filter, dim=-1)
      # Le for in
      etrace_in = torch.squeeze(etrace_in, 0)
      L_in = L.unsqueeze(-1).permute(1, 2, 0).expand(-1, self.n_in, self.n_t)
      L_in_filter = F.conv1d(L_in, rl_gamma_conv.expand(self.n_in, -1, -1), padding=self.n_t, groups=self.n_in)[:, :,
                    :self.n_t]
      Le_in = torch.sum(etrace_in * L_in_filter, dim=-1)
      # W_Critic and w_out filters
      trace_out = F.conv1d(trace_out, rl_gamma_conv.expand(self.n_rec, -1, -1), padding=self.n_t, groups=self.n_rec)[
                  :, :, :self.n_t]
      trace_out = trace_out.expand(self.n_out, -1, -1)

      trace_out_critic = F.conv1d(trace_out, rl_gamma_conv.expand(self.n_rec, -1, -1), padding=self.n_t,
                                  groups=self.n_rec)[:, :, :self.n_t]
      trace_out_critic = trace_out_critic.expand(self.n_out, -1, -1)
      # track variables needed for weight update step
      self.elig_in.append(Le_in)
      self.elig_rec.append(Le_rec)
      self.elig_out.append(trace_out)
      self.elig_critic.append(trace_out_critic)
      self.LH.append(L_H)
      self.H.append(H)
      self.action_tensors_for_wout.append(action_for_w_out)
      self.probs.append(action_taken[0].probs[0])
      


