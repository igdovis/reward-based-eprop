import torch
import torch.nn as nn
from utils import train_reward_AC
from RSNN_ALIF import RSNN_ALIF
from LIF import LIF

lr_layer_norm = (0.05, 0.05, 1.0)
# Network model parameters
n_rec = 128
threshold = 1
tau_mem = 20e-3
tau_out = 20e-3
bias_out = 0.0
gamma = 0.3
w_init_gain = (0.5, 0.1, 0.5)
n_in = 8
n_out = 2
seq_len = 50
rho = 0.985
beta = 0.1

model = RSNN_ALIF(
           n_in,
           n_rec,
           n_out,
           seq_len,
           threshold,
           tau_mem,
           tau_out,
           bias_out,
           gamma,
           dt=1e-3,
           w_init_gain=w_init_gain,
           rho=rho,
           beta=beta,
           lr_layer=lr_layer_norm,
           device="cuda"
)

"""
model = LIF(n_in,
               n_rec,
               n_out,
               seq_len,
               threshold,
               tau_mem,
               tau_out,
               bias_out,
               gamma,
               dt=1e-3,
               classif=True,
               w_init_gain=w_init_gain,
               lr_layer=lr_layer_norm,
               device="cuda"
           )
lif = lif.to(device)
"""

device = torch.device("cuda")
model.to(device)
train_reward_AC(model, "ALIF")
