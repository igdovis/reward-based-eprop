import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['prob', 'value'])

def select_action_AC(state, policy, device, t, training):
  state = torch.from_numpy(state).float().unsqueeze(0).to(device)
  probs, critic_value = policy(state, training, False)
  m = torch.distributions.Categorical(probs)
  action = m.sample()
  policy.saved_actions.append(SavedAction(m.log_prob(action), critic_value))
  policy.action_taken.append((m, action.item()))
  return action.item(), critic_value

def select_action_AC_for_plot(state, policy, device, t, training):
  state = torch.from_numpy(state).float().unsqueeze(0).to(device)
  probs, critic_value, zs, vols, probs_in_rnn = policy(state, training, True)
  m = torch.distributions.Categorical(probs)
  probs = m.probs
  action = m.sample()
  policy.saved_actions.append(SavedAction(m.log_prob(action), critic_value))
  policy.action_taken.append((m, action.item()))
  return action.item(), probs, zs, vols, probs_in_rnn


def train_reward_AC(model, modelType):
  device = torch.device("cuda")
  policy = model.to(device)
  running_reward = 10
  gamma = 0.98
  episodes = 500
  timesteps = 3000
  running_rewards = []
  episode_rewards = []

  optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
  env = gym.make("CartPole-v1")
  env.reset()
  policy.eval()
  with torch.no_grad():
      for e in range(1, episodes + 1):
          state, ep_reward = env.reset(), 0
          optimizer.zero_grad()
          policy.init_grad()
          policy.state_values.append(0)
          for t in range(timesteps):
              action, state_critic_value = select_action_AC(state, policy, device, t, training=False)
              policy.state_values.append(state_critic_value)
              state, reward, done, _ = env.step(action)
              policy.rewards.append(reward)
              if modelType == "LIF":
                policy.grads_batch_ac(torch.from_numpy(state).float().unsqueeze(0).to(device))
              else:
                policy.eligibility_traces(torch.from_numpy(state).float().unsqueeze(0).to(device), True)
              ep_reward = ep_reward + reward
              if done:
                  break
          finish_episode_actor(policy, optimizer, gamma)
          optimizer.step()
          running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
          if e % 10 == 0:
              print(
                  "Episode {}/{} \tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                      e, episodes, ep_reward, running_reward
                  )
              )
          episode_rewards.append(ep_reward)
          running_rewards.append(running_reward)
          if running_reward > env.spec.reward_threshold:
              print(
                  "Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t)
              )
              break
          np.save("running_rewards.npy", np.array(running_rewards))
          np.save("episode_rewards.npy", np.array(episode_rewards))
          torch.save(optimizer.state_dict(), "optimizer.pt")
          torch.save(policy.state_dict(), "policy.pt")

    

def finish_episode_actor(policy, optimizer, gamma):
    eps = np.finfo(np.float32).eps.item()
    R = 0
    saved_actions = policy.saved_actions
    policy_loss = []
    critic_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.as_tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    policy.update_grad(returns)
    # reset rewards and action buffer
    del policy.rewards[:]
    del policy.saved_actions[:]
    del policy.state_values[:]

