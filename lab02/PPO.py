import gymnasium as gym
import os
import time
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(51, 64)
        self.lin2 = nn.Linear(64, 32)
        self.mean_head = nn.Linear(32, 9)
        self.logvar_head = nn.Linear(32, 9)
        
    def forward(self, states: torch.Tensor):
        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        # mean: (-1, 1)
        mean = F.tanh(self.mean_head(x))
        # logvar: (-infty, 0] so that variance is in range [0, 1]
        logvar = -F.relu(self.logvar_head(x))
        return mean, logvar

def calculate_returns_simple(rewards: torch.Tensor, gamma=0.9):
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1]
    for i in range(1, len(returns)):
        returns[-i - 1] = rewards[-i - 1] + gamma * returns[-i]
    return returns

def logprobs(means, logvars, actions):
    # log(Gaussian(µ, σ, x)) = -1/2 * ((x - µ)/σ)^2 - log(sqrt(2π)σ)
    # We can omit the 2π because we ultimately care about probability ratio.
    return 0.5 * torch.pow(actions - means, 2)/torch.exp(logvars) - 0.5 * logvars

def ppo_loss(policy, policy_ref, critic, states, actions, returns, epsilon=0.2, vf_coef=0.5, entropy_coef=0.1):
    means_base, logvars_base = policy(states)
    logprobs_base = logprobs(means_base, logvars_base, actions)
    with torch.no_grad():
        advantages = returns - torch.gather(critic(states), -1, actions)
        logprobs_ref = logprobs(*policy_ref(states), actions)
    logprob_ratio = logprobs_base - logprobs_ref

    policy_loss = torch.min(advantages * log_ratio, advantages * torch.clamp(logprob_ratio, -epsilon, epsilon))
    vf_loss = F.mse_loss(critic(states, actions), returns)

    # E_p[log(Gaussian(µ, σ, x))] = 1/2*log(2πσ^2) + 1/2 = 1/2 * log(2π) + log(σ) + 1/2
    # We only really care about the log(σ) part though, which is equal to log(var) up to a constant factor of 2.
    entropy_bonus = logvars_base.sum(dim=-1).mean()

    return policy_loss + vf_coef * vf_loss + entropy_coef * entropy_bonus

env = gym.make(
    "RotateValveLevel0-v1",
    render_mode="rgb_array",
    sim_backend="gpu",
    control_mode="pd_joint_delta_pos",
    obs_mode="state",
)
env = RecordEpisode(
    env,
    output_dir="Videos",
    save_trajectory=False,
    save_video=True,
    video_fps=30,
    max_steps_per_video=100,
)
# Observation space is 51 dims.
# Action space is [-1, 1]^9.
def do_episode(model):
    obs, info = env.reset()
    done = False

    states = []
    actions = []
    rewards = []

    while not done:
        print("Taking step.")
        mean, logvar = model(obs)
        action = torch.randn(mean.shape).to("cuda") * torch.exp(logvar / 2) + mean

        states.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        done = terminated or truncated

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)

    return (states, actions, rewards, truncated)

# obs, info = env.reset()
# print(info)
model = Model().to("cuda")
ep = do_episode(model)

print(calculate_returns_simple(ep[2]))
print(ep[2])

"""
state = {
    'agent': {
        'qpos': tensor([[0.6708, -0.6970, -0.6773, -0.0490, -0.0246,  0.0112, -0.0041, -0.0011, -0.0080]], device='cuda:0'),
        'qvel': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'),
        'tip_poses': tensor([[-0.0209,  0.1451,  0.0956,  0.0430, -0.5684, -0.3984,  0.7185, -0.1264, -0.0909,  0.0997, -0.0565, -0.5715,  0.4062,  0.7108,  0.1398, -0.0584, 0.0949, -0.6370, -0.6384, -0.3068,  0.3042]], device='cuda:0')
    },
    'extra': {
        'rotate_dir': tensor([1.], device='cuda:0'),
        'valve_qpos': tensor([[0.9170]], device='cuda:0'),
        'valve_qvel': tensor([[0.]], device='cuda:0'),
        'valve_x': tensor([0.6082], device='cuda:0'),
        'valve_y': tensor([0.7938], device='cuda:0'),
        'valve_pose': tensor([[1.9796e-02,  2.4465e-03, -5.5879e-09,  5.1358e-01,  0.0000e+00, 0.0000e+00,  8.5804e-01]], device='cuda:0')
    }
}
"""
# print(obs.shape)
