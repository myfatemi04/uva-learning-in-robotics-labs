import gymnasium as gym
import os
import time
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm
import json

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

NUM_ENVS = 128

def make_base(sizes):
    modules = []
    for i in range(len(sizes) - 1):
        modules.append(nn.Linear(sizes[i], sizes[i + 1]))
        modules.append(nn.GELU())
    return nn.Sequential(*modules)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = make_base([51, 64, 128, 128, 64])
        self.mean_head = nn.Linear(64, 9)
        self.logvar_head = nn.Linear(64, 9)
        
    def forward(self, states: torch.Tensor):
        x = self.base(states)
        # mean: (-1, 1)
        mean = F.tanh(self.mean_head(x))
        # logvar: (-infty, 0] so that variance is in range [0, 1]
        logvar = -F.gelu(self.logvar_head(x))
        return mean, logvar

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = make_base([51, 64, 128, 128, 64])
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, states: torch.Tensor):
        x = self.base(states)
        value = self.value_head(x)
        return value[..., 0]

def logprobs(means, logvars, actions):
    # log(Gaussian(µ, σ, x)) = -1/2 * ((x - µ)/σ)^2 - log(sqrt(2π)σ)
    # We can omit the 2π because we ultimately care about probability ratio.
    return (0.5 * torch.pow(actions - means, 2)/torch.exp(logvars) - 0.5 * logvars).sum(axis=-1)

def reward_estimation(V, states, rewards, gamma, lambda_):
    td_lambda_estimate = torch.zeros(states.shape[:-1], device=device)
    gae_estimate = torch.zeros(states.shape[:-1], device=device)
    
    # Compute values for each state
    values = V(states)  # Assuming V is the value function network
    
    # Initialize eligibility trace
    eligibility_trace = 0
    gae = 0
    
    seqlen = states.shape[-2]
    for t in reversed(range(seqlen)):
        delta = rewards[..., t] + (gamma * values[..., t + 1] if t + 1 < seqlen else 0) - values[..., t]
        
        # Update eligibility trace with lambda_
        eligibility_trace = gamma * lambda_ * eligibility_trace + 1
        gae = gamma * lambda_ * gae + delta
        
        # Update the TD(λ) estimate
        td_lambda_estimate[..., t] = delta * eligibility_trace + values[..., t]
        gae_estimate[..., t] = gae
    
    return td_lambda_estimate, gae_estimate

def ppo_loss(policy, policy_ref, V, states, actions, rewards, epsilon=0.1, vf_coef=0.5, entropy_coef=0.01, gamma=0.9, lambda_=0.8):
    # states.squeeze_(-1)
    # actions.squeeze_(-1)
    next_states = states[1:]
    states = states[:-1]

    means_base, logvars_base = policy(states)
    logprobs_base = logprobs(means_base, logvars_base, actions)
    with torch.no_grad():
        value_estimate, advantage_estimate = reward_estimation(V, states, rewards, gamma, lambda_)
        logprobs_ref = logprobs(*policy_ref(states), actions)
    logprob_ratio = logprobs_base - logprobs_ref

    policy_loss = torch.min(advantage_estimate * logprob_ratio, advantage_estimate * torch.clamp(logprob_ratio, -epsilon, epsilon))
    policy_loss = policy_loss.sum(dim=-1).mean()

    vf_loss = F.mse_loss(V(states).squeeze(-1), value_estimate)

    # E_p[log(Gaussian(µ, σ, x))] = 1/2*log(2πσ^2) + 1/2 = 1/2 * log(2π) + log(σ) + 1/2
    # We only really care about the log(σ) part though, which is equal to log(var) up to a constant factor of 2.
    entropy_bonus = logvars_base.sum(dim=-1).mean() + 1/2 * (1 + math.log(2 * np.pi))

    loss = policy_loss + vf_coef * vf_loss + -entropy_coef * entropy_bonus
    info = {
        "policy": policy_loss.item(),
        "vf": vf_loss.item(),
        "entropy": entropy_bonus.item(),
    }
    
    return (loss, info)

# Observation space is 51 dims.
# Action space is [-1, 1]^9.
def do_episode(model):
    obs, info = env.reset()
    done = False

    states = []
    actions = []
    rewards = []

    while not done:
        mean, logvar = model(obs)
        action = torch.randn(mean.shape).to(device) * torch.exp(logvar / 2) + mean

        states.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        done = terminated.any() or truncated.any()

    # Add terminal state.
    states.append(obs)

    states = torch.stack(states).float()
    actions = torch.stack(actions).float()
    rewards = torch.stack(rewards).float()

    print(states.shape, actions.shape, rewards.shape)

    return (states, actions, rewards, truncated)

env = gym.make(
    "RotateValveLevel0-v1",
    render_mode="rgb_array",
    sim_backend="gpu",
    control_mode="pd_joint_delta_pos",
    obs_mode="state",
    num_envs=NUM_ENVS,
)
env = ManiSkillVectorEnv(
    env,
    auto_reset=True,
    ignore_terminations=False,
)
# env = RecordEpisode(
#     env,
#     output_dir="Videos",
#     save_trajectory=False,
#     save_video=True,
#     video_fps=30,
#     max_steps_per_video=100,
# )

# obs, info = env.reset()
# print(info)
policy = Policy().to(device)
policy_ref = Policy().to(device)
policy_ref.load_state_dict(policy_ref.state_dict())
V = ValueNetwork().to(device)
optimizer = torch.optim.Adam([*policy.parameters(), *V.parameters()], lr=1e-4)

epochs = 10000
ckpt_every = 200
buffer_episodes = 20
# buffer_epochs = 10

out_dir = 0
while os.path.exists("runs/" + str(out_dir)):
    out_dir += 1
os.makedirs("runs/" + str(out_dir))

with tqdm.tqdm(total=epochs) as pbar:
    for epoch in range(epochs):
        # buf_states = []
        # buf_nextstates = []
        # buf_actions = []
        # buf_rewards = []
        # for ep in range(buffer_episodes):
        states, actions, rewards, truncated = do_episode(policy)
            # buf_states.append(states[:-1])
            # buf_nextstates.append(states[1:])
            # buf_actions.append(actions)
            # buf_rewards.append(rewards)
        loss, info = ppo_loss(
            policy,
            policy_ref,
            V,
            states,
            actions,
            rewards,
            epsilon=0.03,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*policy.parameters(), *V.parameters()], 1.0)
        optimizer.step()
        total_reward = rewards.sum(0).mean()
        pbar.set_postfix({**info, "reward": total_reward.item()})
        pbar.update()
        if (epoch + 1) % 10 == 0:
            policy_ref.load_state_dict(policy.state_dict())
        
        if (epoch + 1) % ckpt_every == 0:
            torch.save(policy_ref.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}.pt")

        with open(f'runs/{out_dir}/reward.jsonl', 'a') as f:
            f.write(json.dumps({"epoch": epoch, **info, "reward": total_reward.item()}) + "\n")

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
