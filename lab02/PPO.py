import json
import math
import os
import sys
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

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
    def __init__(self, state_dim=48, act_dim=8):
        super().__init__()

        self.base = make_base([state_dim, 256, 256, 256, 64])
        self.mean_head = nn.Linear(64, act_dim)
        self.logvar_head = nn.Linear(64, act_dim)

    def forward(self, states: torch.Tensor):
        x = self.base(states)
        # mean: (-1, 1)
        mean = F.tanh(self.mean_head(x))
        # logvar: (-infty, 0] so that variance is in range [0, 1]
        logvar = -F.gelu(self.logvar_head(x))
        return mean, logvar


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=48):
        super().__init__()

        self.base = make_base([state_dim, 256, 256, 256, 64])
        self.value_head = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor):
        x = self.base(states)
        value = self.value_head(x)
        return value[..., 0]


def logprobs(means, logvars, actions):
    # log(Gaussian(µ, σ, x)) = -1/2 * ((x - µ)/σ)^2 - log(sqrt(2π)σ)
    # We can omit the 2π because we ultimately care about probability ratio.
    return (
        -0.5 * torch.pow(actions - means, 2) / torch.exp(logvars) - 0.5 * logvars
    ).sum(axis=-1)


def reward_estimation(V, states, rewards, gamma, lambda_):
    gae_estimate = torch.zeros(states.shape[:-1], device=device)
    values = V(states)
    gae = 0

    seqlen = states.shape[-2]
    for t in reversed(range(seqlen)):
        delta = (
            rewards[..., t]
            + (gamma * values[..., t + 1] if t + 1 < seqlen else 0)
            - values[..., t]
        )
        gae = gamma * lambda_ * gae + delta
        gae_estimate[..., t] = gae

    # returns and advantages
    return gae_estimate + values, gae_estimate


def ppo_update(
    optimizer,
    policy,
    policy_ref,
    V,
    V_targ,
    states,
    actions,
    rewards,
    epsilon=0.1,
    vf_coef=1.0,
    entropy_coef=0.1,
    gamma=0.98,
    lambda_=0.96,
    grad_clip_norm=1.0,
    normalize_advantages=True,
    num_iterations=10,
):
    states = states[:-1]
    with torch.no_grad():
        value_estimate, advantage_estimate = reward_estimation(
            V_targ, states, rewards, gamma, lambda_
        )
        logprobs_ref = logprobs(*policy_ref(states), actions)

    policy_losses = []
    vf_losses = []
    entropy_bonuses = []

    for iter_ in range(num_iterations):
        means_base, logvars_base_ = policy(states)
        logvars_base = torch.max(torch.tensor(-4.0, device=device), logvars_base_)
        logprobs_base = logprobs(means_base, logvars_base, actions)
        logprob_ratio = logprobs_base - logprobs_ref
        prob_ratio = torch.exp(logprob_ratio)

        if normalize_advantages:
            advantage_estimate = (advantage_estimate - advantage_estimate.mean()) / (1e-8 + advantage_estimate.std())

        policy_loss = -torch.min(
            advantage_estimate * prob_ratio,
            advantage_estimate * torch.clamp(prob_ratio, 1 - epsilon, 1 + epsilon),
        )
        policy_loss = policy_loss.sum(dim=-1).mean()

        vf_loss = F.mse_loss(V(states).squeeze(-1), value_estimate)

        # E_p[log(Gaussian(µ, σ, x))] = 1/2*log(2πσ^2) + 1/2 = 1/2 * log(2π) + log(σ) + 1/2
        # We only really care about the log(σ) part though, which is equal to log(var) up to a constant factor of 2.
        entropy_bonus = logvars_base.sum(dim=-1).mean() + 1 / 2 * (1 + math.log(2 * np.pi))

        loss = policy_loss + vf_coef * vf_loss + -entropy_coef * entropy_bonus
        # Penalize really negative logvars
        # loss = loss + (-1) * ((logvars_base_ < -4) * logvars_base_).mean()

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_([*policy.parameters(), *V.parameters()], grad_clip_norm)
        loss.backward()
        optimizer.step()

        policy_losses.append(policy_loss.item())
        vf_losses.append(vf_loss.item())
        entropy_bonuses.append(entropy_bonus.item())

    info = {
        "policy": policy_loss.item(),
        "vf": vf_loss.item(),
        "entropy": entropy_bonus.item(),
    }

    return (loss, info)


# Observation space is 51 dims.
# Action space is [-1, 1]^9.
def do_episode(env, model):
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

    states = torch.stack(states).float().detach()
    actions = torch.stack(actions).float().detach()
    rewards = torch.stack(rewards).float().detach()

    return (states, actions, rewards, truncated)


def run_training():
    import random

    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    NUM_ENVS = 128
    env = gym.make(
        # "RotateValveLevel0-v1",
        "StackCube-v1",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=NUM_ENVS,
    )
    env = ManiSkillVectorEnv(
        env,
        auto_reset=False,
        ignore_terminations=False,
    )

    policy = Policy().to(device)
    policy_ref = Policy().to(device)
    policy_ref.load_state_dict(policy_ref.state_dict())
    V = ValueNetwork().to(device)
    V_targ = ValueNetwork().to(device)
    V_targ.load_state_dict(V.state_dict())
    optimizer = torch.optim.Adam([*policy.parameters(), *V.parameters()], lr=1e-4)

    epochs = 10000
    ckpt_every = 25
    ppo_update_iterations = 10
    vf_sync_every = 5

    out_dir = 0
    while os.path.exists("runs/" + str(out_dir)):
        out_dir += 1
    os.makedirs("runs/" + str(out_dir))

    wandb.init(
        project="cs6501-lab02-ppo",
        name="run_" + str(out_dir)
    )

    with tqdm.tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            policy_ref.load_state_dict(policy.state_dict())
            if (epoch + 1 % vf_sync_every) == 0:
                V_targ.load_state_dict(V.state_dict())
            # batched content.
            states, actions, rewards, truncated = do_episode(env, policy)
            # log current reward.
            total_reward = rewards.sum(0).mean()

            # Run a pass over the data.
            loss, info = ppo_update(
                optimizer,
                policy,
                policy_ref,
                V,
                V_targ,
                states,
                actions,
                rewards,
                epsilon=0.2,
                num_iterations=ppo_update_iterations,
            )

            pbar.set_postfix({**info, "reward": total_reward.item()})
            pbar.update()
            with open(f"runs/{out_dir}/reward.jsonl", "a") as f:
                f.write(
                    json.dumps({"epoch": epoch, **info, "reward": total_reward.item()})
                    + "\n"
                )

            wandb.log({
                "epoch": epoch,
                **info,
                "reward": total_reward.item(),
            })

            if (epoch + 1) % ckpt_every == 0:
                torch.save(policy.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}.pt")

            if (epoch + 1) % ckpt_every == 0:
                torch.save(V.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}_vf.pt")


def eval():
    env = gym.make(
        "StackCube-v1",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=100,
    )
    policy = Policy().to(device)
    policy.load_state_dict(torch.load("runs/31/ckpt_650.pt", weights_only=True))

    success = 0
    for i in range(100):
        states, actions, rewards, truncated = do_episode(env, policy)
        success += (~truncated).sum().item()
        print((~truncated).sum().item())

    print(success / 100)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        run_training()
    else:
        eval()
