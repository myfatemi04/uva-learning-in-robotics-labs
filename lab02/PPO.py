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

NUM_ENVS = 512
# ENV = "OpenCabinetDrawer-v1"
# STATE_DIM = 44
# ACTION_DIM = 13
ENV = "PickCube-v1"
STATE_DIM = 42
ACTION_DIM = 8

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_base(sizes):
    modules = []
    for i in range(len(sizes) - 1):
        modules.append(layer_init(nn.Linear(sizes[i], sizes[i + 1])))
        modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class Policy(nn.Module):
    def __init__(self, state_dim=STATE_DIM, act_dim=ACTION_DIM):
        super().__init__()

        self.base = make_base([state_dim, 256, 256, 256])
        self.mean_head = nn.Linear(256, act_dim)
        # self.logvar_head = nn.Linear(64, act_dim)
        self.logvar = nn.Parameter(-torch.ones(1, act_dim), requires_grad=True)

    def forward(self, states: torch.Tensor):
        x = self.base(states)
        # mean: (-1, 1)
        mean = F.tanh(self.mean_head(x))
        # logvar: (-infty, 0] so that variance is in range (0, 1]
        # logvar = -F.elu(self.logvar_head(x))
        return mean, self.logvar.expand_as(mean)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()

        self.base = make_base([state_dim, 256, 256, 256])
        self.value_head = nn.Linear(256, 1)

    def forward(self, states: torch.Tensor):
        x = self.base(states)
        value = self.value_head(x)
        return value[..., 0]


def logprobs(means, logvars, actions):
    # log(Gaussian(µ, σ, x)) = -1/2 * ((x - µ)/σ)^2 - log(sqrt(2π)σ)
    # We can omit the 2π because we ultimately care about probability ratio.
    return (
        -0.5 * torch.pow(actions - means, 2) / (1e-8 + torch.exp(logvars)) - 0.5 * logvars
    ).sum(axis=-1)


def reward_estimation(V, states, rewards, dones, gamma, lambda_):
    gae_estimate = torch.zeros(states.shape[:-1], device=device)
    values = V(states)
    gae = 0

    seqlen = states.shape[0] - 1
    for t in reversed(range(seqlen)):
        delta = (
            rewards[t]
            + (gamma * values[t + 1] if t + 1 < seqlen else 0)
            - values[t]
        )
        gae = gamma * lambda_ * gae * (1.0 - dones[t + 1]) + delta
        gae_estimate[t] = gae

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
    dones,
    epsilon=0.1,
    vf_coef=1.0,
    entropy_coef=0.1,
    gamma=0.98,
    lambda_=0.96,
    grad_clip_norm=1.0,
    target_kl=0.1,
    normalize_advantages=True,
    num_iterations=4,
    minibatch_size=800,
):
    states = states[:-1]
    with torch.no_grad():
        return_estimate, advantage_estimate = reward_estimation(
            V_targ, states, rewards, dones, gamma, lambda_
        )
        logprobs_ref = logprobs(*policy_ref(states), actions)

    if normalize_advantages:
        advantage_estimate = (advantage_estimate - advantage_estimate.mean()) / (
            1e-8 + advantage_estimate.std()
        )

    # flatten.
    states_flat = states.reshape((-1, states.shape[-1]))
    return_estimates_flat = return_estimate.reshape((-1,))
    advantage_estimates_flat = advantage_estimate.reshape((-1,))
    logprobs_ref_flat = logprobs_ref.reshape((-1,))
    actions_flat = actions.reshape((-1, actions.shape[-1]))
    indices = np.arange(states.shape[0])

    policy_losses = []
    vf_losses = []
    entropy_bonuses = []

    for iter_ in range(num_iterations):
        np.random.shuffle(indices)

        for i in range(0, len(indices), minibatch_size):
            minibatch_indices = indices[i:i + minibatch_size]

            means_base, logvars_base = policy(states_flat[minibatch_indices])
            # logvars_base = torch.max(torch.tensor(-8.0, device=device), logvars_base_)
            logprobs_base = logprobs(means_base, logvars_base, actions_flat[minibatch_indices])
            logprob_ratio = logprobs_base - logprobs_ref_flat[minibatch_indices]
            prob_ratio = torch.exp(logprob_ratio)

            policy_loss = -torch.min(
                advantage_estimates_flat[minibatch_indices] * prob_ratio,
                advantage_estimates_flat[minibatch_indices] * torch.clamp(prob_ratio, 1 - epsilon, 1 + epsilon),
            )
            policy_loss = policy_loss.mean()

            vf_loss = F.mse_loss(
                V(states_flat[minibatch_indices]),
                return_estimates_flat[minibatch_indices],
            )

            # E_p[log(Gaussian(µ, σ, x))] = 1/2*log(2πσ^2) + 1/2 = 1/2 * log(2π) + log(σ) + 1/2
            # We only really care about the log(σ) part though, which is equal to log(var) up to a constant factor of 2.
            entropy_bonus = logvars_base.sum(dim=-1).mean() + 1 / 2 * (
                1 + math.log(2 * np.pi)
            )

            loss = policy_loss + vf_coef * vf_loss + -entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*policy.parameters(), *V.parameters()], grad_clip_norm
            )
            optimizer.step()

            policy_losses.append(policy_loss.item())
            vf_losses.append(vf_loss.item())
            entropy_bonuses.append(entropy_bonus.item())

            with torch.no_grad():
                approx_kl = ((prob_ratio - 1) - logprob_ratio).mean()

            if target_kl is not None and approx_kl > target_kl:
                break

    info = {
        "policy": policy_loss.item(),
        "vf": vf_loss.item(),
        "entropy": entropy_bonus.item(),
        "logvar": logvars_base.mean().item(),
        "mean": means_base.mean().item(),
    }

    return (loss, info)


def do_episode(env, model, steps: int, deterministic=False):
    obs, info = env.reset()
    done = False

    states = []
    actions = []
    rewards = []
    dones = []

    for _ in range(steps):
        mean, logvar = model(obs)
        if deterministic:
            action = mean
        else:
            action = torch.randn(mean.shape).to(device) * torch.exp(logvar / 2) + mean

        states.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        dones.append(terminated | truncated)
    
    # Add terminal state.
    states.append(obs)

    states = torch.stack(states).float().detach()
    actions = torch.stack(actions).float().detach()
    rewards = torch.stack(rewards).float().detach()
    dones = torch.stack(dones).float().detach()

    return (states, actions, rewards, truncated, dones)


def run_training():
    import random

    import numpy as np
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    env = gym.make(
        ENV,
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=NUM_ENVS,
    )
    env = ManiSkillVectorEnv(
        env,
        auto_reset=False,
        ignore_terminations=True,
    )

    policy = Policy().to(device)
    policy_ref = Policy().to(device)
    policy_ref.load_state_dict(policy_ref.state_dict())
    V = ValueNetwork().to(device)
    V_targ = ValueNetwork().to(device)
    V_targ.load_state_dict(V.state_dict())
    optimizer = torch.optim.Adam([*policy.parameters(), *V.parameters()], lr=3e-4, weight_decay=1e-5)

    EPISODE_STEPS = 50
    epochs = 10000
    ckpt_every = 25
    ppo_update_iterations = 4
    vf_sync_every = 5
    epsilon = 0.2
    entropy_bonus = 0.0
    grad_clip_norm = 1.0
    # gamma = 0.98
    gamma = 0.8
    lambda_ = 0.9
    target_kl = 0.1
    # lambda_ = 0.96

    out_dir = 0
    while os.path.exists("runs/" + str(out_dir)):
        out_dir += 1
    os.makedirs("runs/" + str(out_dir))

    wandb.init(
        project="cs6501-lab02-ppo",
        name="run_" + str(out_dir),
        config={
            "environment": ENV,
            "num_envs": NUM_ENVS,
            "ppo_update_iterations": 10,
            "vf_sync_every": 5,
            "entropy_bonus": entropy_bonus,
            "epsilon": epsilon,
            "gamma": gamma,
            "lambda": lambda_,
            "grad_clip_norm": grad_clip_norm,
        },
    )

    with tqdm.tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            policy_ref.load_state_dict(policy.state_dict())
            if (epoch + 1 % vf_sync_every) == 0:
                V_targ.load_state_dict(V.state_dict())
            # batched content.
            states, actions, rewards, truncated, dones = do_episode(env, policy, steps=EPISODE_STEPS)
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
                dones,
                epsilon=epsilon,
                entropy_coef=entropy_bonus,
                gamma=gamma,
                lambda_=lambda_,
                num_iterations=ppo_update_iterations,
                grad_clip_norm=grad_clip_norm,
                target_kl=target_kl,
            )

            pbar.set_postfix({**info, "reward": total_reward.item()})
            pbar.update()
            with open(f"runs/{out_dir}/reward.jsonl", "a") as f:
                f.write(
                    json.dumps({"epoch": epoch, **info, "reward": total_reward.item()})
                    + "\n"
                )

            wandb.log(
                {**info, "reward": total_reward.item()},
                step=epoch,
            )

            if (epoch + 1) % ckpt_every == 0:
                torch.save(policy.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}.pt")

            if (epoch + 1) % ckpt_every == 0:
                torch.save(V.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}_vf.pt")


def eval():
    env = gym.make(
        ENV,
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=1,
    )
    print(env.action_space)
    env = RecordEpisode(env, output_dir="Videos", save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=100)
    policy = Policy().to(device)
    policy.load_state_dict(torch.load("runs/44/ckpt_7200.pt", weights_only=True))

    success = 0
    for i in range(100):
        states, actions, rewards, truncated = do_episode(
            env, policy, deterministic=True
        )
        print(actions)
        success += (~truncated).sum().item()
        print((~truncated).sum().item(), rewards.sum().item())

    print(success / 100)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        run_training()
    else:
        eval()
# policy = Policy()
# mean, logvar = policy(torch.zeros((1, 42)))
# print(mean)
# logvar.sum().backward()
# print(policy.logvar.grad)
