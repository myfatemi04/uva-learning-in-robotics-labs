import math
import os
import random
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tqdm
import wandb
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

BLANK_TENSOR = torch.tensor(0)


@dataclass
class Rollout:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    logprobs: torch.Tensor
    dones: (
        torch.Tensor
    )  # dones[t] indicates that after taking action[t], a "done" signal was received.
    infos: list[dict]
    returns: torch.Tensor
    advantages: torch.Tensor

    def flatten(self):
        return Rollout(
            # Remove the "final" state.
            self.states[:-1].reshape((-1, self.states.shape[-1])),
            self.actions.reshape((-1, self.actions.shape[-1])),
            self.rewards.reshape((-1,)),
            self.logprobs.reshape((-1,)),
            # note: "dones" and "infos" lose their meaning when flattened.
            BLANK_TENSOR,  # self.dones.reshape((-1,)),
            [],  # self.infos,
            self.returns.reshape((-1,)),
            self.advantages.reshape((-1,)),
        )

    @staticmethod
    def from_raw_states(
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        logprobs: torch.Tensor,
        dones: torch.Tensor,
        value_estimates: torch.Tensor,
        infos: list[dict],
        gamma: float,
        lambda_: float,
        normalize_advantages: bool,
    ):
        advantages = torch.zeros(states.shape[:-1], device=device)
        gae = 0

        seqlen = actions.shape[0]
        for t in reversed(range(seqlen)):
            delta = rewards[t] + gamma * value_estimates[t + 1] - value_estimates[t]

            # If action t received a "done" signal, then future rewards should not percolate beyond
            # episode boundaries.
            gae = gamma * lambda_ * gae * (1.0 - dones[t]) + delta
            advantages[t] = gae

        returns = advantages + value_estimates
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return Rollout(
            states,
            actions,
            rewards,
            logprobs,
            dones,
            infos,
            returns,
            advantages,
        )

    def sample_batch(self, batch_size: int):
        indexes = torch.randperm(self.states.shape[0])[:batch_size]
        return Rollout(
            self.states[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.logprobs[indexes],
            BLANK_TENSOR,  # self.dones[indexes],
            [],  # self.infos[indexes],
            self.returns[indexes],
            self.advantages[indexes],
        )

    def __getitem__(self, slice):
        return Rollout(
            self.states[slice],
            self.actions[slice],
            self.rewards[slice],
            self.logprobs[slice],
            BLANK_TENSOR,  # self.dones[slice],
            [],  # self.infos[slice],
            self.returns[slice],
            self.advantages[slice],
        )

    def __len__(self):
        return self.actions.shape[0]


@dataclass
class ActionStep:
    mean: torch.Tensor
    logstd: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    actions: torch.Tensor


def compute_logprobs(
    means: torch.Tensor, logstds: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    # log(Gaussian(µ, σ, x)) = -1/2 * ((x - µ)/σ)^2 - log(sqrt(2π)σ)
    # We can omit the 2π because we ultimately care about probability ratio.
    return (-0.5 * (actions - means) ** 2 / (1e-8 + torch.exp(logstds)) - logstds).sum(
        dim=-1
    )


def compute_entropy(logstds: torch.Tensor):
    # E_p[log(Gaussian(µ, σ, x))] = 1/2*log(2πσ^2) + 1/2 = 1/2 * log(2π) + log(σ) + 1/2
    return 0.5 * (1 + math.log(2 * math.pi)) + logstds.sum(dim=-1).mean()


def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_base(sizes):
    modules = []
    for i in range(len(sizes) - 1):
        modules.append(layer_init(nn.Linear(sizes[i], sizes[i + 1])))
        modules.append(nn.Tanh())
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.actor_base = make_base([state_dim, 256, 256, 256])
        self.critic_base = make_base([state_dim, 256, 256, 256])
        self.mean_head = nn.Linear(256, action_dim)
        self.logstd = nn.Parameter(
            -1 / 2 * torch.ones(1, action_dim), requires_grad=True
        )
        self.value_head = nn.Linear(256, 1)

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_actor = self.actor_base(states)
        x_critic = self.critic_base(states)

        mean = F.tanh(self.mean_head(x_actor))
        logstd = self.logstd.expand_as(mean)
        value = self.value_head(x_critic).squeeze(-1)

        return mean, logstd, value

    def sample_action(self, states: torch.Tensor, deterministic: bool) -> ActionStep:
        means, logstds, values = self(states)
        actions = means + (
            torch.randn_like(means) * torch.exp(logstds / 2) if not deterministic else 0
        )
        logprobs = compute_logprobs(means, logstds, actions)

        return ActionStep(
            means,
            logstds,
            logprobs,
            values,
            actions,
        )

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self(states)[2]

    def value_logprob_entropy(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means, logstds, values = self(states)
        logprobs = compute_logprobs(means, logstds, actions)
        entropy = compute_entropy(logstds)
        return values, logprobs, entropy


def ppo_update(
    optimizer: torch.optim.Optimizer,  # type: ignore
    model: ActorCritic,
    rollout: Rollout,
    clip_coefficient: float,
    vf_coef: float,
    entropy_coef: float,
    grad_clip_norm: float,
    target_kl: float,
    num_iterations: int,
    minibatch_size: int,
) -> dict:
    rollout_flat = rollout.flatten()

    losses = {
        "policy_loss": [],
        "vf_loss": [],
        "entropy": [],
        "approx_kl": [],
    }

    for _ in range(num_iterations):
        indices = torch.randperm(len(rollout_flat), device=device)

        for i in range(0, len(indices), minibatch_size):
            minibatch_indices = indices[i : i + minibatch_size]
            rollout_minibatch = rollout_flat[minibatch_indices]

            values, logprobs, entropy = model.value_logprob_entropy(
                rollout_minibatch.states, rollout_minibatch.actions
            )
            log_ratio = logprobs - rollout_minibatch.logprobs
            ratio = torch.exp(log_ratio)

            # See if we need to stop early because KL divergence is too high.
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if target_kl is not None and approx_kl > target_kl:
                    break

            policy_loss = -torch.min(
                rollout_minibatch.advantages * ratio,
                rollout_minibatch.advantages
                * torch.clamp(ratio, 1 - clip_coefficient, 1 + clip_coefficient),
            ).mean()
            vf_loss = F.mse_loss(values, rollout_minibatch.returns, reduction="mean")
            loss = policy_loss + vf_coef * vf_loss + -entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            losses["policy_loss"].append(policy_loss.item())
            losses["vf_loss"].append(vf_loss.item())
            losses["entropy"].append(entropy.item())
            losses["approx_kl"].append(approx_kl)

    return losses


def run_rollout(
    env: gym.Env,
    model: ActorCritic,
    steps: int,
    deterministic: bool,
    gamma: float,
    lambda_: float,
    normalize_advantages: bool,
) -> Rollout:
    obs, info = env.reset()
    states = []
    actions = []
    logprobs = []
    rewards = []
    value_estimates = []
    dones = []
    infos = []

    for _ in range(steps):
        frame = model.sample_action(obs, deterministic)

        states.append(obs)
        actions.append(frame.actions)
        logprobs.append(frame.logprobs)
        value_estimates.append(frame.values)

        obs, reward, terminated, truncated, info = env.step(frame.actions)

        rewards.append(reward)
        dones.append(terminated | truncated)
        infos.append(info)

    # Add final state.
    states.append(obs)
    # Add final value.
    value_estimates.append(model.value(obs))

    states = torch.stack(states).float().detach()
    actions = torch.stack(actions).float().detach()
    rewards = torch.stack(rewards).float().detach()
    logprobs = torch.stack(logprobs).float().detach()
    dones = torch.stack(dones).float().detach()
    value_estimates = torch.stack(value_estimates).float().detach()

    return Rollout.from_raw_states(
        states,
        actions,
        rewards,
        logprobs,
        dones,
        value_estimates,
        infos,
        gamma,
        lambda_,
        normalize_advantages,
    )


def run_training():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ENV = "PickCube-v1"
    NUM_ENVS = 512
    EPISODE_STEPS = 50
    EPOCHS = 10000
    CHECKPOINT_EVERY = 25
    PPO_ITERATIONS = 4
    VF_COEFFICIENT = 0.5
    CLIP_COEFFICIENT = 0.2
    ENTROPY_COEFFICIENT = 0.0
    GRADIENT_CLIPPING_NORM = 1.0
    NORMALIZE_ADVANTAGES = True
    MINIBATCH_SIZE = 800
    GAMMA = 0.8
    LAMBDA = 0.9
    TARGET_KL = 0.1

    env = gym.make(
        ENV,
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=NUM_ENVS,
    )
    env = ManiSkillVectorEnv(
        env,  # type: ignore
        auto_reset=False,
        ignore_terminations=True,
        record_metrics=True,
    )
    model = ActorCritic(
        env.single_observation_space.shape[0],  # type: ignore
        env.single_action_space.shape[0],  # type: ignore
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)  # type: ignore

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
            "episode_steps": EPISODE_STEPS,
            "epochs": EPOCHS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "ppo_iterations": PPO_ITERATIONS,
            "vf_coefficient": VF_COEFFICIENT,
            "clip_coefficient": CLIP_COEFFICIENT,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "gradient_clipping_norm": GRADIENT_CLIPPING_NORM,
            "normalize_advantages": NORMALIZE_ADVANTAGES,
            "minibatch_size": MINIBATCH_SIZE,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "target_kl": TARGET_KL,
        },
    )

    global_step = 0

    with tqdm.tqdm(total=EPOCHS) as pbar:
        for epoch in range(EPOCHS):
            rollout = run_rollout(
                env,
                model,
                steps=EPISODE_STEPS,
                deterministic=False,
                gamma=GAMMA,
                lambda_=LAMBDA,
                normalize_advantages=NORMALIZE_ADVANTAGES,
            )
            train_returns = rollout.returns.sum(0).mean().item()
            train_rewards = rollout.rewards.sum(0).mean().item()
            for info in rollout.infos:
                global_step += NUM_ENVS
                if "final_info" in info:
                    done_mask = info["_final_info"]
                    wandb.log(
                        {
                            "train/" + k: v[done_mask].mean()
                            for (k, v) in info["final_info"].items()
                        },
                        global_step,
                    )

            # Run a pass over the data.
            losses = ppo_update(
                optimizer,
                model,
                rollout,
                clip_coefficient=CLIP_COEFFICIENT,
                vf_coef=VF_COEFFICIENT,
                entropy_coef=ENTROPY_COEFFICIENT,
                grad_clip_norm=GRADIENT_CLIPPING_NORM,
                target_kl=TARGET_KL,
                num_iterations=PPO_ITERATIONS,
                minibatch_size=MINIBATCH_SIZE,
            )

            wandb.log(
                {"losses/" + k: losses[k][-1] for k in losses.keys()}, global_step
            )

            pbar.set_postfix(
                {"train/returns": train_returns, "train/rewards": train_rewards}
            )
            pbar.update()

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}.pt")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        run_training()
    # else:
    #     eval()
# policy = Policy()
# mean, logvar = policy(torch.zeros((1, 42)))
# print(mean)
# logvar.sum().backward()
# print(policy.logvar.grad)
