import math
import os
import random
import sys
from dataclasses import dataclass
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tqdm
import wandb
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch.distributions import Normal

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
    value_estimates: torch.Tensor
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
            self.returns[:-1].reshape((-1,)),
            self.value_estimates[:-1].reshape((-1,)),
            self.advantages[:-1].reshape((-1,)),
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
        finite_horizon_gae: bool,
    ):
        advantages = torch.zeros(states.shape[:-1], device=device)
        gae = 0
        seqlen = actions.shape[0]

        if not finite_horizon_gae:
            for t in reversed(range(seqlen)):
                delta = rewards[t] + gamma * value_estimates[t + 1] - value_estimates[t]

                # If action t received a "done" signal, then future rewards should not percolate beyond
                # episode boundaries.
                gae = gamma * lambda_ * gae * (~dones[t]) + delta
                advantages[t] = gae
        else:
            lambda_coefficient_sum = 0
            reward_term_sum = 0
            value_term_sum = 0
            for t in reversed(range(seqlen)):
                # Reinitialize the sums for any environments which just terminated their episode.
                lambda_coefficient_sum = lambda_coefficient_sum * (~dones[t])
                reward_term_sum = reward_term_sum * (~dones[t])
                value_term_sum = value_term_sum * (~dones[t])

                # Sort of inductive.
                # The r_t sum is gamma * r_{t + 1} sum + r_t times the current lambda coefficient sum.
                # This is based on the implementation in the Maniskill baseline.
                """
                1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                """
                lambda_coefficient_sum = 1 + lambda_ * lambda_coefficient_sum
                reward_term_sum = (
                    lambda_ * gamma * reward_term_sum
                    + lambda_coefficient_sum * rewards[t]
                )
                value_term_sum = (
                    lambda_ * gamma * value_term_sum + gamma * value_estimates[t + 1]
                )
                advantages[t] = (
                    reward_term_sum + value_term_sum
                ) / lambda_coefficient_sum - value_estimates[t]

        returns = advantages + value_estimates
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # print(
        #     f"{states.shape=} {actions.shape=} {rewards.shape=} {logprobs.shape=} {dones.shape=} {value_estimates.shape=} {len(infos)=} {returns.shape=} {advantages.shape=}"
        # )

        return Rollout(
            states,
            actions,
            rewards,
            logprobs,
            dones,
            infos,
            returns,
            value_estimates,
            advantages,
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
            self.value_estimates[slice],
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


# def compute_logprobs(
#     means: torch.Tensor, logstds: torch.Tensor, actions: torch.Tensor
# ) -> torch.Tensor:
#     # log(Gaussian(µ, σ, x)) = -1/2 * ((x - µ)/σ)^2 - log(sqrt(2π)σ)
#     # We can omit the 2π because we ultimately care about probability ratio.
#     return (-0.5 * (actions - means) ** 2 / (1e-8 + torch.exp(logstds)) - logstds).sum(
#         dim=-1
#     )


# def compute_entropy(logstds: torch.Tensor):
#     # E_p[log(Gaussian(µ, σ, x))] = 1/2*log(2πσ^2) + 1/2 = 1/2 * log(2π) + log(σ) + 1/2
#     return 0.5 * (1 + math.log(2 * math.pi)) + logstds.sum(dim=-1).mean()


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


def make_critic(state_dim):
    return make_base([state_dim, 256, 256, 256, 1])


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # self.actor_base = make_base([state_dim, 256, 256, 256])
        # self.mean_head = nn.Linear(256, action_dim)
        # layer_init(self.mean_head, std=0.01 * np.sqrt(2))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )

        self.logstd = nn.Parameter(-0.5 * torch.ones(1, action_dim), requires_grad=True)
        self.critic = make_critic(state_dim)
        # self.critic_target = make_critic(state_dim)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        # self.critic_target.requires_grad_(False)

    # def synchronize_critics(self, beta: float):
    #     # Exponential moving average
    #     with torch.no_grad():
    #         for param, target_param in zip(
    #             self.critic.parameters(), self.critic_target.parameters()
    #         ):
    #             target_param.data = beta * target_param.data + (1 - beta) * param.data

    def actor(self, states: torch.Tensor):
        # x_actor = self.actor_base(states)
        # mean = F.tanh(self.mean_head(x_actor))
        mean = self.actor_mean(states)
        logstd = self.logstd.expand_as(mean)
        return mean, logstd

    def sample_action(self, states: torch.Tensor, deterministic: bool):
        means, logstds = self.actor(states)
        dist = Normal(means, torch.exp(logstds))
        if deterministic:
            actions = means
        else:
            actions = dist.sample()

        # Clamp to action space.
        # ^^^ This actually should be done before being sent to the environment,
        # but the original action should be preserved. Otherwise, log probabilities
        # get too unstable.
        # actions = torch.clamp(actions, -1, 1)

        # Sum across all action dimensions.
        # Shape is [batch_size, num_action-dimensions]
        logprobs = dist.log_prob(actions).sum(-1)

        return (actions, logprobs)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)

    def value_target(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)


def ppo_update(
    optimizer: torch.optim.Optimizer,  # type: ignore
    model: ActorCritic,
    rollout: Rollout,
    clip_coefficient: float,
    vf_coef: float,
    entropy_coef: float,
    grad_clip_norm: float,
    target_kl: float | None,
    num_iterations: int,
    minibatch_size: int,
) -> dict:
    rollout_flat = rollout.flatten()

    # Check the rollout.
    for k in [
        "states",
        "actions",
        "rewards",
        "logprobs",
        "returns",
        "value_estimates",
        "advantages",
    ]:
        assert not torch.any(
            torch.isnan(getattr(rollout_flat, k))
        ), f"NaN in rollout_flat.{k}"

    losses = {
        "policy_loss": [],
        "vf_loss": [],
        "entropy": [],
        "approx_kl": [],
    }

    def check_na(value, label: str):
        assert not torch.any(torch.isnan(value)), f"NaN in {label}"

    for iter_num in range(num_iterations):
        # indices = torch.randperm(len(rollout_flat), device=device)
        indices = torch.arange(len(rollout_flat), device=device)

        for i in range(0, len(indices), minibatch_size):
            minibatch_indices = indices[i : i + minibatch_size]
            rollout_minibatch = rollout_flat[minibatch_indices]

            # print(
            #     (
            #         (rollout_minibatch.states[:, 0] - rollout_minibatch.logprobs) > 0
            #     ).any(),
            #     (
            #         (rollout_minibatch.actions[:, 0] - rollout_minibatch.logprobs) > 0
            #     ).any(),
            # )

            values_minibatch = model.value(rollout_minibatch.states)
            means, logstds = model.actor(rollout_minibatch.states)

            check_na(values_minibatch, "values_minibatch")
            check_na(means, "means")
            check_na(logstds, "logstds")
            check_na(torch.exp(logstds), "stddevs")

            dist = Normal(means, torch.exp(logstds))

            # Shape: [batch_size, num_action_dimensions]
            # Therefore, we sum logprobs and entropy across all action dimensions.
            logprobs = dist.log_prob(rollout_minibatch.actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            check_na(logprobs, "logprobs")
            check_na(entropy, "entropy")

            log_ratio = logprobs - rollout_minibatch.logprobs
            ratio = torch.exp(log_ratio)

            check_na(log_ratio, "log_ratio")
            check_na(ratio, "ratio")

            # See if we need to stop early because KL divergence is too high.
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()

                # if iter_num == 0 and i == 0:
                #     idx = (ratio - 1).abs().argmax()
                #     print(
                #         # "logprob=",dist.log_prob(rollout_minibatch.actions).sum(-1)[idx],
                #         # "logprob=",rollout_minibatch.logprobs[idx],
                #         f"{means[idx]=} {logstds[idx]=} {rollout_minibatch.actions[idx]=}",
                #     )
                #     print(
                #         f"{approx_kl=} {(ratio-1).abs().max()=} {log_ratio.abs().max()=}"
                #     )

                if target_kl is not None and approx_kl > target_kl:
                    break

            policy_loss = -torch.min(
                rollout_minibatch.advantages * ratio,
                rollout_minibatch.advantages
                * torch.clamp(ratio, 1 - clip_coefficient, 1 + clip_coefficient),
            ).mean()

            check_na(policy_loss, "policy_loss")

            vf_loss = F.mse_loss(
                values_minibatch, rollout_minibatch.returns, reduction="mean"
            )

            check_na(vf_loss, "vf_loss")

            means_excess = means - torch.clamp(means, -1, 1)
            logstds_excess = logstds - torch.clamp(logstds, -2, 2)
            excess_penalty = (means_excess**2 + logstds_excess**2).mean()

            loss = (
                policy_loss
                + vf_coef * vf_loss
                + -entropy_coef * entropy
                + excess_penalty
            )

            check_na(loss, "loss")

            optimizer.zero_grad()
            loss.backward()

            for param_name, param_value in model.named_parameters():
                if torch.any(torch.isnan(param_value)):
                    print(f"NaN in model.parameters::{param_name}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            losses["policy_loss"].append(policy_loss.item())
            losses["vf_loss"].append(vf_loss.item())
            losses["entropy"].append(entropy.item())
            losses["approx_kl"].append(approx_kl.item())

        # See if we need to stop early because KL divergence is too high.
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()  # type: ignore
            if target_kl is not None and approx_kl > target_kl:
                break

    # var_y = np.var(rollout.returns.cpu().numpy())
    # if var_y != 0:
    #     explained_variance = (
    #         1
    #         - np.var(
    #             rollout_flat.returns.cpu().numpy()
    #             - rollout_flat.value_estimates.detach().cpu().numpy()
    #         )
    #         / var_y
    #     )
    #     print(f"Explained variance: {explained_variance:.2f}")

    return losses


@torch.no_grad()
def run_rollout(
    env: gym.Env,
    model: ActorCritic,
    steps: int,
    deterministic: bool,
    gamma: float,
    lambda_: float,
    normalize_advantages: bool,
    finite_horizon_gae: bool,
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
        (actions_, logprobs_) = model.sample_action(obs, deterministic)
        # obs[:, 0] = torch.arange(len(obs))
        # logprobs_[:] = torch.arange(len(obs))
        # actions_[:, 0] = torch.arange(len(obs))

        values = model.value_target(obs)

        states.append(obs)
        actions.append(actions_)
        logprobs.append(logprobs_)

        value_estimates.append(values)

        # clip action. note though that the original action values must be preserved!
        actions_clamped = torch.clamp(actions_, -1, 1)
        obs, reward, terminated, truncated, info = env.step(actions_clamped)

        rewards.append(reward)
        dones.append(terminated | truncated)
        infos.append(info)

    # Add final state.
    states.append(obs)
    # Add final value.
    value_estimates.append(model.value_target(obs))

    states = torch.stack(states).float().detach()
    actions = torch.stack(actions).float().detach()
    rewards = torch.stack(rewards).float().detach()
    logprobs = torch.stack(logprobs).float().detach()
    dones = torch.stack(dones).bool().detach()
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
        finite_horizon_gae,
    )


def run_training():
    ENV = "PickCube-v1"
    SEED = 1
    NUM_ENVS = 512
    EPISODE_STEPS = 50
    EPOCHS = 10000
    CHECKPOINT_EVERY = 25
    PPO_ITERATIONS = 4
    VF_COEFFICIENT = 0.5
    VF_SYNCHRONIZE_BETA = 0.9
    CLIP_COEFFICIENT = 0.2
    ENTROPY_COEFFICIENT = 0.0
    GRADIENT_CLIPPING_NORM = 0.5
    NORMALIZE_ADVANTAGES = True
    MINIBATCH_SIZE = 800
    GAMMA = 0.8
    LAMBDA = 0.9
    TARGET_KL = 0.1
    FINITE_HORIZON_GAE = False

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

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
        ignore_terminations=False,
        record_metrics=True,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    assert isinstance(
        env.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    state_dim: int = env.single_observation_space.shape[0]  # type: ignore
    action_dim: int = env.single_action_space.shape[0]  # type: ignore
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-5)  # type: ignore

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
            "vf_synchronize_beta": VF_SYNCHRONIZE_BETA,
            "clip_coefficient": CLIP_COEFFICIENT,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "gradient_clipping_norm": GRADIENT_CLIPPING_NORM,
            "normalize_advantages": NORMALIZE_ADVANTAGES,
            "minibatch_size": MINIBATCH_SIZE,
            "finite_horizon_gae": FINITE_HORIZON_GAE,
            "gamma": GAMMA,
            "lambda": LAMBDA,
            "target_kl": TARGET_KL,
        },
    )

    global_step = 0

    with tqdm.tqdm(total=EPOCHS) as pbar:
        for epoch in range(EPOCHS):
            # model.eval()
            rollout_start = time.time()
            rollout = run_rollout(
                env,
                model,
                steps=EPISODE_STEPS,
                deterministic=False,
                gamma=GAMMA,
                lambda_=LAMBDA,
                normalize_advantages=NORMALIZE_ADVANTAGES,
                finite_horizon_gae=FINITE_HORIZON_GAE,
            )
            for done_mask, info in zip(rollout.dones, rollout.infos):
                global_step += NUM_ENVS
                if done_mask.any():
                    wandb.log(
                        {
                            "train/" + k: v[done_mask].float().mean()
                            for (k, v) in info["episode"].items()
                        },
                        global_step,
                    )

            # Run a pass over the data.
            # model.synchronize_critics(VF_SYNCHRONIZE_BETA)
            # model.train()
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

            rollout_end = time.time()
            steps_per_second = (EPISODE_STEPS * NUM_ENVS) / (
                rollout_end - rollout_start
            )

            pbar.set_postfix(
                # {"train/returns": train_returns, "train/rewards": train_rewards}
                {"sps": steps_per_second}
            )
            pbar.update()

            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), f"runs/{out_dir}/ckpt_{epoch + 1}.pt")


def eval():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ENV = "PickCube-v1"
    NUM_ENVS = 100
    DO_RECORDING = False
    # NUM_ENVS = 1
    # DO_RECORDING = True

    assert not DO_RECORDING or NUM_ENVS == 1, "NUM_ENVS must be 1 if recording."

    env = gym.make(
        ENV,
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        obs_mode="state",
        num_envs=NUM_ENVS,
    )
    if DO_RECORDING:
        env = RecordEpisode(
            env,  # type: ignore
            output_dir="Videos_test",
            save_trajectory=False,
            video_fps=30,
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
    model.load_state_dict(torch.load("runs/144/ckpt_200.pt", weights_only=True))

    rollout = run_rollout(
        env,
        model,
        steps=50,
        deterministic=True,
        gamma=0,
        lambda_=0,
        normalize_advantages=False,
        finite_horizon_gae=False,
    )
    stats = {}
    for done_mask, info in zip(rollout.dones, rollout.infos):
        if done_mask.any():
            for k, v in info["episode"].items():
                if k not in stats:
                    stats[k] = []

                stats[k].extend(v[done_mask].detach().cpu().tolist())

    for k in stats.keys():
        print(f"{k}: {sum(stats[k]) / len(stats[k]):.3f} ({NUM_ENVS} trials)")


if __name__ == "__main__":
    match sys.argv[1]:
        case "train":
            run_training()
        case "eval":
            eval()
        case _:
            raise ValueError("Invalid argument")
