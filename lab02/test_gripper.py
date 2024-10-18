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
from PPO import Policy, run_rollout

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

NUM_ENVS = 128
ENV = "PickCube-v1"
STATE_DIM = 42
ACTION_DIM = 8


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
    env = RecordEpisode(
        env,
        output_dir="Videos_gripper_test",
        save_trajectory=False,
        save_video=True,
        video_fps=30,
        max_steps_per_video=100,
    )
    policy = Policy().to(device)
    policy.load_state_dict(torch.load("runs/44/ckpt_7200.pt", weights_only=True))

    success = 0
    for i in range(100):
        states, actions, rewards, truncated = run_rollout(
            env, policy, deterministic=True
        )
        success += (~truncated).sum().item()
        print((~truncated).sum().item(), rewards.sum().item())

    print(success / 100)


eval()
