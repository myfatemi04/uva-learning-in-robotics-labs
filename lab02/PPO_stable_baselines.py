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
from stable_baselines3 import PPO

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

NUM_ENVS = 128
env = gym.make(
    "StackCube-v1",
    render_mode="rgb_array",
    sim_backend="gpu",
    control_mode="pd_joint_delta_pos",
    obs_mode="state",
)
env = ManiSkillVectorEnv(
    env,
    auto_reset=False,
    ignore_terminations=False,
)
orig_reset = env.reset
orig_step = env.step
def alt_reset(*args, **kwargs):
    obs, arg = orig_reset(*args, **kwargs)
    return obs.detach().cpu().numpy(), arg
def alt_step(*args, **kwargs):
    obs, *args = orig_step(*args, **kwargs)
    return obs.detach().cpu().numpy(), *args
env.reset = alt_reset
env.step = alt_step

model = PPO("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()