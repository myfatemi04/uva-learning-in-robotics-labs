# Import required packages
import gymnasium as gym
import os
# import mani_skill.envs
import time
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode
import matplotlib.pyplot as plt

env = gym.make("PickCube-v1", render_mode = "rgb_array", sim_backend="gpu", control_mode="pd_ee_delta_pos", obs_mode="state_dict")
env = RecordEpisode(env, output_dir="Videos", save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=100)
# print(env.action_space)
obs, _ = env.reset(seed=0)
env.unwrapped.print_sim_details() # print verbose details about the configuration
done = False
truncated = False
start_time = time.time()
tx, ty, tz, *_ = obs['extra']['obj_pose'][0].detach().cpu().numpy()

print("Cube position:", tx, ty, tz)

env.close()