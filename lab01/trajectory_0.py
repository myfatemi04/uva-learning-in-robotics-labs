# Import required packages
import gymnasium as gym
# import mani_skill.envs
import time
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode

env = gym.make("PickCube-v1", render_mode = "rgb_array", sim_backend="gpu", control_mode="pd_ee_delta_pos", obs_mode="state_dict")
print(env.agent._controller_configs.keys())
env = RecordEpisode(env, output_dir="Videos", save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=100)
# print(env.action_space)
obs, _ = env.reset(seed=0)
env.unwrapped.print_sim_details() # print verbose details about the configuration
done = False
truncated = False
start_time = time.time()
start_pos = obs['extra']['tcp_pose'][0,:3].detach().cpu()
print(start_pos)
# target_pos = 0.0, 0.4, 0.0
# target_pos = start_pos
# tx,ty,tz = target_pos
tx, ty, tz, *_ = obs['extra']['obj_pose'][0].detach().cpu()

# we can get the model and compute forward kinematics to get the ee pos.

while not done and not truncated:
    # print(obs['agent']['qpos'])
    # random_action = env.action_space.sample()
    # random_action = np.array([0,0,0,1])
    x, y, z, *_ = obs['extra']['tcp_pose'][0].detach().cpu() #obs['agent']['qpos'][0].detach().cpu()
    # print(obs['extra']['tcp_pose'])
    # print(x, y, z)
    random_action = np.array([tx-x,ty-y,tz-z,0])# ,0,0,0,0])
    # print(random_action)
    obs, rew, done, truncated, info = env.step(random_action)
env.close()
N = info["elapsed_steps"].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
