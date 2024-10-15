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
tx, ty, tz, *_ = obs['extra']['obj_pose'][0].detach().cpu()

xyz_poses = np.zeros((3, 50))
i = 0

# while not done and not truncated:
for _ in range(25):
    x, y, z, *_ = obs['extra']['tcp_pose'][0].detach().cpu()
    random_action = np.array([tx-x,ty-y,tz-z,1]) * 2
    obs, rew, done, truncated, info = env.step(random_action)
    
    xyz_poses[:, i] = (x, y, z)
    i += 1

tx, ty, tz = [0,0,0.4]
for _ in range(25):
    x, y, z, *_ = obs['extra']['tcp_pose'][0].detach().cpu()
    random_action = np.array([0,0,0.1,-1])
    obs, rew, done, truncated, info = env.step(random_action)
    
    xyz_poses[:, i] = (x, y, z)
    i += 1

env.close()

# Display joint poses.
os.rename("Videos/0.mp4", "Videos/pick.mp4")

print(xyz_poses[1])

plt.clf()
fig = plt.figure(figsize=(4, 4), dpi=200)
ax = plt.axes(projection='3d')
ax.set_title("XYZ for 'Pick up cube'")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.01, 0.2)
ax.plot3D(*xyz_poses)
plt.savefig("xyz_pick.png")

N = info["elapsed_steps"].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
