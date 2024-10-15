# Import required packages
import gymnasium as gym
import os
# import mani_skill.envs
import time
import numpy as np
from mani_skill.utils.wrappers import RecordEpisode
import matplotlib.pyplot as plt

for direction in ['up', 'forward', 'right']:
    env = gym.make("PickCube-v1", render_mode = "rgb_array", sim_backend="gpu", control_mode="pd_ee_delta_pos", obs_mode="state_dict")
    env = RecordEpisode(env, output_dir="Videos", save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=100)
    # print(env.action_space)
    obs, _ = env.reset(seed=0)
    env.unwrapped.print_sim_details() # print verbose details about the configuration
    done = False
    truncated = False
    start_time = time.time()
    start_pos = obs['extra']['tcp_pose'][0,:3].detach().cpu()

    # tx, ty, tz, *_ = obs['extra']['obj_pose'][0].detach().cpu()
    tx = start_pos[0] + 0.25 * (direction == 'forward')
    ty = start_pos[1] + -0.25 * (direction == 'right')
    tz = start_pos[2] + 0.25 * (direction == 'up')

    joint_poses = np.zeros((9, 50))
    i = 0

    while not done and not truncated:
        x, y, z, *_ = obs['extra']['tcp_pose'][0].detach().cpu()
        random_action = np.array([tx-x,ty-y,tz-z,0])
        obs, rew, done, truncated, info = env.step(random_action)
        
        joint_poses[:, i] = obs['agent']['qpos'][:, :].detach().cpu().numpy()
        i += 1

    env.close()

    # Display joint poses.
    os.rename("Videos/0.mp4", "Videos/" + direction + ".mp4")

    plt.clf()
    plt.figure(figsize=(8, 4), dpi=200)
    names = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
        'panda_finger1',
        'panda_finger2',
    ]
    for i in range(9):
        plt.plot(joint_poses[i, :], label=names[i])
    plt.tight_layout()
    plt.legend()
    plt.savefig(direction + "_joint_plot.png")

    N = info["elapsed_steps"].item()
    dt = time.time() - start_time
    FPS = N / (dt)
    print(f"Frames Per Second = {N} / {dt} = {FPS}")
