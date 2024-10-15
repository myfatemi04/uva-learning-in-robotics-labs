# Import required packages
import gymnasium as gym
import mani_skill.envs
import time
from mani_skill.utils.wrappers import RecordEpisode
# env = gym.make("PegInsertionSide-v1", render_mode = "rgb_array", control_mode="pd_joint_delta_pos", sim_backend="gpu")
env = gym.make("AnymalC-Spin-v1", render_mode = "rgb_array", control_mode="pd_joint_delta_pos", sim_backend="gpu")
env = RecordEpisode(env, output_dir="Videos", save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=100)
obs, _ = env.reset(seed=0)
env.unwrapped.print_sim_details() # print verbose details about the configuration
done = False
truncated = False
start_time = time.time()
while not done and not truncated:
    random_action = env.action_space.sample()
    obs, rew, done, truncated, info = env.step(random_action)
env.close()
N = info["elapsed_steps"].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
