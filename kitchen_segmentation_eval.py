import gym
import d4rl
import numpy as np

from data import D4RLDataset
from utils import make_video

env = gym.make('kitchen-mixed-v0')

dataset = D4RLDataset(dataset_name='kitchen-mixed-v0')

episode_idx = 0
episode = dataset.get_episode(episode_idx)

frames = []

for i in range(len(episode['observations'])):
    print(i)
    env.env.set_state(episode['observations'][i][:30], np.zeros_like(episode['observations'][i][:29]))
    frame = env.render(mode='rgb_array')
    frames.append(frame)

make_video(frames, 'media/videos/kitchen_{episode_idx}')

print("Done")
