import gym
import d4rl
import numpy as np
import matplotlib.pyplot as plt
import torch

import time

from data import D4RLDataset
from utils import make_video

def segment_episode(model, episode, device=torch.device('cpu')):
    ep_len = len(episode['observations'])
    termination_probs = np.zeros(ep_len - 1)
    samples = np.zeros(ep_len - 1)
    for i in range(ep_len - model.max_seq_len + 1):
        obs = torch.tensor(episode['observations'][i:i+model.max_seq_len], device=device).unsqueeze(0)
        actions = torch.tensor(episode['actions'][i:i+model.max_seq_len], device=device).unsqueeze(0)
        _, info = model.forward(obs, actions)
        np_probs = torch.sigmoid(info['segmentation_post_logits']).detach().cpu().numpy().squeeze()
        termination_probs[i:i+model.max_seq_len-1] += np_probs
        samples[i:i+model.max_seq_len-1] += np.ones_like(np_probs)

    probs = termination_probs / samples
    return np.concatenate([probs, np.array([1.0])], axis=0)

env = gym.make('kitchen-mixed-v0')
device = torch.device('cuda:0')

full_model_path = "logs/kitchen/kitchen_policy_segmentation_04-07-2023_20-08-25/full_model.pt"
model = torch.load(full_model_path, map_location=device)

np.random.seed(0)

env_name = 'kitchen-mixed-v0'
dataset = D4RLDataset(dataset_name=env_name)

for _ in range(10):
    episode_idx = np.random.randint(0, len(dataset))
    t = time.time()
    episode = dataset.get_episode(episode_idx)

    with torch.no_grad():
        probs = segment_episode(model, episode, device=device)
    plt.plot(probs)
    plt.savefig(f"media/figures/probs_{episode_idx}.png")
    plt.clf()

    segmentations = probs > 0.5

    segmentation_indices = np.where(segmentations, np.arange(len(segmentations)), np.inf)
    terminal_indices = np.flip(np.minimum.accumulate(np.flip(segmentation_indices)))

    frames = []

    for i in range(len(episode['observations'])):
        env.env.set_state(episode['observations'][i][:30], np.zeros_like(episode['observations'][i][:29]))
        frame = env.render(mode='rgb_array')
        frames.append(frame)

    terminal_state_frames = np.array(frames)[np.array(terminal_indices, dtype=np.int32)]

    make_video(terminal_state_frames, f'media/videos/{env_name}_{episode_idx}_terminal_states')
    # make_video(frames, f'media/videos/{env_name}_{episode_idx}')
    print(f"Episode {episode_idx} took {time.time() - t} seconds")

print("Done")
