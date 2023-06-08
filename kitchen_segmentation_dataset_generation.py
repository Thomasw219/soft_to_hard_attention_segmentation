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

np.random.seed(1)

env_name = 'kitchen-mixed-v0'
dataset = D4RLDataset(dataset_name=env_name)

episodes_with_segmentation = []
total_timesteps = 0
for i in range(len(dataset)):
    t = time.time()
    episode = dataset.get_episode(i)

    with torch.no_grad():
        probs = segment_episode(model, episode, device=device)

    episode['segmentation_probs'] = probs

    episodes_with_segmentation.append(episode)
    total_timesteps += len(episode['observations'])
    print(f"Episode {i} took {time.time() - t} seconds, processed {len(episode['observations'])} timesteps, total timesteps {total_timesteps}")

np.save(f"./data/raw/{env_name}_segmented.npy", episodes_with_segmentation, allow_pickle=True)

print("Done")

