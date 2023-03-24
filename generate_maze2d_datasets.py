import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import os
import gzip
import h5py
import argparse


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for episode in data:
        for k in episode:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32

            episode[k] = np.array(episode[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)



    t = 0
    episodes = []
    while t < args.num_samples:
        data = reset_data()
        env.set_target()
        s = env.reset()
        act = env.action_space.sample()
        done = False
        ts = 0
        for _ in range(args.num_samples):
            position = s[0:2]
            velocity = s[2:4]
            act, done = controller.get_action(position, velocity, env._target)
            if args.noisy:
                act = act + np.random.randn(*act.shape)*0.5

            act = np.clip(act, -1.0, 1.0)
            if ts >= max_episode_steps:
                done = True
            append_data(data, s, act, env._target, done, env.sim.data)

            ns, _, _, _ = env.step(act)

            ts += 1
            t += 1
            if done:
                done = False
                env.set_target()
            else:
                s = ns

            if ts == max_episode_steps:
                break

            if args.render:
                env.render()
        episodes.append(data)
        print(len(episodes))
        print(t)

    os.makedirs('data/raw', exist_ok=True)
    if args.noisy:
        fname = 'data/raw/%s-noisy.hdf5' % args.env_name
    else:
        fname = 'data/raw/%s.npy' % args.env_name
    npify(episodes)
    np.save(fname, episodes, allow_pickle=True)

if __name__ == "__main__":
    main()