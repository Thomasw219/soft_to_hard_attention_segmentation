from abc import ABC, abstractmethod

import numpy as np
import gym
import d4rl

class OrnsteinUhlenbeckProcess:
    def __init__(self, dim, theta, sigma, dt, mu=0):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mu = mu
        self.reset()

    def step(self):
        self.x_t = (
            self.x_t
            + self.theta * (self.mu - self.x_t) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)
        )
        return self.x_t

    def reset(self, x_t=None):
        self.x_t = x_t if x_t is not None else np.random.randn(self.dim) * self.sigma + self.mu

class PiecewiseSineBase(ABC):
    def __init__(
        self,
        signal_length=128,
        dataset_size=1000,
    ):
        self.signal_length = signal_length
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

class FixedSizePiecewiseSine(PiecewiseSineBase):
    def __init__(
        self,
        signal_length=128,
        piece_length=20,
        dataset_size=1000,
    ):
        super().__init__(signal_length=signal_length, dataset_size=dataset_size)
        self.piece_length = piece_length

    def __getitem__(self, index):
        signal = np.empty(self.signal_length)
        idx = 0
        while idx < self.signal_length:
            l = np.minimum(self.piece_length, self.signal_length - idx)
            c = np.random.randint(0, 2)
            if c == 0:
                signal[idx:idx+l] = np.sin(np.linspace(0, 2 * np.pi, self.piece_length))[:l]
            elif c == 1:
                signal[idx:idx+l] = 0
            idx += l
        return signal.reshape((self.signal_length, 1))

class RandomSizePiecewiseSine(PiecewiseSineBase):
    def __init__(
        self,
        signal_length=128,
        piece_length=20,
        dataset_size=1000,
    ):
        super().__init__(signal_length=signal_length, dataset_size=dataset_size)
        self.piece_length = piece_length

    def __getitem__(self, index):
        signal = np.empty(self.signal_length)
        idx = 0
        while idx < self.signal_length:
            sine_length = np.random.randint(10, 30)
            l = np.minimum(sine_length, self.signal_length - idx)
            c = np.random.randint(0, 2)
            if c == 0:
                signal[idx:idx+l] = np.sin(np.linspace(0, 2 * np.pi, sine_length + 1))[:l]
            elif c == 1:
                signal[idx:idx+l] = 0
            idx += l
        return signal.reshape((self.signal_length, 1))

class SinusoidAndRandom(PiecewiseSineBase):
    def __init__(
        self,
        signal_length=128,
        dataset_size=1000,
    ):
        super().__init__(signal_length=signal_length, dataset_size=dataset_size)

    def __getitem__(self, index):
        signal = np.empty(self.signal_length)
        idx = 0
        while idx < self.signal_length:
            s_l = np.random.randint(25, 40)
            l = np.minimum(s_l, self.signal_length - idx)
            c = np.random.randint(0, 3)
            if c == 0:
                signal[idx:idx+l] = np.random.uniform(-1, 1) * np.sin(np.linspace(0, np.random.choice([0, 1, 2, 3, 4]) * np.pi, s_l + 1) + np.random.choice([0, np.pi]))[:l]
            elif c == 1:
                signal[idx:idx+l] = 0
            elif c == 2:
                proc = OrnsteinUhlenbeckProcess(1, 0.03, 0.1, 3)
                proc.reset(x_t=0)
                for i in range(l):
                    signal[idx + i] = proc.step()
            idx += l
        return signal.reshape((self.signal_length, 1))

class ComplicatedSinusoid(PiecewiseSineBase):
    def __init__(
        self,
        signal_length=128,
        dataset_size=1000,
    ):
        super().__init__(signal_length=signal_length, dataset_size=dataset_size)

    def __getitem__(self, index):
        signal = np.empty(self.signal_length)
        idx = 0
        while idx < self.signal_length:
            s_l = np.random.randint(25, 40)
            l = np.minimum(s_l, self.signal_length - idx)
            c = np.random.randint(0, 2)
            if c == 0:
                signal[idx:idx+l] = np.random.uniform(-1, 1) * np.sin(np.linspace(0, np.random.choice([0, 1, 2, 3, 4]) * np.pi, s_l + 1) + np.random.choice([0, np.pi]))[:l]
            elif c == 1:
                signal[idx:idx+l] = 0
            idx += l
        return signal.reshape((self.signal_length, 1))

class PiecewiseLinear(PiecewiseSineBase):
    def __init__(
        self,
        signal_length=128,
        dataset_size=1000,
    ):
        super().__init__(signal_length=signal_length, dataset_size=dataset_size)
        self.indices = [0, 1, 2, 3, 4, 5]
        self.probs = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
        self.values = [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]
        self.lengths = [3, 8, 20, 3, 6, 30]

    def __getitem__(self, index):
        signal = np.empty(self.signal_length)
        idx = 0
        while idx < self.signal_length:
            lin_idx = np.random.choice(self.indices, p=self.probs)
            val = self.values[lin_idx]
            l = np.minimum(self.lengths[lin_idx], self.signal_length - idx)
            signal[idx:idx+l] = val
            idx += l
        return signal.reshape((self.signal_length, 1))

class SimplePiecewiseLinear(PiecewiseSineBase):
    def __init__(
        self,
        signal_length=128,
        dataset_size=1000,
    ):
        super().__init__(signal_length=signal_length, dataset_size=dataset_size)
        self.indices = [0, 1, 2]
        self.probs = [0.5, 0.3, 0.2]
        self.values = [1.0, 0.0, -1.0]
        self.lengths = [3, 3, 3]

    def __getitem__(self, index):
        signal = np.empty(self.signal_length)
        idx = 0
        while idx < self.signal_length:
            lin_idx = np.random.choice(self.indices, p=self.probs)
            val = self.values[lin_idx]
            l = np.minimum(self.lengths[lin_idx], self.signal_length - idx)
            signal[idx:idx+l] = val
            idx += l
        return signal.reshape((self.signal_length, 1))

class GeneratedD4RLDataset:
    def __init__(
            self,
            signal_length=128,
            data_path='data/raw/maze2d-medium-v1.npy',
    ):
        self.signal_length = signal_length
        self.episodes = np.load(data_path, allow_pickle=True)
        self.n_episodes = len(self.episodes)
        self.obs_dim = self.episodes[0]['observations'].shape[-1]
        self.action_dim = self.episodes[0]['actions'].shape[-1]

    def __getitem__(self, index):
        ep = self.episodes[index]
        start_index = np.random.randint(0, ep['observations'].shape[0] - self.signal_length)
        return ep['observations'][start_index:start_index+self.signal_length], ep['actions'][start_index:start_index+self.signal_length]

    def __len__(self):
        return self.n_episodes

class D4RLDataset:
    def __init__(
            self,
            signal_length=128,
            dataset_name='antmaze-large-diverse-v0'
    ):
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
        assert dataset_name == 'antmaze-large-diverse-v0'
        episode_points = [0]
        episode_points.extend([1001 + i for i in range(0, 1000000 - 1000, 1001)])
        self.signal_length = signal_length
        self.episodes = [{k : v[episode_start:episode_end] for k, v in dataset.items()} for episode_start, episode_end in zip(episode_points[:-1], episode_points[1:])]
        self.obs_dim = self.episodes[0]['observations'].shape[-1]
        self.action_dim = self.episodes[0]['actions'].shape[-1]

    def __getitem__(self, index):
        ep = self.episodes[index]
        start_index = np.random.randint(0, ep['observations'].shape[0] - self.signal_length)
        return ep['observations'][start_index:start_index+self.signal_length], ep['actions'][start_index:start_index+self.signal_length]

    def __len__(self):
        return len(self.episodes)

def test_fixed_size_piecewise_sine():
    dataset = FixedSizePiecewiseSine()

    from torch.utils.data import DataLoader
    from time import time
    np.random.seed(0)
    t = time()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(time() - t)
    # print(batch)
    # print(batch.shape)
    # print(batch[0, :, 0])

def test_maze2d_dataset():
    dataset = GeneratedD4RLDataset()

    from torch.utils.data import DataLoader
    from time import time
    np.random.seed(0)
    t = time()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    obs = next(iter(dataloader))
    print(time() - t)
    print(obs)

def test_d4rl_dataset():
    dataset = D4RLDataset()

    from torch.utils.data import DataLoader
    from time import time
    np.random.seed(0)
    t = time()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    obs, act = next(iter(dataloader))
    print(time() - t)
    print(obs, act)

if __name__ == '__main__':
    # test_fixed_size_piecewise_sine()
    # test_maze2d_dataset()
    test_d4rl_dataset()