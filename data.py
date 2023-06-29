from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision import datasets, transforms
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
        self.signal_length = signal_length
        env = gym.make(dataset_name)
        dataset = env.get_dataset()
        if dataset_name == 'antmaze-large-diverse-v0':
            episode_points = [0]
            episode_points.extend([1001 + i for i in range(0, 1000000 - 1000, 1001)])
        elif dataset_name == 'kitchen-mixed-v0' or dataset_name == 'kitchen-partial-v0':
            episode_points = [0]
            episode_points.extend((np.arange(136950)[dataset['terminals']] + 1).tolist())
        else:
            raise NotImplementedError()
        self.episodes = [{k : v[episode_start:episode_end] for k, v in dataset.items()} for episode_start, episode_end in zip(episode_points[:-1], episode_points[1:])]

        min_len = np.inf
        for episode in self.episodes:
            terminations = episode['terminals']
            length = terminations.shape[0]
            min_len = np.minimum(min_len, length)
            assert length > self.signal_length
        print("Min length: ", min_len)

        self.obs_dim = self.episodes[0]['observations'].shape[-1]
        self.action_dim = self.episodes[0]['actions'].shape[-1]

    def get_episode(self, index):
        return self.episodes[index]

    def __getitem__(self, index):
        ep = self.episodes[index]
        start_index = np.random.randint(0, ep['observations'].shape[0] - self.signal_length)
        return ep['observations'][start_index:start_index+self.signal_length], ep['actions'][start_index:start_index+self.signal_length]

    def __len__(self):
        return len(self.episodes)

class StochasticMovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train=True, data_root='./data/moving_mnist',
                    obs_len=64, num_digits=2, context_len=10, image_size=64, deterministic=True, img_transforms=None, channel_first=True):
        path = data_root
        self.seq_len = obs_len + context_len
        self.context_len = context_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = 32
        self.gap_size = image_size - self.digit_size
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1
        self.transforms = img_transforms
        self.channel_first = channel_first

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        c = np.zeros((self.seq_len, 2 * self.num_digits), dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                c[t, 2 * n] = sx
                c[t, 2 * n + 1] = sy
                sy += dy
                sx += dx

        x[x>1] = 1.
        if self.transforms is not None:
            x = self.transforms(x)
        if self.channel_first:
            x = x.transpose(0, 3, 1, 2)
        return x[self.context_len:], x[:self.context_len], c[self.context_len:], c[:self.context_len]

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
    dataset = D4RLDataset(dataset_name='kitchen-mixed-v0')

    from torch.utils.data import DataLoader
    from time import time
    np.random.seed(0)
    t = time()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    obs, act = next(iter(dataloader))
    print(time() - t)
    print(obs.shape, act.shape)
    print(obs[0, :, 30:])

def test_moving_mnist():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = StochasticMovingMNIST(
        train=True,
        seq_len=110,
        image_size=64,
        deterministic=False,
        num_digits=1
        )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # visualize the stochastic moving mnist dataset
    for _, seq in enumerate(data_loader):
        for i in range(seq.shape[1]):
            img = seq[0, i, ...]
            plt.imshow(img)
            plt.pause(0.5)
            # plt.imsave(f"data/moving_mnist_test/{i}.png", img.squeeze().numpy(), cmap='gray')
        break

if __name__ == '__main__':
    # test_fixed_size_piecewise_sine()
    # test_maze2d_dataset()
    test_d4rl_dataset()