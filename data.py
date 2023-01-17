from abc import ABC, abstractmethod

import numpy as np

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
                signal[idx:idx+l] = np.sin(np.linspace(0, 2 * np.pi, l))
            elif c == 1:
                signal[idx:idx+l] = 0
            idx += l
        return signal.reshape((self.signal_length, 1))

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

if __name__ == '__main__':
    test_fixed_size_piecewise_sine()
