import collections
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch

class SchedulerBase(ABC):
    @abstractmethod
    def get_value(self, step):
        raise NotImplementedError()

class LinearScheduler(SchedulerBase):
    def __init__(self, start_value, end_value, start_step, end_step, **kwargs):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def get_value(self, step):
        if step < self.start_step:
            return self.start_value
        elif step > self.end_step:
            return self.end_value
        elif step == self.start_step and step == self.end_step:
            return self.end_value
        else:
            return self.start_value + (self.end_value - self.start_value) * (step - self.start_step) / (self.end_step - self.start_step)

class LogarithmicScheduler(SchedulerBase):
    def __init__(self, start_value, end_value, start_step, end_step, **kwargs):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def get_value(self, step):
        if step < self.start_step:
            return self.start_value
        elif step > self.end_step:
            return self.end_value
        elif step == self.start_step and step == self.end_step:
            return self.end_value
        else:
            return self.start_value * (self.end_value / self.start_value) ** ((step - self.start_step) / (self.end_step - self.start_step))

class GumbelSoftmaxScheduler(SchedulerBase):
    def __init__(self, start_step, N, r, max_temp, min_temp, **kwargs):
        self.start_step = start_step
        self.N = N
        self.r = r
        self.max_temp = max_temp
        self.min_temp = min_temp

    def get_value(self, step):
        if step < self.start_step:
            return self.max_temp
        else:
            return np.maximum(self.max_temp * np.exp(-self.r * np.floor((step - self.start_step) / self.N) * self.N), self.min_temp)

def make_scheduler(cfg):
    if cfg['type'] == 'linear':
        return LinearScheduler(**cfg)
    elif cfg['type'] == 'logarithmic':
        return LogarithmicScheduler(**cfg)
    elif cfg['type'] == 'gumbel_softmax':
        return GumbelSoftmaxScheduler(**cfg)
    else:
        raise ValueError(f'Unknown scheduler type: {cfg["type"]}')

def get_optimizer(cfg, model):
    if cfg['type'] == 'adam':
        return torch.optim.Adam(model.parameters(), **cfg['params'])
    elif cfg['type'] == 'sgd':
        return torch.optim.SGD(model.parameters(), **cfg['params'])
    elif cfg['type'] == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), **cfg['params'])
    else:
        raise NotImplementedError(f"Optimizer type {cfg['type']} not implemented")

def make_video(frames,name):
    writer = imageio.get_writer(name+'.mp4', fps=20)

    for im in frames:
        writer.append_data(im)
    writer.close()

def flatten(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scheduler = make_scheduler({
        'type': 'gumbel_softmax',
        'start_step': 0,
        'N': 1000,
        'r': 1.0e-5,
        'max_temp': 0.8,
        'min_temp': 0.33,
    })

    x = np.arange(600000)
    y = np.array([scheduler.get_value(i) for i in x])

    plt.plot(x, y)
    plt.savefig("figures/scheduler.png")
