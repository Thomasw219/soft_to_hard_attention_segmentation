from abc import ABC, abstractmethod

import numpy as np

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

def make_scheduler(cfg):
    if cfg['type'] == 'linear':
        return LinearScheduler(**cfg)
    elif cfg['type'] == 'logarithmic':
        return LogarithmicScheduler(**cfg)
    else:
        raise ValueError(f'Unknown scheduler type: {cfg["type"]}')
