#! /usr/bin/env python
import math
from bisect import bisect_right
from torch.optim import Optimizer, Adam
import numpy as np

class LR:
    def __init__(self, optimizer=None, policy="Step", warmup_epoch=0,
                 warmup_start_lr=0.01, **kwargs):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.policy = policy
        if policy == 'Step':
            self.LR = StepLR(**kwargs)
        elif policy == 'MultiStep':
            self.LR = MultiStepLR(**kwargs)
        elif policy == 'MultiStepAbs':
            self.LR = MultiStepAbsLR(**kwargs)
        elif policy == 'Cosine':
            self.LR = CosineLR(**kwargs)
        else:
            raise ValueError()

        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

    def __getattr__(self, name):
        return getattr(self.LR, name)

    def step(self, epoch):
        if isinstance(self.optimizer, Adam):
            return 0.001
        lr = self.LR.get_lr(epoch)
        if epoch < self.warmup_epoch:
            lr_start = self.warmup_start_lr
            lr_end = self.LR.get_lr(self.warmup_epoch)
            alpha = (lr_end - lr_start) / self.warmup_epoch
            lr = epoch * alpha + lr_start

        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr

class StepLR:
    def __init__(self, base_lr=0.1, stepsize=None, gamma=0.1):
        self.base_lr = base_lr
        self.stepsize = stepsize
        self.gamma = gamma

    def get_lr(self, epoch):
        if self.stepsize is None or self.stepsize <= 0:
            return self.base_lr
        return self.base_lr * self.gamma ** (epoch // self.stepsize)

class MultiStepLR:
    def __init__(self, base_lr=0.1, milestones=[], gammas=0.1):
        self.base_lr = base_lr
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas] * len(milestones)
        assert len(gammas) == len(milestones)
        self.gammas = gammas

    def get_lr(self, epoch):
        section = bisect_right(self.milestones, epoch)
        return self.base_lr * np.prod(self.gammas[:section])

class MultiStepAbsLR:
    def __init__(self, base_lr=0.1, milestones=[], gammas=0.1):
        self.base_lr = base_lr
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas] * len(milestones)
        assert len(gammas) == len(milestones)
        self.gammas = gammas

    def get_lr(self, epoch):
        section = bisect_right(self.milestones, epoch)
        periods = [self.base_lr] + self.gammas
        return periods[section]

class CosineLR:
    def __init__(self, base_lr=0.1, max_epoch=0):
        self.base_lr = base_lr
        self.max_epoch = max_epoch

    def get_lr(self, epoch):
        if self.max_epoch <=0:
            return self.base_lr
        theta = math.pi * epoch / self.max_epoch
        return self.base_lr * (math.cos(theta) + 1.0) * 0.5

class Coef(object):
    def __init__(self, init=0.1, milestones=[], gammas=0.1, decay='step',
                 last_epoch=-1):
        super().__init__()

        self.init = self._val = init
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas]
        if len(gammas) == 1:
            gammas = gammas * len(milestones)
        self.gammas = gammas
        self.decay = decay
        self.last_epoch = last_epoch

    @property
    def val(self):
        return self._val

    def __item__(self, i):
        return self.step(i)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if len(self.milestones) == 0 or self.decay == 'constant':
            self._val = self.init

        # step decay
        elif self.decay == "step":
            section = bisect_right(self.milestones, epoch)
            f = self.init
            for s in range(section):
                f = f * self.gammas[s]
            self._val = f

        elif self.decay == "step_abs":
            section = bisect_right(self.milestones, epoch)
            periods = [self.init] + self.gammas
            self._val = periods[section]

        else:
            raise ValueError("Unknown decay method: ", self.decay)

        return self._val


def test():
    a = Step(init=0.1)
    for i in [-1, 0, 1]:
        a.last_epoch = i
        print(a.get())
    b = Step(init=0.1, stepsize=10)
    for i in [-1, 0, 5, 10, 15]:
        b.last_epoch = i
        print(b.get())
    c = MultiStep(init=0.1)
    for i in [-1, 0, 1]:
        c.last_epoch = i
        print(c.get())
    d = MultiStep(init=0.1, milestones=[10, 20], gammas=[0.3, 0.7])
    for i in [-1, 0, 5, 10, 15, 20, 25]:
        d.last_epoch = i
        print(d.get())
    e = MultiStepAbs(init=0.1)
    for i in [-1, 0, 1]:
        e.last_epoch = i
        print(e.get())
    f = MultiStepAbs(init=0.1, milestones=[10, 20], gammas=[0.3, 0.7])
    for i in [-1, 0, 5, 10, 15, 20, 25]:
        f.last_epoch = i
        print(f.get())

    # c = Coefficient(init=0.1, milestones=[10, 20], gammas=[0.03, 0.07])
    # for i in [-1, 0, 5, 10, 15, 20, 25]:
        # print(c.step(i))
    # c = Coefficient(init=0.1, milestones=[10, 20], gammas=[0.03, 0.07],
                    # decay='step_abs')
    # for i in [-1, 0, 5, 10, 15, 20, 25]:
        # print(c.step(i))

if __name__ == "__main__":
    test()
