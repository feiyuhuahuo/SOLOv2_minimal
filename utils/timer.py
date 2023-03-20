#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import pdb
import torch
import numpy as np

times = {}
times.setdefault('batch', [])
times.setdefault('data', [])
started = False  # Use for starting and stopping the timer
max_len = 60
time_this = time_last = 0.
reset_step = -1


def reset(length=100, reset_at=-1):
    global time_this, time_last
    global times, started, max_len
    global reset_step
    times = {}
    times.setdefault('batch', [])
    times.setdefault('data', [])
    time_this = time_last = 0.
    started = False
    max_len = length
    reset_step = reset_at


def start(step):
    global started, times
    if step == reset_step + 1:
        started = True

        for k, v in times.items():
            assert len(v) == 0, 'Error, time list is not empty when starting.'


def add_batch_time():
    global time_this, time_last
    time_this = time.time()
    if started:
        batch_time = time_this - time_last
        times['batch'].append(batch_time)

        inner_time = 0
        for k, v in times.items():
            if k not in ('batch', 'data'):
                inner_time += v[-1]

        times['data'].append(batch_time - inner_time)

    time_last = time_this


def get_times(time_name):
    return_time = []
    for name in time_name:
        return_time.append(np.mean(times[name]))

    return return_time


class counter:
    def __init__(self, name):
        self.name = name
        self.times = times
        self.started = started
        self.max_len = max_len

        for v in times.values():
            if len(v) >= self.max_len:  # pop the first item if the list is full
                v.pop(0)

    def __enter__(self):
        if self.started:
            torch.cuda.synchronize()
            self.times.setdefault(self.name, [])
            self.times[self.name].append(time.perf_counter())

    def __exit__(self, e, ev, t):
        if self.started:
            torch.cuda.synchronize()
            self.times[self.name][-1] = time.perf_counter() - self.times[self.name][-1]
