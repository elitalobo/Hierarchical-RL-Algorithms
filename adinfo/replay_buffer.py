import torch
import random
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # else:
        #     return
        self.buffer[self.position] = (
            state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size,flag=False):
        batch = random.sample(self.buffer, batch_size)
        if flag==True:
            batch = self.buffer[0:batch_size]
        #batch = self.buffer[0:batch_size]
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def clear(self):
        del self.buffer
        self.buffer = []
        self.position=0

    def __len__(self):
        return len(self.buffer)

class ReplayBufferWeighted:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done,p):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # else:
        #     return
        self.buffer[self.position] = (
            state, action, reward, next_state, done,p)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size,flag=False):
        batch = random.sample(self.buffer, batch_size)
        if flag==True:
            batch = self.buffer[0:batch_size]
        #batch = self.buffer[0:batch_size]
        state, action, reward, next_state, done,p = map(
            np.stack, zip(*batch))

        return state, action, reward, next_state, done,p

    def clear(self):
        del self.buffer
        self.buffer = []
        self.position=0

    def __len__(self):
        return len(self.buffer)