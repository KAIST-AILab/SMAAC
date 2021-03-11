import os
import random
import numpy as np
import csv
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(list(self.memory), n)
