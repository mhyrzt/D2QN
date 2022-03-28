import random
import numpy as np
from collections import deque


class ReplyBuffer:
    def __init__(self, max_len, batch_size):
        self.dones = deque(maxlen=max_len)
        self.actions = deque(maxlen=max_len)
        self.rewards = deque(maxlen=max_len)
        self.next_states = deque(maxlen=max_len)
        self.current_states = deque(maxlen=max_len)

        self.maxlen = max_len
        self.batch_size = batch_size

    def add(self, current_state, action, reward, next_state, done):
        self.dones.append(done)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.current_states.append(current_state)

    def _get_ids(self):
        n = range(len(self.current_states))
        return random.sample(n, self.batch_size)

    def _select_from(self, arr, ids):
        return np.array([arr[i] for i in ids])

    def sample(self):
        ids = self._get_ids()

        dones = self._select_from(self.dones, ids)
        actions = self._select_from(self.actions, ids)
        rewards = self._select_from(self.rewards, ids)
        next_states = self._select_from(self.next_states, ids)
        current_states = self._select_from(self.current_states, ids)

        return (current_states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.current_states)

    def is_filled(self):
        return len(self) >= self.maxlen

    def can_sample(self):
        return len(self) >= self.batch_size
