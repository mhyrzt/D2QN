
from ctypes.wintypes import tagRECT
from os import stat
import gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import signal

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from model import Model
from epsilon import Epsilon
from reply_buffer import ReplyBuffer

GAMMA = 0.99
MAX_LEN = 5_000
BATCH_SIZE = 128
UPDATE_EVERY = 25

EPISODES = 1_000
LOG_EACH = 5
RENDER_EACH = 100
BREAK = False


env = gym.make("LunarLander-v2")
buffer = ReplyBuffer(MAX_LEN, BATCH_SIZE)
nA = env.action_space.n
nS = env.observation_space.shape[0]

online_model = Model(nS, nA)
target_model = online_model.copy()

epsilon = Epsilon(env, online_model, 1, 0.995)
optimizer = optim.Adam(online_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

history = {
    "action_loss": [],
    "episode_reward": [],
    "episode_epsilon": []
}

def handler(signum, frame):
    global BREAK
    BREAK = True
signal.signal(signal.SIGINT, handler)

for episode in range(EPISODES):
    s = env.reset()
    done = False
    episode_reward = 0

    while not done:
        if episode % RENDER_EACH == 0:
            env.render()
        
        a = epsilon.take_action(s)
        ns, r, done, info = env.step(a)

        buffer.add(s, a, r, ns, float(done))
        s = ns
        episode_reward += r

        if not buffer.can_sample():
            continue
        
        # SAMPLING
        states, actions, rewards, next_states, dones = buffer.sample()
        dones = torch.Tensor(dones).to(online_model.device).unsqueeze(1)
        rewards = torch.Tensor(rewards).to(online_model.device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(online_model.device).unsqueeze(1)
        
        # ESTIMATING TARGET VALUES (Y)
        arg_max_q_online = online_model(next_states).max(1)[1].unsqueeze(1).detach()
        target_q_values = target_model(next_states).gather(1, arg_max_q_online).detach()
        target_q_values = rewards + target_q_values * GAMMA * (1 - dones)
        
        # CALCULATING CURRENT VALUES (X)
        current_q_values = online_model(states).gather(1, actions)
        
        loss = (current_q_values - target_q_values).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                
        
    
    if episode % UPDATE_EVERY == 0:
        target_model.load_state_dict(online_model.state_dict())

    history["episode_reward"].append(episode_reward)
    history["episode_epsilon"].append(epsilon.epsilon)
    
    if episode % LOG_EACH == 0:
        sep = "\t\t"
        print(f"#{episode}{sep}reward={episode_reward}{sep}eps={epsilon.epsilon}")
    
    epsilon.decrease()
    
    
    if BREAK:
        break

env.close()
plt.plot(history["episode_reward"])
plt.title("Rewards")
plt.grid()
plt.show()