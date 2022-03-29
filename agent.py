from model import Model
from epsilon import Epsilon
import torch
import torch.nn as nn
import torch.optim as optim
from reply_buffer import ReplyBuffer


class Agent:
    def __init__(self, env, **kwargs):
        self.env = env
        self.gamma = kwargs.get("gamma", 0.99)
        self.hiddens = kwargs.get("hiddens", (32, 32, 32, 32))
        self.batch_size = kwargs.get("batch_size", 128)

        self.buffer = ReplyBuffer(kwargs.get(
            "max_len", 5_000), self.batch_size)

        self.online_model = Model(*self.get_dims())
        self.target_model = self.online_model.copy()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=1e-3)
        self.device = self.online_model.device

        self.epsilon = Epsilon(env, self.online_model, kwargs.get(
            "epsilon", 1), kwargs.get("decay", 0.999), kwargs.get("min_epsilon", 1e-3))

    def get_dims(self):
        nA = self.env.action_space.n
        nS = self.env.observation_space.shape[0]
        return nS, nA, self.hiddens

    def render(self):
        self.env.render()

    def explore(self, state):
        action = self.epsilon.take_action(state)
        next_state, reward, done, _ = self.env.step(action)
        self.buffer.add(state, action, reward, next_state, float(done))
        return next_state, done, reward

    def can_train(self):
        return self.buffer.can_sample()

    def update(self):
        self.target_model.load_state_dict(self.online_model.state_dict())
        
    def sample(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()
        dones = torch.Tensor(dones).to(self.device).unsqueeze(1)
        rewards = torch.Tensor(rewards).to(self.device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        return states, actions, rewards, next_states, dones
    
    def calculate_targets(self, next_states, rewards, dones):
        arg_max_q_online = self.online_model(next_states).max(1)[1].unsqueeze(1).detach()
        target_q_values = self.target_model(next_states).gather(1, arg_max_q_online).detach()
        target_q_values = rewards + target_q_values * self.gamma * (1 - dones)
        return target_q_values
    
    def calculate_currents(self, states, actions):
        current_q_values = self.online_model(states).gather(1, actions)
        return current_q_values
    
    def optimize(self):
        if not self.can_train():
            return
        
        states, actions, rewards, next_states, dones = self.sample()
        target_q_values = self.calculate_targets(next_states, rewards, dones)
        current_q_values = self.calculate_currents(states, actions)
        
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, filename):
        self.online_model.save(filename)
    
    def load(self, filename):
        self.online_model.load(filename)
        self.update()