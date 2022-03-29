import gym
from agent import Agent
from time import sleep
from trainer import ENV_NAME

FPS = 1.0 / 30.0

env = gym.make(ENV_NAME)
agent = Agent(env)
agent.load(f"{ENV_NAME}-agent.pt")

done = False
state = env.reset()
total = 0
while not done:
    env.render()
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    total += reward
    sleep(FPS)
env.close()
print(f"TOTAL REWARDS = {total}")