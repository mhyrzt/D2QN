import gym
import signal
from agent import Agent
from history import History

BREAK = False
EPISODES = 50
LOG_EACH = 10
RENDER_EACH = 100
UPDATE_EVERY = 25

def handler(signum, frame):
    global BREAK
    BREAK = True
signal.signal(signal.SIGINT, handler)

env = gym.make("CartPole-v1")
hist = History()
agent = Agent(env)

for episode in range(EPISODES):
    done = False
    state = env.reset()
    episode_reward = 0
    
    while not done:
        if episode % RENDER_EACH == 0:
            env.render()
        
        state, done, reward = agent.explore(state)
        episode_reward += reward
        
        agent.optimize()
    
    hist.add(episode_reward, agent.epsilon.epsilon)
    if episode % UPDATE_EVERY == 0:
        agent.update()
    if episode % LOG_EACH == 0:
        hist.log(episode)
    
    agent.epsilon.decrease()
    
    if BREAK:
        break

env.close()
hist.plot()