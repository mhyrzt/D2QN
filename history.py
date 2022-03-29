from matplotlib import pyplot as plt

class History:
    def __init__(self):
        self.rewards = []
        self.epsilons = []
    
    def add(self, reward, epsilon):
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
    
    def plot(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title("Rewards")
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.epsilons)
        plt.title("Epsilons")
        plt.grid()
        
        plt.show()
    
    def log(self, episode, SEP="\t\t"):
        r = self.rewards[-1]
        e = self.epsilons[-1]
        print(f"#{episode}{SEP}reward={r}{SEP}eps={e}")