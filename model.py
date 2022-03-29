import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input, output, hiddens=(32, 32, 32)):
        super().__init__()
        self.inp = nn.Linear(input, hiddens[0])
        
        self.fcs = nn.ModuleList()
        for i in range(len(hiddens) - 1):
            fc = nn.Linear(hiddens[i], hiddens[i + 1])
            self.fcs.append(fc)
            
        self.out = nn.Linear(hiddens[-1], output)
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self._tensor(x)
        x = F.relu(self.inp(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = self.out(x)
        return x
    
    def _tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.Tensor(np.array(x)).to(self.device)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def save(self, filename: str):
        torch.save(self.state_dict(),  filename)
    
    def load(self, filename: str):
        try:
            self.load_state_dict(torch.load(filename))
        except:
            pass
        return self
    