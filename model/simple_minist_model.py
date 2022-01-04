import torch
from torch import nn


class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.flattern=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self,x):
        x=x.type(torch.float)
        x=self.flattern(x)
        logits=self.linear_relu_stack(x)
        return logits

