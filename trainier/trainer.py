import torch
from torch import nn
from functools import partial
class Trainer:
    def __init__(self,
                model:nn.Module,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_fn=partial(torch.optim.SGD,lr=1e-3),
                device='cuda',
                batch_size=64):
        
        self.optimizer=optimizer_fn(model.parameters())
        self.losser=loss_fn
        self.batch_size=batch_size
        self.device=device
        self.model=model.to(device)

    def eval(self,stream):
        self.model.train()
        for x,y in stream.vanila_batch(self.batch_size):
            x,y=x.to(self.device),y.to(self.device)
            pred=self.model(x)
            loss=self.losser(pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            yield loss
    






