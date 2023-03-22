import numpy as np
import torch

a = torch.tensor([[2,2,3],[4,5,6]])

a = a - a.max()
a = a.exp()
a = a.sum(1,keepdim=True)
print("---")