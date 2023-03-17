import torch

a = torch.tensor([[2,3,4],[1,2,3]])

b = torch.tensor([[5,6,9],[7,8,9]])

c = torch.cat((a,b),dim=1)


print(c)