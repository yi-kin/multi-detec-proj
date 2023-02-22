import torch

a = torch.tensor([[2,3,4],[1,2,3]])
print(a.size())
b = torch.tensor([2,3])
print(b.size())
c = torch.matmul(a,b)

c = c.unsqueeze(dim=0)
print(c)