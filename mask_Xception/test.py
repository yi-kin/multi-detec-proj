import torch
import torchvision



# 定义一个二维 tensor
t = torch.tensor([[1, 2, 3], [11,12,13], [7, 8, 9]])

# # 找到每一行的最大值的下标
# _, indices = t.max(dim=1)
# t = [j[i] for i,j in zip(indices,t)]
a = [2,1]
t
print("t:",t[a])
# print(indices)  # 输出: tensor([2, 2, 2]