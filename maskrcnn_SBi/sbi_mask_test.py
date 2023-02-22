import torch
from PIL import Image

# a = torch.Tensor([3,2,1])
# b = a.numpy()
# print(b)
# A = [*map(lambda x: x - 1, b)]
# print(A)


def del_tensor_ele(arr,index):
    arr1 = arr[0:index]

    return arr1
tensor1 = torch.Tensor([1,2,3,4,5,6,7])
for i in range(2):
    tensor1 = del_tensor_ele(tensor1,-1) # 删除tensor1中索引为1的元素
print(tensor1)
