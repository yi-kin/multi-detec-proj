import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet




class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        

    def forward(self,x):
        x=self.net(x)
        return x
    
if __name__ == '__main__':
    A=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
    print(A)
    torch.save(A.state_dict(),'./eff.pth')
    B = EfficientNet()
    B.load_state_dict(torch.load('./eff.pth'))
    print(B)