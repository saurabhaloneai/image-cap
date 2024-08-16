import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN,self).__init__()
        resnet=models.resnet50(pretrained=True)
        for p in resnet.parameters():
            p.requires_grad_(False)
        modules=list(resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
    def forward(self,images):
        features=self.resnet(images)
        features=features.permute(0,2,3,1)
        features=features.view(features.size(0),-1,features.size(-1))
        return features #(batch_size,49,2048)
        