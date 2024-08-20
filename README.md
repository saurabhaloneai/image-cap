# image captioningggg 🐳

## Dataset 

## Architecture

### ARC : 

#### - Econder (resnet50)

#### - Attention(block)

#### - Deonder (lstms)

# EncoderCNN: 

### Overview

It uses a pre-trained ResNet-50 network to extract features from images. these features are then passed to the decoder, which generates captions.

### initialization

```python
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # freeze resnet parameters to prevent updating during training
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)


- the EncoderCNN uses a ResNet-50 model that has been pre-trained on the ImageNet dataset.

- we freeze its parameters so they don't change during training. 

- this way, ResNet-50 can focus on extracting useful features from the input images.(and also we don't need to trian it from scrath)

- after that we remove the last layers of ResNet-50, which are designed for classification, because we only need the feature maps, not the classification output.

### forward pass 

```python 
def forward(self, images):
    features = self.resnet(images)                                   
    # reshape features for attention
    features = features.permute(0, 2, 3, 1)                          
    features = features.view(features.size(0), -1, features.size(-1)) 
    return features
```

**Input**

- the input(images) with shape (batch_size, 3, 224, 224). 

- 224x224 is the image size, and 3 is for the RGB color channels.

**Passing Through ResNet-50**

- After passing, the feature map shape is (batch_size, 2048, 7, 7). 

- 2048 is the number of features, and 7x7 is the reduced spatial size of the image.

**Reshaping for Attention** 

- we then rearrange the dimensions so the spatial information comes first. 

- after permuting, the shape is (batch_size, 7, 7, 2048).

- then we flatten the spatial dimensions ton so we can fed to attn.

- the output shape becomes (batch_size, 49, 2048). 49 is 7x7, and 2048 is the number of features.

**Output**

- the final output is a feature tensor with shape (batch_size, num_features, encoder_dim), which is passed to the decoder.


## Training 

## TODO 