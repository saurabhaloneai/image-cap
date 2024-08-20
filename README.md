# image captioningggg 🐳

## 1. Dataset 

## 2. Model 

### ARC : 

#### - Econder (resnet50)

#### - Attention(block)

#### - Deonder (lstms)

# EncoderCNN: ResNet-based Image Feature Extractor

## Overview

The `EncoderCNN` class implements a convolutional neural network (CNN) encoder based on a pre-trained ResNet-50 architecture. It's designed to extract spatial features from input images for subsequent use in attention-based decoding tasks.

## Architecture Details

### Initialization

1. Loads a pre-trained ResNet-50 model:

   ```resnet = models.resnet50(pretrained=True)

Freezes all ResNet parameters:

for param in resnet.parameters():
    param.requires_grad_(False)

Removes the final two layers (adaptive average pooling and fully connected):
pythonCopymodules = list(resnet.children())[:-2]

Wraps remaining layers in a sequential module:
pythonCopyself.resnet = nn.Sequential(*modules)


Forward Pass
The forward method processes input images through the modified ResNet:

Feature extraction:
pythonCopyfeatures = self.resnet(images)

Dimension reordering:
pythonCopyfeatures = features.permute(0, 2, 3, 1)

Reshaping:
pythonCopyfeatures = features.view(features.size(0), -1, features.size(-1))


Mathematical Formulation
Let $X \in \mathbb{R}^{B \times C \times H \times W}$ be the input tensor, where:

$B$ is the batch size
$C$ is the number of input channels (typically 3 for RGB images)
$H$ is the height of the input image
$W$ is the width of the input image

The ResNet-50 transformation can be represented as:
$F = f_{ResNet}(X)$
Where $F \in \mathbb{R}^{B \times 2048 \times H' \times W'}$, with $H'$ and $W'$ being the reduced spatial dimensions.
The permute operation changes the order of dimensions:
$F_{permuted} = F_{B \times W' \times H' \times 2048}$
The final reshaping operation produces:
$F_{reshaped} = F_{B \times (H' \cdot W') \times 2048}$
Output Dimensions
For a standard input size of 224x224 pixels:

Input: $(B, 3, 224, 224)$
After ResNet: $(B, 2048, 7, 7)$
After permute: $(B, 7, 7, 2048)$
Final output: $(B, 49, 2048)$

where $B$ is the batch size.


## 3. Training 

## 4. Inference 

