# image captioningggg 🐳

## Dataset 

## Architecture

> [!IMPORTANT]
>
> Econder(resnet50)
>
> Attention(block)
>
> Deonder(lstms)

# EncoderCNN: 

### Overview

It uses a pre-trained (ResNet-50)[https://github.com/saurabhaloneai/History-of-Deep-Learning/blob/main/02-optimization-and-regularization/03-residuals/resnet.ipynb] network to extract features from images. these features are then passed to the decoder, which generates captions.


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
```

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

- the input(images) with shape `(batch_size, 3, 224, 224)`. 

- 224x224 is the image size, and 3 is for the RGB color channels.

**Passing Through ResNet-50**

- After passing, the feature map shape is `(batch_size, 2048, 7, 7)`. 

- 2048 is the number of features, and 7x7 is the reduced spatial size of the image.

**Reshaping for Attention** 

- we then rearrange the dimensions so the spatial information comes first. 

- after permuting, the shape is `(batch_size, 7, 7, 2048)`.

- then we flatten the spatial dimensions ton so we can fed to attn.

- the output shape becomes `(batch_size, 49, 2048)`. `49` is `7x7`, and `2048` is the number of features.

**Output**

- the final output is a feature tensor with shape `(batch_size, num_features, encoder_dim)`, which is passed to the decoder.

# Attention: 

### Overview

- the Attn is used in decoder and it focus on different parts of the image while generating captions. 

- it calculates attention scores that tell the model which parts of the image to pay more attention to at each step in the caption gen process.

```python
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        # linear layers for attention
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)
```
- the Atten uses three linear layers.(it uses this to cal attention scores.) 

- **self.W**: takes the hidden state of the decoder `(decoder_dim)` and maps it to the attention space `(attention_dim)`.

- **self.U**: takes the features from the encoder `(encoder_dim)` and also maps them to the attention space `(attention_dim)`.

- **self.A**: reduces the combined attention space to a single value, which will represent the attention score.


### Forward Pass

```python
def forward(self, features, hidden_state):
    u_hs = self.U(features)     
    w_ah = self.W(hidden_state) 
    combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) 
    attention_scores = self.A(combined_states)        
    attention_scores = attention_scores.squeeze(2)    
    alpha = F.softmax(attention_scores, dim=1)    # attention_weight      
    # apply attention weights to features
    attention_weights = features * alpha.unsqueeze(2)  
    attention_weights = attention_weights.sum(dim=1)  
    return alpha, attention_weights
```
> breakign down the whole code of attn

**Input Shapes:**

- **features**: shape `(batch_size, num_features, encoder_dim)`. 

- these are the features extracted by the encoder, where `num_features` is the flattened spatial dimension (49 if the feature map is `7x7`), and `encoder_dim` is typically `2048`.
- **hidden_state**: shape `(batch_size, decoder_dim)`. this is the current hidden state of the lstm decoder.

**Mapping to attention space:**

- **self.u(features)**:

  - each feature from the encoder is passed through `self.u`, which maps it from the encoder dimension to the attention dimension.

  - if `attention_dim = 512`, this operation changes the shape of `features` to `(batch_size, num_features, attention_dim)`.
  - math-f:
    $$u_{hs} = u \times features$$
    where `u` is a weight matrix of shape `(encoder_dim, attention_dim)`.

- **self.w(hidden_state)**:

  - the decoder’s hidden state is passed through `self.w`, mapping it from the decoder dimension to the attention dimension.

  - this changes the shape of `hidden_state` to `(batch_size, attention_dim)`.
  - mathematically: 
    $$
    w_{ah} = w \times hidden\_state
    $$
    where `w` is a weight matrix of shape `(decoder_dim, attention_dim)`.

**Combining**

- **combining**: the mapped features and hidden state are combined by adding them together. 

- since `w_ah` has shape `(batch_size, attention_dim)`, we use `.unsqueeze(1)` to add an extra dimension so it can be added to `u_hs`.
- **activation**: a tanh activation is applied to the combined result. this function adds non-linearity, allowing the model to learn complex relationships.
- mathematically: 
  $$
  combined\_states = tanh(u\_hs + w\_ah)
  $$
  shape of `combined_states`: `(batch_size, num_features, attention_dim)`.

**Computing Attn Scores:**

- **self.a(combined_states)**: the combined states are passed through `self.a`, which reduces the attention dimension to a single score for each feature.

- mathematically: 
  $$
  attention\_scores = a \times combined\_states
  $$
  where `a` is a weight matrix of shape `(attention_dim, 1)`.
  - shape of `attention_scores`: `(batch_size, num_features, 1)`.
  - the `squeeze(2)` operation removes the singleton dimension, giving `attention_scores` the shape `(batch_size, num_features)`.

**Softmax For Attn Weights:** 

- the attention scores are passed through a softmax function to normalize them into probabilities.

- mathematically: 
  $$
  \alpha = softmax(attention\_scores)
  $$
  shape of `alpha`: `(batch_size, num_features)`. each value in `alpha` represents how much attention the model should give to each feature.

**Atten Weights:**

- **reshaping**: the attention weights `alpha` are reshaped with `.unsqueeze(2)` to match the shape of `features`.

- **weighted sum**: the features are multiplied by the attention weights and then summed over the spatial dimensions to produce the final attention-weighted context vector.
- mathematically: 
  $$
  attention\_weights = \sum (\alpha \times features)
  $$
  shape of `attention_weights`: `(batch_size, encoder_dim)`. this vector is used by the decoder to generate the next word in the caption.

**Output**

- the `attention` class returns two things:

  1. **alpha**: the attention weights, which show where the model is focusing its attention.

  2. **attention_weights**: the context vector, which combines the encoder features according to the attention weights.

