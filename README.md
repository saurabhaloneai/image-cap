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

  - math eqn : 
    
    <p>$$u_{hs} = u \times features$$</p>

    - `u` is a weight matrix of shape `(encoder_dim, attention_dim)`.

- **self.w(hidden_state)**:

  - the decoder’s hidden state is passed through `self.w`, mapping it from the decoder dimension to the attention dimension.

  - this changes the shape of `hidden_state` to `(batch_size, attention_dim)`.

  - math eqn : 

    <p>$$w_{ah} = w \times hidden\_state $$</p>

    - `w` is a weight matrix of shape `(decoder_dim, attention_dim)`.

**Combining**

- **combining**: the mapped features and hidden state are combined by adding them together. 

- since `w_ah` has shape `(batch_size, attention_dim)`, we use `.unsqueeze(1)` to add an extra dimension so it can be added to `u_hs`.
- **activation**: a tanh activation is applied to the combined result. this function adds non-linearity, allowing the model to learn complex relationships.
- math eqn : 

  <p>$$combined\_states = tanh(u\_hs + w\_ah)$$</p>

  - shape of `combined_states`: `(batch_size, num_features, attention_dim)`.

**Computing Attn Scores:**

- **self.a(combined_states)**: the combined states are passed through `self.a`, which reduces the attention dimension to a single score for each feature.

- math eqn : 
  <p>$$attention\_scores = a \times combined\_states$$</p>

  - `a` is a weight matrix of shape `(attention_dim, 1)`.
  - shape of `attention_scores`: `(batch_size, num_features, 1)`.
  - the `squeeze(2)` operation removes the singleton dimension, giving `attention_scores` the shape `(batch_size, num_features)`.

**Softmax For Attn Weights:** 

- the attention scores are passed through a softmax function to normalize them into probabilities.

- math eqn : 

  <p>$$\alpha = softmax(attention\_scores)$$</p>

  - shape of `alpha`: `(batch_size, num_features)`. each value in `alpha` represents how much attention the model should give to each feature.

**Atten Weights:**

- **reshaping**: the attention weights `alpha` are reshaped with `.unsqueeze(2)` to match the shape of `features`.

- **weighted sum**: the features are multiplied by the attention weights and then summed over the spatial dimensions to produce the final attention-weighted context vector.
- mathe eqn : 

  <p>$$attention\_weights = \sum (\alpha \times features)$$</p>

  - shape of `attention_weights`: `(batch_size, encoder_dim)`. this vector is used by the decoder to generate the next word in the caption.

**Output**

- the `attention` class returns two things:

  1. **alpha**: the attention weights, which show where the model is focusing its attention.

  2. **attention_weights**: the context vector, which combines the encoder features according to the attention weights.


# DecoderRNN: 

### Overview

- the DecoderRNN gen captions for the given image features. 

- it takes the attention-weighted context vectors from the Attn class and the pre-gen word to produce the next word in the sequence. 
- it uses an LSTM (Long Short-Term Memory) to handle the text data(captions).

```python
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        # embedding layer using bert tokenizer vocabulary
        self.embedding = nn.Embedding(len(BertTokenizer.from_pretrained('bert-base-uncased')), embed_size)
        # initialize hidden and cell states
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        # lstm cell for sequence generation
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        # fc layer for output
        self.fcn = nn.Linear(decoder_dim, self.embedding.num_embeddings)
        self.drop = nn.Dropout(drop_prob)
```

- **Attention** : the attn is to focus on different parts of the image during caption generation.

- **Embedding Layer** : it transforms input words (in token form) into dense vectors of size embed_size. 
this layer uses the BERT tokenizer's vocabulary, which is loaded from the bert-base-uncased model.

-**Hidden and Cell State Init**:
  - self.init_h and self.init_c are linear layers that initialize the LSTM's hidden state (h) and cell state (c) based on the mean of the encoded image features.

- **LSTM Cell** : The core of the decoder is an LSTM cell(handle sequential data)
It takes both the embedded word vector and the context vector from the Attention as inputs.

- **Fully Connected Layer**: the last linear layer (self.fcn) maps the LSTM's output to the vocabulary size, generating scores for each word in the vocabulary.

- **Dropout**: it is applied to the LSTM output to prevent overfitting.(but still we overfit somehow-> need to use other tech also).


### Forward Pass

```python
def forward(self, features, captions):
    embeds = self.embedding(captions)
    h, c = self.init_hidden_state(features)
    seq_length = captions.size(1) - 1 
    batch_size = captions.size(0)
    num_features = features.size(1)
    
    # initialize tensors to store predictions and attention weights
    preds = torch.zeros(batch_size, seq_length, self.embedding.num_embeddings).to(features.device)
    alphas = torch.zeros(batch_size, seq_length, num_features).to(features.device)
            
    # generate sequence
    for s in range(seq_length):
        alpha, context = self.attention(features, h)
        lstm_input = torch.cat((embeds[:, s], context), dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))
        output = self.fcn(self.drop(h))
        preds[:, s] = output
        alphas[:, s] = alpha  
    
    return preds, alphas
```
> brekaing into pieaces 

**Input Shapes**:

- features: Shape (batch_size, num_features, encoder_dim). These are the encoded image features passed from the encoder.

- captions: Shape (batch_size, seq_length). These are the target captions, with each word represented as a token.

**Word Embeddings**:

- The input captions are passed through the embedding layer to convert each word token into a dense vector.

- embeds=Embedding(captions)

- shape of embeds: (batch_size, seq_length, embed_size).

**Init Hidden and Cell States**:

- Hidden State (h) and Cell State (c) are initialized using the mean of the image features.

- h=init_h(mean(features))

- shape of h and c: (batch_size, decoder_dim).

**Loop Through Sequence**:

- We iterate over each time step s in the sequence (except the last one, hence seq_length - 1).

**Attention**:

- At each time step, the attention mechanism computes the context vector based on the current hidden state and the image features.

- Context Vector (context): It is the weighted sum of the image features
- Attention Weights (alpha): They show the importance of each feature for the current word generation.
- alpha,context=Attention(features,h)
- Shape of context: (batch_size, encoder_dim).

**LSTM Cell Input**:

- The LSTM cell takes the concatenation of the current word's embedding (embeds[:, s]) and the context vector (context).

- Concatenation: This combines the current word information and the relevant image features.
- Shape of lstm_input: (batch_size, embed_size + encoder_dim).

**LSTM Cell Update**:

- The LSTM cell updates the hidden and cell states based on its input.

- h,c=LSTMCell(lstm_input,(h,c))

- Shape of h and c: (batch_size, decoder_dim).

**Output Generation**:

- The hidden state h is passed through a dropout layer and then through the fully connected layer self.fcn to generate the output scores for the next word.
- Output (output): It contains the logits for each word in the vocabulary.

- output=fcn(drop(h))
- Shape of output: (batch_size, vocab_size).

**Storing Predictions and Attention Weights**:

- The output logits are stored in preds at the current time step s.
- The attention weights alpha are stored in alphas for each time step.

**Return**:

- The model returns the predictions (preds) and the attention weights (alphas).