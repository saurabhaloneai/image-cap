import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertTokenizer

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        # freeze resnet parameters to prevent updating during training
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)                                   
        # reshape features for attention
        features = features.permute(0, 2, 3, 1)                          
        features = features.view(features.size(0), -1, features.size(-1)) 
        return features

class Attention(nn.Module): # BahdanauAttention
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        # linear layers for attention
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)
        
    def forward(self, features, hidden_state):
        # calculate attention weights
        u_hs = self.U(features)     
        w_ah = self.W(hidden_state) 
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) 
        attention_scores = self.A(combined_states)        
        attention_scores = attention_scores.squeeze(2)    
        alpha = F.softmax(attention_scores, dim=1)    #attention_weight      
        # apply attention weights to features
        attention_weights = features * alpha.unsqueeze(2)  
        attention_weights = attention_weights.sum(dim=1)  
        return alpha, attention_weights

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
    
    def init_hidden_state(self, encoder_out):
        # initialize hidden states
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)
        return h, c

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            drop_prob=drop_prob
        )
        
    def forward(self, images, captions):
        # encode images and decode captions
        features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions)
        return outputs, alphas