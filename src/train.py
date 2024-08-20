#import 
from src.model import EncoderDecoder
from src.data_loader import train_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import os
from transformers import BertTokenizer

# using BERT tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Hyperparams
embed_size=300
attention_dim=256
encoder_dim=2048
decoder_dim=512
learning_rate = 3e-4

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# Initialize the model
model = EncoderDecoder(
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

# Initialize criterion and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training loop 

def train(model, data_loader, criterion, optimizer, num_epochs, tokenizer):
    # multiple GPUs support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # mixed precision training setup
    scaler = GradScaler()
    print_every = 150
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for idx, (image, captions) in enumerate(data_loader):
            image, captions = image.to(device), captions.to(device)
            
            optimizer.zero_grad()
            
            # autocast for mixed precision
            with autocast():
                outputs, _ = model(image, captions)
                targets = captions[:, 1:]  # shifted target for teacher forcing
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.reshape(-1))
            # backpropagation with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            # calculating accuracy, ignoring padding index
            _, predicted = outputs.max(2)
            mask = targets != tokenizer.pad_token_id
            correct_predictions += (predicted == targets).masked_select(mask).sum().item()
            total_predictions += mask.sum().item()
            
            if (idx + 1) % print_every == 0:
                avg_loss = running_loss / print_every
                accuracy = correct_predictions / total_predictions
                print(f"Epoch: {epoch}/{num_epochs}, Batch: {idx+1}/{len(data_loader)}, Loss: {avg_loss:.5f}, Accuracy: {accuracy:.5f}")
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # evaluate the model on the entire data loader
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        
        with torch.inference_mode():
            for image, captions in data_loader:
                image, captions = image.to(device), captions.to(device)
                with autocast():
                    outputs, _ = model(image, captions)
                    targets = captions[:, 1:]
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.reshape(-1))
                eval_loss += loss.item()
                
                _, predicted = outputs.max(2)
                mask = targets != tokenizer.pad_token_id
                eval_correct += (predicted == targets).masked_select(mask).sum().item()
                eval_total += mask.sum().item()
        
        avg_eval_loss = eval_loss / len(data_loader)
        eval_accuracy = eval_correct / eval_total
        print(f"Epoch: {epoch}/{num_epochs}, Validation Loss: {avg_eval_loss:.5f}, Validation Accuracy: {eval_accuracy:.5f}")
        
        # save the model 
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), "model_weights.pth")
            else:
                torch.save(model.state_dict(), "model_weights.pth")
            print(f"model saved at epoch {epoch}")

# Train the model

num_epochs = 100

train(model, train_loader, criterion, optimizer, num_epochs, tokenizer)