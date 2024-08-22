#import 
from src.model import EncoderDecoder
from src.data_loader import train_loader, test_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import os
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim

# using BERT tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # multiple GPUs support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    # mixed precision training setup
    scaler = GradScaler()
    print_every = 150
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for idx, (image, captions) in enumerate(train_loader):
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
                print(f"Epoch: {epoch}/{num_epochs}, Batch: {idx+1}/{len(train_loader)}, Loss: {avg_loss:.5f}, Accuracy: {accuracy:.5f}")
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.inference_mode():
            for image, captions in test_loader:
                image, captions = image.to(device), captions.to(device)
                with autocast():
                    outputs, _ = model(image, captions)
                    targets = captions[:, 1:]
                    loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.reshape(-1))
                test_loss += loss.item()
                
                _, predicted = outputs.max(2)
                mask = targets != tokenizer.pad_token_id
                test_correct += (predicted == targets).masked_select(mask).sum().item()
                test_total += mask.sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        print(f"Epoch: {epoch}/{num_epochs}, Test Loss: {avg_test_loss:.5f}, Test Accuracy: {test_accuracy:.5f}")
        
        # save the model 
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), "model_weights.pth")
            else:
                torch.save(model.state_dict(), "model_weights.pth")
            print(f"Model saved at epoch {epoch}")
# Train the model
num_epochs = 100
train(model, train_loader, test_loader, criterion, optimizer, num_epochs, tokenizer)