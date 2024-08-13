from PIL import Image
from io import BytesIO
import requests
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
       
        return text.lower().split()  
    def build_vocabulary(self, sentence_list):  
        frequencies = {}
        idx = 4  

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

class PokemonDataset(Dataset):
    def __init__(self, root_dir, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform

        # Load dataset
        ds = load_dataset("reach-vb/pokemon-blip-captions", split="train[:500]")
        self.imgs = ds['image']
        self.captions = ds['text']

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]  # Assuming 'image' field contains image file paths
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class PokemonDataset(Dataset):
    def __init__(self, dataset, transform=None, freq_threshold=5):
        self.dataset = dataset
        self.transform = transform

       
        self.imgs = [item['image'] for item in dataset] 
        self.captions = [item['text'] for item in dataset]

       
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_data = self.imgs[index]

        
        if isinstance(img_data, str):
            if img_data.startswith("http"):
                
                response = requests.get(img_data)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
               
                img = Image.open(img_data).convert("RGB")
        elif isinstance(img_data, Image.Image):
            
            img = img_data.convert("RGB")
        else:
            raise TypeError("Unexpected type for img_data")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
from torch.utils.data import DataLoader
from datasets import load_dataset

def get_loader(
    split='train[:500]',
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    freq_threshold=5,
    transform=None
):
    # Load dataset
    dataset = load_dataset("reach-vb/pokemon-blip-captions", split=split)
    
    # Initialize the custom dataset
    pokemon_dataset = PokemonDataset(dataset, transform=transform, freq_threshold=freq_threshold)

    # Define padding index
    pad_idx = pokemon_dataset.vocab.stoi["<PAD>"]

    # Initialize the collate function
    collate_fn = MyCollate(pad_idx=pad_idx)

    # Create DataLoader
    loader = DataLoader(
        dataset=pokemon_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    return loader, pokemon_dataset

# Example usage
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

data_loader, dataset = get_loader(
    split='train[:500]',
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    freq_threshold=5,
    transform=transform
)

# Iterate through the data loader
for imgs, captions in data_loader:
    print(imgs.shape)  # Should be (batch_size, channels, height, width)
    print(captions.shape)  # Should be (batch_size, max_caption_length)
    break  # Just for demonstration; remove in actual code
