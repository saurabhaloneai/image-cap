import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import spacy
import torchvision.transforms as transforms

spacy_eng = spacy.load("en_core_web_sm") # for tokenization 

class Vocabulary:
    
    def __init__(self,freq_threshold):
        
        self.itos = {0:"<PAD",1:"<SOS",2:"<EOS>",3:"UNK"} # int to sting 
        
        
        self.stoi = {v:k for k,v in self.itos.items()} # string to int tokens
        
        self.freq_threshold = freq_threshold
        
    def __len__(self): 
        return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        
        return [token.text.lower() for token in spacy_eng.tokenizer(text)] #tokenization with spacy
        
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        index = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                # append teh word if it freq is above threshold.
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1
                    
    
    def numericalize(self,text):
        tokenized_text = self.tokenize(text)
        
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    


class FlickrDataset(Dataset):
    
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        
        self.root_dir = root_dir
        self.captions_file = pd.read_csv(captions_file)
        self.transform = transform 
        
        # get the image img from the captions file(image,captions)
        
        self.img = self.df['image']
        self.cap = self.df['caption']
        
        # call the vocab funs 
        
        self.vocab = Vocabulary(freq_threshold) #create the vocab from captions 
        self.vocab.build_vocab(self.cap.tolist())
        
        
    def __init__(self):
        return len(self.df)
    
    
    def __getitem__(self, index) :
        
        caption = self.captions[index]
        img_name = self.imgs[index]
        img_location = os.path.jion(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        
        if self.transform is not None:
            img = self.transform(img)
        
        #text -> num
        
        captions_vec = []
        captions_vec += [self.vocab.stoi["<SOS>"]]
        captions_vec += self.vocab.numericalize(caption)
        captions_vec += [self.vocab.stoi["<EOS>"]]
        
        # to tensor 
        
        out = torch.tensor(captions_vec)
        
        return img, out
    

# adding PADDING to adjus the shapes

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    loader, dataset = get_loader(
        "flickr8k/images/", "flickr8k/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)