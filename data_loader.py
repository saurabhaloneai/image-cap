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

spacy_eng = spacy.load("en_core_web_sm") # for token tokenization 