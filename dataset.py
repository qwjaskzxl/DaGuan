import os
import torch
import numpy as np
import jieba
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import random
from config import get_args
from time import time
from prefetch_generator import BackgroundGenerator


class PretrainDataset(Dataset):
    def __init__(self, cfg, root='data/'):
        self.max_seq_len = cfg.max_seq_len
        with open(root + 'corpus_word.txt', 'r') as f:
            self.item = f.readlines()
        # with open(root + 'vocab_freq.json', 'r') as f:
        #     self.vocab = json.load(f)

    def __len__(self):
        print(len(self.item))
        return len(self.item)

    def __getitem__(self, idx):
        return self.item[idx]


class TextDataset(Dataset):
    def __init__(self, cfg, mode, root='data/'):
        self.max_seq_len = cfg.max_seq_len
        self.mode = mode
        if mode == 'train':
            with open(root + 'train_word_min20.txt', 'r') as f1, open(root + 'label.txt', 'r') as f2:
                self.text = f1.readlines()[:90000]
                self.label = f2.readlines()[:90000]
        elif mode == 'val':
            with open(root + 'train_word_min20.txt', 'r') as f1, open(root + 'label.txt', 'r') as f2:
                self.text = f1.readlines()[90000:]
                self.label = f2.readlines()[90000:]
        elif mode == 'test':
            with open(root + 'test_word_min20.txt', 'r') as f:
                self.text = f.readlines()

        print(self.mode, 'size:', len(self.text))
        # with open(root + 'vocab_freq.json', 'r') as f:
        #     self.vocab = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.tokenize(self.text[idx])
        if len(text) > self.max_seq_len:
            half = self.max_seq_len // 2
            text = text[:half] + text[-half:]
        else:
            text.extend([0] * (self.max_seq_len - len(text)))
        text = torch.LongTensor(text)
        ret = {'text': text,}
        if self.mode == 'train' or self.mode == 'val':
            ret['label'] = int(self.label[idx].strip())-1
        return ret

    def tokenize(self, text):
        words = [int(w) for w in text.strip().split(' ')]
        return words

if __name__ == '__main__':
    cfg = get_args()
    print(cfg)
