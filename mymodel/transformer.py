import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=0) #0.60
        self.fc = nn.Linear(cfg.emb_dim, 19)#0.701
        # self.lstm = nn.LSTM()
        self.fc1 = nn.Linear(cfg.emb_dim, 128)
        self.fc2 = nn.Linear(128, 19) #0.716/0.720
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.embed(x)#.half()
        x = x.sum(dim=1)
        # x = self.fc(x)
        x = self.fc1(x)
        # x = self.relu(self.dropout(x))
        x = self.fc2(x)
        # x = x.permute(0,2,1)
        # print(x.shape)
        # x = x.sum(dim=2)
        # print(x.mean().item(), x.std().item())
        return x



