from config import get_args
from dataset import TextDataset
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from mymodel.transformer import TransformerModel
from tqdm import tqdm

def eval(model,valset,cfg):

    val_params = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        'pin_memory': True,
        "num_workers": cfg.workers
    }
    val_loader = DataLoader(valset, **val_params)
    model = model.eval().cuda()
    # model.load_state_dict(torch.load('trained_models/emb0.811.pth'))

    total,N_T = 0,0

    for i, data in enumerate(val_loader):
        text = data['text'].cuda()
        label = data['label'].cuda()
        y = model(text)
        y_pred = torch.argmax(y, dim=1)
        total += label.size(0)
        N_T += (y_pred == label).sum().float()

    return N_T/total

def train(cfg):
    cudnn.benchmark = True
    trainset = TextDataset(cfg, 'train')
    valset = TextDataset(cfg, 'val')
    train_params = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        'pin_memory': True,
        "num_workers": cfg.workers
    }
    train_loader = DataLoader(trainset,**train_params)
    model = TransformerModel(cfg).train().cuda().half()
    # model = nn.DataParallel(model, device_ids=[0])

    optimizer = torch.optim.AdamW(model.parameters(), cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    epoch_loss = []
    best_acc = 0.8
    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(train_loader)
        total,N_T = 0,0

        for i, data in enumerate(progress_bar):
            optimizer.zero_grad()
            text = data['text'].cuda()
            label = data['label'].cuda()
            y = model(text)
            loss = loss_fn(y, label)

            total += label.size(0)
            y_pred = torch.argmax(y, dim=1)
            N_T += (y_pred == label).sum().float()

            progress_bar.set_description('Epoch: %d/%d'%(epoch + 1, cfg.num_epochs))
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)
            acc_tra = N_T / total

            # if (i+1) % 1000 == 0:
            #     acc_val = eval(model, cfg)
            #     progress_bar.write('Batch loss: %.3f\tTotal loss: %.3f\tAcc_tra: %.3f\tAcc_val: %.3f'
            #                            %(loss, total_loss, acc_tra, acc_val))
            loss.backward()
            optimizer.step()

        # acc_val = eval(model, valset, cfg)
        # progress_bar.write('Batch loss: %.3f\tTotal loss: %.3f\tAcc_tra: %.3f\tAcc_val: %.3f'
        #                    % (loss, total_loss, acc_tra, acc_val))
        # if acc_val > best_acc:
        #     best_acc = acc_val
        #     torch.save(model.state_dict(), 'trained_models/emb%.3f.pth'%(acc_val))

if __name__ == '__main__':
    cfg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % cfg.gpu_id
    train(cfg)
