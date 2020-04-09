import os
from dataset import TextDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from mymodel.transformer import TransformerModel
from tqdm import tqdm
from config import get_args


def test(cfg):
    cudnn.benchmark = True
    trainset = TextDataset(cfg, 'test')
    test_params = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        'pin_memory': True,
        'drop_last' : False,
        "num_workers": cfg.workers
    }
    data_loader = DataLoader(trainset, **test_params)

    model = TransformerModel(cfg).eval().cuda()
    model.load_state_dict(torch.load('trained_models/emb0.811.pth'))

    progress_bar = tqdm(data_loader)
    result = []
    for i, data in enumerate(progress_bar):
        text = data['text'].cuda()
        y = model(text)
        y_pred = torch.argmax(y, dim=1)
        for pred in y_pred:
            result.append(pred.item())

    with open('results/ans.csv','w') as f:
        f.write("id,class"+"\n")
        for i in range(len(result)): #是列数，即句子数
            f.write(str(i)+','+str(result[i]+1)+'\n')

if __name__ == '__main__':
    cfg = get_args()
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '%d' % cfg.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % cfg.gpu_id
    test(cfg)
