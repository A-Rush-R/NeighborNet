import os
import sys
import pickle as pkl
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt

from dataloader.supervise_movienet import load_data, load_transfer, load_data_abl
from model.NeighborNet import SLNet
from loss import bce
from warm_up import warmup_decay_cosine


from metric import metric

# ==========================
# 📌 Utility Functions
# ==========================
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves model state to a file."""
    torch.save(state, filename)

def write_pkl(path, data):
    """Writes a pickle file."""
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1

def trans_graph(graph, gpu):
    """Converts graph data to a CUDA-compatible format."""
    graph = graph.to(torch.float).cuda(gpu)
    return Variable(graph, requires_grad=False)

def adjust_lr(optimizer):
    """Adjusts learning rate by reducing it by 10x."""
    for param in optimizer.param_groups:
        param['lr'] *= 0.1
    return 1

# ==========================
# 📌 Training Function
# ==========================
def train_epoch(train_loader, model, optimizer, lr_scheduler, gpu=0):
    """Runs a single training epoch."""
    model.train()
    progress = tqdm(train_loader)
    
    for sample in progress:
        data, graphs, inxs, label = [s.cuda(gpu) for s in sample[:4]]
        hop_gh = trans_graph(graphs, gpu)
        
        pred = model(data, hop_gh, inxs)
        loss = bce(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        progress.set_postfix(loss=f'{loss.item():.8f}')

# ==========================
# 📌 Testing Function
# ==========================
def test_epoch(test_loader, model, metrics=['map', 'miou', 'f1'], gpu=0):
    """Evaluates the model on a test dataset."""
    model.eval()
    pred_list, label_list, path_list = [], [], []
    
    with torch.no_grad():
        for sample in tqdm(test_loader):
            paths, data, graphs, inxs, label = sample
            data, inxs = data.cuda(gpu), inxs.cuda(gpu)
            hop_gh = trans_graph(graphs, gpu)
            pred = model(data, hop_gh, inxs)
            
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            path_list.append(paths)
    
    return metric(path_list, pred_list, label_list, needs=metrics)

# ==========================
# 📌 Main Training Pipeline
# ==========================
def main(sample_path, split_path, batch=64, epochs=10, gpu=0, model_path=None, save_path=None):
    """Trains NeighborNet on MovieNet dataset."""
    train_loader = load_data(sample_path, split_path, batch, topk=5)
    test_loader = load_data(sample_path, split_path, 512, mode='test', topk=5)
    
    model = SLNet(2048, embed_dim=1024, att_drop=0.1, topk=5, seg_sz=20, tnei=2, mode='fine').cuda(gpu)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_decay_cosine(len(train_loader), len(train_loader) * (epochs - 1)))
    
    max_miou, max_map = 0, 0
    for epoch in range(epochs):
        train_epoch(train_loader, model, optimizer, lr_scheduler, gpu)
        met, _ = test_epoch(test_loader, model, ['map', 'miou', 'f1'], gpu)
        
        if save_path and (max_miou < met['mIoU'] or max_map < met['mAP']):
            max_miou, max_map = met['mIoU'], met['mAP']
            save_checkpoint({'state_dict': model.state_dict(), 'mIoU': max_miou, 'mAP': max_map}, f'{save_path}/epoch_{epoch+1}.pth.tar')
        
        print(f"Epoch {epoch+1}: mAP={met['mAP']:.3f}, mIoU={met['mIoU']:.3f}, F1={met['F1']:.3f}")

# ==========================
# 📌 Example Usage
# ==========================
if __name__ == "__main__":
    sample_path = r'/data/OpenDataLab___MovieNet/raw/gendatasetk2'
    split_path = r'/data/OpenDataLab___MovieNet/raw/movie1K.split.v1.json'
    save_path = r'/data/manan/ouput_k_2'
    model_path = None  # Set pre-trained model path if available
    
    main(sample_path, split_path, batch=512, epochs=10, save_path=save_path, model_path=model_path)
