import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.supervise_movienet import load_data, load_transfer, load_data_abl
from model.NeighborNet import SLNet
from loss import bce, sigmoid_focal
from warm_up import warmup_decay_cosine
from metric import metric
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl

torch.cuda.set_device(0)

import logging
import sys

import os
import logging
import matplotlib.pyplot as plt

# Initialize logging
logging.basicConfig(
    filename='training_logs.log',  # Log file
    level=logging.INFO,           # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
)

def initialize_visualization():
    """Prepare placeholders for visualization."""
    global epoch_losses, epoch_maps, epoch_mious
    epoch_losses = []
    epoch_maps = []
    epoch_mious = []

def update_visualization(loss, mAP, mIoU):
    """Update metrics for visualization."""
    epoch_losses.append(loss)
    epoch_maps.append(mAP)
    epoch_mious.append(mIoU)

def plot_visualizations(save_path=None):
    """Plot and save training metrics visualizations."""
    epochs = range(1, len(epoch_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot mAP and mIoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_maps, label='mAP')
    plt.plot(epochs, epoch_mious, label='mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Performance Metrics')
    plt.legend()

    if save_path:
        plt.savefig(f"{save_path}/training_metrics.png")
    plt.show()

# Modify train_epoch to return the average loss
def train_epoch(
        trainload,
        model,
        opti,
        lr_sh,
        gpu=0
):
    model.train()
    progress = tqdm(trainload)
    epoch_loss = 0
    for i, sample in enumerate(progress):
        data, graphs, inxs, label = sample[0], sample[1], sample[2], sample[3]

        data = data.cuda(gpu)
        hop_gh = trans_graph(graphs, gpu)
        inxs = inxs.cuda(gpu)
        label = label.cuda(gpu)

        pred = model(data, hop_gh, inxs)
        loss = bce(pred, label)

        opti.zero_grad()
        loss.backward()
        opti.step()

        lr_sh.step()
        progress.set_postfix(loss=f'{loss.item():.8f}')
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(trainload)
    return average_loss

def test_epoch(
        testload,
        model,
        need,
        gpu=0,
):
    predlist = []
    labelist = []
    pathlist = []
    scos = 0
    dcos = 0
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate((tqdm(testload))):
           paths, data, graphs, inxs, label = sample[0], sample[1], sample[2], sample[3], sample[4]

           data = data.cuda(gpu)
           hop_gh = trans_graph(graphs, gpu)
           inxs = inxs.cuda(gpu)

           pred = model(data, hop_gh, inxs)



           predlist.append(pred.data.cpu().numpy())
           labelist.append(label.data.cpu().numpy())
           pathlist.append(paths)
    met, moviePL = metric(pathlist, predlist, labelist, needs=need)
    return met, moviePL

# Update the main function
def main(
        sample_path,
        split_path=None,
        batch=64,
        epoch=2,
        gpu=0,
        model_path=None,
        save_path=None,
):
    # Initialize visualization data
    initialize_visualization()

    trainload = load_data(sample_path, split_path, batch, topk=5)
    testload = load_data(sample_path, split_path, 512, mode='test', topk=5)

    model = SLNet(2048, embed_dim=1024, att_drop=0.1, topk=5, seg_sz=20, tnei=2, mode='fine')
    if model_path is not None:
        pretrain = torch.load(model_path, map_location='cpu')['state_dict']
        model.load_state_dict(pretrain)
    model.cuda(gpu)

    max_miou = 0
    max_map = 0
    opti = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    iter_num = len(trainload)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opti,
        warmup_decay_cosine(iter_num, iter_num * (epoch - 1))
    )

    for i in range(epoch):
        # Training
        avg_loss = train_epoch(trainload, model, opti, lr_scheduler, gpu)

        # Testing
        met, _ = test_epoch(testload, model, ['map', 'miou', 'f1'], gpu=gpu)

        # Update visualizations
        update_visualization(avg_loss, met['mAP'], met['mIoU'])

        # Log metrics
        logging.info(
            f"Epoch {i+1}/{epoch}: Loss={avg_loss:.6f}, mAP={met['mAP']:.3f}, mIoU={met['mIoU']:.3f}, F1={met['F1']:.3f}"
        )
        print(
            f"Epoch {i+1}/{epoch}: Loss={avg_loss:.6f}, mAP={met['mAP']:.3f}, mIoU={met['mIoU']:.3f}, F1={met['F1']:.3f}"
        )

        # Save model
        if save_path is not None and (max_miou < met['mIoU'] or max_map < met['mAP']):
            max_miou = met['mIoU']
            max_map = met['mAP']
            save_checkpoint(
                {
                    'state_dict': model.state_dict(),
                    'miou': max_miou,
                    'map': max_map,
                    'f1': met['F1'],
                    'optim': opti.state_dict()
                },
                f"{save_path}/epoch_{i+1}.pth.tar"
            )

    # Plot and save visualizations
    if save_path:
        plot_visualizations(save_path)
    else:
        plot_visualizations()

    return 1
def adjust_lr(optimizer):
    for param in optimizer.param_groups:
        param['lr'] *= 0.1
    return 1


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1
# ------------- Utilize -------------  #
def trans_graph(graph, gpu):
    graph = graph.to(torch.float)
    graph = graph.cuda(gpu)
    graph = Variable(graph, requires_grad=False)
    return graph


if __name__=='__main__':
    
   sample_path = r'/data/OpenDataLab___MovieNet/raw/dataset2'
   split_path = r'/data/OpenDataLab___MovieNet/raw/movie1K.split.v1.json'
   save_path = r'/data/OpenDataLab___MovieNet/raw/modelsavepath'

    # ablation studies

_ = main(sample_path, split_path, batch=512, epoch=2, save_path=save_path, model_path=None)

