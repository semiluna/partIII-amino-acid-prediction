import argparse
import math
import random 
# import wandb 
import os
from res_dataset import RESDataset

import torch 
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')
from torch_geometric.loader import DataLoader as geom_DataLoader

import gvp
# from atom3d.datasets import LMDBDataset
import torch_geometric
from functools import partial
import gvp.atom3d
import torch.nn as nn
import tqdm, torch, time, os
import numpy as np
from atom3d.util import metrics
import sklearn.metrics as sk_metrics
from collections import defaultdict
import scipy.stats as stats

DATASET_PATH = '/Users/antoniaboca/Downloads/split-by-cath-topology/data'

parser = argparse.ArgumentParser()
parser.add_argument('task', metavar='TASK', choices=[
        'PSR', 'RSR', 'PPI', 'RES', 'MSP', 'SMP', 'LBA', 'LEP'
    ], help="{PSR, RSR, PPI, RES, MSP, SMP, LBA, LEP}")
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                   help='number of threads for loading data, default=4')
parser.add_argument('--smp-idx', metavar='IDX', type=int, default=None,
                   choices=list(range(20)),
                   help='label index for SMP, in range 0-19')
parser.add_argument('--lba-split', metavar='SPLIT', type=int, choices=[30, 60],
                    help='identity cutoff for LBA, 30 (default) or 60', default=30)
parser.add_argument('--batch', metavar='SIZE', type=int, default=8,
                    help='batch size, default=8')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120,
                    help='maximum time between evaluations on valset, default=120 minutes')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20,
                    help='maximum time per evaluation on valset, default=20 minutes')
parser.add_argument('--epochs', metavar='N', type=int, default=50,
                    help='training epochs, default=50')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--load', metavar='PATH', default=None, 
                    help='initialize first 2 GNN layers with pretrained weights')
parser.add_argument('--load_checkpoint', type=str, default=None)
parser.add_argument('--wandb_id', type=str, default=None)
parser.add_argument('--max_len', type=int, default=None)
parser.add_argument('--sample_per_item', type=int, default=None)
parser.add_argument('--data_file', type=str, default=DATASET_PATH)
parser.add_argument('--wandb_group', type=str, default='gvp-pytorch-debug')
parser.add_argument('--id', type=str, default='debug')
os.makedirs('models', exist_ok=True)

args = parser.parse_args()
# if args.load_checkpoint or args.test:
#     run = wandb.init(config=args, project='part3-res-prediction-diss', id=args.wandb_id, resume='must')
# else:
#     run = wandb.init(config=args, project='part3-res-prediction-diss', group=args.wandb_group)

print = partial(print, flush=True)

models_dir = 'models'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())

def main():
    datasets = get_datasets(args.task, args.lba_split, args.data_file, args.max_len, args.sample_per_item)
    dataloader = partial(geom_DataLoader, 
                    num_workers=args.num_workers, batch_size=args.batch)
    if args.task not in ['PPI', 'RES']:
        dataloader = partial(dataloader, shuffle=True)
        
    trainset, valset, testset = map(dataloader, datasets)    
    model = get_model(args.task).to(device)
    
    if args.test:
        test(model, testset)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 0
        if args.load_checkpoint:
            start_epoch = load(model, args.load_checkpoint, optimizer)
        train(model, trainset, valset, optimizer, start_epoch)

def log(epoch, stats, step=0):
    for key, value in stats.items():
        with open(f'{args.id}_model_stats.csv', 'a+')  as handle:
            print(f'{epoch},{key},{value},{step}', file=handle)

def test(model, testset):
    model.load_state_dict(torch.load(args.test)['model_state_dict'])
    model.eval()
    t = tqdm.tqdm(testset)
    metrics = get_metrics(args.task)
    targets, predicts, ids = [], [], []
    with torch.no_grad():
        for batch in t:
            pred = forward(model, batch, device)
            label = get_label(batch, args.task, args.smp_idx)
            if args.task == 'RES':
                pred = pred.argmax(dim=-1)
            if args.task in ['PSR', 'RSR']:
                ids.extend(batch.id)
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))

    for name, func in metrics.items():
        if args.task in ['PSR', 'RSR']:
            func = partial(func, ids=ids)
        value = func(targets, predicts)
        print(f"{name}: {value}")
        log('end', {f'test_{name}': value})

def train(model, trainset, valset, optimizer, start_epoch = 0):
    best_path, best_val = None, np.inf
    for epoch in range(start_epoch, args.epochs):
        print(f' \n\n########## STARTING EPOCH {epoch} ###############')
        model.train()
        loss = loop(trainset, model, epoch, optimizer=optimizer, max_time=args.train_time)
        log(epoch, {'train_loss_epoch_end': loss})
        path = f"{models_dir}/{args.task}_{model_id}_{epoch}.pt"
        torch.save({
            'model_state_dict': model.state_dict(), 
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

        print(f'\nEPOCH {epoch} TRAIN loss: {loss:.8f}')
        model.eval()
        with torch.no_grad():
            loss = loop(valset, model, epoch, max_time=args.val_time, valid=True)
        log(epoch, {'val_loss': loss})
        print(f'\nEPOCH {epoch} VAL loss: {loss:.8f}')
        if loss < best_val:
            best_path, best_val = path, loss
        print(f'BEST {best_path} VAL loss: {best_val:.8f}')
    
def loop(dataset, model,epoch, optimizer=None, max_time=None, valid=False):
    start = time.time()
    loss_fn = get_loss(args.task)
    total_loss, total_count = 0, 0
    if valid:
        targets, predicts = [], []
    for idx, batch in enumerate(dataset):
        # if max_time and (time.time() - start) > 60*max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            out = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
        label = get_label(batch, args.task, args.smp_idx)
        loss_value = loss_fn(out, label)
        total_loss += float(loss_value)
        total_count += 1
        
        if valid:
            out = out.argmax(dim=-1)
            predicts.extend(list(out.cpu().numpy()))
            targets.extend(list(label.cpu().numpy()))

        if optimizer:
            try:
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
                
        if idx % 1000 == 0:
            if not valid:
                print(f'STEP {idx} | LOSS: {total_loss/total_count}')
                log(epoch, {'train_loss': total_loss/total_count}, step=idx)
            
    if valid:
        acc = metrics.accuracy(targets, predicts)
        print(f'EPOCH {epoch} | VALID ACCURACY: {acc}')
        log(epoch, {'valid_acc': acc}, step=idx)
    
    if not valid:
        print(f'STEP {idx} | LOSS: {total_loss/total_count}')
        log(epoch, {'train_loss': total_loss/total_count}, step=idx)

    return total_loss / total_count

def load(model, path, optimizer=None):
    params = torch.load(path)
    
    model.load_state_dict(params['model_state_dict'])
    optimizer.load_state_dict(params['optimizer_state_dict'])
    epoch = params['epoch']

    return epoch
        
#######################################################################

def get_label(batch, task, smp_idx=None):
    if type(batch) in [list, tuple]: batch = batch[0]
    if task == 'SMP':
        assert smp_idx is not None
        return batch.label[smp_idx::20]
    return batch.label

def get_metrics(task):
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])
        
    correlations = {
        'pearson': partial(_correlation, metrics.pearson),
        'kendall': partial(_correlation, metrics.kendall),
        'spearman': partial(_correlation, metrics.spearman)
    }
    mean_correlations = {f'mean {k}' : partial(v, glob=False) \
                            for k, v in correlations.items()}
    
    return {                       
        'RSR' : {**correlations, **mean_correlations},
        'PSR' : {**correlations, **mean_correlations},
        'PPI' : {'auroc': metrics.auroc},
        'RES' : {'accuracy': metrics.accuracy},
        'MSP' : {'auroc': metrics.auroc, 'auprc': metrics.auprc},
        'LEP' : {'auroc': metrics.auroc, 'auprc': metrics.auprc},
        'LBA' : {**correlations, 'rmse': partial(sk_metrics.mean_squared_error, squared=False)},
        'SMP' : {'mae': sk_metrics.mean_absolute_error}
    }[task]
            
def get_loss(task):
    if task in ['PSR', 'RSR', 'SMP', 'LBA']: return nn.MSELoss() # regression
    elif task in ['PPI', 'MSP', 'LEP']: return nn.BCELoss() # binary classification
    elif task in ['RES']: return nn.CrossEntropyLoss() # multiclass classification
    
def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = batch[0].to(device), batch[1].to(device)
    else:
        batch = batch.to(device)
    return model(batch)

def get_datasets(task, lba_split=30, data_file=None, max_len=None, sample_per_item=None):
    if task == 'RES':
        if data_file is None:
            data_file = '/Users/antoniaboca/Downloads/split-by-cath-topology/data'
        trainset = RESDataset(os.path.join(data_file, 'train'), max_len=max_len, sample_per_item=sample_per_item)
        valset = RESDataset(os.path.join(data_file, 'val'), max_len=max_len, sample_per_item=sample_per_item)
        testset = RESDataset(os.path.join(data_file, 'test'), max_len=max_len, sample_per_item=sample_per_item)


        # split_path = '/Users/antoniaboca/Downloads/split-by-cath-topology/indices/'
        # dataset = partial(gvp.atom3d.RESDataset, data_path)        
        # trainset = dataset(split_path=split_path+'train_indices.txt')
        # valset = dataset(split_path=split_path+'val_indices.txt')
        # testset = dataset(split_path=split_path+'test_indices.txt')
        
    return trainset, valset, testset

def get_model(task):
    return {
        'RES' : gvp.atom3d.RESModel,
        'PPI' : gvp.atom3d.PPIModel,
        'RSR' : gvp.atom3d.RSRModel,
        'PSR' : gvp.atom3d.PSRModel,
        'MSP' : gvp.atom3d.MSPModel,
        'LEP' : gvp.atom3d.LEPModel,
        'LBA' : gvp.atom3d.LBAModel,
        'SMP' : gvp.atom3d.SMPModel
    }[task]()

if __name__ == "__main__":
    main()
