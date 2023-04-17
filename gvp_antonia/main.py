import os
import math
import numpy as np
import time
import argparse
import random
import pickle
import signal

import sys

# Get the path to the parent directory of the gvp directory
gvp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory of gvp to the module search path
sys.path.append(gvp_dir)

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, IterableDataset

from torch_geometric.loader import DataLoader as geom_DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins.environments import SLURMEnvironment

# from atom3d.datasets import LMDBDataset
from gvp_antonia.lmdb_dataset import LMDBDataset

# from gvp_antonia.models.equiformer import GraphAttentionTransformer
from gvp_antonia.models.gvp import RES_GVP
# from gvp_antonia.models.mace import RES_MACEModel
from gvp_antonia.models.eqgat import RES_EQGATModel

from gvp_antonia.protein_graph import AtomGraphBuilder, _element_alphabet

DATASET_PATH = '/Users/antoniaboca/Downloads/split-by-cath-topology/data'
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)

MODEL_SELECT = {'gvp': RES_GVP, 'mace': None, 'eqgat': RES_EQGATModel, 'equiformer': None }

class ModelWrapper(pl.LightningModule):
    def __init__(self, 
        model_name, 
        lr, 
        example, 
        dropout, 
        patience_scheduler: int = 10,
        factor_scheduler: float = 0.75,
        **model_args
    ):
        super().__init__()
        model_cls = MODEL_SELECT[model_name]
        self.model = model_cls(example, dropout, **model_args)
        self.lr = lr
        self.patience_scheduler = patience_scheduler
        self.factor_scheduler = factor_scheduler

        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=self.factor_scheduler,
            patience=self.patience_scheduler,
            min_lr=1e-7,
            verbose=True,
        )

        schedulers = [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }
        ]
        return [optimiser], schedulers

    def training_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = graph.label.to(self.device)
        loss = self.loss_fn(out, labels)
        acc = torch.sum(torch.argmax(out, dim=-1) == labels)
        self.log('train_loss', loss)

        return {'loss': loss, 'acc': acc, 'n_graphs': len(labels)}
    
    def training_epoch_end(self, outputs):
        correct_graphs = 0
        total_graphs = 0
        total_loss = 0.0
        for output in outputs:
            correct_graphs += output['acc']
            total_graphs += output['n_graphs']
            total_loss += output['loss']
        acc = 1.0 * correct_graphs / total_graphs
        total_loss /= len(outputs)

        self.log('train_acc_on_epoch_end', acc, sync_dist=True)
        self.log('train_loss_on_epoch_end', total_loss, sync_dist=True)

    def validation_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = graph.label.to(self.device)
        loss = self.loss_fn(out, labels)
        acc = torch.sum(torch.argmax(out, dim=-1) == labels)
        self.log('val_loss', loss, batch_size = len(labels))

        return {'loss': loss, 'acc': acc, 'n_graphs': len(labels)}
    
    def validation_epoch_end(self, outputs):
        correct_graphs = 0
        total_graphs = 0
        total_loss = 0.0
        for output in outputs:
            correct_graphs += output['acc']
            total_graphs += output['n_graphs']
            total_loss += output['loss']
        acc = 1.0 * correct_graphs / total_graphs
        total_loss /= len(outputs)

        self.log('val_acc_on_epoch_end', acc, sync_dist=True)
        self.log('val_loss_on_epoch_end', total_loss, sync_dist=True)
    
    def test_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = graph.label.to(self.device)
        
        loss = self.loss_fn(out, labels)

        acc = torch.sum(torch.argmax(out, dim=-1) == labels)
        self.log('test_acc', acc, batch_size=len(labels))
        self.log('test_loss', loss, batch_size=len(labels))

        return {'loss': loss, 'acc': acc, 'n_graphs': len(labels)}
    
    def test_epoch_end(self, outputs):
        correct_graphs = 0
        total_graphs = 0
        total_loss = 0.0
        for output in outputs:
            correct_graphs += output['acc']
            total_graphs += output['n_graphs']
            total_loss += output['loss']

        acc = 1.0 * correct_graphs / total_graphs
        total_loss /= len(outputs)

        self.log('test_acc_on_epoch_end', acc, sync_dist=True)
        self.log('test_loss_on_epoch_end', total_loss, sync_dist=True)
        
        return {'accuracy': acc, 'test_loss': total_loss}


class RESDataset(IterableDataset):
    def __init__(self, dataset_path, max_len=None, sample_per_item=None, shuffle=False):
        self.dataset = LMDBDataset(dataset_path)
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
        self.shuffle = shuffle
        self.max_len = max_len
        self.sample_per_item = sample_per_item

    def __iter__(self):
        length = len(self.dataset)
        if self.max_len:
            length = min(length, self.max_len)
        indices = list(range(length))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(indices)
        else:  
            per_worker = int(math.ceil(length / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, length)
            gen = self._dataset_generator(indices[iter_start:iter_end])
        return gen

    def _dataset_generator(self, indices):
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            item = self.dataset[idx]

            atoms = item['atoms']
            # Mask the residues one at a time for a single graph
            if self.sample_per_item:
                limit = self.sample_per_item
            else:
                limit = len(item['labels'])
            for sub in item['labels'][:limit].itertuples():
                _, num, aa = sub.subunit.split('_')
                num, aa = int(num), _amino_acids(aa)
                if aa == 20:
                    continue
                assert aa is not None

                my_atoms = atoms.iloc[item['subunit_indices'][sub.Index]].reset_index(drop=True)
                ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                if len(ca_idx) != 1: continue
                graph = self.graph_builder(my_atoms)
                graph.label = aa
                graph.ca_idx = int(ca_idx)
                graph.ensemble = item['id']
                yield graph


def train(args):
    pl.seed_everything(args.seed)
    pinned = args.gpus > 0
    train_dataloader = geom_DataLoader(RESDataset(os.path.join(args.data_file, 'train'), shuffle=True, 
                        max_len=args.max_len, sample_per_item = args.sample_per_item), 
                        batch_size=args.batch_size, num_workers=args.data_workers, pin_memory=pinned)
    val_dataloader = geom_DataLoader(RESDataset(os.path.join(args.data_file, 'val'), 
                        max_len=args.max_len, sample_per_item = args.sample_per_item), 
                        batch_size=args.batch_size, num_workers=args.data_workers, pin_memory=pinned)
    test_dataloader = geom_DataLoader(RESDataset(os.path.join(args.data_file, 'test'), 
                        max_len=args.max_len, sample_per_item = args.sample_per_item), 
                        batch_size=args.batch_size, num_workers=args.data_workers, pin_memory=pinned)


    example = next(iter(train_dataloader))
    model = ModelWrapper(args.model, args.lr, example, args.dropout, n_layers=args.n_layers)

    root_dir = os.path.join(CHECKPOINT_PATH, args.model)
    os.makedirs(root_dir, exist_ok=True)
    if args.resume_checkpoint is None:
        wandb_logger = WandbLogger(project='part3-res-prediction-diss')
    else:
        wandb_logger = WandbLogger(project='part3-res-prediction-diss', id=args.wandb_id, resume='must')

    # lt.monkey_patch()
    if args.gpus > 0:
        if args.slurm:
            plugins = [SLURMEnvironment(requeue_signal=signal.SIGHUP)]
        else:
            plugins = None

        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[
                ModelCheckpoint(mode="max", monitor="val_acc_on_epoch_end"), 
                ModelCheckpoint(mode="max", monitor="epoch"), # saves last completed epoch 
                LearningRateMonitor()],   
            log_every_n_steps=1,
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=args.gpus,
            num_nodes=args.num_nodes,
            strategy='ddp',
            logger=wandb_logger,
            plugins=plugins
        ) 
    else:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[
                ModelCheckpoint(mode="max", monitor="val_acc_on_epoch_end"),
                ModelCheckpoint(mode="max", monitor="epoch"), # saves last completed epoch 
                LearningRateMonitor()],
            log_every_n_steps=1,
            max_epochs=args.epochs,
            logger=wandb_logger
        )

    ckpt_path = args.resume_checkpoint
    print('Start training...')
    start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    end = time.time()
    print('TRAINING TIME: {:.4f} (s)'.format(end - start))
    best_model = ModelWrapper(args.model, args.lr, example, 0.0, n_layers=args.n_layers)
    best_model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    
    test_result = trainer.test(best_model, test_dataloader)
    print(test_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gvp', choices=['gvp', 'mace', 'eqgat', 'equiformer'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--data_file', type=str, default=DATASET_PATH)
    parser.add_argument('--data_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--sample_per_item', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--slurm', action='store_true', help='Whether or not this is a SLURM job.')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--wandb_id', type=str, default=None)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()