import os
import math
import numpy as np
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from atom3d.datasets import LMDBDataset

from gvp import GVP_GNN
from protein_graph import AtomGraphBuilder, _element_alphabet

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

class RES_GVP(nn.Module):
    def __init__(self, example, **model_args):
        super().__init__()
        ns, _ = _DEFAULT_V_DIM
        self.gvp = GVP_GNN.init_from_example(example, **model_args)
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2*ns, 20)
        )   
    
    def forward(self, graph):
        out = self.gvp(graph)
        out = self.dense(out)
        return out

MODEL_SELECT = {'gvp': RES_GVP }

class ModelWrapper(pl.LightningModule):
    def __init__(self, model_cls, lr, example, **model_args):
        super().__init__()
        self.model = model_cls(example, device=self.device, **model_args)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        return optimiser

    def training_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = torch.tensor(graph.label, dtype=torch.long).unsqueeze(-1)
        
        loss = self.loss_fn(out, labels)
        acc = out[0, torch.argmax(out[0], dim=-1)] == labels[0]
        self.log('train_loss', loss, batch_size=1)

        return {'loss': loss, 'acc': acc}
    
    def training_epoch_end(self, outputs):
        sum = 0
        total_loss = 0.0
        for output in outputs:
            sum += output['acc']
            total_loss += output['loss']
        acc = 1.0 * sum / len(outputs)

        self.log('train_acc_on_epoch_end', acc, batch_size=1)
        self.log('train_loss_on_epoch_end', total_loss, batch_size=1)

    def validation_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = torch.tensor(graph.label, dtype=torch.long).unsqueeze(-1)

        loss = self.loss_fn(out, labels)
        acc = out[0, torch.argmax(out[0], dim=-1)] == labels[0]
        self.log('val_loss', loss, batch_size=1)

        return {'loss': loss, 'acc': acc}
    
    def validation_epoch_end(self, outputs):
        sum = 0
        total_loss = 0.0
        for output in outputs:
            sum += output['acc']
            total_loss += output['loss']
        acc = 1.0 * sum / len(outputs)

        self.log('val_acc_on_epoch_end', acc, batch_size=1)
        self.log('val_loss_on_epoch_end', total_loss, batch_size=1)
    
    def test_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = torch.tensor(graph.label, dtype=torch.long).unsqueeze(-1)
        
        loss = self.loss_fn(out, labels)
        acc = out[0, torch.argmax(out[0], dim=-1)] == labels[0]
        self.log('test_acc', acc, batch_size=1)
        self.log('test_loss', loss, batch_size=1)

        return {'loss': loss, 'acc': acc}
    
    def test_epoch_end(self, outputs):
        sum = 0
        total_loss = 0.0
        for output in outputs:
            sum += output['acc']
            total_loss += output['loss']
        acc = 1.0 * sum / len(outputs)

        self.log('test_acc_on_epoch_end', acc)
        self.log('test_loss_on_epoch_end', total_loss)
        
        return {'accuracy': acc, 'test_loss': total_loss}

class RESDataset(IterableDataset):
    def __init__(self, dataset_path, shuffle=False):
        self.dataset = LMDBDataset(dataset_path)
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
        self.shuffle = shuffle

    def __iter__(self):
        length = len(self.dataset)
        indices = list(range(length))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(indices)
        else:  
            per_worker = int(math.ceil(length / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(indices[iter_start:iter_end])
        return gen

    def _dataset_generator(self, indices):
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            item = self.dataset[idx]

            atoms = item['atoms']
            # Mask the residues one at a time for a single graph
            for sub in item['labels'].itertuples():
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
                yield graph
                

def train(args):
    pl.seed_everything(42)
    train_dataloader = DataLoader(RESDataset(os.path.join(args.data_file, 'train'), shuffle=True), 
                        batch_size=None)
    val_dataloader = DataLoader(RESDataset(os.path.join(args.data_file, 'val')), 
                        batch_size=None)
    test_dataloader = DataLoader(RESDataset(os.path.join(args.data_file, 'test')), 
                        batch_size=None)

    pl.seed_everything()
    example = next(iter(train_dataloader))
    model_cls = MODEL_SELECT[args.model]
    model = ModelWrapper(model_cls, args.lr, example, n_layers=args.n_layers)

    root_dir = os.path.join(CHECKPOINT_PATH, args.model)
    os.makedirs(root_dir, exist_ok=True)

    # wandb_logger = WandbLogger(project='part3-res-prediction-diss')

    if args.gpus > 0:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", 
                                        monitor="val_acc")],
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=args.gpus,
            strategy='ddp',
            # logger=wandb_logger,
        ) 
    else:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", 
                                        monitor="val_acc")],
            max_epochs=args.epochs,
            # logger=wandb_logger
        )

    print('Start training...')
    start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)
    end = time.time()
    print('TRAINING TIME: {:.4f} (s)'.format(end - start))
    best_model = ModelWrapper.load_from_checkpoint(
                                trainer.checkpoint_callback.best_model_path)
    
    test_result = trainer.test(best_model, test_dataloader)
    print(test_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gvp', choices=['gvp'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--data_file', type=str, default=DATASET_PATH)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()