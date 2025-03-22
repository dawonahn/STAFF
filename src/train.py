import time
import copy
import torch
import torch.optim as optim
from knn import *
from read import *


def rec_loss(rec, val):
    return ((rec - val) ** 2).sum()

class Trainer:
    """
    A general Trainer class to train tensor decomposition models.
    """
    def __init__(self, model, tensor, cfg, wandb):

        self.cfg = cfg
        self.model = model
        self.tensor = tensor
        self.wandb = wandb
        self.print_iter = 1
        self.device = cfg.device

        if self.cfg.tf == 'cpd':
            self.reduce = 'sum'
            self.loss_fn = rec_loss
        else:
            self.loss_fn = nn.MSELoss()
            self.reduce = 'mean'

        if self.cfg.sampling:
            self.train_batch = self.train_fair_knn_one_batch
        else:
            self.train_batch = self.train_one_batch

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)

    def train(self):
        self.break_loop = False
        self.stop = 0
        self.old_err = 1e+10
        self.best_err = 1e+10
        self.training_res = []
        start = time.time()
        for n_iter in range(1, self.cfg.n_iters):
            self.batch_loss = 0
            self.fair_batch_loss = 0    
            self.model.train()
            for batch in self.tensor.dataloader:
                self.optimizer.zero_grad()
                self.train_batch(batch)
                self.optimizer.step() 
             
            self.eval_validation(n_iter)
            if self.break_loop:
                break
            
        end = time.time()
        self.elapsed_time = end - start
        self.model.load_state_dict(self.best_model)
        self.model.best_iter = self.best_iter

    def train_one_batch(self, batch):

        idx, val = batch[0], batch[1]
        rec = self.model(idx)
        rec_loss = self.loss_fn(rec, val)
        loss = rec_loss
        loss.backward()
        self.batch_loss += rec_loss.item()        

    def train_fair_knn_one_batch(self, batch):

        idx, val = batch[0], batch[1]
        rec = self.model(idx)
        rec_loss = self.loss_fn(rec, val)
        fair_loss = 0    

        for synth_data in self.tensor.synth_data:
            fair_loss = fair_loss + self.knn_regularizer(synth_data, idx)
        loss = rec_loss + fair_loss * self.cfg.wd2
        loss.backward()

        self.batch_loss += rec_loss.item()        
        self.fair_batch_loss += fair_loss.item()

    def knn_regularizer(self, synth_data, idx=None):
        '''
            dim: mode of the tensor where this regularize will be applied (sdim or nsdim) 
            idx1: original index of entities
            idx2: fake index of entities
        '''
        dim, dist, graph, idx1, idx2 = iter(synth_data)
        factor = self.model.factors[dim]
        reg = ((factor[idx1] - factor[idx2]) ** 2).sum(1) # row-wise

        return (reg).sum()
    
    def eval_validation(self, n_iter):
        self.model.eval()
        with torch.no_grad():
            train_nre, train_rmse = self.model.eval_(self.tensor.train_i, self.tensor.train_v)
            valid_nre, valid_rmse = self.model.eval_(self.tensor.valid_i, self.tensor.valid_v)
            self.training_res.append([train_nre, train_rmse, valid_nre, valid_rmse])

        if self.wandb:
            self.wandb.log({'train_nre':train_nre , 'train_rmse':train_rmse,
                            'valid_rmse':valid_rmse, 'valid_nre': valid_nre})
            
        if n_iter % self.print_iter == 0:
            if (self.cfg.fair_type) or (self.cfg.sampling == 'knn') or (self.cfg.regularizer):
                print(f"Iters:{n_iter:>4} || training loss: {self.batch_loss:.5f}\t"
                      f"fair loss: {self.fair_batch_loss:.5f}\t"
                      f"Train RMSE: {train_rmse:.5f} Valid RMSE: {valid_rmse:.5f}\t")
            else:
                print(f"Iters:{n_iter:>4} || training loss: {self.batch_loss:.5f}\t"
                      f"Train RMSE: {train_rmse:.5f} Valid RMSE: {valid_rmse:.5f}\t")

        if valid_rmse != valid_rmse:
            self.break_loop = True
        if self.model.cfg.tf == 'costco':
                stop_type = 25
        else:
            stop_type = 5
        if self.old_err <= valid_rmse:
            if self.stop >= stop_type:
                self.break_loop = True
            self.stop += 1
        else:
            if self.best_err > valid_rmse:
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_iter = n_iter
                self.best_err = valid_rmse

        self.old_err = valid_rmse

