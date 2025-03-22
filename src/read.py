import os
import json
import copy
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from knn import *


class COODataset(Dataset):
    def __init__(self, idxs, vals):
        self.idxs = idxs
        self.vals = vals

    def __len__(self):
        return self.vals.shape[0]

    def __getitem__(self, idx):
        return self.idxs[idx], self.vals[idx]
        
class TensorDataset(object):
    def __init__(self, cfg, path, name, verbose=True):
        '''
        path: data path
        name: data name
        device: device
        '''
        self.cfg = cfg
        self.path = path
        self.name = name
        self.device = cfg.device

        if verbose:
            print("***********************************************************************************") 
            print(f"[1] Read {name} self...")
        self.df = pd.read_csv(os.path.join(path, f"{name}.tensor") , sep='\t')

        # set metadata of dataset to attr.
        # Do preprocess related with sensitive attributes
        if verbose:
            print(f"[2] Read metadata...")
        self.dct = json.load(open(os.path.join(path, 'data.json')))[name]
        for key, value in self.dct.items():
            setattr(self, key, value)
        
        # Normalize tensor values ranging from 0 to 1
        if name.startswith('lastfm'):
            print(f"[3] No normalization; values are already binary...")
        else:
            print(f"[3] Normalize values in the range of 0 to 1...")
            self._normalize()

        # Split obeserved entries of tensors into training/valid/test
        if verbose:
            print(f"[4] Split the tensor into training/validation/test")
        self._split() #self.tensor
        # Preprocess sensitive attributes related stuff
        if verbose:
            print(f"[5] Make statistics of group information")
        self._preprocess_sa()

        # Sampling tensor entries
        if verbose:
            print(f"[6] Change the date type into torch")
        self._cast()

        # Print statistics of data
        if verbose:
            print(f"[7] Read {name} tensor done...!")

        self._describe()
        if verbose:
            print("***********************************************************************************") 

    def _normalize(self):
        ''' Normalize tensor values from 0 to 1.
        '''
        if self.name.startswith('chicago'):
            self.df[self.val] = np.log(self.df[self.val]) + 1

        max_val = self.df[self.val].max()
        min_val = self.df[self.val].min()
        self.df[self.val] = (self.df[self.val] - min_val) / (max_val - min_val)

    def _split(self):
        ''' Split the tensor into training/validation/test/leave-one-out based on time.
        '''
        # # # Random
        self.train_df, tmp = train_test_split(self.df, test_size=0.2, random_state=1, shuffle=True)
        self.valid_df, self.test_df = train_test_split(tmp, test_size=0.5, random_state=1, shuffle=True)
        if self.cfg.unfair != 0:
            print(f" [4 - 1] Sparsify the minority group to make it more unfair")
            min_df = self.train_df[self.train_df[self.s_attr].isin(self.minority)]
            maj_df = self.train_df[self.train_df[self.s_attr].isin(self.majority)]
            min_df, _ =  train_test_split(min_df, train_size=self.cfg.unfair, random_state=1, shuffle=True)
            self.train_df = pd.concat([min_df, maj_df])

    def _preprocess_sa(self):
        ''' Preprocess steps for fairness evaluation 
        '''
        # Find minority/majority groups
        self.label1_cnt = self.train_df.groupby(self.s_attr).count().iloc[:, 0][self.majority]
        self.label2_cnt = self.train_df.groupby(self.s_attr).count().iloc[:, 0][self.minority]
        self.maj_nnz_cnt = self.label1_cnt
        self.min_nnz_cnt = self.label2_cnt

        # Sensitive attributes
        if self.name != 'traffic':
            self.s_attr_df = pd.DataFrame(self.s_attr_dct, columns=[self.s_attr_mode,  self.s_attr])
            self.maj_idx = self.s_attr_df[self.s_attr_df[self.s_attr].isin(self.majority)][self.s_attr_mode].values.astype(int)
            self.min_idx = self.s_attr_df[self.s_attr_df[self.s_attr].isin(self.minority)][self.s_attr_mode].values.astype(int)
            self.features = np.vstack([
                    (self.s_attr_df[self.s_attr].isin(self.majority)).values.astype(int),
                    (self.s_attr_df[self.s_attr].isin(self.minority)).values.astype(int)]).T
            self.maj_idx_cnt = self.maj_idx.shape[0]
            self.min_idx_cnt = self.min_idx.shape[0]
        
    def _cast(self):
        ''' Cast a numpy into torch type.
        '''
        cols2 = self.cols
        device = self.device
        self.train_i = torch.LongTensor(self.train_df[cols2].values).to(device)
        self.valid_i = torch.LongTensor(self.valid_df[cols2].values).to(device)
        self.test_i = torch.LongTensor(self.test_df[cols2].values).to(device)

        self.train_v = torch.FloatTensor(self.train_df[self.val].values).to(device)
        self.valid_v = torch.FloatTensor(self.valid_df[self.val].values).to(device)
        self.test_v = torch.FloatTensor(self.test_df[self.val].values).to(device)

        if self.cfg.sampling:
            if self.cfg.sampling == 'random':
                self.original_train_i = copy.deepcopy(self.train_i)
                self.original_train_v = copy.deepcopy(self.train_v)

    def _describe(self):
        ''' Print basic statistics of data.
        '''
        print(f"Tensor      || {self.cols}; {self.val}")
        print(f"NNZ         || {self.sizes}; {self.train_df.shape[0]} | {self.valid_df.shape[0]} | {self.test_df.shape[0]}")
        print(f"Sens. Attr  || {self.s_attr_mode}, {self.s_attr}: maj({self.majority}) min({self.minority})")
        print(f"Entity      || Majority: {self.maj_idx_cnt} Minority: {self.min_idx_cnt}")
        print(f"NNZ         || Majority: {self.maj_nnz_cnt.values} Minority: {self.min_nnz_cnt.values}")

    def load_data(self, cfg=None):
        if cfg:
            self.cfg = cfg

        dataset = COODataset(self.train_i, self.train_v)
        self.dataloader = DataLoader(dataset, batch_size=self.cfg.bs, shuffle=True)
            
def read_augment(tensor, cfg, wandb=None):
    '''
    Read augmentation results.
    '''
    print("***********************************************************************************") 
    print("Augment entities with fair K-NN graph ")
    knn_augmentation(tensor, cfg)

    # Augmentation done
    tensor.train_i = torch.LongTensor(tensor.train_df.iloc[:, :3].values.astype(int)).to(cfg.device)
    tensor.train_v = torch.FloatTensor(tensor.train_df[tensor.val].values).reshape(-1).to(cfg.device)
    
    dataset = COODataset(tensor.train_i, tensor.train_v)
    tensor.dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=True)