import torch
import torch.nn as nn
import tensorly as tl
from metric import *
from tensorly import check_random_state
tl.set_backend('pytorch')


def gen_random(random_state, size):
    ''' 
    Make random values with a given size.
    '''
    #if len(size) >1, size must be tuple
    rng = check_random_state(random_state)
    random_value = torch.FloatTensor(rng.random_sample(size))
    return random_value

class CPD(nn.Module):

    def __init__(self, cfg):
        super(CPD, self).__init__()

        self.cfg = cfg
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        self.nmode = len(self.sizes)
        # Factor matrices
        self.factors = nn.ParameterList([nn.Parameter(gen_random(cfg.random, (mode, self.rank)))
                                         for mode in self.sizes])
    
    def recon(self, idxs, factors=None):

        '''
        Reconstruct a tensor entry with a given index
        '''
        if factors is None:
            factors = [self.factors[m][idxs[:, m]].unsqueeze(-1) for m in range(self.nmode)]
        concat = torch.concat(factors, dim=-1) # NNZ x rank x nmode
        rec = torch.prod(concat, dim=-1)    # NNZ x rank
        return rec.sum(-1)

    def forward(self, idxs):
        return self.recon(idxs)

    def eval_(self, idxs, vals):
        '''
        Evaluate a model on NRE and RMSE.
        '''
        with torch.no_grad():
            rec = self.recon(idxs)
            nre = nre_(vals, rec).item()
            rmse = rmse_(vals, rec).item()
        return nre, rmse

class CoSTCo(nn.Module):
    def __init__(self, cfg):
        super(CoSTCo, self).__init__()
        nc = int(cfg.nc)
        self.cfg = cfg
        self.rating = cfg.rating
        self.rank = cfg.rank
        self.sizes = cfg.sizes
        # print(cfg.nc, cfg.sizes)

        self.factors = nn.ParameterList([nn.Parameter(gen_random(cfg.random, (mode, self.rank)))
                                         for mode in self.sizes])
        # self.factors = nn.ModuleList([nn.Embedding(self.sizes[i], self.rank)
                                    #  for i in range(len(self.sizes))])
        self.conv1 = nn.Conv2d(1, nc, kernel_size=(1, len(self.sizes)), padding=0)
        self.conv2 = nn.Conv2d(nc, nc, kernel_size=(self.rank, 1), padding=0)
        self.fc1 = nn.Linear(nc, nc)
        self.fc2 = nn.Linear(nc, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.last_act = nn.Sigmoid() if self.rating == 'bin' else nn.ReLU()
        self._initialize()
        
    def _initialize(self):
        for i in range(len(self.factors)):
            nn.init.kaiming_uniform_(self.factors[i].data)
        
    def recon(self, idxs, factors=None):

        '''
        Reconstruct a tensor entry with a given index
        '''
        if factors is None:
            factors = [self.factors[m][idxs[:, m]].reshape(-1, self.rank, 1)
                      for m in range(len(self.sizes))]
        x = torch.cat(factors, dim=-1)
        x = x.reshape(-1, 1, self.rank, len(self.sizes))# NNZ_batch x 1 x rank x nmode 
        
        # CNN on stacked factors
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Aggregate with mlps
        x = x.view(-1, x.size(1))
        x = self.relu(self.fc1(x))
        x = self.last_act(self.fc2(x))
        return x.reshape(-1)

    def forward(self, idxs):
        return self.recon(idxs)

    def eval_(self, idxs, vals):
        with torch.no_grad():
            rec = self.recon(idxs)
            nre = nre_(vals, rec).item()
            rmse = rmse_(vals, rec).item()
        return nre, rmse
    