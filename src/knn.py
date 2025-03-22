import copy
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from utils import *


def find_interaction(mode, entities, val, train_df, graph):
    '''
    Generate augmented nonzero entries.
    '''
    # Duplicated entries will be removed or averaged later
    cols = train_df.columns[:3].tolist()
    cols2 = copy.deepcopy(cols)
    cols2.remove(mode)
    org_entity = train_df.groupby(mode).sample(30, replace=True)
    lst = []
    for i in range(len(entities)):
        try:
            tmp = train_df[train_df[mode].isin(graph[i, 1:])].sample(30, replace=True)[cols2 + [val]]
            tmp[mode] = graph[i, 0]
            lst.append(tmp)
        except:
            pass

    fake_entity = pd.concat(lst)[cols + [val]]
    cols = org_entity.columns[:3].tolist()

    org_entity.loc[:, 'dtype'] = 'org'
    fake_entity.loc[:, 'dtype'] = 'fake'
    synth_entity = pd.concat([org_entity, fake_entity])
    synth_entity = synth_entity.groupby(cols + ['dtype'])[val].mean().reset_index()

    return org_entity, fake_entity, synth_entity

def search_fair_knn(factor, K, features, gamma):
    '''
    Find fairness-aware k-nn.
    '''
    x = factor    
    y = features
    sim = 1 - squareform(pdist(x, 'cosine')) # similarity between entities
    s_sim = squareform(pdist(y, 'cosine'))   # different groups, sim ++
    
    # gamma ++ -> fairness ++ -> (different group ++)
    m_sim = gamma * sim + (1-gamma) * s_sim
    
    # sim ++ , closer neighbor
    graph = np.argsort(m_sim)[:, ::-1][:, :K]
    sims = np.stack([m_sim[i, graph[i]] for i in range(graph.shape[0])])

    graph = np.hstack([np.arange(graph.shape[0]).reshape(-1, 1), graph]) # add myself
    sims = np.hstack([np.ones(graph.shape[0]).reshape(-1, 1), sims]) # add myself

    return sims, graph

def avg_recon(model, factor, indices, dim, tf='cpd'):
    '''
        Predict values with averaged factors with fairness-aware knn.
    '''
    if tf == 'cpd':
        with torch.no_grad():
            dummy_factors = copy.deepcopy(model.factors)
            dummy_factors[dim] = factor
            facs = [dummy_factors[m][indices[:, m]].unsqueeze(-1) for m in range(3)]
            concat = torch.concat(facs, dim=-1) # NNZ x rank x nmode
            rec = torch.prod(concat, dim=-1)    # NNZ x rank
            rec = rec.sum(-1).detach().cpu().numpy()
    else:
        original_factor = copy.deepcopy(model.factors[dim].data)
        model.factors[dim].data = factor.data
        rec = model(indices)
        model.factors[dim].data = original_factor.data
        rec = rec.detach().cpu().numpy()

    return rec


def generate_knn_aug(dim, mode, factor, model, tensor, cfg):
    '''
        Generate augmentation.
    '''
    entities = np.arange(tensor.sizes[dim])

    dist, graph = search_fair_knn(factor, cfg.K, tensor.features, cfg.gamma)

    # Make synthetic factors and corresponding nonzeros
    train_df = tensor.train_df
    org_df, fake_df, df = find_interaction(mode, entities, tensor.val, train_df, graph)

    # Make synthetic entitis' factors by averaging factors    
    new_factor = torch.FloatTensor(factor[graph].mean(1)).to(cfg.device)
       
    cur_df = df
    synth_i = cur_df.iloc[:, :3].values
    synth_i = torch.LongTensor(synth_i).to(cfg.device)
    cur_df['pred'] = avg_recon(model, new_factor, synth_i, dim, tf = cfg.aug_tf) 

    return dist, graph, cur_df

def knn_augmentation(tensor, cfg):
    '''
        Generate augmentation.
    '''
    tensor.aug_df = []
    tensor.synth_data = []
    # Load the pretrained model
    model = load_aug_model(cfg)        
    tensor.original_size = tensor.sizes.copy()
    dct = {}    
    dims = [int(i) for i in cfg.aug_modes.split(',')]
    
    for dim in dims:        
        mode = tensor.cols[dim]
        print(f"Augmentation for the '{mode}' mode ")
        if cfg.aug_training: 
            factor = model.factors[dim].data.cpu().numpy()                

            dct['dist'], dct['graph'], dct['df'] = generate_knn_aug(dim, mode, factor, model, tensor, cfg)
            save_augmentation(mode, dct, cfg)   
        else:
            dct = read_augmentation(mode, cfg)
        group_idx = range(tensor.sizes[dim])
        preprocess_augmentation(dim, mode, group_idx, dct, tensor, cfg)

    tensor.aug_df = pd.concat(tensor.aug_df)
    tensor.train_df = pd.concat([tensor.train_df, tensor.aug_df])


def preprocess_augmentation(dim, mode, group_idx, dct, tensor, cfg):
    '''
        Preprocess indices.
    '''
    # Predict values with a pretrained tf model.
    dct['df'].loc[dct['df'].dtype == 'fake', tensor.val] = dct['df'].loc[dct['df'].dtype == 'fake', 'pred']      

    idx_dct = {j: i + tensor.sizes[dim] for i, j in enumerate(group_idx)}
    dct['df'][mode]  = dct['df'][mode].apply(lambda x: idx_dct[x])
    tensor.sizes[dim] = max(dct['df'][mode].unique().max() + 1, tensor.sizes[dim] + len(group_idx))
    idx1 = list(idx_dct.keys()) # original entities
    idx2 = list(idx_dct.values()) # new entities
    tensor.synth_data.append((dim, torch.FloatTensor(dct['dist']).to(cfg.device),
                                torch.LongTensor(dct['graph']).to(cfg.device), idx1, idx2))

    dct['df']['aug_mode'] = mode
    tensor.aug_df.append(dct['df'])

