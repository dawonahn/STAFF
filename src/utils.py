
import os
import json
import torch
import pandas as pd
from model import *
from dotmap import DotMap


def get_model(cfg):
    ''' Get tensor factorization model.
    '''     
    if cfg.tf == 'cpd':
        f = CPD
    if cfg.tf == 'costco':
        f = CoSTCo
    model = f(cfg).to(cfg.device)
    return model

def load_aug_model(cfg):
    ''' Load tensor factorization model for augmentation.
    '''     
    file_names = [cfg.name, cfg.aug_tf, str(float(cfg.unfair))]
    aug_cfg_file = '_'.join(file_names)
    aug_dct = json.load(open(os.path.join(cfg.opath, 'best_models', f'{aug_cfg_file}.json')))
    aug_cfg = aug_dct[str(cfg.random)]
    aug_cfg['sizes'] = eval(aug_cfg['sizes'])
    aug_cfg = DotMap(aug_cfg)
    aug_cfg.random = cfg.random
    aug_cfg.device = cfg.device

    model_state = torch.load(os.path.join(cfg.opath, aug_cfg['mpath']),
                            map_location=cfg.device)['model_state']    
    model = get_model(aug_cfg)
    model.load_state_dict(model_state)
    model.eval()

    return model

def read_augmentation(mode, cfg):
    ''' Read augmentation results: distance, graph, and df (nonzero entries)
    '''       
    out_path = '/'.join([cfg.opath, cfg.name, 'sampling'])
    comm_filename = '_'.join([str(cfg.unfair), cfg.aug_tf, str(cfg.gamma), str(cfg.K), str(cfg.random)])
    dct = {}
    cols = ['dist', 'graph', 'df']
    for k in cols:
        filename = os.path.join(out_path, comm_filename + '_' + k + '.csv')
        df = pd.read_csv(filename)
        if k == 'df':
            dct[k] = df
        else:
            dct[k] = df.values
        print(f"Read a {k} file from [{filename}] {dct[k].shape}")

    return dct

def save_augmentation(mode, dct, cfg):
    ''' Save augmentation results: distance, graph, and df (nonzero entries)
    '''    
    out_path = '/'.join([cfg.opath, cfg.name, 'sampling'])
    comm_filename = '_'.join([str(cfg.unfair), cfg.aug_tf, str(cfg.gamma), str(cfg.K), str(cfg.random)])

    os.makedirs(out_path, exist_ok=True)
    for k, v in dct.items():
        filename = os.path.join(out_path, comm_filename + '_' + k + '.csv')
        pd.DataFrame(v).to_csv(filename, index=False)
        print(f"Save file as [{filename}]")

def read_checkpoints(config):
    '''Read a model checkpoint.
    '''
    out_path = os.path.join(config.opath, config.name, config.tf)
    model_path = os.path.join(out_path, f'{config.wnb_name}.pt')
    model = torch.load(model_path)

    return model

def save_checkpoints(model, config):
    '''Save a trained model.
    '''
    out_path = os.path.join(config.opath, config.name, config.tf)
    model_path = os.path.join(out_path, f'{config.wnb_name}.pt')
    os.makedirs(out_path, exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), model_path)

    return model_path



        
