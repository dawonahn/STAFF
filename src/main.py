import wandb
import argparse
from model import *
from read import *
from train import *
from utils import *
from dotmap import DotMap


def parse_args():

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--wnb_project', type=str, help="Wnb project name")
    parser.add_argument('--dpath', type=str, default='./data', help='Dataset path')
    parser.add_argument('--opath', type=str, default='./output', help='Training result path')
    parser.add_argument('--name', type=str, default='lastfm_time', help='Dataset name')
    parser.add_argument('--unfair', type=float, default=0, help="Ratio of sparsification")
    
    # Augmentation
    parser.add_argument('--sampling', type=str, nargs="?", help="knn")
    parser.add_argument('--K', type=int, nargs="?", default=3, help='# of neighbors')
    parser.add_argument('--gamma', type=float, nargs="?", help="Adjust the contribution of group for graph")
    parser.add_argument('--wd2', type=float, nargs="?", help="Penalty for a fairness regularizer")
    parser.add_argument('--aug_training', type=bool, nargs="?", help='Whether training the augmentation or using the saved augmentation')
    parser.add_argument('--only_aug_save', type=bool, default=False, help="Option for only making augmentation")

    # Model and its hyper parameters
    parser.add_argument('--tf', type=str, help='tensor')
    parser.add_argument('--rank', type=int, help='Rank size of tf')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--wd', type=float, help='Weight decay')
    parser.add_argument('--bs', type=int, help='Batch size', nargs='?', default=1024)
    parser.add_argument('--n_iters', type=int, help='Iteration for training', default=10000)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--random', type=int, help='Random seed')

    # CoSTCo hyper-params
    parser.add_argument('--nc', type=int, help='Hidden dimension', nargs='?')

    args = parser.parse_args()
    dict_args = DotMap(dict(args._get_kwargs()))
    dict_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dict_args.aug_number = int(dict_args.random - 1) 

    return dict_args

ENTITY=""
PROJECT=""

def main():

    cfg = parse_args()
    cfg.verbose = True

    # wandb = None
    run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            config = cfg,
    )
    cfg.wnb_name = wandb.run.name

    # Read datasets
    tensor = TensorDataset(cfg=cfg, path=cfg.dpath, name=cfg.name)
    tensor.load_data()
    cfg.sizes = tensor.sizes

    if cfg.sampling:
        read_augment(tensor, cfg, wandb)
        cfg.sizes = tensor.sizes

    if cfg.only_aug_save is False:
        model = get_model(cfg)
        trainer = Trainer(model, tensor, cfg, wandb)
        trainer.train()
    
        # Evaluate and save the model
        evaluate_model(model, tensor, wandb)
        save_checkpoints(model, cfg)

if __name__ ==  '__main__':
    main()

