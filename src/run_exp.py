import sys
import yaml
import itertools
import subprocess
from main import main as run

def load_config(config_path, rpath='./config'):
    with open(f"{rpath}/{config_path}.yaml", "r") as file:
        return yaml.safe_load(file)

def generate_combinations(hparams):
    return itertools.product(hparams)

def main(file, dataset, gpu_device, random_val):
    all_configs = load_config(file)
    fixed_configs = all_configs['config']
    exp_configs = all_configs['datasets'][dataset]
    hparams_configs = exp_configs['hparams']

    command = [ f"CUDA_VISIBLE_DEVICES={gpu_device}","python", "./src/main.py",
                f"--name {dataset}", f"--random {random_val}"]
    fixed_args = [f"--{key} {value}" for key, value in fixed_configs.items()]
    exp_args = []
    for ufair, vals in exp_configs['unfair'].items():
        exp_args.append(f'--unfair {ufair}')

        key_params = list(hparams_configs.keys()) + list(vals.keys())
        val_params = list(hparams_configs.values()) + list(vals.values())
        
        combinations = itertools.product(*val_params)
        for comb in combinations:
            exp_args = [f'--unfair {ufair}'] + [f'--{k} {v}' for k, v in zip(key_params, comb)]
            
            total_args = [*command, *fixed_args, *exp_args]
            print(" ".join(total_args))
            subprocess.run(" ".join(total_args), shell=True)

if __name__ == "__main__":
    
    file = sys.argv[1]  
    dataset = sys.argv[2]  
    gpu_device = sys.argv[3]  
    random_val = sys.argv[4]  
    
    main(file, dataset, gpu_device, random_val)
