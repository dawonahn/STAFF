import torch
from dotmap import DotMap


def recon(factors, idxs):
    '''
    Reconstruct COO type tensor with a given indices.
    '''
    nmode = len(factors)
    facs = [factors[m][idxs[:, m]] for m in range(nmode)]
    rec = torch.einsum('bi,bi,bi->b', facs) # Element-wise product and sum
    return rec

def rmse_(val, rec):
    '''
    Calculate Root Mean Squared Error (RMSE)
    '''
    return torch.sqrt(torch.mean((val-rec) ** 2))

def nre_(val, rec):
    '''
    Calculate Normalized Reconstruction Error (NRE)
    '''
    return torch.norm(val-rec)/torch.norm(val)

def eval_(factors, idxs, vals):
    '''
    Evaluate a model on NRE and RMSE.
    '''
    rec = recon(factors, idxs)
    nre = nre_(vals, rec).item()
    rmse = rmse_(vals, rec).item()    
    return nre, rmse

def evaluate_model(model, tensor, wandb=None, verbose=True):
    '''
    Evaluate fairness of model.
    '''
    res = DotMap()

    res.test_nre, res.test_rmse = model.eval_(tensor.test_i, tensor.test_v)
    print(f"Test NRE: {res.test_nre:.4f} Test RMSE: {res.test_rmse:.4f}")
    df_list = [tensor.train_df, tensor.valid_df, tensor.test_df]
    idx_list = [tensor.train_i, tensor.valid_i, tensor.test_i]
    for df, idx in zip(df_list, idx_list):
        rec_v = model.recon(idx)
        df['Recon'] = rec_v.detach().cpu().numpy()
        df['Error'] = (df[tensor.val] - df['Recon']) ** 2
    
    err_col = ['Error', 'Recon']
    _maj = tensor.majority[0]
    _min = tensor.minority[0]

    if verbose:
        print("***********************************************************************************") 
        print("Calculate group fairness...")
    # Group difference
    mean_df = tensor.test_df[[tensor.s_attr] + err_col].groupby(tensor.s_attr)[err_col].mean()
    for col in err_col:
        res[f'MAD_{col}'] = mean_df[col][_maj] - mean_df[col][_min] # MADE or MADR
        res[f'Group0_{col}'] = mean_df[col][_maj]
        res[f'Group1_{col}'] = mean_df[col][_min]
    if verbose:        
        print("***********************************************************************************") 

    if wandb:
        wandb.log(res)
    return res