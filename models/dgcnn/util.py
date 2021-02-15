import torch
import torch.nn as nn
import torch.nn.functional as F

def find_knn(x, k):
    with torch.no_grad():
        idx = torch.cdist(x.transpose(1,2), x.transpose(1,2))
        return idx.topk(k=k, dim=-1, largest=False)[1]   
    
def get_graph_feature(x, k, idx=None):
    batch_size, num_dims, num_points = x.shape
    
    ## Calculate kNN if are not given
    idx = find_knn(x, k) if idx is None else idx
    
    ## Expand idx and x to same shape
    idx = idx.unsqueeze(1).expand(batch_size, num_dims, num_points, -1)
    x = x.unsqueeze(-1).expand_as(idx)
    
    ## Gather an concatenate kNN
    out = torch.gather(x, 2, idx)
    out -= x
    out = torch.cat((out, x), dim=1)
    return out