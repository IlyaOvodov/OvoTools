import random
import numpy as np
import torch

SEED = 241075

# TODO
#PYHTONHASHSEED
#https://github.com/n01z3/kaggle-pneumothorax-segmentation/blob/master/n15_train.py#L32-L49
#https://pytorch.org/docs/stable/generated/torch.set_deterministic.html#torch.set_deterministic

def set_reproducibility(seed = SEED):
    '''
    attempts to make calculations reproducible
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True

def reproducibility_worker_init_fn(seed = SEED):
    def worker_init_fn(worker_id):
        np.random.seed(SEED)
    return worker_init_fn