import os
import random
import numpy as np
import torch
import math

# for get_cosine_scheduler_with_warmup
from torch.optim.lr_scheduler import LambdaLR

def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
        
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def search_folders_name(dirname):
    filelist = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                if (path + '\\') not in filelist:
                    filelist.append("%s\\"%(path))
    return filelist

def search_folders(dirname):
    filelist = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                filelist.append("%s\\%s"%(path,filename))
    return filelist

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list