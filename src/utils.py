import os
import random

import numpy as np
import torch
import datetime


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_time():
    timenow = str(datetime.datetime.now()).split('.')[0]
    timenow = '-'.join(timenow.split())
    return timenow