import os
import pickle
import random
import uuid

import numpy as np
import torch
import datetime
import json
from transformers import AutoTokenizer, DebertaV2Tokenizer


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
    uid = str(uuid.uuid4())[:4]
    return timenow + '-' + uid


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_tokenizer(checkpoint):
    return AutoTokenizer.from_pretrained(checkpoint, trim_offsets=False)


def get_uid():
    return str(uuid.uuid4())[:4]


def check_if_fix_offsets(checkpoint):
    do_fix_offsets_names = ['electra', 'ernie', 'albert', 'funnel', 'mpnet']
    for name in do_fix_offsets_names:
        if name in checkpoint:
            return True
    return False

if __name__ == "__main__":
    # tok = get_tokenizer('microsoft/deberta-v3-large')
    tok = get_tokenizer('roberta-large')
    print(tok(['i am a student'], return_offsets_mapping=True))
