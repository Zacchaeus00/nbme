import os
import random

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
    return timenow


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def get_tokenizer(checkpoint):
    if 'deberta-v3' in checkpoint:
        from transformers import DebertaV2TokenizerFast
        return DebertaV2TokenizerFast.from_pretrained(checkpoint)
    return AutoTokenizer.from_pretrained(checkpoint)

if __name__ == "__main__":
    # tok = get_tokenizer('microsoft/deberta-v3-large')
    tok = get_tokenizer('roberta-large')
    tok(['i am a student'], return_offsets_mapping=True)