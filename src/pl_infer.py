import os
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)
from transformers import DataCollatorForTokenClassification

from data_utils import NBMEDatasetInfer
from model_utils import NBMEModel
from eval_utils import get_char_logits
from arguments import parse_args_pl_infer
from utils import get_tokenizer, save_pickle
from tqdm import tqdm

cfg = parse_args_pl_infer()
pl_df = pd.read_pickle(cfg.data_path)
pprint(vars(cfg))
print(f'{len(pl_df)} rows')

tokenizer = get_tokenizer(cfg.pretrained_checkpoint)
model = NBMEModel(cfg.pretrained_checkpoint).cuda()
results = {}  # {id: char_logits}
for fold in range(5):
    pl_df_fold = pl_df[pl_df['fold'] == fold].reset_index(drop=True)
    pl_dataset_fold = NBMEDatasetInfer(tokenizer, pl_df_fold)
    dataset_len = len(pl_dataset_fold)
    maxlen = max([len(x['input_ids']) for x in pl_dataset_fold])
    dataloader = DataLoader(pl_dataset_fold, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=DataCollatorForTokenClassification(tokenizer), pin_memory=True)
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, f'{fold}.pt')))
    model.eval()
    preds = []
    for b in tqdm(dataloader, total=dataset_len // cfg.batch_size + 1):
        b = {k: v.cuda() for k, v in b.items()}
        pred = model(**b).logits.squeeze()  # [bs, maxlen, 1]
        pred = F.pad(input=pred, pad=(0, maxlen - pred.shape[1]), mode='constant', value=-100).cpu().numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)  # [n, maxlen]
    char_logits = get_char_logits(pl_df_fold['pn_history'].values, preds, tokenizer, do_fix_offsets=cfg.do_fix_offsets)
    results.update({k: v for k, v in zip(pl_df_fold['id'], char_logits)})
print(f'{len(results)} results')
save_pickle(results, os.path.join(cfg.model_dir, 'pl_logits.pkl'))