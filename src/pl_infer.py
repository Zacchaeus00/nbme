import json
import os
from pprint import pprint
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)
from transformers import DataCollatorForTokenClassification

from data_utils import NBMEDatasetInfer
from model_utils import NBMEModel
from eval_utils import get_char_logits, get_spans
from arguments import parse_args_pl_infer
from utils import get_tokenizer, check_if_fix_offsets
from tqdm import tqdm

cfg = parse_args_pl_infer()
pl_df = pd.read_pickle(cfg.data_path)
blend_log = json.load(open(cfg.blend_log, 'r'))
pprint(vars(cfg))
print('blend log:', blend_log)
assert set(blend_log['individual'].keys()) == set(cfg.model_dirs), "model dirs not match"

pl_df['char_logits_blend'] = pl_df['pn_history'].apply(lambda x: np.zeros(len(x)))
for i, pretrained_ckpt in enumerate(cfg.pretrained_checkpoints):
    print(i, pretrained_ckpt)
    model_dir = cfg.model_dirs[i]
    tokenizer = get_tokenizer(pretrained_ckpt)
    model = NBMEModel(pretrained_ckpt).cuda()
    results = {}  # {id: char_logits}
    for fold in range(5):
        pl_df_fold = pl_df[pl_df['fold'] == fold].reset_index(drop=True)
        pl_dataset_fold = NBMEDatasetInfer(tokenizer, pl_df_fold)
        dataset_len = len(pl_dataset_fold)
        maxlen = max([len(x['input_ids']) for x in pl_dataset_fold])
        dataloader = DataLoader(pl_dataset_fold, batch_size=cfg.batch_size, shuffle=False,
                                collate_fn=DataCollatorForTokenClassification(tokenizer), pin_memory=True)
        del pl_dataset_fold
        gc.collect()
        model.load_state_dict(torch.load(os.path.join(model_dir, f'{fold}.pt')))
        model.eval()
        preds = []
        for b in tqdm(dataloader, total=dataset_len // cfg.batch_size + 1):
            b = {k: v.cuda() for k, v in b.items()}
            pred = model(**b).logits.squeeze()  # [bs, maxlen, 1]
            pred = F.pad(input=pred, pad=(0, maxlen - pred.shape[1]), mode='constant', value=-100).cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)  # [n, maxlen]
        print(f'preds memory size : {preds.size * preds.itemsize / (1024**2)} MB')
        if_fix_offsets = check_if_fix_offsets(pretrained_ckpt)
        print('if_fix_offsets:', if_fix_offsets)
        char_logits = get_char_logits(pl_df_fold['pn_history'].values, preds, tokenizer, do_fix_offsets=if_fix_offsets)
        del preds
        gc.collect()
        results.update({k: v for k, v in zip(pl_df_fold['id'], char_logits)})
        del char_logits
        gc.collect()
    pl_df = pl_df.merge(pd.DataFrame({'id': list(results.keys()), model_dir: list(results.values())}), on='id',
                        how='left')
    del results
    gc.collect()

all_char_logits = []
weights = []
for i, model_dir in enumerate(cfg.model_dirs):
    all_char_logits.append(pl_df[model_dir].values)
    weights.append(blend_log['weights'][model_dir])
pl_df['char_logits_blend'] = np.average(np.array(all_char_logits), weights, axis=0)
all_spans = get_spans(pl_df['char_logits_blend'].values, pl_df['pn_history'].values)
results = []
for spans in all_spans:
    result = []
    for s, e in spans:
        result.append(f'{s} {e}')
    results.append(result)
pl_df['location'] = results
annotations = []
for i in range(len(pl_df)):
    annotation = []
    text = pl_df.loc[i, 'pn_history']
    spans = all_spans[i]
    for s, e in spans:
        annotation.append(text[s:e])
    annotations.append(annotation)
pl_df['annotation'] = annotations
pl_df['annotation_length'] = pl_df['annotation'].apply(len)
uid = cfg.blend_log.split('.')[-2][-4:]
pl_df.to_pickle(os.path.join(cfg.out_dir, f'pl-{uid}.pkl'))
