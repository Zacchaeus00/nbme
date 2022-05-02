import os
import time
s0 = time.time()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from arguments import parse_args_infer
from data_utils import NBMEDatasetInfer, preprocess_features
from eval_utils import get_char_logits, my_get_results
from model_utils import NBMEModel

torch.set_grad_enabled(False)

cfg = parse_args_infer()
WEIGHTS = {f'w{i}': 1 for i in range(len(cfg.pretrained_checkpoints))}
FIXOFFS = [False for _ in range(len(cfg.pretrained_checkpoints))]

test_df = pd.read_csv(os.path.join(cfg.data_dir, 'train.csv'))
features = preprocess_features(pd.read_csv(os.path.join(cfg.data_dir, 'features.csv')))
pn = pd.read_csv(os.path.join(cfg.data_dir, 'patient_notes.csv'))
test_df = test_df.merge(pn, on='pn_num', how='left')
test_df = test_df.merge(features, on='feature_num', how='left')
test_df['len'] = test_df['pn_history'].apply(len) + test_df['feature_text'].apply(len)
test_df = test_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.loc[:int(cfg.portion*len(test_df)),:]
test_df = test_df.sort_values(by=['len']).reset_index(drop=True)

char_logits_blend = [np.zeros(len(text)) for text in test_df.pn_history.values]
for i, ckpt in enumerate(cfg.pretrained_checkpoints):
    s = time.time()
    w = WEIGHTS[f'w{i}']
    print(f'{i} {ckpt}')
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trim_offsets=False)
    test_dataset = NBMEDatasetInfer(tokenizer, test_df, cache=cfg.cache)
    maxlen = max([len(x['input_ids']) for x in test_dataset])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                                 collate_fn=DataCollatorForTokenClassification(tokenizer), pin_memory=cfg.pin_memory)
    model = NBMEModel(ckpt).cuda()
    preds_folds = []
    for fold in range(1):
        model.eval()
        preds = []
        for b in tqdm(test_dataloader, total=len(test_dataset) // cfg.batch_size + 1):
            b = {k: v.cuda() for k, v in b.items()}
            pred = model(**b).logits.squeeze()  # [bs, maxlen, 1]
            pred = F.pad(input=pred, pad=(0, maxlen - pred.shape[1]), mode='constant', value=-100).cpu().numpy()
            preds.append(pred)
        preds = np.concatenate(preds, axis=0)  # [n, maxlen]
        preds_folds.append(preds)
    preds_folds = np.stack(preds_folds)
    preds = np.mean(preds_folds, axis=0)
    char_logits = get_char_logits(test_df['pn_history'].values, preds, tokenizer, do_fix_offsets=FIXOFFS[i])
    for j in range(len(test_df)):
        char_logits_blend[j] += w * char_logits[j]
    print(f'{i} {ckpt} runtime {time.time()-s}s')

results = my_get_results(char_logits_blend, test_df.pn_history.values)
test_df['location'] = results
sub = pd.read_csv("../input/nbme-score-clinical-patient-notes/sample_submission.csv")
sub = sub[['id']].merge(test_df[['id', "location"]], how="left", on="id")
runtime = time.time()-s0
print(f'total runtime {runtime/3600} hrs, estimate submission runtime {2*5*runtime/cfg.portion/3600} hrs')
