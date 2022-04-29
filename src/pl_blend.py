import pickle
import json
import os
import pandas as pd
import numpy as np
import gc
from pprint import pprint
from arguments import parse_args_pl_blend
from eval_utils import get_spans
from utils import save_json
from tqdm import tqdm

cfg = parse_args_pl_blend()
print(vars(cfg))
uid = cfg.blend_log[-9:-5]
print('uid:', uid)
blend_log = json.load(open(cfg.blend_log, 'r'))
pprint(blend_log)
pl_df = pd.read_pickle(cfg.data_path)

char_logits_blend = [np.zeros(len(text)) for text in pl_df.pn_history.values]
for model_dir, w in tqdm(blend_log['weights'].items()):
    char_logits = pickle.load(open(os.path.join(model_dir, 'pl_logits.pkl'), 'rb'))
    df = pl_df.merge(pd.DataFrame({'id': list(char_logits.keys()), 'char_logits': list(char_logits.values())}),
                     on='id',
                     how='left')
    char_logits = df.char_logits.values
    char_logits = [w * char_logits_ for char_logits_ in char_logits]
    for i in range(len(pl_df)):
        char_logits_blend[i] += char_logits[i]
del df, char_logits
gc.collect()

all_spans = get_spans(char_logits_blend, pl_df.pn_history.values, th=1e-6)
locations = []
for spans in all_spans:
    locs = []
    for s, e in spans:
        locs.append(f'{s} {e}')
    locations.append(locs)
pl_df['location'] = locations

annotations = []
for i, spans in enumerate(all_spans):
    annos = []
    for s, e in spans:
        annos.append(pl_df.loc[i, 'pn_history'][s:e])
    annotations.append(annos)
pl_df['annotation'] = annotations
pl_df['annotation_length'] = pl_df['annotation'].apply(len)
pl_df.to_pickle(os.path.join(cfg.out_dir, f'train_pl_{uid}.pkl'))
save_json(blend_log, os.path.join(cfg.out_dir, f'blend-{uid}.json'))
