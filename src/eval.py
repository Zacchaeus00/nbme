import os

import numpy as np
import pandas as pd
import torch

torch.set_grad_enabled(False)
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from data_utils import NBMEDatasetInfer
from model_utils import NBMEModel
from eval_utils import get_char_logits, get_predictions, get_score, create_labels_for_scoring, \
    my_get_results
from arguments import parse_args_eval
from utils import get_tokenizer, save_json, save_pickle

cfg = parse_args_eval()
df = pd.read_pickle(cfg.data_path)

tokenizer = get_tokenizer(cfg.pretrained_checkpoint)
model = NBMEModel(cfg.pretrained_checkpoint).cuda()
print(vars(cfg))

scores = []
oof_preds = {}
for fold in range(5):
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    val_dataset = NBMEDatasetInfer(tokenizer, val_df)
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, f'{fold}.pt')))
    model.eval()
    maxlen = max([len(x['input_ids']) for x in val_dataset])
    test_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                                 collate_fn=DataCollatorForTokenClassification(tokenizer), pin_memory=True)
    preds = []
    for b in tqdm(test_dataloader, total=len(val_dataset) // 8 + 1):
        b = {k: v.cuda() for k, v in b.items()}
        pred = model(**b).logits  # [bs, maxlen, 1]
        pred = pred.view(pred.shape[0], pred.shape[1])  # [bs, maxlen]
        pred = F.pad(input=pred, pad=(0, maxlen - pred.shape[1]), mode='constant', value=-100).cpu().numpy()
        preds.append(pred)
    preds = np.concatenate(preds, axis=0)  # [n, maxlen]
    char_logits = get_char_logits(val_df['pn_history'].values, preds, tokenizer, do_fix_offsets=cfg.do_fix_offsets)
    oof_preds.update({k: v for k, v in zip(val_df['id'], char_logits)})
    results = my_get_results(char_logits, val_df['pn_history'].values)
    preds = get_predictions(results)
    scores.append(get_score(create_labels_for_scoring(val_df), preds))
    print(f'fold {fold} score: {scores[-1]}')

print(f'cv score: {np.mean(scores)}')
save_json({'scores': scores, 'cv avg': np.mean(scores)}, os.path.join(cfg.model_dir, 'eval.json'))
save_pickle(oof_preds, os.path.join(cfg.model_dir, 'oof.pkl'))
