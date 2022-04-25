import os

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from data_utils import NBMEDatasetInfer, preprocess_features
from model_utils import NBMEModel
from eval_utils import get_char_logits
from arguments import parse_args_infer
from utils import get_tokenizer, save_pickle
from tqdm import tqdm

cfg = parse_args_infer()
test = pd.read_csv(os.path.join(cfg.data_dir, 'test.csv'))
features = preprocess_features(pd.read_csv(os.path.join(cfg.data_dir, 'features.csv')))
patient_notes = pd.read_csv(os.path.join(cfg.data_dir, 'patient_notes.csv'))
test = test.merge(features, on=['feature_num', 'case_num'], how='left')
test = test.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

tokenizer = get_tokenizer(cfg.pretrained_checkpoint)
model = NBMEModel(cfg.pretrained_checkpoint)
test_dataset = NBMEDatasetInfer(tokenizer, test)
print(f'{len(test)} rows')
print(vars(cfg))

test_preds = []
for fold in range(5):
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, f'{fold}.pt')))
    args = TrainingArguments(
        output_dir=f".",
        do_train=False,
        group_by_length=True,
    )
    trainer = Trainer(
        model,
        args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    predictions = trainer.predict(test_dataset).predictions  # [n, maxlen, 1]
    predictions = predictions.reshape(len(test), -1)
    char_logits = get_char_logits(test['pn_history'].values, predictions, tokenizer)
    test_preds.append(char_logits)

blended_preds = {}
for i in tqdm(range(len(test))):
    thispred = np.array([test_preds[j][i] for j in range(5)])
    assert thispred.shape[1] == test.loc[i, 'pn_history'], f"index {i}: logit shape {thispred.shape}, but text len {test.loc[i, 'pn_history']}"
    blended_preds[test.loc[i, 'id']] = np.mean(thispred, axis=0)

name = cfg.model_dir.split('/')[-1]
save_pickle(blended_preds, os.path.join(cfg.out_dir, f'{name}.pkl'))
