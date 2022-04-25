import os

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from data_utils import NBMEDatasetInfer
from model_utils import NBMEModel
from eval_utils import get_char_logits, get_results, get_predictions, get_score, create_labels_for_scoring
from arguments import parse_args_eval
from utils import get_tokenizer, save_json

cfg = parse_args_eval()
df = pd.read_pickle(cfg.data_path)

tokenizer = get_tokenizer(cfg.pretrained_checkpoint)
model = NBMEModel(cfg.pretrained_checkpoint)
print(vars(cfg))

scores = []
for fold in range(5):
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    val_dataset = NBMEDatasetInfer(tokenizer, val_df)
    model.load_state_dict(torch.load(os.path.join(cfg.model_dir, f'{fold}.pt')))
    args = TrainingArguments("./")
    trainer = Trainer(
        model,
        args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    predictions = trainer.predict(val_dataset).predictions  # [n, maxlen, 1]
    predictions = predictions.reshape(len(val_df), -1)
    char_logits = get_char_logits(val_df['pn_history'].values, predictions, tokenizer)
    results = get_results(char_logits)
    preds = get_predictions(results)
    scores.append(get_score(create_labels_for_scoring(val_df), preds))
    print(f'fold {fold} score: {scores[-1]}')


print(f'cv score: {np.mean(scores)}')
save_json({'scores': scores, 'cv avg': np.mean(scores)}, os.path.join(cfg.model_dir, 'eval.json'))
