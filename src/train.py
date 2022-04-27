import os
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_ENTITY"] = "zacchaeus"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "nbme"

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from pathlib import Path

from data_utils import NBMEDataset
from model_utils import NBMEModel
from eval_utils import get_char_logits, get_results, get_predictions, get_score, create_labels_for_scoring, compute_metrics
from arguments import parse_args_train
from utils import seed_everything, get_time, save_json, get_tokenizer, save_pickle

timenow = get_time()
cfg = parse_args_train()
seed_everything(cfg.seed)
df = pd.read_pickle(cfg.data_path)
tokenizer = get_tokenizer(cfg.pretrained_checkpoint)
print(timenow)
print(vars(cfg))

scores = []
oof_preds = {}
for fold in range(5):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    name = f"{timenow}_fold{fold}"
    args = TrainingArguments(
        output_dir=f"../ckpt/{name}",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        warmup_ratio=0.2,
        fp16=True,
        report_to='wandb',
        dataloader_num_workers=4,
        group_by_length=True,
        run_name=name,
        metric_for_best_model="nbme_f1",
        save_total_limit=2,
        label_names=['label'],
        seed=cfg.seed,
    )
    model = NBMEModel(cfg.pretrained_checkpoint)
    trainer = Trainer(
        model,
        args,
        train_dataset=NBMEDataset(tokenizer, train_df),
        eval_dataset=NBMEDataset(tokenizer, val_df),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics
    )
    trainer.train()
    predictions = trainer.predict(NBMEDataset(tokenizer, val_df)).predictions  # [n, maxlen, 1]
    predictions = predictions.reshape(len(val_df), -1)
    char_logits = get_char_logits(val_df['pn_history'].values, predictions, tokenizer)
    oof_preds.update({k: v for k, v in zip(val_df['id'], char_logits)})
    results = get_results(char_logits)
    preds = get_predictions(results)
    scores.append(get_score(create_labels_for_scoring(val_df), preds))
    print(f'fold {fold} score: {scores[-1]}')

    # save ckpt
    Path(f"../ckpt/{timenow}/").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"../ckpt/{timenow}/{fold}.pt")
    shutil.rmtree(f"../ckpt/{name}")

# save hyp
print(f'cv score: {np.mean(scores)}')
save_json({**vars(cfg), 'score': np.mean(scores)}, f"../ckpt/{timenow}/config.json")
save_pickle(oof_preds, f"../ckpt/{timenow}/oof.pkl")
