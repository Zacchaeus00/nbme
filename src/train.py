import os
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_ENTITY"] = "zacchaeus"
os.environ["WANDB_MODE"] = "offline"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from data_utils import NBMEDataset
from model_utils import NBMEModel
from eval_utils import get_char_logits, get_results, get_predictions, get_score, create_labels_for_scoring, compute_metrics
from arguments import parse_args
from utils import seed_everything, get_time, save_json

timenow = get_time()
cfg = parse_args()
seed_everything(cfg.seed)
df = pd.read_pickle(cfg.data_path)
tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint)

scores = []
for fold in range(5):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    name = f"{timenow}_fold{fold}"
    args = TrainingArguments(
        output_dir=f"../ckpt/{name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
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
        metric_for_best_model="pearson",
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
    results = get_results(char_logits)
    preds = get_predictions(results)
    scores.append(get_score(create_labels_for_scoring(val_df), preds))
    print(f'fold {fold} score: {scores[-1]}')

    # save ckpt
    torch.save(model.state_dict(), f"../ckpt/{timenow}/{fold}.pt")
    shutil.rmtree(f"../ckpt/{name}")

# save hyp
print(f'cv score: {np.mean(scores)}')
save_json({**vars(cfg), 'score': np.mean(scores)}, f"../ckpt/{timenow}/config.json")
