import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_ENTITY"] = "zacchaeus"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "nbme"

import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from pathlib import Path

from data_utils import LineByLineTextDataset
from arguments import parse_args_pretrain
from utils import get_time, get_tokenizer, seed_everything

timenow = get_time()
cfg = parse_args_pretrain()
seed_everything(cfg.seed)
df = pd.read_csv(cfg.data_path)
tokenizer = get_tokenizer(cfg.pretrained_checkpoint)
dataset = LineByLineTextDataset(tokenizer, df['pn_history'].tolist(), 512)
print(timenow)
print(vars(cfg))
print(f'{len(dataset)} rows')
print('sample data:', dataset[0])

args = TrainingArguments(
    output_dir=f"../ckpt/{timenow}",
    save_strategy="epoch",
    learning_rate=cfg.lr,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.epochs,
    weight_decay=cfg.weight_decay,
    warmup_ratio=0.2,
    fp16=True,
    report_to='wandb',
    dataloader_num_workers=4,
    group_by_length=True,
    run_name=timenow,
    save_total_limit=2,
    seed=cfg.seed,
)
model = AutoModelForMaskedLM.from_pretrained(cfg.pretrained_checkpoint)
trainer = Trainer(
    model,
    args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_prob),
)
trainer.train()
