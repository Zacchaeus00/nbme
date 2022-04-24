import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

from data_utils import NBMEDataset
from arguments import parse_args
from utils import seed_everything

cfg = parse_args()
fold = 0
df = pd.read_pickle(cfg.data_path)
train_df = df[df['fold'] != fold].reset_index(drop=True)
val_df = df[df['fold'] == fold].reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_checkpoint)
seed_everything(cfg.seed)
args = TrainingArguments(
        output_dir=f"./ckpt",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        fp16=True,
        report_to='none',
        dataloader_num_workers=4,
        group_by_length=True
)
trainer = Trainer(
        AutoModelForTokenClassification.from_pretrained(cfg.pretrained_checkpoint, num_labels=2),
        args,
        train_dataset=NBMEDataset(tokenizer, train_df),
        eval_dataset=NBMEDataset(tokenizer, val_df),
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
)
trainer.train()