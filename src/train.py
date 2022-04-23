import pandas as pd
from transformers import AutoTokenizer

from data_utils import NBMEDataset

fold = 0
df = pd.read_pickle('../data/train_processed.pkl')
train_df = df[df['fold']!=fold].reset_index(drop=True)
val_df = df[df['fold']==fold].reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
train_dataset = NBMEDataset(tokenizer, train_df)
val_dataset = NBMEDataset(tokenizer, val_df)
print(len(train_dataset), len(val_dataset))
print(train_dataset[0])