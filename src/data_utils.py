import torch
from torch.utils.data import Dataset
import numpy as np


def prepare_input(tokenizer, text, feature_text):
    inputs = tokenizer(text, feature_text,
                       add_special_tokens=True,
                       return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def create_label(tokenizer, text, annotation_length, location_list):
    encoded = tokenizer(text,
                        add_special_tokens=True,
                        return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']
    ignore_idxes = np.where(np.array(encoded.sequence_ids()) != 0)[0]
    label = np.zeros(len(offset_mapping))
    label[ignore_idxes] = -100
    if annotation_length != 0:
        for location in location_list:
            for loc in [s.split() for s in location.split(';')]:
                start_idx = -1
                end_idx = -1
                start, end = int(loc[0]), int(loc[1])
                for idx in range(len(offset_mapping)):
                    if (start_idx == -1) & (start < offset_mapping[idx][0]):
                        start_idx = idx - 1
                    if (end_idx == -1) & (end <= offset_mapping[idx][1]):
                        end_idx = idx + 1
                if start_idx == -1:
                    start_idx = end_idx
                if (start_idx != -1) & (end_idx != -1):
                    label[start_idx:end_idx] = 1
    return torch.tensor(label, dtype=torch.float)


class NBMEDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.tokenizer,
                               self.pn_historys[item],
                               self.feature_texts[item])
        label = create_label(self.tokenizer,
                             self.pn_historys[item],
                             self.annotation_lengths[item],
                             self.locations[item])
        return {**inputs, 'label': label}

if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    df = pd.read_pickle('../data/train_processed.pkl')
    dataset = NBMEDataset(tokenizer, df)
    # print(dataset[0])
    print(tokenizer.pad([dataset[0], dataset[1]], padding=True, return_tensors='pt'))
    # print(tokenizer.decode(dataset[0]['input_ids']))
    # lengs = []
    # for d in dataset:
    #     lengs.append(len(d['input_ids']))
    # print(np.mean(lengs), np.max(lengs), np.min(lengs)) # 229.59685314685314 417 53
    print(list(dataset[0].keys()))