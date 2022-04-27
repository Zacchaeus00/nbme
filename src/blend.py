import pickle as pk
import pandas as pd
import optuna
import numpy as np
import os
from utils import get_uid, save_json

from eval_utils import get_spans, get_score, create_labels_for_scoring
from arguments import parse_args_blend

cfg = parse_args_blend()
for dir_ in cfg.result_dirs:
    assert os.path.isdir(dir_), f'{dir_} not exists'

train = pd.read_pickle(cfg.data_path)
train_labels = create_labels_for_scoring(train)
log = {'individual': {}, 'blend': 0, 'weights': {}}

for i, res_dir in enumerate(cfg.result_dirs):
    print(res_dir)
    char_logits = pk.load(open(os.path.join(res_dir, 'oof.pkl'), 'rb'))
    ids = list(char_logits.keys())
    spans = get_spans(list(char_logits.values()))
    df = pd.DataFrame({'id': ids, f'prediction{i}': spans, f'char_logits{i}': char_logits.values()})
    train = train.merge(df, on='id', how='left')
    score = get_score(train_labels, train[f'prediction{i}'].values)
    log['individual'][res_dir] = score
    print(score)
    print()


def objective(trial):
    n = len(cfg.result_dirs)
    weights = np.ones(n)
    for i in range(n):
        weights[i] = trial.suggest_float(f'w{i}', 0, 1)
    char_logits_blend = []
    for i in range(len(train)):
        logits = []
        for j in range(len(cfg.result_dirs)):
            logits.append(train.loc[i, f'char_logits{j}'])
        char_logits_blend.append(np.average(logits, weights=weights, axis=0))
    spans = get_spans(char_logits_blend)
    score = get_score(train_labels, spans)
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=cfg.n_trials)
print('best blend weights:', study.best_params)
print('best blend score:', study.best_trial.values[0])

log['blend'] = study.best_trial.values[0]
for i, res_dir in enumerate(cfg.result_dirs):
    log['weights'][res_dir] = study.best_params[f'w{i}']
name = f'blend-{get_uid()}.json'
for res_dir in cfg.result_dirs:
    save_json(log, os.path.join(res_dir, name))
