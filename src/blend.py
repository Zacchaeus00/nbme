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
    df = pd.DataFrame({'id': ids, f'char_logits{i}': char_logits.values()})
    train = train.merge(df, on='id', how='left')
    spans = get_spans(train[f'char_logits{i}'].tolist(), train['pn_history'].tolist())
    df = pd.DataFrame({'id': train['id'].tolist(), f'prediction{i}': spans})
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
    spans = get_spans(char_logits_blend, train['pn_history'].tolist())
    score = get_score(train_labels, spans)
    return score


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)
print('best blend weights:', study.best_params)
print('best blend score:', study.best_trial.values[0])

log['blend'] = study.best_trial.values[0]
for i, res_dir in enumerate(cfg.result_dirs):
    log['weights'][res_dir] = study.best_params[f'w{i}']
uid = get_uid()
print('uid:', uid)
for res_dir in cfg.result_dirs:
    log['self'] = res_dir
    save_json(log, os.path.join(res_dir, f'blend-{uid}.json'))
save_json(log, os.path.join(cfg.out_dir, f'blend-{uid}.json'))

n = len(cfg.result_dirs)
weights = np.ones(n)
for i in range(n):
    weights[i] = study.best_params[f'w{i}']
char_logits_blend = []
for i in range(len(train)):
    logits = []
    for j in range(len(cfg.result_dirs)):
        logits.append(train.loc[i, f'char_logits{j}'])
    char_logits_blend.append(np.average(logits, weights=weights, axis=0))
spans = get_spans(char_logits_blend, train['pn_history'].tolist())
train['pred_spans'] = spans
annotations = []
for i, spans_ in enumerate(spans):
    text = train.loc[i, 'pn_history']
    annos = []
    for s, e in spans_:
        annos.append(text[s:e])
    annotations.append(annos)
train['pred_annotation'] = annotations
train.drop('location_for_create_labels', axis=1, inplace=True)
for i, _ in enumerate(cfg.result_dirs):
    train.drop(f'char_logits{i}', axis=1, inplace=True)
    train.drop(f'prediction{i}', axis=1, inplace=True)
train.to_pickle(os.path.join(cfg.out_dir, f'blend-{uid}-oof.pkl'))
