import argparse


def parse_args_train():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--pretrained_checkpoint', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-base')
    arg('--data_path', type=str, default='../data/train_processed.pkl')
    arg('--epochs', type=int, default=5)
    arg('--batch_size', type=int, default=16)
    arg('--accumulation_steps', type=int, default=1)
    arg('--lr', type=float, default=2e-5)
    arg('--weight_decay', type=float, default=0.01)
    arg('--seed', type=int, default=42)
    return parser.parse_args()

def parse_args_infer():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--pretrained_checkpoint', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-base')
    arg('--model_dir', type=str, required=True)
    arg('--data_dir', type=str, default='../data/nbme-score-clinical-patient-notes/')
    arg('--out_dir', type=str, default='../output/')
    return parser.parse_args()


def parse_args_eval():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--data_path', type=str, default='../data/train_processed.pkl')
    arg('--pretrained_checkpoint', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-base')
    arg('--model_dir', type=str, required=True)
    return parser.parse_args()

def parse_args_pretrain():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('--pretrained_checkpoint', type=str, default='/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-base')
    arg('--data_path', type=str, default='../data/nbme-score-clinical-patient-notes/patient_notes.csv')
    arg('--epochs', type=int, default=1)
    arg('--batch_size', type=int, default=4)
    arg('--accumulation_steps', type=int, default=1)
    arg('--lr', type=float, default=1e-5)
    arg('--weight_decay', type=float, default=0.0)
    arg('--mlm_prob', type=float, default=0.2)
    return parser.parse_args()