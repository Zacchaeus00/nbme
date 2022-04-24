import argparse


def parse_args():
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
