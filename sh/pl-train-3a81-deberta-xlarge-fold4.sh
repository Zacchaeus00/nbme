#!/bin/bash
#SBATCH --job-name=pl-train-3a81-deberta-xlarge-fold4
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=8           
#SBATCH --mem=16GB
#SBATCH --time=167:00:00
#SBATCH --mail-type=END             
#SBATCH --mail-user=yw3642@nyu.edu  
#SBATCH --output=log/%x-%A.out
#SBATCH --error=log/%x-%A.err
#SBATCH --gres=gpu:1                
#SBATCH -p aquila                   
#SBATCH --nodelist=agpu7

module purge                        
module load anaconda3 cuda/11.1.1              

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/nbme/src     

echo "START"               
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate kaggle          
python -u pl_train.py \
--pretrained_checkpoint /gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-02:28:41-4c32/checkpoint-84292 \
--pl_path ../data/train_pl_3a81.pkl \
--epochs 1 \
--batch_size 2 \
--accumulation_steps 1 \
--lr 5e-6 \
--weight_decay 0.0 \
--seed 116 \
--fold 4
echo "FINISH"                       