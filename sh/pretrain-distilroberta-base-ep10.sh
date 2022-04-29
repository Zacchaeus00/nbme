#!/bin/bash
#SBATCH --job-name=pretrain-distilroberta-base-ep10
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
python -u pretrain.py --pretrained_checkpoint /gpfsnyu/scratch/yw3642/hf-models/distilroberta-base --epochs 10 \
--batch_size 16 --accumulation_steps 1 --lr 3e-5 --weight_decay 0.0 --mlm_prob 0.2 --seed 6078
echo "FINISH"                       