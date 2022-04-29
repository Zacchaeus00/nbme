#!/bin/bash
#SBATCH --job-name=pl-infer-2
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
python -u pl_infer.py \
--blend_log /gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:22:20-efef/blend-0eeb.json \
--pretrained_checkpoints \
/gpfsnyu/scratch/yw3642/hf-models/roberta-large \
/gpfsnyu/scratch/yw3642/hf-models/facebook_muppet-roberta-large \
--model_dirs \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-20:19:05-967e \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:22:20-efef
echo "FINISH"                       