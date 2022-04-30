#!/bin/bash
#SBATCH --job-name=pl-blend-1
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32GB
#SBATCH --time=167:00:00
#SBATCH --mail-type=END             
#SBATCH --mail-user=yw3642@nyu.edu  
#SBATCH --output=log/%x-%A.out
#SBATCH --error=log/%x-%A.err
#SBATCH --constraint=cpu # use this if you want to only use cpu
#SBATCH -p aquila                   

module purge                        
module load anaconda3 cuda/11.1.1              

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/nbme/src     

echo "START"               
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate kaggle          
python -u pl_blend.py --blend_log /gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-30-11:54:28-624d/blend-3a81.json
#1. roberta-large
#2. muppet-roberta-large
#3. roberta-base
#4. bart-large
#5. bart-large-mnli
#6. deberta-large
#7. deberta-base
#8. deberta-large-mnli
#9. deberta-v3-large
#10. ernie-large
#11. electra-large-discriminator
#12. funnel-large
#13. deberta-v3-base
echo "FINISH"                       