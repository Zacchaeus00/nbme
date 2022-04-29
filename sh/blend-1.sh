#!/bin/bash
#SBATCH --job-name=blend-1
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=8           
#SBATCH --mem=16GB                  
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
python -u blend.py --result_dirs \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-20:19:05-967e \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:22:20-efef \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-29-08:40:24-f2bb \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:24:05-3be3 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:24:05-45aa \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:23:06-cf4f \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-29-10:11:32-1998 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:24:05-bc37 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-28-23:23:06-f755 \
--n_trials 1000
#roberta-large
#muppet-roberta-large
#roberta-base
#bart-large
#bart-large-mnli
#deberta-large
#deberta-base
#deberta-large-mnli
#deberta-v3-large
echo "FINISH"                       