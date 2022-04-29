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
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-29-12:27:21-20e9 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-29-10:16:54-b5c1 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-29-10:26:43-39ad \
--n_trials 1000
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
echo "FINISH"                       