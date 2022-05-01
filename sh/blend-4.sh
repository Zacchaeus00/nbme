#!/bin/bash
#SBATCH --job-name=blend-4
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32
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
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-30-13:07:35-93c8 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-30-15:15:29-c9b6 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:16:11-d3e3 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:18:24-7816 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:16:11-622e \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-06:09:31-d229 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-20:38:10-0673 \
--n_trials 1000 \
--n_jobs 32
#pl-b7ce
#1. roberta-large
#2. muppet-roberta-large
#pl-3a81
#3. bart-large
#4. deberta-base
#5. roberta-base
#6. bart-large-mnli
#7. deberta-v3-base
echo "FINISH"                       