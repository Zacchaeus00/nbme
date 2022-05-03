#!/bin/bash
#SBATCH --job-name=blend-13
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=4
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
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:18:24-7816 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-14:53:42-e7c0 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-12:45:14-3b86 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-13:11:10-5a15 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-23:06:10-147b \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-02-00:57:31-6812 \
--n_trials 2000 \
--n_jobs 1
#4. deberta-base 8911
#9. electra-large 8917
#12. deberta-large-mnli 8927
#13. deberta-v3-large 8928
#15. deberta-xlarge
#16. deberta-v2-xlarge-mnli
echo "FINISH"                       