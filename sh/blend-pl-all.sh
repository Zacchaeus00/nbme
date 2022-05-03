#!/bin/bash
#SBATCH --job-name=blend-pl-all
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
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-30-13:07:35-93c8 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-30-15:15:29-c9b6 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:16:11-d3e3 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:18:24-7816 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-04:16:11-622e \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-06:09:31-d229 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-20:38:10-0673 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-14:53:42-136a \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-14:53:42-e7c0 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-15:49:03-f7f8 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-12:43:23-edad \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-12:45:14-3b86 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-13:11:10-5a15 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-18:41:00-f514 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-01-23:06:10-147b \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-05-02-00:57:31-6812 \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-30-00:28:29-93e0 \
--n_trials 5000 \
--n_jobs 1
#pl-b7ce
#1. roberta-large 8912
#2. muppet-roberta-large 8910
#pl-3a81
#3. bart-large 8893
#4. deberta-base 8911
#5. roberta-base 8872
#6. bart-large-mnli 8901
#7. deberta-v3-base 8895
#8. ernie-large 8916
#9. electra-large 8917
#10. roberta-large-mnli 8911
#11. deberta-large 8930
#12. deberta-large-mnli 8927
#13. deberta-v3-large 8928
#14. funnel-large 8896
#15. deberta-xlarge
#16. deberta-v2-xlarge-mnli
#finetune
#17. deberta-v2-xlarge-mnli
echo "FINISH"                       