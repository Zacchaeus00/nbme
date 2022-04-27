#!/bin/bash
#SBATCH --job-name=blend-0
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
python -u blend.py --result_dirs \
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-26-20:11:47-edb5 \#muppet-roberta-large
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-26-20:11:47-22c0 \#deberta-large
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-26-20:11:47-952e \#deberta-v3-large
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-26-13:16:38-8c3f \#bart-large
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-26-17:57:25-9d71 \#bart-large-mnli
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-26-09:27:10-4c7e \#roberta-large
/gpfsnyu/scratch/yw3642/nbme/ckpt/2022-04-27-02:53:06-6b2f \#roberta-large (seed1)
echo "FINISH"                       