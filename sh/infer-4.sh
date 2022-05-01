#!/bin/bash
#SBATCH --job-name=infer-4
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB                  
#SBATCH --time=167:00:00
#SBATCH --mail-type=END             
#SBATCH --mail-user=yw3642@nyu.edu  
#SBATCH --output=log/%x-%A.out
#SBATCH --error=log/%x-%A.err
#SBATCH --gres=gpu:1                
#SBATCH -p aquila                   
#SBATCH --nodelist=gpu6

module purge                        
module load anaconda3 cuda/11.1.1              

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/nbme/src     

echo "START"               
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate kaggle          
python -u infer.py \
--pretrained_checkpoints \
/gpfsnyu/scratch/yw3642/hf-models/roberta-large \
/gpfsnyu/scratch/yw3642/hf-models/facebook_muppet-roberta-large \
/gpfsnyu/scratch/yw3642/hf-models/facebook_bart-large \
/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-base \
/gpfsnyu/scratch/yw3642/hf-models/roberta-base \
/gpfsnyu/scratch/yw3642/hf-models/facebook_bart-large-mnli \
/gpfsnyu/scratch/yw3642/hf-models/microsoft_deberta-v3-base \
/gpfsnyu/scratch/yw3642/hf-models/nghuyong_ernie-2.0-large-en \
/gpfsnyu/scratch/yw3642/hf-models/google_electra-large-discriminator
echo "FINISH"                       