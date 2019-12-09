#!/bin/bash -l
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p nlp
conda activate /scratch/gobi1/mtian/my_project
python3 <<EOF
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
EOF
