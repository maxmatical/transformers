#!/bin/bash
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH -p nlp
python3 ./examples/run_squad_max.py \
    --model_type albert \
    --model_name_or_path albert-xlarge-v2 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file /scratch/gobi1/mtian/SQUAD/train-v1.1.json \
    --predict_file /scratch/gobi1/mtian/SQUAD/dev-v1.1.json \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --beta1 0.9 \
    --beta2 0.98 \
    --adam_epsilon 1e-8 \
    --lr_scheduler 'cosine' \
    --num_train_epochs 3 \
    --save_steps 1000 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /scratch/gobi1/mtian/models/albert_squadv1_max_beta2_98_lr_3e-5_3epochs/ \
    --overwrite_output_dir \
    --gradient_accumulation_steps 24 \
    --per_gpu_eval_batch_size=2   \
    --per_gpu_train_batch_size=2   \
