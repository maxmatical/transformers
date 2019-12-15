#!/bin/bash
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p nlp
python3 ./examples/run_squad_max.py \
    --model_type albert \
    --model_name_or_path albert-large-v2 \
    --do_train \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file /scratch/gobi1/mtian/SQUAD/train-v2.0.json \
    --predict_file /scratch/gobi1/mtian/SQUAD/dev-v2.0.json \
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --beta1 0.9 \
    --beta2 0.999 \
    --adam_epsilon 1e-8 \
    --lr_scheduler 'linear' \
    --max_steps 8144 \
    --save_steps 500 \
    --warmup_steps 814 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ./output/models/albert_squad_max_beta2_999_v2_lowercase_linear/ \
    --overwrite_output_dir \
    --gradient_accumulation_steps 16 \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \