#!/bin/bash
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p nlp
python3 ./examples/run_squad_max.py \
    --model_type albert \
    --model_name_or_path /scratch/gobi1/mtian/models/albert_squadv1_max_beta2_98_lr_5e-5/ \
    --do_train \
    --do_lower_case \
    --train_file /scratch/gobi1/mtian/BioASQ/BioASQ-train-factoid-4b.json  \
    --predict_file /scratch/gobi1/mtian/BioASQ/BioASQ-test-factoid-4b-1.json \
    --learning_rate 5e-6 \
    --weight_decay 0 \
    --beta1 0.9 \
    --beta2 0.98 \
    --adam_epsilon 1e-8 \
    --lr_scheduler 'cosine' \
    --num_train_epochs 6 \
    --save_steps 1000 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /scratch/gobi1/mtian/models/bioasq_albert_6epoch_lre5-6_sq/ \
    --overwrite_output_dir \
    --gradient_accumulation_steps 16 \
    --per_gpu_eval_batch_size=2   \
    --per_gpu_train_batch_size=2   \



