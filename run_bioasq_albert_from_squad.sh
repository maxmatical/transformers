#!/bin/bash
#SBATCH --mem=12G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p nlp
python3 ./examples/run_squad_max.py \
    --model_type albert \
    --model_name_or_path ./output/models/albert_squad_max_beta2_98/ \
    --do_train \
    --train_file /scratch/gobi1/mtian/BioASQ/BioASQ-train-factoid-4b.json  \
    --predict_file /scratch/gobi1/mtian/BioASQ/BioASQ-test-factoid-4b-1.json \
    --fp16
    --learning_rate 3e-5 \
    --weight_decay 0 \
    --beta1 0.9 \
    --beta2 0.98 \
    --adam_epsilon 1e-4 \
    --lr_scheduler 'cosine' \
    --num_train_epochs 2 \
    --max_steps 5000 \
    --save_steps 500 \
    --warmup_steps 500 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir ./output/models/albert_bioasq_max_beta2_98_from_squad/ \
    --overwrite_output_dir \
    --gradient_accumulation_steps 16 \
    --per_gpu_eval_batch_size=3   \
    --per_gpu_train_batch_size=3   \