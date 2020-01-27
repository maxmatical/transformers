srun --mem=12G -c 2 --gres=gpu:1 -p interactive --pty bash
python3 ./examples/run_squad_max.py \
   --model_type albert \
   --model_name_or_path ./output/models/bioasq_albert_v2_lre3-5/ \
   --do_eval \
   --do_lower_case \
   --train_file /scratch/gobi1/mtian/BioASQ/BioASQ-train-factoid-4b.json  \
   --predict_file /scratch/gobi1/mtian/BioASQ/BioASQ-test-factoid-4b-1.json \
   --learning_rate 3e-5 \
   --num_train_epochs 5 \
   --save_steps 30000 \
   --max_seq_length 384 \
   --doc_stride 128 \
   --output_dir ./output/models/bioasq_albert_v2_lre3-5/4b1 \
   --overwrite_output_dir \
   --gradient_accumulation_steps 6 \
   --per_gpu_eval_batch_size=2