# 1
srun --mem=12G -c 2 --gres=gpu:1 -p interactive --pty bash

#2 
# models can be bioasq_albertx or biosaq_bert models
# change model type, model path, and output_dir


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
   --output_dir ./output/models/bioasq_albertx_v2_lre3-5/ \
   --overwrite_output_dir \
   --gradient_accumulation_steps 6 \
   --per_gpu_eval_batch_size=2


#3
# change nbest_path here
python3 transform_nbset2bioasqform.py --nbest_path=./output/models/bioasq_albert_v2_lre3-5/nbest_predictions_.json --output_path=./

# 4
# /h/mtian/Evaluation-Measures
# move output to /scratch/gobi1/mtian/
# /scratch/gobi1/mtian/BioASQform_BioASQ-answer.json (path to file)
# /scratch/gobi1/mtian/BioASQ (path to bioasq)


# path to transformers for BioASQ-golden
# use output_path in step 3 for BioASQform_BioASQ-answer.json
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    /h/mtian/transformers/BioASQ-golden/4B1_golden.json \
    ./BioASQform_BioASQ-answer.json