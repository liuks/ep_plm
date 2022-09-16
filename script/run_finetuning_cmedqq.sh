#!/bin/bash
set -x
# The training and test data are placed in the data/finetune/finetune_data directory
# train.txt dev.txt eval.txt
cd ..
data_dir=data/finetune
model_name=base_model
task_name=cmedqq
learning_rate=3e-4
num_trials=1

function finetune()
{
export CUDA_VISIBLE_DEVICES=0
python run_finetuning.py \
    --data-dir ${data_dir} \
    --model-name ${model_name} \
    --hparams "{\"num_trials\": "${num_trials}", \"task_names\": [\"${task_name}\"], \"max_seq_length\":128, \"model_size\": \"base\", \"do_train\": true, \"do_eval\": true, \"do_test\": true, \"num_train_epochs\": 3, \"learning_rate\": ${learning_rate}, \"vocab_size\": 21128, \"train_batch_size\": 64, \"eval_batch_size\": 16, \"predict_batch_size\": 16, \"save_checkpoints_steps\": 1000, \"write_test_outputs\":true }"
}

function calculate_pr()
{
  finetune_model_dir=${data_dir}/models/${model_name}/finetuning_models
  for i in `seq 1 ${num_trials}`;
  do
  	input_path=${finetune_model_dir}/${task_name}_eval_${i}
  	output_path=${finetune_model_dir}/${task_name}_eval_${i}_pr
  	python calculate_pr.py $input_path $output_path 1
  done
}

finetune
calculate_pr
