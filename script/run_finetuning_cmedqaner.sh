#!/bin/bash
set -x
# The training and test data are placed in the data/finetune/finetune_data directory.
# Converting json data to BIO annotation format
# train.txt dev.txt eval.txt
cd ..
data_dir=data/finetune
model_name=base_model
task_name=cmedqaner
num_trials=1
learning_rate=3e-4
lable_id_path=$data_dir/models/$model_name/finetuning_tfrecords/${task_name}_tfrecords/${task_name}_label_mapping.pkl

function finetune()
{
export CUDA_VISIBLE_DEVICES=3
python run_finetuning.py \
    --data-dir ${data_dir} \
    --model-name ${model_name} \
    --hparams "{\"num_trials\": "${num_trials}", \"task_names\": [\"${task_name}\"], \"do_train\": true, \"do_eval\":true, \"do_test\":true, \"model_size\": \"base\", \"vocab_size\": 21128, \"save_checkpoints_steps\": 1000, \"max_seq_length\": 128, \"write_test_outputs\":true, \"num_train_epochs\": 10, \"learning_rate\":${learning_rate}}"
}

function calculate_pr()
{
  finetune_model_dir=${data_dir}/models/${model_name}/finetuning_models
  for i in `seq 1 ${num_trials}`;
  do
  input_path=${finetune_model_dir}/${task_name}_eval_${i}
  tmp=${finetune_model_dir}/${task_name}_eval_tmp
  python convert_tagid_to_tagname.py $lable_id_path  $input_path  $tmp
  perl conlleval.pl -d '\t' < $tmp
  python  calculate_pr.py   $tmp ${tmp}_${i}_pr  B-body,B-crowd,B-department,B-disease,B-drug,B-feature,B-physiology,B-symptom,B-test,B-time,B-treatment,I-body,I-crowd,I-department,I-disease,I-drug,I-feature,I-physiology,I-symptom,I-test,I-time,I-treatment
  done
}

finetune
calculate_pr

