#!/bin/bash
set -x
cd ..
data_dir=data/finetune
model_name=base_model
task_name=cehrner
num_trials=1
lable_id_path=$data_dir/models/$model_name/finetuning_tfrecords/${task_name}_tfrecords/${task_name}_label_mapping.pkl
learning_rate=3e-4

function finetune()
{
export CUDA_VISIBLE_DEVICES=0
python run_finetuning.py \
    --data-dir ${data_dir} \
    --model-name ${model_name} \
    --hparams "{\"num_trials\": "${num_trials}", \"task_names\": [\"${task_name}\"], \"do_train\": true, \"do_eval\":true, \"do_test\":true, \"model_size\": \"base\", \"vocab_size\": 21128, \"save_checkpoints_steps\": 1000, \"max_seq_length\": 128, \"write_test_outputs\":true, \"learning_rate\":${learning_rate}}"
}

# calculate precision，recall and F1-score
function calculate_pr()
{
  finetune_model_dir=${data_dir}/models/${model_name}/finetuning_models
  for i in `seq 1 ${num_trials}`;
  do
  	input_path=${finetune_model_dir}/${task_name}_eval_${i}
  	tmp=${finetune_model_dir}/${task_name}_eval_tmp
  	python convert_tagid_to_tagname.py $lable_id_path  $input_path  $tmp
  	perl conlleval.pl -d '\t' < $tmp
  	python  calculate_pr.py   $tmp ${tmp}_${i}_pr B-实验室检验,B-影像检查,B-手术,B-疾病和诊断,B-症状,B-药物,B-解剖部位,I-实验室检验,I-影像检查,I-手术,I-疾病和诊断,I-症状,I-药物,I-解剖部位
  done
}

finetune
calculate_pr


