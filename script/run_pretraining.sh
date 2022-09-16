#!/bin/bash
set -x

cd ..

D=data/raw
pretrain_data_dir=pretrain_tfrecords
model_name=base_model

horovodrun -np 4 -H localhost:4 \
      python3 run_pretraining_hvd.py  \
      --data-dir $D  \
      --model-name ${model_name} \
      --hparams "{\"model_size\":\"base\", \"pretrain_data_dir\": \"${pretrain_data_dir}\", \"vocab_size\": 21128, \"train_batch_size\": 16, \"num_train_steps\": 400000000, \"learning_rate\":2e-4, \"eval_batch_size\":16,\"model_hparam_overrides\": {\"hidden_size\": 768, \"num_hidden_layers\": 12, \"num_attention_heads\": 12}}"

