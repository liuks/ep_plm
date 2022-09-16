## Introduction
This repository contains the code and resources for COLING 2022 paper *"A Domain Knowledge Enhanced Pre-Trained Language Model for Vertical Search: Case Study on Medicinal Products"*.

```
@inproceedings{liuks2022,
  title={A Domain Knowledge Enhanced Pre-Trained Language Model for Vertical Search: Case Study on Medicinal Products},
  author={Kesong Liu, Jianhui Jiang and Feifei Lyu},
  booktitle={COLING},
  year={2022}
}
```

Our code is developed based on [ELECTRA](https://github.com/google-research/electra). Following ELECTRA's replaced token detection (RTD) pre-training, we leverage biomedical entity masking (EM) strategy to learn better contextual word representations. Furthermore, we propose a novel pre-training task, *product attribute prediction* (PAP), to inject product knowledge into the pre-trained language model efficiently by leveraging medicinal product databases directly.


## Usage
- Pre-training: Please refer to the files "build_pretraining_dataset.py" and "run_pretraining.py" to build training samples and perform multi-task pre-training.
- Training data: Please refer to the "data" directory for *SAMPLE* data.
- Fine-tuning and evaluation: Please refer to the files in the "script" directory.
