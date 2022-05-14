# DoPrompt

PyTorch implementation of Domain Prompt for Domain Generalization (DoPrompt). This implementation is based on the DomainBed.

## Overview

Architecture of Network:

TBD

## Training

Refer to [DomainBed Readme](README_domainbed.md) for more details on commands running jobs. The training setting sweeps across multiple hyperparameters. Here we select some hyperparameters that can reach a good result.

```sh
# OfficeHome
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0 --algorithm DoPrompt --output_dir results/exp \
     --hparams '{"lr": 1e-5, "lr_classifier": 1e-3, "prompt_dim": 4, "lambda": 1.0}'
```

## Collect Results

```sh
python -m domainbed.scripts.collect_results --input_dir=results
```

## Requirements

```sh
pip install -r requirements.txt
```

## Citation

TBD
