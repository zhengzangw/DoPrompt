# Domain-Prompt

Replicate resnet-50 results

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset VLCS --test_env 0 \
     --algorithm ERM --output_dir vit/debug --hparams '{"vit_base_16": 0, "lr": 5e-5, "weight_decay": 1e-4, "resnet_dropout": 0.1}'
```

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset VLCS --test_env 0 \
     --algorithm ERM --output_dir vit/debug --hparams '{"vit_base_16": 1, "lr": 5e-5, "weight_decay": 1e-4, "resnet_dropout": 0.1}'
```
