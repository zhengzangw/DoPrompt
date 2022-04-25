# Domain-Prompt

## Launch single job

Replicate resnet-50 results:

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0 \
     --algorithm ERM --output_dir vit/debug --hparams '{"vit_base_16": 0, "lr": 5e-5, "weight_decay": 1e-4, "resnet_dropout": 0.1}'
```

ViT baseline results:

- OfficeHome:

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 1 \
     --algorithm ERM --output_dir vit_16_base_result/erm_off_c_0425_01/exp --hparams '{"lr": 5e-6}'
```

- PACS

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset PACS --test_env 0 \
     --algorithm ERM --output_dir vit_16_base_result/erm_pacs_a_0425_01/exp --hparams '{"lr": 5e-6}'
```

- VLCS

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset VLCS --test_env 2 \
     --algorithm ERM --output_dir vit_16_base_result/erm_vlcs_s_0425_01/exp --hparams '{"lr": 5e-6}'
```

- DomainNet

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset DomainNet --test_env 2 \
     --algorithm ERM --output_dir vit_16_base_result/erm_dn_p_0425_01/exp --hparams '{"lr": 5e-6}'
```

## Launch multiple jobs

## Collect Results

```sh
python -m domainbed.scripts.collect_results --input_dir=vit_16_base_result/erm_off_a_1
```
