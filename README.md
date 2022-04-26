# Domain-Prompt

## Launch single job

Replicate resnet-50 results:

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0 \
     --algorithm ERM --output_dir vit/resnet_erm_off_a_01/exp --hparams '{"vit_base_16": 0, "lr": 5e-5, "weight_decay": 1e-4}'
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
     --algorithm ERM --output_dir vit_16_base_result/erm_vlcs_s_0425_01/exp --hparams '{"lr": 1e-5}'
```

- DomainNet

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset DomainNet --test_env 0 \
     --algorithm ERM --output_dir vit_16_base_result/erm_dn_c_0426_11/exp --hparams '{"lr": 5e-5}' --seed 1
```

## Launch multiple jobs

```sh
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/erm_vlcs_0426_01 \
       --algorithms ERM \
       --datasets VLCS \
       --n_trials 3 \
       --hparams '{"lr": 5e-6}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/erm_pacs_0426_01 \
       --algorithms ERM \
       --datasets PACS \
       --n_trials 3 \
       --hparams '{"lr": 5e-6}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/erm_off_0425_99 \
       --algorithms ERM \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5}'
```

```sh
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/coral_off_0426_01 \
       --algorithms CORAL \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5, "mmd_gamma": 0.1}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/dann_off_0426_01 \
       --algorithms DANN \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5, "lambda": 0.1, "mlp_width": 768, "mlp_layer": 2}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/gdro_off_0426_01 \
       --algorithms GroupDRO \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5, "groupdro_eta":0.01}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/mldg_off_0426_01 \
       --algorithms MLDG \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5, "mldg_beta": 10}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/irm_off_0426_01 \
       --algorithms IRM \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5, "irm_lambda": 0.1, "irm_penalty_anneal_iters": 500}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/mixup_off_0426_01 \
       --algorithms Mixup \
       --datasets OfficeHome \
       --n_trials 3 \
       --hparams '{"lr": 1e-5, "mixup_alpha": 0.2}'
```

## Collect Results

```sh
python -m domainbed.scripts.collect_results --input_dir=vit_16_base_result/erm_off_a_1
```
