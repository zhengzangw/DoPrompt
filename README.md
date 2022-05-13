# Domain-Prompt

## Ours

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset DomainNet --test_env 0 \
     --algorithm DoPrompt --output_dir vit_prompt_ours/dopropt_dn_0_0510_88/exp --seed 0 \ 
     --hparams '{"lr": 5e-5, "lr_classifier": 5e-3, "lr_prompt": 5e-2, "weight_decay": 1e-2, "prompt_dim": 50}' 
```

## Launch single job

Replicate resnet-50 results:

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0 \
     --algorithm ERM --output_dir vit_16_base_resultresnet_erm_off_a_01/exp --hparams '{"vit_base_16": 0, "lr": 5e-5, "weight_decay": 1e-4}'
```

ViT baseline results:

- OfficeHome:

```sh
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 1 \
     --algorithm ERM --output_dir vit_16_base_result/erm_off_c_0505_01/exp --hparams '{"lr": 1e-5, "attention_dropout": 0.5}'

python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset OfficeHome --test_env 0 \
     --algorithm Prompt --output_dir vit_prompt/prompt_off_0_0506_01 --hparams '{"lr": 1e-5, "lr_classifier": 1e-4, "lr_prompt": 1e-3}'
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
python -m domainbed.scripts.train --data_dir=./domainbed/data/ --steps 5001 --dataset DomainNet --test_env 4 \
     --algorithm ERM --output_dir vit_16_base_result/erm_dn_r_0427_03/exp --hparams '{"lr": 5e-5}' --seed 2
```

## Launch multiple jobs

```sh
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/erm_vlcs_0503_91 \
       --algorithms ERM SMA MMD CORAL Mixup DANN CDANN GroupDRO MLDG IRM \
       --datasets VLCS \
       --n_trials 2 \
       --hparams '{"lr": 5e-6, "lr_classifier": 5e-5, "mixup_alpha": 0.2, "mmd_gamma": 0.1, "groupdro_eta": 0.01, "mldg_beta": 1, "lambda": 0.1, "mlp_width": 768, "mlp_layer": 1, "irm_lambda": 0.1, "irm_penalty_anneal_iters": 500}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/prompt_vlcs_0505_99 \
       --algorithms Prompt \
       --datasets VLCS \
       --n_trials 1 \
       --hparams '{"lr": 5e-6, "lr_classifier": 5e-5}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_16_base_result/rsc_off_0505_99 \
       --algorithms RSC \
       --datasets OfficeHome \
       --n_trials 1 \
       --hparams '{"lr": 1e-5, "lr_classifier": 1e-4}'


python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_prompt_ours/ours_off_0509_01 \
       --algorithms MyPrompt2 \
       --datasets OfficeHome \
       --n_trials 2 \
       --hparams '{"lr": 1e-5, "lr_classifier": 1e-3, "lr_prompt": 1e-2, "weight_decay": 1e-2, "prompt_dim": 50}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_prompt_ours/ours_vlcs_0509_01 \
       --algorithms MyPrompt2 \
       --datasets VLCS \
       --n_trials 2 \
       --hparams '{"lr": 1e-5, "lr_classifier": 1e-3, "lr_prompt": 1e-2, "weight_decay": 1e-2, "prompt_dim": 50}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_prompt_ours/ours_pacs_0509_01 \
       --algorithms MyPrompt2 \
       --datasets PACS \
       --n_trials 2 \
       --hparams '{"lr": 1e-5, "lr_classifier": 1e-3, "lr_prompt": 1e-2, "weight_decay": 1e-2, "prompt_dim": 50}'

python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_prompt_ours/ours_dn_0509_01 \
       --algorithms MyPrompt2 \
       --datasets DomainNet \
       --n_trials 1 \
       --hparams '{"lr": 5e-5, "lr_classifier": 5e-3, "lr_prompt": 5e-2, "weight_decay": 1e-2, "prompt_dim": 50}'
python -m domainbed.scripts.sweep launch --data_dir=./domainbed/data/ --command_launcher local \
       --steps 5001 --single_test_envs --n_hparams 1 \
       --output_dir=vit_prompt_ours/ours_dn_0509_02 \
       --algorithms MyPrompt2 \
       --datasets DomainNet \
       --n_trials 1 \
       --hparams '{"lr": 5e-5, "lr_classifier": 5e-4, "lr_prompt": 5e-3, "weight_decay": 1e-2, "prompt_dim": 50}'
```

## Collect Results

```sh
python -m domainbed.scripts.collect_results --input_dir=vit_16_base_result/erm_off_a_1
```
