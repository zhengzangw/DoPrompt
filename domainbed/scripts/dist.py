import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import combinations
import math

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.torchmisc import dataloader
from domainbed.algorithms import PrependPrompt


@torch.no_grad()
def cal_feat_mean(network, loader, weights, device, domain=None, algorithm=None):
    network.featurizer.network.encoder.layers = network.featurizer.network.encoder.layers[:1]
    
    feat_cls_mean = torch.zeros(network.num_classes, network.featurizer.n_outputs).cuda()
    feat_cls_cnt = torch.zeros(network.num_classes).cuda()
    feat_mean = torch.zeros(network.featurizer.n_outputs).cuda()
    cnt = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        if algorithm == "DomainPrompt":
            if domain is not None:
                domain_prompts = network.x_domain_prompt(x, domain)
                domain_prompts = network.x_domain_prompt(x, 2)
                with PrependPrompt(network.featurizer, domain_prompts):
                    feat = network.featurizer(x)
            else:
                domain_prompts = []
                for i in range(network.num_domains):
                    domain_prompt = network.x_domain_prompt(x, i)
                    domain_prompts.append(domain_prompt)
                domain_prompts = torch.stack(domain_prompts, dim=1).mean(dim=1)
                domain_prompts = network.x_domain_prompt(x, 2)
                with PrependPrompt(network.featurizer, domain_prompts):
                    feat = network.featurizer(x)
        elif algorithm in ["Prompt"]:
            domain_prompts = network.prompt_tokens.repeat(len(x), 1, 1)
            with PrependPrompt(network.featurizer, domain_prompts):
                feat = network.featurizer(x)
        elif algorithm == "PromptDeep":
            domain_prompts = [t.repeat(len(x), 1, 1) for t in network.deep_prompt_tokens]
            with PrependPromptDeep(network.featurizer, domain_prompts):
                feat = network.featurizer(x)
        elif algorithm == "DoPrompt":
            all_z, _ = network.forward_first(x)
            hint = all_z.detach()
            all_bias = network.project(hint).reshape(-1, network.num_domains, network.prompt_dim)
            all_bias = F.softmax(all_bias, dim=1)
            domain_prompts = network.x_domain_prompt_comb(all_bias)
            with PrependPrompt(network.featurizer, domain_prompts):
                feat = network.featurizer(x)
        else:
            feat = network.featurizer(x)
        
        # update feat_mean
        feat_mean += feat.sum(dim=0)
        cnt += feat.shape[0]
        # update feat_cls_mean
        for i in range(feat.shape[0]):
            feat_cls_mean[y[i]] += feat[i]
            feat_cls_cnt[y[i]] += 1
    if cnt > 0:
        feat_mean /= cnt
    for i in range(network.num_classes):
        if feat_cls_cnt[i] > 0:
            feat_cls_mean[i] /= feat_cls_cnt[i]
    return feat_mean, feat_cls_mean


@torch.no_grad()
def cosine_dist(x1, x2):
    x1 = x1.view(1, -1)
    x2 = x2.view(1, -1)
    d = 1 - F.cosine_similarity(x1, x2)
    return d.item()


@torch.no_grad()
def domain_class_insight(feat_means, feat_cls_means):
    in_name = [x for x in feat_means.keys() if 'in' in x]
    out_name = [x for x in feat_means.keys() if 'out' in x]
    stats = {'cross_domain': {}, 'in_domain': {}, 'in_domain_in_cls': {}, 'in_domain_cross_cls': {}, 'cross_domain_in_cls': {}}
    # distance of a same domain
    print("=== Section 1: in-domain distance ===")
    for i, o in zip(in_name, out_name):
        d = cosine_dist(feat_means[i], feat_means[o])
        stats['in_domain'][i] = d
        print(f"dist({i}, {o}) = {d}")
    # distance between different domains
    print("=== Section 2: cross-domain distance ===")
    for d1, d2 in combinations(in_name, 2):
        d = cosine_dist(feat_means[d1], feat_means[d2])
        stats['cross_domain'][(d1, d2)] = d
        print(f"dist({d1}, {d2}) = {d}")
    # average distance within classes
    print("=== Section 3: in-domain in-class distance ===")
    for d1_in, d1_out in zip(in_name, out_name):
        cls_dist = 0.0
        num_classes = len(feat_cls_means[d1_in])
        cnt = 0
        for c1 in range(num_classes):
            cls_dist += cosine_dist(feat_cls_means[d1_in][c1], feat_cls_means[d1_out][c1])
            cnt += 1
        cls_dist /= cnt
        stats['in_domain_in_cls'][d1_in] = cls_dist
        print(f"class_avg_dist({d1_in}) = {cls_dist}")
    # average distance between classes in different domains
    print("=== Section 4: in-domain cross-class distance ===")
    for d1 in in_name:
        cls_dist = 0.0
        num_classes = len(feat_cls_means[d1])
        class_pairs = combinations(range(num_classes), 2)
        cnt = 0
        for c1, c2 in class_pairs:
            cls_dist += cosine_dist(feat_cls_means[d1][c1], feat_cls_means[d1][c2])
            cnt += 1
        cls_dist /= cnt
        stats['in_domain_cross_cls'][d1] = cls_dist
        print(f"class_avg_dist({d1}) = {cls_dist}")
    # cross domain in class distance
    print("=== Section 5: cross-domain in-class distance ===")
    for d1, d2 in combinations(in_name, 2):
        cls_dist = 0.0
        num_classes = len(feat_cls_means[d1])
        cnt = 0
        for c1 in range(num_classes):
            cls_dist += cosine_dist(feat_cls_means[d1][c1], feat_cls_means[d2][c1])
            cnt += 1
        cls_dist /= cnt
        stats['cross_domain_in_cls'][(d1, d2)] = cls_dist
        print(f"class_avg_dist({d1}, {d2}) = {cls_dist}")
    # Cross-domain / in-domain
    print("=== Section 6: cross-domain / in-domain ===")
    for d1, d2 in combinations(in_name, 2):
        ratio = stats['cross_domain'][(d1, d2)] / ((stats['in_domain'][d1] + stats['in_domain'][d2])/2)
        print(f"[domain] cross/in({d1}, {d2}) = {ratio}")
    # In-domain cross class / in class
    print("=== Section 7: in-domain cross-class / in-class ===")
    for d1 in in_name:
        print(f"[in-domain class] cross/in({d1}) = {stats['in_domain_cross_cls'][d1] / stats['in_domain_in_cls'][d1]}")
    # In-class cross domain / in domain
    print("=== Section 8: in-class cross-domain / in-domain ===")
    for d1, d2 in combinations(in_name, 2):
        ratio = stats['cross_domain_in_cls'][(d1, d2)] / ((stats['in_domain'][d1] + stats['in_domain'][d2])/2)
        print(f"[in-class domain] cross/in({d1}, {d2}) = {ratio}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--restore', type=str, default=None)
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    eval_class = dataloader
    eval_loaders = [eval_class(
        dataset=env,
        batch_size=128,
        num_workers=8)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]
    
    # domain mapping
    cnt = 0
    domain_mapping = {x: None for x in args.test_envs}
    for i in range(len(in_splits)):
        if i not in args.test_envs:
            domain_mapping[i] = cnt
            cnt += 1

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    if args.restore:
        ckpt = torch.load(args.restore)
        algorithm.load_state_dict(ckpt["model_dict"])
        print("restored from {}".format(args.restore))

    algorithm.eval()
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    feat_means = dict()
    feat_cls_means = dict()
    for name, loader, weights in evals:
        feat_mean, feat_cls_mean = cal_feat_mean(algorithm, loader, weights, device, domain=domain_mapping[int(name[3])], algorithm=args.algorithm)
        feat_means[name] = feat_mean
        feat_cls_means[name] = feat_cls_mean
    domain_class_insight(feat_means=feat_means, feat_cls_means=feat_cls_means,)
    algorithm.train()