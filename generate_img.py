import datetime
import math
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch import optim
from torch.nn import Parameter
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from tqdm.auto import tqdm
import accelerate
import einops
from omegaconf import OmegaConf
from loguru import logger
import taming.models.vqgan
from libs.nat_misc import GraphNATSchedule

import utils
from libs.inception import InceptionV3
from dataset import get_dataset
from PIL import Image
from torch._C import _distributed_c10d

_distributed_c10d.set_debug_level(_distributed_c10d.DebugLevel.INFO)
import setproctitle
setproctitle.setproctitle('liang')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="configs/GraphNAT_L_infer.yaml")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--benchmark', type=int, default=0)
    # parser.add_argument('--mode', type=str, choices=['pretrain', 'search', 'eval'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ignored_keys', type=str, default=[], nargs='+')
    parser.add_argument('--pretrained_path', type=str, default="assert/graph_nnet_ema.pth")
    parser.add_argument('--gen_steps', type=int, default=8)
    parser.add_argument('--eval_n', type=int, default=50000)
    # AutoNAT parameters
    parser.add_argument('--beta_alpha_beta', type=float, nargs='+', default=(12, 3))
    parser.add_argument('--test_bsz', type=int, default=100)
    parser.add_argument('--sample_folder_dir', type=str,
                        default='sampling_npz/graph_nnet')
    parser.add_argument('--searched_strategy', type=str, default="configs/AutoNAT_L-T8_strategy.yaml")
    
    args = parser.parse_args()
    return args


@torch.cuda.amp.autocast(enabled=True)
def decode(_batch):
    return autoencoder.decode_code(_batch)


def cfg_nnet(x, scale, **kwargs):
    _cond = nnet_ema(x, **kwargs)
    kwargs['context'] = einops.repeat(empty_ctx, '1 ... -> B ...', B=x.size(0))
    _uncond = nnet_ema(x, **kwargs)
    res = _cond + scale * (_cond - _uncond)
    return res

device = "cuda:0"

args = get_args()
config = OmegaConf.load(args.config)


autoencoder = taming.models.vqgan.get_model()
codebook_size = autoencoder.n_embed
config.nnet.codebook_size = codebook_size

autoencoder.to(device)


empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(device)

train_state = utils.initialize_graph_adapter_train_state(config, device, args)


nnet_ema = train_state.nnet_ema

ckpt = torch.load(args.pretrained_path, map_location='cpu')

nnet_ema.load_state_dict(ckpt)

schedule = GraphNATSchedule(codebook_size=codebook_size, device=device, **config.muse,
                           beta_alpha_beta=args.beta_alpha_beta)

class_ids = [17, 24, 90, 175, 220, 248, 281, 388, 292]
for c in class_ids:
    batch_size = 50
    contexts = torch.tensor([c] * batch_size, device=device).reshape(batch_size, 1)

    prev_sd = OmegaConf.load(args.searched_strategy)
    print("start generating......")
    samples = schedule.generate_auto(args.gen_steps, len(contexts), cfg_nnet, decode, context=contexts,**prev_sd)

    samples = torch.clamp(samples, 0.0, 1.0)
    generated_image_256 = (samples * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    sample_folder_dir = f"sampling_img/{c}-2"
    os.makedirs(sample_folder_dir, exist_ok=True)

    for i, sample in enumerate(generated_image_256):
        Image.fromarray(sample).save(f"{sample_folder_dir}/{i:06d}.png")
