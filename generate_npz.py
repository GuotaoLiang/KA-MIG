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


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    # del samples
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="configs/GraphNAT_L_infer.yaml")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--benchmark', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ignored_keys', type=str, default=[], nargs='+')
    parser.add_argument('--pretrained_path', type=str, default="assert/graph_nnet_ema.pth")
    parser.add_argument('--gen_steps', type=int, default=8)
    parser.add_argument('--eval_n', type=int, default=50000)
    # AutoNAT parameters
    parser.add_argument('--beta_alpha_beta', type=float, nargs='+', default=(12, 3))
    parser.add_argument('--test_bsz', type=int, default=125)
    parser.add_argument('--sample_folder_dir', type=str,
                        default='output_sampling_npz')
    parser.add_argument('--searched_strategy', type=str, default="configs/AutoNAT_L-T8_strategy.yaml")
    
    args = parser.parse_args()
    return args


def LSimple(x0, nnet, schedule, **kwargs):
    timesteps, labels, xn = schedule.sample(x0)
    pred = nnet(xn, timesteps=timesteps, **kwargs)
    loss = schedule.loss(pred, labels)
    masked_token_ratio = xn.eq(schedule.mask_ind).sum().item() / xn.shape[0] / xn.shape[1]
    return loss, masked_token_ratio


@logger.catch()
def train(config, args):
    logger.add(os.path.join(args.output_dir, 'output.log'), level='INFO')
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    sample_folder_dir = args.sample_folder_dir
    accelerate.utils.set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        logger.info('Setting seed: {}'.format(args.seed))
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    logger.info(f'Process {accelerator.process_index} using device: {device}')

    # assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes
    args.batch_size = mini_batch_size
    logger.info(f'Using mini-batch size {mini_batch_size} per device')

    config.ckpt_root = os.path.join(args.output_dir, 'ckpts')
    config.searched_strategies_dir = os.path.join(args.output_dir, 'searched_strategies')
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.searched_strategies_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    logger.info(f'Run on {accelerator.num_processes} devices')

    global_batch_size = args.test_bsz * accelerator.num_processes
    autoencoder = taming.models.vqgan.get_model()
    codebook_size = autoencoder.n_embed
    config.nnet.codebook_size = codebook_size
    autoencoder.to(device)

    empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(device)

    train_state = utils.initialize_graph_adapter_train_state(config, device, args)

    _, nnet_ema, _ = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer)

    @torch.cuda.amp.autocast(enabled=True)
    def encode(_batch):
        return autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)

    @torch.cuda.amp.autocast(enabled=True)
    def decode(_batch):
        return autoencoder.decode_code(_batch)


    def get_test_generator():
        while True:
            yield torch.randint(0, 1000, (args.test_bsz, 1), device=device)

    schedule = GraphNATSchedule(codebook_size=codebook_size, device=device, **config.muse,
                           beta_alpha_beta=args.beta_alpha_beta)

    def cfg_nnet(x, scale, **kwargs):
        _cond = nnet_ema(x, **kwargs)
        kwargs['context'] = einops.repeat(empty_ctx, '1 ... -> B ...', B=x.size(0))
        _uncond = nnet_ema(x, **kwargs)
        res = _cond + scale * (_cond - _uncond)
        return res

    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    nnet_ema.module.load_state_dict(ckpt)

    @torch.no_grad()
    def eval_step_auto(n_samples, **nat_conf):
        # logger.info(f'evaluating with {temperature} and {guidance_scale}')
        test_generator = get_test_generator()

        batch_size = args.test_bsz * accelerator.num_processes

        total = 0

        for _batch_size in tqdm(utils.amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc=sample_folder_dir):
            contexts = next(test_generator)
            samples = schedule.generate_auto(args.gen_steps, len(contexts), cfg_nnet, decode, context=contexts,
                                        **nat_conf)
            # samples = samples.clamp_(0., 1.)
            samples = torch.clamp(samples, 0.0, 1.0)
            generated_image_256 = (samples * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(generated_image_256):
                index = i * accelerator.num_processes+ rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size



    prev_sd = OmegaConf.load(args.searched_strategy)
    eval_step_auto(n_samples=args.eval_n, **prev_sd)
            
    npz_path = create_npz_from_sample_folder(sample_folder_dir, num=50_000)
    print(f"Created NPZ file: {npz_path}")

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = get_args()
    config = OmegaConf.load(args.config)
    train(config, args)
