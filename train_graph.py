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
from datetime import datetime
import utils
from libs.inception import InceptionV3
from dataset import get_dataset
from PIL import Image
from torch._C import _distributed_c10d
import yaml
_distributed_c10d.set_debug_level(_distributed_c10d.DebugLevel.INFO)

import setproctitle
setproctitle.setproctitle('liang')

today = datetime.today()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="configs/GraphNAT_L.yaml")
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--benchmark', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['pretrain', 'search', 'eval'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ignored_keys', type=str, default=[], nargs='+')
    parser.add_argument('--gen_steps', type=int, default=8)
    parser.add_argument('--eval_n', type=int, default=50000)
    # AutoNAT parameters
    parser.add_argument('--beta_alpha_beta', type=float, nargs='+', default=(12, 3))
    parser.add_argument('--test_bsz', type=int, default=125)
    parser.add_argument('--reference_image_path', type=str,
                        default='assets/fid_stats_imagenet256_guided_diffusion.npz')
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
    best_fid = 999999
    history_fid = {

    }

    args.output_dir = args.output_dir + '/' +str(today.strftime("%Y-%m-%d-%H"))
    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, 'output.log'), level='INFO')
    if args.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        logger.info(f'yaml is \n {OmegaConf.to_yaml(config)}')
        logger.info('Setting seed: {}'.format(args.seed))
    logger.info(f'Process {accelerator.process_index} using device: {device}')

    # assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size
    
    args.batch_size = mini_batch_size
    logger.info(f'Using mini-batch size {mini_batch_size} per device')

    config.ckpt_root = os.path.join(args.output_dir, 'ckpts')
    # config.searched_strategies_dir = os.path.join(args.output_dir, 'searched_strategies')
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        # os.makedirs(config.searched_strategies_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    logger.info(f'Run on {accelerator.num_processes} devices')

    # prepare for fid calc
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()
    inception.requires_grad_(False)
    # load npz file
    with np.load(args.reference_image_path) as f:
        m2, s2 = f['mu'][:], f['sigma'][:]
        m2, s2 = torch.from_numpy(m2).to(device), torch.from_numpy(s2).to(device)

    autoencoder = taming.models.vqgan.get_model()
    codebook_size = autoencoder.n_embed
    config.nnet.codebook_size = codebook_size
    autoencoder.to(device)

    # load npy dataset
    dataset = get_dataset(**config.dataset)
    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True
                                      )
    # for cfg:
    empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(device)

    train_state = utils.initialize_graph_adapter_train_state(config, device, args)

    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)

    assert len(optimizer.param_groups) == 1
    lr_scheduler = train_state.lr_scheduler
    if args.resume and not bool(os.listdir(config.ckpt_root)):
        train_state.resume(args.resume, ignored_keys=args.ignored_keys)
    else:
        train_state.resume(config.ckpt_root)

    @torch.cuda.amp.autocast(enabled=True)
    def encode(_batch):
        return autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)

    @torch.cuda.amp.autocast(enabled=True)
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    def get_data_generator():
        while True:
            for data in train_dataset_loader:
                yield data

    data_generator = get_data_generator()

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

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        with torch.no_grad():
            _z = _batch[0]
            context = _batch[1]
        loss, masked_token_ratio = LSimple(_z, nnet, schedule, context=context)
        metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
        metric_logger.update(masked_token_ratio=masked_token_ratio)
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step(train_state.step)
        train_state.ema_update(config.get('ema_rate', 0.9999))
        metric_logger.update(loss_scaler=accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.)
        metric_logger.update(grad_norm=utils.get_grad_norm_(optimizer.param_groups[0]['params']))

        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'],
                    **{k: v.value for k, v in metric_logger.meters.items()})

    @torch.no_grad()
    def eval_step(n_samples, temperature, guidance_scale):
        logger.info(f'evaluating with {temperature} and {guidance_scale}')
        test_generator = get_test_generator()

        batch_size = args.test_bsz * accelerator.num_processes

        idx = 0

        pred_tensor = torch.empty((n_samples, 2048), device=device)
        for _batch_size in tqdm(utils.amortize(n_samples, batch_size), disable=not accelerator.is_main_process,
                                desc=f'sample2dir temp {temperature} guidance {guidance_scale}'):
            contexts = next(test_generator)
            samples = schedule.generate(args.gen_steps, len(contexts), cfg_nnet, decode, 
                                                temperature=temperature,
                                                guidance_scale=guidance_scale,
                                                context=contexts,
                                                )
            samples = samples.clamp_(0., 1.)

            pred = inception(samples.float())[0]

            # Apply global spatial average pooling if needed
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)
            pred_tensor[idx:idx + pred.shape[0]] = pred

            idx = idx + pred.shape[0]

        pred_tensor = pred_tensor[:idx].to(device)
        pred_tensor = accelerator.gather(pred_tensor)

        pred_tensor = pred_tensor[:n_samples]

        m1 = torch.mean(pred_tensor, dim=0)
        pred_centered = pred_tensor - pred_tensor.mean(dim=0)
        s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

        m1 = m1.double()
        s1 = s1.double()

        fid = utils.calc_fid(m1, s1, m2, s2)

        if accelerator.is_main_process:
            logger.info(f'FID{n_samples}={fid}')
        return {f'fid{n_samples}': fid}

        
    @torch.no_grad()
    def eval_step_auto(n_samples, **nat_conf):
        # logger.info(f'evaluating with {temperature} and {guidance_scale}')
        test_generator = get_test_generator()

        batch_size = args.test_bsz * accelerator.num_processes

        idx = 0

        pred_tensor = torch.empty((n_samples, 2048), device=device)
        for _batch_size in tqdm(utils.amortize(n_samples, batch_size), disable=not accelerator.is_main_process):
            contexts = next(test_generator)
            samples = schedule.generate_auto(args.gen_steps, len(contexts), cfg_nnet, decode, context=contexts,
                                        **nat_conf)
            samples = samples.clamp_(0., 1.)

            pred = inception(samples.float())[0]

            # Apply global spatial average pooling if needed
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)
            pred_tensor[idx:idx + pred.shape[0]] = pred

            idx = idx + pred.shape[0]

        pred_tensor = pred_tensor[:idx].to(device)
        pred_tensor = accelerator.gather(pred_tensor)

        pred_tensor = pred_tensor[:n_samples]

        m1 = torch.mean(pred_tensor, dim=0)
        pred_centered = pred_tensor - pred_tensor.mean(dim=0)
        s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

        m1 = m1.double()
        s1 = s1.double()

        fid = utils.calc_fid(m1, s1, m2, s2)

        if accelerator.is_main_process:
            logger.info(f'FID{n_samples}={fid}')
        return {f'fid{n_samples}': fid}


    @torch.no_grad()
    def generate_images():
        save_path = os.path.join(args.output_dir, 'images')
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Generating images...path is {save_path}")
        # fmt: off
        imagenet_class_names = ["Jay", "Castle", "coffee mug", "desk", "Husky", "Valley", "Red wine", "Coral reef",
                                "Mixing bowl", "Cleaver", "Vine Snake", "Bloodhound", "Barbershop", "Ski", "Otter",
                                "Snowmobile"]
        # fmt: on
        imagenet_class_ids = torch.tensor(
            [17, 483, 504, 526, 248, 979, 966, 973, 659, 499, 59, 163, 424, 795, 360, 802],
            device=accelerator.device,
            dtype=torch.long,
        ).reshape(-1, 1)
        images = schedule.generate(args.gen_steps,
                                    len(imagenet_class_ids),
                                    cfg_nnet,
                                    decode,
                                    temperature=7,
                                    guidance_scale=1,
                                    context=imagenet_class_ids,
                                    )
        images = 2.0 * images - 1.0
        images = torch.clamp(images, -1.0, 1.0)
        images = (images + 1.0) / 2.0
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        for idx, img in enumerate(pil_images):
            p = os.path.join(save_path, f'{imagenet_class_names[idx]}.png')
            img.save(p)


    # if args.mode == 'pretrain':
    logger.info(f'Start fitting, step={train_state.step}, mixed_precision={accelerator.mixed_precision}')
    prev_sd = OmegaConf.load(args.searched_strategy)
    metric_logger = utils.MetricLogger()
    # res = eval_step_auto(n_samples=args.eval_n,
    #                         **prev_sd)
    while train_state.step < config.train.n_steps:
        nnet.train()
        data_time_start = time.time()
        batch = next(data_generator)
        if isinstance(batch, list):
            batch = tree_map(lambda x: x.to(device), next(data_generator))
        else:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        metric_logger.update(data_time=time.time() - data_time_start)
        metrics = train_step(batch)

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            res = eval_step_auto(n_samples=args.eval_n,
                            **prev_sd)
            if accelerator.is_main_process:
                logger.info(f'Evaluated {args.eval_n} samples with strategy {train_state.step}:')
                res_fid = res['fid50000']
                with open(os.path.join(args.output_dir, "step_fid.txt"), "a") as f:
                    f.write(f"{train_state.step}\t{res_fid}\n")
                # if best_fid > res_fid:
                #     best_fid = res_fid
            if res['fid50000'] < 2.58:
                logger.info(f'Save checkpoint {train_state.step}...')
                if accelerator.local_process_index == 0:
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
                

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logger.info(f'[step {train_state.step}]: {metrics}')

            
        
            

    logger.info(f'Finish fitting, step={train_state.step}')


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = get_args()
    config = OmegaConf.load(args.config)
    
    train(config, args)
