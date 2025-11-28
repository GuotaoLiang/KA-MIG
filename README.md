# Improved Masked Image Generation with Knowledge-Augmented Token Representations
This repo contains the official PyTorch implementation of [Improved Masked Image Generation with Knowledge-Augmented Token Representations](https://arxiv.org/abs/2511.12032).

## Overview

Masked image generation (MIG) has demonstrated remarkable efficiency and high-fidelity images by enabling parallel token prediction. Existing methods typically rely solely on the model itself to learn semantic dependencies among visual token sequences. However, directly learning such semantic dependencies from data is challenging because the individual tokens lack clear semantic meanings, and these sequences are usually long. To address this limitation, we propose a novel Knowledge-Augmented Masked Image Generation framework, named KA-MIG, which introduces explicit knowledge of token-level semantic dependencies (i.e., extracted from the training data) as priors to learn richer representations for improving performance. In particular, we explore and identify three types of advantageous token knowledge graphs, including two positive and one negative graphs (i.e., the co-occurrence graph, the semantic similarity graph, and the position-token incompatibility graph). Based on three prior knowledge graphs, we design a graph-aware encoder to learn token and position-aware representations. After that, a lightweight fusion mechanism is introduced to integrate these enriched representations into the existing MIG methods. Resorting to such prior knowledge, our method effectively enhances the model's ability to capture semantic dependencies, leading to improved generation quality. 

![motivation.png](assets/motivation.png)
*Motivation and framework overview. (a) The training pipeline of Masked Image Generation (MIG). (b) The inference process in MIG, where existing works primarily focus on improving sampling quality. (c) Our proposed framework incorporates external prior knowledge graphs to learn richer representations for enhancing MIG.*

![model.png](assets/model.png)

![generation_images.png](assets/generation_images.jpg)

## Installation
We support PyTorch==2.0.1 and torchvision==0.15.2. Please install them following the official instructions.

Clone this repo and install the required packages:

```shell
git clone https://github.com/GuotaoLiang/KA-MIG.git
pip install -r requirements.txt
```

## Data Preparation

- The ImageNet dataset should be prepared as follows:

```
data
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
```

- Tokenizing the ImageNet dataset:
Use [this link](https://drive.google.com/file/d/13S_unB87n6KKuuMdyMnyExW0G1kplTbP/view?usp=sharing) to download the pre-trained VQGAN tokenizer and put it in `assets/vqgan_jax_strongaug.ckpt`. Then run the following command to tokenize the ImageNet-256 dataset:

```shell
python extract_imagenet_feature.py --path data --split train
```
This command will save the tokenized ImageNet dataset to `assets/imagenet256_vq_features`. 

- Prepare FID-stats:
Download the FID-stats from [this link](https://drive.google.com/file/d/1C7DgARuZi9-InTYOgpkE3pggkJB6DMZD/view?usp=drive_link) and put it in `assets/fid_stats` directory.

- Prepare pre-trained inception model for FID calculation:
Download the pre-trained inception model from [this link](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth) and put it in `assets/pt_inception-2015-12-05-6726825d.pth`.

## Pre-trained Model

### Backbone Model (AutoNAT)
Download the pre-trained backbone model, namely AutoNAT-L, from [this link](https://drive.google.com/file/d/12NN62Vg1DiOd-3sWkEnBQRIBBbuiVkNl/view?usp=sharing) and put it in `assets/nnet_ema.pth`.

### Our Pre-trained Model
Download our pre-trained KA-MIG model from [this link](https://drive.google.com/file/d/1jBJIappmKaeJ2QB51h8u78Ghpelh5snm/view?usp=sharing) and put it in `assets/graph_nnet_ema.pth`.

### Graph Prior Knowledge
Download the class_pos_graph from [this link](graph...) and put it in `graph_prior/`.

We also provide `train_img_50K_token.pkl` which contains 50K images sampled from ImageNet for constructing the co-occurrence graph and the semantic similarity graph. The code for constructing these two graphs is provided in the `graph_construction/` directory.



## Training

To train the KA-MIG model, run the following command:

```shell
export ACCELERATE_MIXED_PRECISION=fp16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_graph.py
```

## Image Generation

To generate images using the trained model, run:

```shell
python generate_img.py
```

## Evaluation

To evaluate the model performance, run the following command:

```shell
export ACCELERATE_MIXED_PRECISION=fp16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 generate_npz.py
```

We also provide 50K generated images for FID evaluation. You can download them from [this link](https://drive.google.com/file/d/1jBJIappmKaeJ2QB51h8u78Ghpelh5snm/view?usp=sharing).

## Citation

If you find our work useful for your research, please consider citing

```
@article{liang2025improved,
  title={Improved Masked Image Generation with Knowledge-Augmented Token Representations},
  author={Liang, Guotao and Zhang, Baoquan and Wen, Zhiyuan and Han, Zihao and Ye, Yunming},
  journal={arXiv preprint arXiv:2511.12032},
  year={2025}
}
```

## Acknowledgements

Our implementation is based on
- [U-ViT](https://github.com/baofff/U-ViT) (Pre-training code and network architecture)
- [MaskGIT](https://github.com/google-research/maskgit) (NAT sampling code)
- [MAGE](https://github.com/LTH14/mage?tab=readme-ov-file) (VQGAN weights)
- [VQGAN](https://github.com/CompVis/taming-transformers) (VQGAN code)
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) (official implementation of FID in PyTorch)
- [AutoNAT](https://github.com/LeapLabTHU/ImprovedNAT) (Backbone model implementation)

We thank the authors for their excellent work.

## Contact

If you have any questions, please send mail to [lianggt@pcl.ac.cn](mailto:lianggt@pcl.ac.cn) or [23B951062@stu.hit.edu.cn](mailto:23B951062@stu.hit.edu.cn) (Harbin Institute of Technology, Shenzhen).
