export ACCELERATE_MIXED_PRECISION=fp16
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_graph.py