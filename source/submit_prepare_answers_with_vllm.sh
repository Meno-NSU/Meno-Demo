#!/bin/sh

#SBATCH --job-name=rrncb_answers
#SBATCH --error=/storage2/bi/rrncb_answers.err 
#SBATCH --output=/storage2/bi/rrncb_answers.log 
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32G

. "/storage2/bi/miniconda/etc/profile.d/conda.sh"
conda activate vllm_env
export PIP_CACHE_DIR=/storage2/bi/.cache/pip
export HOME=/storage2/bi
export HF_HOME=/storage2/bi/huggingface_cached
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128
export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
nvidia-smi -L
nvcc -V
python -V
python -m pip list | grep torch
python -m pip list | grep evaluate
python -m pip list | grep sacrebleu
python -m pip list | grep annoy

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python -u process_questions_with_vllm.py -m /storage2/bi/models/Qwen2.5-14B-Instruct --gpu 0.85 --reranker encoder --minibatch 32 --eval
