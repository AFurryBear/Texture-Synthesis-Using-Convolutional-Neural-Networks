#!/bin/bash
#SBATCH -N 1
#SBATCH -p turing
#SBATCH -D /gpfs/laur/data/xiongy/visualencoding/visual_stimuli/TextureSynthesis/
#SBATCH -w lnx-cm-21004.mpibr.local
#SBATCH -o ./logs/job.out.%j
#SBATCH -e ./logs/job.err.%j
cd /gpfs/laur/data/xiongy/visualencoding/visual_stimuli/TextureSynthesis;
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3 #specifies the GPU to use on the node
.venv/bin/python syn_texture.py rock-0.pgm $1 2
