#!/bin/bash

for ((i=0;i<10;i++));
do sbatch --job-name=texture_gen_${i} /gpfs/laur/data/xiongy/visualencoding/visual_stimuli/TextureSynthesis/yirong_sbatch_gpu.sh $i;
done