#!/bin/bash

for ((i=0;i<3;i++));
do sbatch --job-name=texture_gen2_${i} /gpfs/laur/data/xiongy/visualencoding/visual_stimuli/TextureSynthesis/yirong_sbatch_gpu.sh $i;
done