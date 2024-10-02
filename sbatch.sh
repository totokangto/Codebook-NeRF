#!/usr/bin/bash

#SBATCH -J nerf-sr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=128G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y4
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out


# train: original llff nerf
# bash scripts/train_llff.sh

# train
# bash scripts/train_llff_downX.sh

# test
# bash scripts/test_llff_downX.sh

# # Refinement
# python warp.py
bash scripts/train_llff_refine.sh

# test
# bash scripts/test_llff_refine.sh

# bash scripts/train_blender.sh
# bash scripts/train_blender_downX.sh

# bash scripts/test_blender.sh
# bash scripts/test_blender_downX.sh

exit 0
