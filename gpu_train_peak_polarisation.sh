#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=5000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:40
#SBATCH --output=out_peak_polarisation.out

module load python/3.11.5 # Make sure to choose a version that suits your application
module load scipy-stack/2025a
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt

export WANDB_DIR=~/foundational_possum
export WANDB_MODE=offline
export WANDB_API_KEY=d0994170ead572e9093a51d04a45b0b56c2f45e2

python -u -m scripts.run_training -c config_peak_polarisation_gpu
