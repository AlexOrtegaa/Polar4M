#!/bin/bash
#SBATCH --constraint=A100
#SBATCH --time=13-23
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=20
#SBATCH --job-name=vqvae
#SBATCH --ntasks-per-node=1
#SBATCH --exclude compute-0-18

pwd; hostname; date

nvidia-smi

# torch device should have same order as nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "CUDA_DEVICE_ORDER set to $CUDA_DEVICE_ORDER"

export HF_HOME="/share/nas2/walml/cache/huggingface"
export HF_DATASETS_CACHE='/state/partition1/walml/cache/huggingface/datasets'  # load on node

PYTHON=/share/nas2/walml/miniconda3/envs/zoobot39_cu118_dev/bin/python

# CONFIG="config_stokes_i"
CONFIG="config_peak_polarisation"

# run from foundational_possum directory
srun $PYTHON -m scripts.run_training -c $CONFIG

# copy results to dropbox (run externally)
# rsync -avz -e 'ssh -A -J walml@external.jb.man.ac.uk' walml@galahad.ast.man.ac.uk:"/share/nas2/walml/repos/foundationalmodel-possum/checkpoints/*/epoch_1*" .
