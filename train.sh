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


#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=notifications.out

module load python/3.11.5 # Make sure to choose a version that suits your application
module load scipy-stack/2025a
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r requirements.txt

python -u -m scripts.run_training -c config_peak_polarisation_gpu
