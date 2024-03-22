#!/bin/sh
#SBATCH --account {{account_name}}
#SBATCH --qos turing
#SBATCH --time 0-0:30:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name mtqe-{{experiment_name}}
#SBATCH --output ./slurm_train_logs/{{experiment_name}}-train-%j.out

module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

CONDA_ENV_PATH="/bask/projects/v/vjgo8416-8416-mt-qual-est/mtqe_env"

conda activate ${CONDA_ENV_PATH}
{{python_call}}
