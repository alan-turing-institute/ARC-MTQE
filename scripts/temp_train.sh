#!/bin/sh
#SBATCH --account vjgo8416-mt-qual-est
#SBATCH --qos turing
#SBATCH --time 0-3:00:0
#SBATCH --nodes 1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node=1
#SBATCH --job-name mtqe-test-bask
#SBATCH --output ./slurm_train_logs/test-bask-train-%j.out

module purge
module load baskerville
#module load Miniconda3/4.10.3
#export CONDA_PKGS_DIRS=/tmp
#eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

#CONDA_ENV_PATH="/bask/projects/v/vjgo8416-mt-qual-est/mtqe_env"

#conda activate ${CONDA_ENV_PATH}
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0
source /bask/projects/v/vjgo8416-mt-qual-est/.cache/pypoetry/virtualenvs/mtqe-OL1VZIKQ-py3.10/bin/activate
export PYTHONPATH=$PYTHONPATH:/bask/projects/v/vjgo8416-mt-qual-est/.cache/pypoetry/virtualenvs/mtqe-OL1VZIKQ-py3.10/bin/python
python train_ced.py
