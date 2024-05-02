#!/bin/sh
#SBATCH --account {{account_name}}
#SBATCH --qos turing
#SBATCH --time {{time}}
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --tasks-per-node 1
#SBATCH --mem {{memory}}
#SBATCH --job-name mtqe-{{experiment_name}}
#SBATCH --output ./slurm_train_logs/{{experiment_name}}-train-%j.out


module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-mt-qual-est/.cache/pypoetry/virtualenvs/mtqe-OL1VZIKQ-py3.10/bin/activate
export PYTHONPATH=$PYTHONPATH:/bask/projects/v/vjgo8416-mt-qual-est/.cache/pypoetry/virtualenvs/mtqe-OL1VZIKQ-py3.10/bin/python

{{python_call}}
