cd /bask/projects/v/vjgo8416-mt-qual-est
module restore system
module load bask-apps/test
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

CONDA_ENV_PATH="/bask/projects/v/vjgo8416-8416-mt-qual-est/mtqe_env"


conda activate ${CONDA_ENV_PATH}
