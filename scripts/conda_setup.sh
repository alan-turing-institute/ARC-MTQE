# cd /bask/projects/v/vjgo8416-mt-qual-est
module restore system
module load bask-apps/test
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your Conda environment (modify as appropriate)
# N.B. this path will be created by the subsequent commands if it doesn't already exist
CONDA_ENV_PATH="/bask/projects/v/vjgo8416-mt-qual-est/mtqe_env"

# Create the environment. Only required once.
conda create --yes --prefix "${CONDA_ENV_PATH}"
# Activate the environment
conda activate "${CONDA_ENV_PATH}"
# Choose your version of Python
conda install --yes python=3.9
