# This will loop through the slurm files in an experiment group and create a job for each of them
# Usage: scripts/slurm_train_all.sh drop-only
for FILE in find ./train_scripts/${1}*
do
    sbatch $FILE
done
