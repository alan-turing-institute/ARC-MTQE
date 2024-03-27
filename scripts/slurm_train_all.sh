# This will loop through the slurm files in an experiment group and create a job for each of them
# Usage: scripts/slurm_train_all.sh drop-only
for FILE in find ./slurm_scripts/${1}/
do
  if [[$FILE == *.sh ]]
  then
    sbatch $FILE
  fi
done
