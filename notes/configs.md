Describe the sections in the YAML config files.

### slurm [only required if running from a slurm file]
These contain parameters for the slurm file when the model is run on an HPC cluster. In our case, we used the University of Birmingham's [Baskerville](https://docs.baskerville.ac.uk) system and our parameters in this section are specific to that system.

### wandb
Parameters that identify where to log the results in wandb.

### seeds
These are the initial random seeds to be used. Each experiment will run once for each of the seeds listed.

### hparams
These are the parameters that are used to define the model, such as the learning rate or dropout value. Any parameters listed here will be used across all experiments.

### model_checkpoint [optional]
The parameters that determine how and when a model checkpoint is saved. These parameters will be the valid

### early_stopping [optional]
The parameters for early stopping - if required.

### model_path [optional]
If the model weights should be initialised from a saved checkpoint, the path can be listed here. Otherwise COMETKiwi-22 is taken as a starting point.

### trainer_config
The parameters used by the Pytorch Lightning Trainer object.

### experiments
The data in this section defines the various experiments to be run given the above parameters. Each experiment will be defined by a name and the training and development data to be used. They might also have their own `slurm` or `hparams` parameter values as these might be different by experiment.
