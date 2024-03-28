# Baskerville

# Account and login

Follow the [Baskerville instructions](https://docs.baskerville.ac.uk/logging-on/#first-time-access) for creating an account.

Make sure you add your local SSH key to your Baskerville account. You can copy it on a Mac using the below command:

```bash
cat ~/.ssh/id_rsa.pub | pbcopy
```

You can then login with:

```bash
ssh <username>@login.baskerville.ac.uk
```

which will prompt you for your one time password (OTP). You can only use a single OTP once per SSH session so might need to wait for it to regenerate if you want to use multiple sessions. This takes you to a login node and you need to navigate to your project from there:

```bash
cd /bask/projects/v/<project_space_name>
```

Check out the [Baskerville docs](https://docs.baskerville.ac.uk) for tips on getting started and running jobs. Or search for available applications and where to load them from [here](https://apps.baskerville.ac.uk/search).

# Setting up a project

If your repository is private, you will need to git clone through SSH. Create an SSH key on Baskerville:

```bash
git config user.name "<your_username>"
ssh-keygen -t rsa -b 4096 -C "<your_email>"
cat ~/.ssh/id_rsa.pub
```

then on GitHub go to `Settings/SSH AND GPG keys/New SSH key` and add the newly generated Baskerville SSH key to your account. You can now clone your repository:

```bash
cd /bask/projects/v/<project_space_name>
git clone git@github.com:alan-turing-institute/<arc_git_repo>.git
```

Load make to download and preprocess data:

```bash
module load bask-apps/live
module load make/4.3-GCCcore-12.2.0
make data
```

To create a link to the project from your home directory:

```
ln -s /bask/projects/v/<project_space_name>
```

# Setting up Poetry

To use Poetry:

```bash
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0
```

Change the Poetry cache-dir to the project directory (it is the user directory by default). This will also change where the virtual environment is created.

```bash
poetry config cache-dir /bask/projects/v/<project_space_name>/.cache/pypoetry
```

Then you can install your project:

```bash
poetry install
```

Follow the login instructions in the README for HuggingFace and WandB.

To activate your poetry environment:

```
source /bask/projects/v/<project_space_name>/.cache/pypoetry/virtualenvs/<env_name>/bin/activate

export PYTHONPATH=$PYTHONPATH:/bask/projects/v/<project_space_name>/.cache/pypoetry/virtualenvs/<env_name>/bin/python
```

# Running jobs

Create directory for logs (in the scripts directory):

```bash
mkdir slurm_train_logs
```

Submit the job:

```bash
sbatch <slurm script>.sh
```
