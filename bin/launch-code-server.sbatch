#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G
#SBATCH --job-name=code_server
#SBATCH --mail-type=ALL
#SBATCH --output=bin/%x-%j-slurm.out
#SBATCH --error=bin/%x-%j-slurm.err

# use srun to launch code server in order to reserve a port
srun --resv-ports=1 ./bin/launch-code-server.srun
