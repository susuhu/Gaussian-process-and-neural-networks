#!/bin/bash

#SBATCH --time=04:00:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # number of GPUS
#SBATCH --ntasks=1              # limit to one node
#SBATCH --cpus-per-task=1       # number of processor cores (i.e. threads)
#SBATCH --partition=ml          # defines access to hardware and software modules
#SBATCH --mem-per-cpu=8000M     # memory per CPU core
#SBATCH -J "convgp-test"         # job name
#SBATCH -A p_sp_hu  # credit to your project

#SBATCH -o /scratch/ws/0/suhu478b-gpvenv/slurm-%j.out     # save output messages %j is job-id
#SBATCH -e /scratch/ws/0/suhu478b-gpvenv/slurm-%j.err     # save error messages %j is job-id

#SBATCH --mail-type=end# send email notification when job finished
#SBATCH --mail-user=susu.hu@mailbox.tu-dresden.de

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
module load modenv/ml # loads the ml environment
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 # loads the tensorflow module
# module load matplotlib

source /scratch/ws/0/suhu478b-gpvenv/gpflow1/bin/activate # Source python venv

python /scratch/ws/0/suhu478b-gpvenv/GP_MNIST.py

exit 0