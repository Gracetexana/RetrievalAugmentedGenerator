#!/bin/bash -l
# NOTE the -l flag!
# This is an example job file for a single core CPU bound program
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler
# Please copy this file to your home directory and modify it 
# to suit your needs.
#
# If you need any help, please [submit a ticket](https://help.rit.edu/sp?id=rc_request) or contact us on Slack.
#
# Name of the job - You'll probably want to customize this.
#SBATCH -J RAG
# Standard out and Standard Error output files
#SBATCH -o RAG.o
#SBATCH -e RAG.e
# To send slack notifications, set the address below and remove one of the '#' sings
#SBATCH --mail-user=slack:@gtl1500
# notify on state change: BEGIN, END, FAIL, OR ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Request 0 days, 0 hours, 30 minutes, and 0 seconds
#SBATCH -t 0-0:30:0
#Put the job in the appropriate partition matching the account and request one core
#SBATCH -A malont -p tier3 -c 1
#Job membory requirements in MB=m (default), GB=g, or TB=t
#SBATCH --mem=10g
#SBATCH --gres=gpu:a100:1
#
# Your job script goes below this line.
#

spack load cuda/sgx3wdz

source /home/gtl1500/miniconda3/etc/profile.d/conda.sh
conda activate personalRAG

python3 -u "/home/gtl1500/RetrievalAugmentedGenerator/RAG.py"
