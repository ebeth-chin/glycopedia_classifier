#!/bin/bash


#SBATCH --time=14:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 1 processor core(s) per node X 2 threads per core
#SBATCH --partition=gpu    # standard node(s)
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --signal=USR2
#SBATCH --mail-user={your-email@supercoolscienceemail.com}
#SBATCH --mail-type=ALL


module load cuda

set -x

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export CUDA_VISIBLE_DEVICES="0,1"

HOSTNAMES=`srun hostname -s | sort | uniq`

TF_CONFIG='{"cluster": {"worker": ['
for node in $HOSTNAMES
do
TF_CONFIG+='"'$node':12345", '
done
TF_CONFIG="${TF_CONFIG::-2}"
TF_CONFIG+=']}, "task": {"index": 0, "type": "worker"}}'

export TF_CONFIG

#srun -n1 -N1 python3 02_tiny_cnn.py
srun -n1 -N1 python3 03_finetune.py
