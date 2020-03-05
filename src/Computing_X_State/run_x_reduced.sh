#!/bin/bash

## output files

#SBATCH --output=../logs/temp/output_%A_%a.log.txt
#SBATCH --error=../logs/temp/output_%A_%a.err.txt

#

# Estimated running time. The job will be killed when it runs 15 min longer     than this time.

#SBATCH --time=7-00:00:00

#

## Resources

## -p gpu/batch  |job type

## -N            |number of nodes

## -n            |number of tasks

##SBATCH -p largemem

#SBATCH --mem=64000

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=Kapil.Devkota@tufts.edu

module load anaconda/3
source activate bcb-env

######################### Parameters to change #################################
folder=../Reduced_Graphs_2/Graph${SLURM_ARRAY_TASK_ID}
reduced_dims=${1}
NODELIST=${folder}/node_list_r.json
RED_GRAPH=${folder}/graph_r.txt

OP_STATE=${folder}/R_state_l_1.00_${reduced_dims}.npy

python run_x_reduced.py -j ${NODELIST} -r ${reduced_dims} ${RED_GRAPH} ${OP_STATE}

folder_name=state_computation
if [ ! -d "../logs/${folder_name}" ]
then
    mkdir ../logs/${folder_name}
fi

if [ ! -d "../logs/${folder_name}/${SLURM_ARRAY_JOB_ID}" ]
then
    mkdir ../logs/${folder_name}/${SLURM_ARRAY_JOB_ID}
    mkdir ../logs/${folder_name}/${SLURM_ARRAY_JOB_ID}/op
    mkdir ../logs/${folder_name}/${SLURM_ARRAY_JOB_ID}/er
fi

cp ../logs/temp/output_${SLURM_ARRAY_JOB_ID}*.log.txt ../logs/${folder_name}/${SLURM_ARRAY_JOB_ID}/op/
# rm logs/temp/output_${SLURM_ARRAY_JOB_ID}*.log.txt

cp ../logs/temp/output_${SLURM_ARRAY_JOB_ID}*.err.txt ../logs/${folder_name}/${SLURM_ARRAY_JOB_ID}/er/
# rm logs/temp/output_${SLURM_ARRAY_JOB_ID}*.err.txt
