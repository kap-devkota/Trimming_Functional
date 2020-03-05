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

## SBATCH -p largemem

#SBATCH --mem=4000

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=Kapil.Devkota@tufts.edu

module load anaconda/3
source activate bcb-env

######################### Parameters to Change ###################################
graph_folder=
graph_outputs_folder=
o_folder=${graph_folder}/Reduced_Graphs/Graph${SLURM_ARRAY_TASK_ID}
graph=${o_folder}/graph_r.txt
op=${o_folder}/graph_r_int.txt
dict=${o_folder}/label_to_int_label.json

##################################################################################

python convert_graph_labels_to_id.py -n ${dict} -o ${op} ${graph}

logs_f=${graph_folder}/logs/annotate_ppi

if [ ! -d ${logs_f} ]
then
    mkdir ${logs_f}
fi

if [ ! -d ${logs_f}/${SLURM_ARRAY_JOB_ID} ]
then
    mkdir ${logs_f}/${SLURM_ARRAY_JOB_ID}
    mkdir ${logs_f}/${SLURM_ARRAY_JOB_ID}/op
    mkdir ${logs_f}/${SLURM_ARRAY_JOB_ID}/er
fi

cp ../logs/temp/output_${SLURM_ARRAY_JOB_ID}*.log.txt ${logs_f}/${SLURM_ARRAY_JOB_ID}/op/
# rm logs/temp/output_${SLURM_ARRAY_JOB_ID}*.log.txt

cp ../logs/temp/output_${SLURM_ARRAY_JOB_ID}*.err.txt ${logs_f}/${SLURM_ARRAY_JOB_ID}/er/
# rm logs/temp/output_${SLURM_ARRAY_JOB_ID}*.err.txt
