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

#SBATCH --mem=32000

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=Kapil.Devkota@tufts.edu

module load anaconda/3
source activate bcb-env

#rm Reduced_Graph/Graph${SLURM_ARRAY_TASK_ID}/graph_[dl]*
op_folder=../Reduced_Graph/Graph${SLURM_ARRAY_TASK_ID}/ranked_p_full_n_10000_dm_l1
ip_file=../Reduced_Graph/Graph${SLURM_ARRAY_TASK_ID}/ncomponent_f_align_p_10000/projected_-1_no_0.txt
dict=../Reduced_Graph/Graph${SLURM_ARRAY_TASK_ID}/node_list.json
fname=ranked_p_full_n_10000_dm_l1.txt
if [ ! -d $op_folder ]
then
    mkdir $op_folder
fi

python compute_annotations.py -d $dict -o $op_folder/$fname $ip_file

folder_name=annotate

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
