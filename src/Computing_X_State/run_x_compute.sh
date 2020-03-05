#!/bin/bash

#SBATCH --output=../logs/output_%A.log.txt
#SBATCH --error=../logs/output_%A.err.txt
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=END
#SBATCH --mail-user=Kapil.Devkota@tufts.edu

##########################################################################################

module load anaconda/3
source activate bcb-env

##########################################################################################

positional=()
while [ "${1}" != "" ]
do
    case ${1} in
	-la ) shift
	    LAMBDA=${1}
	    ;;
	* ) positional+=(${1})
    esac
    shift
done

######################### Parameters to change ###########################################

exp_dir=../Experiments/${positional[0]}
NODELIST=${exp_dir}/node_list.json
RED_GRAPH=${exp_dir}/graph_f.txt
OP_STATE=${exp_dir}/X_r_inf_lambda_${LAMBDA}.npy

##########################################################################################

python gen_x_state.py -j ${NODELIST} -p ${LAMBDA} ${RED_GRAPH} ${OP_STATE}

##########################################################################################
if [ ! -d ${exp_dir}/logs ]
then
    mkdir ${exp_dir}/logs
fi

folder_name=${exp_dir}/logs/state_computation

if [ ! -d ${folder_name} ]
then
    mkdir ${folder_name}
fi

if [ ! -d "${folder_name}/${SLURM_JOB_ID}" ]
then
    mkdir ${folder_name}/${SLURM_JOB_ID}
fi

cp ../logs/output_${SLURM_JOB_ID}.log.txt ${folder_name}/${SLURM_JOB_ID}/log.txt
cp ../logs/output_${SLURM_JOB_ID}.err.txt ${folder_name}/${SLURM_JOB_ID}/er.txt



