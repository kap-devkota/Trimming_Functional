#!/bin/bash

no_iter=1

exp=D_3_Complete
red=
lambda=1

./run_x_compute.sh ${lambda} ${exp} ${red}
# sbatch --array=1-${no_iter} run_x_compute.sh ${lambda} ${exp} ${red}