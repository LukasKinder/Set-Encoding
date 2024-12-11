#!/bin/bash

#MODEL=$1 MAX_TOEKSN=$2 TOKENS_PER_DOCUMENT=$3 N_GENERATE=$4 USE_SET_ENCODING=$5 

# falcon
sbatch run_generic.sh tiiuae/falcon-7b-instruct 6000 1500 300
sbatch run_generic.sh tiiuae/falcon-7b-instruct 6000 1500 300 "--use_set_encoding"

exit 0

# phi-3
sbatch run_generic.sh microsoft/Phi-3-mini-4k-instruct 6000 1500 300
sbatch run_generic.sh microsoft/Phi-3-mini-4k-instruct 6000 1500 300 "--use_set_encoding"