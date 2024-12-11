#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S

cd ..
cd ..
source venv/bin/activate
cd ContextWindow/Multi-News

# Get command-line arguments
MODEL=$1
MAX_TOEKSN=$2
TOKENS_PER_DOCUMENT=$3
N_GENERATE=$4
USE_SET_ENCODING=$5 


PYTHON_ARGS="--model_id $MODEL --max_tokens $MAX_TOEKSN --max_tokens_per_document $TOKENS_PER_DOCUMENT --n_generate $N_GENERATE"

if [ "$USE_SET_ENCODING" == "--use_set_encoding" ]; then
    PYTHON_ARGS="$PYTHON_ARGS $USE_SET_ENCODING"
fi


echo python main.py $PYTHON_ARGS
python main.py $PYTHON_ARGS