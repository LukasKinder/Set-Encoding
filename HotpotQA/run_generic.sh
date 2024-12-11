#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S

cd ..
source venv/bin/activate
cd HotpotQA

# Get command-line arguments
MODEL=$1
BATCH_SIZE=$2
MAX_TOKENS=$3
TOKENIZER_PATH=$4
USE_SET_ENCODING=$5 


PYTHON_ARGS="--model_id $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS --tokenizer_path $TOKENIZER_PATH"

if [ "$USE_SET_ENCODING" == "--use_set_encoding" ]; then
    PYTHON_ARGS="$PYTHON_ARGS $USE_SET_ENCODING"
fi


echo python main.py $PYTHON_ARGS
python main.py $PYTHON_ARGS
