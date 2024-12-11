#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S

cd ..
source venv/bin/activate
cd MCQ

# Get command-line arguments
MODEL=$1
BATCH_SIZE=$2
MAX_BATCHES=$3
EVAL_TYPE=$4
TOKENIZER_PATH=$5 # optional
FIRST_TOKENS_DIFFER=$6 #optional
USE_SET_ENCODING=$7 # optional

# Check if required arguments are provided
if [ -z "$MODEL" ] || [ -z "$BATCH_SIZE" ] || [ -z "$MAX_BATCHES" ] || [ -z "$EVAL_TYPE" ]; then
    echo "Usage: $0 <model> <batch_size> <max_batches> <eval_type> [<tokenizer_path>] [--first_tokens_differ] [--use_set_encoding] "
    exit 1
fi

# Construct Python command arguments
PYTHON_ARGS="--batch_size $BATCH_SIZE --max_batches $MAX_BATCHES --eval_type $EVAL_TYPE"

if [ "$USE_SET_ENCODING" == "--use_set_encoding" ]; then
    PYTHON_ARGS="$PYTHON_ARGS $USE_SET_ENCODING"
fi

if [ ! -z "$TOKENIZER_PATH" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --tokenizer_path $TOKENIZER_PATH"
fi

if [ "$FIRST_TOKENS_DIFFER" == "--first_tokens_differ" ]; then
    PYTHON_ARGS="$PYTHON_ARGS $FIRST_TOKENS_DIFFER"
fi

# Execute the Python script with the constructed arguments
echo python main.py "$MODEL" $PYTHON_ARGS
python main.py "$MODEL" $PYTHON_ARGS
