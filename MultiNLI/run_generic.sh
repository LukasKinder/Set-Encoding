#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --gres=gpu:1
#SBATCH --partition=4090

cd ..
source venv/bin/activate
cd MultiNLI

# Get command-line arguments
MODEL=$1
BATCH_SIZE=$2
ORDER=$3
USE_SET_ENCODING=$4 # 
CALLIBRATE=$5

PYTHON_ARGS="--model_id $MODEL --batch_size $BATCH_SIZE --order $ORDER"

if [ "$USE_SET_ENCODING" == "--use_set_encoding" ]; then
    PYTHON_ARGS="$PYTHON_ARGS $USE_SET_ENCODING"
fi

if [ "$CALLIBRATE" == "--callibrate" ]; then
    PYTHON_ARGS="$PYTHON_ARGS $CALLIBRATE"
fi

echo python main.py $PYTHON_ARGS
python main.py $PYTHON_ARGS
