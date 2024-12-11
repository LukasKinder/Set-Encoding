#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --partition=L40S

# python train.py --save_path ./finetuned_models/llama3 --max_tokens 4000 --num_epochs 3 --use_set_encoding True --model_id "meta-llama/Meta-Llama-3-8B-instruct" --learning_rate 5e-6 --accumulation_steps 16 --batch_size 1

# Assign command-line arguments to variables
SAVE_PATH=$1
MAX_TOKENS=$2
NUM_EPOCHS=$3
USE_SET_ENCODING=$4
MODEL_ID=$5
LEARNING_RATE=$6
ACCUMLATION_STEPS=$7
BATCH_SIZE=$8

cd ../..
source venv/bin/activate
cd ContextWindow/HotpotQA

echo python train.py --save_path $SAVE_PATH --max_tokens $MAX_TOKENS --num_epochs $NUM_EPOCHS --use_set_encoding $USE_SET_ENCODING --model_id $MODEL_ID --learning_rate $LEARNING_RATE --accumulation_steps $ACCUMLATION_STEPS --batch_size $BATCH_SIZE
python train.py --save_path $SAVE_PATH --max_tokens $MAX_TOKENS --num_epochs $NUM_EPOCHS --use_set_encoding $USE_SET_ENCODING --model_id $MODEL_ID --learning_rate $LEARNING_RATE --accumulation_steps $ACCUMLATION_STEPS --batch_size $BATCH_SIZE