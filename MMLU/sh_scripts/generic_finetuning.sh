#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S


# Check if the correct number of arguments are passed
if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <save_path> <use_set_encoding> <model_id> <learning_rate> <accumulation_steps> <batch_size> <balance_proportion>"
  exit 1
fi

# Assign command-line arguments to variables
SAVE_PATH=$1
USE_SET_ENCODING=$2
MODEL_ID=$3
LEARNING_RATE=$4
ACCUMLATION_STEPS=$5
BATCH_SIZE=$6
PROPORTION=$7

cd ..
cd ..
source venv/bin/activate
cd MCQ

echo python finetune_trainer.py --save_path $SAVE_PATH --use_set_encoding $USE_SET_ENCODING --model_id $MODEL_ID --learning_rate $LEARNING_RATE --accumulation_steps $ACCUMLATION_STEPS --batch_size $BATCH_SIZE --balance $PROPORTION
python finetune_trainer.py --save_path $SAVE_PATH --use_set_encoding $USE_SET_ENCODING --model_id $MODEL_ID --learning_rate $LEARNING_RATE --accumulation_steps $ACCUMLATION_STEPS --batch_size $BATCH_SIZE --balance $PROPORTION