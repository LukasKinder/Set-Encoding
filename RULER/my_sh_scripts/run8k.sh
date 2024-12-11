#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=4090

cd ../../../..
source venv/bin/activate
cd NeedleInHaystack/RULER/scripts

# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="Llama-3-8b-set-1k" # the path that stores generated task samples and model predictions.
MODEL_DIR="../.." # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization

BENCHMARK="synthetic"
TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

SEQ_LENGTHS=(
    8192
)

NUM_SAMPLES=10

TOKENIZER_PATH="meta-llama/Meta-Llama-3-8B-instruct"
MODEL_PATH="meta-llama/Meta-Llama-3-8B-instruct"
TOKENIZER_TYPE="hf"
MODEL_TEMPLATE_TYPE="meta-llama3"
MODEL_FRAMEWORK="hf"


for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        echo "MAX_SEQUECE_LENGTH = ${MAX_SEQ_LENGTH}; TASK = ${TASK}"
        python data/my_prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES}

        python pred/run_my_model.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            ${STOP_WORDS}

    done
    
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done