
HF_CACHE_DIR="../huggingface_datasets"
MAX_SOURCE_LEN=4096
MAX_TARGET_LEN=512
NUM_BEAMS=4
LR=0.001
PREPROCESSING_NUM_WORKERS=128

MODEL_NAME_OR_PATH="KETI-AIR/long-ke-t5-small"
OUTPUT_DIR="long-ke-t5-small-summarization"
PER_DEV_TRAIN_BATCH_SIZE=4
PER_DEV_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2


OUTPUT_DIR="long-ke-t5-base-summarization"
MODEL_NAME_OR_PATH="KETI-AIR/long-ke-t5-base"
PER_DEV_TRAIN_BATCH_SIZE=1
PER_DEV_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
EVAL_ACCUMULATION_STEPS=8


DATASET_NAME="KETI-AIR/aihub_summary_and_report"
OUTPUT_DIR_NAME=aihub_summary_and_report
HF_DATA_DIR="../data/downstreams"
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/nikl_summarization"
OUTPUT_DIR_NAME=nikl_summarization
HF_DATA_DIR="../data/downstreams"
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_book_summarization"
OUTPUT_DIR_NAME=aihub_book_summarization
HF_DATA_DIR="../data/downstreams"
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


# default config for KETI-AIR/aihub_dialog_summarization: roberta_prepended_single_punct (utterance sep token: </s>, add speaker's name, add punctuation for each line)
DATASET_NAME="KETI-AIR/aihub_dialog_summarization"
OUTPUT_DIR_NAME=aihub_dialog_summarization
HF_DATA_DIR="../data/downstreams"
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_document_summarization"
DATASET_CONFIG_NAME="law"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_document_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_document_summarization"
DATASET_CONFIG_NAME="magazine"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_document_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_document_summarization"
DATASET_CONFIG_NAME="news"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_document_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="paper_entire"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="paper_section"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="patent_entire"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="patent_section"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
python -m torch.distributed.launch --nproc_per_node 8 run_summarization.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --num_beams $NUM_BEAMS \
    --overwrite_output_dir \
    --predict_with_generate \
    --do_train --do_eval --evaluation_strategy "epoch" --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}
