

MODEL_NAME_OR_PATH="KETI-AIR/ke-t5-small"
OUTPUT_DIR="ke-t5-small-summarization_lre-3"
HF_CACHE_DIR="../huggingface_datasets"
MAX_SOURCE_LEN=512
MAX_TARGET_LEN=512
NUM_BEAMS=4
PER_DEV_TRAIN_BATCH_SIZE=4
PER_DEV_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
LR=0.001

DATASET_NAME=cnn_dailymail
DATASET_CONFIG_NAME="3.0.0"
OUTPUT_DIR_NAME=cnn_dailymail
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME=big_patent
OUTPUT_DIR_NAME=big_patent
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME=multi_news
OUTPUT_DIR_NAME=multi_news
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="ccdv/pubmed-summarization"
OUTPUT_DIR_NAME=pubmed-summarization
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="ccdv/arxiv-summarization"
OUTPUT_DIR_NAME=arxiv-summarization
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


# default config for mediasum: roberta_prepended (utterance sep token: </s>, add speaker's name)
DATASET_NAME="ccdv/mediasum"
OUTPUT_DIR_NAME=mediasum
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


# default config for mediasum: roberta_prepended (utterance sep token: </s>, add speaker's name)
DATASET_NAME="ccdv/WCEP-10"
OUTPUT_DIR_NAME=WCEP-10
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_summary_and_report"
OUTPUT_DIR_NAME=aihub_summary_and_report
HF_DATA_DIR="../data/downstreams"
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/nikl_summarization"
OUTPUT_DIR_NAME=nikl_summarization
HF_DATA_DIR="../data/downstreams"
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_book_summarization"
OUTPUT_DIR_NAME=aihub_book_summarization
HF_DATA_DIR="../data/downstreams"
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


# default config for KETI-AIR/aihub_dialog_summarization: roberta_prepended_single_punct (utterance sep token: </s>, add speaker's name, add punctuation for each line)
DATASET_NAME="KETI-AIR/aihub_dialog_summarization"
OUTPUT_DIR_NAME=aihub_dialog_summarization
HF_DATA_DIR="../data/downstreams"
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_document_summarization"
DATASET_CONFIG_NAME="law"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_document_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_document_summarization"
DATASET_CONFIG_NAME="magazine"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_document_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_document_summarization"
DATASET_CONFIG_NAME="news"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_document_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}


DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="paper_entire"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="paper_section"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="patent_entire"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}

DATASET_NAME="KETI-AIR/aihub_paper_summarization"
DATASET_CONFIG_NAME="patent_section"
HF_DATA_DIR="../data/downstreams"
OUTPUT_DIR_NAME=aihub_paper_summarization/$DATASET_CONFIG_NAME
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --hf_data_dir $HF_DATA_DIR \
    --source_prefix "summarize: " \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --val_max_target_length $MAX_TARGET_LEN \
    --num_beams $NUM_BEAMS \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME}
