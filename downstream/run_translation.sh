
HF_CACHE_DIR="../huggingface_datasets"
HF_DATA_DIR="../data/downstreams"
MAX_SOURCE_LEN=512
MAX_TARGET_LEN=512
NUM_BEAMS=4
PREPROCESSING_NUM_WORKERS=128
LR=0.001
SCHEDULER_TYPE="linear"
EPOCHS=3


MODEL_NAME_OR_PATH="kimsan0622/long-ke-t5-small-2"
OUTPUT_DIR="long-ke-t5-small-2-translation_lre-3"
PER_DEV_TRAIN_BATCH_SIZE=32
PER_DEV_EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2

OUTPUT_DIR="long-ke-t5-base-translation_lre-3"
MODEL_NAME_OR_PATH="kimsan0622/long-ke-t5-base"
PER_DEV_TRAIN_BATCH_SIZE=8
PER_DEV_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=8


# ko2en
# SOURCE_LANG="ko"
# TARGET_LANG="en"
# SOURCE_PREFIX="translate_ko2en: "

# DATASET_NAME="KETI-AIR/aihub_koenzh_food_translation"
# OUTPUT_DIR_NAME=aihub_koenzh_food_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_scitech_translation"
# OUTPUT_DIR_NAME=aihub_scitech_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_scitech20_translation"
# OUTPUT_DIR_NAME=aihub_scitech20_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_socialtech20_translation"
# OUTPUT_DIR_NAME=aihub_socialtech20_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_spoken_language_translation"
# OUTPUT_DIR_NAME=aihub_spoken_language_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}


# en2ko
SOURCE_LANG="en"
TARGET_LANG="ko"
SOURCE_PREFIX="translate_en2ko: "

# DATASET_NAME="KETI-AIR/aihub_koenzh_food_translation"
# OUTPUT_DIR_NAME=aihub_koenzh_food_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_scitech_translation"
# OUTPUT_DIR_NAME=aihub_scitech_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_scitech20_translation"
# OUTPUT_DIR_NAME=aihub_scitech20_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

# DATASET_NAME="KETI-AIR/aihub_socialtech20_translation"
# OUTPUT_DIR_NAME=aihub_socialtech20_translation
# accelerate launch run_translation_no_trainer.py \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --source_lang ${SOURCE_LANG} \
# --target_lang ${TARGET_LANG} \
# --source_prefix ${SOURCE_PREFIX} \
# --max_source_length ${MAX_SOURCE_LEN} \
# --max_target_length ${MAX_TARGET_LEN} \
# --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
# --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
# --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
# --lr_scheduler_type ${SCHEDULER_TYPE} \
# --learning_rate ${LR} \
# --num_train_epochs ${EPOCHS} \
# --num_beams ${NUM_BEAMS} \
# --eval_on_last_epoch \
# --output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}

DATASET_NAME="KETI-AIR/aihub_spoken_language_translation"
OUTPUT_DIR_NAME=aihub_spoken_language_translation
accelerate launch run_translation_no_trainer.py \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--dataset_name ${DATASET_NAME} \
--hf_cache_dir ${HF_CACHE_DIR} \
--hf_data_dir ${HF_DATA_DIR} \
--source_lang ${SOURCE_LANG} \
--target_lang ${TARGET_LANG} \
--source_prefix ${SOURCE_PREFIX} \
--max_source_length ${MAX_SOURCE_LEN} \
--max_target_length ${MAX_TARGET_LEN} \
--preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
--per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
--lr_scheduler_type ${SCHEDULER_TYPE} \
--learning_rate ${LR} \
--num_train_epochs ${EPOCHS} \
--num_beams ${NUM_BEAMS} \
--eval_on_last_epoch \
--output_dir ${OUTPUT_DIR}/${OUTPUT_DIR_NAME}/${SOURCE_LANG}2${TARGET_LANG}
