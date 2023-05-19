
# MODEL_NAME_OR_PATH="KETI-AIR/ke-t5-small"
# OUTPUT_DIR="ke-t5-small-seq_cls_lre-3"

MODEL_NAME_OR_PATH="kimsan0622/long-ke-t5-small-b"
OUTPUT_DIR="long-ke-t5-small-2-seq_cls_lre-3"
PER_DEV_TRAIN_BATCH_SIZE=8
PER_DEV_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1


HF_CACHE_DIR="../huggingface_datasets"
MAX_SEQ_LEN=512
LR=0.00001
NUM_TRAIN_EPOCHS=50
SUMMARY_TYPE="attn"
LR_SCHEDULER_TYPE="constant"


# DATASET_NAME="glue"
# DATASET_CONFIG_NAME="cola"
# OUTPUT_DIR_NAME=${DATASET_NAME}_${DATASET_CONFIG_NAME}
# accelerate launch run_seq_classification_no_trainer.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --dataset_name ${DATASET_NAME} \
#     --dataset_config_name ${DATASET_CONFIG_NAME} \
#     --hf_cache_dir $HF_CACHE_DIR \
#     --max_length $MAX_SEQ_LEN \
#     --learning_rate $LR \
#     --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
#     --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME} \
#     --summary_type ${SUMMARY_TYPE} \
#     --num_train_epochs ${NUM_TRAIN_EPOCHS} \
#     --find_unused_parameters \
#     --with_tracking


DATASET_NAME="klue"
DATASET_CONFIG_NAME="sts"
OUTPUT_DIR_NAME=${DATASET_NAME}_${DATASET_CONFIG_NAME}
accelerate launch run_seq_classification_no_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --hf_cache_dir $HF_CACHE_DIR \
    --max_length $MAX_SEQ_LEN \
    --learning_rate $LR \
    --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME} \
    --summary_type ${SUMMARY_TYPE} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --find_unused_parameters \
    --with_tracking


# DATASET_NAME="glue"
# DS_LIST="cola mnli mrpc qnli qqp rte sst2 stsb wnli"
# for DATASET_CONFIG_NAME in $DS_LIST
# do
# OUTPUT_DIR_NAME=${DATASET_NAME}_${DATASET_CONFIG_NAME}
# accelerate launch run_seq_classification_no_trainer.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --dataset_name ${DATASET_NAME} \
#     --dataset_config_name ${DATASET_CONFIG_NAME} \
#     --hf_cache_dir $HF_CACHE_DIR \
#     --max_length $MAX_SEQ_LEN \
#     --learning_rate $LR \
#     --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
#     --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME} \
#     --summary_type ${SUMMARY_TYPE} \
#     --num_train_epochs ${NUM_TRAIN_EPOCHS} \
#     --find_unused_parameters \
#     --with_tracking
# done


# DATASET_NAME="KETI-AIR/kor_corpora"
# DS_LIST="khsd.split kornli.split korsts.split nsmc.split qpair.split"
# for DATASET_CONFIG_NAME in $DS_LIST
# do
# OUTPUT_DIR_NAME=${DATASET_NAME}_${DATASET_CONFIG_NAME}
# accelerate launch run_seq_classification_no_trainer.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --dataset_name ${DATASET_NAME} \
#     --dataset_config_name ${DATASET_CONFIG_NAME} \
#     --hf_cache_dir $HF_CACHE_DIR \
#     --max_length $MAX_SEQ_LEN \
#     --learning_rate $LR \
#     --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
#     --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME} \
#     --summary_type ${SUMMARY_TYPE} \
#     --num_train_epochs ${NUM_TRAIN_EPOCHS} \
#     --find_unused_parameters \
#     --eval_on_test \
#     --with_tracking
# done


# DATASET_NAME="klue"
# DS_LIST="nli sts ynat"
# for DATASET_CONFIG_NAME in $DS_LIST
# do
# OUTPUT_DIR_NAME=${DATASET_NAME}_${DATASET_CONFIG_NAME}
# accelerate launch run_seq_classification_no_trainer.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --dataset_name ${DATASET_NAME} \
#     --dataset_config_name ${DATASET_CONFIG_NAME} \
#     --hf_cache_dir $HF_CACHE_DIR \
#     --max_length $MAX_SEQ_LEN \
#     --learning_rate $LR \
#     --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
#     --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --output_dir $OUTPUT_DIR/${OUTPUT_DIR_NAME} \
#     --summary_type ${SUMMARY_TYPE} \
#     --num_train_epochs ${NUM_TRAIN_EPOCHS} \
#     --find_unused_parameters \
#     --with_tracking
# done
