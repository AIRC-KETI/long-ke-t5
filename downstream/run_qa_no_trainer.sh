
OUTPUT_DIR="long-ke-t5-small-2-qa_lre-3"
MODEL_NAME_OR_PATH="kimsan0622/long-ke-t5-small-2"
PER_DEV_TRAIN_BATCH_SIZE=16
PER_DEV_EVAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1

OUTPUT_DIR="long-ke-t5-base-qa_lre-3"
MODEL_NAME_OR_PATH="kimsan0622/long-ke-t5-base"
PER_DEV_TRAIN_BATCH_SIZE=4
PER_DEV_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4


LEARNING_RATE=0.001
LR_SCHEDULE="linear"
LR_SCHEDULE_LIST="linear constant"


DS_LIST="squad squad_kor_v1 squad_kor_v2"
for DATASET in $DS_LIST
do
msl=384
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=512
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=1024
ds=256
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=2048
ds=512
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=4096
ds=1024
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

done

DATASET=squad_v2
msl=384
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=512
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=1024
ds=256
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=2048
ds=512
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=4096
ds=1024
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}




DS_LIST="KETI-AIR/aihub_news_mrc KETI-AIR/aihub_admin_docs_mrc"
for DATASET in $DS_LIST
do
DATASET_CONFIG_NAME="squad.v2.like"
msl=384
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=512
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=1024
ds=256
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=2048
ds=512
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=4096
ds=1024
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

done



DS_LIST="KETI-AIR/aihub_news_mrc KETI-AIR/aihub_admin_docs_mrc"
for DATASET in $DS_LIST
do
DATASET_CONFIG_NAME="squad.v1.like"
msl=384
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=512
ds=128
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=1024
ds=256
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=2048
ds=512
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

msl=4096
ds=1024
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name $DATASET \
  --dataset_config_name ${DATASET_CONFIG_NAME} \
  --per_device_train_batch_size $PER_DEV_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEV_EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --hf_data_dir "../data/downstreams" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

done

