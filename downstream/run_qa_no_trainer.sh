OUTPUT_DIR=test
LEARNING_RATE=0.001
LR_SCHEDULE="linear"
msl=384
ds=128
DATASET=squad_v2

accelerate launch run_qa_no_trainer.py \
  --model_name_or_path "KETI-AIR/long-ke-t5-small" \
  --dataset_name $DATASET \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}


OUTPUT_DIR=test
LEARNING_RATE=0.001
LR_SCHEDULE="linear"
msl=384
ds=128
DATASET=squad_v2

accelerate launch run_qa_no_trainer.py \
  --model_name_or_path "KETI-AIR/ke-t5-small" \
  --dataset_name $DATASET \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --max_seq_length ${msl} \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride ${ds} \
  --learning_rate ${LEARNING_RATE} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --version_2_with_negative \
  --output_dir ${OUTPUT_DIR}/${DATASET}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}






# OUTPUT_DIR=test2
# LEARNING_RATE=0.001
# LR_SCHEDULE="linear"

# LR_SCHEDULE_LIST="linear constant"
# for LR_SCHEDULE in $LR_SCHEDULE_LIST
# do

# DS_LIST="squad squad_kor_v1 squad_kor_v2"
# for DATASET in $DS_LIST
# do
# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=1024
# ds=256
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=2048
# ds=512
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=4096
# ds=1024
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}
# done

# DATASET=squad_v2
# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=1024
# ds=256
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=2048
# ds=512
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=4096
# ds=1024
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}


# DS_LIST="KETI-AIR/aihub_news_mrc KETI-AIR/aihub_admin_docs_mrc"
# for DATASET in $DS_LIST
# do
# DATASET_CONFIG_NAME="squad.v2.like"
# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=1024
# ds=256
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=2048
# ds=512
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=4096
# ds=1024
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --version_2_with_negative \
#   --output_dir ${OUTPUT_DIR}/${DATASET}_${DATASET_CONFIG_NAME}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}
# done



# DS_LIST="KETI-AIR/aihub_news_mrc KETI-AIR/aihub_admin_docs_mrc"
# for DATASET in $DS_LIST
# do
# DATASET_CONFIG_NAME="squad.v1.like"
# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=1024
# ds=256
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=2048
# ds=512
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=4096
# ds=1024
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/lkt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=384
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}

# msl=512
# ds=128
# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name $DATASET \
#   --dataset_config_name ${DATASET_CONFIG_NAME} \
#   --per_device_train_batch_size 16 \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length ${msl} \
#   --hf_cache_dir "../huggingface_datasets" \
#   --hf_data_dir "../data/downstreams" \
#   --doc_stride ${ds} \
#   --learning_rate ${LEARNING_RATE} \
#   --lr_scheduler_type ${LR_SCHEDULE} \
#   --output_dir test/${DATASET}_${DATASET_CONFIG_NAME}/kt5_${msl}_${ds}_lr${LEARNING_RATE}_${LR_SCHEDULE}
# done

# done
