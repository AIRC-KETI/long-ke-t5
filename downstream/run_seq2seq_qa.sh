
# DATASET="squad"
# msl=384
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --predict_with_generate \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 1e-3 \
#   --num_train_epochs 3 \
#   --num_beams 4 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_t5_${msl}_${ds}


DATASET="squad_v2"
msl=384
ds=128
python run_seq2seq_qa.py \
  --model_name_or_path "t5-small" \
  --dataset_name ${DATASET} \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --hf_cache_dir ../huggingface_datasets \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-3 \
  --num_train_epochs 3 \
  --num_beams 4 \
  --max_seq_length ${msl} \
  --doc_stride ${ds} \
  --unanswerable_text "unanswerable" \
  --version_2_with_negative \
  --output_dir test/${DATASET}_seq2seq_t5_${msl}_${ds}_imp


# DATASET="squad"
# msl=384
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --predict_with_generate \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 1e-3 \
#   --num_train_epochs 3 \
#   --num_beams 4 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_kt5_${msl}_${ds}


DATASET="squad_v2"
msl=384
ds=128
python run_seq2seq_qa.py \
  --model_name_or_path "KETI-AIR/ke-t5-small" \
  --dataset_name ${DATASET} \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --hf_cache_dir ../huggingface_datasets \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-3 \
  --num_train_epochs 3 \
  --num_beams 4 \
  --max_seq_length ${msl} \
  --doc_stride ${ds} \
  --unanswerable_text "unanswerable" \
  --version_2_with_negative \
  --output_dir test/${DATASET}_seq2seq_kt5_${msl}_${ds}_imp




# DATASET="squad_v2"

# msl=384
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --predict_with_generate \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 1e-3 \
#   --num_train_epochs 3 \
#   --num_beams 4 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --version_2_with_negative \
#   --output_dir test/${DATASET}_seq2seq_kt5_${msl}_${ds}

# msl=512
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --version_2_with_negative \
#   --output_dir test/${DATASET}_seq2seq_kt5_${msl}_${ds}

# msl=384
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --version_2_with_negative \
#   --output_dir test/${DATASET}_seq2seq_lkt5_${msl}_${ds}

# msl=512
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --version_2_with_negative \
#   --output_dir test/${DATASET}_seq2seq_lkt5_${msl}_${ds}

# msl=1024
# ds=512
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --version_2_with_negative \
#   --output_dir test/${DATASET}_seq2seq_lkt5_${msl}_${ds}




# DATASET="squad"

# msl=384
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_kt5_${msl}_${ds}

# msl=512
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_kt5_${msl}_${ds}

# msl=384
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_lkt5_${msl}_${ds}

# msl=512
# ds=128
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_lkt5_${msl}_${ds}

# msl=1024
# ds=512
# python run_seq2seq_qa.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name ${DATASET} \
#   --context_column context \
#   --question_column question \
#   --answer_column answers \
#   --do_train \
#   --do_eval \
#   --hf_cache_dir ../huggingface_datasets \
#   --per_device_train_batch_size 32 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length ${msl} \
#   --doc_stride ${ds} \
#   --output_dir test/${DATASET}_seq2seq_lkt5_${msl}_${ds}





