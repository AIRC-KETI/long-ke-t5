

# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/ke-t5-base" \
#   --dataset_name squad \
#   --max_seq_length 384 \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride 128 \
#   --output_dir test/squad


# accelerate launch run_qa_no_trainer.py \
#   --model_name_or_path "KETI-AIR/long-ke-t5-small" \
#   --dataset_name squad \
#   --per_device_train_batch_size 64 \
#   --per_device_eval_batch_size 64 \
#   --max_seq_length 384 \
#   --hf_cache_dir "../huggingface_datasets" \
#   --doc_stride 128 \
#   --output_dir test/squad_long


accelerate launch run_qa_no_trainer.py \
  --model_name_or_path "google/long-t5-tglobal-base" \
  --dataset_name squad \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --max_seq_length 1024 \
  --hf_cache_dir "../huggingface_datasets" \
  --doc_stride 256 \
  --output_dir test/squad_t5_1k
