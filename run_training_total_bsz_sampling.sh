

accelerate launch run_lm_no_trainer_total_bsz_sampling.py \
--config_name ./config/long-ke-t5-tglobal-small.json \
--tokenizer_name vocab/ko_en/spiece/vs64000 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--block_size 4096 \
--logging_steps 1000 \
--num_train_epochs 1 \
--checkpointing_steps 100000 \
--save_dataset_to_disk saved_data_sorted \
--gradient_accumulation_steps 4 \
--with_tracking \
--use_adafactor \
--find_unused_parameters \
--output_dir small_model_sorted


# --find_unused_parameters \ # when the sequence length is less than global block size(16), teh long-t5 doesn't use `global_relative_attention_bias`
