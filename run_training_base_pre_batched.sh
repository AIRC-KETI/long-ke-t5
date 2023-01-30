accelerate launch run_lm_no_trainer_batched.py \
--config_name ./config/long-ke-t5-tglobal-base.json \
--tokenizer_name vocab/ko_en/spiece/ko20000vs64000 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--block_size 4096 \
--logging_steps 1000 \
--num_train_epochs 5 \
--checkpointing_steps 30000 \
--load_dataset_from_disk saved_batched_data \
--gradient_accumulation_steps 2 \
--with_tracking \
--use_adafactor \
--find_unused_parameters \
--resume_from_checkpoint base_model_batched/step_570000 \
--output_dir base_model_batched

# --find_unused_parameters \