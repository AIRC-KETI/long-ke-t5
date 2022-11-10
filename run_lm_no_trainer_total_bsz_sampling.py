#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
import functools
from itertools import chain
from pathlib import Path
from datetime import timedelta

import datasets
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, DistributedDataParallelKwargs
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    DPTForDepthEstimation,
    LongT5ForConditionalGeneration,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from transformers.optimization import Adafactor
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from typing import Optional, Sized
import torch.distributed as dist
from torch.utils.data import Sampler, BatchSampler, Dataset

def split_n_parts(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

class RandomTotalBSZSampler(Sampler):
    def __init__(
            self, 
            data_source: Optional[Sized],
            batch_size: int,
            num_processes: int,
            shuffle: bool=True,
        ) -> None:
        super().__init__(data_source)
        self.data_source = data_source

        self.num_processes = num_processes
        self.batch_size = batch_size
        self.total_batch_size = batch_size * num_processes
        self.shuffle = shuffle

        self.total_size = len(self.data_source)
        self.num_total_batches = self.total_size // self.total_batch_size
        self.num_total_batch_idx = self.num_total_batches * self.total_batch_size

    def __iter__(self):

        range_total = list(range(self.total_size))
        range_total_chunked = [
            range_total[i:i + self.total_batch_size] 
                for i in range(
                    0, 
                    self.total_size, 
                    self.total_batch_size
                    )
            ]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            batch_indices = torch.randperm(self.num_total_batches, generator=g).tolist()  # type: ignore[arg-type]
        else:
            batch_indices = list(range(self.num_total_batches))  # type: ignore[arg-type]

        sample_indices = [sample for idx in batch_indices for sample in range_total_chunked[idx]]
        if self.total_size % self.batch_size != 0:
            sample_indices += range_total[self.num_total_batch_idx:]

        return iter(sample_indices)

    def __len__(self) -> int:
        return self.total_size

class RandomBatchSampler(Sampler):
    def __init__(
        self, 
        dataset: Dataset, 
        batch_size:int,  # batch per process
        num_processes: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False) -> None:

        self.dataset = dataset
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.total_batch_size = batch_size * num_processes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.total_size = len(dataset)

        # calc number of bathces
        if self.total_size % self.total_batch_size == 0:
            self.num_total_batches = self.total_size // self.total_batch_size
        else:
            length = self.total_size // self.total_batch_size
            self.num_total_batches =  length if self.drop_last else length + 1

        self.num_batches = self.num_total_batches * num_processes

    def __iter__(self):

        range_total = list(range(self.total_size))
        
        _remainder = self.total_size%self.total_batch_size
        padding_size = self.total_batch_size - _remainder
        if not self.drop_last:
            range_total += range_total[-padding_size:]
        else:
            range_total = range_total[:-_remainder]

        range_total_chunked = [
            range_total[i:i + self.total_batch_size] 
                for i in range(
                    0, 
                    self.total_size, 
                    self.total_batch_size
                    )
            ]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            batch_indices = torch.randperm(self.num_total_batches, generator=g).tolist()  # type: ignore[arg-type]
        else:
            batch_indices = list(range(self.num_total_batches))  # type: ignore[arg-type]

        
        assert len(batch_indices) == len(range_total_chunked), f"len(batch_indices):[{len(batch_indices)}] == len(range_total_chunked):[{len(range_total_chunked)}]"
        assert self.num_total_batches == len(range_total_chunked), f"self.num_total_batches:[{self.num_total_batches}] == len(range_total_chunked):[{len(range_total_chunked)}]"

        batched_indices = [batch for idx in batch_indices for batch in list(split_n_parts(range_total_chunked[idx], self.num_processes))]

        return iter(batched_indices)

    def __len__(self) -> int:
        return self.num_batches


logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# accelerate launch run_lm_no_trainer.py

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="psg_datasets.py",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="base",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default="huggingface_datasets",
        help="The path to cache directory for huggingface datasets.",
    )
    parser.add_argument(
        "--hf_data_dir1",
        type=str,
        default="data/sent_score/en_subset",
        help="The path to data directory for huggingface datasets.",
    )
    parser.add_argument(
        "--hf_data_dir2",
        type=str,
        default="data/sent_score/ko_corpora",
        help="The path to data directory for huggingface datasets.",
    )
    parser.add_argument(
        "--keep_in_memory",
        action="store_true",
        help="Whether to copy the dataset in-memory.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=1,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--gsg_type",
        type=str,
        default="ind-uniq",
        help="A type of priciple sentence selection",
        choices=["lead", "random", "ind-orig", "ind-uniq", "seq-orig", "seq-uniq"]
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument("--save_dataset_to_disk", type=str, default=None, help="Where to store the preprocessed datasets")


    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="google/long-t5-tglobal-base",
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="vocab/ko_en/spiece/vs64000",
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_portion", type=float, default=0.02, help="Portion of total training steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="test", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=4096,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=256,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=0, help="Number of steps for logging (stdout)."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help="Whether to use adafactor optimizer.",
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Whether to find unused parameters",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    kwargs_handlers = [
            InitProcessGroupKwargs(timeout=timedelta(days=10))
        ]
    
    if args.find_unused_parameters:
        kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        kwargs_handlers=kwargs_handlers , **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_path:
        config = AutoConfig.from_pretrained(args.model_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if len(tokenizer.additional_special_tokens) < 1:
        logger.warning("There is no 'additional_special_tokens'. The special token '<mask2>' is added.")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mask2>"]})

    if args.model_path:
        model = LongT5ForConditionalGeneration.from_pretrained(
            args.model_path
        )
    else:
        logger.info("Training new model from scratch")
        model = LongT5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))


    # load dataset
    raw_datasets_1st = load_from_disk(
            args.save_dataset_to_disk,
            keep_in_memory=args.keep_in_memory,
        )

    column_names = raw_datasets_1st["train"].column_names
    if "input_length" in column_names:
        raw_datasets_1st = raw_datasets_1st.remove_columns(["input_length"])
        
    train_dataset = raw_datasets_1st["train"]
    eval_dataset = raw_datasets_1st["validation"]

    # DataLoaders creation:
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=tokenizer.pad_token_id,
    #     pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    # )

    # collate tokens func
    def collate_tokens(values, pad_idx, max_length=None, pad_to_multiple_of=None):
        if max_length is None:
            max_length = max(len(l) for l in values)
        if pad_to_multiple_of is not None:
            max_length = (
                (max_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )
        new_value = []
        for value in values:
            remainder = [pad_idx] * (max_length - len(value))
            new_value.append(value + remainder)
        return new_value
    
    def dynamic_collate_fn(samples, pad_id=0, max_length=None, padding_max_length=None, pad_to_multiple_of=None):
        if len(samples) == 0:
            return {}
        decoder_max_length = max_length
        if decoder_max_length is not None:
            decoder_max_length = decoder_max_length // 4
        
        decoder_padding_max_length = padding_max_length
        if decoder_padding_max_length is not None:
            decoder_padding_max_length = decoder_padding_max_length // 4
        
        input_ids = [s["input_ids"] if max_length is None else s["input_ids"][:max_length] for s in samples]
        attention_mask = [s["attention_mask"] if max_length is None else s["attention_mask"][:max_length] for s in samples]
        labels = [s["labels"] if decoder_max_length is None else s["labels"][:decoder_max_length] for s in samples]
        decoder_attention_mask = [s["decoder_attention_mask"] if decoder_max_length is None else s["decoder_attention_mask"][:decoder_max_length] for s in samples]
        return {
            "input_ids": torch.tensor(collate_tokens(input_ids, pad_id, max_length=padding_max_length, pad_to_multiple_of=pad_to_multiple_of)),
            "attention_mask": torch.tensor(collate_tokens(attention_mask, 0, max_length=padding_max_length, pad_to_multiple_of=pad_to_multiple_of)),
            "labels": torch.tensor(collate_tokens(labels, pad_id, max_length=decoder_padding_max_length, pad_to_multiple_of=pad_to_multiple_of)),
            "decoder_attention_mask": torch.tensor(collate_tokens(decoder_attention_mask, 0, max_length=decoder_padding_max_length, pad_to_multiple_of=pad_to_multiple_of)),
        }
        
    data_collator = functools.partial(
        dynamic_collate_fn, 
        pad_id=tokenizer.pad_token_id,
        max_length=args.block_size,
        padding_max_length=args.block_size if args.pad_to_max_length else None,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        item = train_dataset[index]
        logger.info(f"Sample {index} of the training set: {item}.")
        logger.info("input_ids: {}\n".format(tokenizer.decode(item["input_ids"])))
        logger.info("labels: {}\n".format(tokenizer.decode(item["labels"])))

    # create total batch size level random sampler for train_dataloader
    # Random sampling in units of total batch size.
    # train_batchsampler = RandomBatchSampler(
    #         train_dataset, 
    #         batch_size=args.per_device_train_batch_size,
    #         num_processes=accelerator.num_processes, 
    #     )
    train_sampler = RandomTotalBSZSampler(
        train_dataset, 
            batch_size=args.per_device_train_batch_size,
            num_processes=accelerator.num_processes, 
    )
    train_batchsampler = BatchSampler(train_sampler, batch_size=args.per_device_train_batch_size, drop_last=False)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_batchsampler, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_adafactor:
        optimizer = Adafactor(
            optimizer_grouped_parameters, 
            scale_parameter=True, 
            relative_step=True, 
            warmup_init=True, 
            lr=None
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.warmup_portion > 0:
        args.num_warmup_steps = int(args.max_train_steps/max(min(args.warmup_portion, 1), 0))

    if args.use_adafactor:
        lr_scheduler = None
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch
    perplexity=0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            
            if (args.logging_steps>0 and completed_steps % args.logging_steps == 0 and completed_steps > 0) and args.with_tracking and step % args.gradient_accumulation_steps == 0:
                mean_loss = total_loss.item()/(step+1)
                perplexity = math.exp(mean_loss)
                logger.info("train_loss: {:.3f}, perplexity: {:.3f}".format(mean_loss, perplexity))

            if completed_steps >= args.max_train_steps:
                break
        
        if isinstance(checkpointing_steps, int):
            if accelerator.sync_gradients:
                output_dir = f"step_{completed_steps }"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
