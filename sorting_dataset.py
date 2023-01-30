import os
import argparse
import functools

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, T5Tokenizer
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, load_from_disk


def calc_input_len(examples):
    input_length = [len(ex) for ex in examples["input_ids"]]
    examples["input_length"] = input_length
    return examples

def parse_args():
    parser = argparse.ArgumentParser(description="Sorting data along the length of input_ids")
    parser.add_argument(
        "--input_saved_dataset",
        type=str,
        default="saved_data_vocab2",
        help="The path to saved dataset that needs to be sorted.",
    )
    parser.add_argument(
        "--output_saved_dataset",
        type=str,
        default="saved_data_sorted_vocab2",
        help="The path to output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="The number of workers for preprocessing.",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()


    dataset = load_from_disk(args.input_saved_dataset)
    dataset = dataset.map(
                    calc_input_len,
                    batched=True,
                    num_proc=args.workers,
                    desc="add length of input_ids on dataset",
                )
    sorted_dataset = dataset.sort("input_length")
    sorted_dataset = sorted_dataset.remove_columns("input_length")
    sorted_dataset.save_to_disk(args.output_saved_dataset)



if __name__=="__main__":
    main()
