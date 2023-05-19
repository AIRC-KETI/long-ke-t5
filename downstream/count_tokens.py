import os
import json
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="The path to cache directory for huggingface datasets.",
    )
    parser.add_argument(
        "--hf_data_dir",
        type=str,
        default=None,
        help="The path to data directory for huggingface datasets.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the target texts.",
    )
    parser.add_argument(
        "--is_translation",
        action="store_true",
        help="Count the translation datafeature type.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the result.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    raw_datasets = load_dataset(
            args.dataset_name, 
            args.dataset_config_name, 
            cache_dir=args.hf_cache_dir,
            data_dir=args.hf_data_dir
            )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    column_names = raw_datasets["train"].column_names
    input_column = args.input_column if args.input_column is not None else "text"
    target_column = args.target_column

    if args.is_translation:
        supported_languages = raw_datasets["train"].features['translation'].languages
        if input_column not in supported_languages:
            raise ValueError(
                    f"--input_column' value '{input_column}' needs to be one of: {', '.join(supported_languages)}"
                )
        if target_column is not None and target_column not in supported_languages:
            raise ValueError(
                    f"--target_column' value '{target_column}' needs to be one of: {', '.join(supported_languages)}"
                )
    else:
        if input_column not in column_names:
            raise ValueError(
                    f"--input_column' value '{input_column}' needs to be one of: {', '.join(column_names)}"
                )
        if target_column is not None and target_column not in column_names:
            raise ValueError(
                    f"--target_column' value '{target_column}' needs to be one of: {', '.join(column_names)}"
                )

    def tokenize_function(examples):
        if args.is_translation:
            output = tokenizer([ex[input_column] for ex in examples["translation"]])
            if target_column is not None:
                output["target_ids"] = tokenizer([ex[target_column] for ex in examples["translation"]]).input_ids
            return output

        output = tokenizer(examples[input_column])
        if target_column is not None:
            output["target_ids"] = tokenizer(examples[target_column]).input_ids
        return output

    processed_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    cnt_result = {}
    if target_column is not None:
        cnt_result_target = {}

    for k, ds in processed_datasets.items():
        _length = []
        if target_column is not None:
            _length_target = []
        
        for sample in tqdm(ds, desc="Count the {} split...".format(k)):
            _length.append(len(sample["input_ids"]))
            if target_column is not None:
                _length_target.append(len(sample["target_ids"]))
        cnt_result[k] = _length
        if target_column is not None:
            cnt_result_target[k] = _length_target

    def percentile(lst, per=50, is_sorted=True):
        if not is_sorted:
            lst = sorted(lst)
        lstLen = len(lst)
        if lstLen == 1:
            return lst[0]

        index = lstLen/100*per
        has_rem = (index - int(index) != 0)
        index = int(index)
    
        if not has_rem:
            return lst[index]
        elif index == lstLen - 1:
            return (lst[index] + lst[index - 1])/2.0
        else:
            return (lst[index] + lst[index + 1])/2.0

    os.makedirs(args.output_dir, exist_ok=True)

    len_stat = {
        input_column: {},
    }
    if target_column is not None:
        len_stat[target_column] = {}

    for k, _length in cnt_result.items():
        sorted_length = sorted(_length)
        len_stat[input_column][k] = {
            "average": sum(sorted_length)/len(sorted_length),
            "median": percentile(sorted_length, per=50),
            "max": sorted_length[-1],
            "90th percentile": percentile(sorted_length, per=90),
        }
    if target_column is not None:
        for k, _length in cnt_result_target.items():
            sorted_length = sorted(_length)
            len_stat[target_column][k] = {
                "average": sum(sorted_length)/len(sorted_length),
                "median": percentile(sorted_length, per=50),
                "max": sorted_length[-1],
                "90th percentile": percentile(sorted_length, per=90),
            }
    json.dump(len_stat, open(os.path.join(args.output_dir, "data_stat.json"), "w"), indent=4)


if __name__=="__main__":
    main()
