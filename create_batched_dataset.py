
import os
import argparse
import json

from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--load_from_saved_data",
        type=str,
        default="saved_data_sorted_vocab2",
        help="The path to saved data.",
    )
    parser.add_argument("--output_dir", type=str, default="batched_data_small", help="Where to store the final model.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    args = parser.parse_args()

    return args

MAX_SEQ_LEN = 4096
# BSZ_SEQ_LEN_LIST = [(192, 128), (32, 512), (16, 1024), (8, 2048), (3, MAX_SEQ_LEN)]
BSZ_SEQ_LEN_LIST = [(448, 128), (96, 512), (48, 1024), (22, 2048), (8, MAX_SEQ_LEN)]


def pad_func(values, pad_idx, max_length=MAX_SEQ_LEN):
    new_value = []
    for value in values:
        value = value[:max_length]
        remainder = [pad_idx] * (max_length - len(value))
        new_value.append(value + remainder)
    return new_value


def create_batch_range(dataset, bsz_seq_list, msl):
    total_len = len(dataset)
    offset = 0
    
    while offset < total_len:
        bsz_state = []
        for idx, (bsz, seq_len) in enumerate(bsz_seq_list):
            if offset+bsz < total_len:
                if len(dataset[offset+bsz]["input_ids"][:msl]) <= seq_len:
                    bsz_state.append(idx)
        if len(bsz_state) > 0:
            sbsz, sseq_len = bsz_seq_list[min(bsz_state)]
        else:
            sbsz = total_len - offset
            seq_len = msl
        
        batched_item = dataset[offset:offset+sbsz]
        batched_input_ids = pad_func(batched_item["input_ids"], 0, max_length=sseq_len)
        batched_attention_mask = pad_func(batched_item["attention_mask"], 0, max_length=sseq_len)
        batched_labels = pad_func(batched_item["labels"], 0, max_length=sseq_len//4)
        batched_decoder_attention_mask = pad_func(batched_item["decoder_attention_mask"], 0, max_length=sseq_len//4)
        
        yield {
            "input_ids" : batched_input_ids,
            "attention_mask" : batched_attention_mask,
            "labels" : batched_labels,
            "decoder_attention_mask" : batched_decoder_attention_mask,
        }, offset, total_len
        offset += sbsz



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_datasets = load_from_disk(
                args.load_from_saved_data
            )
    train_datasets = raw_datasets["train"]
    validation_datasets = raw_datasets["validation"]


    with open(os.path.join(args.output_dir, "train.jsonl"), "w") as f:
        for batch, offset, total_len in create_batch_range(train_datasets, BSZ_SEQ_LEN_LIST, MAX_SEQ_LEN):
            f.write(json.dumps(batch)+"\n")
            print("train[{}/{}]".format(offset, total_len))
    
    with open(os.path.join(args.output_dir, "validation.jsonl"), "w") as f:
        for batch, offset, total_len in create_batch_range(validation_datasets, BSZ_SEQ_LEN_LIST, MAX_SEQ_LEN):
            f.write(json.dumps(batch)+"\n")
            print("validation[{}/{}]".format(offset, total_len))



if __name__=="__main__":
    main()
    