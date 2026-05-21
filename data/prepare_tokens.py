from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


TOKENIZER = None
DATASET = None
ARGS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare token shards for LLaMA pretraining")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-documents", type=int, default=0, help="0 means use the full split")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def init_worker(args: argparse.Namespace) -> None:
    global TOKENIZER, DATASET, ARGS
    TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer)
    DATASET = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    ARGS = args


def tokenize_worker(worker_idx: int) -> np.ndarray:
    global TOKENIZER, DATASET, ARGS

    eos_token_id = TOKENIZER.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")

    total_docs = len(DATASET) if ARGS.num_documents <= 0 else min(len(DATASET), ARGS.num_documents)
    shards = []
    iterator = range(worker_idx, total_docs, ARGS.num_workers)
    if worker_idx == 0:
        iterator = tqdm(iterator, total=(total_docs + ARGS.num_workers - 1) // ARGS.num_workers)

    for doc_idx in iterator:
        if doc_idx % ARGS.num_shards != ARGS.shard_rank:
            continue
        text = DATASET[doc_idx][ARGS.text_column]
        tokens = TOKENIZER.encode(text, add_special_tokens=False)
        shards.append(np.asarray(tokens + [eos_token_id], dtype=np.uint32))

    if not shards:
        return np.empty(0, dtype=np.uint32)
    return np.concatenate(shards)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.num_workers, initializer=init_worker, initargs=(args,)) as pool:
        chunks = list(pool.imap(tokenize_worker, range(args.num_workers)))

    output = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint32)
    output_path = os.path.join(args.output_dir, f"tokens_{args.shard_rank}.bin")
    output.tofile(output_path)
    print(f"Wrote {output.shape[0]} tokens to {output_path}")


if __name__ == "__main__":
    main()
