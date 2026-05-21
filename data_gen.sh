for r in 0 1 2 3 4 5 6 7; do
  python data/prepare_tokens.py \
    --tokenizer lmsys/vicuna-7b-v1.5 \
    --dataset-name allenai/c4 \
    --dataset-config en \
    --split train \
    --shard-rank $r \
    --num-shards 8 \
    --num-documents 10000000 \
    --num-workers 8 \
    --output-dir data/C4
done