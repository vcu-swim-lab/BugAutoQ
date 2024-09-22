#!/bin/bash

SCRIPTS_DIR=script/models/
EMB_DIR=embeddings

DATA_DIR=data

RESULTS=$DATA_DIR/ranking.csv

DEVICE=cuda
CUDA_NO=0

python $SCRIPTS_DIR/main.py \
        --embeddings $EMB_DIR/vectors_pad.txt \
        --post-tsv $DATA_DIR/post_data.tsv \
        --train-ids $DATA_DIR/train_ids.txt \
        --test-ids $DATA_DIR/test_ids.txt \
        --qa-tsv $DATA_DIR/qa_data.tsv \
        --utility-tsv $DATA_DIR/utility_data.tsv \
        --output-ranking-file $RESULTS \
        --device $DEVICE \
        --cuda-no $CUDA_NO \
        --batch-size 256 \
        --n-epochs 10 \
        --max-p-len 300 \
        --max-q-len 100 \
        --max-a-len 100
