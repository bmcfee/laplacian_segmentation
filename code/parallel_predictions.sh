#!/bin/bash

NUM_JOBS=10


predictor() {
    dataset=$1
    algo=$2
    file=$3
    prediction="$(basename "$file" .pickle).lab"

    ./feature_segmenter.py "${file}" \
                            "../data/predictions/${dataset}/${algo}/${prediction}" 
}
export -f predictor

parallel -j ${NUM_JOBS} predictor BEATLES_TUT laplacian {1} ::: ../data/features/BEATLES_TUT/*.pickle
parallel -j ${NUM_JOBS} predictor SALAMI laplacian {1} ::: ../data/features/SALAMI/*.pickle

