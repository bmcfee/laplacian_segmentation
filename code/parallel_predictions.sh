#!/bin/bash

NUM_JOBS=10


predictor() {
    dataset=$1
    algo=$2
    file=$3
    args=${@:4}

    prediction="$(basename "$file" .pickle).lab"

    ./feature_segmenter.py $args "${file}" \
                            "../data/predictions/${dataset}/${algo}/${prediction}" 
}
export -f predictor

for data in BEATLES_TUT SALAMI 
    do
        for m in $(seq 2 10)
            do
                parallel -j ${NUM_JOBS} predictor $data laplacian_$m {1} -m $m ::: ../data/features/$data/*.pickle
            done
    done

# parallel -j ${NUM_JOBS} predictor BEATLES_TUT laplacian {1} ::: ../data/features/BEATLES_TUT/*.pickle
#parallel -j ${NUM_JOBS} predictor SALAMI laplacian {1} ::: ../data/features/SALAMI/*.pickle

