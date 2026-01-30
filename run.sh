#!/bin/bash

# mvals=(10 100 200)
mvals=(200)
# kvals=(5 4 3 2 1)
kvals=(3 2 1)
# nvals=($(for i in {11..18}; do echo $((2**i)); done))
nvals=($(for i in {11..16}; do echo $((2**i)); done))

for m in "${mvals[@]}"; do
    for k in "${kvals[@]}"; do
        for n in "${nvals[@]}"; do
            python profile_async.py --n "$n" --m "$m" --k "$k"
            exit_status=$?
            if [ $exit_status -ne 0 ]; then
                break
            fi
        done
    done
done

for m in "${mvals[@]}"; do
    for n in "${nvals[@]}"; do
        python profile_async.py --n "$n" --m "$m" --k "1" --force_sparse
        exit_status=$?
        if [ $exit_status -ne 0 ]; then
            break
        fi
    done
done
