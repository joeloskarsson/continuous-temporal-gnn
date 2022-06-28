#!/bin/sh

for ds in bay la
do
    for on in 1.0 0.75 0.5 0.25 0.1
    do
        python preprocess_dataset.py --dataset $ds --obs_nodes $on
    done
done

