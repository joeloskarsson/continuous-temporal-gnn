#!/bin/sh

for ds in\
    bay_node_1.0\
    bay_node_0.75\
    bay_node_0.5\
    bay_node_0.25\
    bay_node_0.1\
    la_node_1.0\
    la_node_0.75\
    la_node_0.5\
    la_node_0.25\
    la_node_0.1
do
    python baselines.py --dataset $ds --baseline previous --test 1
done


