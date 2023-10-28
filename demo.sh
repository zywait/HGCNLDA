#!/usr/bin/env bash

dataset="dataset1"
model="HGCNLDA"
comment="save_dir"
epochs=300
neighbor_num=5
lr=1e-2
embedding_dim=128

python main.py --model ${model} --dataset_name ${dataset} --comment ${comment} \
--epochs ${epochs} --lnc_neighbor_num ${neighbor_num} --disease_neighbor_num ${neighbor_num} \
--mi_neighbor_num ${neighbor_num} --embedding_dim ${embedding_dim} --lr ${lr}