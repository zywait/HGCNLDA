#!/usr/bin/env bash

emb_dim_array=(32 64 128)
neighbor_num_array=(3 5 10)
alpha_array=(0.25 0.5 0.75)
gamma_array=(0 1 2 5)

dataset_array=( "dataset1" "dataset2")
model="HGCNLDA_fc"
comment="focal_loss"
lr=1e-2

for dataset in ${dataset_array[*]};do
  for alpha in ${alpha_array[*]};do
    for gamma in ${gamma_array[*]};do
      for neighbor_num in ${neighbor_num_array[*]};do
        for embedding_dim in ${emb_dim_array[*]};do
          if [ $dataset == "dataset2" ]
          then
              epochs=150
          fi
          python main.py --model ${model} --dataset_name ${dataset} --comment "E:/${comment}/${dataset}/alpha=${alpha}gamma=${gamma}_neighbor_num=${neighbor_num}_embedding_dim=${embedding_dim}" \
          --epochs ${epochs} --lnc_neighbor_num ${neighbor_num} --disease_neighbor_num ${neighbor_num} \
          --mi_neighbor_num ${neighbor_num} --embedding_dim ${embedding_dim} --lr ${lr} --fl_alpha ${alpha} \
          --fl_gamma ${gamma}
        done
      done
    done
  done
done