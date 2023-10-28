#!/usr/bin/env bash

emb_dim_array=(32 64 128)
neighbor_num_array=(3 5 10) 

dataset_array=( "dataset1" "dataset2") 
models=("HGCNLDA_cep" "HGCNLDA_wcep" "HGCNLDA_wceb")  
comment="emp/ablation"  
lr=1e-2 
epochs=300  
for dataset in ${dataset_array[*]};do
  for model in ${models[*]};do
      for neighbor_num in ${neighbor_num_array[*]};do
        for embedding_dim in ${emb_dim_array[*]};do
          if [ $dataset == "dataset2" ]
          then
              epochs=150
          fi
          python main.py --model ${model} --dataset_name ${dataset} --comment "E:/${comment}/${dataset}/neighbor_num=${neighbor_num}_embedding_dim=${embedding_dim}" \
          --epochs ${epochs} --lnc_neighbor_num ${neighbor_num} --disease_neighbor_num ${neighbor_num} \
          --mi_neighbor_num ${neighbor_num} --embedding_dim ${embedding_dim} --lr ${lr}
        done
      done
  done
done