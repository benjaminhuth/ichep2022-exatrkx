#!/bin/bash

cp configs/processing-smear.yaml LightningModules/Processing/processing-smear.yaml
cp configs/embedding-smear.yaml LightningModules/Embedding/embedding-smear.yaml
cp configs/filter-smear.yaml LightningModules/Filter/filter-smear.yaml
cp configs/gnn-smear.yaml LightningModules/GNN/gnn-smear.yaml
cp configs/graph-modifier-smear.yaml LightningModules/GraphModifier/graph-modifier-smear.yaml

export PYTHONWARNINGS="ignore"

export EXATRKX_DATA=$HOME/exatrkx/data/training_data_1K

export PROCESSING_OUTPUT=tmp/processing_output
export EMBEDDING_OUTPUT=tmp/embedding_output
export MODIFIER_OUTPUT=tmp/modifier_output
export FILTER_OUTPUT=tmp/filter_output
export GNN_OUTPUT=tmp/gnn_output
export SEGMENTING_OUTPUT=tmp/segmenting_output

export PROJECT_NAME="ODD-1K-smear"

traintrack "$@" configs/pipeline.yaml

rm LightningModules/Processing/processing-smear.yaml
rm LightningModules/Embedding/embedding-smear.yaml
rm LightningModules/Filter/filter-smear.yaml
rm LightningModules/GNN/gnn-smear.yaml
