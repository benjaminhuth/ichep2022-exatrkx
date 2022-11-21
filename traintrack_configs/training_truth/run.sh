#!/bin/bash

cp configs/processing.yaml LightningModules/Processing/processing.yaml
cp configs/embedding.yaml LightningModules/Embedding/embedding.yaml
cp configs/filter.yaml LightningModules/Filter/filter.yaml
cp configs/gnn.yaml LightningModules/GNN/gnn.yaml

export PYTHONWARNINGS="ignore"

export EXATRKX_DATA=$HOME/exatrkx/data/training_data_1K

export PROCESSING_OUTPUT=tmp/processing_output
export EMBEDDING_OUTPUT=tmp/embedding_output
export FILTER_OUTPUT=tmp/filter_output
export GNN_OUTPUT=tmp/gnn_output
export SEGMENTING_OUTPUT=tmp/segmenting_output

export PROJECT_NAME="ODD-1K-truth"

traintrack "$@" configs/pipeline.yaml

rm LightningModules/Processing/processing.yaml
rm LightningModules/Embedding/embedding.yaml
rm LightningModules/Filter/filter.yaml
rm LightningModules/GNN/gnn.yaml
