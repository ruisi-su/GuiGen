#!/bin/bash

# --------------------------------------
# This script is used to run txt_generation.py
# --------------------------------------

# Quit if there're any errors
set -e

date=$1

TXT_MODEL_PATH=models/model.$date.best.txt.chkpt

CUDA_VISIBLE_DEVICES=0 python txt_generate.py \
	--txt_model_path "$TXT_MODEL_PATH" \
	--no_cuda

