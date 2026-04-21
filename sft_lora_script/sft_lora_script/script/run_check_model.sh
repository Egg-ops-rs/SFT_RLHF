#!/bin/bash

python evaluate_model.py --model_path /root/autodl-tmp/finetune_output/final_complete_model --test_file /root/autodl-tmp/data/sft/deepspeek_sft_dataset_2000.jsonl --batch_size 8 --max_length 1024 --device auto --sample_size 400 --max_new_tokens 1024