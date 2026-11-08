#!/bin/bash
# Full pipeline on one GPU machine (tested on one 96 GB GPU; the 32B judge needs ~70 GB in bfloat16).
# Times on that machine: build ~25 min (14 CPU workers), CLIP ~2 min, Qwen-7B ~35 min, Qwen-32B ~2 h.
set -e
python prepare_data.py
python build_pairs.py
python judge.py clip
python judge.py qwen7b
python judge.py qwen32b
python analyze.py
