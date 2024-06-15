#!/bin/bash

# Change to the script's running directory
cd "$(dirname "$0")"

export PYTHONPATH=

# Source the virtual environment
source ~/fmcv/vllm/bin/activate

# Execute the Python script with sudo
uvicorn demo_llama_3_v_chinese:app --reload  --port 8500 --host=0.0.0.0 --ssl-keyfile key.pem --ssl-certfile cert.pem
