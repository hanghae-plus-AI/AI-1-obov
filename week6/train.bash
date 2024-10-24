#!/bin/bash

# 필요한 패키지 설치
sudo apt-get update
sudo apt-get install -y python3-pip git

# Python 및 관련 패키지 설치
pip3 install torch transformers datasets wandb evaluate

# WANDB API Key가 저장된 파일에서 키를 읽어와 로그인
if [ -f "wandb_api_key.txt" ]; then
    WANDB_API_KEY=$(cat wandb_api_key.txt)
    wandb login $WANDB_API_KEY
else
    echo "wandb_api_key.txt 파일을 찾을 수 없습니다. 파일에 API 키를 저장해주세요."
    exit 1
fi

# 모델과 데이터셋을 지정하여 스크립트를 실행합니다.
python3 train.py \
    --model_name_or_path openai-community/openai-gpt \
    --per_device_train_batch_size 8 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --output_dir /tmp/test-clm \
    --save_total_limit 1 \
    --logging_steps 100 \
    --eval_strategy epoch \