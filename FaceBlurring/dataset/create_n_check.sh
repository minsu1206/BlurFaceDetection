#!/bin/sh

# 데이터셋을 생성하고, 생성된 label과 함께 확인하는 스크립트
python create_blurimg.py --blur $1 --calc $2
python dataset.py --option blur --calc $2
