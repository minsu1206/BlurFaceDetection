#!/bin/bash
python train.py --config base_regression --save checkpoint_sample/base_regression --device cuda:2
python train.py --config edgenext_regression --save checkpoint_sample/edgenext_regression --device cuda:2
python train.py --config yolov5n_regression --save checkpoint_sample/yolov5n_regression --device cuda:2

python test.py --config base_regression --device cuda:2 --save checkpoint_sample/base_regression --resume checkpoint_sample/base_regression/checkpoint_0.ckpt
python test.py --config edgenext_regression --device cuda:2 --save checkpoint_sample/edgenext_regression --resume checkpoint_sample/edgenext_regression/checkpoint_0.ckpt
python test.py --config yolov5n_regression --device cuda:2 --save checkpoint_sample/yolov5n_regression --resume checkpoint_sample/yolov5n_regression/checkpoint_0.ckpt

python recorder.py --checkpoints checkpoint_sample/base_regression checkpoint/yolov5n_regression checkpoint/edgenext_regression --save_path model_3_compare.png

