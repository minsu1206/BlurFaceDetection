#!/bin/sh
# VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace
python embedding.py --label psnr --option deepface --model ArcFace
python embedding.py --label ssim --option deepface --model ArcFace
python embedding.py --label degree --option deepface --model ArcFace
python embedding.py --label psnr --option deepface --model Facenet
python embedding.py --label ssim --option deepface --model Facenet
python embedding.py --label degree --option deepface --model Facenet
python embedding.py --label psnr --option deepface --model VGG-Face
python embedding.py --label ssim --option deepface --model VGG-Face
python embedding.py --label degree --option deepface --model VGG-Face
python embedding.py --label psnr --option deepface --model DeepFace
python embedding.py --label ssim --option deepface --model DeepFace
python embedding.py --label degree --option deepface --model DeepFace