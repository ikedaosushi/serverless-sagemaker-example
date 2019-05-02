#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

TRAIN_FILE_NAME = "train.csv"
MODEL_FILE_NAME = "model.joblib"

def train(train_dir: Path, model_dir: Path):
    # S3からダウンロードされたファイルを読み込み
    train_file = train_dir/TRAIN_FILE_NAME # /opt/ml/input/data/train/train.csv
    df = pd.read_csv(train_file, engine='python')

    # 説明変数と目的変数に分ける
    X = df.drop('y', axis=1)
    y = df['y']

    # 学習
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # 書き出し
    joblib.dump(clf, model_dir/MODEL_FILE_NAME)

if __name__ == '__main__':
    model_dir = os.environ['SM_MODEL_DIR'] # /opt/ml/model
    train_dir = os.environ['SM_CHANNEL_TRAIN'] # /opt/ml/input/data/train 
    train_dir = Path(train_dir)
    model_dir = Path(model_dir)
    train(train_dir, model_dir)

def model_fn(model_dir: str):
    """Predictで使う用の関数。学習されたモデルを返す"""
    model_path = Path(model_dir)/MODEL_FILE_NAME
    clf = joblib.load(model_path)

    return clf