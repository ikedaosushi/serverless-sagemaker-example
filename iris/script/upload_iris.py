#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from dotenv import load_dotenv
from sklearn.datasets import load_iris
import pandas as pd
import boto3

load_dotenv(".env")

S3_BUCKET =  os.environ.get("S3_BUCKET")
TRAIN_BASE_KEY = "train_data"
TRAIN_FILE_NAME = "train.csv"

# 一旦データを保存する用のディレクトリを作成
DATA_DIR = Path(__file__).parents[1]/'data' # => Rootディレクトリ/data
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=True)


def main():
    # irisデータをロード
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['y'] = iris.target

    # dataディレクトリに保存
    train_path = DATA_DIR/TRAIN_FILE_NAME # => data/train.csv
    df.to_csv(train_path, index=False)

    # S3にアップロード
    key = f"{TRAIN_BASE_KEY}/{TRAIN_FILE_NAME}" # => train_data/train.csv
    client = boto3.client("s3")
    client.upload_file(
        Filename=str(train_path),
        Bucket=S3_BUCKET,
        Key=key
    )

if __name__ == "__main__":
    main()