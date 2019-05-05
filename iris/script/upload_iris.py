#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path

from dotenv import load_dotenv
from sklearn.datasets import load_iris
import pandas as pd
import boto3

# CloudFormationから環境変数を読み出し
## CFのStack設定
SERVICE_NAME = "sagemaker-serverless-example"
ENV = os.environ.get("ENV", "dev")
STACK_NAME = f"{SERVICE_NAME}-{ENV}"

## Outputsを{Key: Valueの形で読み出し}
stack = boto3.resource('cloudformation').Stack(STACK_NAME)
outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.outputs}

S3_BUCKET = outputs["S3Bucket"]
S3_TRAIN_BASE_KEY = outputs["S3TrainBaseKey"]
TRAIN_FILE_NAME = "train.csv"

def main():
    # irisデータをロード
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['y'] = iris.target

    # dataディレクトリに保存
    data_dir = Path(__file__).parents[1]/'data' # => Rootディレクトリ/data
    # データを保存する用のディレクトリを作成
    if not data_dir.is_dir():
        data_dir.mkdir(parents=True)

    train_path = data_dir/TRAIN_FILE_NAME # => data/train.csv
    df.to_csv(train_path, index=False)

    # S3にアップロード
    key = f"{S3_TRAIN_BASE_KEY}/{TRAIN_FILE_NAME}" # => train_data/train.csv
    client = boto3.client("s3")
    client.upload_file(
        Filename=str(train_path),
        Bucket=S3_BUCKET,
        Key=key
    )

if __name__ == "__main__":
    main()