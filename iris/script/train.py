import os
import json
import argparse
from pathlib import Path

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

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
S3_MODEL_BASE_KEY = outputs["S3ModelBaseKey"]

SM_ROLE_ARN = outputs["SmRoleArn"]
SM_ENDPOINT_NAME = outputs["SmEndpointName"]

INPUT_PATH = f"s3://{S3_BUCKET}/{S3_TRAIN_BASE_KEY}"
OUTPUT_PATH = f's3://{S3_BUCKET}/{S3_MODEL_BASE_KEY}'


def main(update_endpoint=False):
    script_path = str(Path(__file__).parent/"src/iris.py")
    train_instance_type = "ml.m5.large"
    initial_instance_count = 1
    hosting_instance_type = "ml.t2.medium"

    sagemaker_session = sagemaker.Session()
    # 学習
    sklearn = SKLearn(
        entry_point=script_path,
        train_instance_type=train_instance_type,
        role=SM_ROLE_ARN,
        sagemaker_session=sagemaker_session,
        output_path=OUTPUT_PATH
    )
    sklearn.fit({'train': INPUT_PATH})

    # デプロイ
    sklearn.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=hosting_instance_type,
        endpoint_name=SM_ENDPOINT_NAME,
        update_endpoint=update_endpoint
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-endpoint', action='store_true')
    args = parser.parse_args()
    update_endpoint = args.update_endpoint
    main(update_endpoint)