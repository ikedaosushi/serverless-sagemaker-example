import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

load_dotenv(".env")

SM_ROLE_ARN = os.environ.get("SM_ROLE_ARN")
S3_BUCKET =  os.environ.get("S3_BUCKET")
SM_ENDPOINT_NAME = os.environ.get("SM_ENDPOINT_NAME")

TRAIN_KEY = "train_data"
OUTPUT_KEY = "artifacts"
INPUT_PATH = f"s3://{S3_BUCKET}/{TRAIN_KEY}"
OUTPUT_PATH = f's3://{S3_BUCKET}/{OUTPUT_KEY}'


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