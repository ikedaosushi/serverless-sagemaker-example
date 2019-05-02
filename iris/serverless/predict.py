import os
import json

from sagemaker.sklearn.model import SKLearnPredictor

SM_ENDPOINT_NAME =  os.environ.get("SM_ENDPOINT_NAME")

def predict(event, contect):
    data = json.loads(event["body"])['data']
    predictor = SKLearnPredictor(endpoint_name=SM_ENDPOINT_NAME)
    y = predictor.predict([data]).tolist()
    body = {
        "y": y
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response