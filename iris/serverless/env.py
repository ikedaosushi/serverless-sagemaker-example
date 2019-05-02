import os
import json

def see_environment(enent, context):
    SM_ROLE_ARN = os.environ.get("SM_ROLE_ARN")
    SM_ENDPOINT_NAME =  os.environ.get("SM_ENDPOINT_NAME")
    S3_BUCKET =  os.environ.get("S3_BUCKET")
    resp = {
        "SM_ROLE_ARN": SM_ROLE_ARN,
        "SM_ENDPOINT_NAME": SM_ENDPOINT_NAME,
        "S3_BUCKET": S3_BUCKET
    }

    return resp
