import os
import json

def hello(event, context):
    resp = {
        "statusCode": 200,
        "body": json.dumps(event)
    }

    return resp
