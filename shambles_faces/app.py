import json
import base64
import cv2
import numpy as np
import random
import boto3


def load_input_image(img_bin):
    img_array = np.frombuffer(img_bin,dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def convet_base64(img):
    _, encoded = cv2.imencode(".jpg", img)
    return base64.b64encode(encoded).decode("ascii")


def absolute_position(width, height, b_box):
    w = width * b_box['Width']
    h = height * b_box['Height']
    l = width * b_box['Left']
    t = height * b_box['Top']

    return (int(w), int(h), int(l), int(t))


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    img_bin = base64.b64decode(event['body'])


    ### Rekognition
    client = boto3.client('rekognition')
    response = client.detect_faces(
        Image={
            'Bytes': img_bin,
        },
        Attributes=[
            'DEFAULT',
        ]
    )


    ## Shambles
    img_input = load_input_image(img_bin)
    height, width, channels = img_input.shape[:3]
    img_output = img_input.copy()

    bounding_box = list(map(lambda x: x['BoundingBox'], response['FaceDetails']))

    l1 = list(range(len(bounding_box)))
    l2 = random.sample(l1, len(l1))

    for index in l1:
        (w, h, l, t) = absolute_position(width, height, bounding_box[index])
        clop = img_input[t: t + h, l: l + w]

        (w, h, l, t) = absolute_position(width, height, bounding_box[l2[index]])
        clop = cv2.resize(clop, (w, h))

        img_output[t: t + h, l: l + w] = clop

    ## 

    return {
        "statusCode": 200,
        "body": convet_base64(img_output),
        "isBase64Encoded": True
    }
