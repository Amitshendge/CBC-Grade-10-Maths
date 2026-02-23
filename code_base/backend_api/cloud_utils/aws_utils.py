# aws.py
import json
import boto3
from functools import lru_cache
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION


def _client_kwargs():
    kwargs = {"region_name": AWS_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return kwargs


@lru_cache()
def get_s3_client():
    return boto3.client("s3", **_client_kwargs())

@lru_cache()
def get_secret_manager_client():
    return boto3.client("secretsmanager", **_client_kwargs())

def get_secret(secret_name: str) -> str:
    if not secret_name:
        raise ValueError("Secret name is empty.")
    client = get_secret_manager_client()
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

def upload_file(bucket: str, key: str, data: bytes):
    s3 = get_s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=data)

def delete_file(bucket: str, key: str):
    s3 = get_s3_client()
    s3.delete_object(Bucket=bucket, Key=key)
    
def read_file(bucket: str, key: str) -> bytes:
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()

def create_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    s3 = get_s3_client()
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )
    return url
