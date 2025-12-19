import os
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError


AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

_s3_client: Optional[boto3.client] = None


def _get_s3_client() -> boto3.client:
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    if not AWS_REGION or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("Missing AWS configuration for S3")
    _s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    return _s3_client


def upload_file_and_get_key(
    local_path: str,
    key: str,
    delete_local: bool = True,
) -> Optional[str]:
    """
    Upload a file to S3 and return the S3 object key on success.
    Optionally delete the local file after successful upload.
    """
    if not S3_BUCKET_NAME:
        return None

    try:
        s3 = _get_s3_client()
        s3.upload_file(local_path, S3_BUCKET_NAME, key)
        if delete_local and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return key
    except (BotoCoreError, ClientError, FileNotFoundError) as e:
        print(f"S3 upload error for {local_path}: {e}")
        return None


def generate_presigned_url(
    key: str,
    expires_in: int = 15 * 60,
) -> Optional[str]:
    """
    Generate a pre-signed GET URL for an S3 object key.
    """
    if not S3_BUCKET_NAME:
        return None

    try:
        s3 = _get_s3_client()
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except (BotoCoreError, ClientError) as e:
        print(f"S3 presign error for key {key}: {e}")
        return None


