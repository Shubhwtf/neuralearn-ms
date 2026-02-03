import os
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


def _get_env_var(name: str) -> Optional[str]:
    value = os.getenv(name)
    if not value:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


AWS_REGION = _get_env_var("AWS_REGION")
AWS_ACCESS_KEY_ID = _get_env_var("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _get_env_var("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = _get_env_var("S3_BUCKET_NAME")
S3_LOCATION_PREFIX = _get_env_var("S3_LOCATION_PREFIX") or "eda-ms"

_s3_client: Optional[boto3.client] = None


def _is_s3_configured() -> bool:
    return bool(AWS_REGION and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME)


def _get_s3_client() -> Optional[boto3.client]:
    global _s3_client
    
    if not _is_s3_configured():
        return None
    
    if _s3_client is not None:
        return _s3_client
    
    try:
        config = Config(
            signature_version='s3v4',
            region_name=AWS_REGION
        )
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            config=config,
        )
        return _s3_client
    except Exception:
        return None


def upload_file_and_get_key(
    local_path: str,
    key: str,
    delete_local: bool = True,
) -> Optional[str]:
    if not _is_s3_configured():
        return None

    s3 = _get_s3_client()
    if s3 is None:
        return None

    try:
        if not os.path.exists(local_path):
            return None

        if not key.startswith(f"{S3_LOCATION_PREFIX}/"):
            key = f"{S3_LOCATION_PREFIX}/{key}"
        
        s3.upload_file(local_path, S3_BUCKET_NAME, key)
        if delete_local and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return key
    except (BotoCoreError, ClientError, FileNotFoundError, Exception):
        return None


def generate_presigned_url(
    key: str,
    expires_in: int = 15 * 60,
) -> Optional[str]:
    if not _is_s3_configured():
        return None

    s3 = _get_s3_client()
    if s3 is None:
        return None

    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except (BotoCoreError, ClientError, Exception):
        return None


