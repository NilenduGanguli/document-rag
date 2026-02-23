import boto3
import tempfile
import os
from botocore.exceptions import ClientError
from botocore.config import Config
from app.core.config import settings


def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=f"{'https' if settings.MINIO_SECURE else 'http'}://{settings.MINIO_ENDPOINT}",
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # MinIO ignores region but boto3 requires one
    )


def ensure_bucket_exists() -> None:
    """Create the configured bucket if it does not already exist."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=settings.MINIO_BUCKET)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
            client.create_bucket(Bucket=settings.MINIO_BUCKET)
        else:
            raise


def upload_file(object_key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """
    Upload *data* to MinIO under *object_key* and return the canonical S3 URI.

    Returns
    -------
    str
        ``s3://<bucket>/<object_key>``
    """
    ensure_bucket_exists()
    client = _get_client()
    client.put_object(
        Bucket=settings.MINIO_BUCKET,
        Key=object_key,
        Body=data,
        ContentType=content_type,
    )
    return f"s3://{settings.MINIO_BUCKET}/{object_key}"


def download_to_tempfile(s3_uri: str) -> str:
    """
    Download an object identified by *s3_uri* (``s3://<bucket>/<key>``) to a
    temporary local file and return the path.

    The caller is responsible for deleting the temp file when done.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {s3_uri}")

    path_part = s3_uri[len("s3://"):]
    bucket, _, key = path_part.partition("/")

    client = _get_client()
    _, ext = os.path.splitext(key)
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    client.download_file(bucket, key, tmp_path)
    return tmp_path
