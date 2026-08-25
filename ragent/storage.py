"""Object storage for raw uploads, converted PDFs and page renders.

MinIO locally, any S3 in production. Documents live here and never in the
message queue, so a message that sits in a queue across a deploy cannot go stale
and the broker never becomes a second copy of the corpus.
"""

from __future__ import annotations

import hashlib
from typing import Any

import aioboto3

from ragent.config import get_settings

__all__ = ["sha256_of", "put_object", "get_object", "object_uri", "presign"]

_session = aioboto3.Session()


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_uri(key: str) -> str:
    return f"s3://{get_settings().s3_bucket}/{key}"


def _key_of(uri: str) -> str:
    prefix = f"s3://{get_settings().s3_bucket}/"
    return uri[len(prefix) :] if uri.startswith(prefix) else uri


def _client() -> Any:
    settings = get_settings()
    return _session.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name="us-east-1",
    )


async def put_object(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    async with _client() as s3:
        await s3.put_object(
            Bucket=get_settings().s3_bucket, Key=key, Body=data, ContentType=content_type
        )
    return object_uri(key)


async def get_object(uri: str) -> bytes:
    async with _client() as s3:
        response = await s3.get_object(Bucket=get_settings().s3_bucket, Key=_key_of(uri))
        return await response["Body"].read()


async def presign(uri: str, expires: int = 3600) -> str:
    """Short-lived URL so the browser fetches page renders directly.

    Streaming them back through the API would double the bytes over the wire for
    no benefit — the viewer requests a lot of page images while panning.
    """
    async with _client() as s3:
        return await s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": get_settings().s3_bucket, "Key": _key_of(uri)},
            ExpiresIn=expires,
        )
