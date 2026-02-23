import httpx
import tempfile
import os

def download_file(uri: str) -> str:
    """
    Download a file from the given URI to a local path and return it.

    Supported schemes
    -----------------
    s3://      — download from MinIO/S3 via s3_service
    http(s)://  — plain HTTP download to a temp file
    file://    — strip prefix, treat as local path
    (bare)     — treat directly as a local path
    """
    if uri.startswith("s3://"):
        from app.services.s3_service import download_to_tempfile
        return download_to_tempfile(uri)
    if uri.startswith("http://") or uri.startswith("https://"):
        response = httpx.get(uri)
        response.raise_for_status()
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as f:
            f.write(response.content)
        return path
    if uri.startswith("file://"):
        return uri.replace("file://", "")
    return uri
