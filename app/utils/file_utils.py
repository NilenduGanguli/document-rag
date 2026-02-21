import httpx
import tempfile
import os

def download_file(uri: str) -> str:
    """
    Downloads a file from a URI to a temporary local file.
    If it's already a local file path, returns the path.
    """
    if uri.startswith("http://") or uri.startswith("https://"):
        response = httpx.get(uri)
        response.raise_for_status()
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as f:
            f.write(response.content)
        return path
    elif uri.startswith("file://"):
        return uri.replace("file://", "")
    return uri
