import hashlib
import os
from pathlib import Path

import requests
from tqdm import tqdm


def _get_hash(path: Path) -> str:
    file_hash = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def download(path: Path, public_key: str) -> None:
    url = "https://cloud-api.yandex.net/v1/disk/public/resources"
    params = {"public_key": f"https://disk.yandex.ru/d/{public_key}"}

    response = requests.get(url, params=params).json()
    download_url = response["file"]
    file_size = response["size"]
    sha256 = response["sha256"]

    response = requests.get(download_url, stream=True)

    if path.is_file() and os.path.getsize(path) == file_size:
        print(f"File already downloaded: {path}")
        if _get_hash(path) == sha256:
            return

    with tqdm(total=file_size, unit="B", unit_scale=True) as progress_bar:
        with open(path, "wb") as f:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                f.write(data)
