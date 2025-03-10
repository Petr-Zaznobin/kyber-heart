import shutil
from enum import Enum
from pathlib import Path

from . import yadisk


class Type(Enum):
    PTB_XL = 1
    MIT_BIH_AF = 2
    THREE_BASES = 3


def _download(
        name: str,
        path_dir: Path,
        name_zip: str,
        exist_ok: bool,
        public_key: str) -> None:
    path_dir.mkdir(parents=True, exist_ok=True)
    path_zip = path_dir / name_zip
    if not exist_ok and path_zip.exists():
        print(f"Archive already exists: {path_zip}")
        return
    yadisk.download(path_zip, public_key)
    shutil.unpack_archive(path_zip, path_dir, format="zip")
    print(f"{name} is downloaded and unzipped")


def _download_ptb_xl(path_dir: Path, exist_ok: bool) -> None:
    _download("PTB-XL", path_dir, "ptb_xl.zip", exist_ok, "Uzm7r0IFlE2cSw")


def _download_mit_bih_af(path_dir: Path, exist_ok: bool) -> None:
    _download("MIT-BIH Atrial Fibrillation", path_dir, "mit_bih_af.zip", exist_ok, "XyVrZired_NRrw")


def _download_three_bases(path_dir: Path, exist_ok: bool) -> None:
    _download("'Three bases'", path_dir, "three_bases.zip", exist_ok, "3GvQyz3wWMbK1w")
    _download(
        "'Three bases' bad records",
        path_dir,
        "three_bases_bad_records.zip",
        exist_ok,
        "R3xTmgtjDRTqSQ"
    )


def download(database: Type, path_dir: Path | str = "datasets", exist_ok: bool = False) -> None:
    if isinstance(path_dir, str):
        path_dir = Path(path_dir)

    type_to_func = {
        Type.PTB_XL: _download_ptb_xl,
        Type.MIT_BIH_AF: _download_mit_bih_af,
        Type.THREE_BASES: _download_three_bases
    }

    type_to_func[database](path_dir, exist_ok)
