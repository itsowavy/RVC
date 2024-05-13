import os

from app.constants import PTH_DIR_PATH, INDEX_DIR_PATH
from app.utils import download_from_s3


def download_index_pth(speaker_name: str):
    download_from_s3(f"pth/{speaker_name}.pth", os.path.join(PTH_DIR_PATH, f"{speaker_name}.pth"))
    download_from_s3(f"index/{speaker_name}.index", os.path.join(INDEX_DIR_PATH, f"{speaker_name}.index"))
