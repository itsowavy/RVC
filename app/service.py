import os

import boto3

from app import constants
from app.constants import PTH_DIR_PATH, INDEX_DIR_PATH


def download_from_s3(key: str, file_path: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=constants.AWS_ACCESS_KEY,
        aws_secret_access_key=constants.AWS_SECRET_KEY
    )
    s3.download_file('voicechanger-resource', key, file_path)


def download_index_pth(speaker_name: str):
    download_from_s3(f"pth/{speaker_name}.pth", os.path.join(PTH_DIR_PATH, f"{speaker_name}.pth"))
    download_from_s3(f"index/{speaker_name}.index", os.path.join(INDEX_DIR_PATH, f"{speaker_name}.index"))
