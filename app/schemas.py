from typing import List

from pydantic import BaseModel

from app.io_device import IODevice


class StreamRequest(BaseModel):
    input_device: IODevice
    output_device: IODevice
    pitch: int


class SettingResponse(StreamRequest):
    input_devices_list: List[IODevice]
    output_devices_list: List[IODevice]


class ConvertRequest(BaseModel):
    source_path: str
    save_dir_path: str
    pitch: int


class RecordRequest(BaseModel):
    input_device: IODevice
    save_dir_path: str
    pitch: int
