from typing import List, Optional

from pydantic import BaseModel

from app.io_device import IODevice
from app.speaker import SpeakerStatus


class StreamRequest(BaseModel):
    input_device: IODevice
    output_device: IODevice
    pitch: int
    speaker: str


class SettingResponse(BaseModel):
    input_device: Optional[IODevice]
    output_device: Optional[IODevice]
    input_devices_list: Optional[List[IODevice]]
    output_devices_list: Optional[List[IODevice]]
    pitch: int


class ConvertRequest(BaseModel):
    source_path: str
    save_dir_path: str
    pitch: int


class RecordRequest(BaseModel):
    input_device: IODevice
    save_dir_path: str
    pitch: int
    speaker: str


class SpeakerResponse(BaseModel):
    name: str
    status: str


class SpeakersListResponse(BaseModel):
    speakers: List[SpeakerResponse]
