import json
from enum import Enum


class SpeakerStatus(Enum):
    AVAILABLE = 'available'
    UNAVAILABLE = 'unavailable'
    DOWNLOADING = 'downloading'


class Speaker:
    name: str
    pth_name: str
    index_name: str
    sid: int
    status: SpeakerStatus

    def __init__(self, name: str, pth_path: str, index_name: str, sid: int,
                 status: SpeakerStatus = SpeakerStatus.UNAVAILABLE):
        self.name = name
        self.pth_name = pth_path
        self.index_name = index_name
        self.sid = sid
        self.status = status

    def to_json(self):
        speaker_dict = self.__dict__.copy()
        speaker_dict['status'] = self.status.value
        return json.dumps(speaker_dict)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        data['status'] = SpeakerStatus(data['status'])
        return cls(**data)
