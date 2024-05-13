import json
from enum import Enum
from typing import List

from pydantic import BaseModel


class DeviceType(Enum):
    INPUT = 'input'
    OUTPUT = 'output'


class IODevice(BaseModel):
    class Config:
        use_enum_values = True

    type: DeviceType
    index: int
    api_name: str
    device_name: str

    def exists_in_list(self, device_list: List['IODevice']) -> bool:
        for device in device_list:
            if self.api_name == device.api_name and self.device_name == device.device_name:
                return True
        return False

    def to_json(self):
        return self.json()

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        data['type'] = DeviceType(data['type'])
        return cls(**data)
