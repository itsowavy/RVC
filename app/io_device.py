import json
from enum import Enum
from typing import List

import sounddevice as sd
from pydantic import BaseModel


class DeviceType(Enum):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'


class IODevice(BaseModel):
    class Config:
        use_enum_values = True

    type: DeviceType
    index: int
    api_name: str
    device_name: str

    def format_device_name(self):
        return f"{self.device_name} ({self.api_name})"

    def exists_in_list(self, device_list: List['IODevice']) -> bool:
        for device in device_list:
            if self.api_name == device.api_name and self.device_name == device.device_name:
                return True
        return False

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        data['type'] = DeviceType(data['type'])
        return cls(**data)


def set_devices(self, input_device, output_device):
    sd.default.device[0] = self.input_devices_indices[
        self.input_devices.index(input_device)
    ]
    sd.default.device[1] = self.output_devices_indices[
        self.output_devices.index(output_device)
    ]
    print("Input device: %s:%s", str(sd.default.device[0]), input_device)
    print("Output device: %s:%s", str(sd.default.device[1]), output_device)
