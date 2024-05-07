import json
import os
from datetime import datetime
from typing import List

import numpy as np
import sounddevice as sd
import torch

from app.constants import SETTING_FILE_PATH
from app.io_device import IODevice, DeviceType


def load_setting():
    if not os.path.exists(SETTING_FILE_PATH):
        return None, None, 0
    with open(SETTING_FILE_PATH, "r") as f:
        data = json.load(f)
        pitch = data["pitch"] if data["pitch"] else 0
        input_device = IODevice.from_json(data["input_device"]) if data["input_device"] else None
        output_device = IODevice.from_json(data["output_device"]) if data["output_device"] else None

    return input_device, output_device, pitch


def save_setting(input_device: IODevice, output_device: IODevice, pitch: int):
    data_json = {
        "pitch": pitch,
        "input_device": input_device.input_device.to_json() if input_device else None,
        "output_device": output_device.output_device.to_json() if output_device else None
    }
    with open(SETTING_FILE_PATH, "w") as f:
        json.dump(data_json, f)


def set_io_devices(input_device: IODevice, output_device: IODevice):
    sd.default.device = (input_device.index, output_device.index if output_device else None)


def get_io_devices(update: bool = True):
    if update:
        sd._terminate()
        sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    input_devices: List[IODevice] = []
    output_devices: List[IODevice] = []
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            if devices[device_idx]["max_input_channels"] > 0:
                input_devices.append(
                    IODevice(
                        api_name=hostapi["name"], device_name=devices[device_idx]["name"],
                        index=devices[device_idx]["index"], type=DeviceType.INPUT
                    )
                )
            elif devices[device_idx]["max_output_channels"] > 0:
                output_devices.append(
                    IODevice(
                        api_name=hostapi["name"], device_name=devices[device_idx]["name"],
                        index=devices[device_idx]["index"], type=DeviceType.OUTPUT
                    )
                )
    return input_devices, output_devices


def get_device_samplerate():
    return int(
        sd.query_devices(device=sd.default.device[0])["default_samplerate"]
    )


def get_device_channels():
    max_input_channels = sd.query_devices(device=sd.default.device[0])[
        "max_input_channels"
    ]
    max_output_channels = sd.query_devices(device=sd.default.device[1])[
        "max_output_channels"
    ]
    return min(max_input_channels, max_output_channels, 2)


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out ** 2)
        + b * (fade_in ** 2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


def get_filepath(dir_path):
    current_datetime = datetime.now().strftime("%Y.%m.%d_%H-%M-%S.%f")[:-3] + ".wav"
    filepath = os.path.join(dir_path, current_datetime)
    return filepath


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)
