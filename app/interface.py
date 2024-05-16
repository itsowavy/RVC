import os
import sys
import threading
import time

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

import infer.lib.rtrvc as rtrvc
from app.constants import PTH_DIR_PATH, INDEX_DIR_PATH
from app.schemas import StreamRequest, RecordRequest
from app.utils import get_device_samplerate, get_device_channels, phase_vocoder, printt, set_io_devices
from app.config import Config, Status
from configs.config import Config as VcConfig
from infer.modules.gui import TorchGate
from infer.modules.vc.modules import VC


# 싱글톤 클래스
class Interface:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Interface, cls).__new__(cls)
        return cls._instance

    def __init__(self, device, inp_q, opt_q):
        self.config = Config()
        self.status = Status()
        self.device = device
        self.inp_q = inp_q
        self.opt_q = opt_q
        self.vc_config = VcConfig()
        self.vc = VC(self.vc_config)

    @classmethod
    def get_instance(cls):
        return cls._instance

    def _change_stream_flag(self, is_stream: bool):
        self.status.flag_stream = is_stream

    def _change_record_flag(self, is_record: bool):
        self.status.flag_record = is_record

    def _change_convert_flag(self, is_convert: bool):
        self.status.flag_convert = is_convert

    def is_streaming(self) -> bool:
        return self.status.flag_stream

    def is_recording(self) -> bool:
        return self.status.flag_record

    def is_converting(self) -> bool:
        return self.status.flag_conversion

    def save_setting(self):
        self.config.save()

    def get_user_setting(self):
        return self.config.get_user_setting()

    def get_latency(self):
        return self.status.latency_stream

    def set_stream_config(self, values: StreamRequest):
        self.config.input_device = values.input_device
        self.config.output_device = values.output_device
        self.config.pitch = values.pitch
        self.config.pth_path = os.path.join(PTH_DIR_PATH, f"{values.speaker}.pth")
        self.config.index_path = os.path.join(INDEX_DIR_PATH, f"{values.speaker}.index")
        set_io_devices(self.config.input_device, self.config.output_device)

    def set_record_config(self, values: RecordRequest, file_path):
        audio_file = sf.SoundFile(file_path, mode='w', samplerate=self.config.samplerate, channels=2, format='WAV')
        self.config.output_file = audio_file
        self.config.input_device = values.input_device
        self.config.output_device = None
        self.config.pth_path = os.path.join(PTH_DIR_PATH, f"{values.speaker}.pth")
        self.config.index_path = os.path.join(INDEX_DIR_PATH, f"{values.speaker}.index")
        self.config.pitch = values.pitch
        set_io_devices(self.config.input_device, None)

    def convert_single(self, sid, input_file_path, pitch, f0_file, f0_method, file_index,
                       index_rate, filter_radius, resample_sr, rms_mix_rate, protect):

        return self.vc.vc_single(sid, input_file_path, pitch, f0_file, f0_method, file_index, None,
                                 index_rate, filter_radius, resample_sr, rms_mix_rate, protect)

    def stop_vc(self):
        self._change_stream_flag(False)
        self._change_record_flag(False)
        if self.stream is not None:
            self.stream.abort()
            self.stream.close()
            self.stream = None
        if self.config.output_file is not None and not self.config.output_file.closed:
            self.config.output_file.close()
        self.config.output_file = None

    def start_vc(self, is_record=False):
        torch.cuda.empty_cache()
        if is_record:
            self._change_record_flag(True)
        self.rvc = rtrvc.RVC(
            self.config.pitch,
            self.config.formant,
            self.config.pth_path,
            self.config.index_path,
            self.config.index_rate,
            self.config.n_cpu,
            self.inp_q,
            self.opt_q,
            self.vc_config,
            self.rvc if hasattr(self, 'rvc') else None
        )
        self.config.samplerate = get_device_samplerate()

        self.config.channels = get_device_channels()
        self.zc = self.config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.config.block_time
                    * self.config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.config.crossfade_time
                    * self.config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.config.extra_time
                    * self.config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.vc_config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.vc_config.device,
            dtype=torch.float32,
        )
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.vc_config.device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
                                 self.block_frame + self.sola_buffer_frame + self.sola_search_frame
                             ) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.vc_config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.vc_config.device)
        if self.rvc.tgt_sr != self.config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.config.samplerate,
                dtype=torch.float32,
            ).to(self.vc_config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.vc_config.device)
        self.start_stream()

    def start_stream(self):
        if not self.is_streaming():
            self._change_stream_flag(True)
            # if (
            #     "WASAPI" in self.gui_config.sg_hostapi
            #     and self.gui_config.sg_wasapi_exclusive
            # ):
            #     extra_settings = sd.WasapiSettings(exclusive=True)
            # else:
            extra_settings = None
            self.stream = sd.Stream(
                callback=self.__audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()

    def __audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc:]
            indata = indata[2 * self.zc - self.zc // 2:]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc: (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2:]
        self.input_wav[: -self.block_frame] = self.input_wav[
                                              self.block_frame:
                                              ].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(
            self.vc_config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
                                                      self.block_frame_16k:
                                                      ].clone()
        # input noise reduction and resampling
        if self.config.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                                                          self.block_frame:
                                                          ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame:]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += (
                self.nr_buffer * self.fade_out_window
            )
            self.input_wav_denoise[-self.block_frame:] = input_wav[
                                                         : self.block_frame
                                                         ]
            self.nr_buffer[:] = input_wav[self.block_frame:]
            self.input_wav_res[-self.block_frame_16k - 160:] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc:]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1):] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc:])[
                160:
                ]
            )
        # infer
        if self.config.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame:].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame:].clone()
        # output noise reduction
        if self.config.O_noise_reduce and self.config.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                                                      self.block_frame:
                                                      ].clone()
            self.output_buffer[-self.block_frame:] = infer_wav[-self.block_frame:]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.config.rms_mix_rate < 1 and self.config.function == "vc":
            if self.config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame:]
            else:
                input_wav = self.input_wav[self.extra_frame:]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.vc_config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.vc_config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.config.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
                     None, None, : self.sola_buffer_frame + self.sola_search_frame
                     ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input ** 2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.vc_config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        printt("sola_offset = %d", int(sola_offset))
        infer_wav = infer_wav[sola_offset:]
        if "privateuseone" in str(self.vc_config.device) or not self.config.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        self.sola_buffer[:] = infer_wav[
                              self.block_frame: self.block_frame + self.sola_buffer_frame
                              ]

        audio_block = (
            infer_wav[: self.block_frame]
            .repeat(self.config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        if self.is_recording():
            self.config.output_file.write(audio_block)
        else:
            outdata[:] = audio_block

        total_time = time.perf_counter() - start_time
        self.status.latency_stream = int(total_time * 1000)
        printt("Infer time: %.2f", total_time)

    def __soundinput(self):
        with sd.Stream(
            channels=2,
            callback=self.__audio_callback,
            blocksize=self.block_frame,
            samplerate=self.config.samplerate,
            dtype="float32",
        ):
            while self.flag_vc:
                time.sleep(self.config.block_time)
                print("Audio block passed.")

        print("ENDing VC")
