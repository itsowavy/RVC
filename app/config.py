import multiprocessing

from app.utils import load_setting, save_setting
from app.constants import INDEX_DIR_PATH, PTH_DIR_PATH


class Config:
    def __init__(self):
        self.input_device, self.output_device, self.pitch = load_setting()
        self.channels = 2
        self.block_time: float = 0.3
        self.buffer_num: int = 1
        self.threhold: int = -40
        self.crossfade_time: float = 0.15
        self.extra_time: float = 1.5
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.index_rate = 0.3
        self.samplerate: int = 40000
        self.formant = 0.0
        self.pth_path: str = PTH_DIR_PATH
        self.index_path: str = INDEX_DIR_PATH
        self.n_cpu = min(multiprocessing.cpu_count(), 8)
        self.f0method = 'rmvpe'
        self.function = "vc"
        self.rms_mix_rate: float = 0.0
        self.use_pv: bool = False
        self.output_file = None

    def save(self):
        save_setting(self.input_device, self.output_device, self.pitch)

    def get_user_setting(self):
        return self.input_device, self.output_device, self.pitch

    def set_output_file(self, sound_file):
        self.output_file = sound_file


class Status:
    def __init__(self):
        self.flag_stream = False
        self.flag_record = False
        self.flag_conversion = False
        self.latency_stream = 0
