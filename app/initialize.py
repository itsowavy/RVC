import multiprocessing
from multiprocessing import Queue
from multiprocessing import cpu_count

import torch

from app.interface import Interface
from app.utils import load_latest_speakers, save_speakers_to_json


class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q

    def run(self):
        import numpy as np
        import pyworld

        while 1:
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.opt_q.put(ts)


def initialize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_q, opt_q = _generate_in_out_queue()
    _initialize_harvest_workers(inp_q, opt_q)
    Interface(device=device, inp_q=inp_q, opt_q=opt_q)
    _initialize_speaker_status()


def _generate_in_out_queue():
    inp_q = Queue()
    opt_q = Queue()

    return inp_q, opt_q


def _initialize_harvest_workers(input_queue, output_queue):
    n_cpu = min(cpu_count(), 8)
    for _ in range(n_cpu):
        p = Harvest(input_queue, output_queue)
        p.daemon = True
        p.start()


def _initialize_speaker_status():
    speakers = load_latest_speakers()
    save_speakers_to_json(speakers)