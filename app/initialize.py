import multiprocessing
from multiprocessing import Queue
from multiprocessing import cpu_count

import torch

from app.interface import Interface


class Harvest(multiprocessing.Process):
    def __init__(self, inp_q, opt_q):
        multiprocessing.Process.__init__(self)
        self.inp_q = inp_q
        self.opt_q = opt_q


def initialize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp_q, opt_q = _generate_in_out_queue()
    _initialize_harvest_workers(inp_q, opt_q)
    interface = Interface(device=device, inp_q=inp_q, opt_q=opt_q)

    return interface


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

