import os
import signal
import sys

from fastapi.middleware.cors import CORSMiddleware

from app.utils import get_io_devices

now_dir = os.getcwd()
sys.path.append(now_dir)
from utils import get_filepath
import uvicorn
from fastapi import FastAPI, HTTPException
from schemas import StreamRequest, SettingResponse, RecordRequest
from interface import Interface
from initialize import initialize

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
def start_up():
    initialize()
    # rvc_for_realtime.manager = Manager()
    # rvc_for_realtime.config = Config()
    # rvc.convert.config = Config()


@app.get("/ping", status_code=200)
def health_check():
    return {"OK"}


@app.post("/turn-off")
def shut_down():
    interface = Interface.get_instance()
    interface.save_setting()
    os.kill(os.getpid(), signal.SIGTERM)


@app.get("/setting", status_code=200, response_model=SettingResponse)
def get_setting():
    interface = Interface.get_instance()
    input_device, output_device, pitch = interface.get_user_setting()
    input_devices, output_devices = get_io_devices()
    return SettingResponse(
        pitch=pitch,
        input_device=input_device,
        output_device=output_device,
        input_devices_list=input_devices,
        output_devices_list=output_devices
    )


@app.get("/stream/latency", status_code=200)
def stream_latency():
    interface = Interface.get_instance()
    return {"latency": interface.get_latency()}


@app.post("/stream", status_code=200)
def stream_start(request: StreamRequest):
    interface = Interface.get_instance()
    if interface.is_streaming():
        return HTTPException(status_code=400, detail="Stream already started.")
    interface.set_stream_config(request)
    interface.save_setting()
    interface.start_vc()

    return {"success": True}


@app.delete("/stream")
def stream_stop():
    interface = Interface.get_instance()
    if not interface.is_streaming():
        return HTTPException(status_code=400, detail="Stream already stopped.")
    interface.stop_vc()

    return {"success": True}


@app.post("/record", status_code=200)
def record_start(request: RecordRequest):
    interface = Interface.get_instance()
    if interface.is_recording():
        return HTTPException(status_code=400, detail="Record already started.")
    filepath = get_filepath(request.save_dir_path)
    interface.set_record_config(request, filepath)
    interface.start_vc(is_record=True)

    return {"success": True}


@app.delete("/record", status_code=200)
def record_stop():
    interface = Interface.get_instance()
    if not interface.is_recording():
        return HTTPException(status_code=400, detail="Record already started.")
    interface.stop_vc()

    return {"success": True}


# @app.post("/convert", status_code=200)
# def convert_file(request: ConvertRequest):
#     get_vc(0.33, 0.33)
#     samplerate, audio_output = vc_single(
#         0, request.source_path, request.pitch, None, "rmvpe", "",
#         0.75, 33, 3, 0.25, 0.33)
#     filepath = get_filepath(request.save_dir_path)
#     save_audio(samplerate, audio_output, filepath)

# return {"success": True}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
