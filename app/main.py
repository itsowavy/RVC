import os
import signal
from multiprocessing import Manager, freeze_support

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
load_dotenv("sha256.env")

if __name__ == '__main__':
    import infer.lib.rtrvc as rtrvc
    from app.speaker import SpeakerStatus
    from app.utils import get_io_devices, load_speakers_from_json, save_speakers_to_json
    from app.service import download_index_pth
    from .setup import initialize, shut_down
    from .interface import Interface
    from .utils import get_filepath, save_audio
from .schemas import StreamRequest, SettingResponse, RecordRequest, SpeakersListResponse, SpeakerResponse, \
    ConvertRequest

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


@app.get("/ping", status_code=200)
def health_check():
    return {"OK"}


@app.post("/turn-off")
def turn_off():
    interface = Interface.get_instance()
    interface.save_setting()
    shut_down()
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


@app.get("/speakers", status_code=200, response_model=SpeakersListResponse)
def get_speakers():
    speakers = load_speakers_from_json()
    speakers_list_resp = [
        SpeakerResponse(name=speaker.name, status=speaker.status.value)
        for speaker in speakers
    ]
    return SpeakersListResponse(speakers=speakers_list_resp)


@app.get("/speakers/downloads/{speaker_name}", status_code=200)
def get_speakers_downloads(speaker_name: str):
    speakers = load_speakers_from_json()
    try:
        speaker = next((speaker for speaker in speakers if speaker.name == speaker_name))
    except StopIteration:
        raise HTTPException(status_code=404, detail="Speaker Not Found.")
    if speaker.status is not SpeakerStatus.UNAVAILABLE:
        raise HTTPException(status_code=400, detail="Speaker Status Error.")
    speaker.status = SpeakerStatus.DOWNLOADING
    save_speakers_to_json(speakers)
    try:
        download_index_pth(speaker_name)
    except Exception:
        speaker.status = SpeakerStatus.UNAVAILABLE
        save_speakers_to_json(speakers)
        raise HTTPException(status_code=500, detail="Speaker Download Failed.")
    speaker.status = SpeakerStatus.AVAILABLE
    save_speakers_to_json(speakers)

    return {"success": True}


@app.get("/stream/latency", status_code=200)
def stream_latency():
    interface = Interface.get_instance()
    return {"latency": interface.get_latency()}


@app.post("/stream", status_code=200)
def stream_start(request: StreamRequest):
    interface = Interface.get_instance()
    if interface.is_streaming():
        return HTTPException(status_code=400, detail="Stream already started.")
    speakers = load_speakers_from_json()
    try:
        speaker = next((speaker for speaker in speakers if speaker.name == request.speaker))
    except StopIteration:
        raise HTTPException(status_code=404, detail="Speaker Not Found.")
    if speaker.status is not SpeakerStatus.AVAILABLE:
        raise HTTPException(status_code=400, detail="Speaker Status Error.")
    interface.set_stream_config(request, speaker.sid)
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
    speakers = load_speakers_from_json()
    try:
        speaker = next((speaker for speaker in speakers if speaker.name == request.speaker))
    except StopIteration:
        raise HTTPException(status_code=404, detail="Speaker Not Found.")
    if speaker.status is not SpeakerStatus.AVAILABLE:
        raise HTTPException(status_code=400, detail="Speaker Status Error.")
    file_path = get_filepath(request.save_dir_path)
    interface.set_record_config(request, file_path, speaker.sid)
    interface.start_vc(is_record=True)

    return {"success": True}


@app.delete("/record", status_code=200)
def record_stop():
    interface = Interface.get_instance()
    if not interface.is_recording():
        return HTTPException(status_code=400, detail="Record already started.")
    interface.stop_vc()

    return {"success": True}


@app.post("/convert", status_code=200)
def convert_file(request: ConvertRequest):
    speakers = load_speakers_from_json()
    try:
        speaker = next((speaker for speaker in speakers if speaker.name == request.speaker))
    except StopIteration:
        raise HTTPException(status_code=404, detail="Speaker Not Found.")
    if speaker.status is not SpeakerStatus.AVAILABLE:
        raise HTTPException(status_code=400, detail="Speaker Status Error.")
    sid = speaker.pth_name
    sid_id = speaker.sid
    interface = Interface.get_instance()
    interface.vc.get_vc(sid, 0.33, 0.33)
    _, (tgt_sr, audio_opt) = interface.convert_single(
        sid=sid_id,
        input_file_path=request.source_path,
        pitch=request.pitch,
        f0_file="",
        f0_method="rmvpe",
        # file_index=os.path.join(INDEX_DIR_PATH, f"{request.speaker}.index"),
        file_index=request.speaker,
        index_rate=0.75,
        filter_radius=33,
        resample_sr=0,
        rms_mix_rate=0.25,
        protect=0.33
    )
    filepath = get_filepath(request.save_dir_path)
    save_audio(tgt_sr, audio_opt, filepath)

    return {"success": True}


def main():
    rtrvc.mm = Manager()
    uvicorn.run(app, host="127.0.0.1", port=28000)


if __name__ == '__main__':
    freeze_support()
    main()
