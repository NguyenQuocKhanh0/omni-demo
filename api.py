# =======================
# SETTINGS
# =======================

APP_TITLE = "KhanhTTS-OmniVoice FastAPI"

MY_AUDIO_DIR = "myAudio"
OUTPUT_DIR = "outputs"
TEMP_DIR = "tmp_uploads"
VOICE_DB_PATH = f"{MY_AUDIO_DIR}/voices.json"

MAX_TEXT_CHARS = 1000000
MAX_TEXT_WORDS = 500000

DEFAULT_STEPS = 20
DEFAULT_SPEED = 1.0
DEFAULT_LANGUAGE = "auto"
DEFAULT_GUIDANCE_SCALE = 2.0
DEFAULT_T_SHIFT = 0.1
DEFAULT_LAYER_PENALTY_FACTOR = 5.0
DEFAULT_POSITION_TEMPERATURE = 5.0
DEFAULT_CLASS_TEMPERATURE = 0.0
DEFAULT_NUM_THREADS = 2

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}


# =======================
# IMPORTS
# =======================

import gc
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import traceback
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =======================
# DIRECTORIES
# =======================

MY_AUDIO_PATH = Path(MY_AUDIO_DIR)
OUTPUT_PATH = Path(OUTPUT_DIR)
TEMP_PATH = Path(TEMP_DIR)
VOICE_DB_FILE = Path(VOICE_DB_PATH)

MY_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
TEMP_PATH.mkdir(parents=True, exist_ok=True)


# =======================
# VOICE DB HELPERS
# =======================

_voice_db_lock = threading.Lock()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def safe_voice_id(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or f"voice_{uuid.uuid4().hex[:8]}"


def load_voice_db() -> Dict[str, Any]:
    if not VOICE_DB_FILE.exists():
        return {}

    try:
        with open(VOICE_DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_voice_db(data: Dict[str, Any]):
    tmp_path = VOICE_DB_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, VOICE_DB_FILE)


def get_voice_by_name_or_id(voice_name: str) -> Dict[str, Any]:
    voice_name = (voice_name or "").strip()
    if not voice_name:
        raise HTTPException(status_code=400, detail="voice_name không được rỗng.")

    with _voice_db_lock:
        db = load_voice_db()

    if voice_name in db:
        return db[voice_name]

    lowered = voice_name.lower()
    for item in db.values():
        if item.get("name", "").lower() == lowered:
            return item

    raise HTTPException(status_code=404, detail=f"Không tìm thấy voice: {voice_name}")


def validate_audio_ext(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"File audio không hợp lệ. Cho phép: {sorted(ALLOWED_AUDIO_EXTS)}",
        )
    return suffix


async def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = validate_audio_ext(upload.filename)
    temp_path = TEMP_PATH / f"{uuid.uuid4().hex}{suffix}"

    with open(temp_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return str(temp_path)


def validate_gen_text(text: str):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập text cần tạo giọng.")

    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text quá dài. Tối đa {MAX_TEXT_CHARS} ký tự.",
        )

    if len(text.split()) > MAX_TEXT_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"Text quá dài. Tối đa {MAX_TEXT_WORDS} từ.",
        )


def build_file_url(request: Request, path_prefix: str, filename: str) -> str:
    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/{path_prefix}/{filename}"


# =======================
# WORKER THREAD
# =======================

_job_q = queue.Queue()
_worker_ready = threading.Event()
_worker_error = {"err": None}


def tts_worker():
    """
    Toàn bộ import/load/infer của model nằm trong thread này.
    Không import model ở top-level để tránh load model ngoài worker.
    """
    try:
        print("🚀 Loading TTS model inside dedicated worker thread...")

        import torch
        from model import text_to_speech
        from utils.proccess_wav import enhance_ref_audio, transcribe_ref_audio

        print("✅ TTS model loaded inside worker thread!")
        _worker_ready.set()

        while True:
            item = _job_q.get()
            if item is None:
                break

            job_type, payload, resp_q = item

            try:
                if job_type == "tts":
                    ref_audio_path = payload.get("ref_audio_path")
                    ref_text = payload.get("ref_text")
                    gen_text = payload["gen_text"]

                    use_enhance_ref_audio = bool(payload.get("use_enhance_ref_audio", False))

                    if not gen_text.strip():
                        raise ValueError("Please enter text content to generate voice.")

                    if len(gen_text) > MAX_TEXT_CHARS:
                        raise ValueError(f"Text too long, max {MAX_TEXT_CHARS} chars.")

                    if len(gen_text.split()) > MAX_TEXT_WORDS:
                        raise ValueError(f"Text too long, max {MAX_TEXT_WORDS} words.")

                    enhanced_ref_audio = None

                    if not ref_audio_path:
                        prompt_wav_path = None
                        ref_text = None
                    else:
                        need_transcribe_ref_text = not ref_text or not ref_text.strip()

                        if use_enhance_ref_audio or need_transcribe_ref_text:
                            enhanced_ref_audio = enhance_ref_audio(ref_audio_path)

                        if need_transcribe_ref_text:
                            ref_text = transcribe_ref_audio(enhanced_ref_audio)
                            if not ref_text:
                                raise ValueError("Không nhận dạng được Reference Text.")

                        ref_text = ref_text.strip()
                        prompt_wav_path = enhanced_ref_audio if use_enhance_ref_audio else ref_audio_path

                    fd, out_path = tempfile.mkstemp(suffix=".wav", dir=OUTPUT_DIR)
                    os.close(fd)

                    tts_result = text_to_speech(
                        texts=gen_text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=ref_text,
                        inference_timesteps=int(payload.get("steps", DEFAULT_STEPS)),
                        speed=float(payload.get("speed", DEFAULT_SPEED)),
                        language=payload.get("language", DEFAULT_LANGUAGE),
                        guidance_scale=float(payload.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
                        t_shift=float(payload.get("t_shift", DEFAULT_T_SHIFT)),
                        layer_penalty_factor=float(payload.get("layer_penalty_factor", DEFAULT_LAYER_PENALTY_FACTOR)),
                        position_temperature=float(payload.get("position_temperature", DEFAULT_POSITION_TEMPERATURE)),
                        class_temperature=float(payload.get("class_temperature", DEFAULT_CLASS_TEMPERATURE)),
                        num_threads=int(payload.get("num_threads", DEFAULT_NUM_THREADS)),
                        out_path=out_path,
                    )

                    if isinstance(tts_result, dict):
                        out_path = tts_result.get("out_path", out_path)
                        result = {
                            "out_path": out_path,
                            "rtf": tts_result.get("rtf"),
                            "elapsed_time": tts_result.get("elapsed_time"),
                            "audio_duration": tts_result.get("audio_duration"),
                            "num_chunks": tts_result.get("num_chunks"),
                            "num_threads": tts_result.get("num_threads"),
                            "devices": tts_result.get("devices"),
                        }
                    else:
                        result = {
                            "out_path": out_path,
                            "rtf": None,
                            "elapsed_time": None,
                            "audio_duration": None,
                            "num_chunks": None,
                            "num_threads": None,
                            "devices": None,
                        }

                    if enhanced_ref_audio and enhanced_ref_audio != ref_audio_path:
                        try:
                            os.remove(enhanced_ref_audio)
                        except Exception:
                            pass

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    resp_q.put(("ok", result))

                elif job_type == "prepare_voice":
                    src_audio_path = payload["src_audio_path"]
                    ref_text = payload.get("ref_text")
                    enhance_audio = bool(payload.get("enhance_audio", True))
                    infer_ref_text_if_empty = bool(payload.get("infer_ref_text_if_empty", True))

                    need_transcribe = infer_ref_text_if_empty and (not ref_text or not ref_text.strip())

                    enhanced_audio = None
                    prepared_audio_path = src_audio_path

                    if enhance_audio or need_transcribe:
                        enhanced_audio = enhance_ref_audio(src_audio_path)

                    if enhance_audio:
                        prepared_audio_path = enhanced_audio

                    if need_transcribe:
                        ref_text = transcribe_ref_audio(enhanced_audio)
                        if not ref_text:
                            raise ValueError("Không nhận dạng được Reference Text.")

                    resp_q.put(
                        (
                            "ok",
                            {
                                "prepared_audio_path": prepared_audio_path,
                                "ref_text": (ref_text or "").strip(),
                                "enhanced": bool(enhance_audio),
                            },
                        )
                    )

                elif job_type == "infer_ref_text":
                    ref_audio_path = payload["ref_audio_path"]

                    if not ref_audio_path:
                        raise ValueError("Upload ref audio trước.")

                    enhanced = enhance_ref_audio(ref_audio_path)
                    text = transcribe_ref_audio(enhanced)

                    try:
                        os.remove(enhanced)
                    except Exception:
                        pass

                    if not text:
                        raise ValueError("Không nhận dạng được nội dung.")

                    resp_q.put(("ok", text))

                else:
                    raise ValueError(f"Unknown job_type: {job_type}")

            except Exception:
                resp_q.put(("err", traceback.format_exc()))
            finally:
                _job_q.task_done()

    except Exception:
        _worker_error["err"] = traceback.format_exc()
        _worker_ready.set()


def run_job(job_type: str, payload: dict):
    resp_q = queue.Queue(maxsize=1)
    _job_q.put((job_type, payload, resp_q))
    status, data = resp_q.get()

    if status == "ok":
        return data

    raise HTTPException(status_code=500, detail=data)


_worker_thread = threading.Thread(target=tts_worker, daemon=True)
_worker_thread.start()
_worker_ready.wait()

if _worker_error["err"] is not None:
    raise RuntimeError("Worker init failed:\n" + _worker_error["err"])


# =======================
# FASTAPI APP
# =======================

app = FastAPI(title=APP_TITLE)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/myAudio", StaticFiles(directory=MY_AUDIO_DIR), name="myAudio")


# =======================
# SCHEMAS
# =======================

class TTSRequest(BaseModel):
    gen_text: str = Field(..., description="Text cần tạo giọng.")
    voice_name: Optional[str] = Field("Nam - Trầm Ấm", description="Tên voice hoặc voice_id đã upload.")

    steps: int = DEFAULT_STEPS
    speed: float = DEFAULT_SPEED
    language: str = DEFAULT_LANGUAGE
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    t_shift: float = DEFAULT_T_SHIFT
    layer_penalty_factor: float = DEFAULT_LAYER_PENALTY_FACTOR
    position_temperature: float = DEFAULT_POSITION_TEMPERATURE
    class_temperature: float = DEFAULT_CLASS_TEMPERATURE
    num_threads: int = DEFAULT_NUM_THREADS

    use_enhance_ref_audio: bool = False


# =======================
# ROUTES
# =======================

@app.get("/")
def root():
    return {
        "name": APP_TITLE,
        "status": "ok",
        "docs": "/docs",
        "voices": "/voices",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "worker_alive": _worker_thread.is_alive(),
    }


@app.post("/voices")
async def upload_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
    enhance_audio: bool = Form(True),
    infer_ref_text_if_empty: bool = Form(True),
):
    """
    Upload voice vào thư mục myAudio.

    - name: tên giọng
    - audio: file wav/mp3/flac...
    - ref_text: optional
    - enhance_audio=True: khử nhiễu rồi lưu bản đã khử nhiễu
    - infer_ref_text_if_empty=True: nếu ref_text trống thì tự transcribe
    """
    voice_id = safe_voice_id(name)
    temp_audio_path = await save_upload_to_temp(audio)

    try:
        prepared = run_job(
            "prepare_voice",
            {
                "src_audio_path": temp_audio_path,
                "ref_text": ref_text,
                "enhance_audio": enhance_audio,
                "infer_ref_text_if_empty": infer_ref_text_if_empty,
            },
        )

        prepared_audio_path = prepared["prepared_audio_path"]
        prepared_ref_text = prepared["ref_text"]

        final_ext = ".wav" if enhance_audio else Path(audio.filename).suffix.lower()
        final_audio_path = MY_AUDIO_PATH / f"{voice_id}{final_ext}"

        shutil.copy2(prepared_audio_path, final_audio_path)

        item = {
            "voice_id": voice_id,
            "name": name,
            "audio_path": str(final_audio_path),
            "ref_text": prepared_ref_text,
            "enhanced": bool(enhance_audio),
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }

        with _voice_db_lock:
            db = load_voice_db()
            db[voice_id] = item
            save_voice_db(db)

        return {
            "status": "ok",
            "voice": item,
        }

    finally:
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass


@app.get("/voices")
def list_voices(request: Request):
    """
    Xem danh sách các giọng đã upload.
    """
    with _voice_db_lock:
        db = load_voice_db()

    voices = []
    for voice_id, item in db.items():
        filename = Path(item["audio_path"]).name
        voices.append(
            {
                "voice_id": voice_id,
                "name": item.get("name"),
                "ref_text": item.get("ref_text"),
                "enhanced": item.get("enhanced"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "audio_url": build_file_url(request, "myAudio", filename),
            }
        )

    return {
        "count": len(voices),
        "voices": voices,
    }


@app.get("/voices/{voice_name}")
def get_voice(voice_name: str, request: Request):
    item = get_voice_by_name_or_id(voice_name)
    filename = Path(item["audio_path"]).name

    return {
        **item,
        "audio_url": build_file_url(request, "myAudio", filename),
    }


@app.get("/voices/{voice_name}/audio")
def get_voice_audio(voice_name: str):
    item = get_voice_by_name_or_id(voice_name)
    audio_path = Path(item["audio_path"])

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="File audio không tồn tại.")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name,
    )


@app.delete("/voices/{voice_name}")
def delete_voice(voice_name: str):
    item = get_voice_by_name_or_id(voice_name)
    voice_id = item["voice_id"]
    audio_path = Path(item["audio_path"])

    with _voice_db_lock:
        db = load_voice_db()
        db.pop(voice_id, None)
        save_voice_db(db)

    try:
        if audio_path.exists():
            audio_path.unlink()
    except Exception:
        pass

    return {
        "status": "ok",
        "deleted": voice_id,
    }


@app.post("/voices/{voice_name}/infer-text")
def infer_voice_text(voice_name: str):
    """
    Infer lại ref_text từ voice đã lưu.
    """
    item = get_voice_by_name_or_id(voice_name)
    text = run_job(
        "infer_ref_text",
        {
            "ref_audio_path": item["audio_path"],
        },
    )

    item["ref_text"] = text
    item["updated_at"] = now_iso()

    with _voice_db_lock:
        db = load_voice_db()
        db[item["voice_id"]] = item
        save_voice_db(db)

    return {
        "status": "ok",
        "voice_id": item["voice_id"],
        "ref_text": text,
    }


@app.post("/tts")
def tts(req: TTSRequest, request: Request):
    """
    Tạo TTS bằng voice đã upload.

    Nếu không truyền voice_name thì model dùng DEFAULT_REF_AUDIO/DEFAULT_REF_TEXT trong model.py.
    """
    validate_gen_text(req.gen_text)

    ref_audio_path = None
    ref_text = None
    used_voice = None

    if req.voice_name:
        voice = get_voice_by_name_or_id(req.voice_name)
        ref_audio_path = voice["audio_path"]
        ref_text = voice.get("ref_text")
        used_voice = {
            "voice_id": voice["voice_id"],
            "name": voice["name"],
        }

    result = run_job(
        "tts",
        {
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "gen_text": req.gen_text,
            "steps": req.steps,
            "speed": req.speed,
            "language": req.language,
            "guidance_scale": req.guidance_scale,
            "t_shift": req.t_shift,
            "layer_penalty_factor": req.layer_penalty_factor,
            "position_temperature": req.position_temperature,
            "class_temperature": req.class_temperature,
            "num_threads": req.num_threads,
            "use_enhance_ref_audio": req.use_enhance_ref_audio,
        },
    )

    out_path = Path(result["out_path"])
    filename = out_path.name

    return {
        "status": "ok",
        "voice": used_voice,
        "audio_url": build_file_url(request, "outputs", filename),
        "download_url": build_file_url(request, "outputs", filename),
        "rtf": result.get("rtf"),
        "elapsed_time": result.get("elapsed_time"),
        "audio_duration": result.get("audio_duration"),
        "num_chunks": result.get("num_chunks"),
        "num_threads": result.get("num_threads"),
        "devices": result.get("devices"),
    }


@app.post("/tts-with-audio")
async def tts_with_audio(
    request: Request,
    gen_text: str = Form(...),
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    use_enhance_ref_audio: bool = Form(True),
    steps: int = Form(DEFAULT_STEPS),
    speed: float = Form(DEFAULT_SPEED),
    language: str = Form(DEFAULT_LANGUAGE),
    guidance_scale: float = Form(DEFAULT_GUIDANCE_SCALE),
    t_shift: float = Form(DEFAULT_T_SHIFT),
    layer_penalty_factor: float = Form(DEFAULT_LAYER_PENALTY_FACTOR),
    position_temperature: float = Form(DEFAULT_POSITION_TEMPERATURE),
    class_temperature: float = Form(DEFAULT_CLASS_TEMPERATURE),
    num_threads: int = Form(DEFAULT_NUM_THREADS),
):
    """
    Tạo TTS bằng file ref audio upload trực tiếp, không lưu vào myAudio.

    Logic giống Gradio:
    - Nếu ref_text trống thì tự enhance rồi transcribe.
    - Nếu use_enhance_ref_audio=True thì TTS dùng bản đã khử nhiễu.
    - Nếu không có ref_audio thì dùng default ref trong model.py.
    """
    validate_gen_text(gen_text)

    temp_audio_path = None

    try:
        if ref_audio is not None:
            temp_audio_path = await save_upload_to_temp(ref_audio)

        result = run_job(
            "tts",
            {
                "ref_audio_path": temp_audio_path,
                "ref_text": ref_text,
                "gen_text": gen_text,
                "steps": steps,
                "speed": speed,
                "language": language,
                "guidance_scale": guidance_scale,
                "t_shift": t_shift,
                "layer_penalty_factor": layer_penalty_factor,
                "position_temperature": position_temperature,
                "class_temperature": class_temperature,
                "num_threads": num_threads,
                "use_enhance_ref_audio": use_enhance_ref_audio,
            },
        )

        out_path = Path(result["out_path"])
        filename = out_path.name

        return {
            "status": "ok",
            "audio_url": build_file_url(request, "outputs", filename),
            "download_url": build_file_url(request, "outputs", filename),
            "rtf": result.get("rtf"),
            "elapsed_time": result.get("elapsed_time"),
            "audio_duration": result.get("audio_duration"),
            "num_chunks": result.get("num_chunks"),
            "num_threads": result.get("num_threads"),
            "devices": result.get("devices"),
        }

    finally:
        if temp_audio_path:
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


@app.get("/outputs/{filename}")
def download_output(filename: str):
    file_path = OUTPUT_PATH / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file output.")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )
    
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8618,
        workers=1,
    )