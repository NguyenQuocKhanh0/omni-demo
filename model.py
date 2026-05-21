import time
import threading
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import soundfile as sf
from utils.proccess_text import easy_normalize, split_text_into_chunks, normalize_vietnamese_tts
from utils.detect_english import g2p,to_custom
import soundfile as sf
import os
import shutil
import re  # <-- thêm
import random
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import torchaudio
from omnivoice import OmniVoice
# =========================
# SETTINGS
# =========================


CHECKPOINT_DIR = "kjanh/KhanhTTS-OmniVoice"

DEFAULT_REF_AUDIO = "example/refvoice.wav"
DEFAULT_REF_TEXT = (
    "có người từng nói với cô, đó là hơi thở của mùa đông, "
    "hơi thở của đất trời, hơi thở của tình yêu."
)

USE_MULTI_GPU = True
GPU_IDS = None
# None  -> tự dùng toàn bộ GPU đang có: cuda:0, cuda:1, ...
# [0]   -> chỉ dùng cuda:0
# [0,1] -> dùng cuda:0 và cuda:1

WORKERS_PER_GPU = 1
# Nên để 1 trước.
# Nếu mỗi GPU còn dư VRAM và model generate không ăn hết GPU,
# có thể thử 2, nhưng thường 1 là ổn định nhất.

MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# =========================
# LOAD MODEL PER GPU
# =========================

def get_devices():
    if not torch.cuda.is_available():
        return ["cpu"]

    total_gpu = torch.cuda.device_count()

    if GPU_IDS is None:
        gpu_ids = list(range(total_gpu))
    else:
        gpu_ids = [i for i in GPU_IDS if 0 <= i < total_gpu]

    if not USE_MULTI_GPU:
        gpu_ids = gpu_ids[:1]

    if len(gpu_ids) == 0:
        return ["cpu"]

    return [f"cuda:{i}" for i in gpu_ids]


DEVICES = get_devices()

models = {}
model_locks = {}

for device in DEVICES:
    print(f"Loading OmniVoice on {device}, dtype={MODEL_DTYPE}")

    model = OmniVoice.from_pretrained(
        CHECKPOINT_DIR,
        device_map=device,
        dtype=MODEL_DTYPE,
    )

    if hasattr(model, "eval"):
        model.eval()

    models[device] = model
    model_locks[device] = threading.Lock()

print("Loaded devices:", DEVICES)


# =========================
# HELPER
# =========================

def trim_leading_silence_torch(
    wav: torch.Tensor,
    sample_rate: int,
    silence_thresh: float = 0.086,
    chunk_ms: int = 10,
    extend_ms: int = 20,
    ratio: float = 0.95,  # % sample phải dưới ngưỡng để coi là im lặng
):
    wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
    norm_wav = wav_np / (np.max(np.abs(wav_np)) + 1e-8)

    chunk_size = int(sample_rate * chunk_ms / 1000)
    total_chunks = int(len(norm_wav) / chunk_size)

    start_idx = 0
    for i in range(total_chunks):
        chunk = norm_wav[i * chunk_size : (i + 1) * chunk_size]
        # Tính tỷ lệ sample dưới ngưỡng
        silent_ratio = np.mean(np.abs(chunk) < silence_thresh)
        if silent_ratio < ratio:  # nếu ít hơn 95% sample im lặng → coi là có tiếng
            start_idx = max(0, i * chunk_size - int(sample_rate * extend_ms / 1000))
            break

    return wav[:, start_idx:]


def trim_edge_silence_np(
    audio_np: np.ndarray,
    sample_rate: int,
    silence_thresh: float = 0.086,
    chunk_ms: int = 10,
    extend_ms: int = 20,
    ratio: float = 0.95,
):
    """
    Cắt khoảng lặng ở ĐẦU và CUỐI từng chunk audio.

    extend_ms giúp giữ lại một ít âm thanh trước/sau đoạn có tiếng
    để tránh cắt quá sát vào lời nói.
    """
    audio_np = np.asarray(audio_np, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    if audio_np.size == 0:
        return audio_np.astype(np.float32)

    max_amp = np.max(np.abs(audio_np))
    if max_amp < 1e-8:
        return audio_np.astype(np.float32)

    norm = audio_np / (max_amp + 1e-8)

    chunk_size = max(1, int(sample_rate * chunk_ms / 1000))
    extend_size = max(0, int(sample_rate * extend_ms / 1000))

    first_voice_idx = None
    last_voice_idx = None

    for start in range(0, len(norm), chunk_size):
        end = min(start + chunk_size, len(norm))
        block = norm[start:end]

        if block.size == 0:
            continue

        silent_ratio = np.mean(np.abs(block) < silence_thresh)

        # Nếu không đủ im lặng theo ratio thì xem là bắt đầu/còn lời nói
        if silent_ratio < ratio:
            if first_voice_idx is None:
                first_voice_idx = start
            last_voice_idx = end

    # Không detect được lời nói thì giữ nguyên để tránh xóa nhầm cả chunk
    if first_voice_idx is None or last_voice_idx is None:
        return audio_np.astype(np.float32)

    cut_start = max(0, first_voice_idx - extend_size)
    cut_end = min(len(audio_np), last_voice_idx + extend_size)

    return audio_np[cut_start:cut_end].astype(np.float32)

import re
import re
from typing import List


def trim_internal_silence_segment_torch(
    wav: torch.Tensor,
    sample_rate: int,
    silence_thresh: float = 0.05,
    max_silence_ms: int = 350,
):
    """
    Cắt các đoạn silence liên tục ở GIỮA audio sao cho
    mỗi đoạn silence không dài quá max_silence_ms.
    """
    if wav.dim() == 2:
        wav = wav.squeeze(0)

    wav_np = wav.cpu().numpy().astype(np.float32)

    # Normalize để detect silence ổn định
    max_amp = np.max(np.abs(wav_np)) + 1e-8
    norm = wav_np / max_amp

    max_silence = int(sample_rate * max_silence_ms / 1000)

    segments = []
    i = 0
    T = len(norm)

    while i < T:
        if abs(norm[i]) < silence_thresh:
            # bắt đầu silence segment
            start = i
            while i < T and abs(norm[i]) < silence_thresh:
                i += 1
            end = i
            length = end - start

            if length <= max_silence:
                # giữ nguyên
                segments.append(wav_np[start:end])
            else:
                # giữ phần ĐẦU và CUỐI silence, cắt giữa
                keep = max_silence // 2
                head = wav_np[start : start + keep]
                tail = wav_np[end - (max_silence - keep) : end]
                segments.append(np.concatenate([head, tail]))
        else:
            # voice segment
            start = i
            while i < T and abs(norm[i]) >= silence_thresh:
                i += 1
            segments.append(wav_np[start:i])

    out = np.concatenate(segments) if segments else np.zeros(0, np.float32)
    return torch.from_numpy(out).unsqueeze(0)

def normalize_text_for_tts(chunk, language):
    if language == "auto":
        chunk = easy_normalize(chunk)
        chunk = normalize_vietnamese_tts(chunk)
        text = g2p(chunk)

    elif language == "en":
        # chunk = easy_normalize(chunk)
        text = to_custom(chunk, "en")

    elif language == "none":
        text = chunk

    else:
        chunk = easy_normalize(chunk)
        chunk = normalize_vietnamese_tts(chunk)
        text = chunk

    return text


def audio_to_numpy(audio):
    if isinstance(audio, list):
        audio_t = torch.cat(audio, dim=-1)
    else:
        audio_t = torch.as_tensor(audio)

    audio_t = audio_t.detach().float()

    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)

    return audio_t.squeeze(0).cpu().numpy().astype(np.float32)


def cuda_context(device):
    if str(device).startswith("cuda"):
        gpu_id = int(str(device).split(":")[1])
        return torch.cuda.device(gpu_id)
    return nullcontext()


def autocast_context(device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


# =========================
# MULTI-GPU TEXT TO SPEECH
# =========================

def text_to_speech(
    texts,
    prompt_wav_path=None,
    prompt_text=None,
    inference_timesteps=32,
    out_path="col.wav",
    silence_ms=250,
    speed=1.0,
    language="vi",
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    num_threads=2,
    trim_chunk_silence=True,
    chunk_silence_thresh=0.086,
    chunk_trim_ms=10,
    chunk_extend_ms=20,
    chunk_silence_ratio=0.95,
):
    sr = 24000
    start_time = time.perf_counter()
    texts = texts.replace("AI","ây ai").replace("IT","ai ti")
    chunks = list(split_text_into_chunks(texts))
    silence = np.zeros(int(sr * silence_ms / 1000.0), dtype=np.float32)

    if len(chunks) == 0:
        final = np.zeros(0, np.float32)
        sf.write(out_path, final, sr)

        return {
            "out_path": out_path,
            "rtf": None,
            "elapsed_time": 0.0,
            "audio_duration": 0.0,
            "num_chunks": 0,
            "num_threads": 0,
            "devices": DEVICES,
        }

    # Mỗi GPU tối đa WORKERS_PER_GPU worker.
    # Mặc định WORKERS_PER_GPU=1 để tránh nhiều thread tranh cùng một model/GPU.
    max_workers = min(
        int(num_threads),
        len(chunks),
        len(DEVICES) * int(WORKERS_PER_GPU),
    )

    max_workers = max(1, max_workers)

    def pick_device(worker_index):
        return DEVICES[worker_index % len(DEVICES)]

    def process_chunk(i, chunk, device):
        model = models[device]

        text = normalize_text_for_tts(chunk, language)

        generate_kwargs = dict(
            text=text,
            num_step=inference_timesteps,
            speed=speed,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            layer_penalty_factor=layer_penalty_factor,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
        )

        if prompt_wav_path is not None:
            generate_kwargs["ref_audio"] = prompt_wav_path
            generate_kwargs["ref_text"] = prompt_text
        else:
            generate_kwargs["ref_audio"] = DEFAULT_REF_AUDIO
            generate_kwargs["ref_text"] = DEFAULT_REF_TEXT

        with model_locks[device]:
            with torch.inference_mode():
                with cuda_context(device):
                    with autocast_context(device):
                        audio = model.generate(**generate_kwargs)

        audio_np = audio_to_numpy(audio)

        if trim_chunk_silence:
            audio_np = trim_edge_silence_np(
                audio_np,
                sample_rate=sr,
                silence_thresh=chunk_silence_thresh,
                chunk_ms=chunk_trim_ms,
                extend_ms=chunk_extend_ms,
                ratio=chunk_silence_ratio,
            )

        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

        return i, text, audio_np, device

    results = [None] * len(chunks)

    if len(chunks) > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for i, chunk in enumerate(chunks):
                device = pick_device(i)
                futures.append(
                    executor.submit(process_chunk, i, chunk, device)
                )

            for future in as_completed(futures):
                i, text, audio_np, device = future.result()

                results[i] = {
                    "text": text,
                    "audio": audio_np,
                    "device": device,
                }

                print(f"Done chunk {i + 1}/{len(chunks)} on {device}")

    else:
        device = DEVICES[0]

        for i, chunk in enumerate(chunks):
            i, text, audio_np, device = process_chunk(i, chunk, device)

            results[i] = {
                "text": text,
                "audio": audio_np,
                "device": device,
            }

            print(f"Done chunk {i + 1}/{len(chunks)} on {device}")

    elapsed_time = time.perf_counter() - start_time

    wavs = []

    for i, item in enumerate(results):
        print(item["text"])

        if i > 0:
            wavs.append(silence)

        wavs.append(item["audio"])

    final = np.concatenate(wavs) if wavs else np.zeros(0, np.float32)

    final_t = torch.from_numpy(final).unsqueeze(0)

    final_t = trim_internal_silence_segment_torch(
        final_t,
        sample_rate=sr,
        silence_thresh=0.05,
        max_silence_ms=450,
    )

    final = final_t.squeeze(0).cpu().numpy().astype(np.float32)

    sf.write(out_path, final, sr)

    audio_duration = len(final) / sr
    rtf = elapsed_time / audio_duration if audio_duration > 0 else None

    used_devices = sorted(set(item["device"] for item in results if item is not None))

    return {
        "out_path": out_path,
        "rtf": rtf,
        "elapsed_time": elapsed_time,
        "audio_duration": audio_duration,
        "num_chunks": len(chunks),
        "num_threads": max_workers,
        "devices": used_devices,
    }