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
        chunk = easy_normalize(chunk)
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
    silence_ms=170,
    speed=1.0,
    language="vi",
    guidance_scale=2.0,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    num_threads=2,
):
    sr = 24000
    start_time = time.perf_counter()

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

# def text_to_speech(
#     texts,
#     prompt_wav_path=None,
#     prompt_text=None,
#     inference_timesteps=32,
#     out_path="col.wav",
#     silence_ms=170,
#     speed=1.0,
#     language="vi",
#     guidance_scale=2.0,
#     t_shift=0.1,
#     layer_penalty_factor=5.0,
#     position_temperature=5.0,
#     class_temperature=0.0,
# ):
#     sr = 24000

#     silence = np.zeros(int(sr * silence_ms / 1000.0), dtype=np.float32)
#     wavs = []

#     # Nếu có prompt_wav_path thì dùng cố định từ đầu đến cuối
#     fixed_ref_audio_path = prompt_wav_path
#     fixed_ref_text = prompt_text

#     # Nếu không có prompt_wav_path thì sẽ lấy chunk đầu tiên làm ref cố định
#     first_chunk_ref_audio_path = None
#     first_chunk_ref_text = None

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         for i, chunk in enumerate(split_text_into_chunks(texts)):
#             with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
#                 chunk = easy_normalize(chunk)
#                 if language == "auto":
#                     chunk = normalize_vietnamese_tts(chunk)
#                     text = g2p(chunk)
#                 elif language == "en":
#                     text = to_custom(chunk, "en")
#                 elif language == "none":
#                     text = chunk
#                 else:
#                     chunk = normalize_vietnamese_tts(chunk)
#                     text = chunk

#                 print(text)

#                 if fixed_ref_audio_path is not None:
#                     # Có prompt_wav_path: luôn dùng prompt gốc
#                     audio = model.generate(
#                         text=text,
#                         ref_audio=fixed_ref_audio_path,
#                         ref_text=fixed_ref_text,
#                         num_step=inference_timesteps,
#                         speed=speed,
#                         guidance_scale=guidance_scale,
#                         t_shift=t_shift,
#                         layer_penalty_factor=layer_penalty_factor,
#                         position_temperature=position_temperature,
#                         class_temperature=class_temperature,
#                     )

#                 elif first_chunk_ref_audio_path is not None:
#                     # Không có prompt_wav_path:
#                     # Các chunk sau dùng chunk đầu tiên làm ref cố định
#                     audio = model.generate(
#                         text=text,
#                         ref_audio=first_chunk_ref_audio_path,
#                         ref_text=first_chunk_ref_text,
#                         num_step=inference_timesteps,
#                         speed=speed,
#                         guidance_scale=guidance_scale,
#                         t_shift=t_shift,
#                         layer_penalty_factor=layer_penalty_factor,
#                         position_temperature=position_temperature,
#                         class_temperature=class_temperature,
#                     )

#                 else:
#                     # Chunk đầu tiên khi không có prompt_wav_path: sinh random
#                     audio = model.generate(
#                         text=text,
#                         num_step=inference_timesteps,
#                         speed=speed,
#                         guidance_scale=guidance_scale,
#                         t_shift=t_shift,
#                         layer_penalty_factor=layer_penalty_factor,
#                         position_temperature=position_temperature,
#                         class_temperature=class_temperature,
#                     )

#             # audio: (T,) hoặc (1, T)
#             if isinstance(audio, list):
#                 audio_t = torch.cat(audio, dim=-1)
#             else:
#                 audio_t = torch.as_tensor(audio)

#             audio_t = audio_t.float()

#             if audio_t.dim() == 1:
#                 audio_t = audio_t.unsqueeze(0)

#             # Trim silence đầu chỉ cho các chunk sau
#             if i > 0:
#                 audio_t = trim_leading_silence_torch(
#                     audio_t,
#                     sample_rate=sr,
#                     silence_thresh=0.086,
#                     chunk_ms=10,
#                     extend_ms=20,
#                     ratio=0.95,
#                 )

#             audio_np = audio_t.squeeze(0).cpu().numpy().astype(np.float32)

#             if i > 0:
#                 wavs.append(silence)

#             wavs.append(audio_np)

#             # Chỉ lưu chunk đầu tiên làm ref nếu ban đầu không có prompt_wav_path
#             if fixed_ref_audio_path is None and i == 0:
#                 first_chunk_ref_audio_path = os.path.join(tmp_dir, "first_chunk_ref.wav")
#                 sf.write(first_chunk_ref_audio_path, audio_np, sr)
#                 first_chunk_ref_text = text

#         final = np.concatenate(wavs) if wavs else np.zeros(0, np.float32)

#         final_t = torch.from_numpy(final).unsqueeze(0)

#         final_t = trim_internal_silence_segment_torch(
#             final_t,
#             sample_rate=sr,
#             silence_thresh=0.05,
#             max_silence_ms=450,
#         )

#         final = final_t.squeeze(0).cpu().numpy()

#         sf.write(out_path, final, sr)

#     return out_path
