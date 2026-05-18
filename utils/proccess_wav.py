from typing import List, Tuple
import numpy as np
from pydub import AudioSegment
import os
from chunkformer import ChunkFormerModel
from clearvoice import ClearVoice
# ======================= ASR + CLEARVOICE + AUDIO PROCESSING =======================

ASR_MODEL = None
CLEARVOICE_MODEL = None
REF_AUDIO_CACHE = {}  # cache: đường dẫn input -> đường dẫn output đã xử lý



def get_asr_model() -> ChunkFormerModel:
    """Lazy-load ChunkFormer (ASR, chạy trên CPU)."""
    global ASR_MODEL
    if ASR_MODEL is None:
        ASR_MODEL = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")
    return ASR_MODEL


def get_clearvoice_model() -> ClearVoice:
    """Lazy-load ClearVoice để khử nhiễu ref audio."""
    global CLEARVOICE_MODEL
    if CLEARVOICE_MODEL is None:
        CLEARVOICE_MODEL = ClearVoice(
            task="speech_enhancement",
            model_names=["MossFormer2_SE_48K"],
        )
    return CLEARVOICE_MODEL

model = get_asr_model()
cv = get_clearvoice_model()


def find_silent_regions(
    audio: AudioSegment,
    silence_thresh: float = 0.05,  # biên độ sau chuẩn hoá [-1, 1]
    chunk_ms: int = 10,
    min_silence_len: int = 200,
) -> List[Tuple[int, int]]:
    """
    Tìm các khoảng lặng (start_ms, end_ms) trong AudioSegment dựa trên biên độ.
    """
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    norm = samples / (2 ** (audio.sample_width * 8 - 1))
    sr = audio.frame_rate

    chunk_size = max(1, int(sr * chunk_ms / 1000))
    total_chunks = len(norm) // chunk_size

    silent_regions: List[Tuple[int, int]] = []
    start = None
    for i in range(total_chunks):
        chunk = norm[i * chunk_size: (i + 1) * chunk_size]
        if chunk.size == 0:
            continue

        if np.all((chunk > -silence_thresh) & (chunk < silence_thresh)):
            if start is None:
                start = i
        else:
            if start is not None:
                dur = (i - start) * chunk_ms
                if dur >= min_silence_len:
                    silent_regions.append((start * chunk_ms, i * chunk_ms))
                start = None

    if start is not None:
        dur = (total_chunks - start) * chunk_ms
        if dur >= min_silence_len:
            silent_regions.append((start * chunk_ms, total_chunks * chunk_ms))

    return silent_regions


def trim_leading_trailing_silence(
    audio: AudioSegment,
    silence_thresh: float = 0.05,
    chunk_ms: int = 10,
    min_silence_len: int = 200,
) -> AudioSegment:
    """
    Bỏ khoảng lặng đầu/cuối file.
    """
    duration = len(audio)
    silent_regions = find_silent_regions(
        audio,
        silence_thresh=silence_thresh,
        chunk_ms=chunk_ms,
        min_silence_len=min_silence_len,
    )

    if not silent_regions:
        return audio

    start_trim = 0
    end_trim = duration

    # khoảng lặng đầu file
    first_start, first_end = silent_regions[0]
    if first_start <= 0:
        start_trim = max(start_trim, first_end)

    # khoảng lặng cuối file
    last_start, last_end = silent_regions[-1]
    if last_end >= duration:
        end_trim = min(end_trim, last_start)

    return audio[start_trim:end_trim]


def compress_internal_silence(
    audio: AudioSegment,
    max_silence_ms: int = 300,
    silence_thresh: float = 0.05,
    chunk_ms: int = 10,
    min_silence_len: int = 50,
) -> AudioSegment:
    """
    Rút ngắn khoảng lặng giữa file:
    - Khoảng lặng <= max_silence_ms: giữ nguyên
    - Khoảng lặng > max_silence_ms: cắt còn max_silence_ms
    """
    duration = len(audio)
    silent_regions = find_silent_regions(
        audio,
        silence_thresh=silence_thresh,
        chunk_ms=chunk_ms,
        min_silence_len=min_silence_len,
    )
    if not silent_regions:
        return audio

    new_audio = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    cursor = 0

    for s_start, s_end in silent_regions:
        # phần có tiếng nói trước khoảng lặng
        if s_start > cursor:
            new_audio += audio[cursor:s_start]

        silence_len = s_end - s_start
        if silence_len <= max_silence_ms:
            new_audio += audio[s_start:s_end]
        else:
            new_audio += audio[s_start: s_start + max_silence_ms]

        cursor = s_end

    # phần còn lại sau khoảng lặng cuối
    if cursor < duration:
        new_audio += audio[cursor:]

    return new_audio


def select_subsegment_by_silence(
    audio: AudioSegment,
    min_len_ms: int = 5000,
    max_len_ms: int = 10000,
    silence_thresh: float = 0.05,
    chunk_ms: int = 10,
    min_silence_len: int = 200,
) -> AudioSegment:
    """
    Nếu audio > max_len_ms, chọn 1 đoạn dài trong khoảng [min_len_ms, max_len_ms],
    cắt tại điểm nằm trong khoảng lặng để tránh cắt dính giọng nói.
    """
    duration = len(audio)
    if duration <= max_len_ms:
        return audio

    silent_regions = find_silent_regions(
        audio,
        silence_thresh=silence_thresh,
        chunk_ms=chunk_ms,
        min_silence_len=min_silence_len,
    )

    if not silent_regions:
        # không tìm được khoảng lặng -> lấy đoạn giữa
        target_len = min(max_len_ms, duration)
        start = max(0, (duration - target_len) // 2)
        end = start + target_len
        return audio[start:end]

    # boundary là midpoint của khoảng lặng (chắc chắn nằm trong vùng im lặng)
    boundaries = [0]
    for s_start, s_end in silent_regions:
        mid = (s_start + s_end) // 2
        if 0 < mid < duration:
            boundaries.append(mid)
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))

    # ưu tiên đoạn đầu tiên thỏa 5–10s
    for i in range(len(boundaries)):
        for j in range(i + 1, len(boundaries)):
            seg_len = boundaries[j] - boundaries[i]
            if min_len_ms <= seg_len <= max_len_ms:
                return audio[boundaries[i]:boundaries[j]]

    # nếu không có đoạn nào nằm trọn trong [min, max], chọn đoạn gần max_len nhất
    best_i, best_j, best_diff = 0, None, None
    for i in range(len(boundaries)):
        for j in range(i + 1, len(boundaries)):
            seg_len = boundaries[j] - boundaries[i]
            if seg_len >= min_len_ms:
                diff = abs(seg_len - max_len_ms)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_i, best_j = i, j

    if best_j is not None:
        return audio[boundaries[best_i]:boundaries[best_j]]

    # fallback cuối cùng
    target_len = min(max_len_ms, duration)
    start = max(0, (duration - target_len) // 2)
    end = start + target_len
    return audio[start:end]


def enhance_ref_audio(input_path: str) -> str:
    """
    Pipeline xử lý WAV cho TTS:
    - ClearVoice khử nhiễu
    - Bỏ khoảng lặng đầu/cuối
    - Rút ngắn khoảng lặng giữa > 0.3s thành 0.3s
    - Nếu audio > 10s: chọn 1 đoạn 5–10s, cắt tại khoảng lặng
    Trả về đường dẫn file wav đã xử lý.
    """
    if not input_path:
        raise ValueError("No input audio path for enhancement.")

    # cache để cùng 1 file không phải xử lý nhiều lần
    if input_path in REF_AUDIO_CACHE:
        return REF_AUDIO_CACHE[input_path]

    

    # 1) khử nhiễu
    try:
        cv_out = cv(input_path=input_path, online_write=False)
        base = os.path.basename(input_path)
        name, ext = os.path.splitext(base)
        if not ext:
            ext = ".wav"
        denoised_path = os.path.join(os.path.dirname(input_path), f"{name}_denoised{ext}")
        cv.write(cv_out, output_path=denoised_path)
    except Exception as e:
        print(f"[ClearVoice] Error during denoising, fallback to original: {e}")
        denoised_path = input_path

    # 2) pydub xử lý khoảng lặng + length
    audio = AudioSegment.from_file(denoised_path)

    # bỏ khoảng lặng đầu/cuối
    audio = trim_leading_trailing_silence(audio)

    # rút ngắn khoảng lặng giữa
    audio = compress_internal_silence(audio, max_silence_ms=300)

    # nếu >10s thì chọn đoạn trong khoảng 5–10s
    audio = select_subsegment_by_silence(audio, min_len_ms=5000, max_len_ms=10000)

    # 3) ghi ra file mới
    enhanced_path = os.path.join(os.path.dirname(denoised_path), f"{name}_enhanced.wav")
    audio.export(enhanced_path, format="wav")

    REF_AUDIO_CACHE[input_path] = enhanced_path
    os.remove(denoised_path)
    return enhanced_path

def split_audio_by_silence(
    audio: AudioSegment,
    silence_thresh: float = 0.025,
    chunk_ms: int = 10,
    min_silence_len: int = 200,
    min_segment_len: int = 200,
) -> List[Tuple[int, int]]:
    """
    Từ AudioSegment, trả về các đoạn có tiếng nói (non-silent)
    được tách bằng khoảng lặng.
    """
    duration = len(audio)
    silent_regions = find_silent_regions(
        audio,
        silence_thresh=silence_thresh,
        chunk_ms=chunk_ms,
        min_silence_len=min_silence_len,
    )

    segments: List[Tuple[int, int]] = []
    cur_start = 0

    for s_start, s_end in silent_regions:
        if cur_start < s_start:
            if s_start - cur_start >= min_segment_len:
                segments.append((cur_start, s_start))
        cur_start = s_end

    if cur_start < duration and duration - cur_start >= min_segment_len:
        segments.append((cur_start, duration))

    # nếu không tìm được đoạn nào, lấy cả file
    if not segments:
        segments.append((0, duration))

    return segments


def transcribe_ref_audio(audio_path: str) -> str:
    """
    ASR theo yêu cầu:
    - Cắt âm thanh theo khoảng lặng
    - ASR từng đoạn
    - Nối text bằng dấu phẩy
    """
    if not audio_path:
        raise ValueError("No audio path for ASR.")

    # model = get_asr_model()
    audio = AudioSegment.from_file(audio_path)
    segments = split_audio_by_silence(audio)

    texts = []
    base, _ = os.path.splitext(audio_path)

    for idx, (start_ms, end_ms) in enumerate(segments):
        seg_audio = audio[start_ms:end_ms]
        seg_path = f"{base}_seg_{idx}.wav"
        seg_audio.export(seg_path, format="wav")

        try:
            transcription = model.endless_decode(
                audio_path=seg_path,
                chunk_size=32,
                left_context_size=0,
                right_context_size=0,
                total_batch_duration=400,
                return_timestamps=False,
            )
        except TypeError:
            transcription = model.endless_decode(
                audio_path=seg_path,
                chunk_size=32,
                left_context_size=0,
                right_context_size=0,
                total_batch_duration=400,
            )

        if isinstance(transcription, str):
            text = transcription
        else:
            text = str(transcription)

        text = text.strip()
        if text:
            texts.append(text)
        os.remove(seg_path)

    return ", ".join(texts) + "."
