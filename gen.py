import os
import random
import math
from collections import defaultdict, Counter

from model import text_to_speech

# ==========================
# Config
# ==========================
INPUT_FILE = "merged_shuffled_unique_text.txt"
OUTPUT_DIR = "audio"

prompt_wav_path = "tin_nhanh.wav"
prompt_text = "đến tiệm mua chai thuốc trừ sâu, người đàn ông hỏi một câu khiến nhân viên giật mình lấy lại chai thuốc."
inference_timesteps = 64

NUM_SAMPLES = 4000

# Ước lượng tốc độ đọc.
# Nếu thấy text bị xếp sai bucket, chỉnh số này.
# Số càng lớn => cùng một text bị ước lượng ngắn hơn.
CHARS_PER_SECOND = 15.0

# Bucket từ 0-1s, 1-2s, ..., 19-20s, >20s
MAX_BUCKET_SEC = 27

# Chế độ lấy mẫu:
# "balanced"               : cố gắng lấy đều giữa các khoảng
# "balanced_long_less"     : gần đều, nhưng data rất dài sẽ thấp hơn chút
TARGET_MODE = "balanced_long_less"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ==========================
# Helper
# ==========================
def estimate_duration_sec(text: str) -> float:
    """
    Ước lượng duration dựa trên độ dài text.
    Có thể đổi sang word count nếu muốn:
    return len(text.split()) / WORDS_PER_SECOND
    """
    return len(text) / CHARS_PER_SECOND


def get_bucket_id(text: str) -> int:
    """
    Return:
        0  -> 0-1s
        1  -> 1-2s
        ...
        19 -> 19-20s
        20 -> >20s
    """
    est_sec = estimate_duration_sec(text)
    bucket = int(est_sec)

    if bucket >= MAX_BUCKET_SEC:
        return MAX_BUCKET_SEC

    return bucket


def bucket_name(bucket_id: int) -> str:
    if bucket_id >= MAX_BUCKET_SEC:
        return f">{MAX_BUCKET_SEC}s"
    return f"{bucket_id}-{bucket_id + 1}s"


def get_bucket_weight(bucket_id: int) -> float:
    """
    Weight dùng để phân bổ số lượng target cho từng bucket.

    balanced:
        bucket nào cũng weight = 1

    balanced_long_less:
        các bucket ngắn/trung bình gần đều,
        bucket rất dài giảm nhẹ để tránh bị over quá nhiều.
    """
    if TARGET_MODE == "balanced":
        return 1.0

    if TARGET_MODE == "balanced_long_less":
        # 0-10s: gần đều
        if bucket_id <= 10:
            return 1.0

        # Sau 10s giảm nhẹ dần
        # 11s -> 0.95, 12s -> 0.90, ...
        return max(0.45, 1.0 - 0.05 * (bucket_id - 10))

    raise ValueError(f"TARGET_MODE không hợp lệ: {TARGET_MODE}")


def allocate_targets(buckets, total_samples):
    """
    Phân bổ số lượng cần lấy cho từng bucket theo weight,
    nhưng không vượt quá số data hiện có trong bucket.
    Nếu bucket nào thiếu data, phần còn lại được chia sang bucket khác.
    """
    capacities = {b: len(items) for b, items in buckets.items() if len(items) > 0}
    weights = {b: get_bucket_weight(b) for b in capacities}

    targets = {b: 0 for b in capacities}

    remaining_samples = total_samples
    active_buckets = set(capacities.keys())

    while remaining_samples > 0 and active_buckets:
        weight_sum = sum(weights[b] for b in active_buckets)

        raw_allocs = {}
        for b in active_buckets:
            raw = remaining_samples * weights[b] / weight_sum
            raw_allocs[b] = raw

        # Lấy phần nguyên trước
        assigned_this_round = 0
        for b in list(active_buckets):
            cap_left = capacities[b] - targets[b]
            give = min(int(math.floor(raw_allocs[b])), cap_left)

            targets[b] += give
            assigned_this_round += give

        # Phần dư chia tiếp theo fractional part
        leftover = remaining_samples - assigned_this_round

        sorted_buckets = sorted(
            active_buckets,
            key=lambda b: (
                -(raw_allocs[b] - math.floor(raw_allocs[b])),
                -b,  # nếu bằng nhau thì ưu tiên bucket dài hơn
            )
        )

        progress = True
        while leftover > 0 and progress:
            progress = False

            for b in sorted_buckets:
                if leftover <= 0:
                    break

                if targets[b] < capacities[b]:
                    targets[b] += 1
                    leftover -= 1
                    progress = True

        new_total_selected = sum(targets.values())
        remaining_samples = total_samples - new_total_selected

        active_buckets = {
            b for b in active_buckets
            if targets[b] < capacities[b]
        }

        if assigned_this_round == 0 and not progress:
            break

    return targets


def print_distribution(title, ids, lines):
    counter = Counter(get_bucket_id(lines[i]) for i in ids)
    total = len(ids)

    print("\n" + title)
    print("=" * len(title))

    for b in range(MAX_BUCKET_SEC + 1):
        count = counter.get(b, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"{bucket_name(b):>6} : {count:6d} file ({pct:6.2f}%)")


# ==========================
# Tạo thư mục output
# ==========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Đọc toàn bộ text
# ==========================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

if len(lines) < NUM_SAMPLES:
    raise ValueError(
        f"File chỉ có {len(lines)} dòng, không đủ {NUM_SAMPLES} dòng."
    )

# ==========================
# Gom data theo bucket độ dài
# ==========================
buckets = defaultdict(list)

for idx, text in enumerate(lines):
    b = get_bucket_id(text)
    buckets[b].append(idx)

# In phân bố toàn bộ data gốc
all_ids = list(range(len(lines)))
print_distribution("Phân bố toàn bộ data gốc theo duration ước lượng", all_ids, lines)

# ==========================
# Tính số lượng cần lấy mỗi bucket
# ==========================
targets = allocate_targets(buckets, NUM_SAMPLES)

print("\nTarget lấy mẫu theo bucket")
print("=========================")
for b in range(MAX_BUCKET_SEC + 1):
    available = len(buckets.get(b, []))
    target = targets.get(b, 0)
    print(f"{bucket_name(b):>6} : target={target:6d} / available={available:6d}")

# ==========================
# Lấy mẫu theo bucket
# ==========================
selected_ids = []

for b, target_count in targets.items():
    ids_in_bucket = buckets[b]

    if target_count >= len(ids_in_bucket):
        selected_ids.extend(ids_in_bucket)
    else:
        selected_ids.extend(random.sample(ids_in_bucket, target_count))

# Nếu vì lý do nào đó vẫn thiếu, fill thêm bằng weighted sampling ưu tiên data dài
if len(selected_ids) < NUM_SAMPLES:
    selected_set = set(selected_ids)
    remaining_ids = [i for i in range(len(lines)) if i not in selected_set]

    need = NUM_SAMPLES - len(selected_ids)

    # Ưu tiên dòng dài hơn
    remaining_ids.sort(
        key=lambda i: estimate_duration_sec(lines[i]),
        reverse=True
    )

    selected_ids.extend(remaining_ids[:need])

# Nếu bị dư, cắt lại
selected_ids = selected_ids[:NUM_SAMPLES]

# Xáo trộn thứ tự generate
random.shuffle(selected_ids)

print_distribution("Phân bố data đã chọn theo duration ước lượng", selected_ids, lines)

# ==========================
# Generate audio
# ==========================
for idx in selected_ids:
    text = lines[idx]

    txt_path = os.path.join(OUTPUT_DIR, f"{idx}.txt")
    wav_path = os.path.join(OUTPUT_DIR, f"{idx}.wav")

    # Lưu text
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Generating {idx}.wav")

    try:
        text_to_speech(
            texts=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            inference_timesteps=inference_timesteps,
            out_path=wav_path,
            language="none",
        )
    except Exception as e:
        print(f"Error at line {idx}: {e}")

print("Done!")
