import os
import random

from model import text_to_speech

# ==========================
# Config
# ==========================
INPUT_FILE = "merged_shuffled_unique_text.txt"
OUTPUT_DIR = "audio"

prompt_wav_path = "tin_nhanh.wav"
prompt_text = "đến tiệm mua chai thuốc trừ sâu người đàn ông hỏi một câu khiến nhân viên giật mình lấy lại chai thuốc."
inference_timesteps = 64

NUM_SAMPLES = 1000

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

# Lấy ngẫu nhiên 1000 dòng
# idx là số thứ tự dòng trong file (bắt đầu từ 0)
selected_ids = random.sample(range(len(lines)), NUM_SAMPLES)

# Nếu muốn id bắt đầu từ 1:
# selected_ids = [i + 1 for i in random.sample(range(len(lines)), NUM_SAMPLES)]

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
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            inference_timesteps=inference_timesteps,
            out_path=wav_path,
            language="none",
        )
    except Exception as e:
        print(f"Error at line {idx}: {e}")

print("Done!")
