# app_local.py
from pathlib import Path
import tempfile
import threading
import queue
import traceback
import gc
import random
import torch
import gradio as gr


# ======================= APP SETTINGS =======================

APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Đặt 5 file wav mẫu vào thư mục samples cùng cấp file app_local.py.
# Có thể đổi đường dẫn tùy ý, ví dụ: "/home/user/voices/wav1.wav".
SAMPLE_REF_VOICES = [
    {"label": "Postcast", "audio_path": APP_DIR / "example" / "podcast.wav", "ref_text": "có người từng nói với cô, đó là hơi thở của mùa đông, hơi thở của đất trời, hơi thở của tình yêu."},
    {"label": "Bóng đá", "audio_path": APP_DIR / "example" / "bongda2.wav", "ref_text": "sai lầm của các cầu thủ đông á thanh hóa vừa rồi là sai lầm của trịnh văn lợi."},
    {"label": "Cờ vua", "audio_path": APP_DIR / "example" / "covua2.wav", "ref_text": "trắng nhanh chóng xuất chiêu la hán đẩy xe rồng xê ép hai chiếu vỗ mặt."},
    {"label": "Cute girl", "audio_path": APP_DIR / "example" / "cutegirl.wav", "ref_text": "tiếp đến là mỗi người sẽ được một bát súp miso."},
    {"label": "Nữ nhẹ nhàng", "audio_path": APP_DIR / "example" / "girlamap.wav", "ref_text": "kiểm soát cảm xúc, thực chất là một quá trình đánh giá lại bản thân, để tìm thấy tự do, thoát khỏi sự cuốn hút của chính bản ngã."},
]

RANDOM_TEXTS = [
    '''Ngày xuân con én đưa thoi,
Thiều quang chín chục đã ngoài sáu mươi.
Cỏ non xanh tận chân trời,
Cành lê trắng điểm một vài bông hoa.

Thanh minh trong tiết tháng ba,
Lễ là tảo mộ hội là đạp Thanh.
Gần xa nô nức yến anh,
Chị em sắm sửa bộ hành chơi xuân.
Dập dìu tài tử giai nhân,
Ngựa xe như nước áo quần như nêm.
Ngổn ngang gò đống kéo lên,
Thoi vàng vó rắc tro tiền giấy bay.
''',
    '''Tà tà bóng ngả về tây,
Chị em thơ thẩn dan tay ra về.
Bước dần theo ngọn tiểu khê,
Lần xem phong cảnh có bề thanh thanh.
Nao nao dòng nước uốn quanh,
Dịp cầu nho nhỏ cuối ghềnh bắc ngang.
Sè sè nấm đất bên đàng,
Dàu dàu ngọn cỏ nửa vàng nửa xanh.
Rằng Sao trong tiết thanh minh,
Mà đây hương khói vắng tanh thế mà?''',
    '''“Break the ice” có nghĩa là phá vỡ sự ngại ngùng khi bắt đầu nói chuyện.
“Once in a blue moon” có nghĩa là một việc rất hiếm khi xảy ra.
“Hit the books” có nghĩa là bắt đầu học tập chăm chỉ.
“Under the weather” có nghĩa là cảm thấy không khỏe.
“Piece of cake” có nghĩa là việc gì đó rất dễ dàng.''',
    '''Dự báo thời tiết 15/5: Hà Nội chạm mốc 38 độ, miền Bắc xuất hiện diễn biến lạ trước ngày mưa giông quay lại''',
    '''Chồng tao nào phải như ai,
Điều này hẳn miệng những người thị phi!
Vội vàng xuống lệnh ra uy,
Đứa thì vả miệng đứa thì bẻ răng.
Trong ngoài kín mít như bưng.
Nào ai còn dám nói năng một lời!
Buồng đào khuya sớm thảnh thơi,
Ra vào một mực nói cười như không.''',
'''Trong kỳ quay thưởng này, mỗi tỉnh sẽ phát hành vé số riêng và công bố kết quả độc lập, tạo nên hệ thống giải thưởng đa dạng và cơ hội trúng thưởng cao hơn cho người tham gia.
Xổ số miền Nam sử dụng loại vé 6 chữ số với mệnh giá phổ biến 10.000 đồng, mỗi kỳ quay có tổng cộng 9 hạng giải từ giải tám đến giải đặc biệt, trong đó giải đặc biệt có giá trị cao nhất lên tới 2 tỷ đồng cho vé trùng khớp toàn bộ dãy số.''',
'''Đêm ấy mưa rất to. Con ngõ nhỏ chỉ còn tiếng nước chảy lộp bộp từ mái tôn xuống chiếc xô nhựa cũ đặt trước cửa.

Ông Thành ngồi một mình trong tiệm sửa đồng hồ đã mở hơn ba mươi năm. Tiệm bé đến mức chỉ cần bước ba bước là chạm bức tường đối diện. Trên tường treo đầy đồng hồ cũ, nhưng lạ nhất là tất cả đều chạy chậm hơn đúng năm phút.

Người trong phố từng hỏi vì sao ông không chỉnh lại.

Ông chỉ cười:
“Chậm một chút cũng tốt.”

Gần nửa đêm, cửa tiệm bật mở. Một cậu bé mặc áo mưa xanh đứng ngoài hiên, tay ôm chiếc đồng hồ quả quýt đã gỉ sét.

“Ông sửa được không ạ?”

Ông Thành đeo kính lên, nhận lấy món đồ. Vừa mở nắp đồng hồ, bàn tay ông bỗng khựng lại.''',
'''The rain kept falling, but somehow the city felt quieter than before.''',

]

# ======================= STYLE =======================

custom_css = """
#app-container { max-width: 1000px; margin: 0 auto; }
.gradio-container { background: radial-gradient(circle at top, #ffffff 0, #f9fafb 55%); color: #111827; }

#title-block h1 {
  font-size: 2.4rem !important;
  font-weight: 800 !important;
  background: linear-gradient(120deg, #f97316, #eab308, #22c55e);
  -webkit-background-clip: text;
  color: transparent;
  text-align: center;
}
#title-block p { text-align:center; font-size: 0.95rem; color: #6b7280; }

.sample-card {
  border-radius: 16px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid rgba(148, 163, 184, 0.6);
  box-shadow: 0 18px 28px rgba(148, 163, 184, 0.35);
}

.sample-voice-btn button {
  min-height: 36px !important;
  padding: 8px 10px !important;
  border-radius: 12px !important;
  font-size: 0.86rem !important;
}
"""

# ======================= SINGLE INFERENCE WORKER =======================

_job_q: "queue.Queue[tuple]" = queue.Queue()
_worker_ready = threading.Event()
_worker_error = {"err": None}


def tts_worker():
    """
    Toàn bộ import/load/compile/infer đều nằm trong cùng 1 thread này.
    """
    try:
        print("🚀 Loading TTS model inside dedicated worker thread...")

        # IMPORT Ở ĐÂY, không import model_cpm ở top-level nữa
        from model import text_to_speech
        from utils.proccess_wav import enhance_ref_audio, transcribe_ref_audio

        # Nếu model_cpm load model global khi import thì nó cũng diễn ra trong thread này
        # Nếu model load lazy ở lần gọi đầu thì vẫn OK vì cũng nằm trong thread này

        print("✅ TTS model loaded inside worker thread!")
        _worker_ready.set()

        while True:
            item = _job_q.get()
            if item is None:
                break

            job_type, payload, resp_q = item

            try:
                if job_type == "tts":
                    ref_audio_path = payload["ref_audio_path"]
                    ref_text = payload["ref_text"]
                    gen_text = payload["gen_text"]
                    steps = payload["steps"]
                    speed = payload["speed"]
                    language = payload["language"]
                    guidance_scale = payload["guidance_scale"]
                    t_shift = payload["t_shift"]
                    layer_penalty_factor = payload["layer_penalty_factor"]
                    position_temperature = payload["position_temperature"]
                    class_temperature = payload["class_temperature"]
                    use_enhance_ref_audio = payload["use_enhance_ref_audio"]

                    if not gen_text.strip():
                        raise ValueError("Please enter text content to generate voice.")

                    if len(gen_text.split()) > 50000:
                        raise ValueError("Text too long (max 500 words).")

                    if not ref_audio_path:
                        ref_text = None
                        prompt_wav_path = None
                    else:
                        need_transcribe_ref_text = not ref_text or not ref_text.strip()
                        enhanced_ref_audio = None

                        # Chỉ enhance khi TTS cần dùng bản enhance, hoặc khi cần transcribe ref text.
                        # Khi bỏ chọn use_enhance_ref_audio:
                        # - text_to_speech dùng ref_audio_path gốc
                        # - transcribe_ref_audio vẫn dùng enhanced_ref_audio nếu cần infer text
                        if use_enhance_ref_audio or need_transcribe_ref_text:
                            enhanced_ref_audio = enhance_ref_audio(ref_audio_path)

                        if need_transcribe_ref_text:
                            ref_text = transcribe_ref_audio(enhanced_ref_audio)
                            if not ref_text:
                                raise ValueError("Không nhận dạng được Reference Text.")

                        # if not ref_text.strip().endswith((".", ",")):
                        #     ref_text += "."
                        ref_text = ref_text.strip()
                        # print(ref_text)

                        prompt_wav_path = enhanced_ref_audio if use_enhance_ref_audio else ref_audio_path

                    fd, out_path = tempfile.mkstemp(suffix=".wav")

                    tts_result = text_to_speech(
                        texts=gen_text,
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=ref_text,
                        inference_timesteps=int(steps),
                        speed=float(speed),
                        language=language,
                        guidance_scale=guidance_scale,
                        t_shift=t_shift,
                        layer_penalty_factor=layer_penalty_factor,
                        position_temperature=position_temperature,
                        class_temperature=class_temperature,
                        out_path=out_path,
                    )

                    if isinstance(tts_result, dict):
                        out_path = tts_result.get("out_path", out_path)
                        rtf = tts_result.get("rtf")
                        elapsed_time = tts_result.get("elapsed_time")
                        audio_duration = tts_result.get("audio_duration")
                    else:
                        rtf = None
                        elapsed_time = None
                        audio_duration = None

                    if rtf is None:
                        rtf_text = "RTF: N/A"
                    else:
                        rtf_text = f"RTF: {rtf:.3f}x"

                        if elapsed_time is not None and audio_duration is not None:
                            rtf_text += f"\nInfer time: {elapsed_time:.2f}s\nAudio duration: {audio_duration:.2f}s"
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    resp_q.put(("ok", (out_path, rtf_text)))

                elif job_type == "infer_ref_text":
                    ref_audio_path = payload["ref_audio_path"]

                    if not ref_audio_path:
                        raise ValueError("Upload ref audio trước.")

                    enhanced = enhance_ref_audio(ref_audio_path)
                    text = transcribe_ref_audio(enhanced)

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


_worker_thread = threading.Thread(target=tts_worker, daemon=True)
_worker_thread.start()
_worker_ready.wait()

if _worker_error["err"] is not None:
    raise RuntimeError("Worker init failed:\n" + _worker_error["err"])


# ======================= UI HELPERS =======================

def clear_ref_text_on_audio_upload(_):
    return ""


def clear_ref_text_on_audio_clear():
    return ""


def select_sample_voice(sample_index: int):
    item = SAMPLE_REF_VOICES[sample_index]
    audio_path = Path(item["audio_path"]).expanduser()

    if not audio_path.exists():
        raise gr.Error(f"Không tìm thấy file voice mẫu: {audio_path}")

    return str(audio_path), item.get("ref_text", "")


def get_random_text():
    if not RANDOM_TEXTS:
        return ""
    return random.choice(RANDOM_TEXTS)


def run_job(job_type: str, payload: dict):
    resp_q = queue.Queue(maxsize=1)
    _job_q.put((job_type, payload, resp_q))
    status, data = resp_q.get()

    if status == "ok":
        return data

    raise gr.Error(data)


def infer_tts(ref_audio_path, use_enhance_ref_audio, ref_text, gen_text, steps, speed, language, guidance_scale, t_shift, layer_penalty_factor, position_temperature, class_temperature):
    return run_job(
        "tts",
        {
            "ref_audio_path": ref_audio_path,
            "use_enhance_ref_audio": use_enhance_ref_audio,
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
        },
    )


def infer_ref_text_ui(ref_audio_path):
    return run_job(
        "infer_ref_text",
        {
            "ref_audio_path": ref_audio_path,
        },
    )


# ======================= UI =======================

def build_demo():
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        with gr.Column(elem_id="app-container"):
            gr.Markdown(
                """
<div id="title-block">
<h1>🎤 KhanhTTS-OmniVoice – Zero-shot TTS</h1>
<p>sao chép giọng nói chỉ với 1 giây ⚡</p>
</div>
"""
            )

            gr.Markdown("### 🎙️ Chọn nhanh Reference Voice")
            sample_voice_buttons = []
            with gr.Row():
                for sample in SAMPLE_REF_VOICES:
                    sample_voice_buttons.append(
                        gr.Button(
                            sample["label"],
                            size="sm",
                            elem_classes=["sample-voice-btn"],
                        )
                    )

            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(label="🔊 Reference Voice (1s-9s)", type="filepath")
                    use_enhance_ref_audio = gr.Checkbox(
                        value=True,
                        label="Khử nhiễu âm thanh đầu vào",
                    )
                    ref_text = gr.Textbox(label="📝 Reference Text (optional)", lines=3)

                    # Chỉ clear Reference Text khi người dùng upload/clear thủ công.
                    # Không dùng .change ở đây vì .change cũng chạy khi nút voice mẫu set ref_audio.
                    ref_audio.upload(
                        clear_ref_text_on_audio_upload,
                        inputs=ref_audio,
                        outputs=ref_text,
                        queue=False,
                    )
                    ref_audio.clear(
                        clear_ref_text_on_audio_clear,
                        inputs=None,
                        outputs=ref_text,
                        queue=False,
                    )

                    btn_infer_text = gr.Button("✨ Infer Text từ audio (optional)")
                    btn_random_text = gr.Button("🎲 Random Text")
                    gen_text = gr.Textbox(
                        label="📝 Text to Generate",
                        placeholder="Nhập nội dung text bạn muốn tổng hợp...",
                        lines=6,
                        max_length=1000,
                        info=f"Bản demo, tối đa 1000 ký tự để tránh quá tải.",
                    )

                    steps = gr.Slider(8, 64, value=40, step=1, label="Steps (Càng lớn càng tốt nhưng càng chậm)")
                    speed = gr.Slider(0.8, 1.2, value=1.0, step=0.01, label="Speed")

                with gr.Column():
                    output_audio = gr.Audio(label="🎧 Output", type="filepath")
                    rtf_output = gr.Textbox(
                        label="⚡ RTF",
                        value="RTF: ",
                        lines=3,
                        interactive=False,
                    )

                    btn_run = gr.Button("🔥 Generate Voice", variant="primary")
                    language = gr.Dropdown(
                        choices=["auto", "vi", "en", "none"],
                        value="auto",
                        label="Language"
                    )
                    guidance_scale = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Guidance Scale (độ bám giọng ref)")
                    t_shift = gr.Slider(0, 0.5, value=0.1, step=0.01, label="Sự dịch chuyển bước thời gian cho lịch trình nhiễu.")
                    layer_penalty_factor = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="Hình phạt được áp dụng cho các lớp mã sâu hơn, khuyến khích các lớp sớm hơn (thấp hơn) được giải mã trước.")
                    position_temperature = gr.Slider(0.0, 10.0, value=5.0, step=0.1, label="Giá trị càng cao thì tính ngẫu nhiên càng tăng.")
                    class_temperature = gr.Slider(0.0, 10.0, value=0.0, step=0.1, label="Giá trị càng cao thì tính ngẫu nhiên càng tăng.")
                    # btn_run = gr.Button("🔥 Generate Voice", variant="primary")

            for i, btn_sample_voice in enumerate(sample_voice_buttons):
                btn_sample_voice.click(
                    lambda sample_index=i: select_sample_voice(sample_index),
                    inputs=None,
                    outputs=[ref_audio, ref_text],
                    queue=False,
                )

            btn_random_text.click(
                get_random_text,
                inputs=None,
                outputs=gen_text,
                queue=False,
            )

            btn_run.click(
                infer_tts,
                inputs=[
                    ref_audio,
                    use_enhance_ref_audio,
                    ref_text,
                    gen_text,
                    steps,
                    speed,
                    language,
                    guidance_scale,
                    t_shift,
                    layer_penalty_factor,
                    position_temperature,
                    class_temperature,
                ],
                outputs=[output_audio, rtf_output],
                queue=False,
                concurrency_limit=1,
            )

            btn_infer_text.click(
                infer_ref_text_ui,
                inputs=ref_audio,
                outputs=ref_text,
                queue=False,
                concurrency_limit=1,
            )

        gr.HTML(
            """
            <div style="
                margin-top:20px;
                padding:16px 18px;
                border-radius:14px;
                background: #fff7ed;
                border: 1px solid #fed7aa;
                color: #374151;
            ">
            <h3 style="margin-top:0; margin-bottom:8px;">☕ Ủng hộ dự án này</h3>
            <p style="font-size:14px; line-height:1.6; margin-bottom:12px;">
            Việc huấn luyện các mô hình TTS chất lượng cao đòi hỏi tài nguyên GPU đáng kể.
            Nếu bạn thấy mô hình này hữu ích, vui lòng xem xét hỗ trợ quá trình phát triển:
            </p>

            <div style="margin-bottom:12px;">
            <a href="https://buymeacoffee.com/khanh20017n" target="_blank">
                <img
                src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-orange?logo=buy-me-a-coffee"
                alt="Buy Me a Coffee"
                />
            </a>
            </div>

            <img
            src="https://huggingface.co/kjanh/ViVoxCPM-1.5/resolve/main/asserts/aa8d6020dd54530a0a45.jpg"
            width="100"
            style="border-radius:8px; margin-bottom:12px;"
            />

            <p style="font-size:14px; margin-bottom:0;">
            Mọi sự ủng hộ của các bạn là niềm động lực giúp mình phát triển
            các mô hình tốt hơn trong tương lai ❤️
            </p>
            </div>
            """
        )

    return demo


# ======================= MAIN =======================

if __name__ == "__main__":
    demo = build_demo()

    # Không cần queue cho app này nếu bạn muốn tránh mọi rắc rối xung quanh queue
    # demo.queue(max_size=128)

    demo.launch(
        server_name="0.0.0.0",
        server_port=2397,
        share=True,
        max_threads=4,  # tùy, không còn ảnh hưởng tới thread inference chính
    )