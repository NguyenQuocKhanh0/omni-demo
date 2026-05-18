
from typing import List, Dict, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification
)
import unicodedata
import re
import logging
from typing import List, Optional, Tuple
import re
from itertools import chain
from typing import List, Dict, Optional
import logging
from functools import reduce


LABEL_LIST = ["O", "B-EN", "I-EN"]
LABEL2ID = {l:i for i,l in enumerate(LABEL_LIST)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

model_name = "kjanh/detect_english"
model_detect = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(LABEL_LIST),
    id2label=ID2LABEL, label2id=LABEL2ID
).eval().to("cpu")

tokenizer_detect = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokens_to_pred_spans(offsets: List[Tuple[int,int]], pred_ids: List[int]) -> List[Tuple[int,int]]:
    spans=[]; cur=None
    for (start,end), lid in zip(offsets, pred_ids):
        if start==end: continue
        lab = ID2LABEL.get(lid,"O")
        if lab=="B-EN":
            if cur: spans.append(cur)
            cur=[start,end]
        elif lab=="I-EN":
            if cur: cur[1]=end
            else: cur=[start,end]
        else:
            if cur: spans.append(cur); cur=None
    if cur: spans.append(cur)
    return [tuple(x) for x in spans]
    
def merge_close_spans(spans: List[Dict], max_gap: int = 2) -> List[Dict]:
    if not spans:
        return []
    merged = [spans[0]]
    for cur in spans[1:]:
        prev = merged[-1]
        if cur["start"] - prev["end"] <= max_gap:
            # gộp lại
            prev["end"] = cur["end"]
        else:
            merged.append(cur)
    return merged


def infer_spans(text: str, tokenizer, model, max_length: int = 256) -> List[Dict]:
    text = text.lower()
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()
    with torch.no_grad():
        out = model(**{k: v for k, v in enc.items() if k != "offset_mapping"})
        pred_ids = out.logits.argmax(-1)[0].tolist()
    spans = tokens_to_pred_spans(offsets, pred_ids)
    spans = [{"start": s, "end": e} for (s, e) in spans]
    spans = merge_close_spans(spans, max_gap=2)
    # print(spans)
    return spans


def is_letter(ch: str) -> bool:
    if not ch:
        return False
    # Nếu người dùng lỡ truyền vào tổ hợp có dấu (e + ◌́), chuẩn hoá về NFC:
    ch = unicodedata.normalize("NFC", ch)
    # Chỉ chấp nhận đúng 1 ký tự sau chuẩn hoá
    if len(ch) != 1:
        return False
    # Nhóm 'L*' của Unicode: Lu, Ll, Lt, Lm, Lo
    return unicodedata.category(ch).startswith('L')

# Ví dụ:
# tests = [","]
# print({t: is_letter(t) for t in tests})


def to_custom(text: str, lang: str) -> str:
    if lang.startswith("vi"):
        return text
    elif lang.startswith("en"):
        return " ".join(f"`{word}" for word in text.split())
    return text.replace("``","`")


def flatten(phs):
    """Phẳng hóa list-of-lists (hoặc trả lại list nếu đã phẳng)."""
    if not phs:
        return []
    if isinstance(phs[0], (list, tuple)):
        return list(chain.from_iterable(phs))
    return list(phs)

def g2p_chunk(text: str, lang: str):
    tokens = []
    start = 0
    for t in text:
        if is_letter(t):
            break
        start = start + 1
        
    # Giữ lại: khoảng trắng (\s+), từ (\w+), ký tự khác [^\w\s]
    if start > 0 :
        tokens.extend(flatten(text[0:start]))
    phs = to_custom(text[start:], lang)   # có thể trả về list-of-lists
    tokens.extend(flatten(phs))
    return tokens

TAG_RE = re.compile(r"\[(vi|en(?:-[a-z]{2})?)\]", re.IGNORECASE)

def _norm_lang(tag: str) -> str:
    t = tag.lower()
    if t == "vi":
        return "vi"
    if t == "en":
        return "en-us"          # default cho [en]
    return t                    # en-us / en-gb ...

def _maybe_add_space(tokens_all: List[str], next_chunk: str):
    """Chỉ chèn 1 space khi cần (tránh dính phoneme nếu g2p_chunk strip whitespace)."""
    if not tokens_all:
        return
    last = tokens_all[-1]
    if not last:
        return
    if last[-1].isspace():
        return
    if next_chunk and next_chunk[0].isspace():
        return
    tokens_all.append(" ")

def parse_tagged_segments(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Trả về list (lang, chunk) theo tag.
    - lang=None: đoạn không được gắn tag -> cần auto (infer_spans)
    - lang='vi'/'en-us'...: đoạn đã xác định
    """
    segs: List[Tuple[Optional[str], str]] = []

    cur_lang: Optional[str] = None
    last = 0
    for m in TAG_RE.finditer(text):
        s, e = m.span()
        # text trước tag
        if s > last:
            segs.append((cur_lang, text[last:s]))
        cur_lang = _norm_lang(m.group(1))
        last = e

    # phần còn lại sau tag cuối
    if last < len(text):
        segs.append((cur_lang, text[last:]))

    return segs

def g2p_auto(text: str) -> str:
    """
    Logic cũ: infer_spans để tách các đoạn EN, còn lại coi là VI.
    """
    spans = infer_spans(text, tokenizer_detect, model_detect)
    spans = sorted(spans, key=lambda x: x["start"])

    tokens_all: List[str] = []
    last = 0
    for sp in spans:
        s, e = sp["start"], sp["end"]

        # trước EN -> VI
        if s > last:
            vi_chunk = text[last:s]
            
            if vi_chunk:
                tokens_all.extend(g2p_chunk(vi_chunk, "vi"))

        # EN
        en_chunk = text[s:e]
        if en_chunk:
            _maybe_add_space(tokens_all, en_chunk)
            tokens_all.extend(g2p_chunk(en_chunk, "en-us"))

        last = e

    # đuôi -> VI
    if last < len(text):
        vi_chunk = text[last:]
        
        if vi_chunk:

            tokens_all.extend(g2p_chunk(vi_chunk, "vi"))

    return "".join(tokens_all)

def g2p(text: str) -> str:
    """
    - Nếu có tag [vi]/[en]...: dùng tag để chia.
      + đoạn nào lang=None (chưa tag) => g2p_auto trên đoạn đó
    - Nếu không có tag: g2p_auto toàn chuỗi (như cũ)
    """
    try:
        segs = parse_tagged_segments(text)
        has_any_tag = any(lang is not None for lang, _ in segs)

        if not has_any_tag:
            return g2p_auto(text)

        tokens_all: List[str] = []
        for lang, chunk in segs:
            if not chunk:
                continue

            if lang is None:
                # phần chưa xác định -> auto detect EN/VI
                tokens_all.append(g2p_auto(chunk))
            else:
                # phần đã xác định -> phonemize thẳng
                if lang.startswith("en"):
                    _maybe_add_space(tokens_all, chunk)
                tokens_all.extend(g2p_chunk(chunk, lang))

        return "".join(tokens_all)

    except Exception as ex:
        logging.warning(f"Tokenization of mixed texts failed: {ex}")
        return ""
    

# text = g2p("Xin chào, tôi sẽ go to school vào ngày mai.")
# print(text)
# text = g2p("OmniVoice supports inline non-verbal symbols and pronunciation hints within the input text.")
# print(text)