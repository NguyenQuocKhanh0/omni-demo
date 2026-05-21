import re
import re
import unicodedata
from typing import List

PHONE_MIN_DIGITS = 9
PHONE_MAX_DIGITS = 11

# ================== CẤU HÌNH ==================
ONSETS = {
    "",
    "b", "B",
    "c", "C", "ch", "Ch",
    "d", "D", "đ", "Đ",
    "g", "G", "gh", "Gh", "gi", "Gi",
    "h", "H",
    "k", "K", "kh", "Kh",
    "l", "L",
    "m", "M",
    "n", "N", "ng", "Ng", "ngh", "Ngh", "nh", "Nh",
    "ph", "Ph",
    "q", "Q", "qu", "Qu",
    "r", "R",
    "s", "S",
    "t", "T", "th", "Th", "tr", "Tr",
    "v", "V",
    "x", "X"
}

CODAS = {"", "c", "ch", "m", "n", "ng", "nh", "p", "t"}

NUCLEI = {
    # nguyên âm đơn
    "a", "ă", "â", "e", "ê", "i", "y", "o", "ô", "ơ", "u", "ư",

    # nguyên âm đôi
    "ai", "ao", "au", "ay", "âu", "ây", "ia", "iê", "yê", "êu", "eo" "iu", "oa", "oe", "oă", "oa", "oi", "oe", "oo", "ôô", "ơi", "ua", "ue", "ua", "uâ", "uă", "uâ", "ui", "ưi", "uo", "ươ", "ưu", "uơ", "uy",

    # nguyên âm ba
    "oai", "oao", "uao", "oeo", "iêu", "yêu", "uôi", "ươu", "uyu", "uyê", "oay", "uây", "ươi",
}

VIETNAMESE_CHARS = set("aăâbcdđeêghiklmnoôơpqrstuưvxy")


def strip_tone_marks(text):
    text = text.lower()
    text = text.replace("à", "a").replace("á", "a").replace("ả", "a").replace("ã", "a").replace("ạ", "a")
    text = text.replace("ằ", "ă").replace("ắ", "ă").replace("ẳ", "ă").replace("ẵ", "ă").replace("ặ", "ă")
    text = text.replace("ầ", "â").replace("ấ", "â").replace("ẩ", "â").replace("ẫ", "â").replace("ậ", "â")
    text = text.replace("è", "e").replace("é", "e").replace("ẻ", "e").replace("ẽ", "e").replace("ẹ", "e")
    text = text.replace("ề", "ê").replace("ế", "ê").replace("ể", "ê").replace("ễ", "ê").replace("ệ", "ê")
    text = text.replace("ì", "i").replace("í", "i").replace("ỉ", "i").replace("ĩ", "i").replace("ị", "i")
    text = text.replace("ò", "o").replace("ó", "o").replace("ỏ", "o").replace("õ", "o").replace("ọ", "o")
    text = text.replace("ồ", "ô").replace("ố", "ô").replace("ổ", "ô").replace("ỗ", "ô").replace("ộ", "ô")
    text = text.replace("ờ", "ơ").replace("ớ", "ơ").replace("ở", "ơ").replace("ỡ", "ơ").replace("ợ", "ơ")
    text = text.replace("ù", "u").replace("ú", "u").replace("ủ", "u").replace("ũ", "u").replace("ụ", "u")
    text = text.replace("ừ", "ư").replace("ứ", "ư").replace("ử", "ư").replace("ữ", "ư").replace("ự", "ư")
    text = text.replace("ỳ", "y").replace("ý", "y").replace("ỷ", "y").replace("ỹ", "y").replace("ỵ", "y")
    return text


def is_valid_vietnamese_syllable(token):
    s = strip_tone_marks(token)

    if not s or not s.isalpha():
        return False

    for ch in s:
        if ch not in VIETNAMESE_CHARS:
            return False

    onsets_sorted = sorted({x.lower() for x in ONSETS}, key=len, reverse=True)
    codas_sorted = sorted(CODAS, key=len, reverse=True)

    for onset in onsets_sorted:
        if not s.startswith(onset):
            continue

        rest = s[len(onset):]

        for coda in codas_sorted:
            if coda:
                if not rest.endswith(coda):
                    continue
                nucleus = rest[:-len(coda)]
            else:
                nucleus = rest

            if not nucleus:
                continue

            if nucleus not in NUCLEI:
                continue

            # một vài ràng buộc chính tả cơ bản
            if onset == "q" and not nucleus.startswith("u"):
                continue
            if onset == "qu" and nucleus.startswith("u"):
                # tránh kiểu "quu..."
                continue

            if onset == "gh" and nucleus[:1] not in {"e", "ê", "i"}:
                continue
            if onset == "ngh" and nucleus[:1] not in {"e", "ê", "i"}:
                continue
            if onset == "g" and nucleus[:1] in {"e", "ê", "i"}:
                continue
            if onset == "ng" and nucleus[:1] in {"e", "ê", "i"}:
                continue

            return True

    return False

# ================== CẤU HÌNH ==================
CHEM_ALLOW_NO_DIGIT = False
FOUR_DIGIT_AS_DIGIT_BY_DIGIT = True

# ================== TỪ ĐỌC CƠ BẢN ==================
DIGIT_NORMAL = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

# Dùng cho dạng đọc từng chữ số
DIGIT_BY_DIGIT = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "lăm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

# Đọc tên chữ cái
LETTER_VI = {
    "A": "a",
    "B": "bê",
    "b": "bê",
    "C": "xê",
    "c": "xê",
    "D": "đê",
    "d": "đê",
    "E": "e",
    "F": "ép",
    "f": "ép",
    "G": "gờ",
    "g": "gờ",
    "H": "hát",
    "h": "hát",
    "I": "i",
    "J": "di",
    "j": "di",
    "K": "ca",
    "k": "ca",
    "L": "lờ",
    "l": "lờ",
    "M": "mờ",
    "m": "mờ",
    "N": "nờ",
    "n": "nờ",
    "O": "ô",
    "P": "pê",
    "p": "pê",
    "Q": "quy",
    "q": "quy",
    "R": "rờ",
    "r": "rờ",
    "S": "ét",
    "s": "ét",
    "T": "tê",
    "t": "tê",
    "U": "u",
    "V": "vê",
    "v": "vê",
    "W": "vê kép",
    "w": "vê kép",
    "X": "ích",
    "x": "ích",
    "Y": "i",
    "Z": "dét",
    "z": "dét",
}

SCALE_NAMES = [
    "",
    "nghìn",
    "triệu",
    "tỷ",
    "nghìn tỷ",
    "triệu tỷ",
    "tỷ tỷ",
]

COMMON_VI_PRONOUNCEABLE = {
    "ai", "ao", "ba", "be", "bo", "bu", "ca", "co", "cu", "da", "de", "do", "du",
    "ga", "go", "ha", "he", "ho", "la", "le", "lo", "lu", "ma", "me", "mo", "mu",
    "na", "ne", "no", "nu", "pa", "po", "ra", "re", "ro", "sa", "se", "so", "ta",
    "te", "to", "ti", "tu", "va", "ve", "vo", "xa", "xe", "xo"
}


# ================== ĐỌC SỐ ==================
def read_digit_normal(ch):
    return DIGIT_NORMAL[str(ch)]


def read_digit_by_digit(num_str):
    return " ".join(DIGIT_BY_DIGIT[ch] for ch in str(num_str))


def read_unit_after_tens(unit):
    unit = int(unit)

    if unit == 1:
        return "mốt"
    if unit == 4:
        return "tư"
    if unit == 5:
        return "lăm"

    return DIGIT_NORMAL[str(unit)]


def read_two_digits(n):
    n = int(n)

    if n < 10:
        return DIGIT_NORMAL[str(n)]

    ten = n // 10
    unit = n % 10

    if ten == 1:
        if unit == 0:
            return "mười"
        if unit == 5:
            return "mười lăm"
        return "mười " + DIGIT_NORMAL[str(unit)]

    if unit == 0:
        return DIGIT_NORMAL[str(ten)] + " mươi"

    return DIGIT_NORMAL[str(ten)] + " " + read_unit_after_tens(unit)


def read_three_digits(n, force_full=False):
    n = int(n)

    if n == 0:
        return ""

    hundred = n // 100
    rest = n % 100
    parts = []

    if hundred > 0:
        parts.append(DIGIT_NORMAL[str(hundred)] + " trăm")
    elif force_full:
        parts.append("không trăm")

    if rest > 0:
        if rest < 10:
            if hundred > 0 or force_full:
                parts.append("linh " + DIGIT_NORMAL[str(rest)])
            else:
                parts.append(DIGIT_NORMAL[str(rest)])
        else:
            parts.append(read_two_digits(rest))

    return " ".join(parts)


def split_groups_3(num_str):
    num_str = str(num_str)
    groups = []
    while num_str:
        groups.insert(0, num_str[-3:])
        num_str = num_str[:-3]
    return groups


def read_large_number(num_str):
    num_str = str(num_str).replace(".", "")
    num_str = num_str.lstrip("0") or "0"

    if num_str == "0":
        return "không"

    groups = split_groups_3(num_str)
    total_groups = len(groups)
    parts = []

    for i, group in enumerate(groups):
        group_value = int(group)
        if group_value == 0:
            continue

        scale_index = total_groups - i - 1
        scale_name = SCALE_NAMES[scale_index] if scale_index < len(SCALE_NAMES) else ""

        is_first_group = i == 0
        force_full = (not is_first_group) and len(group) == 3 and group_value < 100

        if len(group) <= 2 and is_first_group:
            group_text = read_two_digits(group_value)
        else:
            group_text = read_three_digits(group_value, force_full=force_full)

        if scale_name:
            parts.append(group_text + " " + scale_name)
        else:
            parts.append(group_text)

    return " ".join(parts)


def read_number_auto(num_str, force_large=False):
    raw = str(num_str)

    if "." in raw:
        return read_large_number(raw)

    clean = raw.lstrip("0") or "0"

    if force_large:
        return read_large_number(clean)

    if len(clean) == 1:
        return read_digit_normal(clean)

    if len(clean) == 2:
        return read_two_digits(clean)

    if len(clean) == 3:
        return read_three_digits(clean)

    if len(clean) == 4 and FOUR_DIGIT_AS_DIGIT_BY_DIGIT:
        return read_large_number(clean)

    return read_large_number(clean)


# ================== ĐỌC CHỮ CÁI / TOKEN ==================
def read_letter_vi(ch):
    return LETTER_VI.get(ch, ch)


def read_token_spelled(token):
    return " ".join(read_letter_vi(ch) if ch.isalpha() else DIGIT_NORMAL[ch] for ch in token)

def is_all_consonants(text: str) -> bool:
    for ch in text:
        if not ch.isalpha():  # bỏ qua nếu không phải chữ cái
            return False
        if ch in "aeiouyAEIOUY":  # nguyên âm
            return False
    return len(text) > 0

def is_probably_non_vietnamese_compact_word(token):
    if not token or not token.isalpha():
        return False

    # VIẾT HOA TOÀN BỘ thì xem như viết tắt
        # and not is_valid_vietnamese_syllable(token.lower()) 
    if token.isupper() or sum(1 for c in token if c.isupper()) >= 2:
        return True
    if is_all_consonants(token):
        return True

    
    # Viết thường toàn bộ: chỉ đánh vần nếu không phải âm tiết tiếng Việt
    # if token.islower():
    #     return not is_valid_vietnamese_syllable(token)

    # Viết hoa chữ cái đầu kiểu "Trong", "Học", "Việt"
    # thì thử hạ về chữ thường để kiểm tra âm tiết tiếng Việt
    # lowered = token.lower()
    # if is_valid_vietnamese_syllable(lowered):
    #     return False

    # CamelCase / tên khó đọc / từ ngoại lai thì mới đánh vần
    return False
# ================== CHUẨN HÓA NGÀY THÁNG ==================
import re

# ===== SETTINGS =====
DATE_END_BOUNDARY = r"(?=$|[\s\.,;:!\?…\)\]\}\"'“”‘’»›])"
DATE_START_BOUNDARY = r"(?<![\w/])"


def normalize_dates(text):
    """
    Chuẩn hóa các dạng ngày/tháng phổ biến:
    - ngày 03/02/2026 -> ngày ba tháng hai năm hai không hai sáu
    - 03/02/2026      -> ba tháng hai năm hai không hai sáu
    - 15/05/2026:\n   -> mười lăm tháng năm năm hai không hai sáu:\n
    - tháng 03/2026   -> tháng ba năm hai không hai sáu
    - tháng 2/26      -> tháng hai năm hai mươi sáu

    Cho phép ngày/tháng đứng sát dấu câu, xuống dòng, ngoặc...
    Dấu câu phía sau được giữ nguyên.
    """
    text = str(text)

    def read_year(year: str) -> str:
        year = str(year).strip()
        if len(year) == 4:
            return read_digit_by_digit(year)
        return read_number_auto(year)

    def is_valid_day(day: str) -> bool:
        try:
            return 1 <= int(day) <= 31
        except ValueError:
            return False

    def is_valid_month(month: str) -> bool:
        try:
            return 1 <= int(month) <= 12
        except ValueError:
            return False

    # Dạng:
    # tháng 03/2026
    # tháng 03/2026:
    # tháng 03/2026,\n
    month_year_pattern = re.compile(
        rf"(?i){DATE_START_BOUNDARY}(tháng)\s+(\d{{1,2}})\s*/\s*(\d{{2,4}}){DATE_END_BOUNDARY}"
    )

    def repl_month_year(match):
        month = match.group(2)
        year = match.group(3)

        if not is_valid_month(month):
            return match.group(0)

        return " ".join([
            "tháng",
            read_number_auto(month),
            "năm",
            read_year(year),
        ])

    text = month_year_pattern.sub(repl_month_year, text)

    # Dạng:
    # ngày 03/02/2026
    # 03/02/2026
    # 15/05/2026:\n
    # 03/02.
    # 03/02,
    # 03/02)
    date_pattern = re.compile(
        rf"(?i){DATE_START_BOUNDARY}(ngày\s+)?(\d{{1,2}})\s*/\s*(\d{{1,2}})(?:\s*/\s*(\d{{2,4}}))?{DATE_END_BOUNDARY}"
    )

    def repl_date(match):
        has_ngay = bool(match.group(1))
        day = match.group(2)
        month = match.group(3)
        year = match.group(4)

        if not is_valid_day(day) or not is_valid_month(month):
            return match.group(0)

        parts = []

        if has_ngay:
            parts.append("ngày")

        parts.append(read_number_auto(day))
        parts.append("tháng")
        parts.append(read_number_auto(month))

        if year:
            parts.append("năm")
            parts.append(read_year(year))

        return " ".join(parts)

    return date_pattern.sub(repl_date, text)

# ================== CHUẨN HÓA BIỂU THỨC TOÁN / DẤU + - * x : ==================
def read_left_operand(token):
    token = token.strip()

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+|\d+", token):
        return read_number_auto(token)

    if re.fullmatch(r"[A-Z]", token):
        return read_letter_vi(token)

    if re.fullmatch(r"[A-Za-z]+", token):
        return read_token_spelled(token)

    return token


def read_right_operand(token):
    token = token.strip()

    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+|\d+", token):
        return read_number_auto(token)

    if re.fullmatch(r"[A-Z]", token):
        return read_letter_vi(token)

    if re.fullmatch(r"[A-Za-z]+", token):
        return read_token_spelled(token)

    return token


import re

def normalize_math_operators(text):
    """
    Chỉ đọc toán tử khi nằm sau:
    - một số
    - hoặc một chữ cái in hoa

    và trước:
    - một số
    - hoặc một token chữ

    Riêng dấu '-' nằm sát giữa 2 chữ cái thì coi là dấu nối từ:
    "hà-nội" -> "hà nội"

    Fix lỗi:
    "5 xem á" không được hiểu thành "5 x em á"
    """

    text = str(text)

    # Chữ cái Unicode, gồm cả tiếng Việt
    LETTER = r"[^\W\d_]"
    WORD = rf"{LETTER}+"

    NUMBER = r"\d{1,3}(?:\.\d{3})+|\d+"
    LEFT_OPERAND = rf"(?:{NUMBER}|[A-Z])"
    RIGHT_OPERAND = rf"(?:{NUMBER}|{WORD})"

    # '-' sát giữa 2 chữ cái thì đổi thành khoảng trắng
    text = re.sub(rf"(?<={LETTER})-(?={LETTER})", " ", text)

    # x/X chỉ là toán tử nhân nếu không nằm trong một từ chữ cái
    # Đúng:
    #   5x6
    #   5 x 6
    #   A x B
    # Sai:
    #   5 xem
    #   3 xăng
    OPERATOR = rf"(?:[+\-*:]|(?<!{LETTER})[xX](?!{LETTER}))"

    binary_pattern = re.compile(
        rf"(?<!\w)"
        rf"({LEFT_OPERAND})"
        rf"\s*({OPERATOR})\s*"
        rf"({RIGHT_OPERAND})"
        rf"(?!\w)"
    )

    op_map = {
        "+": "cộng",
        "-": " ",
        "*": "nhân",
        "x": "nhân",
        "X": "nhân",
        ":": "chia",
    }

    def repl_binary(match):
        left = read_left_operand(match.group(1))
        op = op_map[match.group(2)]
        right = read_right_operand(match.group(3))

        if op.strip():
            return f"{left} {op} {right}"
        return f"{left} {right}"

    text = binary_pattern.sub(repl_binary, text)

    unary_pattern = re.compile(
        rf"(?<!\w)"
        rf"({LEFT_OPERAND})"
        rf"\s*([+\-])"
        rf"(?!\w)"
    )

    def repl_unary(match):
        left = read_left_operand(match.group(1))
        op = "cộng" if match.group(2) == "+" else ""

        if op:
            return f"{left} {op}"
        return left

    text = unary_pattern.sub(repl_unary, text)

    # Các dấu + còn lại: đổi thành dấu phẩy
    text = re.sub(r"\s*\+\s*", ", ", text)

    # Các dấu - còn lại: đổi thành dấu phẩy
    text = re.sub(r"\s*-\s*", ", ", text)

    # Dọn khoảng trắng thừa nhẹ
    text = re.sub(r"[ \t]+", " ", text)

    return text

# ================== CHUẨN HÓA CÔNG THỨC HÓA HỌC ==================
def looks_like_chemical_formula(token):
    has_upper = any(ch.isupper() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    has_alpha = any(ch.isalpha() for ch in token)

    if not has_alpha:
        return False

    if not has_upper:
        return False

    if has_digit:
        alpha_count = sum(ch.isalpha() for ch in token)
        digit_count = sum(ch.isdigit() for ch in token)

        if alpha_count == 1 and digit_count >= 3:
            return False

        return True

    return CHEM_ALLOW_NO_DIGIT


def read_chemical_formula(token):
    parts = []
    for ch in token:
        if ch.isalpha():
            parts.append(read_letter_vi(ch))
        elif ch.isdigit():
            parts.append(DIGIT_NORMAL[ch])
        else:
            parts.append(ch)
    return " ".join(parts)


def normalize_chemical_formulas(text):
    pattern = re.compile(r"(?<![\wÀ-ỹ])([A-Za-z0-9]+)(?![\wÀ-ỹ])")

    def repl(match):
        token = match.group(1)
        if looks_like_chemical_formula(token):
            return read_chemical_formula(token)
        return token

    return pattern.sub(repl, text)


# ================== CHUẨN HÓA TỪ VIẾT TẮT / TỪ KHÓ ĐỌC ==================
def normalize_spelled_tokens(text):
    pattern = re.compile(r"(?<![\wÀ-ỹ])([A-Za-z]+(?:['’][A-Za-z]+)*)(?![\wÀ-ỹ])")

    def repl(match):
        token = match.group(1)

        # từ có apostrophe như here's, it's, don't -> giữ nguyên
        if "'" in token or "’" in token:
            return token

        if is_probably_non_vietnamese_compact_word(token):
            return read_token_spelled(token)

        return token

    return pattern.sub(repl, text)


# ================== CHUẨN HÓA SỐ ==================
def normalize_dotted_numbers(text):
    pattern = re.compile(r"(?<![\w])\d{1,3}(?:\.\d{3})+(?![\w])")

    def repl(match):
        return read_large_number(match.group(0))

    return pattern.sub(repl, text)

def read_digit_sequence(num_str):
    """Đọc từng chữ số, giữ cả số 0 ở đầu. Phù hợp cho số điện thoại/mã số."""
    digits = re.sub(r"\D", "", str(num_str))
    return " ".join(DIGIT_NORMAL[ch] for ch in digits)


def normalize_phone_numbers(text):
    """
    Đọc số điện thoại hoặc dãy số bắt đầu bằng 0 theo từng chữ số.
    Mặc định nhận 9-11 chữ số, có thể viết liền hoặc ngăn bằng khoảng trắng, '.', '-'.
    Ví dụ: 0987654321, 0987 654 321, 0987-654-321.
    """
    pattern = re.compile(
        rf"(?<![\w/])0(?:[ .-]?\d){{{PHONE_MIN_DIGITS - 1},{PHONE_MAX_DIGITS - 1}}}(?![\w/])"
    )

    def repl(match):
        return read_digit_sequence(match.group(0))

    return pattern.sub(repl, text)


def normalize_plain_numbers(text):
    pattern = re.compile(r"(?<![\w/])\d+(?![\w/])")

    def repl(match):
        s = match.group(0)
        start, end = match.span()

        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""

        # tránh số thập phân kiểu 3.14
        if prev_char == "." and start >= 2 and text[start - 2].isdigit():
            return s

        if next_char == "." and end + 1 < len(text) and text[end + 1].isdigit():
            return s

        # Dãy số bắt đầu bằng 0 cần đọc từng chữ số để không mất số 0 đầu.
        if len(s) >= 2 and s.startswith("0"):
            return read_digit_sequence(s)

        return read_number_auto(s)

    return pattern.sub(repl, text)

import re

def normalize_case(text):
    def transform(word):
        # Nếu toàn bộ chữ cái đều viết hoa → giữ nguyên
        if word.isalpha() and word.isupper():
            return word
        # Nếu chỉ viết hoa chữ cái đầu → chuyển thành chữ thường
        if word[:1].isupper() and word[1:].islower():
            return word.lower()
        return word

    # Tách theo từ nhưng vẫn giữ dấu câu
    return re.sub(r'\b\w+\b', lambda m: transform(m.group()), text)
# ================== HÀM CHÍNH ==================
import re
import unicodedata

def remove_special_chars_keep_punctuation(text):
    """
    Thay thế kí tự đặc biệt thành dấu phẩy, nhưng giữ lại:
    - tất cả chữ cái Unicode: tiếng Việt, Anh, Trung, Nhật, Hàn, Thái, Nga...
    - dấu thanh / dấu phụ Unicode
    - chữ số
    - khoảng trắng
    - dấu câu

    Thay thành dấu phẩy:
    - emoji
    - kí hiệu đặc biệt
    - kí hiệu tiền tệ
    - kí hiệu toán học
    - control characters
    """

    text = str(text)
    text = unicodedata.normalize("NFC", text)

    kept_chars = []

    for ch in text:
        cat = unicodedata.category(ch)

        # L = Letter
        # M = Mark
        # N = Number
        # P = Punctuation
        # Z = Separator
        if cat[0] in {"L", "M", "N", "P", "Z"}:
            kept_chars.append(ch)
        else:
            kept_chars.append(",")

    text = "".join(kept_chars)
    return text

import re

def normalize_currency(text: str) -> str:
    def replacer(match):
        number = match.group(1)
        symbol = match.group(2)

        if symbol in ["đ", "₫"]:
            return f"{number} đồng"
        elif symbol == "$":
            return f"{number} đô"
        elif symbol == "k":
            return f"{number} ca"
        return match.group(0)

    pattern = r'(\d+(?:\.\d+)?)\s*(đ|₫|\$|k)(?![A-Za-zÀ-ỹ])'
    return re.sub(pattern, replacer, text)

import re
import re

def normalize_number_punctuation(text: str) -> str:
    def is_between_digits(s, idx):
        return (
            idx > 0 and idx < len(s) - 1
            and s[idx - 1].isdigit()
            and s[idx + 1].isdigit()
        )

    def is_valid_thousand_format(s: str, sep: str) -> bool:
        """
        Đúng khi sep là dấu tách hàng nghìn hợp lệ.

        Ví dụ đúng:
        1.000
        12.345
        123.456
        123.456.000

        Ví dụ sai:
        6.1.5
        1.05
        1234.567
        """
        parts = s.split(sep)

        if len(parts) < 2:
            return False

        if not all(part.isdigit() for part in parts):
            return False

        if not (1 <= len(parts[0]) <= 3):
            return False

        return all(len(part) == 3 for part in parts[1:])

    def replace_sep_between_digits(s: str, sep: str, word: str) -> str:
        result = []

        for i, c in enumerate(s):
            if c == sep and is_between_digits(s, i):
                result.append(f" {word} ")
            else:
                result.append(c)

        return ''.join(result)

    def normalize_mixed_separators(token: str) -> str:
        """
        Xử lý số có cả . và ,
        Ví dụ:
        1.234,56  -> 1234 phẩy 56
        1,234.56  -> 1234 chấm 56
        6.1,5     -> 6 chấm 1 phẩy 5
        """

        candidates = [
            {
                "decimal_sep": ",",
                "decimal_word": "phẩy",
                "thousand_sep": ".",
            },
            {
                "decimal_sep": ".",
                "decimal_word": "chấm",
                "thousand_sep": ",",
            },
        ]

        for cfg in candidates:
            decimal_sep = cfg["decimal_sep"]
            thousand_sep = cfg["thousand_sep"]
            decimal_word = cfg["decimal_word"]

            decimal_idx = token.rfind(decimal_sep)

            if decimal_idx == -1:
                continue

            int_part = token[:decimal_idx]
            frac_part = token[decimal_idx + 1:]

            if not frac_part.isdigit() or len(frac_part) == 0:
                continue

            if thousand_sep in int_part:
                if is_valid_thousand_format(int_part, thousand_sep):
                    int_part = int_part.replace(thousand_sep, "")
                    return f"{int_part} {decimal_word} {frac_part}"

            elif int_part.isdigit():
                return f"{int_part} {decimal_word} {frac_part}"

        # Nếu không nhận diện được format chuẩn thì đọc từng dấu giữa số
        token = replace_sep_between_digits(token, ".", "chấm")
        token = replace_sep_between_digits(token, ",", "phẩy")
        return token

    def process_token(token: str) -> str:
        # Chỉ xử lý cụm có số
        if not re.search(r"\d", token):
            return token

        dots_idx = [
            i for i, c in enumerate(token)
            if c == "." and is_between_digits(token, i)
        ]
        commas_idx = [
            i for i, c in enumerate(token)
            if c == "," and is_between_digits(token, i)
        ]

        dots = len(dots_idx)
        commas = len(commas_idx)

        # Không có dấu . hoặc , nằm giữa số
        if dots == 0 and commas == 0:
            return token

        # Chỉ có dấu chấm
        if dots > 0 and commas == 0:
            if is_valid_thousand_format(token, "."):
                return token.replace(".", "")

            return replace_sep_between_digits(token, ".", "chấm")

        # Chỉ có dấu phẩy
        if commas > 0 and dots == 0:
            if is_valid_thousand_format(token, ","):
                return token.replace(",", "")

            return replace_sep_between_digits(token, ",", "phẩy")

        # Có cả dấu chấm và dấu phẩy
        return normalize_mixed_separators(token)

    tokens = re.findall(r"\d+[.,\d]*|\D+", str(text))
    return "".join(process_token(t) for t in tokens)

def remove_all_weird(text: str) -> str:
    return ''.join(
        ch for ch in text
        if ch.isprintable()
    )
    
def normalize_vietnamese_tts(text):
    text = str(text)
    text = normalize_dates(text)
    text = text.replace(">","lớn hơn ").replace("<","nhỏ hơn ").replace("=","bằng ").replace("≥","lớn hơn hoặc bằng ").replace("≤","nhỏ hơn hoặc bằng ")
    text = text.replace("~", " ").replace("%", " phần trăm ").replace("v.v","vân vân ")
    text = re.sub(r'(?<=\d)h(?=\d)', ' giờ ', text)
    text = re.sub(r'(?<=\d)\s*h\b', ' giờ', text)
    text = normalize_currency(text)
    
    # text = normalize_case(text)
    text = text.replace("₂", "2").replace("₃", "3").replace("₄", "4")
    text = text.replace("+"," cộng ")
    
    text = normalize_phone_numbers(text)
    text = normalize_math_operators(text)
    text = normalize_chemical_formulas(text)
    text = normalize_spelled_tokens(text)
    text = normalize_dotted_numbers(text)
    text = normalize_plain_numbers(text)
    text = (
        text.replace("0", "không ")
            .replace("1", "một ")
            .replace("2", "hai ")
            .replace("3", "ba ")
            .replace("4", "bốn ")
            .replace("5", "năm ")
            .replace("6", "sáu ")
            .replace("7", "bảy ")
            .replace("8", "tám ")
            .replace("9", "chín ")
    )
    text = text.replace("(",", ").replace(")", ", ").replace(" ️ "," ")
    text = remove_special_chars_keep_punctuation(text)
    text = text.replace("/"," ").replace("\\"," ")
    # dọn dấu phẩy và khoảng trắng
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,+", ", ", text)

    text = text.replace(",.",".").replace(". ,", ".").replace(",!", "!").replace("! ,", "!").replace(",?", "?").replace("? ,", "?").replace(",;", ";").replace("; ,", ";").replace(",:", ":").replace(": ,", ":")
    text = text.replace(".,", ".").replace(", .", ".").replace(":,",",")
    text = text.replace("  "," ").replace("  "," ")
    text = text.replace("/"," ").replace("\\"," ")
    text = remove_all_weird(text)
    text = text.lower()
    return text

def easy_normalize(text):
    text = text.replace("~", " ").replace("%", " phần trăm ").replace("v.v","vân vân ").replace("@"," a còng ").replace("#"," thăng ").replace("*"," sao ")
    text = normalize_number_punctuation(text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Tách giữa chữ và số
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)

    # Gộp nhiều khoảng trắng liên tiếp
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = text.replace("(",", ").replace(")", ", ").replace(" ️ "," ").replace("_"," ").replace("m²","mét vuông").replace("°C"," độ xê")
    text = remove_special_chars_keep_punctuation(text)
    
    # dọn dấu phẩy và khoảng trắng
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",\s*,+", ", ", text)

    text = text.replace(",.",".").replace(". ,", ".").replace(",!", "!").replace("! ,", "!").replace(",?", "?").replace("? ,", "?").replace(",;", ";").replace("; ,", ";").replace(",:", ":").replace(": ,", ":")
    text = text.replace(".,", ".").replace(", .", ".").replace(":,",",").replace(". .",".")
    text = text.replace("  "," ").replace("  "," ")
    return text



import re
from typing import List

import re
from typing import List, Dict, Any, Optional

import re
from typing import List, Optional

import re
from typing import List, Optional

import re
import math
from typing import List, Optional

# ===== SETTINGS =====
MIN_CHARS = 80
MAX_CHARS = 200
FORCE_PERIOD = False
MAX_PAUSES = 5

# Cho phép thêm tối đa bao nhiêu chunk so với số chunk tối thiểu.
# Tăng số này nếu muốn các đoạn đều hơn nữa, nhưng sẽ tạo nhiều chunk hơn.
# Đặt = 0 nếu muốn số chunk ít nhất có thể.
BALANCE_EXTRA_CHUNKS = 2

SENTENCE_END_PUNCT = ".!?…"
SOFT_SPLIT_PUNCT = ",;:"
CLOSING_CHARS = "”’\"')]}»"


def split_text_into_chunks(
    s: str,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    force_period: bool = FORCE_PERIOD,
    max_pauses: int = MAX_PAUSES,
    balance_extra_chunks: int = BALANCE_EXTRA_CHUNKS,
) -> List[str]:
    """
    Tách text thành các chunk phục vụ TTS.

    Ưu tiên:
        1. Không cắt giữa câu nếu chưa bắt buộc.
        2. Chỉ cắt trong câu khi câu/đoạn đó dài hơn max_chars.
        3. Sau khi có các câu/piece hợp lệ, chọn cách gom sao cho độ dài chunk đồng đều nhất.
        4. max_chars là ràng buộc cứng.
        5. min_chars và max_pauses là ràng buộc mềm.
        6. Không tạo dấu chấm đầu câu, không tạo '..'.
    """
    if min_chars < 1:
        raise ValueError("min_chars phải >= 1.")
    if max_chars < min_chars:
        raise ValueError("max_chars phải >= min_chars.")
    if max_pauses < 1:
        raise ValueError("max_pauses phải >= 1.")
    if balance_extra_chunks < 0:
        raise ValueError("balance_extra_chunks phải >= 0.")

    all_split_punct = SENTENCE_END_PUNCT + SOFT_SPLIT_PUNCT

    def normalize_text(text: str) -> str:
        text = text or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("“", "").replace("”", "")
        text = text.replace(":", ",")

        lines = [line.strip() for line in text.split("\n")]
        parts = []

        for line in lines:
            if not line:
                continue

            if parts:
                prev = parts[-1].rstrip()

                if prev and prev[-1] not in SENTENCE_END_PUNCT + SOFT_SPLIT_PUNCT:
                    parts[-1] = prev + "."

            parts.append(line)

        text = " ".join(parts)

        text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
        text = re.sub(r"([,;:!?])\s*\.", r"\1", text)
        text = re.sub(r"\.\s*\.", ".", text)
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\s+", " ", text).strip()

        text = text.lstrip(" \t\n\r.,;:!?…")

        return text

    s = normalize_text(s)

    if not s:
        return []

    def is_number_separator(text: str, idx: int) -> bool:
        return (
            text[idx] in ".,"
            and idx > 0
            and idx + 1 < len(text)
            and text[idx - 1].isdigit()
            and text[idx + 1].isdigit()
        )

    def core_end_idx(text: str) -> int:
        i = len(text.rstrip()) - 1

        while i >= 0 and text[i] in CLOSING_CHARS:
            i -= 1

        return i

    def ends_with_punct(text: str) -> bool:
        i = core_end_idx(text)

        return i >= 0 and text[i] in all_split_punct

    def replace_last_punct(text: str, new_punct: str) -> str:
        i = core_end_idx(text)

        if i >= 0 and text[i] in all_split_punct:
            return text[:i] + new_punct + text[i + 1:]

        return text.rstrip() + new_punct

    def strip_end_punct(text: str) -> str:
        text = text.strip()

        while text:
            i = core_end_idx(text)

            if i >= 0 and text[i] in all_split_punct:
                text = (text[:i] + text[i + 1:]).strip()
            else:
                break

        return text.strip()

    def clean_chunk_text(text: str) -> str:
        text = (text or "").strip()

        text = text.lstrip(" \t\n\r.,;:!?…")
        text = re.sub(r"\s+([,.;:!?…])", r"\1", text)
        text = re.sub(r"([,;:!?])\s*\.", r"\1", text)
        text = re.sub(r"\.\s*\.", ".", text)
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def finalize(text: str, is_last: bool) -> str:
        text = clean_chunk_text(text)

        if not text:
            return ""

        if force_period:
            return strip_end_punct(text) + "."

        if ends_with_punct(text):
            if is_last:
                i = core_end_idx(text)

                if i >= 0 and text[i] in SOFT_SPLIT_PUNCT:
                    return replace_last_punct(text, ".")

            return text

        return text + ("." if is_last else ",")

    def final_len(text: str, is_last: bool) -> int:
        return len(finalize(text, is_last))

    def count_pauses(text: str) -> int:
        return sum(
            1
            for i, ch in enumerate(text)
            if ch in all_split_punct and not is_number_separator(text, i)
        )

    def split_into_sentences(text: str) -> List[str]:
        sentences = []
        start = 0
        i = 0

        while i < len(text):
            ch = text[i]

            if ch in SENTENCE_END_PUNCT and not is_number_separator(text, i):
                j = i + 1

                while j < len(text) and text[j] in CLOSING_CHARS:
                    j += 1

                if j == len(text) or text[j].isspace():
                    sentence = clean_chunk_text(text[start:j])

                    if sentence:
                        sentences.append(sentence)

                    start = j

                    while start < len(text) and text[start].isspace():
                        start += 1

                    i = start
                    continue

            i += 1

        tail = clean_chunk_text(text[start:])

        if tail:
            sentences.append(tail)

        return sentences

    def find_punct_cut(text: str, limit: int) -> Optional[int]:
        limit = min(limit, len(text))

        for i in range(limit - 1, -1, -1):
            ch = text[i]

            if ch in all_split_punct and not is_number_separator(text, i):
                end = i + 1

                while end < len(text) and end < limit and text[end] in CLOSING_CHARS:
                    end += 1

                return end

        return None

    def find_space_cut(text: str, limit: int) -> Optional[int]:
        limit = min(limit, len(text))

        for i in range(limit, 0, -1):
            if i < len(text) and text[i].isspace():
                raw = text[:i].strip()

                if raw:
                    return i

        return None

    def split_long_sentence(sentence: str) -> List[str]:
        parts = []
        rest = clean_chunk_text(sentence)

        while rest and final_len(rest, is_last=True) > max_chars:
            cut = find_punct_cut(rest, max_chars)

            if cut is None:
                cut = find_space_cut(rest, max(1, max_chars - 1))

            if cut is None:
                cut = max(1, max_chars - 1)

            raw = clean_chunk_text(rest[:cut])

            if not raw:
                raw = clean_chunk_text(rest[:max(1, max_chars - 1)])
                cut = len(raw)

            parts.append(raw)
            rest = clean_chunk_text(rest[cut:])

        if rest:
            parts.append(rest)

        return parts

    def make_segment(pieces: List[str], i: int, j: int) -> str:
        return clean_chunk_text(" ".join(pieces[i:j]))

    def choose_balanced_chunks(pieces: List[str]) -> List[str]:
        """
        Gom các câu/piece thành chunk bằng quy hoạch động.

        Mục tiêu:
            1. Không chunk nào vượt max_chars.
            2. Độ dài các chunk càng đều càng tốt.
            3. Nếu độ đều tương đương, ưu tiên ít vượt max_pauses hơn.
            4. Nếu vẫn tương đương, ưu tiên ít vi phạm min_chars hơn.
        """
        n = len(pieces)

        if n == 0:
            return []

        if n == 1:
            return [pieces[0]]

        segment_cache = {}

        for i in range(n):
            text = ""

            for j in range(i + 1, n + 1):
                if text:
                    text = clean_chunk_text(text + " " + pieces[j - 1])
                else:
                    text = clean_chunk_text(pieces[j - 1])

                non_last_text = finalize(text, is_last=False)
                last_text = finalize(text, is_last=True)

                non_last_len = len(non_last_text)
                last_len = len(last_text)

                if j < n and non_last_len <= max_chars:
                    segment_cache[(i, j, False)] = {
                        "text": text,
                        "final": non_last_text,
                        "length": non_last_len,
                        "pause_over": max(0, count_pauses(non_last_text) - max_pauses),
                        "under": max(0, min_chars - non_last_len),
                    }

                if j == n and last_len <= max_chars:
                    segment_cache[(i, j, True)] = {
                        "text": text,
                        "final": last_text,
                        "length": last_len,
                        "pause_over": max(0, count_pauses(last_text) - max_pauses),
                        "under": max(0, min_chars - last_len),
                    }

                if non_last_len > max_chars and last_len > max_chars:
                    break

        whole_text = make_segment(pieces, 0, n)
        total_len = final_len(whole_text, is_last=True)

        k_min = max(1, math.ceil(total_len / max_chars))
        k_max = min(n, k_min + balance_extra_chunks)

        def solve_for_k(k: int):
            target = total_len / k

            dp = [[None for _ in range(n + 1)] for _ in range(k + 1)]
            prev = [[None for _ in range(n + 1)] for _ in range(k + 1)]

            dp[0][0] = (0.0, 0, 0)

            for kk in range(1, k + 1):
                min_j = kk
                max_j = n - (k - kk)

                for j in range(min_j, max_j + 1):
                    best_score = None
                    best_i = None

                    for i in range(kk - 1, j):
                        if dp[kk - 1][i] is None:
                            continue

                        is_last_segment = j == n
                        info = segment_cache.get((i, j, is_last_segment))

                        if info is None:
                            continue

                        length = info["length"]
                        pause_over = info["pause_over"]
                        under = info["under"]

                        balance_error = (length - target) ** 2

                        prev_score = dp[kk - 1][i]

                        score = (
                            prev_score[0] + balance_error,
                            prev_score[1] + pause_over,
                            prev_score[2] + under,
                        )

                        if best_score is None or score < best_score:
                            best_score = score
                            best_i = i

                    if best_score is not None:
                        dp[kk][j] = best_score
                        prev[kk][j] = best_i

            if dp[k][n] is None:
                return None

            ranges = []
            kk = k
            j = n

            while kk > 0:
                i = prev[kk][j]
                is_last_segment = j == n
                info = segment_cache[(i, j, is_last_segment)]

                ranges.append((i, j, info))

                j = i
                kk -= 1

            ranges.reverse()

            chunks = [item[2]["text"] for item in ranges]
            lengths = [item[2]["length"] for item in ranges]

            pause_over_sum = sum(item[2]["pause_over"] for item in ranges)
            under_sum = sum(item[2]["under"] for item in ranges)

            length_range = max(lengths) - min(lengths)
            balance_sse = sum((length - target) ** 2 for length in lengths)

            global_score = (
                length_range,
                balance_sse,
                pause_over_sum,
                under_sum,
                k,
            )

            return {
                "chunks": chunks,
                "score": global_score,
            }

        results = []

        for k in range(k_min, k_max + 1):
            result = solve_for_k(k)

            if result is not None:
                results.append(result)

        # Nếu chưa tìm được cách chia hợp lệ trong khoảng cho phép,
        # tiếp tục tăng số chunk cho tới khi tìm được.
        if not results:
            for k in range(k_max + 1, n + 1):
                result = solve_for_k(k)

                if result is not None:
                    results.append(result)
                    break

        if not results:
            return pieces

        results.sort(key=lambda item: item["score"])

        return results[0]["chunks"]

    sentences = split_into_sentences(s)

    pieces = []

    for sentence in sentences:
        if final_len(sentence, is_last=True) <= max_chars:
            pieces.append(sentence)
        else:
            pieces.extend(split_long_sentence(sentence))

    raw_chunks = choose_balanced_chunks(pieces)

    chunks = []

    for idx, chunk in enumerate(raw_chunks):
        final_chunk = finalize(chunk, is_last=(idx == len(raw_chunks) - 1))

        if final_chunk:
            chunks.append(final_chunk)

    return chunks