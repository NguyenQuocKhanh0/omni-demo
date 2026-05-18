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
        return read_digit_by_digit(clean)

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
    text = text.replace("~", " ").replace("%", " phần trăm ").replace("v.v","vân vân ")
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


# ===== SETTINGS =====
MIN_CHARS = 80
MAX_CHARS = 200
FORCE_PERIOD = False
MAX_PAUSES = 5


def split_text_into_chunks(
    s: str,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    force_period: bool = FORCE_PERIOD,
    max_pauses: int = MAX_PAUSES,
) -> List[str]:
    """
    Tách text thành các chunk phục vụ TTS.

    Ưu tiên cao nhất:
        1. Mỗi chunk nên nằm trong [min_chars, max_chars].
        2. Chỉ cho phép chunk < min_chars khi:
            - toàn bộ text < min_chars, hoặc
            - không có cách cắt/gộp hợp lệ để vừa giữ max_chars vừa kết thúc bằng dấu chấm.
        3. Không để chunk > max_chars, trừ trường hợp bất khả kháng như một từ đơn dài hơn max_chars.

    Ưu tiên tiếp theo:
        - Ưu tiên kết thúc chunk bằng dấu '.'.
        - Nếu không tìm được dấu '.' hợp lệ trong khoảng [min_chars, max_chars],
          mới fallback sang ',' hoặc khoảng trắng.
        - Không tách tại '.' hoặc ',' nếu dấu đó nằm giữa 2 chữ số, ví dụ 3.14, 1,000.
        - max_pauses là ràng buộc phụ: ưu tiên giữ <= max_pauses nếu không làm vỡ min/max.
    """
    if min_chars < 1:
        raise ValueError("min_chars phải >= 1.")
    if max_chars < min_chars:
        raise ValueError("max_chars phải >= min_chars.")
    if max_pauses < 1:
        raise ValueError("max_pauses phải >= 1.")

    s = s or ""
    s = (
        s.replace("\n", ". ")
        .replace(":", ",")
        .replace(". .", ".").replace(". .", ".").replace("..", ".")
        .replace(",.", ", ")
        .replace(", .", ".")
        .replace("“", "")
        .replace("”", "")
    )
    s = re.sub(r"\s+", " ", s).strip()

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

    def count_pauses(text: str) -> int:
        return sum(
            1
            for i, ch in enumerate(text)
            if ch in ".,"
            and not is_number_separator(text, i)
        )

    def strip_end_punct(text: str) -> str:
        return text.strip().rstrip(".,").strip()

    def finalize(text: str, is_last: bool) -> str:
        text = text.strip()
        if not text:
            return ""

        if force_period:
            return strip_end_punct(text) + "."

        if text.endswith("."):
            return text

        if text.endswith(","):
            if is_last:
                return strip_end_punct(text) + "."
            return text

        return text + ("." if is_last else ",")

    def final_len(text: str, is_last: bool) -> int:
        return len(finalize(text, is_last))

    def fits_pause(text: str, is_last: bool) -> bool:
        return count_pauses(finalize(text, is_last)) <= max_pauses

    def tail_is_ok(tail: str) -> bool:
        tail = tail.strip()
        if not tail:
            return True

        return final_len(tail, is_last=True) >= min_chars

    def add_candidate(
        candidates: List[Dict[str, Any]],
        rest: str,
        end: int,
        kind: str,
    ) -> None:
        raw = rest[:end].strip()
        tail = rest[end:].strip()

        if not raw:
            return

        n = final_len(raw, is_last=False)

        if min_chars <= n <= max_chars:
            candidates.append(
                {
                    "end": end,
                    "kind": kind,
                    "raw": raw,
                    "tail": tail,
                    "tail_ok": tail_is_ok(tail),
                    "pause_ok": fits_pause(raw, is_last=False),
                }
            )

    def collect_candidates(rest: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        for i, ch in enumerate(rest):
            if ch in ".," and not is_number_separator(rest, i):
                kind = "period" if ch == "." else "comma"
                add_candidate(candidates, rest, i + 1, kind)

        # Fallback cuối: cắt theo khoảng trắng.
        for i, ch in enumerate(rest):
            if ch.isspace():
                add_candidate(candidates, rest, i, "space")

        return candidates

    def choose_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        kind_rank = {
            "period": 0,
            "comma": 1,
            "space": 2,
        }

        # Ưu tiên:
        # 1. Không làm phần còn lại bị < min_chars.
        # 2. Ít dấu ngắt nhịp hơn max_pauses nếu có thể.
        # 3. Kết thúc bằng '.', rồi ',', rồi khoảng trắng.
        # 4. Trong cùng loại, lấy điểm cắt xa nhất để giảm số chunk.
        for need_tail_ok in (True, False):
            pool = [c for c in candidates if c["tail_ok"]] if need_tail_ok else candidates
            if not pool:
                continue

            for need_pause_ok in (True, False):
                pool2 = [c for c in pool if c["pause_ok"]] if need_pause_ok else pool
                if not pool2:
                    continue

                pool2.sort(key=lambda c: (kind_rank[c["kind"]], -c["end"]))
                return pool2[0]

        return None

    def fallback_cut(rest: str) -> int:
        """
        Khi không tìm được '.', ',' hoặc khoảng trắng nào tạo được chunk trong [min, max],
        vẫn ưu tiên không vượt max_chars.

        Nếu không có khoảng trắng phù hợp, bắt buộc cắt cứng.
        """
        max_raw_len = max_chars - 1

        limit = min(len(rest), max_raw_len)

        for i in range(limit, -1, -1):
            if i < len(rest) and rest[i].isspace():
                raw = rest[:i].strip()
                if raw:
                    return i

        return max(1, max_raw_len)

    chunks: List[str] = []
    rest = s

    while rest:
        rest = rest.strip()
        if not rest:
            break

        rest_final = finalize(rest, is_last=True)

        # Nếu phần còn lại đã không vượt max_chars thì giữ nguyên,
        # kể cả khi nó < min_chars. Đây là trường hợp bất khả kháng ở chunk cuối.
        if len(rest_final) <= max_chars:
            chunks.append(rest_final)
            break

        candidates = collect_candidates(rest)
        chosen = choose_candidate(candidates)

        if chosen is not None:
            raw = chosen["raw"]
            end = chosen["end"]
        else:
            end = fallback_cut(rest)
            raw = rest[:end].strip()

        chunk = finalize(raw, is_last=False)
        chunks.append(chunk)

        rest = rest[end:].strip()

    def try_merge(a: str, b: str) -> Optional[str]:
        b_is_last = b.strip().endswith(".")

        merged_body = (strip_end_punct(a) + " " + b.strip()).strip()
        merged = finalize(merged_body, is_last=b_is_last)

        if len(merged) <= max_chars and count_pauses(merged) <= max_pauses:
            return merged

        return None

    # Gộp các chunk < min_chars nếu có thể mà không vượt max_chars.
    i = 0
    while i < len(chunks) and len(chunks) > 1:
        if len(chunks[i]) >= min_chars:
            i += 1
            continue

        if i + 1 < len(chunks):
            merged = try_merge(chunks[i], chunks[i + 1])
            if merged is not None:
                chunks[i] = merged
                del chunks[i + 1]
                continue

        if i - 1 >= 0:
            merged = try_merge(chunks[i - 1], chunks[i])
            if merged is not None:
                chunks[i - 1] = merged
                del chunks[i]
                i = max(i - 1, 0)
                continue

        i += 1

    return [c for c in chunks if c.strip()]