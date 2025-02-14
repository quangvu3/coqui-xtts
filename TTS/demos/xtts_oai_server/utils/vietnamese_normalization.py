import re
from underthesea import text_normalize

# Dictionary to map numbers to Vietnamese words
number_to_words = {
    0: 'không',
    1: 'một',
    2: 'hai',
    3: 'ba',
    4: 'bốn',
    5: 'năm',
    6: 'sáu',
    7: 'bảy',
    8: 'tám',
    9: 'chín',
    10: 'mười',
    100: 'trăm',
    1000: 'nghìn',
    1000000: 'triệu',
    1000000000: 'tỷ'
}

# Dictionary to map Roman numerals to integers
roman_to_int = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000
}

# Function to convert Roman numerals to integers
def roman_to_integer(roman):
    total = 0
    prev_value = 0
    for char in reversed(roman):
        value = roman_to_int.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total
    
currency_symbols ={
    '~': '~ ',
    '%': 'phần trăm',
    '$': 'đô la',
    '₫': 'đồng',
    'đ': 'đồng',
    '€': 'ơ rô',
    '£': 'bảng',
    '¥': 'yên',
    '₹': 'ru pi',
    '₽': 'rúp',
    '₺': 'li ra',
    '₩': 'uôn',
}

def currency_symbol_to_word(currency_sign):
    if currency_sign in currency_symbols:
        return currency_symbols[currency_sign]
    return currency_sign
    
def detect_number_format(number_str):
    # Check if the number contains a comma and a dot
    if ',' in number_str and '.' in number_str:
        # If the last comma is after the last dot, it's Vietnamese
        if number_str.rfind(',') > number_str.rfind('.'):
            # Validate Vietnamese format
            if re.match(r'^\d{1,3}(?:\.\d{3})*(?:,\d+)?$', number_str):
                return "Vietnamese"
            else:
                return "Invalid"
        # Otherwise, it's US
        else:
            # Validate US format
            if re.match(r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$', number_str):
                return "US"
            else:
                return "Invalid"
    # If only commas are present
    elif ',' in number_str:
        if re.match(r'^\d{1,3}(?:,\d{3})*(?:\.\d+)?$', number_str):
            return "US"
        elif re.match(r'^(\d+,\d+)?$', number_str):
            return "Vietnamese"
        else:
            return "Invalid"
    # If only dots are present
    elif '.' in number_str:
        if re.match(r'^\d{1,3}(?:\.\d{3})*(?:,\d+)?$', number_str):
            return "Vietnamese"
        elif re.match(r'^(\d+\.\d+)?$', number_str):
            return "US"
        else:
            return "Invalid"
    # If no separators are present, assume Vietnamese (default)
    else:
        return "Vietnamese"

# Function to convert numbers to Vietnamese words
def number_to_vietnamese_words(number_str):
    number_str = str(number_str)
    if detect_number_format(number_str) == 'Invalid':
        return number_str
        
    if detect_number_format(number_str) == 'US': # convert US number to Vietnamese one: 1,234.5 to 1234,5
        number = re.sub(r'\.', ',', re.sub(r',', '', number_str))
    else: # remove any dot inside number
        number = re.sub(r'\.', '', number_str)

    if isinstance(number, str) and ',' in number:
        # Handle decimal numbers (e.g., "120,57")
        integer_part, decimal_part = number.split(',')
        integer_words = _convert_integer_part(int(integer_part))
        decimal_words = _convert_decimal_part(decimal_part)
        return f"{integer_words} phẩy {decimal_words}"
    else:
        # Handle integer numbers
        return _convert_integer_part(int(number))

# Helper function to convert the integer part of a number
def _convert_integer_part(number):
    if number == 0:
        return number_to_words[0]

    words = []
    
    # Handle billions
    if number >= 1000000000:
        billion = number // 1000000000
        words.append(_convert_integer_part(billion))
        words.append(number_to_words[1000000000])
        number %= 1000000000
    
    # Handle millions
    if number >= 1000000:
        million = number // 1000000
        words.append(_convert_integer_part(million))
        words.append(number_to_words[1000000])
        number %= 1000000
    
    # Handle thousands
    if number >= 1000:
        thousand = number // 1000
        words.append(_convert_integer_part(thousand))
        words.append(number_to_words[1000])
        number %= 1000
        if number < 100 and number > 0:
            words.append('không trăm')
        if number < 10 and number > 0:
            words.append('không')
    
    # Handle hundreds
    if number >= 100:
        hundred = number // 100
        words.append(number_to_words[hundred])
        words.append(number_to_words[100])
        number %= 100
        if number > 0 and number < 10:
            words.append('lẻ')  # Add "lẻ" for numbers like 106 (một trăm lẻ sáu)
    
    # Handle tens and units
    if number >= 20:
        ten = number // 10
        words.append(number_to_words[ten])
        words.append('mươi')
        number %= 10
    elif number >= 10:
        words.append(number_to_words[10])
        number %= 10
    
    # Handle units (1-9)
    if number > 0:
        if number == 5 and len(words) > 1 and not words[-1] in['lẻ', 'không']: w = 'lăm'
        elif number == 1 and len(words) > 1 and not words[-1] in ['lẻ', 'mười', 'không']: w = 'mốt'
        else:  w = number_to_words[number]
        words.append(w)
    
    return ' '.join(words)


# Helper function to convert the decimal part of a number
def _convert_decimal_part(decimal_part):
    words = []
    for digit in decimal_part:
        words.append(number_to_words[int(digit)])
    return ' '.join(words)

# abbreviation replacement    
abbreviation_map = {
    "AI": "Ây Ai",
    "ASEAN": "A Xê An",
    "ATGT": "An toàn giao thông",
    "BCA": "Bộ Công an",
    "BCH": "Ban chấp hành",
    "BCHTW": "Ban Chấp hành Trung ương",
    "BCT": "Bộ Chính trị",
    "BGD": "Bộ Giáo dục",
    "BKH": "Bộ Khoa học và Công nghệ",
    "BNN": "Bộ Nông nghiệp",
    "BQP": "Bộ Quốc phòng",
    "BTC": "Ban tổ chức",
    "BTL": "Bộ Tư lệnh",
    "BYT": "Bộ Y tế",
    "CA" : "công an",
    "CAND" : "Công an nhân dân",
    "CNCS": "chủ nghĩa cộng sản",
    "CNTB": "chủ nghĩa tư bản",
    "CNXH": "chủ nghĩa xã hội",
    "CNY": "nhân dân tệ",
    "CSGT": "Cảnh sát giao thông",
    "CTN": "Chủ tịch nước",
    "ĐBQH": "Đại biểu Quốc hội",
    "ĐBSCL": "Đồng bằng sông Cửu Long",
    "ĐCS": "Đảng cộng sản",
    "ĐH": "Đại học",
    "ĐHBK": "Đại học Bách khoa",
    "ĐHKHTN": "Đại học Khoa học tự nhiên",
    "ĐHQG": "Đại học Quốc gia",
    "ĐSQ": "Đại sứ quán",
    "EU": "Ơ u",
    "GD": "Giáo dục",
    "HCM": "Hồ Chí Minh",
    "HĐBA": "Hội đồng bảo an",
    "HĐND": "Hội đồng nhân dân",
    "HĐQT": "Hội đồng quản trị",
    "HN": "Hà Nội",
    "HV": "Học viện",
    "KHXH&NV": "Khoa học Xã hội và Nhân văn",
    "KT": "Kinh tế",
    "KTQS": "Kỹ thuật Quân sự",
    "LĐ": "lao động",
    "KHKT": "khoa học kỹ thuật",
    "km": "ki lô mét",
    "LHQ": "Liên Hiệp Quốc",
    "NATO": "Na tô",
    "ND": "nhân dân",
    "NHNN": "ngân hàng nhà nước",
    "NXB": "Nhà xuất bản",
    "PCCC": "Phòng cháy chữa cháy",
    "PTTH": "Phổ thông trung học",
    "PTCS": "Phổ thông cơ sở",
    "QĐND" : "Quân đội nhân dân",
    "QĐNDVN" : "Quân đội nhân dân Việt Nam",
    "QG": "Quốc gia",
    "QK": "Quân khu",
    "sau CN": "sau công nguyên",
    "SG": "Sài Gòn",
    "TAND": "Tòa án nhân dân",
    "TBCN": "tư bản chủ nghĩa",
    "TBT": "Tổng bí thư",
    "TCN": "trước công nguyên",
    "TCT": "Tổng công ty",
    "THCS": "Trung học cơ sở",
    "THPT": "Trung học phổ thông",
    "TNHH": "Trách nhiệm hữu hạn",
    "TNHH MTV": "Trách nhiệm hữu hạn một thành viên",
    "TP": "thành phố",
    "TP.": "thành phố",
    "TPHCM": "Thành phố Hồ Chí Minh",
    "TT": "Thủ tướng",
    "TTCK": "Thị trường chứng khoán",
    "TTTC": "Thị trường tài chính",
    "TTCP": "Thủ tướng chính phủ",
    "TTNT": "Trí tuệ nhân tạo",
    "TTXVN": "Thông tấn xã Việt Nam",
    "TƯ": "Trung ương",
    "TW": "Trung ương",
    "UB": "Ủy ban",
    "UBND": "Ủy ban nhân dân",
    "VH": "Văn hóa",
    "VKSND": "Viện kiểm sát nhân dân",
    "VN": "Việt Nam",
    "VND": "Việt Nam đồng",
    "XH": "Xã hội",
    "XHCN": "xã hội chủ nghĩa",
    "%": "phần trăm",
    "@": "a còng",
    "&": "và",
}

abbreviation_pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in abbreviation_map.keys()) + r')\b')
def replace_abbreviations(text):
    def replacement(match):
        return abbreviation_map[match.group(0)]
    return abbreviation_pattern.sub(replacement, text)


def convert_abbreviations(text):
    """Converts abbreviations like M.A.S.H. to MASH"""
    return re.sub(r"([A-Z]\.){2,}", lambda match: "".join(c for c in match.group(0) if c.isalpha()), text)


# Function to normalize Vietnamese text
def normalize_vietnamese_text(text):
    text = text_normalize(text)
    
    def replace_slash_with_word(text):
        def replacement(match):
            word = match.group(1)
            if word in ['ngày', 'giờ', 'tháng', 'quí', 'quý', 'năm']:
                return f" mỗi {word}"
            else:
                return f" trên {word}"
        return re.sub(r'/(\w+)', replacement, text)

    # find and replace "/word" with "per word"
    text = replace_slash_with_word(text)

    # Convert standalone currency amounts (e.g., $200, ₫200, €50, £75, ¥1000)
    def replace_currency(match):
        currency_sign = match.group(1)
        amount = match.group(2)
        return f"{number_to_vietnamese_words(amount)} {currency_symbol_to_word(currency_sign)}"
    text = re.sub(r'([$₫đ€£¥₹₽₩₺])([\d.,]+)', replace_currency, text)
    
    # (reverse case) convert standalone currency amounts (e.g., 200$, 200đ, 50€, 75£, 1000¥)
    def replace_currency_suffix(match):
        amount = match.group(1)
        currency_sign = match.group(2)
        return f"{number_to_vietnamese_words(amount)} {currency_symbol_to_word(currency_sign)}"
    text = re.sub(r'([\d.,]+)([$₫đ€£¥₹₽₩₺%])', replace_currency_suffix, text)
    
    # in case symbol [¥] is used for Chinese currency and followed by CNY
    text = text.replace('yên CNY', 'nhân dân tệ')
    
    # Replace abbreviations
    text = convert_abbreviations(text)
    text = replace_abbreviations(text)
    
    # Convert Roman numerals to integers
    def replace_roman(match):
        roman_numeral = match.group()
        return str(roman_to_integer(roman_numeral))
    # Replace Roman numerals with integers
    text = re.sub(r'\b[IVXLCDM]+\b', replace_roman, text)

    # Convert standalone numbers to words
    text = re.sub(r'\b[\d.,]+\b', lambda match: number_to_vietnamese_words(match.group()), text)
    
    # Fix common grammar errors
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\s([,\.])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,\.])(\S)', r'\1 \2', text)  # Add space after punctuation
    
    text = ( text.replace("..", ".")
                .replace("!.", "!")
                .replace("?.", "?")
                .replace(" .", ".")
                .replace(" ,", ",")
                .replace(" (", ", ")
                .replace(") ", ", ")
    )

    return text.strip()

