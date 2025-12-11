# utils.py - Вспомогательные функции
import re
import numpy as np
import librosa
from num2words import num2words as num2words_lib


def convert_numbers(match):
    """Конвертирует числа в слова с учетом контекста"""
    num_str = match.group()

    try:
        # Пробуем преобразовать в число
        num = float(num_str) if '.' in num_str else int(num_str)

        # ОСОБЫЙ СЛУЧАЙ: ГОДА (4-значные числа)
        if len(num_str) == 4 and 1000 <= num <= 2099:
            # Года произносим по цифрам
            digits = {
                '0': 'ноль', '1': 'один', '2': 'два', '3': 'три',
                '4': 'четыре', '5': 'пять', '6': 'шесть', '7': 'семь',
                '8': 'восемь', '9': 'девять'
            }
            return ' '.join(digits[digit] for digit in num_str)

        # ОСОБЫЙ СЛУЧАЙ: НОМЕРА ТЕЛЕФОНОВ, КОДЫ (6-12 цифр)
        if len(num_str) in [6, 7, 8, 10, 11, 12]:
            # Произносим по цифрам
            digits = {
                '0': 'ноль', '1': 'один', '2': 'два', '3': 'три',
                '4': 'четыре', '5': 'пять', '6': 'шесть', '7': 'семь',
                '8': 'восемь', '9': 'девять'
            }
            return ' '.join(digits[digit] for digit in num_str)

        # ОБЫЧНЫЕ ЧИСЛА
        if '.' in num_str:
            whole_part = num2words_lib(int(num), lang='ru')
            decimal_part = ' '.join(digits.get(d, d) for d in num_str.split('.')[1])
            return f"{whole_part} целых {decimal_part}"

        # Для целых чисел
        return num2words_lib(num, lang='ru')

    except (ValueError, OverflowError):
        return num_str


def preprocess_text_tts(text: str) -> str:
    """Предобработка текста для TTS"""
    # Шаг 1: Преобразуем год (особый случай)
    text = re.sub(r'\b(19|20)\d{2}\b', convert_numbers, text)

    # Шаг 2: Преобразуем номера телефонов
    text = re.sub(r'\b\d{6,12}\b', convert_numbers, text)

    # Шаг 3: Преобразуем обычные числа
    text = re.sub(r'\b\d+(?:\.\d+)?\b', convert_numbers, text)

    # Шаг 4: Обработка специальных символов
    replacements = {
        '%': ' процент',
        '$': ' доллар',
        '€': ' евро',
        '₽': ' рубль',
        '£': ' фунт',
        '¥': ' иена',
        '+': ' плюс',
        '*': ' умножить на',
        '÷': ' разделить на',
        '=': ' равно',
        '/': ' на',
        '>': ' больше',
        '<': ' меньше'
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Шаг 5: Обработка процентов после чисел
    text = re.sub(r'(\d+)\s*процент', r'\1 процент', text)

    # Шаг 6: Обработка валют
    text = re.sub(r'(\d+)\s*(доллар|евро|рубль|фунт|иена)', r'\1 \2', text)

    # Шаг 7: Очистка лишних символов
    text = re.sub(r'[«»"“”„\[\]()]', '', text)

    # Шаг 8: Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def high_quality_speed_up(audio: np.ndarray, speed_factor: float, sample_rate: int) -> np.ndarray:
    """Качественное ускорение аудио"""
    if abs(speed_factor - 1.0) < 0.01 or speed_factor < 1.0:
        return audio

    if speed_factor > 2.0:
        print(f"  Внимание: большое ускорение {speed_factor:.2f}x, ограничиваем до 2.0x")
        speed_factor = 2.0

    n_fft = 2048
    hop_length = n_fft // 4

    try:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
        stft_stretched = librosa.phase_vocoder(stft, rate=speed_factor, hop_length=hop_length)
        audio_stretched = librosa.istft(stft_stretched, hop_length=hop_length, window='hann')
        return audio_stretched.astype(np.float32)
    except Exception:
        target_samples = int(len(audio) / speed_factor)
        return audio[:target_samples]


def apply_smooth_fade(audio: np.ndarray, fade_in_ms: int = 5, fade_out_ms: int = 10, sample_rate: int = 48000) -> np.ndarray:
    """Применяет плавное затухание"""
    if len(audio) < 100:
        return audio

    fade_in_samples = int(fade_in_ms * sample_rate / 1000)
    if fade_in_samples > 0 and len(audio) > fade_in_samples:
        fade_in = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in

    fade_out_samples = int(fade_out_ms * sample_rate / 1000)
    if fade_out_samples > 0 and len(audio) > fade_out_samples:
        fade_out = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out

    return audio


def trim_excessive_silence(audio: np.ndarray, threshold_db: float = -30) -> np.ndarray:
    """Убирает слишком длинную тишину"""
    if len(audio) == 0:
        return audio

    audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
    above_threshold = audio_db > threshold_db

    if not np.any(above_threshold):
        return audio[:int(0.1 * 48000)]

    speech_start = np.argmax(above_threshold)
    speech_end = len(audio) - np.argmax(above_threshold[::-1])

    padding = int(0.05 * 48000)
    start = max(0, speech_start - padding)
    end = min(len(audio), speech_end + padding)

    return audio[start:end]


def get_available_voices():
    """Возвращает список доступных голосов"""
    return [
        ("Aidar (мужской)  ", "aidar"),
        ("Baya (женский)   ", "baya"),
        ("Kseniya (женский)", "kseniya"),
        ("Xenia (женский)  ", "xenia")
    ]


def get_speech_models():
    """Возвращает список моделей распознавания речи"""
    return [
        ("Small (быстрая)", "small"),
        ("Base (стандартная)", "base"),
        ("Medium (улучшенная)", "medium"),
        ("Large (точная)", "large")
    ]


def get_translate_models():
    """Возвращает список моделей переводчика"""
    return [
        ("Small (быстрая)", "small"),
        ("Base (стандартная)", "base"),
        ("Large (точная)", "large")
    ]