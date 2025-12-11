# processor2.py - Тестовый класс для обработки
import os
import tempfile
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import librosa
import numpy as np
import subprocess
import re
from num2words import num2words as num2words_lib
from silero import silero_tts


class MediaProcessor:
    def __init__(self, log_callback=None, progress_callback=None):
        # Определение устройства
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = "CUDA (NVIDIA GPU)"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "MPS (Apple Silicon GPU)"
        else:
            device = torch.device('cpu')
            device_name = "CPU"

        self.device = device
        self.device_name = device_name
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.sample_rate = 48000

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def log_device_info(self):
        if self.log_callback:
            self.log_callback(f"[INFO] Используется устройство: {self.device_name}")

    def update_progress(self, step, total_steps, description=""):
        if self.progress_callback:
            progress = int((step / total_steps) * 100)
            self.progress_callback(progress, description)

    def extract_audio_from_video(self, video_path):
        """Извлечение аудио из видео через ffmpeg"""
        self.log("Извлекаем аудио из видео через ffmpeg...")

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio_path = tmp.name

            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ac', '2',
                '-ar', '44100',
                '-y',
                audio_path
            ]

            self.log(f"Запускаем ffmpeg...")

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if process.returncode == 0:
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    self.log(f"✓ Аудио успешно извлечено: {audio_path}")
                    self.log(f"✓ Размер аудиофайла: {file_size_mb:.2f} MB")

                    try:
                        audio, sr = librosa.load(audio_path, sr=None)
                        duration = len(audio) / sr
                        self.log(f"✓ Длительность аудио: {duration:.2f} секунд")
                    except:
                        pass

                    return audio_path
                else:
                    self.log("✗ Ошибка: аудиофайл не создан или пустой")
                    return None
            else:
                self.log(f"✗ Ошибка ffmpeg (код {process.returncode}):")
                if process.stderr:
                    error_lines = process.stderr.strip().split('\n')[-10:]
                    for line in error_lines:
                        self.log(f"  {line}")
                return None

        except Exception as e:
            self.log(f"✗ Исключение при извлечении аудио: {e}")
            return None

    def transcribe_with_timestamps(self, audio_path, model_size="medium"):
        """Транскрибирование с временными метками - упрощенная версия"""
        self.log(f"Загружаем модель Whisper ({model_size})...")

        try:
            model = whisper.load_model(model_size)
            self.log("Транскрибируем аудио...")

            # Простой вызов без word_timestamps для совместимости
            result = model.transcribe(audio_path, language='en')

            segments = []
            for segment in result['segments']:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                })

            self.log(f"✓ Транскрибировано {len(segments)} сегментов")
            return segments

        except Exception as e:
            self.log(f"✗ Ошибка при транскрибировании: {e}")
            return None

    class Translator:
        def __init__(self, model_size="base", log_callback=None):
            self.log_callback = log_callback

            # Определяем устройство
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

            # ВЫБИРАЕМ МОДЕЛЬ FACEBOOK В ЗАВИСИМОСТИ ОТ РАЗМЕРА
            if model_size == "large":
                self.model_name = "facebook/nllb-200-distilled-1.3B"
                self.model_type = "nllb"
            elif model_size == "base":
                self.model_name = "facebook/nllb-200-distilled-600M"
                self.model_type = "nllb"
            else:  # small
                # Для small можно использовать более легкую модель Facebook
                self.model_name = "facebook/m2m100_418M"
                self.model_type = "m2m100"

            self.log(f"Загружаем модель Facebook: {self.model_name}")
            self.log(f"Тип модели: {self.model_type}")

            try:
                # Загружаем модель Facebook
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.log("✓ Модель Facebook загружена успешно")

                # Настраиваем языковые коды в зависимости от типа модели
                if self.model_type == "nllb":
                    self.src_lang = "eng_Latn"
                    self.tgt_lang = "rus_Cyrl"
                elif self.model_type == "m2m100":
                    self.src_lang = "en"
                    self.tgt_lang = "ru"

            except Exception as e:
                self.log(f"✗ Ошибка загрузки модели Facebook: {e}")
                raise

        def log(self, message):
            if self.log_callback:
                self.log_callback(message)

        def translate_text(self, text):
            """ПРАВИЛЬНЫЙ метод перевода для Facebook NLLB"""
            text = ' '.join(text.split())
            if not text or len(text.strip()) == 0:
                return ""

            try:
                # Для NLLB: добавляем языковой код исходного текста
                prepared_text = f"{self.src_lang} {text}"

                inputs = self.tokenizer(
                    prepared_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Получаем ID для русского языка
                generate_kwargs = {
                    'max_length': 512,
                    'num_beams': 4,
                    'early_stopping': True,
                }

                if self.model_type == "nllb":
                    # Проверяем какие ID есть у токенизатора
                    try:
                        # Пробуем разные варианты русского языка
                        possible_lang_codes = [
                            "rus_Cyrl",  # Основной код
                            "ru",  # Альтернативный
                            "rus",  # Еще вариант
                            "__rus_Cyrl__",  # Полный токен
                        ]

                        forced_bos_token_id = None
                        for lang_code in possible_lang_codes:
                            try:
                                # Пробуем получить ID
                                if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                                    token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
                                    if token_id is not None and token_id > 0:
                                        forced_bos_token_id = token_id
                                        self.log(f"  Найден ID {token_id} для языка '{lang_code}'")
                                        break
                            except:
                                continue

                        if forced_bos_token_id is None:
                            # Если не нашли, ищем вручную в словаре токенов
                            self.log("  Ищем русский язык в токенах...")
                            tokenizer_vocab = self.tokenizer.get_vocab()
                            for token, token_id in tokenizer_vocab.items():
                                if "rus" in token.lower() or "ru" in token.lower():
                                    forced_bos_token_id = token_id
                                    self.log(f"  Найден токен: '{token}' -> ID: {token_id}")
                                    break

                        if forced_bos_token_id:
                            generate_kwargs['forced_bos_token_id'] = forced_bos_token_id
                            self.log(f"  Используем forced_bos_token_id: {forced_bos_token_id}")
                        else:
                            self.log("  ⚠ Не удалось найти ID русского языка, пробуем без него")

                    except Exception as e:
                        self.log(f"  ⚠ Ошибка поиска ID языка: {e}")

                # Генерация перевода
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generate_kwargs)

                # Декодирование результата
                translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Проверяем результат
                self.log(f"  Результат: '{translated[:100]}...'")

                # Если перевод не сработал, возвращаем заглушку
                if not translated or translated.strip() == "" or translated == text:
                    self.log("  ⚠ Переводчик не смог перевести, используем заглушку")
                    return f"[НЕ ПЕРЕВЕДЕНО: {text[:50]}...]"

                return translated

            except Exception as e:
                self.log(f"✗ Ошибка перевода: {e}")
                return text

        def translate_segments(self, segments):
            """Перевод сегментов с валидацией"""
            self.log(f"Переводим сегменты с помощью Facebook {self.model_type}...")
            translated_segments = []

            for i, segment in enumerate(segments):
                if not segment['text'].strip():
                    continue

                self.log(f"Переводим сегмент {i + 1}/{len(segments)}: {segment['text'][:50]}...")
                translated_text = self.translate_text(segment['text'])

                # Проверяем что переводчик вернул
                if translated_text == "[ОШИБКА ПЕРЕВОДА]":
                    self.log(f"  ❌ КРИТИЧЕСКАЯ ОШИБКА: Переводчик не смог перевести!")
                    # Возвращаем оригинальный текст с пометкой
                    translated_text = f"[НЕ ПЕРЕВЕДЕНО: {segment['text'][:30]}...]"

                translated_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'original_text': segment['text'],
                    'translated_text': translated_text,
                    'duration': segment['end'] - segment['start']
                })

                if (i + 1) % 5 == 0:
                    self.log(f"  Переведено {i + 1}/{len(segments)} сегментов")

            self.log(f"✓ Перевод Facebook моделью завершен")
            return translated_segments

    class SileroTTSWithTiming:
        def __init__(self, voice="aidar", log_callback=None):
            self.log_callback = log_callback
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.voice = voice
            self.sample_rate = 48000

            self.log("Загружаем Silero TTS...")
            try:
                self.model, _ = silero_tts(language='ru', speaker='v5_1_ru')
                self.model.to(self.device)
                self.log("✓ TTS загружен")
            except Exception as e:
                self.log(f"✗ Ошибка загрузки TTS: {e}")
                raise

        def log(self, message):
            if self.log_callback:
                self.log_callback(message)

        def preprocess_text(self, text: str) -> str:
            """Предобработка текста"""

            def convert_numbers(match):
                num_str = match.group()
                try:
                    num = float(num_str) if '.' in num_str else int(num_str)

                    if len(num_str) == 4 and 1000 <= num <= 2099:
                        digits = {
                            '0': 'ноль', '1': 'один', '2': 'два', '3': 'три',
                            '4': 'четыре', '5': 'пять', '6': 'шесть', '7': 'семь',
                            '8': 'восемь', '9': 'девять'
                        }
                        return ' '.join(digits[digit] for digit in num_str)

                    if len(num_str) in [6, 7, 8, 10, 11, 12]:
                        digits = {
                            '0': 'ноль', '1': 'один', '2': 'два', '3': 'три',
                            '4': 'четыре', '5': 'пять', '6': 'шесть', '7': 'семь',
                            '8': 'восемь', '9': 'девять'
                        }
                        return ' '.join(digits[digit] for digit in num_str)

                    if '.' in num_str:
                        whole_part = num2words_lib(int(num), lang='ru')
                        decimal_part = ' '.join(digits.get(d, d) for d in num_str.split('.')[1])
                        return f"{whole_part} целых {decimal_part}"

                    return num2words_lib(num, lang='ru')
                except (ValueError, OverflowError):
                    return num_str

            # Шаг 1: Преобразуем год
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

            # Шаг 5-8: Очистка
            text = re.sub(r'(\d+)\s*процент', r'\1 процент', text)
            text = re.sub(r'(\d+)\s*(доллар|евро|рубль|фунт|иена)', r'\1 \2', text)
            text = re.sub(r'[«»"“”„\[\]()]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        def high_quality_speed_up(self, audio: np.ndarray, speed_factor: float) -> np.ndarray:
            """Ускорение аудио"""
            if abs(speed_factor - 1.0) < 0.01 or speed_factor < 1.0:
                return audio

            if speed_factor > 2.0:
                self.log(f"  Внимание: большое ускорение {speed_factor:.2f}x, ограничиваем до 2.0x")
                speed_factor = 2.0

            n_fft = 2048
            hop_length = n_fft // 4

            try:
                stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
                stft_stretched = librosa.phase_vocoder(stft, rate=speed_factor, hop_length=hop_length)
                audio_stretched = librosa.istft(stft_stretched, hop_length=hop_length, window='hann')
                return audio_stretched.astype(np.float32)
            except Exception as e:
                self.log(f"  Ошибка при ускорении: {e}")
                target_samples = int(len(audio) / speed_factor)
                return audio[:target_samples]

        def apply_smooth_fade(self, audio: np.ndarray, fade_in_ms: int = 5, fade_out_ms: int = 10) -> np.ndarray:
            """Плавное затухание"""
            if len(audio) < 100:
                return audio

            fade_in_samples = int(fade_in_ms * self.sample_rate / 1000)
            if fade_in_samples > 0 and len(audio) > fade_in_samples:
                fade_in = np.linspace(0, 1, fade_in_samples)
                audio[:fade_in_samples] *= fade_in

            fade_out_samples = int(fade_out_ms * self.sample_rate / 1000)
            if fade_out_samples > 0 and len(audio) > fade_out_samples:
                fade_out = np.linspace(1, 0, fade_out_samples)
                audio[-fade_out_samples:] *= fade_out

            return audio

        def trim_excessive_silence(self, audio: np.ndarray, threshold_db: float = -30) -> np.ndarray:
            """Убираем тишину"""
            if len(audio) == 0:
                return audio

            audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
            above_threshold = audio_db > threshold_db

            if not np.any(above_threshold):
                return audio[:int(0.1 * self.sample_rate)]

            speech_start = np.argmax(above_threshold)
            speech_end = len(audio) - np.argmax(above_threshold[::-1])

            padding = int(0.05 * self.sample_rate)
            start = max(0, speech_start - padding)
            end = min(len(audio), speech_end + padding)

            return audio[start:end]

        def generate_audio_segment(self, text: str, target_duration: float) -> np.ndarray:
            """Генерация аудио сегмента"""
            original_text = text
            text = self.preprocess_text(text.strip())

            if not text or len(text) < 2:
                self.log("  Пустой текст, возвращаем тишину")
                return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)

            if len(text) > 350:
                self.log(f"  Текст слишком длинный ({len(text)} символов), обрезаем")
                text = text[:340] + "..."

            try:
                with torch.no_grad():
                    wav = self.model.apply_tts(
                        text=text,
                        speaker=self.voice,
                        sample_rate=self.sample_rate,
                        put_accent=True,
                        put_yo=True
                    )

                audio = wav.cpu().numpy().squeeze()
                audio = self.trim_excessive_silence(audio)

                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.95

                current_duration = len(audio) / self.sample_rate

                if current_duration > target_duration:
                    speed_factor = current_duration / target_duration
                    self.log(f"  Ускоряем в {speed_factor:.2f} раз")
                    audio = self.high_quality_speed_up(audio, speed_factor)
                    current_duration = len(audio) / self.sample_rate

                target_samples = int(target_duration * self.sample_rate)

                if len(audio) > target_samples:
                    excess_samples = len(audio) - target_samples
                    fade_out_samples = min(100, excess_samples)
                    if fade_out_samples > 0:
                        audio[target_samples - fade_out_samples:target_samples] *= np.linspace(1, 0, fade_out_samples)
                    audio = audio[:target_samples]

                elif len(audio) < target_samples:
                    silence_samples = target_samples - len(audio)
                    audio = np.pad(audio, (0, silence_samples), mode='constant')

                audio = self.apply_smooth_fade(audio, fade_in_ms=5, fade_out_ms=10)

                return audio.astype(np.float32)

            except Exception as e:
                self.log(f"  ✗ Ошибка при генерации аудио: {e}")
                self.log(f"  Текст вызвавший ошибку: '{original_text[:100]}...'")
                import traceback
                self.log(f"  Детали ошибки: {traceback.format_exc()}")
                return np.zeros(int(target_duration * self.sample_rate), dtype=np.float32)

        def test_voice(self, text="Привет, это тестовое сообщение для проверки голоса."):
            try:
                self.log(f"Тестируем голос {self.voice}...")
                audio = self.generate_audio_segment(text, 3.0)
                return audio
            except Exception as e:
                self.log(f"✗ Ошибка при тестировании голоса: {e}")
                return None

    def process_media_file(self, file_path, speech_model="medium",
                           translate_model="base", voice="aidar",
                           output_dir=None):
        """Основной процесс обработки"""
        self.log("=" * 60)
        self.log("Начинаем обработку файла...")

        # Проверка файла
        if not os.path.exists(file_path):
            self.log(f"✗ Ошибка: Файл не найден: {file_path}")
            return None

        # Определяем тип файла
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.wmv', '.m4v', '.mpg', '.mpeg')
        audio_extensions = ('.wav', '.mp3', '.ogg', '.m4a', '.flac', '.aac', '.wma')

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in video_extensions:
            is_video = True
        elif file_ext in audio_extensions:
            is_video = False
        else:
            self.log(f"✗ Неподдерживаемый формат файла: {file_ext}")
            return None

        # Шаг 1: Извлечение аудио
        if is_video:
            self.update_progress(1, 10, "Извлечение аудио из видео...")
            audio_path = self.extract_audio_from_video(file_path)
            if not audio_path:
                return None
        else:
            self.update_progress(1, 10, "Загрузка аудиофайла...")
            audio_path = file_path

        # Шаг 2: Получение длительности
        try:
            original_audio, orig_sr = librosa.load(audio_path, sr=None)
            original_duration = len(original_audio) / orig_sr
            self.log(f"✓ Длительность аудио: {original_duration:.2f} секунд")
        except Exception as e:
            self.log(f"✗ Ошибка загрузки аудио: {e}")
            return None

        # Шаг 3: Транскрибирование
        self.update_progress(2, 10, "Транскрибирование...")
        segments = self.transcribe_with_timestamps(audio_path, speech_model)
        if not segments:
            return None

        # Шаг 4: Перевод
        self.update_progress(3, 10, "Перевод...")
        translator = self.Translator(model_size=translate_model, log_callback=self.log)
        translated_segments = translator.translate_segments(segments)
        if not translated_segments:
            return None

        # Шаг 5: Синтез речи
        self.update_progress(4, 10, "Синтез речи...")
        tts = self.SileroTTSWithTiming(voice=voice, log_callback=self.log)

        # Шаг 6: Создание финального аудио
        self.update_progress(5, 10, "Создание финального аудио...")
        final_audio = np.zeros(int(original_duration * self.sample_rate), dtype=np.float32)

        total_segments = len(translated_segments)
        for i, segment in enumerate(translated_segments):
            progress = 5 + int((i / total_segments) * 4)
            self.update_progress(progress, 10, f"Синтез сегмента {i + 1}/{total_segments}...")

            self.log(f"  Озвучиваем: '{segment['translated_text'][:80]}...'")

            synthesized_audio = tts.generate_audio_segment(
                segment['translated_text'],
                segment['duration']
            )

            if synthesized_audio is not None and len(synthesized_audio) > 0:
                start_sample = int(segment['start'] * self.sample_rate)
                end_sample = start_sample + len(synthesized_audio)

                if start_sample < len(final_audio):
                    if end_sample > len(final_audio):
                        synthesized_audio = synthesized_audio[:len(final_audio) - start_sample]
                        end_sample = len(final_audio)

                    final_audio[start_sample:end_sample] = synthesized_audio
            else:
                self.log(f"  ⚠ Ошибка: не удалось сгенерировать аудио для сегмента {i + 1}")

        # Шаг 7: Сохранение результата
        self.update_progress(9, 10, "Сохранение результата...")
        if output_dir is None:
            output_dir = os.path.dirname(file_path)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_translated.wav")

        sf.write(output_path, final_audio, self.sample_rate)

        # Очистка временных файлов
        if is_video and audio_path != file_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass

        self.update_progress(10, 10, "Готово!")
        self.log(f"✓ Результат сохранен: {output_path}")
        self.log(f"✓ Размер файла: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        self.log("=" * 60)

        return output_path