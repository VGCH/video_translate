# main.py - Основной файл приложения
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import pygame
import tempfile
from styles import AppColors, Fonts, AppStyles, ButtonConfig
from processor2 import MediaProcessor
from utils import get_available_voices, get_speech_models, get_translate_models


class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Перевод и закадровая озвучка видео")
        photo = tk.PhotoImage(file='flurry.png')
        # Установка иконки для окна
        self.root.iconphoto(False, photo)
        self.root.geometry("1050x900")
        self.root.configure(bg=AppColors.BG_COLOR)

        # Инициализация pygame для воспроизведения звука
        pygame.mixer.init()

        # Настройка стилей
        self.style = AppStyles.configure_styles()

        # Инициализация процессора (но еще не логируем!)
        self.processor = MediaProcessor(
            log_callback=self.log_message,
            progress_callback=self.update_progress
        )

        self.is_processing = False
        self.test_audio_data = None

        # Создаем виджеты интерфейса
        self.create_widgets()

        # После создания всех виджетов логируем начальную информацию
        self.root.after(100, self.log_initial_info)

    def log_initial_info(self):
        """Логирование начальной информации после создания интерфейса"""
        # Добавьте импорт platform в начале файла если его нет
        # import platform
        # import os

        self.log_message("=" * 60)
        self.log_message("[INFO] Перевод и закадровая озвучка видео")
        self.log_message("[INFO] Инициализация завершена")

        # Логируем информацию об устройстве из процессора
        self.processor.log_device_info()

        # Дополнительная системная информация
        self.log_message("[INFO] Приложение готово к работе")
        self.log_message("=" * 60)

    # Добавьте этот метод в класс SpeechRecognitionApp:
    def log_initial_info(self):
        """Логирование начальной информации после создания интерфейса"""
        # Логируем информацию об устройстве процессора
        self.processor.log_device_info()

    def create_widgets(self):
        # Основной контейнер
        main_container = tk.Frame(self.root, bg=AppColors.BG_COLOR, padx=25, pady=15)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = tk.Label(main_container,
                               text="🈯 Перевод и закадровая озвучка видео",
                               font=Fonts.TITLE,
                               bg=AppColors.BG_COLOR,
                               fg=AppColors.TEXT_WHITE)
        title_label.pack(pady=(0, 15))

        # Фрейм выбора файла
        file_frame = tk.LabelFrame(main_container,
                                   text=" 📁 Выбор файла",
                                   font=Fonts.HEADER,
                                   bg=AppColors.FRAME_BG,
                                   fg=AppColors.TEXT_COLOR,
                                   relief=tk.GROOVE,
                                   bd=0,
                                   padx=15,
                                   pady=15)
        file_frame.pack(fill=tk.X, pady=(0, 25))

        # Поле пути файла и кнопка обзора
        self.file_path = tk.StringVar()
        file_entry_frame = tk.Frame(file_frame, bg=AppColors.FRAME_BG)
        file_entry_frame.pack(fill=tk.X)

        file_entry = tk.Entry(file_entry_frame,
                              textvariable=self.file_path,
                              font=Fonts.BODY,
                              bg=AppColors.TEXT_WHITE,
                              fg=AppColors.TEXT_COLOR,
                              width=80,
                              relief=tk.SOLID,
                              bd=1)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))

        browse_config = ButtonConfig.get_browse_button_config()
        browse_btn = tk.Button(file_entry_frame,
                               command=self.browse_file,
                               cursor='hand2',
                               relief=tk.RAISED,
                               bd=0,
                               **browse_config)
        browse_btn.pack(side=tk.LEFT)

        # Фрейм для настроек моделей
        models_frame = tk.Frame(main_container, bg=AppColors.BG_COLOR)
        models_frame.pack(fill=tk.X, pady=(0, 20))

        # Создаем три колонки
        columns_frame = tk.Frame(models_frame, bg=AppColors.BG_COLOR)
        columns_frame.pack(fill=tk.X)

        # Колонка 1: Распознавание речи
        speech_frame = tk.LabelFrame(columns_frame,
                                     text=" 🔊 Модель распознавания речи",
                                     font=Fonts.HEADER,
                                     bg=AppColors.FRAME_BG,
                                     fg=AppColors.TEXT_COLOR,
                                     relief=tk.GROOVE,
                                     bd=0,
                                     padx=20,
                                     pady=20)
        speech_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        self.speech_model = tk.StringVar(value="base")
        for text, value in get_speech_models():
            rb = tk.Radiobutton(speech_frame,
                                text=text,
                                variable=self.speech_model,
                                value=value,
                                bg=AppColors.FRAME_BG,
                                fg=AppColors.TEXT_COLOR,
                                font=Fonts.BODY,
                                selectcolor=AppColors.ACCENT_COLOR,
                                activebackground=AppColors.FRAME_BG,
                                activeforeground=AppColors.TEXT_COLOR,
                                cursor='hand2')
            rb.pack(anchor=tk.W, pady=4)

        # Колонка 2: Модель переводчика
        translate_frame = tk.LabelFrame(columns_frame,
                                        text=" 🌐 Модель переводчика",
                                        font=Fonts.HEADER,
                                        bg=AppColors.FRAME_BG,
                                        fg=AppColors.TEXT_COLOR,
                                        relief=tk.GROOVE,
                                        bd=0,
                                        padx=20,
                                        pady=20)
        translate_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        self.translate_model = tk.StringVar(value="base")
        for text, value in get_translate_models():
            rb = tk.Radiobutton(translate_frame,
                                text=text,
                                variable=self.translate_model,
                                value=value,
                                bg=AppColors.FRAME_BG,
                                fg=AppColors.TEXT_COLOR,
                                font=Fonts.BODY,
                                selectcolor=AppColors.ACCENT_COLOR,
                                activebackground=AppColors.FRAME_BG,
                                activeforeground=AppColors.TEXT_COLOR,
                                cursor='hand2')
            rb.pack(anchor=tk.W, pady=4)

        # Колонка 3: Голос синтеза с тестированием
        voice_frame = tk.LabelFrame(columns_frame,
                                    text=" 🎵 Голос синтеза речи",
                                    font=Fonts.HEADER,
                                    bg=AppColors.FRAME_BG,
                                    fg=AppColors.TEXT_COLOR,
                                    relief=tk.GROOVE,
                                    bd=0,
                                    padx=20,
                                    pady=20)
        voice_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.voice_model = tk.StringVar(value="aidar")

        # Фрейм для радиокнопок
        voice_radio_frame = tk.Frame(voice_frame, bg=AppColors.FRAME_BG)
        voice_radio_frame.pack(fill=tk.X, pady=(0, 10))

        for text, value in get_available_voices():
            rb_frame = tk.Frame(voice_radio_frame, bg=AppColors.FRAME_BG)
            rb_frame.pack(anchor=tk.W, pady=2)

            rb = tk.Radiobutton(rb_frame,
                                text=text,
                                variable=self.voice_model,
                                value=value,
                                bg=AppColors.FRAME_BG,
                                fg=AppColors.TEXT_COLOR,
                                font=Fonts.BODY,
                                selectcolor=AppColors.ACCENT_COLOR,
                                activebackground=AppColors.FRAME_BG,
                                activeforeground=AppColors.TEXT_COLOR,
                                cursor='hand2',
                                command=self.on_voice_change)
            rb.pack(side=tk.LEFT)

            # Добавляем маленькую кнопку теста для каждого голоса
            '''
            test_btn = tk.Button(rb_frame,
                                 text="🎧",
                                 command=lambda v=value: self.test_voice(v),
                                 bg=AppColors.ACCENT_COLOR,
                                 fg=AppColors.TEXT_LIGHT,
                                 font=('Arial', 8),
                                 relief=tk.RAISED,
                                 bd=1,
                                 padx=5,
                                 pady=1,
                                 cursor='hand2')
            test_btn.pack(side=tk.LEFT, padx=(10, 0))
            '''

        # Поле для тестового текста
        test_text_frame = tk.Frame(voice_frame, bg=AppColors.FRAME_BG)
        test_text_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(test_text_frame,
                 text="Текст для теста:",
                 font=('Arial', 9),
                 bg=AppColors.FRAME_BG,
                 fg=AppColors.TEXT_COLOR).pack(anchor=tk.W)

        self.test_text = tk.StringVar(value="Привет, это тестовое сообщение для проверки.")
        test_entry = tk.Entry(test_text_frame,
                              textvariable=self.test_text,
                              font=Fonts.BODY,
                              bg=AppColors.TEXT_WHITE,
                              fg=AppColors.TEXT_COLOR,
                              width=20)
        test_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Кнопка воспроизведения теста
        self.play_test_btn = tk.Button(test_text_frame,
                                       text="▶ Тест",
                                       command=self.test_current_voice,
                                       bg=AppColors.PROGRESS_COLOR,
                                       fg=AppColors.TEXT_LIGHT,
                                       font=('Arial', 9),
                                       relief=tk.RAISED,
                                       bd=0,
                                       padx=10,
                                       pady=3,
                                       width=3,
                                       cursor='hand2')
        self.play_test_btn.pack(side=tk.LEFT)

        # Фрейм прогресс-бара
        progress_frame = tk.LabelFrame(main_container,
                                       text=" 📊 Прогресс выполнения",
                                       font=Fonts.HEADER,
                                       bg=AppColors.FRAME_BG,
                                       fg=AppColors.TEXT_COLOR,
                                       relief=tk.GROOVE,
                                       bd=0,
                                       padx=10,
                                       pady=10)
        progress_frame.pack(fill=tk.X, pady=(0, 20))

        # Метка состояния
        self.progress_label = tk.Label(progress_frame,
                                       text="Готов к работе",
                                       font=Fonts.HEADER,
                                       bg=AppColors.FRAME_BG,
                                       fg=AppColors.ACCENT_COLOR)
        self.progress_label.pack(anchor=tk.W, pady=(0, 10))

        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            length=90,
                                            mode='determinate',
                                            style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        # Процент выполнения
        self.progress_percent = tk.Label(progress_frame,
                                         text="0%",
                                         font=Fonts.BODY,
                                         bg=AppColors.FRAME_BG,
                                         fg=AppColors.TEXT_COLOR)
        self.progress_percent.pack(anchor=tk.E)

        # Кнопка запуска
        button_frame = tk.Frame(main_container, bg=AppColors.BG_COLOR)
        button_frame.pack(fill=tk.X, pady=(0, 20))

        start_config = ButtonConfig.get_start_button_config("normal")
        self.start_btn = tk.Button(button_frame,
                                   command=self.start_processing,
                                   cursor='hand2',
                                   relief=tk.RAISED,
                                   bd=0,
                                   padx=40,
                                   pady=12,
                                   **start_config)
        self.start_btn.pack()

        # Фрейм лога выполнения
        log_frame = tk.LabelFrame(main_container,
                                  text=" 📝 Лог выполнения",
                                  font=Fonts.HEADER,
                                  bg=AppColors.FRAME_BG,
                                  fg=AppColors.TEXT_COLOR,
                                  relief=tk.GROOVE,
                                  bd=0)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Панель инструментов лога
        log_toolbar = tk.Frame(log_frame, bg=AppColors.FRAME_BG, padx=10, pady=5)
        log_toolbar.pack(fill=tk.X)

        tk.Label(log_toolbar,
                 text="Лог выполнения:",
                 font=Fonts.BODY,
                 bg=AppColors.FRAME_BG,
                 fg=AppColors.TEXT_COLOR).pack(side=tk.LEFT)

        clear_config = ButtonConfig.get_log_clear_button_config()
        clear_btn = tk.Button(log_toolbar,
                              command=self.clear_log,
                              cursor='hand2',
                              relief=tk.RAISED,
                              bd=0,
                              **clear_config)
        clear_btn.pack(side=tk.RIGHT, padx=(5, 0))

        copy_config = ButtonConfig.get_log_copy_button_config()
        copy_btn = tk.Button(log_toolbar,
                             command=self.copy_log,
                             cursor='hand2',
                             relief=tk.RAISED,
                             bd=0,
                             **copy_config)
        copy_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # Текстовое поле для лога
        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                  height=12,
                                                  font=Fonts.LOG,
                                                  bg=AppColors.LOG_BG,
                                                  fg=AppColors.LOG_TEXT,
                                                  insertbackground=AppColors.TEXT_WHITE,
                                                  relief=tk.SUNKEN,
                                                  bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Создаем фрейм для статус-бара с двумя частями
        status_frame = tk.Frame(main_container, bg=AppColors.STATUS_BAR_BG, height=25)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))

        # Левая часть статус-бара
        self.status_bar = tk.Label(status_frame,
                                   text="Готов к работе • Выберите файл для обработки",
                                   bd=0,
                                   relief=tk.SUNKEN,
                                   anchor=tk.W,
                                   bg=AppColors.STATUS_BAR_BG,
                                   fg=AppColors.TEXT_WHITE,
                                   font=Fonts.STATUS)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0), pady=2)

        # Правая часть статус-бара (копирайт)
        copyright_label = tk.Label(status_frame,
                                   text="© CYBEREX TECH, 2025",
                                   bd=0,
                                   relief=tk.SUNKEN,
                                   anchor=tk.E,
                                   bg=AppColors.STATUS_BAR_BG,
                                   fg=AppColors.TEXT_LIGHT,  # Светлый цвет
                                   font=('Arial', 9),
                                   padx=10)
        copyright_label.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=2)

    def on_voice_change(self):
        """Вызывается при изменении выбора голоса"""
        self.log_message(f"[INFO] Выбран голос: {self.voice_model.get()}")

    def test_voice(self, voice=None):
        """Тестирование голоса"""
        if voice is None:
            voice = self.voice_model.get()

        test_text = self.test_text.get()
        if not test_text.strip():
            test_text = "Привет, это тестовое сообщение для проверки голоса."

        self.log_message(f"[TEST] Тестируем голос {voice}...")

        # Отключаем кнопку на время теста
        self.play_test_btn.config(state=tk.DISABLED, text="⏳")

        # Запускаем тест в отдельном потоке
        thread = threading.Thread(target=self._test_voice_thread,
                                  args=(voice, test_text),
                                  daemon=True)
        thread.start()

    def _test_voice_thread(self, voice, text):
        """Поток для тестирования голоса"""
        try:
            # Создаем временный процессор для теста
            test_processor = MediaProcessor(
                log_callback=self.log_message,
                progress_callback=None
            )

            # Создаем TTS для теста
            tts = test_processor.SileroTTSWithTiming(
                voice=voice,
                log_callback=self.log_message
            )

            # Генерируем тестовое аудио
            audio = tts.test_voice(text)

            if audio is not None:
                # Сохраняем во временный файл
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name

                import soundfile as sf
                sf.write(tmp_path, audio, 48000)

                # Воспроизводим
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

                self.log_message(f"[TEST] Воспроизводится тест голоса {voice}")

                # Ждем завершения воспроизведения
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except:
                    pass

                self.log_message("[TEST] Тест завершен")
            else:
                self.log_message("[TEST] Не удалось сгенерировать тестовое аудио")

        except Exception as e:
            self.log_message(f"[TEST] Ошибка при тестировании голоса: {e}")
        finally:
            # Восстанавливаем кнопку
            self.root.after(0, lambda: self.play_test_btn.config(
                state=tk.NORMAL,
                text="▶ Тест"
            ))

    def test_current_voice(self):
        """Тестирование текущего выбранного голоса"""
        self.test_voice(self.voice_model.get())

    def browse_file(self):
        """Открыть диалог выбора файла"""
        filename = filedialog.askopenfilename(
            title="Выберите аудио или видео файл",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.ogg *.m4a *.flac"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_path.set(filename)
            self.status_bar.config(text=f"Выбран файл: {os.path.basename(filename)}")
            self.log_message(f"[INFO] Выбран файл: {filename}")

    def log_message(self, message):
        """Добавить сообщение в лог"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def clear_log(self):
        """Очистить лог"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("[INFO] Лог очищен")

    def copy_log(self):
        """Копировать лог в буфер обмена"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get(1.0, tk.END))
        self.status_bar.config(text="Лог скопирован в буфер обмена")
        self.log_message("[INFO] Лог скопирован в буфер обмена")

    def update_progress(self, value, description=""):
        """Обновить прогресс"""
        self.progress_bar['value'] = value
        self.progress_percent.config(text=f"{value}%")

        if description:
            self.progress_label.config(text=description)
            self.status_bar.config(text=description)

        self.root.update()

    def start_processing(self):
        """Запустить процесс обработки"""
        if self.is_processing:
            self.log_message("[WARN] Обработка уже запущена")
            return

        if not self.file_path.get():
            self.log_message(f"[ERROR] Не выбран файл для обработки!")
            self.status_bar.config(text="Ошибка: Не выбран файл!", fg=AppColors.ERROR_COLOR)
            messagebox.showerror("Ошибка", "Пожалуйста, выберите файл для обработки")
            return

        if not os.path.exists(self.file_path.get()):
            self.log_message(f"[ERROR] Файл не найден: {self.file_path.get()}")
            self.status_bar.config(text="Ошибка: Файл не найден!", fg=AppColors.ERROR_COLOR)
            messagebox.showerror("Ошибка", "Файл не найден")
            return

        # Получаем конфигурацию для кнопки в состоянии обработки
        processing_config = ButtonConfig.get_start_button_config("processing")

        # Обновляем кнопку
        self.start_btn.config(
            state=processing_config["state"],
            text=processing_config["text"],
            bg=processing_config["bg"],
            fg=processing_config["fg"]
        )

        self.is_processing = True

        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(target=self._processing_thread, daemon=True)
        thread.start()

    def _processing_thread(self):
        """Поток обработки файла"""
        try:
            # Получаем параметры
            speech_model = self.speech_model.get()
            translate_model = self.translate_model.get()
            voice = self.voice_model.get()
            file_path = self.file_path.get()

            # Запускаем обработку
            result_path = self.processor.process_media_file(
                file_path=file_path,
                speech_model=speech_model,
                translate_model=translate_model,
                voice=voice,
                output_dir=os.path.dirname(file_path)
            )

            if result_path:
                self.log_message(f"[SUCCESS] Обработка завершена успешно!")
                self.log_message(f"[RESULT] Результат сохранен: {result_path}")

                # Предлагаем открыть папку с результатом
                self.root.after(0, lambda: self._ask_open_folder(result_path))
            else:
                self.log_message(f"[ERROR] Обработка завершена с ошибками")
                messagebox.showerror("Ошибка", "Не удалось обработать файл")

        except Exception as e:
            self.log_message(f"[ERROR] Неожиданная ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
        finally:
            # Восстанавливаем кнопку
            normal_config = ButtonConfig.get_start_button_config("normal")
            self.root.after(0, lambda: self.start_btn.config(
                state=normal_config["state"],
                text=normal_config["text"],
                bg=normal_config["bg"],
                fg=normal_config["fg"]
            ))

            self.is_processing = False
            self.update_progress(0, "Готов к работе")

    def _ask_open_folder(self, file_path):
        """Спросить об открытии папки с результатом"""
        folder = os.path.dirname(file_path)
        answer = messagebox.askyesno(
            "Успешно",
            f"Обработка завершена успешно!\n\n"
            f"Результат сохранен в:\n{file_path}\n\n"
            f"Открыть папку с результатом?"
        )

        if answer:
            try:
                import platform
                import subprocess

                system = platform.system()

                if system == "Windows":
                    os.startfile(folder)
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", folder])
                else:  # Linux
                    subprocess.run(["xdg-open", folder])

            except Exception as e:
                self.log_message(f"[ERROR] Не удалось открыть папку: {e}")


def main():
    root = tk.Tk()
    SpeechRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()