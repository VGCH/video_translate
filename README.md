## Проект переводчика и закадровой озвучки видео с использованием ML-моделей

Работает на следующих моделях:

> Whisper от OpenAI

> Helsinki-NLP от Группы исследований языковых технологий Хельсинкского университета.

> facebook/nllb

> Silero TTS v5_ru

Структура проекта:
```no-highlight
speech_translator/
├── main.py          # Главный файл приложения
├── styles.py        # Стили и конфигурации
├── processor.py     # Класс для обработки медиа
├── utils.py         # Вспомогательные функции
└── requirements.txt # Зависимости
```

# Для работы приложения необходимо установить следующие зависимости/пакеты:

Набор библиотек FFmpeg:

Linux
```no-highlight
sudo apt install ffmpeg
```

Windows
```no-highlight
scoop install ffmpeg
```

MacOS
```no-highlight
brew install ffmpeg
```

Установка необходимых библиотек Python3:
```no-highlight
pip install -r requirements.txt
```

Команда запуска приложения:
```no-highlight
python3 main.py
```

Интерфейс приложения:
![Интерфейс приложения](https://github.com/VGCH/video_translate/blob/main/img/screen1.png)

Интерфейс приложения:
![Интерфейс приложения](https://github.com/VGCH/video_translate/blob/main/img/screen2.png)


Полное описание в статье на Хабр

[Ссылка на статью ](https://habr.com/ru/articles/971644/)
