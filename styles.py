# styles.py - Файл со стилями и цветовой схемой

class AppColors:
    """Цветовая схема приложения"""
    # Основные цвета
    BG_COLOR = '#2c3e50'  # Темно-синий фон
    FRAME_BG = '#ecf0f1'  # Светло-серый фон фреймов
    TEXT_COLOR = '#2c3e50'  # Темный текст
    TEXT_LIGHT = '#7f8c8d'  # Серый текст для кнопок
    TEXT_WHITE = '#ffffff'  # Белый текст

    # Акцентные цвета
    ACCENT_COLOR = '#3498db'  # Синий акцентный
    BUTTON_COLOR = '#3498db'  # Цвет кнопок
    PROGRESS_COLOR = '#2ecc71'  # Зеленый прогресс
    ERROR_COLOR = '#e74c3c'  # Красный ошибка
    WARNING_COLOR = '#f39c12'  # Оранжевый предупреждение

    # Дополнительные цвета
    LOG_BG = '#1a1a1a'  # Фон лога
    LOG_TEXT = '#00ff00'  # Текст лога
    STATUS_BAR_BG = '#34495e'  # Фон статус бара
    DISABLED_GRAY = '#bdc3c7'  # Серый для неактивных элементов


class Fonts:
    """Шрифты приложения"""
    TITLE = ('Arial', 20, 'bold')
    HEADER = ('Arial', 11, 'bold')
    BODY = ('Arial', 10)
    BUTTON = ('Arial', 10, 'bold')
    BUTTON_LARGE = ('Arial', 12, 'bold')
    LOG = ('Courier New', 9)
    STATUS = ('Arial', 10)


class AppStyles:
    """Класс для управления стилями приложения"""

    @staticmethod
    def configure_styles():
        """Настройка стилей tkinter"""
        import tkinter.ttk as ttk

        style = ttk.Style()
        style.theme_use('clam')

        # Стиль для Radiobutton
        style.configure('TRadiobutton',
                        background=AppColors.FRAME_BG,
                        foreground=AppColors.TEXT_COLOR,
                        font=Fonts.BODY)

        # Стиль для Progressbar
        style.configure("Horizontal.TProgressbar",
                        background=AppColors.PROGRESS_COLOR,
                        troughcolor=AppColors.DISABLED_GRAY,
                        bordercolor=AppColors.ACCENT_COLOR,
                        lightcolor=AppColors.PROGRESS_COLOR,
                        darkcolor='#27ae60')

        # Стиль для кнопок
        style.configure('TButton',
                        font=Fonts.BUTTON,
                        padding=6)

        return style


class ButtonConfig:
    """Конфигурация кнопок"""

    # Основная кнопка запуска
    @staticmethod
    def get_start_button_config(state="normal"):
        configs = {
            "normal": {
                "text": "🚀 НАЧАТЬ ОБРАБОТКУ",
                "bg": AppColors.PROGRESS_COLOR,
                "fg": AppColors.TEXT_LIGHT,  # Серый текст
                "font": Fonts.BUTTON_LARGE,
                "state": "normal"
            },
            "processing": {
                "text": "⏳ ОБРАБОТКА...",
                "bg": AppColors.WARNING_COLOR,
                "fg": AppColors.TEXT_COLOR,  # Темный текст для контраста
                "font": Fonts.BUTTON_LARGE,
                "state": "disabled"
            }
        }
        return configs.get(state, configs["normal"])

    # Кнопка обзора файлов
    @staticmethod
    def get_browse_button_config():
        return {
            "text": " 📂 Обзор...",
            "bg": AppColors.BUTTON_COLOR,
            "fg": AppColors.TEXT_LIGHT,  # Серый текст
            "font": Fonts.BUTTON,
            "padx": 25,
            "pady": 8
        }

    # Кнопки лога
    @staticmethod
    def get_log_clear_button_config():
        return {
            "text": "🗑️ Очистить",
            "bg": AppColors.DISABLED_GRAY,
            "fg": AppColors.TEXT_LIGHT,  # Серый текст
            "font": Fonts.BUTTON,
            "padx": 10,
            "pady": 3
        }

    @staticmethod
    def get_log_copy_button_config():
        return {
            "text": "📋 Копировать",
            "bg": AppColors.ACCENT_COLOR,
            "fg": AppColors.TEXT_LIGHT,  # Серый текст
            "font": Fonts.BUTTON,
            "padx": 10,
            "pady": 3
        }