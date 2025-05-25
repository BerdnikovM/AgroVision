import sys
import os
import psutil
import time
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFrame, QTextEdit,
                             QSizePolicy, QScrollArea, QStackedWidget)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QSize, QTimer, QDir

from detect_page import DetectPage
from visualization_page import VisualizationPage
from export_page import ExportPage


class AgroVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgroVision")
        self.setMinimumSize(1280, 720)

        # Установить темную тему
        self.set_dark_theme()

        # Состояние сеанса
        self.session_active = False

        # Счетчик обработанных файлов
        self.processed_count = 0

        # Создать основной макет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Создать боковую панель
        self.create_sidebar()

        # Создать стековый виджет для нескольких страниц
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        # Создать главную страницу
        self.home_page = QWidget()
        self.home_layout = QVBoxLayout(self.home_page)
        self.home_layout.setContentsMargins(0, 0, 0, 0)
        self.home_layout.setSpacing(0)
        self.stacked_widget.addWidget(self.home_page)

        # Создать страницу обнаружения
        self.detect_page = DetectPage(self)
        self.stacked_widget.addWidget(self.detect_page)

        # Создать страницу визуализации
        self.visualization_page = VisualizationPage(self)
        self.stacked_widget.addWidget(self.visualization_page)

        # Создать страницу экспорта
        self.export_page = ExportPage(self)
        self.stacked_widget.addWidget(self.export_page)

        # Создать область содержимого для главной страницы
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self.home_layout.addWidget(self.content_widget)

        # Создать заголовок
        self.create_header()

        # Создайть содержимое приборной панели
        self.dashboard_widget = QWidget()
        self.dashboard_layout = QVBoxLayout(self.dashboard_widget)
        self.dashboard_layout.setContentsMargins(20, 20, 20, 20)
        self.dashboard_layout.setSpacing(20)
        self.content_layout.addWidget(self.dashboard_widget)

        # Создайте раздел метрик
        self.create_metrics_section()

        # Создайте кнопку начала сеанса
        self.create_start_session_button()

        # Создайте секцию вывода журнала
        self.create_log_output_section()

        # Создать раздел с советами
        self.create_tip_section()

        # Инициализируем таймер для обновления CPU
        self.cpu_timer = QTimer(self)
        self.cpu_timer.timeout.connect(self.update_cpu_usage)
        self.cpu_timer.start(1000)  # каждую секунду

        # Инициализируем таймер для обновления количества файлов
        self.files_timer = QTimer(self)
        self.files_timer.timeout.connect(self.update_processed_files_count)
        self.files_timer.start(1000)  # каждую секунду

        # Показывать главную страницу по умолчанию
        self.stacked_widget.setCurrentIndex(0)

        # Создать необходимые папки
        self.create_folders()

        # Изначально отключите кнопки боковой панели.
        self.set_sidebar_buttons_enabled(False)

        # Первоначально обновите счетчик обработанных файлов
        self.update_processed_files_count()

    def set_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(30, 30, 40))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 45))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(35, 35, 45))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(dark_palette)

    def create_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(140)
        self.sidebar.setStyleSheet("background-color: #1a1a24;")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(0)
        self.sidebar_layout.setAlignment(Qt.AlignTop)

        # Кнопка меню
        self.menu_button = QPushButton("≡")
        self.menu_button.setFont(QFont("Arial", 20))
        self.menu_button.setFixedHeight(60)
        self.menu_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: white;
                border: none;
                text-align: center;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
        """)
        self.sidebar_layout.addWidget(self.menu_button)

        # Добавьте разделитель строк
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #2a2a34;")
        self.sidebar_layout.addWidget(line)

        # Кнопка Домой
        self.home_button = QPushButton("Главная")
        self.home_button.setFont(QFont("Arial", 10))
        self.home_button.setFixedHeight(50)
        self.home_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: white;
                border: none;
                text-align: center;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
        """)
        self.home_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.sidebar_layout.addWidget(self.home_button)

        # Кнопка обнаружения
        self.detect_button = QPushButton("Детекция")
        self.detect_button.setFont(QFont("Arial", 10))
        self.detect_button.setFixedHeight(50)
        self.detect_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: white;
                border: none;
                text-align: center;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
            QPushButton:disabled {
                color: #666666;
            }
        """)
        self.detect_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.sidebar_layout.addWidget(self.detect_button)

        # Кнопка визуализации
        self.visualization_button = QPushButton("Визуализация")
        self.visualization_button.setFont(QFont("Arial", 10))
        self.visualization_button.setFixedHeight(50)
        self.visualization_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: white;
                border: none;
                text-align: center;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
            QPushButton:disabled {
                color: #666666;
            }
        """)
        self.visualization_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.sidebar_layout.addWidget(self.visualization_button)

        # Кнопка экспорта
        self.export_button = QPushButton("Экспорт")
        self.export_button.setFont(QFont("Arial", 10))
        self.export_button.setFixedHeight(50)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: white;
                border: none;
                text-align: center;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
            QPushButton:disabled {
                color: #666666;
            }
        """)
        self.export_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        self.sidebar_layout.addWidget(self.export_button)

        self.main_layout.addWidget(self.sidebar)

    def create_header(self):
        self.header = QWidget()
        self.header.setFixedHeight(60)
        self.header.setStyleSheet("background-color: #1a1a24;")
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(20, 0, 20, 0)
        self.header_layout.setSpacing(10)

        # Logo
        logo_path = os.path.join("images", "agrovision_logo.png")
        if os.path.exists(logo_path):
            self.logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            self.logo_label.setPixmap(pixmap.scaled(150, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.logo_label = QLabel("AGROVISION")
            self.logo_label.setFont(QFont("Arial", 16, QFont.Bold))
            self.logo_label.setStyleSheet("color: white;")
        self.header_layout.addWidget(self.logo_label)

        # Левая проставка → Центрирование заголовка
        self.header_layout.addStretch()

        # Название дома
        self.title_label = QLabel("Главная")
        self.title_label.setFont(QFont("Arial", 16))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: white;")
        self.header_layout.addWidget(self.title_label)

        # Правая проставка → Центрирование заголовка
        self.header_layout.addStretch()

        # Version label
        self.version_label = QLabel("v 0.1")
        self.version_label.setFont(QFont("Arial", 10))
        self.version_label.setStyleSheet("color: #7f8c8d;")
        self.header_layout.addWidget(self.version_label)

        # Кнопка настройки
        self.settings_button = QPushButton("⚙")
        self.settings_button.setFont(QFont("Arial", 16))
        self.settings_button.setFixedSize(40, 40)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #1a1a24;
                color: white;
                border: none;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
        """)
        self.header_layout.addWidget(self.settings_button)

        # Добавляем хедер и разделитель в основной layout
        self.content_layout.addWidget(self.header)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #2a2a34;")
        self.content_layout.addWidget(line)

    def create_metrics_section(self):
        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(20, 0, 20, 0)
        metrics_layout.setSpacing(20)

        # создаём карточки
        pf = self.create_metric_card("images/Files_icon.png", "Обработанные файлы", str(self.processed_count), "#1a1a24")
        # найдём внутри pf тот QLabel, который отвечает за основное значение (24pt)
        for w in pf.findChildren(QLabel):
            if w.font().pointSize() == 24:
                self.processed_label = w
                break
        mp = self.create_metric_card("images/mAP_icon.png", "mAP", "0.90", "#1a1a24")
        lt = self.create_metric_card("images/TrainingTime_icon.png", "Последнее время\nобучения", "08.04.2025", "#1a1a24")
        cpu_card, self.cpu_value_label, self.cpu_time_label = self.create_cpu_card("images/Cpu_icon.png")

        # добавляем с равным stretch = 1
        metrics_layout.addWidget(pf, 1)
        metrics_layout.addWidget(mp, 1)
        metrics_layout.addWidget(lt, 1)
        metrics_layout.addWidget(cpu_card, 1)

        self.dashboard_layout.addWidget(metrics_widget)

    def create_metric_card(self, icon, title, value, color, subtitle=None):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {color};
                border-radius: 10px;
            }}
        """)
        # фиксируем одинаковые размеры
        card.setFixedSize(240, 140)
        card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Заголовок
        hl = QHBoxLayout()
        icon_lbl = QLabel()
        # если icon — файл, загружаем pixmap, иначе — текст
        if os.path.isfile(icon):
            pix = QPixmap(icon).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_lbl.setPixmap(pix)
        else:
            icon_lbl.setText(icon)
            icon_lbl.setFont(QFont("Arial", 16))
        hl.addWidget(icon_lbl)
        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Arial", 12))
        title_lbl.setStyleSheet("color: white;")
        title_lbl.setWordWrap(True)
        hl.addWidget(title_lbl)
        hl.addStretch()
        layout.addLayout(hl)

        # Значение
        val_lbl = QLabel(value)
        val_lbl.setFont(QFont("Arial", 24, QFont.Bold))
        val_lbl.setStyleSheet("color: white;")
        val_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(val_lbl)

        # Подзаголовок
        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setFont(QFont("Arial", 12))
            sub_lbl.setStyleSheet("color: #aaaaaa;")
            sub_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(sub_lbl)

        return card

    def create_cpu_card(self, cpu_icon_path):
        """
        Возвращает (frame, value_label, time_label)
        """
        # Создаём карточку через create_metric_card, передаём subtitle initial
        frame = self.create_metric_card(
            cpu_icon_path,
            "ЦП нагрузка",
            "0%",
            "#1a1a24",
            subtitle="00:00:00"
        )

        # Найдём внутри QLabel для value (24pt) и для времени (12pt, серый)
        value_label = None
        time_label = None
        for w in frame.findChildren(QLabel):
            if w.font().pointSize() == 24:
                value_label = w
            elif w.font().pointSize() == 12 and "aaaaaa" in w.styleSheet():
                time_label = w

        return frame, value_label, time_label

    def create_start_session_button(self):
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 10)

        self.start_button = QPushButton("Начать сеанс")
        self.start_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_button.setFixedSize(300, 60)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2e8b57;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3aa76d;
            }
        """)
        self.start_button.clicked.connect(self.toggle_session)
        button_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        self.dashboard_layout.addWidget(button_container)

    def create_log_output_section(self):
        # Section title
        log_title = QLabel("Журнал событий")
        log_title.setFont(QFont("Arial", 16, QFont.Bold))
        log_title.setStyleSheet("color: white; margin-top: 10px;")
        self.dashboard_layout.addWidget(log_title)

        # Текстовая область журнала
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a24;
                color: #cccccc;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Courier New';
                font-size: 12px;
            }
        """)
        self.log_output.setFixedHeight(200)

        # Добавить примеры записей в журнале
        self.log_output.append("[INFO] Система инициализирована")
        self.log_output.append("[INFO] Ожидание начала сеанса...")

        self.dashboard_layout.addWidget(self.log_output)

    def create_tip_section(self):
        tip_frame = QFrame()
        tip_frame.setFrameShape(QFrame.StyledPanel)
        tip_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a24;
                border-radius: 5px;
                padding: 15px;
                margin-top: 10px;
            }
        """)

        tip_layout = QHBoxLayout(tip_frame)

        tip_label = QLabel(
            "Совет: Добро пожаловать в AgroVision! Нажмите 'Начать сеанс', чтобы начать.")
        tip_label.setWordWrap(True)
        tip_label.setFont(QFont("Arial", 12))
        tip_label.setStyleSheet("color: #2e8b57;")

        tip_layout.addWidget(tip_label)
        self.dashboard_layout.addSpacing(20)
        self.dashboard_layout.addWidget(tip_frame)

    def update_cpu_usage(self):
        # Читаем загрузку CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        # Текущее время
        now = datetime.now().strftime("%H:%M:%S")
        # Пишем в лейблы
        if self.cpu_value_label:
            self.cpu_value_label.setText(f"{cpu_percent}%")
        if self.cpu_time_label:
            self.cpu_time_label.setText(now)

    def update_processed_files_count(self):
        # Получить количество файлов в выходной папке
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if os.path.exists(output_folder):
            file_count = len([f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))])

            # Update only if the count has changed
            if file_count != self.processed_count:
                self.processed_count = file_count
                if self.processed_label:
                    self.processed_label.setText(str(self.processed_count))

    def create_folders(self):
        # Создайте выходную папку для визуализации
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Создайте папку экспорта для экспортируемых данных
        export_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        # Создайте папку images, если она не существует.
        images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        # Создайте папку models, если она не существует.
        models_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

    def set_sidebar_buttons_enabled(self, enabled):
        """Включите или отключите кнопки боковой панели, кроме Home"""
        self.detect_button.setEnabled(enabled)
        self.visualization_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)

    def log_message(self, message):
        """Добавьте сообщение с временной меткой в вывод журнала"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_output.append(f"{timestamp} {message}")

    def toggle_session(self):
        """Переключение между запуском и остановкой сеанса"""
        if not self.session_active:
            self.start_session()
        else:
            self.stop_session()

    def start_session(self):
        """Начните новый сеанс с проверки системы"""
        self.log_message("Начата проверка системы...")

        # Очистить предыдущие записи журнала
        self.log_output.clear()
        self.log_message("Начата проверка системы...")

        # Имитация кратковременной задержки для проверки системы
        QApplication.processEvents()
        time.sleep(0.5)

        # Проверьте папку с моделями
        models_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(models_folder):
            self.log_message("Проверка папки Models: ПРОВАЛ! Папка не найдена.")
            self.log_message("Инициализация: Не удалось! Пожалуйста, создайте папку models.")
            return

        # Проверка наличия файлов моделей
        model_files = [f for f in os.listdir(models_folder) if
                       f.endswith('.pt') and os.path.isfile(os.path.join(models_folder, f))]
        if not model_files:
            self.log_message("Проверка папки Models: ПРОВАЛ! Файлы моделей не найдены.")
            self.log_message("Инициализация: Не удалось! Пожалуйста, добавьте файлы моделей в папку models.")
            return

        # Журнал найденных моделей
        model_files_str = ", ".join(model_files)
        self.log_message(f"Проверка папок с моделями: Найдено {len(model_files)} модель(ей). ({model_files_str})")

        # Контрольное вычислительное устройство (моделируемое)
        self.log_message("Вычислительное устройство: Процессор (моделируемый)")

        # Проверьте папки вывода и экспорта
        output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        export_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            self.log_message("Создана папка output.")

        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
            self.log_message("Создана папка export.")

        # Все проверки пройдены
        self.log_message("Инициализация: Успех!")
        self.log_message("Сеанс успешно начался.")

        # Обновление пользовательского интерфейса
        self.session_active = True
        self.start_button.setText("Остановить сеанс")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #cc3333;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #dd4444;
            }
        """)

        # Включите кнопки боковой панели
        self.set_sidebar_buttons_enabled(True)

    def stop_session(self):
        """Stop the current session"""
        self.log_message("Сеанс остановлен.")

        # Обновление пользовательского интерфейса
        self.session_active = False
        self.start_button.setText("Начать сеанс")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2e8b57;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3aa76d;
            }
        """)

        # Отключите кнопки боковой панели и вернитесь на главную страницу
        self.set_sidebar_buttons_enabled(False)
        self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgroVisionApp()
    window.show()
    sys.exit(app.exec_())