import os
import sys
import csv
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
                             QGroupBox, QSplitter, QListWidget, QAbstractItemView,
                             QScrollArea, QFrame, QSizePolicy)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QSize


class ExportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.init_ui()
        self.load_export_files()

    def init_ui(self):
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header with back button and title
        header_layout = QHBoxLayout()

        # Back button
        back_button = QPushButton("←")
        back_button.setFont(QFont("Arial", 16))
        back_button.setFixedSize(40, 40)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
        """)
        back_button.clicked.connect(self.go_back)
        header_layout.addWidget(back_button)

        # Add stretch to push title to center
        header_layout.addStretch()

        # Title label - centered
        title_label = QLabel("Экспорт")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        # Add stretch to push title to center
        header_layout.addStretch()

        # Refresh button
        refresh_button = QPushButton("↻")
        refresh_button.setToolTip("Обновить список файлов")
        refresh_button.setFont(QFont("Arial", 16))
        refresh_button.setFixedSize(40, 40)
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #2a2a34;
            }
        """)
        refresh_button.clicked.connect(self.load_export_files)
        header_layout.addWidget(refresh_button)

        main_layout.addLayout(header_layout)

        # Create splitter for file list and data view
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #2a2a34;
                width: 2px;
            }
        """)

        # File list section
        file_group = QGroupBox("Экспорт файлов")
        file_group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a24;
                color: white;
                border: 1px solid #2a2a34;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)

        file_layout = QVBoxLayout(file_group)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a24;
                color: white;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #2a2a34;
            }
            QListWidget::item:selected {
                background-color: #2a2a44;
            }
            QListWidget::item:hover {
                background-color: #2a2a34;
            }
        """)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentItemChanged.connect(self.file_selected)
        file_layout.addWidget(self.file_list)

        # Data view section
        data_group = QGroupBox("Таблица данных")
        data_group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a24;
                color: white;
                border: 1px solid #2a2a34;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)

        data_layout = QVBoxLayout(data_group)

        # Container for data view
        data_container = QWidget()
        data_container.setStyleSheet("background-color: #1a1a24;")
        data_container_layout = QVBoxLayout(data_container)
        data_container_layout.setContentsMargins(0, 0, 0, 0)

        # File path label
        self.file_path_label = QLabel("Файл не выбран")
        self.file_path_label.setStyleSheet("color: #aaaaaa; background-color: #1a1a24;")
        self.file_path_label.setWordWrap(True)
        data_container_layout.addWidget(self.file_path_label)

        # Table widget for data
        self.data_table = QTableWidget()
        self.data_table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a24;
                color: white;
                border: none;
                gridline-color: #2a2a34;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #2a2a44;
            }
            QHeaderView::section {
                background-color: #2a2a34;
                color: white;
                padding: 5px;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1a1a24;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #2a2a34;
                min-height: 20px;
                border-radius: 7px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setStyleSheet(self.data_table.styleSheet() + """
            QTableWidget {
                alternate-background-color: #1e1e28;
            }
        """)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.verticalHeader().setVisible(False)
        data_container_layout.addWidget(self.data_table)

        # Export button
        self.export_button = QPushButton("Экспорт в CSV")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #2e8b57;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3aa76d;
            }
            QPushButton:disabled {
                background-color: #1e5e3a;
                color: #aaaaaa;
            }
        """)
        self.export_button.clicked.connect(self.export_to_csv)
        self.export_button.setEnabled(False)
        data_container_layout.addWidget(self.export_button)

        data_layout.addWidget(data_container)

        # Add widgets to splitter
        splitter.addWidget(file_group)
        splitter.addWidget(data_group)
        splitter.setSizes([200, 600])  # Set initial sizes

        main_layout.addWidget(splitter)

    def go_back(self):
        """Navigate back to home page"""
        if hasattr(self.parent_app, 'stacked_widget'):
            self.parent_app.stacked_widget.setCurrentIndex(0)

    def load_export_files(self):
        """Load export files from the export directory"""
        self.file_list.clear()

        export_dir = os.path.join(self.app_dir, "export")
        if os.path.exists(export_dir):
            files = [f for f in os.listdir(export_dir) if
                     f.endswith('.csv') and os.path.isfile(os.path.join(export_dir, f))]

            # Sort files by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(export_dir, x)), reverse=True)

            for file in files:
                self.file_list.addItem(file)

    def file_selected(self, current, previous):
        """Handle file selection"""
        if current is None:
            self.export_button.setEnabled(False)
            return

        file_name = current.text()
        file_path = os.path.join(self.app_dir, "export", file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            self.file_path_label.setText(f"Файл не найден: {file_path}")
            self.export_button.setEnabled(False)
            return

        # Update file path label
        self.file_path_label.setText(f"Данные экспортируются в {file_path}")

        # Enable export button
        self.export_button.setEnabled(True)

        # Load CSV data
        self.load_csv_data(file_path)

    def load_csv_data(self, file_path):
        """Load CSV data into table widget"""
        try:
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Get header row

                # Set up table
                self.data_table.clear()
                self.data_table.setRowCount(0)
                self.data_table.setColumnCount(len(headers))
                self.data_table.setHorizontalHeaderLabels(headers)

                # Add data rows
                for row_idx, row in enumerate(reader):
                    self.data_table.insertRow(row_idx)
                    for col_idx, cell in enumerate(row):
                        self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(cell))

                # Resize columns to content
                self.data_table.resizeColumnsToContents()

                # Make sure the last column doesn't stretch too much
                header = self.data_table.horizontalHeader()
                header.setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)

                # Ensure the table fills the available space
                self.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        except Exception as e:
            self.data_table.clear()
            self.data_table.setRowCount(1)
            self.data_table.setColumnCount(1)
            self.data_table.setHorizontalHeaderLabels(["Error"])
            self.data_table.setItem(0, 0, QTableWidgetItem(f"Ошибка загрузки CSV: {str(e)}"))

    def export_to_csv(self):
        """Export data to a new CSV file"""
        if self.file_list.currentItem() is None:
            return

        source_file = os.path.join(self.app_dir, "export", self.file_list.currentItem().text())

        # Open file dialog to select destination
        file_filter = "CSV Files (*.csv)"
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", file_filter
        )

        if destination:
            try:
                # Copy file to destination
                import shutil
                shutil.copy(source_file, destination)
                self.file_path_label.setText(f"Файл, экспортированный в: {destination}")
            except Exception as e:
                self.file_path_label.setText(f"Ошибка при экспорте файла: {str(e)}")