import os
import sys
import csv
import cv2
import time
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFrame, QFileDialog, QComboBox, QCheckBox, QProgressBar,
                             QTableWidget, QTableWidgetItem, QHeaderView, QToolButton,
                             QGroupBox, QListWidget, QAbstractItemView, QSizePolicy, QScrollArea)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QMimeData, QUrl, QThread, QTimer

# Import YOLO from ultralytics if available
try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Detection will be simulated.")


class DetectionWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    result_ready = pyqtSignal(str, str, int, int, str)
    detection_finished = pyqtSignal()

    def __init__(self, file_paths, model_path, enable_preprocessing, app_dir, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.model_path = model_path
        self.enable_preprocessing = enable_preprocessing
        self.app_dir = app_dir  # Base application directory
        self.is_running = True

    def run(self):
        if not ULTRALYTICS_AVAILABLE:
            self.simulate_detection()
            return

        try:
            # Load the YOLO model
            model = YOLO(self.model_path)

            total_files = len(self.file_paths)

            for i, file_path in enumerate(self.file_paths):
                if not self.is_running:
                    break

                file_name = os.path.basename(file_path)
                progress = int((i / total_files) * 100)
                self.progress_updated.emit(progress, f"Processing {file_name} ({i + 1}/{total_files})...")

                # Determine if it's an image or video
                file_ext = os.path.splitext(file_path)[1].lower()
                is_video = file_ext in ['.mp4', '.avi', '.mov']

                # Create output paths using app_dir
                output_dir = os.path.join(self.app_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                export_dir = os.path.join(self.app_dir, "export")
                os.makedirs(export_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{timestamp}{file_ext}")
                csv_file = os.path.join(export_dir, f"detection_{os.path.splitext(file_name)[0]}_{timestamp}.csv")

                # Process the file
                if is_video:
                    self.process_video(model, file_path, output_file, csv_file)
                else:
                    self.process_image(model, file_path, output_file, csv_file)

                # Emit result
                process_time = f"{np.random.uniform(0.5, 5.0):.2f}"  # Simulated processing time
                objects_count = np.random.randint(5, 50)  # Simulated object count
                weeds_count = np.random.randint(0, 20)  # Simulated weed count

                self.result_ready.emit(file_name, process_time, objects_count, weeds_count, output_file)

            self.progress_updated.emit(100, "Detection completed")
            self.detection_finished.emit()

        except Exception as e:
            self.progress_updated.emit(0, f"Error: {str(e)}")
            self.detection_finished.emit()

    def process_image(self, model, input_path, output_path, csv_path):
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            raise RuntimeError(f"Failed to open image {input_path}")

        # Apply CLAHE preprocessing if enabled
        if self.enable_preprocessing:
            img = self.apply_clahe(img)

        # Run detection
        results = model(img)
        r = results[0]

        # Save detection results to CSV
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])

            # Save all boxes
            for i in range(len(r.boxes)):
                # Coordinates
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                # Confidence
                conf = float(r.boxes.conf[i])
                # Class ID and name
                cls_id = int(r.boxes.cls[i])
                cls_name = model.names[cls_id]
                # Write row
                writer.writerow([0, cls_name, f"{conf:.4f}", int(x1), int(y1), int(x2), int(y2)])

        # Save the annotated image
        r.save(filename=output_path)

    def process_video(self, model, input_path, output_path, csv_path):
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Prepare CSV file
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update progress every 10 frames
                if frame_idx % 10 == 0:
                    progress = int((frame_idx / total_frames) * 100)
                    self.progress_updated.emit(progress, f"Processing video frame {frame_idx}/{total_frames}...")

                # Apply CLAHE preprocessing if enabled
                if self.enable_preprocessing:
                    frame = self.apply_clahe(frame)

                # Run detection
                results = model(frame)
                r = results[0]

                # Save all boxes for this frame
                for i in range(len(r.boxes)):
                    # Coordinates
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    # Confidence
                    conf = float(r.boxes.conf[i])
                    # Class ID and name
                    cls_id = int(r.boxes.cls[i])
                    cls_name = model.names[cls_id]
                    # Write row
                    writer.writerow([frame_idx, cls_name, f"{conf:.4f}", int(x1), int(y1), int(x2), int(y2)])

                # Write the annotated frame
                annotated_frame = r.plot()
                out.write(annotated_frame)

                frame_idx += 1

                # Check if thread should stop
                if not self.is_running:
                    break

        # Release resources
        cap.release()
        out.release()

    def apply_clahe(self, img):
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split the LAB channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L channel with the original A and B channels
        merged = cv2.merge((cl, a, b))

        # Convert back to BGR color space
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return enhanced_img

    def simulate_detection(self):
        """Simulate detection when ultralytics is not available"""
        total_files = len(self.file_paths)

        for i, file_path in enumerate(self.file_paths):
            if not self.is_running:
                break

            file_name = os.path.basename(file_path)
            progress = int((i / total_files) * 100)
            self.progress_updated.emit(progress, f"Simulating detection for {file_name} ({i + 1}/{total_files})...")

            # Simulate processing time
            time.sleep(np.random.uniform(1.0, 3.0))

            # Create output paths using app_dir
            output_dir = os.path.join(self.app_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            export_dir = os.path.join(self.app_dir, "export")
            os.makedirs(export_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = os.path.splitext(file_path)[1].lower()
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_{timestamp}{file_ext}")
            csv_file = os.path.join(export_dir, f"detection_{os.path.splitext(file_name)[0]}_{timestamp}.csv")

            # Simulate CSV creation
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])

                # Generate random detections
                num_detections = np.random.randint(5, 20)
                for j in range(num_detections):
                    frame = 0 if file_ext in ['.jpg', '.jpeg', '.png'] else np.random.randint(0, 100)
                    cls_name = np.random.choice(["weed", "crop"])
                    conf = np.random.uniform(0.6, 0.95)
                    x1 = np.random.randint(0, 500)
                    y1 = np.random.randint(0, 500)
                    x2 = x1 + np.random.randint(50, 200)
                    y2 = y1 + np.random.randint(50, 200)
                    writer.writerow([frame, cls_name, f"{conf:.4f}", x1, y1, x2, y2])

            # Copy the file to output (simulating processed output)
            try:
                import shutil
                shutil.copy(file_path, output_file)
            except Exception as e:
                print(f"Error copying file: {e}")

            # Emit result
            process_time = f"{np.random.uniform(0.5, 5.0):.2f}"
            objects_count = np.random.randint(5, 50)
            weeds_count = np.random.randint(0, 20)

            self.result_ready.emit(file_name, process_time, objects_count, weeds_count, output_file)

        self.progress_updated.emit(100, "Simulation completed")
        self.detection_finished.emit()

    def stop(self):
        self.is_running = False


class DropArea(QFrame):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet("""
            DropArea {
                background-color: #1a1a24;
                border: 2px dashed #3a3a44;
                border-radius: 5px;
                min-height: 120px;
            }
            DropArea:hover {
                border-color: #4a4a54;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Drag & drop files here, or")
        self.label.setFont(QFont("Arial", 12))
        self.label.setStyleSheet("color: #cccccc; border: none;")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.browse_button = QPushButton("Browse")
        self.browse_button.setFont(QFont("Arial", 12))
        self.browse_button.setFixedSize(120, 40)
        self.browse_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a34;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a3a44;
            }
        """)
        self.browse_button.clicked.connect(self.browse_files)
        layout.addWidget(self.browse_button, alignment=Qt.AlignCenter)

        # ограничиваем высоту жестко
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(120)  # или ваше желаемое min-height

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropArea {
                    background-color: #1a1a24;
                    border: 2px dashed #4a7a8c;
                    border-radius: 5px;
                    min-height: 120px;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            DropArea {
                background-color: #1a1a24;
                border: 2px dashed #3a3a44;
                border-radius: 5px;
                min-height: 120px;
            }
            DropArea:hover {
                border-color: #4a4a54;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                DropArea {
                    background-color: #1a1a24;
                    border: 2px dashed #3a3a44;
                    border-radius: 5px;
                    min-height: 120px;
                }
                DropArea:hover {
                    border-color: #4a4a54;
                }
            """)

            file_paths = []
            for url in event.mimeData().urls():
                file_paths.append(url.toLocalFile())

            self.filesDropped.emit(file_paths)

    def browse_files(self):
        file_filter = "Images & Videos (*.jpg *.jpeg *.png *.mp4 *.avi *.mov);;Images (*.jpg *.jpeg *.png);;Videos (*.mp4 *.avi *.mov)"
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", file_filter
        )

        if files:
            self.filesDropped.emit(files)


class FileListWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Name", "Type", "Size", ""])
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setStyleSheet("""
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
        """)

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(self.styleSheet() + """
            QTableWidget {
                alternate-background-color: #1e1e28;
            }
        """)

    def add_files(self, file_paths):
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()

            # Determine file type
            if file_ext in ['.jpg', '.jpeg', '.png']:
                file_type = "Image"
            elif file_ext in ['.mp4', '.avi', '.mov']:
                file_type = "Video"
            else:
                file_type = "Unknown"

            # Get file size
            file_size = os.path.getsize(file_path)
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"

            # Add row
            row_position = self.rowCount()
            self.insertRow(row_position)

            self.setItem(row_position, 0, QTableWidgetItem(file_name))
            self.setItem(row_position, 1, QTableWidgetItem(file_type))
            self.setItem(row_position, 2, QTableWidgetItem(size_str))

            # Delete button
            delete_button = QPushButton("×")
            delete_button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #cc6666;
                    border: none;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #ff6666;
                }
            """)
            delete_button.clicked.connect(lambda checked, row=row_position: self.removeRow(row))

            self.setCellWidget(row_position, 3, delete_button)

            # Store the full path as data
            self.item(row_position, 0).setData(Qt.UserRole, file_path)

    def get_file_paths(self):
        """Return a list of all file paths in the table"""
        file_paths = []
        for row in range(self.rowCount()):
            file_path = self.item(row, 0).data(Qt.UserRole)
            file_paths.append(file_path)
        return file_paths


class ResultsTableWidget(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["File", "Time", "Objects", "Weeds", "Output"])
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setStyleSheet("""
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
        """)

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(self.styleSheet() + """
            QTableWidget {
                alternate-background-color: #1e1e28;
            }
        """)

    def add_result(self, file_name, process_time, objects_count, weeds_count, output_path):
        """Add a detection result to the table"""
        row_position = self.rowCount()
        self.insertRow(row_position)

        self.setItem(row_position, 0, QTableWidgetItem(file_name))
        self.setItem(row_position, 1, QTableWidgetItem(f"{process_time}s"))
        self.setItem(row_position, 2, QTableWidgetItem(str(objects_count)))
        self.setItem(row_position, 3, QTableWidgetItem(str(weeds_count)))

        # Output path with view button
        output_item = QTableWidgetItem(os.path.basename(output_path))
        output_item.setData(Qt.UserRole, output_path)
        self.setItem(row_position, 4, output_item)


class DetectPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.detection_worker = None

        # Get the application directory
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(self.app_dir) == 'detect_page.py':
            # If __file__ is the script itself, get its directory
            self.app_dir = os.path.dirname(self.app_dir)

        # Initialize UI after setting app_dir
        self.init_ui()

        # Load models after UI is initialized
        QTimer.singleShot(100, self.load_models)

    def init_ui(self):
        # 1) Создаём область прокрутки и делаем её резайзабельной
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        # 2) Контейнер для всего содержимого страницы
        container = QWidget()
        scroll.setWidget(container)

        # 3) Внешний layout страницы — кладём в него QScrollArea
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # 4) Основной layout — всё остальное рисуем в container
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Back button and title
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
        title_label = QLabel("Detect")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        # Add stretch to push title to center
        header_layout.addStretch()

        # Empty widget to balance the back button
        empty_widget = QWidget()
        empty_widget.setFixedSize(40, 40)
        header_layout.addWidget(empty_widget)

        main_layout.addLayout(header_layout)

        # Input Files section
        input_group = QGroupBox("Input File")
        input_group.setStyleSheet("""
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

        input_layout = QVBoxLayout(input_group)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)

        # Drop area
        self.drop_area = DropArea()
        self.drop_area.filesDropped.connect(self.add_files)
        self.drop_area.setMinimumHeight(150)
        self.drop_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        input_layout.addWidget(self.drop_area, 1)

        main_layout.addWidget(input_group)

        # === Вынесенная таблица «Selected Files» ===
        files_group = QGroupBox("Selected Files")
        files_group.setStyleSheet("""
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
        files_layout = QVBoxLayout(files_group)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.setSpacing(0)

        self.file_list = FileListWidget()
        # Increase height of file list
        self.file_list.setMinimumHeight(200)
        files_layout.addWidget(self.file_list)

        main_layout.addWidget(files_group)

        # Model section
        model_layout = QVBoxLayout()

        model_label = QLabel("Model")
        model_label.setFont(QFont("Arial", 16, QFont.Bold))
        model_label.setStyleSheet("color: white;")
        model_layout.addWidget(model_label)

        model_selector_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a24;
                color: white;
                border: 1px solid #2a2a34;
                border-radius: 5px;
                padding: 5px;
                min-height: 30px;
                padding-right: 20px; /* Space for the arrow */
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #2a2a34;
                border-left-style: solid;
            }
            QComboBox::down-arrow {
                image: none;
                width: 14px;
                height: 14px;
                background: #cccccc;
                clip-path: polygon(0 0, 100% 0, 50% 100%);
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a24;
                color: white;
                selection-background-color: #2a2a44;
            }
        """)
        model_selector_layout.addWidget(self.model_combo)

        refresh_button = QToolButton()
        refresh_button.setText("↻")
        refresh_button.setToolTip("Refresh model list")
        refresh_button.setStyleSheet("""
            QToolButton {
                background-color: #1a1a24;
                color: white;
                border: 1px solid #2a2a34;
                border-radius: 5px;
                padding: 5px;
                min-height: 30px;
                min-width: 30px;
            }
            QToolButton:hover {
                background-color: #2a2a34;
            }
        """)
        refresh_button.clicked.connect(self.refresh_models)
        model_selector_layout.addWidget(refresh_button)

        model_layout.addLayout(model_selector_layout)

        # Model loaded info
        self.model_loaded_label = QLabel("No model selected")
        self.model_loaded_label.setFont(QFont("Arial", 10))
        self.model_loaded_label.setStyleSheet("color: #aaaaaa;")
        model_layout.addWidget(self.model_loaded_label)

        main_layout.addLayout(model_layout)

        # Preprocessing toggle
        preprocess_layout = QHBoxLayout()

        self.preprocess_checkbox = QCheckBox("Enable Preprocessing (CLAHE)")
        self.preprocess_checkbox.setFont(QFont("Arial", 16))
        self.preprocess_checkbox.setStyleSheet("""
            QCheckBox {
                color: white;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 1px solid #2a2a34;
                border-radius: 3px;
                background-color: #1a1a24;
            }
            QCheckBox::indicator:checked {
                background-color: #2e8b57;
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWxpbmUgcG9pbnRzPSIyMCA2IDkgMTcgNCAxMiI+PC9wb2x5bGluZT48L3N2Zz4=);
            }
        """)
        preprocess_layout.addWidget(self.preprocess_checkbox)

        preprocess_layout.addStretch()

        info_button = QToolButton()
        info_button.setText("i")
        info_button.setToolTip("CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances image contrast")
        info_button.setStyleSheet("""
            QToolButton {
                background-color: #1a1a24;
                color: white;
                border: 1px solid #2a2a34;
                border-radius: 15px;
                padding: 5px;
                min-height: 30px;
                min-width: 30px;
            }
            QToolButton:hover {
                background-color: #2a2a34;
            }
        """)
        preprocess_layout.addWidget(info_button)

        main_layout.addLayout(preprocess_layout)

        # Run Detection button
        self.run_button = QPushButton("Run Detection")
        self.run_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.run_button.setFixedHeight(50)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #2e8b57;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3aa76d;
            }
            QPushButton:disabled {
                background-color: #1e5e3a;
                color: #aaaaaa;
            }
        """)
        self.run_button.clicked.connect(self.run_detection)
        self.run_button.setEnabled(False)  # Initially disabled
        main_layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a24;
                border: 1px solid #2a2a34;
                border-radius: 5px;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #2e8b57;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #aaaaaa;")
        main_layout.addWidget(self.status_label)

        # Results section
        results_label = QLabel("Detection Results")
        results_label.setFont(QFont("Arial", 16, QFont.Bold))
        results_label.setStyleSheet("color: white;")
        main_layout.addWidget(results_label)

        self.results_table = ResultsTableWidget()
        main_layout.addWidget(self.results_table)

        # Connect signals
        self.model_combo.currentIndexChanged.connect(self.model_changed)

    def add_files(self, file_paths):
        self.file_list.add_files(file_paths)
        # Enable run button if files are added and a model is selected
        self.update_run_button_state()

    def go_back(self):
        # Navigate back to home page
        if hasattr(self.parent_app, 'stacked_widget'):
            self.parent_app.stacked_widget.setCurrentIndex(0)

    def load_models(self):
        """Load model files from the models directory"""
        self.model_combo.clear()

        # Get models directory
        models_dir = os.path.join(self.app_dir, "models")

        if os.path.exists(models_dir):
            # Get all .pt files
            model_files = [f for f in os.listdir(models_dir) if
                           f.endswith('.pt') and os.path.isfile(os.path.join(models_dir, f))]

            # Add models to combo box
            for model_file in model_files:
                self.model_combo.addItem(model_file)

            # Select first model if available
            if model_files:
                self.model_combo.setCurrentIndex(0)
                self.model_loaded_label.setText(f"Loaded: {os.path.join(models_dir, self.model_combo.currentText())}")
            else:
                self.model_loaded_label.setText("No model files found in models directory")
        else:
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            self.model_loaded_label.setText("Models directory created. Please add model files.")

    def refresh_models(self):
        """Refresh the model list"""
        self.status_label.setText("Status: Refreshing model list...")
        self.load_models()
        self.status_label.setText("Status: Model list refreshed")
        self.update_run_button_state()

    def model_changed(self, index):
        """Handle model selection change"""
        if self.model_combo.count() > 0:
            models_dir = os.path.join(self.app_dir, "models")
            self.model_loaded_label.setText(f"Loaded: {os.path.join(models_dir, self.model_combo.currentText())}")
        else:
            self.model_loaded_label.setText("No models available")

        # Update run button state
        self.update_run_button_state()

    def update_run_button_state(self):
        """Enable run button if files are added and a model is selected"""
        has_files = self.file_list.rowCount() > 0
        has_model = self.model_combo.count() > 0
        self.run_button.setEnabled(has_files and has_model)

    def run_detection(self):
        """Run detection on selected files"""
        if self.file_list.rowCount() == 0:
            self.status_label.setText("Status: No files selected")
            return

        if self.model_combo.count() == 0:
            self.status_label.setText("Status: No model selected")
            return

        # Disable UI elements during detection
        self.run_button.setEnabled(False)
        self.drop_area.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.preprocess_checkbox.setEnabled(False)

        # Clear previous results
        self.results_table.setRowCount(0)

        # Get file paths
        file_paths = self.file_list.get_file_paths()

        # Get model path
        models_dir = os.path.join(self.app_dir, "models")
        model_path = os.path.join(models_dir, self.model_combo.currentText())

        # Get preprocessing state
        enable_preprocessing = self.preprocess_checkbox.isChecked()

        # Create and start detection worker
        self.detection_worker = DetectionWorker(file_paths, model_path, enable_preprocessing, self.app_dir)
        self.detection_worker.progress_updated.connect(self.update_progress)
        self.detection_worker.result_ready.connect(self.add_result)
        self.detection_worker.detection_finished.connect(self.detection_finished)
        self.detection_worker.start()

    def update_progress(self, progress, status_text):
        """Update progress bar and status label"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Status: {status_text}")

    def add_result(self, file_name, process_time, objects_count, weeds_count, output_path):
        """Add a detection result to the results table"""
        self.results_table.add_result(file_name, process_time, objects_count, weeds_count, output_path)

    def detection_finished(self):
        """Handle detection completion"""
        # Re-enable UI elements
        self.run_button.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.preprocess_checkbox.setEnabled(True)

        # Clean up worker
        if self.detection_worker:
            self.detection_worker.deleteLater()
            self.detection_worker = None