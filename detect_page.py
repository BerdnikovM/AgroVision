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
                             QGroupBox, QListWidget, QAbstractItemView, QSizePolicy, QScrollArea,
                             QStackedWidget, QSlider, QMessageBox)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QDragEnterEvent, QDropEvent, QImage
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QMimeData, QUrl, QThread, QTimer, QMutex

# Import YOLO from ultralytics if available
try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Предупреждение: ultralytics не установлен. Обнаружение будет симулировано.")


class CameraThread(QThread):
    """Thread for capturing and processing webcam feed"""
    frame_signal = pyqtSignal(QImage, list)  # Signal to send processed frame and detections
    status_signal = pyqtSignal(str)  # Signal to send status messages
    fps_signal = pyqtSignal(float)  # Signal to send FPS updates

    def __init__(self, model_path, enable_preprocessing, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.enable_preprocessing = enable_preprocessing
        self.running = False
        self.mutex = QMutex()
        self.camera_id = 0  # Default camera ID
        self.confidence_threshold = 0.25  # Default confidence threshold

    def run(self):
        """Main thread function to capture and process webcam feed"""
        self.running = True
        self.status_signal.emit("Инициализация камеры...")

        # Try to open the camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.status_signal.emit("Ошибка: Не удалось открыть веб-камеру")
            self.running = False
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Load the YOLO model
        try:
            if ULTRALYTICS_AVAILABLE:
                self.status_signal.emit("Загрузка модели YOLO...")
                model = YOLO(self.model_path)
                self.status_signal.emit("Модель успешно загружена")
            else:
                self.status_signal.emit("Ультралитикc не доступны. Обнаружение будет моделироваться.")
                model = None
        except Exception as e:
            self.status_signal.emit(f"Модель загрузки ошибок: {str(e)}")
            cap.release()
            self.running = False
            return

        # FPS calculation variables
        fps = 0
        frame_count = 0
        start_time = time.time()

        # Main processing loop
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Ошибка: Не удалось захватить кадр")
                break

            # Mirror the frame for a more natural webcam experience
            frame = cv2.flip(frame, 1)

            # Apply preprocessing if enabled
            if self.enable_preprocessing:
                frame = self.apply_clahe(frame)

            # Process frame with YOLO model
            detections = []
            if ULTRALYTICS_AVAILABLE and model:
                results = model(frame, conf=self.confidence_threshold)

                # Extract detection information
                if results and len(results) > 0:
                    r = results[0]

                    # Process each detection
                    for i in range(len(r.boxes)):
                        # Get box coordinates
                        box = r.boxes[i].xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = box.astype(int)

                        # Get confidence and class
                        conf = float(r.boxes[i].conf[0])
                        cls_id = int(r.boxes[i].cls[0])
                        cls_name = model.names[cls_id]

                        # Add to detections list
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'conf': conf,
                            'class': cls_name
                        })

                # Draw detections on the frame
                frame = self.draw_detections(frame, detections)
            else:
                # Simulate detections if model is not available
                if frame_count % 30 == 0:  # Add simulated detections every 30 frames
                    num_detections = np.random.randint(1, 5)
                    for _ in range(num_detections):
                        x1 = np.random.randint(50, frame.shape[1] - 150)
                        y1 = np.random.randint(50, frame.shape[0] - 150)
                        w = np.random.randint(50, 150)
                        h = np.random.randint(50, 150)
                        x2 = x1 + w
                        y2 = y1 + h
                        conf = np.random.uniform(0.6, 0.95)
                        cls_name = np.random.choice(["weed", "crop"])

                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'conf': conf,
                            'class': cls_name
                        })

                # Draw simulated detections
                frame = self.draw_detections(frame, detections)

            # Convert frame to QImage for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit the processed frame
            self.frame_signal.emit(qt_image, detections)

            # Calculate and update FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                self.fps_signal.emit(fps)
                frame_count = 0
                start_time = time.time()

            # Sleep to reduce CPU usage
            self.msleep(1)

        # Clean up
        cap.release()
        self.status_signal.emit("Камера остановлена")

    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.wait()

    def apply_clahe(self, img):
        """Apply CLAHE preprocessing to enhance image contrast"""
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

    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on the frame"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['conf']
            cls_name = det['class']

            # Determine color based on class
            if cls_name.lower() == "weed":
                color = (0, 0, 255)  # Red for weeds
            else:
                color = (0, 255, 0)  # Green for crops

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Create label with class name and confidence
            label = f"{cls_name}: {conf:.2f}"

            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return frame

    def set_confidence_threshold(self, value):
        """Set the confidence threshold for detections"""
        self.mutex.lock()
        self.confidence_threshold = value
        self.mutex.unlock()


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
                self.progress_updated.emit(progress, f"Обработка {file_name} ({i + 1}/{total_files})...")

                # Start timing the processing
                start_time = time.time()

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

                # Process the file and get actual detection counts
                if is_video:
                    objects_count, weeds_count = self.process_video(model, file_path, output_file, csv_file)
                else:
                    objects_count, weeds_count = self.process_image(model, file_path, output_file, csv_file)

                # Calculate actual processing time
                end_time = time.time()
                process_time = f"{end_time - start_time:.2f}"

                # Emit result with actual values
                self.result_ready.emit(file_name, process_time, objects_count, weeds_count, output_file)

            self.progress_updated.emit(100, "Детекция завершена")
            self.detection_finished.emit()

        except Exception as e:
            self.progress_updated.emit(0, f"Ошибка: {str(e)}")
            self.detection_finished.emit()

    def process_image(self, model, input_path, output_path, csv_path):
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            raise RuntimeError(f"Не удалось открыть изображение {input_path}")

        # Apply CLAHE preprocessing if enabled
        if self.enable_preprocessing:
            img = self.apply_clahe(img)

        # Run detection
        results = model(img)
        r = results[0]

        # Count total objects and weeds
        total_objects = len(r.boxes)
        weed_count = 0

        # Save detection results to CSV and count weeds
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

                # Count weeds
                if cls_name.lower() == "weed":
                    weed_count += 1

                # Write row
                writer.writerow([0, cls_name, f"{conf:.4f}", int(x1), int(y1), int(x2), int(y2)])

        # Save the annotated image
        r.save(filename=output_path)

        return total_objects, weed_count

    def process_video(self, model, input_path, output_path, csv_path):
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize counters
        total_objects = 0
        weed_count = 0

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
                    self.progress_updated.emit(progress, f"Обработка видеокадра {frame_idx}/{total_frames}...")

                # Apply CLAHE preprocessing if enabled
                if self.enable_preprocessing:
                    frame = self.apply_clahe(frame)

                # Run detection
                results = model(frame)
                r = results[0]

                # Count objects in this frame
                frame_objects = len(r.boxes)
                total_objects += frame_objects

                # Save all boxes for this frame
                for i in range(len(r.boxes)):
                    # Coordinates
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    # Confidence
                    conf = float(r.boxes.conf[i])
                    # Class ID and name
                    cls_id = int(r.boxes.cls[i])
                    cls_name = model.names[cls_id]

                    # Count weeds
                    if cls_name.lower() == "weed":
                        weed_count += 1

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

        return total_objects, weed_count

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
            self.progress_updated.emit(progress, f"Моделирование обнаружения для {file_name} ({i + 1}/{total_files})...")

            # Start timing
            start_time = time.time()

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

            # Generate realistic detection counts for simulation
            objects_count = np.random.randint(5, 20)
            weeds_count = np.random.randint(0, min(objects_count, 10))  # Weeds can't exceed total objects

            # Simulate CSV creation with consistent counts
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])

                # Generate detections matching the counts
                weed_detections_written = 0
                for j in range(objects_count):
                    frame = 0 if file_ext in ['.jpg', '.jpeg', '.png'] else np.random.randint(0, 100)

                    # Determine class based on remaining weed count
                    if weed_detections_written < weeds_count:
                        cls_name = "weed"
                        weed_detections_written += 1
                    else:
                        cls_name = "crop"

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
                print(f"Ошибка при копировании файла: {e}")

            # Calculate actual processing time
            end_time = time.time()
            process_time = f"{end_time - start_time:.2f}"

            # Emit result with consistent simulated values
            self.result_ready.emit(file_name, process_time, objects_count, weeds_count, output_file)

        self.progress_updated.emit(100, "Моделирование завершено")
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

        self.label = QLabel("Перетащите сюда файлы или")
        self.label.setFont(QFont("Arial", 12))
        self.label.setStyleSheet("color: #cccccc; border: none;")
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.browse_button = QPushButton("Открыть")
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

        # Set fixed height
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(120)

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
        self.camera_thread = None
        self.is_camera_active = False

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
        # 1) Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 2) Create a stacked widget to switch between file detection and camera detection
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # 3) Create file detection page
        self.file_detection_page = QWidget()
        self.init_file_detection_ui()
        self.stacked_widget.addWidget(self.file_detection_page)

        # 4) Create camera detection page
        self.camera_detection_page = QWidget()
        self.init_camera_detection_ui()
        self.stacked_widget.addWidget(self.camera_detection_page)

        # Start with file detection page
        self.stacked_widget.setCurrentIndex(0)

    def init_file_detection_ui(self):
        # Create a scroll area for the file detection page
        scroll = QScrollArea(self.file_detection_page)
        scroll.setWidgetResizable(True)

        # Container for all content
        container = QWidget()
        scroll.setWidget(container)

        # Outer layout for the file detection page
        outer_layout = QVBoxLayout(self.file_detection_page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Main layout for the container
        main_layout = QVBoxLayout(container)
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
        title_label = QLabel("Детекция")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        # Add stretch to push title to center
        header_layout.addStretch()

        # Camera button
        camera_button = QPushButton("📷")
        camera_button.setToolTip("Переключение на обнаружение камеры")
        camera_button.setFont(QFont("Arial", 16))
        camera_button.setFixedSize(40, 40)
        camera_button.setStyleSheet("""
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
        camera_button.clicked.connect(self.switch_to_camera)
        header_layout.addWidget(camera_button)

        main_layout.addLayout(header_layout)

        # Input Files section
        input_group = QGroupBox("Входные файлы")
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

        # Selected Files section
        files_group = QGroupBox("Входные файлы")
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

        model_label = QLabel("Модель")
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
        refresh_button.setToolTip("Обновить список моделей")
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
        self.model_loaded_label = QLabel("Модель не выбрана")
        self.model_loaded_label.setFont(QFont("Arial", 10))
        self.model_loaded_label.setStyleSheet("color: #aaaaaa;")
        model_layout.addWidget(self.model_loaded_label)

        main_layout.addLayout(model_layout)

        # Preprocessing toggle
        preprocess_layout = QHBoxLayout()

        self.preprocess_checkbox = QCheckBox("Включить предобработку (CLAHE)")
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
        info_button.setToolTip("CLAHE (Contrast Limited Adaptive Histogram Equalization) повышает контрастность изображения")
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
        self.run_button = QPushButton("Запуск детекции")
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
        self.status_label = QLabel("Статус: Готов")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #aaaaaa;")
        main_layout.addWidget(self.status_label)

        # Results section
        results_label = QLabel("Результаты детекции")
        results_label.setFont(QFont("Arial", 16, QFont.Bold))
        results_label.setStyleSheet("color: white;")
        main_layout.addWidget(results_label)

        self.results_table = ResultsTableWidget()
        main_layout.addWidget(self.results_table)

        # Connect signals
        self.model_combo.currentIndexChanged.connect(self.model_changed)

    def init_camera_detection_ui(self):
        # Main layout for camera detection page
        main_layout = QVBoxLayout(self.camera_detection_page)
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
        title_label = QLabel("Детекция камеры")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        # Add stretch to push title to center
        header_layout.addStretch()

        # Files button
        files_button = QPushButton("📁")
        files_button.setToolTip("Переключение на детекцию файлов")
        files_button.setFont(QFont("Arial", 16))
        files_button.setFixedSize(40, 40)
        files_button.setStyleSheet("""
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
        files_button.clicked.connect(self.switch_to_files)
        header_layout.addWidget(files_button)

        main_layout.addLayout(header_layout)

        # Camera feed and controls layout
        camera_layout = QHBoxLayout()

        # Left panel for camera feed
        camera_feed_group = QGroupBox("Камера")
        camera_feed_group.setStyleSheet("""
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

        camera_feed_layout = QVBoxLayout(camera_feed_group)

        # Camera display label
        self.camera_label = QLabel("Камера не запущена")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("color: #aaaaaa; background-color: #1a1a24;")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_feed_layout.addWidget(self.camera_label)

        # FPS counter
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Arial", 10))
        self.fps_label.setStyleSheet("color: white;")
        self.fps_label.setAlignment(Qt.AlignRight)
        camera_feed_layout.addWidget(self.fps_label)

        camera_layout.addWidget(camera_feed_group, 7)  # 70% of width

        # Right panel for controls
        controls_group = QGroupBox("Настройки")
        controls_group.setStyleSheet("""
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

        controls_layout = QVBoxLayout(controls_group)

        # Model selection
        model_label = QLabel("Модель")
        model_label.setFont(QFont("Arial", 14, QFont.Bold))
        model_label.setStyleSheet("color: white;")
        controls_layout.addWidget(model_label)

        self.camera_model_combo = QComboBox()
        self.camera_model_combo.setStyleSheet("""
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
        controls_layout.addWidget(self.camera_model_combo)

        # Preprocessing toggle
        self.camera_preprocess_checkbox = QCheckBox("Включить предобработку (CLAHE)")
        self.camera_preprocess_checkbox.setFont(QFont("Arial", 12))
        self.camera_preprocess_checkbox.setStyleSheet("""
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
        controls_layout.addWidget(self.camera_preprocess_checkbox)

        # Confidence threshold
        threshold_label = QLabel("Порог уверенности")
        threshold_label.setFont(QFont("Arial", 12))
        threshold_label.setStyleSheet("color: white;")
        controls_layout.addWidget(threshold_label)

        threshold_layout = QHBoxLayout()

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(25)  # Default 0.25
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #2a2a34;
                height: 8px;
                background: #1a1a24;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2e8b57;
                border: 1px solid #2e8b57;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: #2e8b57;
                border-radius: 4px;
            }
        """)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value_label = QLabel("0.25")
        self.threshold_value_label.setFont(QFont("Arial", 12))
        self.threshold_value_label.setStyleSheet("color: white;")
        self.threshold_value_label.setMinimumWidth(40)
        threshold_layout.addWidget(self.threshold_value_label)

        controls_layout.addLayout(threshold_layout)

        # Add spacer
        controls_layout.addStretch()

        # Start/Stop camera button
        self.camera_button = QPushButton("Запуск камеры")
        self.camera_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.camera_button.setFixedHeight(50)
        self.camera_button.setStyleSheet("""
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
        self.camera_button.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.camera_button)

        # Capture button
        self.capture_button = QPushButton("Захватить кадр")
        self.capture_button.setFont(QFont("Arial", 14))
        self.capture_button.setFixedHeight(40)
        self.capture_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a34;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a3a44;
            }
            QPushButton:disabled {
                background-color: #1a1a24;
                color: #aaaaaa;
            }
        """)
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setEnabled(False)  # Initially disabled
        controls_layout.addWidget(self.capture_button)

        # Status label
        self.camera_status_label = QLabel("Статус: Готов")
        self.camera_status_label.setFont(QFont("Arial", 10))
        self.camera_status_label.setStyleSheet("color: #aaaaaa;")
        controls_layout.addWidget(self.camera_status_label)

        camera_layout.addWidget(controls_group, 3)  # 30% of width

        main_layout.addLayout(camera_layout)

        # Detection statistics
        stats_group = QGroupBox("Статистика детекции")
        stats_group.setStyleSheet("""
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

        stats_layout = QHBoxLayout(stats_group)

        # Total detections
        self.total_detections_label = QLabel("Всего детекций: 0")
        self.total_detections_label.setFont(QFont("Arial", 12))
        self.total_detections_label.setStyleSheet("color: white;")
        stats_layout.addWidget(self.total_detections_label)

        # Weed detections
        self.weed_detections_label = QLabel("Сорняки: 0")
        self.weed_detections_label.setFont(QFont("Arial", 12))
        self.weed_detections_label.setStyleSheet("color: white;")
        stats_layout.addWidget(self.weed_detections_label)

        # Crop detections
        self.crop_detections_label = QLabel("Культура: 0")
        self.crop_detections_label.setFont(QFont("Arial", 12))
        self.crop_detections_label.setStyleSheet("color: white;")
        stats_layout.addWidget(self.crop_detections_label)

        # Average confidence
        self.avg_confidence_label = QLabel("Ср. Уверенность: 0.00")
        self.avg_confidence_label.setFont(QFont("Arial", 12))
        self.avg_confidence_label.setStyleSheet("color: white;")
        stats_layout.addWidget(self.avg_confidence_label)

        main_layout.addWidget(stats_group)

    def add_files(self, file_paths):
        self.file_list.add_files(file_paths)
        # Enable run button if files are added and a model is selected
        self.update_run_button_state()

    def go_back(self):
        """Navigate back to home page"""
        # Stop any active camera or detection
        if self.is_camera_active:
            self.stop_camera()

        if hasattr(self.parent_app, 'stacked_widget'):
            self.parent_app.stacked_widget.setCurrentIndex(0)

    def load_models(self):
        """Load model files from the models directory"""
        self.model_combo.clear()
        self.camera_model_combo.clear()

        # Get models directory
        models_dir = os.path.join(self.app_dir, "models")

        if os.path.exists(models_dir):
            # Get all .pt files
            model_files = [f for f in os.listdir(models_dir) if
                           f.endswith('.pt') and os.path.isfile(os.path.join(models_dir, f))]

            # Add models to combo boxes
            for model_file in model_files:
                self.model_combo.addItem(model_file)
                self.camera_model_combo.addItem(model_file)

            # Select first model if available
            if model_files:
                self.model_combo.setCurrentIndex(0)
                self.camera_model_combo.setCurrentIndex(0)
                self.model_loaded_label.setText(f"Загружено: {os.path.join(models_dir, self.model_combo.currentText())}")
            else:
                self.model_loaded_label.setText("В каталоге models не найдено ни одного файла модели")
        else:
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            self.model_loaded_label.setText("Создан каталог моделей. Пожалуйста, добавьте файлы моделей.")

    def refresh_models(self):
        """Refresh the model list"""
        self.status_label.setText("Статус: Обновление списка моделей...")
        self.load_models()
        self.status_label.setText("Статус: Список моделей обновлен")
        self.update_run_button_state()

    def model_changed(self, index):
        """Handle model selection change"""
        if self.model_combo.count() > 0:
            models_dir = os.path.join(self.app_dir, "models")
            self.model_loaded_label.setText(f"Загружено: {os.path.join(models_dir, self.model_combo.currentText())}")
        else:
            self.model_loaded_label.setText("Нет моделей в наличии")

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
            self.status_label.setText("Статус: Файлы не выбраны")
            return

        if self.model_combo.count() == 0:
            self.status_label.setText("Статус: Модель не выбрана")
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
        self.status_label.setText(f"Статус: {status_text}")

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

    def switch_to_camera(self):
        """Switch to camera detection mode"""
        self.stacked_widget.setCurrentIndex(1)

        # Sync model selection with file detection page
        if self.model_combo.currentIndex() >= 0:
            self.camera_model_combo.setCurrentIndex(self.model_combo.currentIndex())

        # Sync preprocessing checkbox
        self.camera_preprocess_checkbox.setChecked(self.preprocess_checkbox.isChecked())

    def switch_to_files(self):
        """Switch to file detection mode"""
        # Stop camera if active
        if self.is_camera_active:
            self.stop_camera()

        self.stacked_widget.setCurrentIndex(0)

        # Sync model selection with camera detection page
        if self.camera_model_combo.currentIndex() >= 0:
            self.model_combo.setCurrentIndex(self.camera_model_combo.currentIndex())

        # Sync preprocessing checkbox
        self.preprocess_checkbox.setChecked(self.camera_preprocess_checkbox.isChecked())

    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_active:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start the camera feed with detection"""
        if self.camera_model_combo.count() == 0:
            self.camera_status_label.setText("Статус: Модель не выбрана")
            return

        # Get model path
        models_dir = os.path.join(self.app_dir, "models")
        model_path = os.path.join(models_dir, self.camera_model_combo.currentText())

        # Get preprocessing state
        enable_preprocessing = self.camera_preprocess_checkbox.isChecked()

        # Create and start camera thread
        self.camera_thread = CameraThread(model_path, enable_preprocessing)
        self.camera_thread.frame_signal.connect(self.update_camera_frame)
        self.camera_thread.status_signal.connect(self.update_camera_status)
        self.camera_thread.fps_signal.connect(self.update_fps)

        # Set initial confidence threshold
        threshold_value = self.threshold_slider.value() / 100.0
        self.camera_thread.set_confidence_threshold(threshold_value)

        # Connect threshold slider to camera thread
        self.threshold_slider.valueChanged.connect(self.update_confidence_threshold)

        # Start the thread
        self.camera_thread.start()

        # Update UI
        self.is_camera_active = True
        self.camera_button.setText("Стоп камера")
        self.camera_button.setStyleSheet("""
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
        self.capture_button.setEnabled(True)
        self.camera_model_combo.setEnabled(False)
        self.camera_preprocess_checkbox.setEnabled(False)

    def stop_camera(self):
        """Stop the camera feed"""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
            self.camera_thread.deleteLater()
            self.camera_thread = None

        # Update UI
        self.is_camera_active = False
        self.camera_button.setText("Запуск камеры")
        self.camera_button.setStyleSheet("""
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
        self.capture_button.setEnabled(False)
        self.camera_model_combo.setEnabled(True)
        self.camera_preprocess_checkbox.setEnabled(True)

        # Reset camera label
        self.camera_label.setText("Камера не запущена")

        # Reset statistics
        self.total_detections_label.setText("Всего детекций: 0")
        self.weed_detections_label.setText("Сорняки: 0")
        self.crop_detections_label.setText("Культура: 0")
        self.avg_confidence_label.setText("Ср. Уверенность: 0.00")
        self.fps_label.setText("FPS: 0")

    def update_camera_frame(self, qt_image, detections):
        """Update the camera feed display with the processed frame"""
        # Display the image
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_label.setPixmap(pixmap)

        # Update detection statistics
        total_detections = len(detections)
        weed_count = sum(1 for det in detections if det['class'].lower() == 'weed')
        crop_count = total_detections - weed_count

        # Calculate average confidence
        avg_conf = 0.0
        if total_detections > 0:
            avg_conf = sum(det['conf'] for det in detections) / total_detections

        # Update labels
        self.total_detections_label.setText(f"Всего детекций: {total_detections}")
        self.weed_detections_label.setText(f"Сорняки: {weed_count}")
        self.crop_detections_label.setText(f"Культура: {crop_count}")
        self.avg_confidence_label.setText(f"Ср. Уверенность: {avg_conf:.2f}")

    def update_camera_status(self, status_text):
        """Update the camera status label"""
        self.camera_status_label.setText(f"Статус: {status_text}")

    def update_fps(self, fps):
        """Update the FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_threshold_label(self, value):
        """Update the threshold value label"""
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")

        # Update camera thread if running
        if self.is_camera_active and self.camera_thread:
            self.camera_thread.set_confidence_threshold(threshold)

    def update_confidence_threshold(self, value):
        """Update the confidence threshold in the camera thread"""
        if self.camera_thread and self.camera_thread.isRunning():
            threshold = value / 100.0
            self.camera_thread.set_confidence_threshold(threshold)

    def capture_frame(self):
        """Capture and save the current camera frame"""
        if not self.is_camera_active or not self.camera_label.pixmap():
            return

        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.app_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"camera_capture_{timestamp}.jpg")

        # Save the current pixmap
        if self.camera_label.pixmap().save(file_path):
            self.camera_status_label.setText(f"Статус: Кадр захвачен и сохранен в {file_path}")
        else:
            self.camera_status_label.setText("Статус: Не удалось сохранить захваченный кадр")

    def closeEvent(self, event):
        """Handle application close event"""
        # Stop any running processes
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.stop()
            self.detection_worker.wait()

        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()

        super().closeEvent(event)