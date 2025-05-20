import os
import sys
import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QListWidget, QSplitter, QFileDialog, QScrollArea, QFrame,
                             QSizePolicy, QGroupBox, QAbstractItemView, QSlider, QMessageBox)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QSize, QUrl, QTimer, pyqtSignal, QThread
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget


class VideoThread(QThread):
    """Thread for video playback using OpenCV to ensure compatibility"""
    change_pixmap_signal = pyqtSignal(QImage)
    frame_count_signal = pyqtSignal(int)
    current_frame_signal = pyqtSignal(int)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.running = True
        self.paused = False
        self.position = 0
        self.fps = 30  # Default FPS

    def run(self):
        cap = cv2.VideoCapture(self.file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.file_path}")
            return

        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # Default to 30 FPS if unable to determine

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count_signal.emit(frame_count)

        # Set position if needed
        if self.position > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.position)

        delay = int(1000 / self.fps)  # Delay between frames in ms

        while self.running:
            if not self.paused:
                ret, frame = cap.read()
                if ret:
                    # Convert frame to QImage
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                    # Emit signal with the image
                    self.change_pixmap_signal.emit(qt_image)

                    # Emit current frame position
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.current_frame_signal.emit(current_frame)

                    # Sleep to maintain proper playback speed
                    self.msleep(delay)
                else:
                    # End of video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to beginning
                    self.position = 0
            else:
                # When paused, just sleep to avoid high CPU usage
                self.msleep(100)

        # Clean up
        cap.release()

    def toggle_pause(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False
        self.position = 0
        self.wait()

    def set_position(self, position):
        self.position = position
        # If paused, we need to update the frame immediately
        if self.paused and self.isRunning():
            cap = cv2.VideoCapture(self.file_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ret, frame = cap.read()
                if ret:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_image)
                cap.release()


class VisualizationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_thread = None
        self.current_file_path = None
        self.init_ui()
        self.load_output_files()

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
        title_label = QLabel("Visualization")
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

        # Create splitter for file list and preview
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #2a2a34;
                width: 2px;
            }
        """)

        # File list section
        file_group = QGroupBox("Output Files")
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

        # Preview section
        preview_group = QGroupBox("Preview")
        preview_group.setStyleSheet("""
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

        preview_layout = QVBoxLayout(preview_group)

        # Create a scroll area for the image/video preview
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1a1a24;
                border: none;
            }
        """)

        # Container for image or video
        self.preview_container = QWidget()
        self.preview_container.setStyleSheet("background-color: #1a1a24;")
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.preview_layout.setAlignment(Qt.AlignCenter)

        # Image/video display label
        self.display_label = QLabel("No file selected")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("color: #aaaaaa; background-color: #1a1a24;")
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_layout.addWidget(self.display_label)

        # Video controls
        self.video_controls = QWidget()
        self.video_controls.setStyleSheet("background-color: #1a1a24;")
        video_controls_layout = QVBoxLayout(self.video_controls)

        # Timeline slider
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setStyleSheet("""
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
        self.timeline_slider.setRange(0, 100)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self.set_video_position)
        video_controls_layout.addWidget(self.timeline_slider)

        # Play/Stop buttons
        buttons_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton("▶")
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a34;
                color: white;
                border: none;
                border-radius: 15px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3a3a44;
            }
        """)
        self.play_button.clicked.connect(self.toggle_play)
        buttons_layout.addWidget(self.play_button)

        # Stop button
        self.stop_button = QPushButton("■")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a34;
                color: white;
                border: none;
                border-radius: 15px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3a3a44;
            }
        """)
        self.stop_button.clicked.connect(self.stop_video)
        buttons_layout.addWidget(self.stop_button)

        # Frame counter label
        self.frame_label = QLabel("0/0")
        self.frame_label.setStyleSheet("color: white;")
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        buttons_layout.addWidget(self.frame_label)

        video_controls_layout.addLayout(buttons_layout)

        # Add video controls to preview layout
        self.preview_layout.addWidget(self.video_controls)
        self.video_controls.hide()  # Hide initially

        scroll_area.setWidget(self.preview_container)
        preview_layout.addWidget(scroll_area)

        # Add widgets to splitter
        splitter.addWidget(file_group)
        splitter.addWidget(preview_group)
        splitter.setSizes([200, 600])  # Set initial sizes

        main_layout.addWidget(splitter)

    def go_back(self):
        """Navigate back to home page"""
        # Stop any playing video before navigating away
        self.stop_video()

        if hasattr(self.parent_app, 'stacked_widget'):
            self.parent_app.stacked_widget.setCurrentIndex(0)

    def load_output_files(self):
        """Load output files from the output directory"""
        self.file_list.clear()

        output_dir = os.path.join(self.app_dir, "output")
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

            # Sort files by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)

            for file in files:
                self.file_list.addItem(file)

    def file_selected(self, current, previous):
        """Handle file selection"""
        # Stop any playing video
        self.stop_video()

        if current is None:
            return

        file_name = current.text()
        file_path = os.path.join(self.app_dir, "output", file_name)
        self.current_file_path = file_path

        # Check if file exists
        if not os.path.exists(file_path):
            self.display_label.setText(f"File not found: {file_path}")
            self.video_controls.hide()
            return

        # Determine file type
        file_ext = os.path.splitext(file_name)[1].lower()

        # Handle video files
        if file_ext in ['.mp4', '.avi', '.mov']:
            self.display_label.setText("Loading video...")
            self.video_controls.show()
            self.play_video(file_path)

        # Handle image files
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            self.video_controls.hide()

            # Load image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale pixmap to fit the label while maintaining aspect ratio
                self.display_label.setPixmap(pixmap.scaled(
                    self.display_label.width(),
                    self.display_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
            else:
                self.display_label.setText(f"Failed to load image: {file_path}")

        # Handle unsupported files
        else:
            self.display_label.setText(f"Unsupported file format: {file_ext}")
            self.video_controls.hide()

    def play_video(self, file_path):
        """Play video using OpenCV in a separate thread"""
        try:
            # Create and start video thread
            self.video_thread = VideoThread(file_path)
            self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
            self.video_thread.frame_count_signal.connect(self.update_frame_count)
            self.video_thread.current_frame_signal.connect(self.update_current_frame)
            self.video_thread.start()

            # Update button state
            self.play_button.setText("⏸")

        except Exception as e:
            self.display_label.setText(f"Error playing video: {str(e)}")
            self.video_controls.hide()

    def update_video_frame(self, qt_image):
        """Update the video frame in the display label"""
        pixmap = QPixmap.fromImage(qt_image)

        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.display_label.width(),
            self.display_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.display_label.setPixmap(scaled_pixmap)

    def update_frame_count(self, count):
        """Update the total frame count and slider range"""
        self.timeline_slider.setRange(0, count)
        self.frame_label.setText(f"0/{count}")

    def update_current_frame(self, frame):
        """Update the current frame position"""
        # Update slider without triggering valueChanged
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame)
        self.timeline_slider.blockSignals(False)

        # Update frame counter label
        total = self.timeline_slider.maximum()
        self.frame_label.setText(f"{frame}/{total}")

    def toggle_play(self):
        """Toggle play/pause for video"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.toggle_pause()

            if self.video_thread.paused:
                self.play_button.setText("▶")
            else:
                self.play_button.setText("⏸")
        else:
            # If no video is playing, start playing the current file
            if self.current_file_path and os.path.exists(self.current_file_path):
                self.play_video(self.current_file_path)

    def stop_video(self):
        """Stop video playback"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
            self.play_button.setText("▶")

            # Reset slider and frame counter
            self.timeline_slider.setValue(0)
            self.frame_label.setText("0/0")

    def set_video_position(self, position):
        """Set the video position based on slider value"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_position(position)

            # Update frame counter label
            total = self.timeline_slider.maximum()
            self.frame_label.setText(f"{position}/{total}")

    def resizeEvent(self, event):
        """Handle resize events to update image/video scaling"""
        super().resizeEvent(event)

        # If an image is displayed, rescale it
        if hasattr(self, 'display_label') and self.display_label.pixmap() and not self.display_label.pixmap().isNull():
            current_pixmap = self.display_label.pixmap()
            self.display_label.setPixmap(current_pixmap.scaled(
                self.display_label.width(),
                self.display_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def closeEvent(self, event):
        """Handle close event to properly clean up resources"""
        self.stop_video()
        super().closeEvent(event)