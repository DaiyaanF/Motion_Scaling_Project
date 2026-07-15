import sys
import os
import json
import subprocess
import time
import signal
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QGridLayout,
    QLineEdit, QComboBox, QTabWidget, QFileDialog, QMessageBox,
    QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QFormLayout,
    QSlider, QProgressBar
)
from PyQt5.QtCore import (
    Qt, QProcess, pyqtSignal, QTimer, QThread, pyqtSlot,
    QSettings, QSize, QEvent
)
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QTextCursor, QKeyEvent


class TrialConfigDialog(QDialog):
    """Trial configuration dialog with motion scale display"""
    
    def __init__(self, parent=None, trial_history=None):
        super().__init__(parent)
        self.trial_history = trial_history if trial_history is not None else []
        self.parent_window = parent  # Store reference to main window
        self.setWindowTitle("Trial Configuration")
        self.setModal(True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QFormLayout()
        
        # Participant ID
        self.participant_id = QLineEdit()
        self.participant_id.setPlaceholderText("e.g., P001, John_Doe")
        layout.addRow("Participant ID:", self.participant_id)
        
        # Trial number
        self.trial_number = QSpinBox()
        self.trial_number.setMinimum(1)
        self.trial_number.setMaximum(999)
        layout.addRow("Trial Number:", self.trial_number)
        
        # Video FPS
        self.video_fps = QDoubleSpinBox()
        self.video_fps.setMinimum(1.0)
        self.video_fps.setMaximum(60.0)
        self.video_fps.setValue(15.0)
        self.video_fps.setSingleStep(0.5)
        layout.addRow("Video FPS:", self.video_fps)
        
        # Motion scale display (read-only)
        if self.parent_window and hasattr(self.parent_window, 'scale_value'):
            scale_value = self.parent_window.scale_value
            scale_label = QLabel(f"Current Motion Scale: {scale_value:.2f}")
            scale_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            layout.addRow("Motion Scaling:", scale_label)
        
        # Bag file name
        self.bag_name = QLineEdit()
        self.bag_name.setPlaceholderText("e.g., participant/trial#_pattern_scale")
        layout.addRow("Bag Name:", self.bag_name)
        
        # Auto-generate bag name
        self.auto_generate = QCheckBox("Auto-generate bag name")
        self.auto_generate.setChecked(True)
        self.auto_generate.toggled.connect(self.toggle_bag_name)
        layout.addRow(self.auto_generate)
        
        # Auto-increment trial number
        self.auto_increment = QCheckBox("Auto-increment trial number")
        self.auto_increment.setChecked(True)
        layout.addRow(self.auto_increment)
        
        # Update bag name when other fields change
        self.participant_id.textChanged.connect(self.on_participant_changed)
        self.trial_number.valueChanged.connect(self.update_bag_name)
        
        # Instructions
        instructions = QLabel(
            "<i>Tip: Trial numbers auto-increment for each participant. "
            "Motion scale will be included in bag name if different from 1.0.</i>"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; padding: 5px;")
        layout.addRow(instructions)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
        self.setLayout(layout)
        self.update_bag_name()
        
    def on_participant_changed(self):
        """Update trial number based on participant history"""
        if self.auto_increment.isChecked() and self.participant_id.text():
            # Find the highest trial number for this participant
            participant = self.participant_id.text()
            max_trial = 0
            
            for trial in self.trial_history:
                if trial.get('config', {}).get('participant_id', '') == participant:
                    trial_num = trial.get('config', {}).get('trial_number', 0)
                    max_trial = max(max_trial, trial_num)
            
            # Set to next trial number
            self.trial_number.setValue(max_trial + 1)
        
        self.update_bag_name()
        
    def toggle_bag_name(self, checked):
        self.bag_name.setEnabled(not checked)
        if checked:
            self.update_bag_name()
            
    def update_bag_name(self):
        if self.auto_generate.isChecked():
            participant = self.participant_id.text() or "unknown"
            trial = self.trial_number.value()
            
            # Get pattern from parent window
            pattern = "figure8"  # Default pattern
            if self.parent_window and hasattr(self.parent_window, 'pattern_combo'):
                pattern = self.parent_window.pattern_combo.currentText()
            
            # Get scale value
            scale = 1.0  # Default scale
            if self.parent_window and hasattr(self.parent_window, 'scale_value'):
                scale = self.parent_window.scale_value
            
            # Format the folder name
            scale_str = f"_scale{scale:.2f}".replace(".", "") if abs(scale - 1.0) > 0.01 else ""
            folder_name = f"trial{trial}_{pattern}{scale_str}"
            
            # Set bag name as participant/folder_structure
            bag_name = f"{participant}/{folder_name}"
            
            self.bag_name.setText(bag_name)
            
    def get_config(self):
        config = {
            'participant_id': self.participant_id.text(),
            'trial_number': self.trial_number.value(),
            'video_fps': self.video_fps.value(),
            'bag_name': self.bag_name.text()
        }
        
        # Include motion scale value
        if self.parent_window and hasattr(self.parent_window, 'scale_value'):
            config['motion_scale'] = self.parent_window.scale_value
            
        return config


class ProcessRunner(QWidget):
    """Widget for running and monitoring a single process with graceful shutdown"""
    
    process_started = pyqtSignal(str)
    process_finished = pyqtSignal(str, int)
    output_received = pyqtSignal(str, str)  # process_name, output
    
    def __init__(self, name, command, working_dir=None, parent=None, graceful_shutdown=False):
        super().__init__(parent)
        self.name = name
        self.command = command
        self.working_dir = working_dir
        self.process = None
        self.is_running = False
        self.graceful_shutdown = graceful_shutdown  # Use SIGINT instead of SIGTERM
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout()
        
        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: gray; font-size: 20px;")
        self.status_indicator.setFixedWidth(30)
        
        # Process name
        self.name_label = QLabel(self.name)
        self.name_label.setMinimumWidth(200)
        
        # Start/Stop button
        self.control_btn = QPushButton("Start")
        self.control_btn.clicked.connect(self.toggle_process)
        self.control_btn.setFixedWidth(80)
        
        # Command preview
        self.command_label = QLabel(self.command[:50] + "..." if len(self.command) > 50 else self.command)
        self.command_label.setStyleSheet("color: gray; font-style: italic;")
        
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.name_label)
        layout.addWidget(self.control_btn)
        layout.addWidget(self.command_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def update_command(self, new_command):
        """Update the command for this process runner"""
        self.command = new_command
        self.command_label.setText(new_command[:50] + "..." if len(new_command) > 50 else new_command)
        
    def toggle_process(self):
        if self.is_running:
            self.stop_process()
        else:
            self.start_process()
            
    def start_process(self):
        if self.process is not None:
            return
            
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished_handler)
        
        if self.working_dir:
            expanded_dir = os.path.expanduser(self.working_dir)
            self.process.setWorkingDirectory(expanded_dir)
            
        # Handle shell scripts vs python scripts vs ROS commands differently
        if self.command.startswith('./') or self.command.endswith('.sh'):
            # For shell scripts, use bash
            self.process.start('bash', ['-c', self.command])
        elif 'source' in self.command or 'rosrun' in self.command or 'roslaunch' in self.command or 'roscd' in self.command:
            # For ROS commands or commands with source, use bash -c
            self.process.start('bash', ['-c', self.command])
        else:
            # Parse command for regular executables
            parts = self.command.split()
            program = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            self.process.start(program, args)
        
        if self.process.waitForStarted():
            self.is_running = True
            self.status_indicator.setStyleSheet("color: green; font-size: 20px;")
            self.control_btn.setText("Stop")
            self.process_started.emit(self.name)
            self.output_received.emit(self.name, f"Started: {self.command}\n")
        else:
            self.output_received.emit(self.name, f"Failed to start: {self.command}\n")
            self.process = None
            
    def stop_process(self):
        if self.process is None:
            return
        
        if self.graceful_shutdown:
            # Send SIGINT (Ctrl+C) for graceful shutdown
            try:
                os.kill(self.process.processId(), signal.SIGINT)
                self.output_received.emit(self.name, "Sent SIGINT for graceful shutdown...\n")
            except:
                self.process.terminate()
        else:
            self.process.terminate()
            
        if not self.process.waitForFinished(5000):  # Wait 5 seconds
            self.process.kill()
            self.output_received.emit(self.name, "Force killed process\n")
            
    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        output = data.data().decode('utf-8', errors='replace')
        self.output_received.emit(self.name, output)
        
    def handle_stderr(self):
        data = self.process.readAllStandardError()
        output = data.data().decode('utf-8', errors='replace')
        self.output_received.emit(self.name, f"[ERROR] {output}")
        
    def process_finished_handler(self, exit_code, exit_status):
        self.is_running = False
        self.status_indicator.setStyleSheet("color: gray; font-size: 20px;")
        self.control_btn.setText("Start")
        self.process_finished.emit(self.name, exit_code)
        self.output_received.emit(self.name, f"Process finished with code: {exit_code}\n")
        self.process = None


class TrajectoryRunner(ProcessRunner):
    """Special ProcessRunner for trajectory generation that handles first-run initialization"""
    
    initialization_status_changed = pyqtSignal(str, str)  # status_text, style
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_first_run = True
        self.pending_command = None
        self.initializing = False
        
    def start_process(self):
        """Override to handle first trajectory initialization"""
        if self.is_first_run:
            self.is_first_run = False
            self.initializing = True
            self.pending_command = self.command
            
            # Update status
            self.initialization_status_changed.emit(
                "Trajectory System: Initializing...",
                "QLabel { background-color: #b3b33e; padding: 5px; border-radius: 3px; }"
            )
            
            # First, create and clear an empty trajectory
            self.output_received.emit(self.name, "Initializing trajectory system with empty cloud...\n")
            
            # Step 1: Start with clear pattern
            self.command = "python trajectory_generator.py --pattern clear"
            super().start_process()
            
            # Step 2: Let it run briefly then stop
            QTimer.singleShot(1000, self._continue_initialization)
        else:
            # Normal start
            super().start_process()
    
    def _continue_initialization(self):
        """Continue the initialization process"""
        # Stop the clear pattern
        self.stop_process()
        
        # Wait for process to fully stop
        QTimer.singleShot(500, self._start_desired_pattern)
    
    def _start_desired_pattern(self):
        """Start the actual desired pattern"""
        # Restore the desired command
        self.command = self.pending_command
        self.pending_command = None
        self.initializing = False
        
        # Update the command label to show the correct command
        self.command_label.setText(self.command[:50] + "..." if len(self.command) > 50 else self.command)
        
        # Update status
        self.initialization_status_changed.emit(
            "Trajectory System: Initialized",
            "QLabel { background-color: #2e7d32; padding: 5px; border-radius: 3px; color: white; }"
        )
        
        # Start the actual pattern
        super().start_process()
        self.output_received.emit(self.name, "Trajectory system initialized, pattern started.\n")
    
    def update_command(self, new_command):
        """Override to handle command updates during initialization"""
        if self.initializing and self.pending_command is not None:
            # If we're initializing, update the pending command instead
            self.pending_command = new_command
        else:
            # Normal update
            super().update_command(new_command)


class DVRKUserStudyGUI(QMainWindow):
    """
    Main GUI window for dVRK user study with simple manual motion scaling
    
    Key improvements in this version:
    - Trial configuration is now required before data collection
    - Visual indicators show configuration status
    - Auto-increment trial numbers for each participant
    - Quick configuration option for testing
    - Trajectory system auto-initializes on first use
    - Configuration and history persist between sessions
    - Simple manual motion scaling control
    """
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('DVRK_Study', 'UserStudyGUI')
        self.processes = {}
        self.current_config = None  # Changed from {} to None for clearer logic
        self.trial_history = []
        self.collection_in_progress = False
        
        # Initialize scaling attributes
        self.init_scaling_attributes()
        
        # Create output console early for debugging
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setFont(QFont("Consolas", 9))
        
        self.setup_ui()
        self.load_settings()
        
        # Check if we need to configure trial on startup
        if not self.current_config:
            QTimer.singleShot(500, self.prompt_trial_configuration)
            
    def init_scaling_attributes(self):
        """Initialize scaling-related attributes"""
        self.scale_value = 1.0
        self.scale_publisher = None
        self.scale_publish_timer = None
        
    def setup_ui(self):
        self.setWindowTitle("dVRK User Study Interface - Intel Motion Scaling")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar(main_layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Create tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        
        # Setup tab
        self.create_setup_tab()
        
        # Data collection tab
        self.create_collection_tab()
        
        # Analysis tab
        self.create_analysis_tab()
        
        # Motion Scaling tab
        self.create_scaling_tab()
        
        # History tab
        self.create_history_tab()
        
        # Output console at bottom
        self.create_output_console(splitter)
        
        # Set splitter sizes
        splitter.setSizes([600, 200])
        
        # Status bar
        self.status_bar = self.statusBar()
        self.update_status("Ready")
        
    def create_toolbar(self, layout):
        toolbar_layout = QHBoxLayout()
        
        # Trial configuration button - make it more prominent
        self.config_btn = QPushButton("⚙️ Configure Trial")
        self.config_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """
        )
        self.config_btn.clicked.connect(self.configure_trial)
        toolbar_layout.addWidget(self.config_btn)
        
        # Current trial info
        self.trial_info_label = QLabel("❌ No trial configured - Click 'Configure Trial' to begin")
        self.trial_info_label.setStyleSheet(
            "QLabel { background-color: #ffcccc; padding: 5px; border-radius: 3px; font-weight: bold; }"
        )
        toolbar_layout.addWidget(self.trial_info_label)
        
        toolbar_layout.addStretch()
        
        # Emergency stop button
        self.emergency_stop_btn = QPushButton("🛑 EMERGENCY STOP")
        self.emergency_stop_btn.setStyleSheet(
            "QPushButton { background-color: red; color: white; font-weight: bold; padding: 10px; }"
        )
        self.emergency_stop_btn.clicked.connect(self.emergency_stop)
        toolbar_layout.addWidget(self.emergency_stop_btn)
        
        layout.addLayout(toolbar_layout)
    
    def update_config_bag_name(self):
        """Update the bag name in current config when pattern changes"""
        if self.current_config and self.current_config.get('participant_id'):
            participant = self.current_config['participant_id']
            trial = self.current_config['trial_number']
            pattern = self.pattern_combo.currentText()
            
            scale_str = f"_scale{self.scale_value:.2f}".replace(".", "") if abs(self.scale_value - 1.0) > 0.01 else ""
            folder_name = f"trial{trial}_{pattern}{scale_str}"
            
            self.current_config['bag_name'] = f"{participant}/{folder_name}"
            self.update_trial_info()

    def create_setup_tab(self):
        setup_widget = QWidget()
        setup_layout = QVBoxLayout(setup_widget)
        
        # dVRK Console group
        console_group = QGroupBox("dVRK Console")
        console_layout = QVBoxLayout()
        
        # Path configuration
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("dVRK Config Path:"))
        self.dvrk_path = QLineEdit()
        # Default path - user can modify this
        self.dvrk_path.setText("~/catkin_ws/src/dvrk_config_jhu/jhu-daVinci")
        path_layout.addWidget(self.dvrk_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_path(self.dvrk_path))
        path_layout.addWidget(browse_btn)
        console_layout.addLayout(path_layout)
        
        # Console process
        self.console_runner = ProcessRunner(
            "dVRK Console",
            "source ~/catkin_ws/devel/setup.bash && rosrun dvrk_robot dvrk_console_json -j console-MTML-MTMR.json",
            "~/catkin_ws/src/dvrk_config_jhu/jhu-daVinci"
        )
        self.console_runner.output_received.connect(self.append_output)
        console_layout.addWidget(self.console_runner)
        
        # Update console runner path when dVRK path changes
        self.dvrk_path.textChanged.connect(self.update_dvrk_path)
        
        # Note about manual power/home
        note_label = QLabel("Note: After console starts, manually click 'Power On' and 'Home' in the dVRK console window")
        note_label.setStyleSheet("color: orange; font-style: italic; padding: 5px;")
        console_layout.addWidget(note_label)
        
        console_group.setLayout(console_layout)
        setup_layout.addWidget(console_group)
        
        # Simulation Environment group
        env_group = QGroupBox("Simulation Environment")
        env_layout = QVBoxLayout()
        
        # Environment runner
        self.env_runner = ProcessRunner(
            "AMBF Environment",
            "./run_env_3D_MED_COMPLEX_LND_420006.sh",
            "~/surgical_robotics_challenge"
        )
        self.env_runner.output_received.connect(self.append_output)
        env_layout.addWidget(self.env_runner)
        
        # Teleoperation runner
        self.teleop_runner = ProcessRunner(
            "Teleoperation",
            "./mtm_psm_pair_teleop_420006.sh",
            "~/surgical_robotics_challenge/scripts/surgical_robotics_challenge/teleoperation"
        )
        self.teleop_runner.output_received.connect(self.append_output)
        env_layout.addWidget(self.teleop_runner)
        
        env_group.setLayout(env_layout)
        setup_layout.addWidget(env_group)
        
        setup_layout.addStretch()
        self.tabs.addTab(setup_widget, "Setup")
        
    def create_collection_tab(self):
        collection_widget = QWidget()
        collection_layout = QVBoxLayout(collection_widget)
        
        # Trajectory group
        trajectory_group = QGroupBox("Trajectory Generation")
        trajectory_layout = QVBoxLayout()
        
        # Trajectory initialization status
        init_status_layout = QHBoxLayout()
        self.trajectory_init_label = QLabel("Trajectory System: Not initialized")
        self.trajectory_init_label.setStyleSheet(
            "QLabel { background-color: #ffcccc; padding: 5px; border-radius: 3px; }"
        )
        init_status_layout.addWidget(self.trajectory_init_label)
        init_status_layout.addStretch()
        trajectory_layout.addLayout(init_status_layout)
        
        # Trajectory pattern selection
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["figure8", "circle", "line", "triangle"])
        self.pattern_combo.currentTextChanged.connect(self.update_trajectory_command)
        self.pattern_combo.currentTextChanged.connect(lambda: self.update_trial_info() if self.current_config else None)
        self.pattern_combo.currentTextChanged.connect(self.update_config_bag_name)
        pattern_layout.addWidget(self.pattern_combo)
        pattern_layout.addStretch()
        trajectory_layout.addLayout(pattern_layout)
        
        # Trajectory parameters
        params_layout = QGridLayout()
        
        # Size parameters
        params_layout.addWidget(QLabel("Radius:"), 0, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.001, 0.5)
        self.radius_spin.setValue(0.03)
        self.radius_spin.setSingleStep(0.001)
        self.radius_spin.setSuffix(" m")
        self.radius_spin.valueChanged.connect(self.update_trajectory_command)
        params_layout.addWidget(self.radius_spin, 0, 1)
        
        params_layout.addWidget(QLabel("Length:"), 0, 2)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.001, 0.5)
        self.length_spin.setValue(0.06)
        self.length_spin.setSingleStep(0.001)
        self.length_spin.setSuffix(" m")
        self.length_spin.valueChanged.connect(self.update_trajectory_command)
        params_layout.addWidget(self.length_spin, 0, 3)
        
        # Position parameters
        params_layout.addWidget(QLabel("Pos X:"), 1, 0)
        self.pos_x_spin = QDoubleSpinBox()
        self.pos_x_spin.setRange(-0.5, 0.5)
        self.pos_x_spin.setValue(0.0)
        self.pos_x_spin.setSingleStep(0.001)
        self.pos_x_spin.setSuffix(" m")
        self.pos_x_spin.valueChanged.connect(self.update_trajectory_command)
        params_layout.addWidget(self.pos_x_spin, 1, 1)
        
        params_layout.addWidget(QLabel("Pos Y:"), 1, 2)
        self.pos_y_spin = QDoubleSpinBox()
        self.pos_y_spin.setRange(-0.5, 0.5)
        self.pos_y_spin.setValue(0.0)
        self.pos_y_spin.setSingleStep(0.001)
        self.pos_y_spin.setSuffix(" m")
        self.pos_y_spin.valueChanged.connect(self.update_trajectory_command)
        params_layout.addWidget(self.pos_y_spin, 1, 3)
        
        params_layout.addWidget(QLabel("Pos Z:"), 2, 0)
        self.pos_z_spin = QDoubleSpinBox()
        self.pos_z_spin.setRange(-0.5, 0.5)
        self.pos_z_spin.setValue(0.02)
        self.pos_z_spin.setSingleStep(0.001)
        self.pos_z_spin.setSuffix(" m")
        self.pos_z_spin.valueChanged.connect(self.update_trajectory_command)
        params_layout.addWidget(self.pos_z_spin, 2, 1)
        
        # Orientation parameters
        params_layout.addWidget(QLabel("Yaw:"), 2, 2)
        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(-180, 180)
        self.yaw_spin.setValue(0.0)
        self.yaw_spin.setSingleStep(1.0)
        self.yaw_spin.setSuffix("°")
        self.yaw_spin.valueChanged.connect(self.update_trajectory_command)
        params_layout.addWidget(self.yaw_spin, 2, 3)
        
        trajectory_layout.addLayout(params_layout)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        horizontal_btn = QPushButton("Horizontal")
        horizontal_btn.clicked.connect(lambda: self.apply_preset('horizontal'))
        preset_layout.addWidget(horizontal_btn)
        
        vertical_btn = QPushButton("Vertical")
        vertical_btn.clicked.connect(lambda: self.apply_preset('vertical_yz'))
        preset_layout.addWidget(vertical_btn)
        
        diagonal_btn = QPushButton("Diagonal")
        diagonal_btn.clicked.connect(lambda: self.apply_preset('diagonal'))
        preset_layout.addWidget(diagonal_btn)
        
        preset_layout.addStretch()
        trajectory_layout.addLayout(preset_layout)
        
        # Trajectory runner with graceful shutdown and first-run handling
        self.trajectory_runner = TrajectoryRunner(
            "Trajectory Generator",
            "python trajectory_generator.py --pattern figure8 --radius 0.03",
            "~/intel_motion_scaling/scripts",
            graceful_shutdown=True  # Enable graceful shutdown for trajectory
        )
        self.trajectory_runner.output_received.connect(self.append_output)
        self.trajectory_runner.initialization_status_changed.connect(self.update_trajectory_init_status)
        self.trajectory_runner.process_started.connect(lambda x: self.set_trajectory_controls_enabled(False))
        self.trajectory_runner.process_finished.connect(lambda x, code: self.set_trajectory_controls_enabled(True))

        trajectory_layout.addWidget(self.trajectory_runner)
        
        # Clear trajectory button
        clear_trajectory_btn = QPushButton("Clear Trajectory")
        clear_trajectory_btn.clicked.connect(self.clear_trajectory)
        trajectory_layout.addWidget(clear_trajectory_btn)
        
        # Reset initialization button (for troubleshooting)
        reset_init_btn = QPushButton("Reset Initialization")
        reset_init_btn.setToolTip("Reset the first-run flag to reinitialize trajectory system")
        reset_init_btn.clicked.connect(self.reset_trajectory_initialization)
        trajectory_layout.addWidget(reset_init_btn)
        
        # Set initial command
        self.update_trajectory_command()

        # Create overlay widget for disabled state
        self.trajectory_overlay = QLabel("🔒 Trajectory Running - Controls Locked")
        self.trajectory_overlay.setParent(trajectory_group)
        self.trajectory_overlay.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 152, 0, 0.9);
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                border: 2px solid #ff6b6b;
            }
        """)
        self.trajectory_overlay.setAlignment(Qt.AlignCenter)
        self.trajectory_overlay.hide()

        trajectory_group.setLayout(trajectory_layout)
        collection_layout.addWidget(trajectory_group)
        
        # Motion Scaling Indicator
        scaling_indicator_layout = QHBoxLayout()
        
        self.collection_scale_label = QLabel(f"Motion Scale: {getattr(self, 'scale_value', 1.0):.2f}")
        self.collection_scale_label.setStyleSheet(
            "QLabel { background-color: #e0e0e0; padding: 5px; border-radius: 3px; font-weight: bold; }"
        )
        scaling_indicator_layout.addWidget(self.collection_scale_label)
        
        # Quick scale adjustment buttons
        scale_down_btn = QPushButton("−")
        scale_down_btn.setFixedWidth(40)
        scale_down_btn.clicked.connect(lambda: self.adjust_scale(-0.1))
        scaling_indicator_layout.addWidget(scale_down_btn)
        
        scale_reset_btn = QPushButton("Reset")
        scale_reset_btn.setFixedWidth(60)
        scale_reset_btn.clicked.connect(lambda: self.set_scale_value(1.0))
        scaling_indicator_layout.addWidget(scale_reset_btn)
        
        scale_up_btn = QPushButton("+")
        scale_up_btn.setFixedWidth(40)
        scale_up_btn.clicked.connect(lambda: self.adjust_scale(0.1))
        scaling_indicator_layout.addWidget(scale_up_btn)
        
        scaling_indicator_layout.addStretch()
        collection_layout.addLayout(scaling_indicator_layout)
        
        # Data Collection group - Simplified
        data_group = QGroupBox("Data Collection")
        data_layout = QVBoxLayout()
        
        # Trial configuration reminder
        self.collection_config_reminder = QLabel(
            "⚠️ Configure trial before starting data collection"
        )
        self.collection_config_reminder.setStyleSheet(
            "QLabel { background-color: #fff3cd; padding: 8px; border-radius: 3px; "
            "border: 1px solid #ffeeba; color: #856404; }"
        )
        data_layout.addWidget(self.collection_config_reminder)
        
        # Collection status indicator
        self.collection_status_label = QLabel("Collection Status: Idle")
        self.collection_status_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 5px; font-weight: bold; }"
        )
        data_layout.addWidget(self.collection_status_label)
        
        # Single set of collection controls
        collection_controls = QHBoxLayout()
        self.start_collection_btn = QPushButton("▶️ Start Data Collection")
        self.start_collection_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            """
        )
        self.start_collection_btn.clicked.connect(self.start_data_collection)
        self.stop_collection_btn = QPushButton("⏹️ Stop Data Collection")
        self.stop_collection_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            """
        )
        self.stop_collection_btn.clicked.connect(self.stop_data_collection)
        self.stop_collection_btn.setEnabled(False)
        
        # Quick config button
        self.quick_config_btn = QPushButton("⚡ Quick Config")
        self.quick_config_btn.setToolTip("Configure with default settings (Participant: test, Trial: 1)")
        self.quick_config_btn.clicked.connect(self.quick_configure_trial)
        
        collection_controls.addWidget(self.start_collection_btn)
        collection_controls.addWidget(self.stop_collection_btn)
        collection_controls.addWidget(self.quick_config_btn)
        collection_controls.addStretch()
        data_layout.addLayout(collection_controls)
        
        # Distance tracking runner (hidden from user)
        self.tracking_runner = ProcessRunner(
            "Distance Tracking",
            "python find_closest_point.py bag_name_placeholder",
            "~/intel_motion_scaling/scripts",
            graceful_shutdown=True
        )
        self.tracking_runner.output_received.connect(self.append_output)
        self.tracking_runner.setVisible(False)  # Hide the runner widget
        data_layout.addWidget(self.tracking_runner)
        
        # Timer display
        self.timer_label = QLabel("Collection Time: 00:00")
        self.timer_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        data_layout.addWidget(self.timer_label)
        
        self.collection_timer = QTimer()
        self.collection_timer.timeout.connect(self.update_collection_time)
        self.collection_start_time = None
        
        # Note about topic compatibility
        note_label = QLabel("Note: Distance tracking uses unified '/ambf/env/World/trajectory' topic for all patterns")
        note_label.setStyleSheet("color: green; font-style: italic; padding: 5px;")
        data_layout.addWidget(note_label)
        
        data_group.setLayout(data_layout)
        collection_layout.addWidget(data_group)
        
        collection_layout.addStretch()
        
        # Store reference to the tab widget
        self.collection_tab = collection_widget
        
        self.tabs.addTab(collection_widget, "Data Collection")
        self.collection_tab_index = self.tabs.count() - 1  # Store the index
        
    def create_analysis_tab(self):
        analysis_widget = QWidget()
        analysis_widget.setObjectName("analysis_widget")  # Set object name for finding the tab
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QGridLayout()
        
        self.extract_videos_cb = QCheckBox("Extract Videos")
        self.extract_videos_cb.setChecked(True)
        options_layout.addWidget(self.extract_videos_cb, 0, 0)
        
        self.extract_distances_cb = QCheckBox("Extract Distances")
        self.extract_distances_cb.setChecked(True)
        options_layout.addWidget(self.extract_distances_cb, 0, 1)
        
        self.extract_clutch_cb = QCheckBox("Extract Clutch Data")
        self.extract_clutch_cb.setChecked(True)
        options_layout.addWidget(self.extract_clutch_cb, 1, 0)
        
        self.generate_plots_cb = QCheckBox("Generate Plots")
        self.generate_plots_cb.setChecked(True)
        options_layout.addWidget(self.generate_plots_cb, 1, 1)
        
        options_group.setLayout(options_layout)
        analysis_layout.addWidget(options_group)
        
        # Bag file selection
        bag_group = QGroupBox("Bag File Selection")
        bag_layout = QHBoxLayout()
        
        self.bag_file_path = QLineEdit()
        self.bag_file_path.setPlaceholderText("Select bag file to analyze...")
        bag_layout.addWidget(self.bag_file_path)
        
        browse_bag_btn = QPushButton("Browse")
        browse_bag_btn.clicked.connect(self.browse_bag_file)
        bag_layout.addWidget(browse_bag_btn)
        
        recent_bag_btn = QPushButton("Use Recent")
        recent_bag_btn.setToolTip("Select the most recently created bag file")
        recent_bag_btn.clicked.connect(self.set_last_bag_file)
        bag_layout.addWidget(recent_bag_btn)
        
        bag_group.setLayout(bag_layout)
        analysis_layout.addWidget(bag_group)
        
        # Analysis status section
        status_group = QGroupBox("Analysis Status")
        status_layout = QVBoxLayout()
        
        # Status indicator
        self.analysis_status_label = QLabel("Status: Ready")
        self.analysis_status_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 8px; font-weight: bold; border-radius: 3px; }"
        )
        status_layout.addWidget(self.analysis_status_label)
        
        # Progress info
        self.analysis_progress_label = QLabel("")
        self.analysis_progress_label.setVisible(False)
        status_layout.addWidget(self.analysis_progress_label)
        
        status_group.setLayout(status_layout)
        analysis_layout.addWidget(status_group)
        
        # Analysis runner (hidden from user view)
        self.analysis_runner = ProcessRunner(
            "Analysis",
            "python analysis.py",
            "~/intel_motion_scaling/scripts"
        )
        self.analysis_runner.output_received.connect(self.append_output)
        self.analysis_runner.process_started.connect(self.on_analysis_started)
        self.analysis_runner.process_finished.connect(self.on_analysis_finished)
        self.analysis_runner.setVisible(False)  # Hide the runner widget
        analysis_layout.addWidget(self.analysis_runner)
        
        # Run analysis button - now the main control
        self.run_analysis_btn = QPushButton("▶️ Run Analysis")
        self.run_analysis_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            """
        )
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        
        # Stop analysis button (initially hidden)
        self.stop_analysis_btn = QPushButton("⏹️ Stop Analysis")
        self.stop_analysis_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            """
        )
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setVisible(False)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_analysis_btn)
        button_layout.addWidget(self.stop_analysis_btn)
        button_layout.addStretch()
        analysis_layout.addLayout(button_layout)
        
        # Help text
        help_label = QLabel(
            "<i>Select a bag file and choose analysis options, then click 'Run Analysis' to process the data.</i>"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; padding: 5px;")
        analysis_layout.addWidget(help_label)
        
        analysis_layout.addStretch()
        self.tabs.addTab(analysis_widget, "Analysis")

    def create_scaling_tab(self):
        """Create the manual motion scaling control tab"""
        scaling_widget = QWidget()
        scaling_layout = QVBoxLayout(scaling_widget)
        
        # Motion Scaling Control
        control_group = QGroupBox("Motion Scaling Control")
        control_layout = QVBoxLayout()
        
        # Current scale display
        self.current_scale_display = QLabel("Current Scale: 1.000")
        self.current_scale_display.setStyleSheet(
            "QLabel { font-size: 24px; font-weight: bold; padding: 10px; "
            "background-color: #e0e0e0; border-radius: 5px; text-align: center; }"
        )
        self.current_scale_display.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.current_scale_display)
        
        # Scale slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("0.01"))
        
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(1)  # 0.01
        self.scale_slider.setMaximum(300)  # 3.00
        self.scale_slider.setValue(100)  # 1.00
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_slider.setTickInterval(10)
        self.scale_slider.valueChanged.connect(self.update_scale_from_slider)
        slider_layout.addWidget(self.scale_slider)
        
        slider_layout.addWidget(QLabel("1.50"))
        control_layout.addLayout(slider_layout)
        
        # Precise control spinbox
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(QLabel("Precise Control:"))
        
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setMinimum(0.01)
        self.scale_spinbox.setMaximum(3.00)
        self.scale_spinbox.setValue(1.00)
        self.scale_spinbox.setSingleStep(0.01)
        self.scale_spinbox.setDecimals(3)
        self.scale_spinbox.valueChanged.connect(self.update_scale_from_spinbox)
        spinbox_layout.addWidget(self.scale_spinbox)
        
        spinbox_layout.addStretch()
        control_layout.addLayout(spinbox_layout)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        preset_buttons = [
            ("10%", 0.1),
            ("25%", 0.25),
            ("50%", 0.5),
            ("75%", 0.75),
            ("100%", 1.0),
            ("125%", 1.25),
            ("150%", 1.5)
        ]
        
        for label, value in preset_buttons:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, v=value: self.set_scale_value(v))
            preset_layout.addWidget(btn)
            
        preset_layout.addStretch()
        control_layout.addLayout(preset_layout)
        
        control_group.setLayout(control_layout)
        scaling_layout.addWidget(control_group)
        
        # Visual feedback
        feedback_group = QGroupBox("Visual Feedback")
        feedback_layout = QVBoxLayout()
        
        # Progress bar showing scale
        self.scale_progress = QProgressBar()
        self.scale_progress.setMinimum(0)
        self.scale_progress.setMaximum(300)
        self.scale_progress.setValue(100)
        self.scale_progress.setFormat("%v%")
        self.scale_progress.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-size: 16px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            """
        )
        feedback_layout.addWidget(self.scale_progress)
        
        # Description label
        self.scale_description = QLabel("Normal teleoperation sensitivity")
        self.scale_description.setAlignment(Qt.AlignCenter)
        self.scale_description.setStyleSheet("font-size: 14px; padding: 5px;")
        feedback_layout.addWidget(self.scale_description)
        
        feedback_group.setLayout(feedback_layout)
        scaling_layout.addWidget(feedback_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        self.publish_continuously_cb = QCheckBox("Publish continuously")
        self.publish_continuously_cb.setChecked(True)
        self.publish_continuously_cb.setToolTip(
            "When checked, publishes the scale value continuously. "
            "When unchecked, only publishes when the value changes."
        )
        options_layout.addWidget(self.publish_continuously_cb)
        
        options_group.setLayout(options_layout)
        scaling_layout.addWidget(options_group)
        
        # Notes
        notes_label = QLabel(
            "<i>Note: Motion scaling affects the sensitivity of teleoperation. "
            "Values less than 1.0 reduce movement, values greater than 1.0 amplify movement. "
            "The scale value is published to '/ambf/teleop/scale' topic.</i>"
        )
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet("color: #666; padding: 10px;")
        scaling_layout.addWidget(notes_label)
        
        scaling_layout.addStretch()
        
        # Initialize ROS publisher
        self.init_scale_publisher()
        
        # Start publishing timer
        self.scale_publish_timer = QTimer()
        self.scale_publish_timer.timeout.connect(self.publish_scale_value)
        self.scale_publish_timer.start(100)  # Publish at 10 Hz
        
        self.tabs.addTab(scaling_widget, "Motion Scaling")

    def generate_collection_metadata(self):
        """Generate and save metadata for the current data collection"""
        metadata = {
            # Trial configuration
            'participant_id': self.current_config['participant_id'],
            'trial_number': self.current_config['trial_number'],
            'video_fps': self.current_config['video_fps'],
            'bag_name': self.current_config['bag_name'],
            
            # Timestamp
            'collection_start_time': self.collection_start_time.isoformat(),
            'collection_start_time_unix': self.collection_start_time.timestamp(),
            
            # Motion scaling
            'motion_scale': self.scale_value,
            
            # Trajectory configuration
            'trajectory': {
                'pattern': self.pattern_combo.currentText(),
                'radius': self.radius_spin.value(),
                'length': self.length_spin.value(),
                'position': {
                    'x': self.pos_x_spin.value(),
                    'y': self.pos_y_spin.value(),
                    'z': self.pos_z_spin.value()
                },
                'yaw': self.yaw_spin.value()
            },
            
            # System info
            'system': {
                'gui_version': '1.0',  # You can update this as needed
                'data_directory': os.path.expanduser('~/intel_motion_scaling/data')
            }
        }
        
        # Construct the expected folder path
        bag_name = self.current_config['bag_name']

        data_dir = os.path.expanduser(f"~/intel_motion_scaling/data/{bag_name}")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(data_dir, "collection_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.append_output("System", f"Metadata saved to: {metadata_path}")
        
        return metadata_path

    def init_scale_publisher(self):
        """Initialize ROS publisher for scale value"""
        try:
            import rospy
            from std_msgs.msg import Float32
            
            # Initialize ROS node if not already initialized
            if not rospy.core.is_initialized():
                rospy.init_node('dvrk_gui_scale_publisher', anonymous=True)
                
            self.scale_publisher = rospy.Publisher(
                '/ambf/teleop/scale', 
                Float32, 
                queue_size=1,
                latch=True  # Latch to ensure the value persists
            )
            self.scale_value = 1.0
            self.append_output("System", "Initialized motion scaling publisher")
        except ImportError:
            self.scale_publisher = None
            self.append_output("System", "ROS not available - motion scaling disabled")
            QMessageBox.warning(
                self, "ROS Not Available",
                "ROS is not available. Motion scaling functionality will be disabled."
            )

    def update_scale_from_slider(self, value):
        """Update scale from slider movement"""
        scale = value / 100.0
        self.scale_spinbox.blockSignals(True)
        self.scale_spinbox.setValue(scale)
        self.scale_spinbox.blockSignals(False)
        self.update_scale_value(scale)

    def update_scale_from_spinbox(self, value):
        """Update scale from spinbox"""
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(value * 100))
        self.scale_slider.blockSignals(False)
        self.update_scale_value(value)

    def set_scale_value(self, value):
        """Set scale to a specific value"""
        self.scale_spinbox.setValue(value)
        # This will trigger update_scale_from_spinbox

    def update_scale_value(self, value):
        """Update all scale displays and publish if changed"""
        self.scale_value = value
        
        # Update display
        self.current_scale_display.setText(f"Current Scale: {value:.3f}")
        
        # Update progress bar
        self.scale_progress.setValue(int(value * 100))
        
        # Update color based on scale
        if value < 0.5:
            color = "#ff6b6b"  # Red for low scale
            self.scale_progress.setStyleSheet(
                self.scale_progress.styleSheet().replace("#4CAF50", color)
            )
        elif value > 1.2:
            color = "#3498db"  # Blue for high scale
            self.scale_progress.setStyleSheet(
                self.scale_progress.styleSheet().replace("#4CAF50", color)
            )
        else:
            color = "#4CAF50"  # Green for normal range
            self.scale_progress.setStyleSheet(
                self.scale_progress.styleSheet().replace("#ff6b6b", color).replace("#3498db", color)
            )
        
        # Update description
        if value < 0.3:
            desc = "Very low sensitivity - fine control"
        elif value < 0.7:
            desc = "Reduced sensitivity"
        elif value < 1.1:
            desc = "Normal teleoperation sensitivity"
        elif value < 1.3:
            desc = "Increased sensitivity"
        else:
            desc = "High sensitivity - amplified movement"
        self.scale_description.setText(desc)
        
        # Update collection tab display
        if hasattr(self, 'collection_scale_label'):
            self.collection_scale_label.setText(f"Motion Scale: {value:.2f}")
        
        # Publish immediately if not publishing continuously
        if not self.publish_continuously_cb.isChecked():
            self.publish_scale_value()
            
        # Update trial configuration
        self.update_trial_config_with_scaling()

        # Update bag name if we have a current config
        if self.current_config and self.current_config.get('participant_id'):
            participant = self.current_config['participant_id']
            trial = self.current_config['trial_number']
            pattern = self.pattern_combo.currentText()
            
            scale_str = f"_scale{value:.2f}".replace(".", "") if abs(value - 1.0) > 0.01 else ""
            folder_name = f"trial{trial}_{pattern}{scale_str}"
            
            self.current_config['bag_name'] = f"{participant}/{folder_name}"
            self.update_trial_info()

    def publish_scale_value(self):
        """Publish the current scale value to ROS"""
        if self.scale_publisher is not None and self.publish_continuously_cb.isChecked():
            try:
                from std_msgs.msg import Float32
                msg = Float32()
                msg.data = self.scale_value
                self.scale_publisher.publish(msg)
            except Exception as e:
                pass  # Silently fail to avoid spamming console

    def update_trial_config_with_scaling(self):
        """Update trial configuration to include scaling information"""
        if not self.current_config:
            return
            
        self.current_config['motion_scale'] = self.scale_value
        
        # Update trial info display
        self.update_trial_info()

    def adjust_scale(self, delta):
        """Adjust scale by delta amount"""
        new_value = max(0.01, min(1.5, self.scale_value + delta))
        self.set_scale_value(new_value)

    def create_history_tab(self):
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        # Trial history list
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.load_trial_from_history)
        history_layout.addWidget(QLabel("Trial History:"))
        history_layout.addWidget(self.history_list)
        
        # History controls
        history_controls = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_history)
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self.clear_history)
        export_btn = QPushButton("Export History")
        export_btn.clicked.connect(self.export_history)
        
        history_controls.addWidget(refresh_btn)
        history_controls.addWidget(clear_btn)
        history_controls.addWidget(export_btn)
        history_controls.addStretch()
        history_layout.addLayout(history_controls)
        
        self.tabs.addTab(history_widget, "History")
        
    def create_output_console(self, parent):
        console_group = QGroupBox("Output Console")
        console_layout = QVBoxLayout()
        
        # Use the pre-created output console
        console_layout.addWidget(self.output_console)
        
        # Console controls
        console_controls = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.output_console.clear)
        save_btn = QPushButton("Save Log")
        save_btn.clicked.connect(self.save_log)
        
        console_controls.addWidget(clear_btn)
        console_controls.addWidget(save_btn)
        console_controls.addStretch()
        console_layout.addLayout(console_controls)
        
        console_group.setLayout(console_layout)
        parent.addWidget(console_group)
        
    def prompt_trial_configuration(self):
        """Prompt user to configure trial on startup"""
        reply = QMessageBox.information(
            self,
            "Configure Trial",
            "Welcome to dVRK User Study Interface!\n\n"
            "Please configure your trial settings before starting data collection.",
            QMessageBox.Ok | QMessageBox.Cancel
        )
        if reply == QMessageBox.Ok:
            self.configure_trial()
    
    def configure_trial(self):
        """Open trial configuration dialog"""
        try:
            dialog = TrialConfigDialog(self, self.trial_history)
            
            # Pre-fill with last configuration if available
            if self.current_config:
                dialog.participant_id.setText(self.current_config.get('participant_id', ''))
                dialog.trial_number.setValue(self.current_config.get('trial_number', 1))
                dialog.video_fps.setValue(self.current_config.get('video_fps', 15.0))
                dialog.bag_name.setText(self.current_config.get('bag_name', ''))
                dialog.auto_increment.setChecked(False)
            
            if dialog.exec_():
                self.current_config = dialog.get_config()
                self.update_trial_info()
                self.append_output("System", f"Trial configured: {self.current_config}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open configuration dialog:\n{str(e)}")
            self.append_output("System", f"ERROR: {str(e)}")
            
    def update_trial_info(self):
        if self.current_config:
            info = f"Participant: {self.current_config['participant_id']} | "
            info += f"Trial: #{self.current_config['trial_number']} | "
            info += f"Pattern: {self.pattern_combo.currentText()} | "
            info += f"Scale: {self.scale_value:.2f}"
            self.trial_info_label.setText(info)
            self.trial_info_label.setStyleSheet(
                "QLabel { background-color: #2e7d32; padding: 5px; border-radius: 3px; font-weight: bold; color: white; }"
            )
            
            # Update window title
            self.setWindowTitle(
                f"dVRK User Study Interface - {self.current_config['participant_id']} - "
                f"Trial #{self.current_config['trial_number']}"  # Removed trial_type
            )
            
            # Update the Configure Trial button to show it's configured
            self.config_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    font-weight: bold;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
                """
            )
            self.config_btn.setText("⚙️ Reconfigure Trial")
            
            # Hide collection reminder
            if hasattr(self, 'collection_config_reminder'):
                self.collection_config_reminder.setVisible(False)
                
            # Update tab appearance
            if hasattr(self, 'collection_tab_index'):
                self.tabs.setTabText(self.collection_tab_index, "✅ Data Collection")
        else:
            self.trial_info_label.setText("❌ No trial configured - Click 'Configure Trial' to begin")
            self.trial_info_label.setStyleSheet(
                "QLabel { background-color: #ffcccc; padding: 5px; border-radius: 3px; font-weight: bold; }"
            )
            
            # Reset window title
            self.setWindowTitle("dVRK User Study Interface - Intel Motion Scaling")
            
            # Show collection reminder
            if hasattr(self, 'collection_config_reminder'):
                self.collection_config_reminder.setVisible(True)
                
            # Update tab appearance
            if hasattr(self, 'collection_tab_index'):
                self.tabs.setTabText(self.collection_tab_index, "⚠️ Data Collection")
                
    def quick_configure_trial(self):
        """Quick configuration with default settings"""
        trial_num = len([t for t in self.trial_history if t.get('config', {}).get('participant_id') == 'test']) + 1
        self.current_config = {
            'participant_id': 'test',
            'trial_number': trial_num,
            'video_fps': 15.0,
            'bag_name': "test"
        }

        self.update_trial_info()
        self.append_output("System", f"Quick trial configured: {self.current_config}")
    
    def start_data_collection(self):
        if not self.current_config:
            reply = QMessageBox.question(
                self, "No Trial Configured", 
                "No trial has been configured.\n\n"
                "Would you like to:\n"
                "- Yes: Configure trial now\n"
                "- No: Use quick configuration (test participant)\n"
                "- Cancel: Cancel data collection",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.configure_trial()
                return  # User will need to click Start again after configuring
            elif reply == QMessageBox.No:
                self.quick_configure_trial()
            else:
                return  # Cancel
            
        if not self.trajectory_runner.is_running:
            QMessageBox.warning(self, "Warning", "Please start trajectory generation first!")
            return
            
        # Update tracking runner with bag name argument, pattern, scale, and trial number
        bag_name = self.current_config['bag_name']  # This is just participant ID now
        pattern = self.pattern_combo.currentText()
        scale = self.scale_value
        trial_num = self.current_config['trial_number']
        
        # Pass additional parameters to find_closest_point.py
        new_command = f"python find_closest_point.py {bag_name} --gui-mode --pattern {pattern} --scale {scale} --trial {trial_num}"
        self.tracking_runner.update_command(new_command)
        
        # Start collection
        self.tracking_runner.start_process()
        self.collection_start_time = datetime.now()
        self.collection_timer.start(1000)  # Update every second

        # Generate and save metadata
        try:
            metadata_path = self.generate_collection_metadata()
        except Exception as e:
            self.append_output("System", f"Warning: Failed to save metadata: {str(e)}")
        
        self.collection_in_progress = True
        self.collection_status_label.setText("Collection Status: Recording")
        self.collection_status_label.setStyleSheet(
            "QLabel { background-color: #388e3c; padding: 5px; font-weight: bold; color: white; }"
        )
        
        self.start_collection_btn.setEnabled(False)
        self.stop_collection_btn.setEnabled(True)
        
        # Add to history
        self.trial_history.append({
            'config': self.current_config.copy(),
            'start_time': self.collection_start_time,
            'end_time': None
        })
        
        self.append_output("System", f"Started data collection with command: {new_command}")

    def stop_data_collection(self):
        # Stop the tracking process
        self.tracking_runner.stop_process()
        self.collection_timer.stop()
        
        # Store the bag path for potential analysis
        bag_path = None
        if self.current_config:
            # Construct the bag path based on how find_closest_point.py saves it
            # Note: find_closest_point.py uses a hardcoded path /home/kyu17/...
            # We need to check both the hardcoded path and the home directory path
            
            bag_name = self.current_config['bag_name']
            
            # Try the hardcoded path first (as used in find_closest_point.py)
            hardcoded_dir = f"/home/kyu17/intel_motion_scaling/data/{bag_name}"
            bag_file = os.path.join(hardcoded_dir, "dist_video_clutch.bag")
            
            if os.path.exists(bag_file):
                bag_path = bag_file
            else:
                # Try the home directory path as fallback
                home_dir = os.path.expanduser(f"~/intel_motion_scaling/data/{bag_name}")
                bag_file = os.path.join(home_dir, "dist_video_clutch.bag")
                if os.path.exists(bag_file):
                    bag_path = bag_file
        
        if self.trial_history:
            self.trial_history[-1]['end_time'] = datetime.now()
            self.trial_history[-1]['bag_path'] = bag_path
            
        self.collection_in_progress = False
        self.collection_status_label.setText("Collection Status: Idle")
        self.collection_status_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 5px; font-weight: bold; }"
        )
        
        self.start_collection_btn.setEnabled(True)
        self.stop_collection_btn.setEnabled(False)
        
        self.refresh_history()
        self.append_output("System", "Data collection stopped")
        
        # Ask if user wants to analyze the data immediately
        if bag_path and os.path.exists(bag_path):
            reply = QMessageBox.question(
                self, "Run Analysis?",
                f"Data collection complete!\n\n"
                f"Would you like to analyze the recorded data now?\n\n"
                f"Bag file: {os.path.basename(os.path.dirname(bag_path))}/dist_video_clutch.bag",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Switch to analysis tab
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i).endswith("Analysis"):
                        self.tabs.setCurrentIndex(i)
                        break
                
                # Set the bag file path
                self.bag_file_path.setText(bag_path)
                
                # Give UI time to update
                QTimer.singleShot(100, self.run_analysis)
        
    def clear_trajectory(self):
        """Clear the trajectory by publishing an empty point cloud"""
        # Stop any running trajectory generator first
        if self.trajectory_runner.is_running:
            self.trajectory_runner.stop_process()
            # Wait a bit for process to stop
            QTimer.singleShot(500, self._publish_empty_trajectory)
        else:
            self._publish_empty_trajectory()
    
    def reset_trajectory_initialization(self):
        """Reset the first-run flag for trajectory initialization"""
        self.trajectory_runner.is_first_run = True
        self.trajectory_runner.initializing = False
        self.append_output("System", "Trajectory initialization reset. Next start will reinitialize.")
        
        # Update status label
        self.trajectory_init_label.setText("Trajectory System: Not initialized")
        self.trajectory_init_label.setStyleSheet(
            "QLabel { background-color: #ffcccc; padding: 5px; border-radius: 3px; }"
        )
        
        # If trajectory is running, stop it
        if self.trajectory_runner.is_running:
            self.trajectory_runner.stop_process()
            
    def _publish_empty_trajectory(self):
        """Helper to publish empty trajectory"""
        # Use a temporary process to publish empty trajectory
        clear_command = "python trajectory_generator.py --pattern clear"
        clear_process = QProcess()
        clear_process.setWorkingDirectory(os.path.expanduser("~/intel_motion_scaling/scripts"))
        clear_process.start('bash', ['-c', clear_command])
        
        # Let it run briefly then stop
        QTimer.singleShot(1000, lambda: clear_process.terminate())
        self.append_output("System", "Trajectory cleared")
        
    def update_collection_time(self):
        if self.collection_start_time:
            elapsed = datetime.now() - self.collection_start_time
            minutes = int(elapsed.total_seconds() // 60)
            seconds = int(elapsed.total_seconds() % 60)
            self.timer_label.setText(f"Collection Time: {minutes:02d}:{seconds:02d}")
            
    def browse_path(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)
            
    def update_dvrk_path(self):
        """Update the dVRK console working directory when path changes"""
        self.console_runner.working_dir = self.dvrk_path.text()
        
    def update_trajectory_command(self):
        """Update the trajectory generator command based on current parameters"""
        pattern = self.pattern_combo.currentText()
        
        cmd_parts = [
            "python", "trajectory_generator.py",
            "--pattern", pattern,
            "--radius", str(self.radius_spin.value()),
            "--length", str(self.length_spin.value()),
            "--pos_x", str(self.pos_x_spin.value()),
            "--pos_y", str(self.pos_y_spin.value()),
            "--pos_z", str(self.pos_z_spin.value()),
            "--yaw", str(self.yaw_spin.value())
        ]
        
        new_command = " ".join(cmd_parts)
        self.trajectory_runner.update_command(new_command)

    def start_overlay_animation(self):
        """Create a pulsing animation for the overlay"""
        if not hasattr(self, 'overlay_timer'):
            self.overlay_timer = QTimer()
            self.overlay_opacity = 0.9
            self.overlay_direction = -0.05
            
        def update_opacity():
            self.overlay_opacity += self.overlay_direction
            if self.overlay_opacity <= 0.5 or self.overlay_opacity >= 0.9:
                self.overlay_direction *= -1
            
            if hasattr(self, 'trajectory_overlay'):
                self.trajectory_overlay.setStyleSheet(f"""
                    QLabel {{
                        background-color: rgba(255, 152, 0, {self.overlay_opacity});
                        color: white;
                        font-weight: bold;
                        font-size: 14px;
                        padding: 10px;
                        border-radius: 5px;
                        border: 2px solid #ff6b6b;
                    }}
                """)
        
        self.overlay_timer.timeout.connect(update_opacity)
        self.overlay_timer.start(100)  # Update every 100ms
    
    def set_trajectory_controls_enabled(self, enabled):
        """Enable or disable all trajectory configuration controls"""
        # Pattern selection
        self.pattern_combo.setEnabled(enabled)
        
        # All parameter spinboxes
        self.radius_spin.setEnabled(enabled)
        self.length_spin.setEnabled(enabled)
        self.pos_x_spin.setEnabled(enabled)
        self.pos_y_spin.setEnabled(enabled)
        self.pos_z_spin.setEnabled(enabled)
        self.yaw_spin.setEnabled(enabled)
        
        # Preset buttons
        if hasattr(self, 'horizontal_btn'):
            self.horizontal_btn.setEnabled(enabled)
        if hasattr(self, 'vertical_btn'):
            self.vertical_btn.setEnabled(enabled)
        if hasattr(self, 'diagonal_btn'):
            self.diagonal_btn.setEnabled(enabled)
        if hasattr(self, 'clear_trajectory_btn'):
            self.clear_trajectory_btn.setEnabled(enabled)
        
        # Show/hide overlay
        if hasattr(self, 'trajectory_overlay'):
            if not enabled:
                # Position and show the overlay
                trajectory_group = self.pattern_combo.parent()
                self.trajectory_overlay.resize(trajectory_group.width() - 20, 40)
                self.trajectory_overlay.move(10, 30)
                self.trajectory_overlay.raise_()
                self.trajectory_overlay.show()
                
                # Animate the overlay (pulsing effect)
                self.start_overlay_animation()
            else:
                self.trajectory_overlay.hide()
                if hasattr(self, 'overlay_timer'):
                    self.overlay_timer.stop()
        
        # Apply custom styling to disabled controls
        style_disabled = """
            QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #ffebee;
                color: #999;
                border: 2px solid #ff6b6b;
            }
            QPushButton:disabled {
                background-color: #ffebee;
                color: #999;
                border: 2px solid #ff6b6b;
            }
        """
        
        style_enabled = """
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: default;
                color: default;
                border: default;
            }
            QPushButton {
                background-color: default;
                color: default;
                border: default;
            }
        """
        
        # Apply styling to the trajectory group
        trajectory_group = self.pattern_combo.parent()
        if not enabled:
            trajectory_group.setStyleSheet(trajectory_group.styleSheet() + style_disabled)
        else:
            # Reset to default styling
            trajectory_group.setStyleSheet("")
        
        # Update visual feedback
        if not enabled:
            self.append_output("System", "Trajectory controls locked while running")
        else:
            self.append_output("System", "Trajectory controls unlocked")
    
    def apply_preset(self, preset_name):
        """Apply a preset orientation configuration"""
        if preset_name == 'horizontal':
            self.yaw_spin.setValue(0.0)
        elif preset_name == 'vertical_yz':
            self.yaw_spin.setValue(90.0)
        elif preset_name == 'diagonal':
            self.yaw_spin.setValue(45.0)
        
        self.update_trajectory_command()
            
    def browse_bag_file(self):
        """Browse for bag file with smart default directory"""
        # Start in the data directory if it exists
        default_dir = os.path.expanduser("~/intel_motion_scaling/data")
        if not os.path.exists(default_dir):
            default_dir = os.path.expanduser("~")
        
        # If we have a current config, try to start in that participant's folder
        if self.current_config and self.current_config.get('participant_id'):
            participant_dir = os.path.join(default_dir, self.current_config['participant_id'])
            if os.path.exists(participant_dir):
                default_dir = participant_dir
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Bag File", 
            default_dir,
            "ROS Bag Files (*.bag);;All Files (*.*)"
        )
        if path:
            self.bag_file_path.setText(path)
            self.append_output("System", f"Selected bag file: {path}")
            
            # Auto-detect FPS from filename or path if possible
            if "15fps" in path.lower():
                self.current_config = self.current_config or {}
                self.current_config['video_fps'] = 15.0
            elif "30fps" in path.lower():
                self.current_config = self.current_config or {}
                self.current_config['video_fps'] = 30.0

    def find_recent_bag_files(self, limit=5):
        """Find the most recent bag files in the data directory"""
        data_dir = os.path.expanduser("~/intel_motion_scaling/data")
        if not os.path.exists(data_dir):
            return []
        
        bag_files = []
        
        # Walk through the directory tree to find bag files
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.bag'):
                    full_path = os.path.join(root, file)
                    try:
                        # Get file modification time
                        mtime = os.path.getmtime(full_path)
                        bag_files.append((full_path, mtime))
                    except:
                        pass
        
        # Sort by modification time (newest first)
        bag_files.sort(key=lambda x: x[1], reverse=True)
        
        # Return the paths of the most recent files
        return [path for path, _ in bag_files[:limit]]

    def set_last_bag_file(self):
        """Set the most recently created bag file as the analysis target"""
        recent_files = self.find_recent_bag_files(limit=1)
        if recent_files:
            self.bag_file_path.setText(recent_files[0])
            self.append_output("System", f"Auto-selected most recent bag: {recent_files[0]}")
            
            # Show info message
            QMessageBox.information(
                self, "Recent Bag Selected",
                f"Selected the most recent bag file:\n{os.path.basename(recent_files[0])}\n\n"
                f"Full path: {recent_files[0]}"
            )
        else:
            QMessageBox.warning(
                self, "No Bag Files Found",
                "No bag files found in the data directory.\n"
                "Please record some data first or browse to select a bag file."
            )
            
    def run_analysis(self):
        """Run analysis with direct execution (no separate start button needed)"""
        if not self.bag_file_path.text():
            QMessageBox.warning(self, "Warning", "Please select a bag file first!")
            return
        
        # Check if analysis is already running
        if self.analysis_runner.is_running:
            QMessageBox.information(
                self, "Analysis Running", 
                "Analysis is already in progress. Please wait for it to complete or stop it first."
            )
            return
            
        # Build analysis command
        cmd_parts = ["python", "analysis.py", "-b", self.bag_file_path.text()]

        if self.extract_videos_cb.isChecked():
            cmd_parts.append("-v")
        if self.extract_distances_cb.isChecked():
            cmd_parts.append("-d")
        if self.extract_clutch_cb.isChecked():
            cmd_parts.append("-c")
        if self.generate_plots_cb.isChecked():
            cmd_parts.append("-p")
            
        cmd_parts.extend(["--fps", str(self.current_config.get('video_fps', 15.0) if self.current_config else 15.0)])
        
        new_command = " ".join(cmd_parts)
        
        # Update the command and start the process immediately
        self.analysis_runner.update_command(new_command)
        self.analysis_runner.start_process()
        
        # Log the analysis start
        self.append_output("System", f"Starting analysis of: {self.bag_file_path.text()}")

    def stop_analysis(self):
        """Stop the running analysis"""
        if self.analysis_runner.is_running:
            self.analysis_runner.stop_process()
            self.append_output("System", "Analysis stopped by user")
        
    def on_analysis_started(self, process_name):
        """Handle analysis start event"""
        self.analysis_status_label.setText("Status: Running Analysis...")
        self.analysis_status_label.setStyleSheet(
            "QLabel { background-color: #90EE90; padding: 8px; font-weight: bold; border-radius: 3px; }"
        )
        
        # Update button states
        self.run_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setVisible(True)
        
        # Show progress label
        self.analysis_progress_label.setText("Processing... Check output console for details.")
        self.analysis_progress_label.setVisible(True)
        
        # Switch to analysis tab if not already there
        analysis_tab_index = self.tabs.indexOf(self.tabs.findChild(QWidget, "analysis_widget"))
        if analysis_tab_index >= 0 and self.tabs.currentIndex() != analysis_tab_index:
            # Flash the tab to indicate activity
            self.tabs.setCurrentIndex(analysis_tab_index)

    def on_analysis_finished(self, process_name, exit_code):
        """Handle analysis completion event"""
        if exit_code == 0:
            self.analysis_status_label.setText("Status: Analysis Completed Successfully ✓")
            self.analysis_status_label.setStyleSheet(
                "QLabel { background-color: #2e7d32; padding: 8px; font-weight: bold; border-radius: 3px; color: white; }"
            )
            self.analysis_progress_label.setText("Analysis complete! Check the output folder for results.")
            
            # Show success message
            QMessageBox.information(
                self, "Analysis Complete", 
                "Analysis completed successfully!\n\n"
                "Results have been saved to the analysis folder.\n"
                "Check the output console for file locations."
            )
        else:
            self.analysis_status_label.setText(f"Status: Analysis Failed (Exit code: {exit_code})")
            self.analysis_status_label.setStyleSheet(
                "QLabel { background-color: #ffcccc; padding: 8px; font-weight: bold; border-radius: 3px; }"
            )
            self.analysis_progress_label.setText("Analysis failed. Check the output console for errors.")
            
            # Show error message
            QMessageBox.warning(
                self, "Analysis Failed", 
                f"Analysis failed with exit code: {exit_code}\n\n"
                "Please check the output console for error details."
            )
        
        # Reset button states
        self.run_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setVisible(False)
        
        # Keep progress label visible for reference
        QTimer.singleShot(10000, lambda: self.analysis_progress_label.setVisible(False))  # Hide after 10 seconds
        
    def refresh_history(self):
        self.history_list.clear()
        for trial in self.trial_history:
            config = trial.get('config', {})
            start = trial['start_time'].strftime("%Y-%m-%d %H:%M:%S") if trial.get('start_time') else "Unknown"
            item_text = f"{config.get('participant_id', 'Unknown')} - Trial #{config.get('trial_number', 0)} - {start}"
            
            # Add scale info if present
            if 'motion_scale' in config and config['motion_scale'] != 1.0:
                item_text += f" - Scale: {config['motion_scale']:.2f}"
                
            self.history_list.addItem(item_text)
            
    def load_trial_from_history(self, item):
        index = self.history_list.row(item)
        if 0 <= index < len(self.trial_history):
            self.current_config = self.trial_history[index].get('config', {}).copy()
            self.update_trial_info()
            
            # Restore scale value if present
            if 'motion_scale' in self.current_config:
                self.scale_value = self.current_config['motion_scale']
                if hasattr(self, 'scale_spinbox'):
                    self.scale_spinbox.setValue(self.scale_value)
            
    def clear_history(self):
        reply = QMessageBox.question(
            self, "Clear History", 
            "Are you sure you want to clear the trial history?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.trial_history.clear()
            self.refresh_history()
            
    def export_history(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export History", 
            f"trial_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        if path:
            history_data = []
            for trial in self.trial_history:
                data = trial.get('config', {}).copy()
                data['start_time'] = trial.get('start_time').isoformat() if trial.get('start_time') else None
                data['end_time'] = trial.get('end_time').isoformat() if trial.get('end_time') else None
                history_data.append(data)
                
            with open(path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            self.append_output("System", f"History exported to: {path}")
            
    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", 
            f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        if path:
            with open(path, 'w') as f:
                f.write(self.output_console.toPlainText())
            self.append_output("System", f"Log saved to: {path}")
            
    def emergency_stop(self):
        reply = QMessageBox.question(
            self, "Emergency Stop", 
            "Stop all processes?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Clear trajectory first if collection is in progress
            if self.collection_in_progress:
                self.clear_trajectory()
                
            # Stop all running processes
            for runner in [
                self.console_runner,
                self.env_runner,
                self.teleop_runner,
                self.trajectory_runner,
                self.tracking_runner,
                self.analysis_runner
            ]:
                if runner.is_running:
                    runner.stop_process()
                    
            self.append_output("System", "EMERGENCY STOP - All processes terminated")
            
    def update_status(self, message):
        self.status_bar.showMessage(message)
        
    @pyqtSlot(str, str)
    def update_trajectory_init_status(self, status_text, style):
        """Update the trajectory initialization status label"""
        self.trajectory_init_label.setText(status_text)
        self.trajectory_init_label.setStyleSheet(style)
    
    @pyqtSlot(str, str)
    def append_output(self, process_name, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {process_name}: {text}"
        self.output_console.append(formatted_text)
        
        # Auto-scroll to bottom
        cursor = self.output_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_console.setTextCursor(cursor)
        
    def save_settings(self):
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        self.settings.setValue('dvrk_path', self.dvrk_path.text())
        
        # Save scale value
        if hasattr(self, 'scale_value'):
            self.settings.setValue('motion_scale', self.scale_value)
        
        # Save current configuration
        if self.current_config:
            self.settings.setValue('last_config', json.dumps(self.current_config))
        
        # Save trial history
        if self.trial_history:
            history_data = []
            for trial in self.trial_history[:50]:  # Save last 50 trials
                data = {
                    'config': trial.get('config', {}),
                    'start_time': trial.get('start_time').isoformat() if trial.get('start_time') else None,
                    'end_time': trial.get('end_time').isoformat() if trial.get('end_time') else None
                }
                history_data.append(data)
            self.settings.setValue('trial_history', json.dumps(history_data))
        
    def load_settings(self):
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
            
        dvrk_path = self.settings.value('dvrk_path')
        if dvrk_path:
            self.dvrk_path.setText(dvrk_path)
            self.console_runner.working_dir = dvrk_path
            
        # Load scale value
        scale = self.settings.value('motion_scale')
        if scale and hasattr(self, 'scale_value'):
            try:
                self.scale_value = float(scale)
                if hasattr(self, 'scale_spinbox'):
                    self.scale_spinbox.setValue(self.scale_value)
            except:
                pass
            
        # Load last configuration
        last_config = self.settings.value('last_config')
        if last_config:
            try:
                self.current_config = json.loads(last_config)
                self.update_trial_info()
                self.append_output("System", "Restored last trial configuration")
            except:
                pass
        
        # Load trial history
        trial_history = self.settings.value('trial_history')
        if trial_history:
            try:
                history_data = json.loads(trial_history)
                for data in history_data:
                    trial = {
                        'config': data.get('config', {}),
                        'start_time': datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
                        'end_time': datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
                    }
                    self.trial_history.append(trial)
                self.refresh_history()
                self.append_output("System", f"Restored {len(self.trial_history)} trials from history")
            except:
                pass
            
    def closeEvent(self, event):
        self.save_settings()
        
        # Stop publishing timer
        if hasattr(self, 'scale_publish_timer') and self.scale_publish_timer:
            self.scale_publish_timer.stop()
        
        # Clear trajectory if needed
        if self.trajectory_runner.is_running:
            self.clear_trajectory()
        
        # Check for running processes
        running_processes = []
        for name, runner in [
            ("dVRK Console", self.console_runner),
            ("AMBF Environment", self.env_runner),
            ("Teleoperation", self.teleop_runner),
            ("Trajectory Generator", self.trajectory_runner),
            ("Distance Tracking", self.tracking_runner),
            ("Analysis", self.analysis_runner)
        ]:
            if runner.is_running:
                running_processes.append(name)
                
        if running_processes:
            reply = QMessageBox.question(
                self, "Exit", 
                f"The following processes are still running: {', '.join(running_processes)}\n"
                "Do you want to stop them and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                for _, runner in [
                    ("dVRK Console", self.console_runner),
                    ("AMBF Environment", self.env_runner),
                    ("Teleoperation", self.teleop_runner),
                    ("Trajectory Generator", self.trajectory_runner),
                    ("Distance Tracking", self.tracking_runner),
                    ("Analysis", self.analysis_runner)
                ]:
                    if runner.is_running:
                        runner.stop_process()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark theme
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    # Create and show main window
    window = DVRKUserStudyGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()