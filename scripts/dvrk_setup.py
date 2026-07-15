"""
Standalone dVRK Setup GUI
Extracted from the main dVRK User Study GUI for independent setup management
"""

import sys
import os
import signal
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QLineEdit,
    QFileDialog, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QProcess, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QPalette, QColor, QTextCursor


class ProcessRunner(QWidget):
    """Widget for running and monitoring a single process"""
    
    process_started = pyqtSignal(str)
    process_finished = pyqtSignal(str, int)
    output_received = pyqtSignal(str, str)  # process_name, output
    
    def __init__(self, name, command, working_dir=None, parent=None):
        super().__init__(parent)
        self.name = name
        self.command = command
        self.working_dir = working_dir
        self.process = None
        self.is_running = False
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
            
        # Handle shell scripts vs python scripts vs ROS commands
        if self.command.startswith('./') or self.command.endswith('.sh'):
            self.process.start('bash', ['-c', self.command])
        elif 'source' in self.command or 'rosrun' in self.command or 'roslaunch' in self.command:
            self.process.start('bash', ['-c', self.command])
        else:
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


class DVRKSetupGUI(QMainWindow):
    """Standalone GUI for dVRK system setup"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('DVRK_Setup', 'SetupGUI')
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        self.setWindowTitle("dVRK Setup Manager")
        self.setGeometry(100, 100, 1200, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header
        self.create_header(main_layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Create setup controls
        setup_widget = QWidget()
        setup_layout = QVBoxLayout(setup_widget)
        
        # dVRK Console group
        self.create_console_group(setup_layout)
        
        # Simulation Environment group
        self.create_environment_group(setup_layout)
        
        # Quick Actions
        self.create_quick_actions(setup_layout)
        
        setup_layout.addStretch()
        splitter.addWidget(setup_widget)
        
        # Output console at bottom
        self.create_output_console(splitter)
        
        # Set splitter sizes
        splitter.setSizes([400, 200])
        
        # Status bar
        self.status_bar = self.statusBar()
        self.update_status("Ready")
        
    def create_header(self, layout):
        """Create header with title and emergency stop"""
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("dVRK System Setup")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(title_label)
        
        # System status
        self.system_status_label = QLabel("System Status: Not Started")
        self.system_status_label.setStyleSheet(
            "background-color: #f0f0f0; padding: 8px; border-radius: 4px; font-weight: bold;"
        )
        header_layout.addWidget(self.system_status_label)
        
        header_layout.addStretch()
        
        # Emergency stop button
        self.emergency_stop_btn = QPushButton("🛑 STOP ALL")
        self.emergency_stop_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #d32f2f;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            """
        )
        self.emergency_stop_btn.clicked.connect(self.emergency_stop)
        header_layout.addWidget(self.emergency_stop_btn)
        
        layout.addLayout(header_layout)
        
    def create_console_group(self, layout):
        """Create dVRK Console group"""
        console_group = QGroupBox("dVRK Console")
        console_layout = QVBoxLayout()
        
        # Path configuration
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("dVRK Config Path:"))
        self.dvrk_path = QLineEdit()
        self.dvrk_path.setText("~/catkin_ws/src/dvrk_config_jhu/jhu-daVinci")
        path_layout.addWidget(self.dvrk_path)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_path(self.dvrk_path))
        path_layout.addWidget(browse_btn)
        console_layout.addLayout(path_layout)
        
        # Console process runner
        self.console_runner = ProcessRunner(
            "dVRK Console",
            "source ~/catkin_ws/devel/setup.bash && rosrun dvrk_robot dvrk_console_json -j console-MTML-MTMR.json",
            "~/catkin_ws/src/dvrk_config_jhu/jhu-daVinci"
        )
        self.console_runner.output_received.connect(self.append_output)
        self.console_runner.process_started.connect(self.update_system_status)
        self.console_runner.process_finished.connect(self.update_system_status)
        console_layout.addWidget(self.console_runner)
        
        # Update console runner path when dVRK path changes
        self.dvrk_path.textChanged.connect(self.update_dvrk_path)
        
        # Instructions
        instructions = QLabel(
            "📝 Note: After console starts, manually click 'Power On' and 'Home' in the dVRK console window"
        )
        instructions.setStyleSheet("color: #ff9800; font-style: italic; padding: 5px;")
        console_layout.addWidget(instructions)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
    def create_environment_group(self, layout):
        """Create Simulation Environment group"""
        env_group = QGroupBox("Simulation Environment")
        env_layout = QVBoxLayout()
        
        # Environment runner
        self.env_runner = ProcessRunner(
            "AMBF Environment",
            "./run_env_3D_MED_COMPLEX_LND_420006.sh",
            "~/surgical_robotics_challenge"
        )
        self.env_runner.output_received.connect(self.append_output)
        self.env_runner.process_started.connect(self.update_system_status)
        self.env_runner.process_finished.connect(self.update_system_status)
        env_layout.addWidget(self.env_runner)
        
        # Teleoperation runner
        self.teleop_runner = ProcessRunner(
            "Teleoperation",
            "./mtm_psm_pair_teleop_420006.sh",
            "~/surgical_robotics_challenge/scripts/surgical_robotics_challenge/teleoperation"
        )
        self.teleop_runner.output_received.connect(self.append_output)
        self.teleop_runner.process_started.connect(self.update_system_status)
        self.teleop_runner.process_finished.connect(self.update_system_status)
        env_layout.addWidget(self.teleop_runner)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
    def create_quick_actions(self, layout):
        """Create quick action buttons"""
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout()
        
        # Start all button
        start_all_btn = QPushButton("▶️ Start All")
        start_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            """
        )
        start_all_btn.clicked.connect(self.start_all_processes)
        actions_layout.addWidget(start_all_btn)
        
        # Stop all button
        stop_all_btn = QPushButton("⏹️ Stop All")
        stop_all_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            """
        )
        stop_all_btn.clicked.connect(self.stop_all_processes)
        actions_layout.addWidget(stop_all_btn)
        
        # Refresh status
        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.clicked.connect(self.update_system_status)
        actions_layout.addWidget(refresh_btn)
        
        actions_layout.addStretch()
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
    def create_output_console(self, parent):
        """Create output console"""
        console_group = QGroupBox("Output Console")
        console_layout = QVBoxLayout()
        
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setFont(QFont("Consolas", 9))
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
        
    def browse_path(self, line_edit):
        """Browse for directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)
            
    def update_dvrk_path(self):
        """Update the dVRK console working directory when path changes"""
        self.console_runner.working_dir = self.dvrk_path.text()
        
    def start_all_processes(self):
        """Start all processes in sequence"""
        self.append_output("System", "Starting all processes...")
        
        # Start in order: Console -> Environment -> Teleoperation
        if not self.console_runner.is_running:
            self.console_runner.start_process()
            
        if not self.env_runner.is_running:
            self.env_runner.start_process()
            
        if not self.teleop_runner.is_running:
            self.teleop_runner.start_process()
            
    def stop_all_processes(self):
        """Stop all running processes"""
        self.append_output("System", "Stopping all processes...")
        
        for runner in [self.teleop_runner, self.env_runner, self.console_runner]:
            if runner.is_running:
                runner.stop_process()
                
    def emergency_stop(self):
        """Emergency stop all processes"""
        reply = QMessageBox.question(
            self, "Emergency Stop", 
            "Stop all processes immediately?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.stop_all_processes()
            self.append_output("System", "EMERGENCY STOP - All processes terminated")
            
    def update_system_status(self, *args):
        """Update the system status display"""
        running_count = 0
        for runner in [self.console_runner, self.env_runner, self.teleop_runner]:
            if runner.is_running:
                running_count += 1
                
        if running_count == 0:
            self.system_status_label.setText("System Status: Not Started")
            self.system_status_label.setStyleSheet(
                "background-color: #f0f0f0; padding: 8px; border-radius: 4px; font-weight: bold;"
            )
        elif running_count == 3:
            self.system_status_label.setText("System Status: All Running ✓")
            self.system_status_label.setStyleSheet(
                "background-color: #4CAF50; padding: 8px; border-radius: 4px; font-weight: bold; color: white;"
            )
        else:
            self.system_status_label.setText(f"System Status: Partial ({running_count}/3 running)")
            self.system_status_label.setStyleSheet(
                "background-color: #ff9800; padding: 8px; border-radius: 4px; font-weight: bold; color: white;"
            )
            
    def append_output(self, process_name, text):
        """Append text to output console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {process_name}: {text}"
        self.output_console.append(formatted_text)
        
        # Auto-scroll to bottom
        cursor = self.output_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_console.setTextCursor(cursor)
        
    def save_log(self):
        """Save console log to file"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", 
            f"setup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        if path:
            with open(path, 'w') as f:
                f.write(self.output_console.toPlainText())
            self.append_output("System", f"Log saved to: {path}")
            
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.showMessage(message)
        
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
        self.settings.setValue('dvrk_path', self.dvrk_path.text())
        
    def load_settings(self):
        """Load application settings"""
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
            
    def closeEvent(self, event):
        """Handle window close event"""
        self.save_settings()
        
        # Check for running processes
        running_processes = []
        for name, runner in [
            ("dVRK Console", self.console_runner),
            ("AMBF Environment", self.env_runner),
            ("Teleoperation", self.teleop_runner)
        ]:
            if runner.is_running:
                running_processes.append(name)
                
        if running_processes:
            reply = QMessageBox.question(
                self, "Exit", 
                f"The following processes are still running:\n{', '.join(running_processes)}\n\n"
                "Do you want to stop them and exit?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.stop_all_processes()
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()  # Exit without stopping
            else:
                event.ignore()  # Cancel exit
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
    window = DVRKSetupGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()