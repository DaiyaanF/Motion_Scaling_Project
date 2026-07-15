#!/usr/bin/env python3
import sys
import os
import signal
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt, QProcess, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QPalette, QColor, QTextCursor


class ProcessRunner(QWidget):
    process_started = pyqtSignal(str)
    process_finished = pyqtSignal(str, int)
    output_received = pyqtSignal(str, str)
    
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
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: gray; font-size: 20px;")
        self.status_indicator.setFixedWidth(30)
        
        self.name_label = QLabel(self.name)
        self.name_label.setMinimumWidth(200)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.name_label.setFont(font)
        
        self.control_btn = QPushButton("Start")
        self.control_btn.clicked.connect(self.toggle_process)
        self.control_btn.setFixedWidth(100)
        
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.name_label)
        layout.addWidget(self.control_btn)
        layout.addStretch()
        
        self.setLayout(layout)
        
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
        
        # Start process in new process group so we can kill all children
        self.process.start('setsid', ['bash', '-c', self.command])
        
        if self.process.waitForStarted():
            self.is_running = True
            self.status_indicator.setStyleSheet("color: green; font-size: 20px;")
            self.control_btn.setText("Stop")
            self.process_started.emit(self.name)
            self.output_received.emit(self.name, f"Started: {self.name}\n")
        else:
            self.output_received.emit(self.name, f"Failed to start: {self.name}\n")
            self.process = None
            
    def stop_process(self):
        if self.process is None:
            return
        
        # Kill entire process group (including GUI windows and child processes)
        try:
            pid = self.process.processId()
            # Kill the process group by using negative PID
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception as e:
            # Fallback: try regular kill
            try:
                self.process.terminate()
            except:
                pass
        
        # Wait for processes to terminate
        if not self.process.waitForFinished(3000):
            # Force kill if still running
            try:
                pid = self.process.processId()
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except:
                self.process.kill()
        
        # Additional cleanup: kill known process names as safety net
        if 'ambf' in self.name.lower():
            subprocess.run(['pkill', '-9', '-f', 'ambf_simulator'], stderr=subprocess.DEVNULL)
        elif 'teleop' in self.name.lower():
            subprocess.run(['pkill', '-9', '-f', 'mtm_psm_pair_teleop'], stderr=subprocess.DEVNULL)
        
        self.process = None
        self.is_running = False
        self.status_indicator.setStyleSheet("color: gray; font-size: 20px;")
        self.control_btn.setText("Start")
        self.output_received.emit(self.name, f"Stopped: {self.name}\n")
            
    def handle_stdout(self):
        if self.process:
            data = self.process.readAllStandardOutput()
            output = data.data().decode('utf-8', errors='replace')
            self.output_received.emit(self.name, output)
        
    def handle_stderr(self):
        if self.process:
            data = self.process.readAllStandardError()
            output = data.data().decode('utf-8', errors='replace')
            self.output_received.emit(self.name, f"[ERROR] {output}")
        
    def process_finished_handler(self, exit_code, exit_status):
        self.is_running = False
        self.status_indicator.setStyleSheet("color: gray; font-size: 20px;")
        self.control_btn.setText("Start")
        self.process_finished.emit(self.name, exit_code)
        self.output_received.emit(self.name, f"Process finished: exit {exit_code}\n")
        self.process = None


class SimpleDVRKGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("dVRK Process Manager")
        self.setGeometry(100, 100, 900, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        title = QLabel("dVRK & Simulation Process Manager")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        self.create_process_controls(main_layout)
        self.create_output_console(main_layout)
        
        self.status_bar = self.statusBar()
        self.update_status("Ready")
        
    def create_process_controls(self, layout):
        group = QGroupBox("System Processes")
        group_layout = QVBoxLayout()
        
        # Configure paths relative to current directory or use environment
        dvrk_config_dir = os.path.expanduser(os.getenv('DVRK_CONFIG_DIR', 
            '~/catkin_ws/src/dvrk/dvrk_config_jhu/jhu-daVinci'))
        surg_challenge_dir = os.path.expanduser(os.getenv('SURG_CHALLENGE_DIR',
            '~/surgical_robotics_challenge'))
        teleop_dir = os.path.join(surg_challenge_dir, 'scripts/surgical_robotics_challenge/teleoperation')
        
        self.console_runner = ProcessRunner(
            "dVRK Console",
            "source ~/catkin_ws/devel/setup.bash && rosrun dvrk_robot dvrk_console_json -j console-MTML-MTMR.json",
            dvrk_config_dir
        )
        self.console_runner.output_received.connect(self.append_output)
        group_layout.addWidget(self.console_runner)
        
        self.env_runner = ProcessRunner(
            "AMBF Simulation",
            "./run_env_3D_MED_COMPLEX_LND_420006.sh",
            surg_challenge_dir
        )
        self.env_runner.output_received.connect(self.append_output)
        group_layout.addWidget(self.env_runner)
        
        self.teleop_runner = ProcessRunner(
            "Teleoperation",
            "./mtm_psm_pair_teleop_420006.sh",
            teleop_dir
        )
        self.teleop_runner.output_received.connect(self.append_output)
        group_layout.addWidget(self.teleop_runner)
        
        start_all_layout = QHBoxLayout()
        self.start_all_btn = QPushButton("Start All (Sequential)")
        self.start_all_btn.clicked.connect(self.start_all_sequential)
        start_all_layout.addWidget(self.start_all_btn)
        
        self.stop_all_btn = QPushButton("Stop All")
        self.stop_all_btn.clicked.connect(self.stop_all_processes)
        start_all_layout.addWidget(self.stop_all_btn)
        
        group_layout.addLayout(start_all_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
    def create_output_console(self, layout):
        console_group = QGroupBox("System Output")
        console_layout = QVBoxLayout()
        
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setFont(QFont("Consolas", 9))
        console_layout.addWidget(self.output_console)
        
        console_controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Output")
        clear_btn.clicked.connect(self.output_console.clear)
        console_controls.addWidget(clear_btn)
        console_controls.addStretch()
        console_layout.addLayout(console_controls)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
    def start_all_sequential(self):
        self.append_output("System", "Starting all processes sequentially...")
        
        if not self.console_runner.is_running:
            self.console_runner.start_process()
            QMessageBox.information(self, "Starting", 
                "dVRK Console started. Click OK to continue with Simulation...")
        
        if not self.env_runner.is_running:
            self.env_runner.start_process()
            QMessageBox.information(self, "Starting",
                "AMBF Simulation started. Click OK to continue with Teleoperation...")
        
        if not self.teleop_runner.is_running:
            self.teleop_runner.start_process()
            self.append_output("System", "All processes started")
            
    def stop_all_processes(self):
        self.append_output("System", "Stopping all processes...")
        
        for runner in [self.teleop_runner, self.env_runner, self.console_runner]:
            if runner.is_running:
                runner.stop_process()
                
        self.append_output("System", "All processes stopped")
        
    def update_status(self, message):
        self.status_bar.showMessage(message)
        
    @pyqtSlot(str, str)
    def append_output(self, process_name, text):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {process_name}: {text}"
        self.output_console.append(formatted_text)
        
        cursor = self.output_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_console.setTextCursor(cursor)
            
    def closeEvent(self, event):
        for runner in [self.console_runner, self.env_runner, self.teleop_runner]:
            if runner.is_running:
                runner.stop_process()
                
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(dark_palette)
    
    window = SimpleDVRKGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()