import sys
import os
import json
import subprocess
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QGridLayout,
    QLineEdit, QComboBox, QTabWidget, QFileDialog, QMessageBox,
    QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QFormLayout,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import (
    Qt, QProcess, pyqtSignal, QTimer, QThread, pyqtSlot,
    QSettings, QSize
)
from PyQt5.QtGui import QFont, QPalette, QColor, QTextCursor

# Import matplotlib for embedded plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ROS imports (optional - will work without ROS)
try:
    import rospy
    from std_msgs.msg import String, Float32
    HAS_ROS = True
    print("ROS is available")
except ImportError:
    HAS_ROS = False
    print("ROS not available - some features will be limited")


class ROSSubscriberThread(QThread):
    """Thread for ROS subscriptions"""
    
    attributes_received = pyqtSignal(dict)
    psm1_dist_received = pyqtSignal(float)
    psm2_dist_received = pyqtSignal(float)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.message_count = 0
        
    def run(self):
        if not HAS_ROS:
            self.status_update.emit("ROS not available")
            return
            
        try:
            # Initialize ROS node with unique name
            import random
            node_name = f'dvrk_gui_subscriber_{random.randint(1000, 9999)}'
            
            if not rospy.core.is_initialized():
                rospy.init_node(node_name, anonymous=True, disable_signals=True)
                self.status_update.emit(f"ROS node initialized: {node_name}")
            else:
                self.status_update.emit("ROS already initialized")
                
            # Subscribe to topics
            self.status_update.emit("Subscribing to ROS topics...")
            
            attr_sub = rospy.Subscriber('/experiment/attributes', String, self.attributes_callback)
            psm1_sub = rospy.Subscriber('/ambf/env/psm1/trajectory_dist', Float32, self.psm1_dist_callback)
            psm2_sub = rospy.Subscriber('/ambf/env/psm2/trajectory_dist', Float32, self.psm2_dist_callback)
            
            self.running = True
            self.status_update.emit("ROS thread running, waiting for messages...")
            
            # Process callbacks
            rate = rospy.Rate(30)  # Increased rate for better responsiveness
            while self.running and not rospy.is_shutdown():
                rate.sleep()
                
                # Periodic status update
                if self.message_count > 0 and self.message_count % 100 == 0:
                    self.status_update.emit(f"Received {self.message_count} messages")
                
        except Exception as e:
            self.status_update.emit(f"ROS thread error: {e}")
            import traceback
            traceback.print_exc()
            
    def stop(self):
        self.running = False
        self.status_update.emit("Stopping ROS thread...")
        
    def attributes_callback(self, msg):
        try:
            self.message_count += 1
            data = json.loads(msg.data)
            self.attributes_received.emit(data)
        except Exception as e:
            self.status_update.emit(f"Error in attributes_callback: {e}")
            
    def psm1_dist_callback(self, msg):
        self.message_count += 1
        self.psm1_dist_received.emit(msg.data)
        
    def psm2_dist_callback(self, msg):
        self.message_count += 1
        self.psm2_dist_received.emit(msg.data)


class RealTimePlotWidget(FigureCanvas):
    """Widget for real-time plotting of PSM distances"""
    
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 4))
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Create subplot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Distance (m)')
        self.ax.set_title('PSM2 Distance from Trajectory')
        self.ax.grid(True)
        
        # Data storage
        self.max_points = 200
        self.times = deque(maxlen=self.max_points)
        self.distances = deque(maxlen=self.max_points)
        self.start_time = time.time()
        
        # Plot line
        self.line, = self.ax.plot([], [], 'b-', linewidth=2, label='PSM2')
        self.ax.legend(loc='upper right')
        
        # Initial limits
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 0.1)
        
        self.figure.tight_layout()
        
    def update_plot(self, distance):
        """Update plot with new distance value"""
        current_time = time.time() - self.start_time
        
        self.times.append(current_time)
        self.distances.append(distance)
        
        if len(self.times) > 0:
            # Update line data
            self.line.set_data(list(self.times), list(self.distances))
            
            # Update x-axis to show last 10 seconds
            if current_time > 10:
                self.ax.set_xlim(current_time - 10, current_time)
            else:
                self.ax.set_xlim(0, max(10, current_time + 1))
                
            # Update y-axis dynamically
            if len(self.distances) > 0:
                max_dist = max(self.distances)
                self.ax.set_ylim(0, max(0.01, max_dist * 1.2))
                
            # Redraw
            self.draw()
            
    def clear_plot(self):
        """Clear the plot data"""
        self.times.clear()
        self.distances.clear()
        self.start_time = time.time()
        self.line.set_data([], [])
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 0.1)
        self.draw()


class ExperimentConfigDialog(QDialog):
    """Dialog for configuring experimental attributes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Experiment Configuration")
        self.setModal(True)
        self.resize(800, 600)
        
        self.config = {
            'shapes': [],
            'size_ranges': {},
            'scaling_factors': [],
            'rotations': [],
            'positions': [],
            'repetitions': 1
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()

        # Trajectory type selection
        type_group = QGroupBox("Trajectory Type")
        type_layout = QVBoxLayout()

        self.predefined_radio = QRadioButton("Pre-defined Trajectories (Circle, Figure-8, Line, Triangle)")
        self.autogen_radio = QRadioButton("Auto-generated Trajectories (Bezier Curves)")
        self.predefined_radio.setChecked(True)

        self.predefined_radio.toggled.connect(self.toggle_trajectory_type)
        self.autogen_radio.toggled.connect(self.toggle_trajectory_type)

        type_layout.addWidget(self.predefined_radio)
        type_layout.addWidget(self.autogen_radio)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Shapes configuration (for pre-defined)
        self.shapes_group = QGroupBox("Shapes and Size Ranges")
        shapes_layout = QVBoxLayout()

        # Shape selector
        shape_selector_layout = QHBoxLayout()
        shape_selector_layout.addWidget(QLabel("Available Shapes:"))

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(['circle', 'figure8', 'line', 'triangle'])
        shape_selector_layout.addWidget(self.shape_combo)

        self.add_shape_btn = QPushButton("Add Shape")
        self.add_shape_btn.clicked.connect(self.add_shape)
        shape_selector_layout.addWidget(self.add_shape_btn)
        shape_selector_layout.addStretch()

        shapes_layout.addLayout(shape_selector_layout)
        
        # Shape configuration table
        self.shapes_table = QTableWidget()
        self.shapes_table.setColumnCount(5)
        self.shapes_table.setHorizontalHeaderLabels(['Shape', 'Min Size', 'Max Size', 'Step Size', 'Remove'])
        self.shapes_table.horizontalHeader().setStretchLastSection(True)
        shapes_layout.addWidget(self.shapes_table)
        
        self.shapes_group.setLayout(shapes_layout)
        layout.addWidget(self.shapes_group)
        
        # Motion scaling factors
        scaling_group = QGroupBox("Motion Scaling Factors")
        scaling_layout = QVBoxLayout()
        
        scaling_input_layout = QHBoxLayout()
        scaling_input_layout.addWidget(QLabel("Scaling Factors (comma-separated):"))
        self.scaling_input = QLineEdit("0.5, 1.0, 1.5, 2.0, 3.0")
        scaling_input_layout.addWidget(self.scaling_input)
        
        scaling_layout.addLayout(scaling_input_layout)
        scaling_group.setLayout(scaling_layout)
        layout.addWidget(scaling_group)
        
        # Rotations (optional)
        rotation_group = QGroupBox("Rotation Variations (Optional)")
        rotation_layout = QVBoxLayout()
        
        self.rotation_enabled = QCheckBox("Enable rotation variations")
        rotation_layout.addWidget(self.rotation_enabled)
        
        self.rotation_table = QTableWidget()
        self.rotation_table.setColumnCount(4)
        self.rotation_table.setHorizontalHeaderLabels(['Roll', 'Pitch', 'Yaw', 'Remove'])
        self.rotation_table.setEnabled(False)
        rotation_layout.addWidget(self.rotation_table)
        
        rotation_btn_layout = QHBoxLayout()
        self.add_rotation_btn = QPushButton("Add Rotation")
        self.add_rotation_btn.clicked.connect(self.add_rotation)
        self.add_rotation_btn.setEnabled(False)
        rotation_btn_layout.addWidget(self.add_rotation_btn)
        rotation_btn_layout.addStretch()
        rotation_layout.addLayout(rotation_btn_layout)
        
        self.rotation_enabled.toggled.connect(self.toggle_rotation)
        
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)
        
        # Positions (optional)
        position_group = QGroupBox("Position Variations (Optional)")
        position_layout = QVBoxLayout()
        
        self.position_enabled = QCheckBox("Enable position variations")
        position_layout.addWidget(self.position_enabled)
        
        self.position_table = QTableWidget()
        self.position_table.setColumnCount(4)
        self.position_table.setHorizontalHeaderLabels(['X', 'Y', 'Z', 'Remove'])
        self.position_table.setEnabled(False)
        position_layout.addWidget(self.position_table)
        
        position_btn_layout = QHBoxLayout()
        self.add_position_btn = QPushButton("Add Position")
        self.add_position_btn.clicked.connect(self.add_position)
        self.add_position_btn.setEnabled(False)
        position_btn_layout.addWidget(self.add_position_btn)
        position_btn_layout.addStretch()
        position_layout.addLayout(position_btn_layout)
        
        self.position_enabled.toggled.connect(self.toggle_position)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)
        
        # Repetitions
        rep_layout = QHBoxLayout()
        rep_layout.addWidget(QLabel("Repetitions:"))
        self.repetition_spin = QSpinBox()
        self.repetition_spin.setMinimum(1)
        self.repetition_spin.setMaximum(5)
        self.repetition_spin.setValue(1)
        rep_layout.addWidget(self.repetition_spin)
        rep_layout.addStretch()
        layout.addLayout(rep_layout)

        # Bezier settings (for auto-generated)
        self.bezier_group = QGroupBox("Auto-generated Trajectory Settings")
        bezier_layout = QVBoxLayout()

        arc_layout = QHBoxLayout()
        arc_layout.addWidget(QLabel("Arc Lengths (comma-separated, m):"))
        self.bezier_arc_input = QLineEdit("0.03, 0.05, 0.07")
        self.bezier_arc_input.textChanged.connect(self.update_summary)
        arc_layout.addWidget(self.bezier_arc_input)
        bezier_layout.addLayout(arc_layout)

        bound_layout = QHBoxLayout()
        bound_layout.addWidget(QLabel("Boundary Cube Size (m):"))
        self.bezier_boundary_spin = QDoubleSpinBox()
        self.bezier_boundary_spin.setRange(0.05, 0.2)
        self.bezier_boundary_spin.setValue(0.1)
        self.bezier_boundary_spin.setSingleStep(0.01)
        self.bezier_boundary_spin.setDecimals(3)
        bound_layout.addWidget(self.bezier_boundary_spin)
        bound_layout.addStretch()
        bezier_layout.addLayout(bound_layout)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Completion Threshold (mm):"))
        self.bezier_threshold_spin = QDoubleSpinBox()
        self.bezier_threshold_spin.setRange(1.0, 20.0)
        self.bezier_threshold_spin.setValue(5.0)
        self.bezier_threshold_spin.setSingleStep(0.5)
        self.bezier_threshold_spin.setDecimals(1)
        threshold_layout.addWidget(self.bezier_threshold_spin)
        threshold_layout.addStretch()
        bezier_layout.addLayout(threshold_layout)

        seg_layout = QHBoxLayout()
        seg_layout.addWidget(QLabel("Segments Visible:"))
        self.bezier_segments_spin = QSpinBox()
        self.bezier_segments_spin.setRange(1, 3)
        self.bezier_segments_spin.setValue(2)
        seg_layout.addWidget(self.bezier_segments_spin)
        seg_layout.addStretch()
        bezier_layout.addLayout(seg_layout)

        self.bezier_group.setLayout(bezier_layout)
        self.bezier_group.setVisible(False)
        layout.addWidget(self.bezier_group)

        # Summary
        self.summary_label = QLabel("Total combinations: 0")
        layout.addWidget(self.summary_label)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        self.update_summary()

    def toggle_trajectory_type(self):
        is_predefined = self.predefined_radio.isChecked()
        self.shapes_group.setVisible(is_predefined)
        self.bezier_group.setVisible(not is_predefined)
        self.update_summary()

    def add_shape(self):
        """Add a shape configuration"""
        shape = self.shape_combo.currentText()
        
        row = self.shapes_table.rowCount()
        self.shapes_table.insertRow(row)
        
        self.shapes_table.setItem(row, 0, QTableWidgetItem(shape))
        
        # Size inputs
        min_spin = QDoubleSpinBox()
        min_spin.setRange(0.001, 0.5)  # Allow smaller minimum
        min_spin.setValue(0.02)
        min_spin.setSingleStep(0.005)  # Change single step to 0.005
        min_spin.setDecimals(3)  # ADD THIS: Set 3 decimal places
        min_spin.valueChanged.connect(self.update_summary)
        self.shapes_table.setCellWidget(row, 1, min_spin)
        
        max_spin = QDoubleSpinBox()
        max_spin.setRange(0.001, 0.5)  # Allow smaller minimum
        max_spin.setValue(0.05)
        max_spin.setSingleStep(0.005)  # Change single step to 0.005
        max_spin.setDecimals(3)  # ADD THIS: Set 3 decimal places
        max_spin.valueChanged.connect(self.update_summary)
        self.shapes_table.setCellWidget(row, 2, max_spin)
        
        step_spin = QDoubleSpinBox()
        step_spin.setRange(0.001, 0.1)
        step_spin.setValue(0.005)  # Change default to 0.005
        step_spin.setSingleStep(0.001)  # Keep fine control
        step_spin.setDecimals(3)  # ADD THIS: Set 3 decimal places
        step_spin.valueChanged.connect(self.update_summary)
        self.shapes_table.setCellWidget(row, 3, step_spin)
        
        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda checked, r=row: self.remove_shape_at_row(r))
        self.shapes_table.setCellWidget(row, 4, remove_btn)
        
        self.update_summary()
        
    def remove_shape_at_row(self, row):
        """Remove a shape configuration at specific row"""
        if row < self.shapes_table.rowCount():
            self.shapes_table.removeRow(row)
            # Update remove button connections for remaining rows
            for r in range(self.shapes_table.rowCount()):
                remove_btn = self.shapes_table.cellWidget(r, 4)
                if remove_btn:
                    try:
                        remove_btn.clicked.disconnect()
                    except:
                        pass
                    remove_btn.clicked.connect(lambda checked, row_num=r: self.remove_shape_at_row(row_num))
            self.update_summary()
        
    def toggle_rotation(self, enabled):
        """Enable/disable rotation variations"""
        self.rotation_table.setEnabled(enabled)
        self.add_rotation_btn.setEnabled(enabled)
        if not enabled:
            self.rotation_table.setRowCount(0)
        self.update_summary()
        
    def toggle_position(self, enabled):
        """Enable/disable position variations"""
        self.position_table.setEnabled(enabled)
        self.add_position_btn.setEnabled(enabled)
        if not enabled:
            self.position_table.setRowCount(0)
        self.update_summary()
        
    def add_rotation(self):
        """Add a rotation configuration"""
        row = self.rotation_table.rowCount()
        self.rotation_table.insertRow(row)
        
        for col in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-180, 180)
            spin.setValue(0)
            spin.setSingleStep(15)
            spin.valueChanged.connect(self.update_summary)
            self.rotation_table.setCellWidget(row, col, spin)
            
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda checked, r=row: self.remove_rotation_at_row(r))
        self.rotation_table.setCellWidget(row, 3, remove_btn)
        
        self.update_summary()
        
    def remove_rotation_at_row(self, row):
        """Remove rotation at specific row"""
        if row < self.rotation_table.rowCount():
            self.rotation_table.removeRow(row)
            for r in range(self.rotation_table.rowCount()):
                remove_btn = self.rotation_table.cellWidget(r, 3)
                if remove_btn:
                    try:
                        remove_btn.clicked.disconnect()
                    except:
                        pass
                    remove_btn.clicked.connect(lambda checked, row_num=r: self.remove_rotation_at_row(row_num))
            self.update_summary()
        
    def add_position(self):
        """Add a position configuration"""
        row = self.position_table.rowCount()
        self.position_table.insertRow(row)
        
        for col in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-0.1, 0.1)
            spin.setValue(0.0 if col < 2 else 0.04)
            spin.setSingleStep(0.01)
            spin.valueChanged.connect(self.update_summary)
            self.position_table.setCellWidget(row, col, spin)
            
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda checked, r=row: self.remove_position_at_row(r))
        self.position_table.setCellWidget(row, 3, remove_btn)
        
        self.update_summary()
        
    def remove_position_at_row(self, row):
        """Remove position at specific row"""
        if row < self.position_table.rowCount():
            self.position_table.removeRow(row)
            for r in range(self.position_table.rowCount()):
                remove_btn = self.position_table.cellWidget(r, 3)
                if remove_btn:
                    try:
                        remove_btn.clicked.disconnect()
                    except:
                        pass
                    remove_btn.clicked.connect(lambda checked, row_num=r: self.remove_position_at_row(row_num))
            self.update_summary()
        
    def update_summary(self):
        try:
            num_shape_sizes = 0

            if self.autogen_radio.isChecked():
                arc_text = self.bezier_arc_input.text()
                num_arc_lengths = len([s.strip() for s in arc_text.split(',') if s.strip()])
                num_shape_sizes = num_arc_lengths
            else:
                for row in range(self.shapes_table.rowCount()):
                    min_widget = self.shapes_table.cellWidget(row, 1)
                    max_widget = self.shapes_table.cellWidget(row, 2)
                    step_widget = self.shapes_table.cellWidget(row, 3)

                    if min_widget and max_widget and step_widget:
                        min_val = min_widget.value()
                        max_val = max_widget.value()
                        step_val = step_widget.value()
                        if step_val > 0:
                            sizes = np.arange(min_val, max_val + step_val/2, step_val)
                            sizes = sizes[sizes <= max_val]
                            num_shape_sizes += len(sizes)

            scale_text = self.scaling_input.text()
            num_scales = len([s.strip() for s in scale_text.split(',') if s.strip()])

            num_rotations = max(1, self.rotation_table.rowCount()) if self.rotation_enabled.isChecked() else 1
            num_positions = max(1, self.position_table.rowCount()) if self.position_enabled.isChecked() else 1
            reps = self.repetition_spin.value()

            user_combinations = num_shape_sizes * num_scales * num_rotations * num_positions * reps
            total = user_combinations + 1

            self.summary_label.setText(f"Total combinations: {total} ({user_combinations} configured + 1 initialization)")
        except Exception as e:
            print(f"Error updating summary: {e}")
        
    def get_config(self):
        config = {
            'shapes': [],
            'size_ranges': {},
            'scaling_factors': [],
            'rotations': [],
            'positions': [],
            'repetitions': self.repetition_spin.value()
        }

        if self.autogen_radio.isChecked():
            config['shapes'].append('bezier')

            arc_text = self.bezier_arc_input.text()
            try:
                arc_lengths = [float(s.strip()) for s in arc_text.split(',') if s.strip()]
            except ValueError:
                arc_lengths = [0.03, 0.05, 0.07]

            config['size_ranges']['bezier'] = {
                'arc_lengths': arc_lengths,
                'boundary_size': self.bezier_boundary_spin.value(),
                'completion_threshold': self.bezier_threshold_spin.value() / 1000.0,
                'segments_visible': self.bezier_segments_spin.value()
            }
        else:
            for row in range(self.shapes_table.rowCount()):
                shape_item = self.shapes_table.item(row, 0)
                if shape_item:
                    shape = shape_item.text()
                    config['shapes'].append(shape)

                    min_widget = self.shapes_table.cellWidget(row, 1)
                    max_widget = self.shapes_table.cellWidget(row, 2)
                    step_widget = self.shapes_table.cellWidget(row, 3)

                    if min_widget and max_widget and step_widget:
                        config['size_ranges'][shape] = {
                            'min': min_widget.value(),
                            'max': max_widget.value(),
                            'step': step_widget.value()
                        }

        # Process scaling factors
        scale_text = self.scaling_input.text()
        try:
            config['scaling_factors'] = [float(s.strip()) for s in scale_text.split(',') if s.strip()]
        except ValueError:
            config['scaling_factors'] = [1.0]

        # Process rotations
        if self.rotation_enabled.isChecked() and self.rotation_table.rowCount() > 0:
            for row in range(self.rotation_table.rowCount()):
                rotation = {}
                roll_widget = self.rotation_table.cellWidget(row, 0)
                pitch_widget = self.rotation_table.cellWidget(row, 1)
                yaw_widget = self.rotation_table.cellWidget(row, 2)

                if roll_widget and pitch_widget and yaw_widget:
                    rotation['roll'] = roll_widget.value()
                    rotation['pitch'] = pitch_widget.value()
                    rotation['yaw'] = yaw_widget.value()
                    config['rotations'].append(rotation)
        else:
            config['rotations'] = [{'roll': 0, 'pitch': 0, 'yaw': 0}]

        # Process positions
        if self.position_enabled.isChecked() and self.position_table.rowCount() > 0:
            for row in range(self.position_table.rowCount()):
                position = {}
                x_widget = self.position_table.cellWidget(row, 0)
                y_widget = self.position_table.cellWidget(row, 1)
                z_widget = self.position_table.cellWidget(row, 2)

                if x_widget and y_widget and z_widget:
                    position['x'] = x_widget.value()
                    position['y'] = y_widget.value()
                    position['z'] = z_widget.value()
                    config['positions'].append(position)
        else:
            config['positions'] = [{'x': 0, 'y': 0, 'z': 0.04}]

        return config


class ProcessRunner(QWidget):
    """Widget for running and monitoring a single process"""
    
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
        
        self.control_btn = QPushButton("Start")
        self.control_btn.clicked.connect(self.toggle_process)
        self.control_btn.setFixedWidth(80)
        
        self.command_label = QLabel(self.command[:50] + "..." if len(self.command) > 50 else self.command)
        self.command_label.setStyleSheet("color: gray; font-style: italic;")
        
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.name_label)
        layout.addWidget(self.control_btn)
        layout.addWidget(self.command_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def update_command(self, new_command):
        """Update the command for this runner"""
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
            
        self.process.start('bash', ['-c', self.command])
        
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
            
        # Send SIGINT first for graceful shutdown
        try:
            pid = self.process.processId()
            os.kill(pid, signal.SIGINT)
        except:
            pass
            
        # Wait a moment, then terminate if still running
        if not self.process.waitForFinished(3000):
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()
        
        # Ensure process is cleaned up
        self.process = None
        self.is_running = False
        self.status_indicator.setStyleSheet("color: gray; font-size: 20px;")
        self.control_btn.setText("Start")
        self.output_received.emit(self.name, "Process stopped\n")
            
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
        self.output_received.emit(self.name, f"Process finished with code: {exit_code}\n")
        self.process = None


class DVRKComprehensiveGUI(QMainWindow):
    """Main GUI for comprehensive dVRK user study"""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('DVRK_Study', 'ComprehensiveGUI')
        self.experiment_config = None
        self.collection_in_progress = False
        
        # ROS subscriber thread
        self.ros_thread = None
        
        # Current attributes for display
        self.current_attributes = {}
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        self.setWindowTitle("dVRK Comprehensive Study Interface")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar(main_layout)
        
        # Create splitter
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Create tabs
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        
        # Setup tabs
        self.create_setup_tab()
        self.create_experiment_config_tab()
        self.create_collection_tab()
        self.create_analysis_tab()
        
        # Output console
        self.create_output_console(splitter)
        
        splitter.setSizes([700, 200])
        
        # Status bar
        self.status_bar = self.statusBar()
        self.update_status("Ready")
        
    def create_toolbar(self, layout):
        toolbar_layout = QHBoxLayout()
        
        # Experiment status
        self.experiment_status = QLabel("No experiment configured")
        self.experiment_status.setStyleSheet(
            "QLabel { background-color: #ffcccc; padding: 5px; border-radius: 3px; }"
        )
        toolbar_layout.addWidget(self.experiment_status)
        
        toolbar_layout.addStretch()
        
        # Emergency stop
        self.emergency_stop_btn = QPushButton("🛑 EMERGENCY STOP")
        self.emergency_stop_btn.setStyleSheet(
            "QPushButton { background-color: red; color: white; font-weight: bold; padding: 10px; }"
        )
        self.emergency_stop_btn.clicked.connect(self.emergency_stop)
        toolbar_layout.addWidget(self.emergency_stop_btn)
        
        layout.addLayout(toolbar_layout)
        
    def create_setup_tab(self):
        """Create the setup tab"""
        setup_widget = QWidget()
        setup_layout = QVBoxLayout(setup_widget)
        
        # dVRK Console
        console_group = QGroupBox("dVRK Console")
        console_layout = QVBoxLayout()
        
        self.console_runner = ProcessRunner(
            "dVRK Console",
            "source ~/catkin_ws/devel/setup.bash && rosrun dvrk_robot dvrk_console_json -j console-MTML-MTMR.json",
            "~/catkin_ws/src/dvrk/dvrk_config_jhu/jhu-daVinci"
        )
        self.console_runner.output_received.connect(self.append_output)
        console_layout.addWidget(self.console_runner)
        
        console_group.setLayout(console_layout)
        setup_layout.addWidget(console_group)
        
        # Simulation Environment
        env_group = QGroupBox("Simulation Environment")
        env_layout = QVBoxLayout()
        
        self.env_runner = ProcessRunner(
            "AMBF Environment",
            "./run_env_3D_MED_COMPLEX_LND_420006.sh",
            "~/surgical_robotics_challenge"
        )
        self.env_runner.output_received.connect(self.append_output)
        env_layout.addWidget(self.env_runner)
        
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
        
    def create_experiment_config_tab(self):
        """Create experiment configuration tab"""
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        
        # Participant info
        participant_layout = QHBoxLayout()
        participant_layout.addWidget(QLabel("Participant ID:"))
        self.participant_input = QLineEdit()
        participant_layout.addWidget(self.participant_input)
        participant_layout.addStretch()
        config_layout.addLayout(participant_layout)
        
        # Configure button
        self.configure_btn = QPushButton("Configure Experiment")
        self.configure_btn.clicked.connect(self.configure_experiment)
        config_layout.addWidget(self.configure_btn)
        
        # Configuration display
        self.config_display = QTextEdit()
        self.config_display.setReadOnly(True)
        config_layout.addWidget(self.config_display)
        
        config_layout.addStretch()
        self.tabs.addTab(config_widget, "Experiment Configuration")
        
    def create_collection_tab(self):
        """Create data collection tab with real-time plot"""
        collection_widget = QWidget()
        
        # Main horizontal layout to split controls and plot
        main_layout = QHBoxLayout(collection_widget)
        
        # Left side - controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        controls_layout.addWidget(self.progress_bar)
        
        # Current combination display
        self.current_combo_display = QLabel("Current combination: None")
        self.current_combo_display.setStyleSheet(
            "QLabel { background-color: #f0f0f0; padding: 10px; font-size: 14px; color: black; }"
        )
        controls_layout.addWidget(self.current_combo_display)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_collection_btn = QPushButton("▶️ Start Collection")
        self.start_collection_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
            }
            """
        )
        self.start_collection_btn.clicked.connect(self.start_collection)
        control_layout.addWidget(self.start_collection_btn)
        
        self.stop_collection_btn = QPushButton("⏹️ Stop Collection")
        self.stop_collection_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
            }
            """
        )
        self.stop_collection_btn.clicked.connect(self.stop_collection)
        self.stop_collection_btn.setEnabled(False)
        control_layout.addWidget(self.stop_collection_btn)
        
        control_layout.addStretch()
        controls_layout.addLayout(control_layout)
        
        # Runners (hidden)
        self.attribute_runner = ProcessRunner(
            "Attribute Manager",
            "python3 attribute_manager.py",
            "~/intel_motion_scaling/scripts"
        )
        self.attribute_runner.output_received.connect(self.append_output)
        self.attribute_runner.setVisible(False)
        controls_layout.addWidget(self.attribute_runner)
        
        self.collector_runner = ProcessRunner(
            "Data Collector",
            "python3 comprehensive_collector.py",
            "~/intel_motion_scaling/scripts"
        )
        self.collector_runner.output_received.connect(self.append_output)
        self.collector_runner.setVisible(False)
        controls_layout.addWidget(self.collector_runner)
        
        self.trajectory_runner = ProcessRunner(
            "Trajectory Generator",
            "python3 dynamic_trajectory.py",
            "~/intel_motion_scaling/scripts"
        )
        self.trajectory_runner.output_received.connect(self.append_output)
        self.trajectory_runner.setVisible(False)
        controls_layout.addWidget(self.trajectory_runner)
        
        # Add this after trajectory_runner creation
        self.proximity_runner = ProcessRunner(
            "Proximity Indicator",
            "python3 proximity_indicator.py",
            "~/intel_motion_scaling/scripts"
        )
        self.proximity_runner.output_received.connect(self.append_output)
        self.proximity_runner.setVisible(False)
        controls_layout.addWidget(self.proximity_runner)  # or whatever layout you're using

        controls_layout.addStretch()
        
        # Right side - real-time plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        plot_label = QLabel("Real-time PSM2 Distance")
        plot_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        plot_layout.addWidget(plot_label)
        
        self.realtime_plot = RealTimePlotWidget()
        plot_layout.addWidget(self.realtime_plot)
        
        # Add widgets to main layout
        main_layout.addWidget(controls_widget, 1)
        main_layout.addWidget(plot_widget, 2)
        
        self.tabs.addTab(collection_widget, "Data Collection")
        
    def create_analysis_tab(self):
        """Create analysis tab"""
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Bag file selection
        bag_layout = QHBoxLayout()
        bag_layout.addWidget(QLabel("Bag File:"))
        self.bag_path_input = QLineEdit()
        bag_layout.addWidget(self.bag_path_input)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_bag)
        bag_layout.addWidget(browse_btn)
        
        analysis_layout.addLayout(bag_layout)
        
        # Analysis options
        options_group = QGroupBox("Analysis Options")
        options_layout = QGridLayout()
        
        self.extract_videos_cb = QCheckBox("Extract Videos")
        self.extract_videos_cb.setChecked(True)
        options_layout.addWidget(self.extract_videos_cb, 0, 0)
        
        self.extract_data_cb = QCheckBox("Extract Data CSV")
        self.extract_data_cb.setChecked(True)
        options_layout.addWidget(self.extract_data_cb, 0, 1)
        
        self.generate_plots_cb = QCheckBox("Generate Plots")
        self.generate_plots_cb.setChecked(True)
        options_layout.addWidget(self.generate_plots_cb, 1, 0)
        
        self.generate_report_cb = QCheckBox("Generate Report")
        self.generate_report_cb.setChecked(True)
        options_layout.addWidget(self.generate_report_cb, 1, 1)
        
        options_group.setLayout(options_layout)
        analysis_layout.addWidget(options_group)
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.run_analysis_btn)
        
        # Analysis runner (hidden)
        self.analysis_runner = ProcessRunner(
            "Analysis",
            "python3 analyze_comprehensive.py",
            "~/intel_motion_scaling/scripts"
        )
        self.analysis_runner.output_received.connect(self.append_output)
        self.analysis_runner.setVisible(False)
        analysis_layout.addWidget(self.analysis_runner)
        
        analysis_layout.addStretch()
        self.tabs.addTab(analysis_widget, "Analysis")
        
    def create_output_console(self, parent):
        """Create output console"""
        console_group = QGroupBox("Output Console")
        console_layout = QVBoxLayout()
        
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setFont(QFont("Consolas", 9))
        console_layout.addWidget(self.output_console)
        
        console_controls = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.output_console.clear)
        console_controls.addWidget(clear_btn)
        console_controls.addStretch()
        console_layout.addLayout(console_controls)
        
        console_group.setLayout(console_layout)
        parent.addWidget(console_group)
        
    def configure_experiment(self):
        """Open experiment configuration dialog"""
        dialog = ExperimentConfigDialog(self)
        if dialog.exec_():
            self.experiment_config = dialog.get_config()
            
            # Save config to file
            config_path = os.path.expanduser(
                f"~/intel_motion_scaling/configs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.experiment_config, f, indent=2)
                
            # Update display
            self.config_display.setText(json.dumps(self.experiment_config, indent=2))
            
            # Update status
            self.experiment_status.setText(f"Experiment configured: {len(self.experiment_config.get('shapes', []))} shapes")
            self.experiment_status.setStyleSheet(
                "QLabel { background-color: #90EE90; padding: 5px; border-radius: 3px; }"
            )
            
            self.append_output("System", f"Experiment configuration saved to: {config_path}")
            
    def start_collection(self):
        """Start data collection"""
        if not self.experiment_config:
            QMessageBox.warning(self, "Warning", "Please configure experiment first!")
            return
            
        if not self.participant_input.text():
            QMessageBox.warning(self, "Warning", "Please enter participant ID!")
            return
            
        # Save config file
        config_file = os.path.expanduser(
            f"~/intel_motion_scaling/temp/current_experiment.json"
        )
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(self.experiment_config, f)
            
        # Start attribute manager
        self.attribute_runner.update_command(f"python3 attribute_manager.py --config {config_file}")
        self.attribute_runner.start_process()
        
        # Start collector
        participant_id = self.participant_input.text()
        self.collector_runner.update_command(f"python3 comprehensive_collector.py --participant {participant_id}")
        self.collector_runner.start_process()
        
        # Start dynamic trajectory
        self.trajectory_runner.update_command("python3 dynamic_trajectory.py")
        self.trajectory_runner.start_process()
        # Start proximity indicator (always enabled)
        self.proximity_runner.start_process()
        # Start ROS subscriber thread
        if HAS_ROS:
            self.ros_thread = ROSSubscriberThread()
            self.ros_thread.attributes_received.connect(self.update_attributes_display)
            self.ros_thread.psm1_dist_received.connect(lambda d: None)  # Not used for display
            self.ros_thread.psm2_dist_received.connect(self.update_realtime_plot)
            self.ros_thread.status_update.connect(lambda msg: self.append_output("ROS", msg))
            self.ros_thread.start()
            
            # Wait a moment for thread to start
            QTimer.singleShot(500, lambda: self.append_output("System", "ROS subscriptions started"))
        
        # Clear plot
        self.realtime_plot.clear_plot()
        
        # Update UI
        self.collection_in_progress = True
        self.start_collection_btn.setEnabled(False)
        self.stop_collection_btn.setEnabled(True)
        
        self.append_output("System", "Data collection started")
        
    def stop_collection(self):
        """Stop data collection"""
        # Stop ROS thread
        if self.ros_thread:
            self.ros_thread.stop()
            self.ros_thread.wait()
            self.ros_thread = None
        
        # Stop all processes
        self.attribute_runner.stop_process()
        self.collector_runner.stop_process()
        self.trajectory_runner.stop_process()
        # Stop proximity indicator
        self.proximity_runner.stop_process()
        # Update UI
        self.collection_in_progress = False
        self.start_collection_btn.setEnabled(True)
        self.stop_collection_btn.setEnabled(False)
        
        self.append_output("System", "Data collection stopped")
        
    @pyqtSlot(dict)
    def update_attributes_display(self, attributes):
        """Update display with current attributes"""
        self.current_attributes = attributes
        
        # Check for experiment completion
        if attributes.get('experiment_completed', False):
            self.append_output("System", "Experiment completed - all combinations finished")
            # Auto-stop collection after a short delay
            QTimer.singleShot(2000, self.stop_collection)
            return
        
        # Update progress bar
        current = attributes.get('current_index', 0) + 1
        total = attributes.get('total_combinations', 1)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} ({(current/total)*100:.1f}%)")
        
        # Update combination display
        shape = attributes.get('shape', 'Unknown')
        size = attributes.get('size', 0)
        scale = attributes.get('scale', 1.0)
        combo_id = attributes.get('combination_id', 0)
        
        combo_text = f"Combination {combo_id}: {shape} - Size: {size:.3f}m - Scale: {scale:.2f}"
        self.current_combo_display.setText(combo_text)
        
    @pyqtSlot(float)
    def update_realtime_plot(self, distance):
        """Update real-time plot with PSM2 distance"""
        self.realtime_plot.update_plot(distance)
        
    def browse_bag(self):
        """Browse for bag file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Bag File",
            os.path.expanduser("~/intel_motion_scaling/comprehensive_data"),
            "ROS Bag Files (*.bag)"
        )
        if path:
            self.bag_path_input.setText(path)
            
    def run_analysis(self):
        """Run analysis on bag file"""
        bag_path = self.bag_path_input.text()
        if not bag_path or not os.path.exists(bag_path):
            QMessageBox.warning(self, "Warning", "Please select a valid bag file!")
            return
            
        # Build command
        cmd = f"python3 analyze_comprehensive.py --bag {bag_path}"
        
        if self.extract_videos_cb.isChecked():
            cmd += " --videos"
        if self.extract_data_cb.isChecked():
            cmd += " --data"
        if self.generate_plots_cb.isChecked():
            cmd += " --plots"
        if self.generate_report_cb.isChecked():
            cmd += " --report"
            
        self.analysis_runner.update_command(cmd)
        self.analysis_runner.start_process()
        
        self.append_output("System", f"Running analysis: {cmd}")
        
    def emergency_stop(self):
        """Emergency stop all processes"""
        reply = QMessageBox.question(
            self, "Emergency Stop",
            "Stop all processes?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Stop ROS thread
            if self.ros_thread:
                self.ros_thread.stop()
                self.ros_thread.wait()
                self.ros_thread = None
                
            # Stop all runners
            for runner in [
                self.console_runner,
                self.env_runner, 
                self.teleop_runner,
                self.attribute_runner,
                self.collector_runner,
                self.trajectory_runner,
                self.proximity_runner  # Add this line
            ]:
                if runner.is_running:
                    runner.stop_process()
                    
            self.append_output("System", "EMERGENCY STOP - All processes terminated")
            
    def update_status(self, message):
        self.status_bar.showMessage(message)
        
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
        
    def load_settings(self):
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
            
    def closeEvent(self, event):
        """Clean up on close"""
        self.save_settings()
        
        # Stop ROS thread if running
        if self.ros_thread:
            self.ros_thread.stop()
            self.ros_thread.wait()
        
        # Stop all processes
        for runner in [
            self.console_runner if hasattr(self, 'console_runner') else None,
            self.env_runner if hasattr(self, 'env_runner') else None,
            self.teleop_runner if hasattr(self, 'teleop_runner') else None,
            self.attribute_runner if hasattr(self, 'attribute_runner') else None,
            self.collector_runner if hasattr(self, 'collector_runner') else None,
            self.trajectory_runner if hasattr(self, 'trajectory_runner') else None,
        ]:
            if runner and runner.is_running:
                runner.stop_process()
                
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
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(dark_palette)
    
    window = DVRKComprehensiveGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()