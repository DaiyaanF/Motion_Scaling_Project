#!/usr/bin/env python3
"""
Fixed exhaustive correlation analysis script with proper error handling
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QGroupBox, QTextEdit,
    QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class AnalysisThread(QThread):
    progress_update = pyqtSignal(int, str)
    analysis_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, csv_path, output_dir, suffix):
        super().__init__()
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        
    def run(self):
        try:
            analyzer = CorrelationAnalyzer(self.csv_path, self.output_dir, self.suffix)
            analyzer.run_analysis(self.progress_update.emit)
            self.analysis_complete.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))


class CorrelationAnalyzer:
    def __init__(self, csv_path, output_dir, suffix):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        self.df = pd.read_csv(csv_path)
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and clean data"""
        # Remove invalid shapes
        if 'shape' in self.df.columns:
            self.df = self.df[self.df['shape'].notna()]
            self.df = self.df[~self.df['shape'].isin(['', 'none', 'None', 'dot'])]
        
        # Remove practice trials
        if 'repetition' in self.df.columns:
            self.df = self.df[self.df['repetition'] > 0]
        
        # Encode shape numerically
        if 'shape' in self.df.columns:
            self.shape_mapping = {shape: i for i, shape in enumerate(self.df['shape'].unique())}
            self.df['shape_encoded'] = self.df['shape'].map(self.shape_mapping)
    
    def _safe_polyfit(self, x, y, deg=1):
        """Safely perform polynomial fitting with error handling"""
        try:
            # Remove invalid values
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            if np.sum(mask) < deg + 1:
                return None
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Check for variation
            if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                return None
            
            # Check if all values are identical
            if len(np.unique(x_clean)) < 2 or len(np.unique(y_clean)) < 2:
                return None
            
            # Try polynomial fit
            coeffs = np.polyfit(x_clean, y_clean, deg)
            return coeffs
            
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return None
    
    def _safe_correlation(self, x, y):
        """Safely calculate correlation with error handling"""
        try:
            # Remove invalid values
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            if np.sum(mask) < 3:
                return None, None
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            # Check for variation
            if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
                return 0.0, 1.0
            
            # Calculate correlation
            r, p = stats.pearsonr(x_clean, y_clean)
            return r, p
            
        except (ValueError, TypeError):
            return None, None
    
    def calculate_metrics(self):
        """Calculate average metrics for each combination"""
        metrics_list = []
        
        for combo_id in self.df['combination_id'].unique():
            combo_df = self.df[self.df['combination_id'] == combo_id]
            
            # Basic segmentation - just use the middle 80% of data
            n_points = len(combo_df)
            if n_points < 50:
                continue
            
            start_idx = int(n_points * 0.1)
            end_idx = int(n_points * 0.9)
            segment = combo_df.iloc[start_idx:end_idx]
            
            try:
                metrics_list.append({
                    'combo_id': combo_id,
                    'shape': combo_df['shape'].iloc[0],
                    'shape_encoded': combo_df['shape_encoded'].iloc[0],
                    'size': float(combo_df['size'].iloc[0]),
                    'scale': float(combo_df['scale'].iloc[0]),
                    'repetition': float(combo_df['repetition'].iloc[0]),
                    'rotation_roll': float(combo_df['rotation_roll'].iloc[0]),
                    'rotation_pitch': float(combo_df['rotation_pitch'].iloc[0]),
                    'rotation_yaw': float(combo_df['rotation_yaw'].iloc[0]),
                    'position_x': float(combo_df['position_x'].iloc[0]),
                    'position_y': float(combo_df['position_y'].iloc[0]),
                    'position_z': float(combo_df['position_z'].iloc[0]),
                    'avg_velocity': float(segment['psm2_velocity_magnitude'].mean()),
                    'std_velocity': float(segment['psm2_velocity_magnitude'].std()),
                    'avg_distance': float(segment['psm2_dist'].mean() * 1000),
                    'std_distance': float(segment['psm2_dist'].std() * 1000),
                    'duration': float(segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0])
                })
            except Exception as e:
                print(f"Error processing combo {combo_id}: {e}")
                continue
        
        return pd.DataFrame(metrics_list)
    
    def plot_correlations(self, metrics_df):
        """Create correlation plots with error handling"""
        
        # Parameters to analyze
        params = ['size', 'scale', 'rotation_roll', 'rotation_pitch', 'rotation_yaw']
        
        # Create figure for velocity correlations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, param in enumerate(params):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Plot data points by shape
            for shape in metrics_df['shape'].unique():
                shape_data = metrics_df[metrics_df['shape'] == shape]
                ax.scatter(shape_data[param], shape_data['avg_velocity'], 
                          alpha=0.6, s=30, label=shape)
            
            # Try to add trend line
            x = metrics_df[param].values
            y = metrics_df['avg_velocity'].values
            
            coeffs = self._safe_polyfit(x, y, 1)
            if coeffs is not None:
                p = np.poly1d(coeffs)
                x_range = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(x_range, p(x_range), 'r--', alpha=0.5, linewidth=1)
            
            # Add correlation
            r, p_val = self._safe_correlation(x, y)
            if r is not None:
                ax.text(0.02, 0.98, f'r={r:.3f}\np={p_val:.4f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(param.replace('_', ' ').title(), fontsize=9)
            ax.set_ylabel('Avg Velocity (m/s)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        
        # Hide unused subplots
        for idx in range(len(params), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Parameter vs Velocity Correlations', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'velocity_correlations_{self.suffix}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Create figure for distance correlations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, param in enumerate(params):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Plot data points by shape
            for shape in metrics_df['shape'].unique():
                shape_data = metrics_df[metrics_df['shape'] == shape]
                ax.scatter(shape_data[param], shape_data['avg_distance'], 
                          alpha=0.6, s=30, label=shape)
            
            # Try to add trend line
            x = metrics_df[param].values
            y = metrics_df['avg_distance'].values
            
            coeffs = self._safe_polyfit(x, y, 1)
            if coeffs is not None:
                p = np.poly1d(coeffs)
                x_range = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(x_range, p(x_range), 'r--', alpha=0.5, linewidth=1)
            
            # Add correlation
            r, p_val = self._safe_correlation(x, y)
            if r is not None:
                ax.text(0.02, 0.98, f'r={r:.3f}\np={p_val:.4f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(param.replace('_', ' ').title(), fontsize=9)
            ax.set_ylabel('Avg Distance (mm)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        
        # Hide unused subplots
        for idx in range(len(params), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Parameter vs Distance Error Correlations', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'distance_correlations_{self.suffix}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_correlation_matrix(self, metrics_df):
        """Create correlation matrix heatmap"""
        
        # Select columns for correlation
        corr_columns = ['shape_encoded', 'size', 'scale', 'rotation_roll', 'rotation_pitch', 
                       'rotation_yaw', 'position_x', 'position_y', 'position_z',
                       'avg_velocity', 'avg_distance', 'duration']
        
        # Filter columns that exist
        existing_cols = [col for col in corr_columns if col in metrics_df.columns]
        
        # Calculate correlation matrix
        corr_matrix = metrics_df[existing_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(existing_cols)))
        ax.set_xticklabels(existing_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(existing_cols)))
        ax.set_yticklabels(existing_cols, fontsize=8)
        
        # Add values
        for i in range(len(existing_cols)):
            for j in range(len(existing_cols)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color=color, fontsize=6)
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        plt.title('Parameter Correlation Matrix', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'correlation_matrix_{self.suffix}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix as CSV
        csv_path = os.path.join(self.output_dir, f'correlation_matrix_{self.suffix}.csv')
        corr_matrix.to_csv(csv_path)
    
    def create_summary_report(self, metrics_df):
        """Create summary statistics report"""
        
        # Summary by shape
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity by shape
        ax = axes[0, 0]
        shape_groups = [metrics_df[metrics_df['shape'] == s]['avg_velocity'].values 
                       for s in metrics_df['shape'].unique()]
        bp = ax.boxplot(shape_groups, labels=metrics_df['shape'].unique())
        ax.set_ylabel('Avg Velocity (m/s)', fontsize=9)
        ax.set_title('Velocity by Shape', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Distance by shape
        ax = axes[0, 1]
        shape_groups = [metrics_df[metrics_df['shape'] == s]['avg_distance'].values 
                       for s in metrics_df['shape'].unique()]
        bp = ax.boxplot(shape_groups, labels=metrics_df['shape'].unique())
        ax.set_ylabel('Avg Distance (mm)', fontsize=9)
        ax.set_title('Distance Error by Shape', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Size vs Scale scatter
        ax = axes[1, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['size'], shape_data['scale'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Size (m)', fontsize=9)
        ax.set_ylabel('Scale Factor', fontsize=9)
        ax.set_title('Size vs Scale Distribution', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Duration histogram
        ax = axes[1, 1]
        ax.hist(metrics_df['duration'], bins=20, alpha=0.7, color='blue')
        ax.set_xlabel('Duration (s)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title('Trial Duration Distribution', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Summary Statistics', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'summary_stats_{self.suffix}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, progress_callback):
        """Run the complete analysis"""
        
        try:
            progress_callback(10, "Calculating metrics...")
            metrics_df = self.calculate_metrics()
            
            if metrics_df.empty:
                progress_callback(100, "No valid data found")
                return
            
            progress_callback(30, "Creating correlation plots...")
            self.plot_correlations(metrics_df)
            
            progress_callback(50, "Creating correlation matrix...")
            self.create_correlation_matrix(metrics_df)
            
            progress_callback(70, "Creating summary report...")
            self.create_summary_report(metrics_df)
            
            # Save metrics
            progress_callback(90, "Saving data...")
            metrics_path = os.path.join(self.output_dir, f'metrics_{self.suffix}.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
            progress_callback(100, "Analysis complete!")
            
        except Exception as e:
            progress_callback(100, f"Error: {str(e)}")
            raise


class CorrelationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.output_dir = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Correlation Analysis Tool")
        self.setGeometry(100, 100, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # File selection
        file_group = QGroupBox("Data Selection")
        file_layout = QVBoxLayout()
        
        # CSV selection
        csv_layout = QHBoxLayout()
        csv_layout.addWidget(QLabel("CSV File:"))
        self.csv_path_input = QLineEdit()
        csv_layout.addWidget(self.csv_path_input)
        self.csv_browse_btn = QPushButton("Browse")
        self.csv_browse_btn.clicked.connect(self.browse_csv)
        csv_layout.addWidget(self.csv_browse_btn)
        file_layout.addLayout(csv_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit()
        output_layout.addWidget(self.output_dir_input)
        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_browse_btn)
        file_layout.addLayout(output_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Output File Suffix:"))
        self.suffix_input = QLineEdit("analysis")
        suffix_layout.addWidget(self.suffix_input)
        suffix_layout.addStretch()
        settings_layout.addLayout(suffix_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if path:
            self.csv_path_input.setText(path)
            self.load_csv(path)
            
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)
            self.check_ready()
            
    def load_csv(self, path):
        try:
            df = pd.read_csv(path)
            self.csv_path = path
            
            info = f"Loaded: {os.path.basename(path)}\n"
            info += f"Rows: {len(df)}\n"
            info += f"Combinations: {df['combination_id'].nunique() if 'combination_id' in df else 'N/A'}\n"
            
            if 'shape' in df:
                shapes = df['shape'].dropna().unique()
                info += f"Shapes: {', '.join([str(s) for s in shapes[:5]])}"
            
            self.info_text.setText(info)
            
            if not self.output_dir_input.text():
                self.output_dir_input.setText(os.path.dirname(path))
            
            self.check_ready()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
            
    def check_ready(self):
        ready = bool(self.csv_path and self.output_dir_input.text())
        self.analyze_btn.setEnabled(ready)
        
    def run_analysis(self):
        output_dir = self.output_dir_input.text()
        os.makedirs(output_dir, exist_ok=True)
        
        suffix = self.suffix_input.text() or "output"
        
        self.analysis_thread = AnalysisThread(self.csv_path, output_dir, suffix)
        self.analysis_thread.progress_update.connect(self.update_progress)
        self.analysis_thread.analysis_complete.connect(self.analysis_complete)
        self.analysis_thread.error_occurred.connect(self.handle_error)
        
        self.analyze_btn.setEnabled(False)
        self.analysis_thread.start()
        
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def analysis_complete(self):
        self.analyze_btn.setEnabled(True)
        QMessageBox.information(self, "Complete", 
                               f"Analysis complete!\nResults saved to:\n{self.output_dir_input.text()}")
        
    def handle_error(self, error_msg):
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{error_msg}")


def main():
    app = QApplication(sys.argv)
    window = CorrelationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()