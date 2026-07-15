#!/usr/bin/env python3
"""
Parameter correlation analysis script
Analyzes correlations between experimental parameters (shape, size, rotation)
and performance metrics (PSM2 velocity and distance/error)
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QGroupBox, QTextEdit,
    QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class ParameterAnalysisThread(QThread):
    """Thread for running analysis without blocking GUI"""
    progress_update = pyqtSignal(int, str)
    analysis_complete = pyqtSignal()
    
    def __init__(self, csv_path, output_dir, suffix):
        super().__init__()
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        
    def run(self):
        analyzer = ParameterAnalyzer(self.csv_path, self.output_dir, self.suffix)
        analyzer.run_analysis(self.progress_update.emit)
        self.analysis_complete.emit()


class ParameterAnalyzer:
    """Core parameter correlation analysis class"""
    
    def __init__(self, csv_path, output_dir, suffix):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        
        # Load and process data
        self.df = pd.read_csv(csv_path)
        self._filter_data()
        
    def _filter_data(self):
        """Filter data to remove invalid entries"""
        if 'shape' in self.df.columns:
            self.df = self.df[self.df['shape'].notna()]
            self.df = self.df[~self.df['shape'].isin(['', 'none', 'None', 'dot'])]
        if 'repetition' in self.df.columns:
            self.df = self.df[self.df['repetition'] > 0]
    
    def _segment_combination_data(self, combo_df):
        """Segment data using pull-off detection"""
        combo_df = combo_df.sort_values('timestamp').copy()
        
        # Find engagement point (PSM2 < 5mm)
        engagement_mask = combo_df['psm2_dist'] < 0.005
        if not engagement_mask.any():
            return pd.DataFrame()
        
        engagement_idx = engagement_mask.idxmax()
        engagement_time = combo_df.loc[engagement_idx, 'timestamp']
        
        # Find pull-off
        post_engagement_df = combo_df[combo_df['timestamp'] > engagement_time].copy()
        
        if len(post_engagement_df) > 10:
            pulloff_mask = post_engagement_df['psm2_dist'] > 0.02
            
            if pulloff_mask.any():
                pulloff_idx = pulloff_mask.idxmax()
                tracking_segment = post_engagement_df.loc[:pulloff_idx].copy()
                
                if len(tracking_segment) > 20:
                    # Find pull-off start
                    tracking_segment['error_velocity'] = tracking_segment['psm2_dist'].diff()
                    tracking_segment['error_velocity_smooth'] = tracking_segment['error_velocity'].rolling(
                        window=5, min_periods=1, center=True).mean()
                    tracking_segment['error_acceleration'] = tracking_segment['error_velocity_smooth'].diff()
                    tracking_segment['error_accel_smooth'] = tracking_segment['error_acceleration'].rolling(
                        window=5, min_periods=1, center=True).mean()
                    tracking_segment['accel_change'] = tracking_segment['error_accel_smooth'].diff().abs()
                    
                    search_segment = tracking_segment.iloc[:-10] if len(tracking_segment) > 30 else tracking_segment.iloc[:-5]
                    
                    if len(search_segment) > 10:
                        stable_period = search_segment.iloc[:len(search_segment)//2]
                        if len(stable_period['accel_change'].dropna()) > 0:
                            baseline_std = stable_period['accel_change'].std()
                            threshold = baseline_std * 3
                            
                            significant_changes = search_segment['accel_change'] > threshold
                            sustained_accel = search_segment['error_accel_smooth'] > 0.0001
                            pulloff_start_candidates = significant_changes | sustained_accel
                            
                            if pulloff_start_candidates.any():
                                change_indices = search_segment[pulloff_start_candidates].index
                                first_change_idx = change_indices[0]
                                safe_idx = max(0, search_segment.index.get_loc(first_change_idx) - 3)
                                end_time = search_segment.iloc[safe_idx]['timestamp']
                            else:
                                end_time = search_segment['timestamp'].iloc[int(len(search_segment) * 0.8)]
                        else:
                            end_time = search_segment['timestamp'].iloc[int(len(search_segment) * 0.8)]
                    else:
                        end_time = tracking_segment['timestamp'].iloc[len(tracking_segment)//2]
                else:
                    end_time = tracking_segment['timestamp'].iloc[len(tracking_segment)//2]
            else:
                end_time = post_engagement_df['timestamp'].max()
        else:
            end_time = post_engagement_df['timestamp'].max()
        
        segment = combo_df[(combo_df['timestamp'] >= engagement_time) & 
                          (combo_df['timestamp'] <= end_time)].copy()
        
        return segment
    
    def calculate_combination_metrics(self):
        """Calculate average metrics for each combination"""
        metrics_list = []
        
        for combo_id in self.df['combination_id'].unique():
            combo_df = self.df[self.df['combination_id'] == combo_id]
            segment = self._segment_combination_data(combo_df)
            
            if len(segment) > 50:
                # Calculate average metrics
                avg_velocity = segment['psm2_velocity_magnitude'].mean()
                std_velocity = segment['psm2_velocity_magnitude'].std()
                max_velocity = segment['psm2_velocity_magnitude'].max()
                
                avg_distance = segment['psm2_dist'].mean() * 1000  # Convert to mm
                std_distance = segment['psm2_dist'].std() * 1000
                max_distance = segment['psm2_dist'].max() * 1000
                
                # Get parameters
                shape = combo_df['shape'].iloc[0]
                size = combo_df['size'].iloc[0]
                scale = combo_df['scale'].iloc[0]
                repetition = combo_df['repetition'].iloc[0] if 'repetition' in combo_df else 1
                
                # Get rotation values
                rotation_roll = combo_df.get('rotation_roll', pd.Series([0])).iloc[0]
                rotation_pitch = combo_df.get('rotation_pitch', pd.Series([0])).iloc[0]
                rotation_yaw = combo_df.get('rotation_yaw', pd.Series([0])).iloc[0]
                rotation_magnitude = np.sqrt(rotation_roll**2 + rotation_pitch**2 + rotation_yaw**2)
                
                metrics_list.append({
                    'combo_id': combo_id,
                    'shape': shape,
                    'size': size,
                    'scale': scale,
                    'repetition': repetition,
                    'rotation_roll': rotation_roll,
                    'rotation_pitch': rotation_pitch,
                    'rotation_yaw': rotation_yaw,
                    'rotation_magnitude': rotation_magnitude,
                    'avg_velocity': avg_velocity,
                    'std_velocity': std_velocity,
                    'max_velocity': max_velocity,
                    'avg_distance': avg_distance,
                    'std_distance': std_distance,
                    'max_distance': max_distance,
                    'duration': segment['timestamp'].iloc[-1] - segment['timestamp'].iloc[0]
                })
        
        return pd.DataFrame(metrics_list)
    
    def plot_shape_correlations(self, metrics_df):
        """Plot correlations between shape and performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Shape encoding for correlation calculation
        shape_encoding = {shape: i for i, shape in enumerate(metrics_df['shape'].unique())}
        metrics_df['shape_encoded'] = metrics_df['shape'].map(shape_encoding)
        
        # 1. Shape vs Average Velocity
        ax = axes[0, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter([shape] * len(shape_data), shape_data['avg_velocity'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Shape vs Average PSM2 Velocity', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add statistical test (ANOVA)
        shape_groups = [group['avg_velocity'].values for name, group in metrics_df.groupby('shape')]
        f_stat, p_value = stats.f_oneway(*shape_groups)
        ax.text(0.02, 0.98, f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Shape vs Average Distance Error
        ax = axes[0, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter([shape] * len(shape_data), shape_data['avg_distance'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Shape vs Average PSM2 Distance Error', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        shape_groups = [group['avg_distance'].values for name, group in metrics_df.groupby('shape')]
        f_stat, p_value = stats.f_oneway(*shape_groups)
        ax.text(0.02, 0.98, f'ANOVA: F={f_stat:.2f}, p={p_value:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Box plots for velocity by shape
        ax = axes[0, 2]
        shape_velocity = [group['avg_velocity'].values for name, group in metrics_df.groupby('shape')]
        bp = ax.boxplot(shape_velocity, labels=metrics_df['shape'].unique())
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Velocity Distribution by Shape', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 4. Shape vs Max Velocity
        ax = axes[1, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter([shape] * len(shape_data), shape_data['max_velocity'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Max Velocity (m/s)', fontsize=10)
        ax.set_title('Shape vs Maximum PSM2 Velocity', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 5. Box plots for distance by shape
        ax = axes[1, 1]
        shape_distance = [group['avg_distance'].values for name, group in metrics_df.groupby('shape')]
        bp = ax.boxplot(shape_distance, labels=metrics_df['shape'].unique())
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Distance Error Distribution by Shape', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 6. Shape vs Duration
        ax = axes[1, 2]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter([shape] * len(shape_data), shape_data['duration'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Trial Duration (s)', fontsize=10)
        ax.set_title('Shape vs Trial Duration', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Shape-Performance Correlations', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'shape_correlations_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_size_correlations(self, metrics_df):
        """Plot correlations between size and performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Size vs Average Velocity
        ax = axes[0, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['size'] * 1000, shape_data['avg_velocity'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Size vs Average PSM2 Velocity', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        r, p = stats.pearsonr(metrics_df['size'], metrics_df['avg_velocity'])
        ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add trend line
        z = np.polyfit(metrics_df['size'] * 1000, metrics_df['avg_velocity'], 1)
        p_line = np.poly1d(z)
        x_trend = np.linspace(metrics_df['size'].min() * 1000, metrics_df['size'].max() * 1000, 100)
        ax.plot(x_trend, p_line(x_trend), "r--", alpha=0.5, linewidth=2)
        
        # 2. Size vs Average Distance Error
        ax = axes[0, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['size'] * 1000, shape_data['avg_distance'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Size vs Average PSM2 Distance Error', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        r, p = stats.pearsonr(metrics_df['size'], metrics_df['avg_distance'])
        ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        z = np.polyfit(metrics_df['size'] * 1000, metrics_df['avg_distance'], 1)
        p_line = np.poly1d(z)
        ax.plot(x_trend, p_line(x_trend), "r--", alpha=0.5, linewidth=2)
        
        # 3. Size vs Max Velocity
        ax = axes[0, 2]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['size'] * 1000, shape_data['max_velocity'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Max Velocity (m/s)', fontsize=10)
        ax.set_title('Size vs Maximum PSM2 Velocity', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. Size vs Velocity (normalized by shape)
        ax = axes[1, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            if len(shape_data) > 2:
                ax.scatter(shape_data['size'] * 1000, shape_data['avg_velocity'], 
                          alpha=0.6, s=50, label=shape)
                # Fit line per shape
                z = np.polyfit(shape_data['size'] * 1000, shape_data['avg_velocity'], 1)
                p_line = np.poly1d(z)
                x_shape = np.linspace(shape_data['size'].min() * 1000, 
                                     shape_data['size'].max() * 1000, 50)
                ax.plot(x_shape, p_line(x_shape), "--", alpha=0.5, linewidth=1)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Size-Velocity Correlation by Shape', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 5. Size vs Distance (normalized by shape)
        ax = axes[1, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            if len(shape_data) > 2:
                ax.scatter(shape_data['size'] * 1000, shape_data['avg_distance'], 
                          alpha=0.6, s=50, label=shape)
                z = np.polyfit(shape_data['size'] * 1000, shape_data['avg_distance'], 1)
                p_line = np.poly1d(z)
                x_shape = np.linspace(shape_data['size'].min() * 1000, 
                                     shape_data['size'].max() * 1000, 50)
                ax.plot(x_shape, p_line(x_shape), "--", alpha=0.5, linewidth=1)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Size-Distance Correlation by Shape', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Size vs Duration
        ax = axes[1, 2]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['size'] * 1000, shape_data['duration'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Trial Duration (s)', fontsize=10)
        ax.set_title('Size vs Trial Duration', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Size-Performance Correlations', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'size_correlations_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_rotation_correlations(self, metrics_df):
        """Plot correlations between rotation and performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Rotation Magnitude vs Average Velocity
        ax = axes[0, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['rotation_magnitude'], shape_data['avg_velocity'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Rotation Magnitude (degrees)', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Rotation vs Average PSM2 Velocity', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        r, p = stats.pearsonr(metrics_df['rotation_magnitude'], metrics_df['avg_velocity'])
        ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Rotation Magnitude vs Average Distance Error
        ax = axes[0, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['rotation_magnitude'], shape_data['avg_distance'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Rotation Magnitude (degrees)', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Rotation vs Average PSM2 Distance Error', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        r, p = stats.pearsonr(metrics_df['rotation_magnitude'], metrics_df['avg_distance'])
        ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Individual rotation components (Yaw) vs Velocity
        ax = axes[0, 2]
        ax.scatter(metrics_df['rotation_yaw'], metrics_df['avg_velocity'], 
                  alpha=0.6, s=50, c=metrics_df['rotation_magnitude'], cmap='viridis')
        ax.set_xlabel('Yaw Rotation (degrees)', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Yaw Rotation vs Velocity', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label='Total Rotation')
        
        # 4. Rotation components heatmap
        ax = axes[1, 0]
        rotation_impact = metrics_df[['rotation_roll', 'rotation_pitch', 'rotation_yaw']].values
        performance = metrics_df[['avg_velocity', 'avg_distance']].values
        
        # Calculate correlation matrix
        corr_matrix = np.zeros((3, 2))
        for i, rot in enumerate(['rotation_roll', 'rotation_pitch', 'rotation_yaw']):
            corr_matrix[i, 0] = stats.pearsonr(metrics_df[rot], metrics_df['avg_velocity'])[0]
            corr_matrix[i, 1] = stats.pearsonr(metrics_df[rot], metrics_df['avg_distance'])[0]
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Avg Velocity', 'Avg Distance'])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Roll', 'Pitch', 'Yaw'])
        ax.set_title('Rotation Components Correlation Matrix', fontsize=11)
        plt.colorbar(im, ax=ax)
        
        # Add values
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                       ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
        
        # 5. Rotation vs Max Distance Error
        ax = axes[1, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['rotation_magnitude'], shape_data['max_distance'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Rotation Magnitude (degrees)', fontsize=10)
        ax.set_ylabel('Max Distance Error (mm)', fontsize=10)
        ax.set_title('Rotation vs Maximum Distance Error', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Rotation effect by shape
        ax = axes[1, 2]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            if len(shape_data) > 2:
                # Calculate correlation per shape
                r = stats.pearsonr(shape_data['rotation_magnitude'], shape_data['avg_distance'])[0]
                ax.bar(shape, r, alpha=0.7)
        ax.set_xlabel('Shape', fontsize=10)
        ax.set_ylabel('Rotation-Error Correlation', fontsize=10)
        ax.set_title('Rotation Impact by Shape', fontsize=11)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Rotation-Performance Correlations', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'rotation_correlations_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_scale_correlations(self, metrics_df):
        """Plot correlations between scale factor and performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Scale vs Average Velocity
        ax = axes[0, 0]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['avg_velocity'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=10)
        ax.set_title('Scale Factor vs Average PSM2 Velocity', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        r, p = stats.pearsonr(metrics_df['scale'], metrics_df['avg_velocity'])
        ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Scale vs Average Distance Error
        ax = axes[0, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['avg_distance'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Scale Factor vs Average PSM2 Distance Error', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        r, p = stats.pearsonr(metrics_df['scale'], metrics_df['avg_distance'])
        ax.text(0.02, 0.98, f'r={r:.3f}, p={p:.4f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Scale effectiveness (expected vs actual velocity ratio)
        ax = axes[0, 2]
        metrics_df['velocity_ratio'] = metrics_df['avg_velocity'] / metrics_df['scale']
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['velocity_ratio'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Velocity / Scale Ratio', fontsize=10)
        ax.set_title('Scale Effectiveness', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. Box plots by scale bins
        ax = axes[1, 0]
        scale_bins = pd.cut(metrics_df['scale'], bins=3, labels=['Low', 'Medium', 'High'])
        metrics_df['scale_bin'] = scale_bins
        scale_groups = [group['avg_distance'].values for name, group in metrics_df.groupby('scale_bin')]
        bp = ax.boxplot(scale_groups, labels=['Low', 'Medium', 'High'])
        ax.set_xlabel('Scale Factor Group', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Error Distribution by Scale Group', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 5. Scale vs Duration
        ax = axes[1, 1]
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['duration'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Trial Duration (s)', fontsize=10)
        ax.set_title('Scale Factor vs Trial Duration', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Combined effect: Size * Scale vs Performance
        ax = axes[1, 2]
        metrics_df['size_scale_product'] = metrics_df['size'] * metrics_df['scale']
        for shape in metrics_df['shape'].unique():
            shape_data = metrics_df[metrics_df['shape'] == shape]
            ax.scatter(shape_data['size_scale_product'], shape_data['avg_distance'], 
                      alpha=0.6, s=50, label=shape)
        ax.set_xlabel('Size × Scale Product', fontsize=10)
        ax.set_ylabel('Average Distance Error (mm)', fontsize=10)
        ax.set_title('Combined Size-Scale Effect', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Scale Factor-Performance Correlations', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'scale_correlations_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_summary_table(self, metrics_df):
        """Generate a summary table of all correlations"""
        
        # Calculate all correlations
        parameters = ['size', 'scale', 'rotation_magnitude', 'rotation_roll', 
                     'rotation_pitch', 'rotation_yaw']
        metrics = ['avg_velocity', 'std_velocity', 'max_velocity', 
                  'avg_distance', 'std_distance', 'max_distance', 'duration']
        
        correlation_results = []
        for param in parameters:
            for metric in metrics:
                r, p = stats.pearsonr(metrics_df[param], metrics_df[metric])
                correlation_results.append({
                    'Parameter': param,
                    'Metric': metric,
                    'Correlation': r,
                    'P-value': p,
                    'Significant': 'Yes' if p < 0.05 else 'No'
                })
        
        corr_df = pd.DataFrame(correlation_results)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Correlation matrix heatmap
        pivot_corr = corr_df.pivot(index='Parameter', columns='Metric', values='Correlation')
        im = ax1.imshow(pivot_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax1.set_xticks(range(len(pivot_corr.columns)))
        ax1.set_xticklabels(pivot_corr.columns, rotation=45, ha='right', fontsize=8)
        ax1.set_yticks(range(len(pivot_corr.index)))
        ax1.set_yticklabels(pivot_corr.index, fontsize=9)
        ax1.set_title('Parameter-Performance Correlation Matrix', fontsize=11)
        
        # Add correlation values
        for i in range(len(pivot_corr.index)):
            for j in range(len(pivot_corr.columns)):
                val = pivot_corr.values[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        color=color, fontsize=7)
        
        plt.colorbar(im, ax=ax1, label='Pearson Correlation')
        
        # Significant correlations bar plot
        significant_corr = corr_df[corr_df['Significant'] == 'Yes'].copy()
        significant_corr['Label'] = significant_corr['Parameter'] + '-' + significant_corr['Metric']
        significant_corr = significant_corr.sort_values('Correlation', key=abs)
        
        if len(significant_corr) > 0:
            y_pos = np.arange(len(significant_corr))
            colors = ['green' if x > 0 else 'red' for x in significant_corr['Correlation']]
            
            ax2.barh(y_pos, significant_corr['Correlation'], color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(significant_corr['Label'], fontsize=7)
            ax2.set_xlabel('Correlation Coefficient', fontsize=10)
            ax2.set_title('Significant Correlations (p < 0.05)', fontsize=11)
            ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Parameter-Performance Correlation Summary', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'correlation_summary_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Save correlation table to CSV
        csv_path = os.path.join(self.output_dir, f'correlation_table_{self.suffix}.csv')
        corr_df.to_csv(csv_path, index=False)
        
        return corr_df
    
    def run_analysis(self, progress_callback):
        """Run complete parameter correlation analysis"""
        
        progress_callback(10, "Calculating metrics for each combination...")
        metrics_df = self.calculate_combination_metrics()
        
        if metrics_df.empty:
            progress_callback(100, "No valid data found")
            return
        
        progress_callback(25, "Generating shape correlation plots...")
        self.plot_shape_correlations(metrics_df)
        
        progress_callback(40, "Generating size correlation plots...")
        self.plot_size_correlations(metrics_df)
        
        progress_callback(55, "Generating rotation correlation plots...")
        self.plot_rotation_correlations(metrics_df)
        
        progress_callback(70, "Generating scale factor correlation plots...")
        self.plot_scale_correlations(metrics_df)
        
        progress_callback(85, "Generating correlation summary...")
        corr_summary = self.generate_summary_table(metrics_df)
        
        # Save metrics DataFrame
        metrics_path = os.path.join(self.output_dir, f'combination_metrics_{self.suffix}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        progress_callback(100, "Analysis complete!")


class ParameterCorrelationGUI(QMainWindow):
    """GUI for parameter correlation analysis"""
    
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.output_dir = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Parameter-Performance Correlation Analysis")
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
        self.suffix_input = QLineEdit("participant1")
        suffix_layout.addWidget(self.suffix_input)
        suffix_layout.addStretch()
        settings_layout.addLayout(suffix_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("Run Parameter Correlation Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if path:
            self.csv_path_input.setText(path)
            self.load_csv(path)
            
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if path:
            self.output_dir_input.setText(path)
            self.check_ready()
            
    def load_csv(self, path):
        """Load CSV and display information"""
        try:
            df = pd.read_csv(path)
            self.csv_path = path
            
            if 'shape' in df.columns:
                df = df[df['shape'].notna()]
                df = df[~df['shape'].isin(['', 'none', 'None', 'dot'])]
            
            if 'repetition' in df.columns:
                df = df[df['repetition'] > 0]
            
            total_combos = df['combination_id'].nunique()
            shapes = df['shape'].unique() if 'shape' in df else []
            
            info = f"Total combinations: {total_combos}\n"
            info += f"Shapes found: {', '.join(shapes)}\n"
            info += f"Data points: {len(df)}"
            
            self.info_text.setText(info)
            
            if not self.output_dir_input.text():
                self.output_dir_input.setText(os.path.dirname(path))
            
            self.check_ready()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
            
    def check_ready(self):
        """Check if ready to run analysis"""
        ready = bool(self.csv_path is not None and self.output_dir_input.text())
        self.analyze_btn.setEnabled(ready)
        
    def run_analysis(self):
        """Run the analysis"""
        output_dir = self.output_dir_input.text()
        os.makedirs(output_dir, exist_ok=True)
        
        suffix = self.suffix_input.text() or "output"
        
        try:
            self.analysis_thread = ParameterAnalysisThread(
                self.csv_path, output_dir, suffix
            )
            self.analysis_thread.progress_update.connect(self.update_progress)
            self.analysis_thread.analysis_complete.connect(self.analysis_complete)
            
            self.analyze_btn.setEnabled(False)
            self.analysis_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def analysis_complete(self):
        self.analyze_btn.setEnabled(True)
        QMessageBox.information(
            self, "Complete", 
            f"Parameter correlation analysis complete!\nResults saved to:\n{self.output_dir_input.text()}"
        )


def main():
    app = QApplication(sys.argv)
    window = ParameterCorrelationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()