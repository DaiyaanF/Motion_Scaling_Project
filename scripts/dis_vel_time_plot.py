#!/usr/bin/env python3
"""
Time series analysis script for PSM2 distance and velocity plotting
Generates dual-axis plots showing error (mm) and velocity (mm/s) over time for each combination
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QGroupBox, QTextEdit,
    QProgressBar, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class TimeSeriesAnalysisThread(QThread):
    """Thread for running analysis without blocking GUI"""
    progress_update = pyqtSignal(int, str)
    analysis_complete = pyqtSignal()
    
    def __init__(self, csv_path, output_dir, suffix, plots_per_page):
        super().__init__()
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        self.plots_per_page = plots_per_page
        
    def run(self):
        analyzer = TimeSeriesAnalyzer(self.csv_path, self.output_dir, 
                                     self.suffix, self.plots_per_page)
        analyzer.run_analysis(self.progress_update.emit)
        self.analysis_complete.emit()


class TimeSeriesAnalyzer:
    """Core time series analysis class"""
    
    def __init__(self, csv_path, output_dir, suffix, plots_per_page=6):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        self.plots_per_page = plots_per_page
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self._filter_data()
        
    def _filter_data(self):
        """Filter data to remove invalid entries"""
        # Remove invalid shapes
        if 'shape' in self.df.columns:
            self.df = self.df[self.df['shape'].notna()]
            self.df = self.df[~self.df['shape'].isin(['', 'none', 'None', 'dot'])]
        
        # Remove practice trials (repetition=0)
        if 'repetition' in self.df.columns:
            self.df = self.df[self.df['repetition'] > 0]
    
    def _segment_combination_data(self, combo_df):
        """Segment data using the same protocol as heatmap analysis:
        - Start: approach to < 5mm
        - Stop: detect acceleration change before 2cm pull-off
        """
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
                    # Calculate derivatives to find pull-off start
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
                # No pulloff, use footpedal
                end_time = self._get_footpedal_end_time(combo_df, engagement_time)
        else:
            end_time = self._get_footpedal_end_time(combo_df, engagement_time)
        
        segment = combo_df[(combo_df['timestamp'] >= engagement_time) & 
                          (combo_df['timestamp'] <= end_time)].copy()
        
        return segment
    
    def _get_footpedal_end_time(self, combo_df, engagement_time):
        """Get end time based on footpedal press"""
        footpedal_changes = combo_df['footpedal_count'].diff() > 0
        post_engagement_changes = footpedal_changes & (combo_df['timestamp'] > engagement_time)
        
        if post_engagement_changes.any():
            end_idx = post_engagement_changes.idxmax()
            return combo_df.loc[end_idx, 'timestamp'] - 0.1
        else:
            return combo_df['timestamp'].max()
    
    def plot_combination_timeseries(self, combo_id, combo_data, ax):
        """Plot time series for a single combination with dual y-axes"""
        
        # Reset time to start from 0
        time = combo_data['timestamp'] - combo_data['timestamp'].iloc[0]
        
        # Convert to mm and mm/s
        distance_mm = combo_data['psm2_dist'] * 1000  # Convert to mm
        velocity_mm_s = combo_data['psm2_velocity_magnitude'] * 1000  # Convert to mm/s
        
        # Plot distance on primary y-axis
        color1 = 'tab:red'
        ax.plot(time, distance_mm, color=color1, linewidth=1.5, alpha=0.8, label='Distance Error')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Distance Error (mm)', color=color1, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color1, labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Create secondary y-axis for velocity
        ax2 = ax.twinx()
        color2 = 'tab:blue'
        ax2.plot(time, velocity_mm_s, color=color2, linewidth=1.5, alpha=0.8, label='Velocity')
        ax2.set_ylabel('Velocity (mm/s)', color=color2, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)
        
        # Get attributes for title
        shape = combo_data['shape'].iloc[0] if 'shape' in combo_data else 'unknown'
        size = combo_data['size'].iloc[0] if 'size' in combo_data else 0
        scale = combo_data['scale'].iloc[0] if 'scale' in combo_data else 1
        rep = combo_data['repetition'].iloc[0] if 'repetition' in combo_data else 1
        
        # Add title with attributes
        title = f"ID {combo_id} | {shape} | Size: {size:.3f}m | Scale: {scale:.1f} | Rep: {rep}"
        ax.set_title(title, fontsize=9, pad=10)
        
        # Add statistics
        avg_error = distance_mm.mean()
        std_error = distance_mm.std()
        avg_vel = velocity_mm_s.mean()
        std_vel = velocity_mm_s.std()
        
        stats_text = (f"Error: μ={avg_error:.1f}±{std_error:.1f}mm\n"
                     f"Vel: μ={avg_vel:.1f}±{std_vel:.1f}mm/s")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
        
    def run_analysis(self, progress_callback):
        """Run time series analysis for all combinations"""
        
        # Get all valid combinations
        combinations = sorted(self.df['combination_id'].unique())
        valid_combinations = []
        combo_data_dict = {}
        
        progress_callback(10, "Segmenting combinations...")
        
        for combo_id in combinations:
            combo_df = self.df[self.df['combination_id'] == combo_id]
            segment = self._segment_combination_data(combo_df)
            
            if len(segment) > 50:
                valid_combinations.append(combo_id)
                combo_data_dict[combo_id] = segment
        
        if not valid_combinations:
            progress_callback(100, "No valid combinations found")
            return
        
        # Calculate number of pages needed
        n_combos = len(valid_combinations)
        n_pages = (n_combos + self.plots_per_page - 1) // self.plots_per_page
        
        progress_callback(20, f"Generating plots for {n_combos} combinations...")
        
        # Generate plots page by page
        for page_idx in range(n_pages):
            start_idx = page_idx * self.plots_per_page
            end_idx = min(start_idx + self.plots_per_page, n_combos)
            page_combos = valid_combinations[start_idx:end_idx]
            n_plots = len(page_combos)
            
            # Create figure with subplots
            if n_plots <= 3:
                fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
                if n_plots == 1:
                    axes = [axes]
            else:
                n_cols = 2
                n_rows = (n_plots + 1) // 2
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
                axes = axes.flatten()
            
            # Plot each combination
            for plot_idx, combo_id in enumerate(page_combos):
                ax = axes[plot_idx]
                combo_data = combo_data_dict[combo_id]
                self.plot_combination_timeseries(combo_id, combo_data, ax)
            
            # Hide unused subplots if any
            if n_plots < len(axes):
                for idx in range(n_plots, len(axes)):
                    axes[idx].set_visible(False)
            
            # Add overall title and adjust layout
            page_num = page_idx + 1
            fig.suptitle(f'PSM2 Distance and Velocity Time Series - Page {page_num}/{n_pages}', 
                        fontsize=14, y=1.02)
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(self.output_dir, 
                                      f'timeseries_analysis_{self.suffix}_page{page_num}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            progress = 20 + int((page_idx + 1) / n_pages * 70)
            progress_callback(progress, f"Saved page {page_num}/{n_pages}")
        
        # Generate summary statistics plot
        progress_callback(95, "Generating summary statistics...")
        self.generate_summary_plot(combo_data_dict)
        
        progress_callback(100, "Analysis complete!")
    
    def generate_summary_plot(self, combo_data_dict):
        """Generate summary statistics plot comparing all combinations"""
        
        # Collect statistics for each combination
        stats_list = []
        for combo_id, data in combo_data_dict.items():
            distance_mm = data['psm2_dist'] * 1000
            velocity_mm_s = data['psm2_velocity_magnitude'] * 1000
            
            stats_list.append({
                'combo_id': combo_id,
                'shape': data['shape'].iloc[0] if 'shape' in data else 'unknown',
                'size': data['size'].iloc[0] if 'size' in data else 0,
                'scale': data['scale'].iloc[0] if 'scale' in data else 1,
                'avg_error': distance_mm.mean(),
                'std_error': distance_mm.std(),
                'max_error': distance_mm.max(),
                'avg_velocity': velocity_mm_s.mean(),
                'std_velocity': velocity_mm_s.std(),
                'max_velocity': velocity_mm_s.max(),
                'duration': data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]
            })
        
        stats_df = pd.DataFrame(stats_list)
        
        # Create summary figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Average error by shape
        ax = axes[0, 0]
        shape_errors = stats_df.groupby('shape')['avg_error'].apply(list)
        if len(shape_errors) > 0:
            ax.boxplot(shape_errors.values, labels=shape_errors.index)
            ax.set_ylabel('Average Error (mm)', fontsize=10)
            ax.set_title('Average Tracking Error by Shape', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Average velocity by shape
        ax = axes[0, 1]
        shape_velocities = stats_df.groupby('shape')['avg_velocity'].apply(list)
        if len(shape_velocities) > 0:
            ax.boxplot(shape_velocities.values, labels=shape_velocities.index)
            ax.set_ylabel('Average Velocity (mm/s)', fontsize=10)
            ax.set_title('Average Velocity by Shape', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Error vs Scale
        ax = axes[0, 2]
        for shape in stats_df['shape'].unique():
            shape_data = stats_df[stats_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['avg_error'], 
                      label=shape, alpha=0.6, s=50)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Average Error (mm)', fontsize=10)
        ax.set_title('Error vs Scale Factor', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Velocity vs Scale
        ax = axes[1, 0]
        for shape in stats_df['shape'].unique():
            shape_data = stats_df[stats_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['avg_velocity'], 
                      label=shape, alpha=0.6, s=50)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Average Velocity (mm/s)', fontsize=10)
        ax.set_title('Velocity vs Scale Factor', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Error vs Size
        ax = axes[1, 1]
        for shape in stats_df['shape'].unique():
            shape_data = stats_df[stats_df['shape'] == shape]
            ax.scatter(shape_data['size']*1000, shape_data['avg_error'], 
                      label=shape, alpha=0.6, s=50)
        ax.set_xlabel('Pattern Size (mm)', fontsize=10)
        ax.set_ylabel('Average Error (mm)', fontsize=10)
        ax.set_title('Error vs Pattern Size', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Duration distribution
        ax = axes[1, 2]
        shape_durations = stats_df.groupby('shape')['duration'].apply(list)
        if len(shape_durations) > 0:
            ax.boxplot(shape_durations.values, labels=shape_durations.index)
            ax.set_ylabel('Duration (s)', fontsize=10)
            ax.set_title('Trial Duration by Shape', fontsize=11)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Time Series Analysis Summary Statistics', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'timeseries_summary_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_correlation_plots(self, combo_data_dict):
        """Generate correlation analysis plots between distance and velocity"""
        
        n_combos = len(combo_data_dict)
        n_cols = 3
        n_rows = (n_combos + n_cols - 1) // n_cols
        
        # Create figure for scatter plots
        fig = plt.figure(figsize=(15, 4*n_rows))
        
        for idx, (combo_id, data) in enumerate(combo_data_dict.items()):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Get data
            distance_mm = data['psm2_dist'].values * 1000
            velocity_mm_s = data['psm2_velocity_magnitude'].values * 1000
            
            # Calculate correlations
            correlations = self.calculate_correlations(distance_mm, velocity_mm_s)
            
            # Find best correlation
            best_corr_type = max(correlations.items(), key=lambda x: abs(x[1]))[0]
            best_corr_value = correlations[best_corr_type]
            
            # Create scatter plot based on best correlation type
            if 'lag' in best_corr_type:
                # Extract lag value and plot lagged data
                lag_val = int(best_corr_type.split('_')[1]) if '_' in best_corr_type else 0
                if lag_val > 0 and lag_val < len(distance_mm):
                    x_data = distance_mm[:-lag_val] if lag_val > 0 else distance_mm
                    y_data = velocity_mm_s[lag_val:] if lag_val > 0 else velocity_mm_s
                    plot_label = f'Lag={lag_val} samples'
                else:
                    x_data = distance_mm
                    y_data = velocity_mm_s
                    plot_label = 'No lag'
            elif best_corr_type == 'derivative':
                # Plot velocity vs derivative of distance
                x_data = np.diff(distance_mm) * 10  # Scale for visibility
                y_data = velocity_mm_s[1:]
                plot_label = 'd(Error)/dt × 10'
            elif best_corr_type == 'quadratic':
                x_data = distance_mm
                y_data = velocity_mm_s
                plot_label = 'Quadratic'
            else:
                x_data = distance_mm
                y_data = velocity_mm_s
                plot_label = best_corr_type.capitalize()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
            # Create scatter plot with color gradient for time
            colors = np.arange(len(x_data))
            scatter = ax.scatter(x_data, y_data, c=colors, cmap='viridis', 
                               s=2, alpha=0.6)
            
            # Add trend line
            if len(x_data) > 10:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data.min(), x_data.max(), 100)
                ax.plot(x_trend, p(x_trend), "r-", alpha=0.5, linewidth=2)
            
            # Labels and title
            if best_corr_type == 'derivative':
                ax.set_xlabel('d(Error)/dt × 10 (mm/sample)', fontsize=8)
            else:
                ax.set_xlabel('Distance Error (mm)', fontsize=8)
            ax.set_ylabel('Velocity (mm/s)', fontsize=8)
            
            # Get attributes
            shape = data['shape'].iloc[0] if 'shape' in data else 'unknown'
            scale = data['scale'].iloc[0] if 'scale' in data else 1
            
            ax.set_title(f'ID {combo_id} | {shape} | Scale: {scale:.1f}\n'
                        f'{plot_label} | r={best_corr_value:.3f}', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for time
            if idx == 0:
                cbar = plt.colorbar(scatter, ax=ax, label='Time progression')
                cbar.ax.tick_params(labelsize=6)
        
        plt.suptitle('Distance-Velocity Correlation Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'correlation_analysis_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Create a summary correlation matrix
        self.generate_correlation_matrix(combo_data_dict)
    
    def generate_correlation_matrix(self, combo_data_dict):
        """Generate a matrix showing all correlation types for all combinations"""
        
        # Collect all correlations
        corr_data = []
        for combo_id, data in combo_data_dict.items():
            distance_mm = data['psm2_dist'].values * 1000
            velocity_mm_s = data['psm2_velocity_magnitude'].values * 1000
            correlations = self.calculate_correlations(distance_mm, velocity_mm_s)
            
            # Add metadata
            correlations['combo_id'] = combo_id
            correlations['shape'] = data['shape'].iloc[0] if 'shape' in data else 'unknown'
            correlations['scale'] = data['scale'].iloc[0] if 'scale' in data else 1
            corr_data.append(correlations)
        
        # Create DataFrame
        corr_df = pd.DataFrame(corr_data)
        
        # Select correlation columns (exclude metadata)
        corr_cols = [col for col in corr_df.columns if col not in ['combo_id', 'shape', 'scale']]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Plot 1: Heatmap of correlations
        corr_matrix = corr_df[corr_cols].values
        im = ax1.imshow(corr_matrix.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        ax1.set_xticks(range(len(corr_df)))
        ax1.set_xticklabels([f"{row['combo_id']}" for _, row in corr_df.iterrows()], 
                            rotation=90, fontsize=6)
        ax1.set_yticks(range(len(corr_cols)))
        ax1.set_yticklabels(corr_cols, fontsize=8)
        ax1.set_title('Correlation Types Across All Combinations', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Correlation coefficient', fontsize=8)
        
        # Plot 2: Best correlation by shape
        best_corrs = []
        for _, row in corr_df.iterrows():
            corr_values = {col: abs(row[col]) for col in corr_cols if not pd.isna(row[col])}
            if corr_values:
                best_type = max(corr_values.items(), key=lambda x: x[1])
                best_corrs.append({
                    'shape': row['shape'],
                    'scale': row['scale'],
                    'type': best_type[0],
                    'value': row[best_type[0]]  # Use signed value
                })
        
        best_corr_df = pd.DataFrame(best_corrs)
        
        # Group by shape and correlation type
        shape_groups = best_corr_df.groupby(['shape', 'type'])['value'].mean().unstack(fill_value=0)
        
        if not shape_groups.empty:
            shape_groups.plot(kind='bar', ax=ax2)
            ax2.set_xlabel('Shape', fontsize=10)
            ax2.set_ylabel('Average Correlation', fontsize=10)
            ax2.set_title('Dominant Correlation Types by Shape', fontsize=10)
            ax2.legend(title='Correlation Type', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        plt.suptitle('Correlation Analysis Summary', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'correlation_matrix_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


class TimeSeriesGUI(QMainWindow):
    """GUI for time series analysis"""
    
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.output_dir = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("PSM2 Time Series Analysis Tool")
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
        
        # Output suffix
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Output File Suffix:"))
        self.suffix_input = QLineEdit("participant1")
        suffix_layout.addWidget(self.suffix_input)
        suffix_layout.addStretch()
        settings_layout.addLayout(suffix_layout)
        
        # Plots per page
        plots_layout = QHBoxLayout()
        plots_layout.addWidget(QLabel("Plots per page:"))
        self.plots_spinbox = QSpinBox()
        self.plots_spinbox.setMinimum(1)
        self.plots_spinbox.setMaximum(12)
        self.plots_spinbox.setValue(6)
        plots_layout.addWidget(self.plots_spinbox)
        plots_layout.addStretch()
        settings_layout.addLayout(plots_layout)
        
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
        self.analyze_btn = QPushButton("Run Time Series Analysis")
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
            
            # Remove invalid data
            if 'shape' in df.columns:
                df = df[df['shape'].notna()]
                df = df[~df['shape'].isin(['', 'none', 'None', 'dot'])]
            
            if 'repetition' in df.columns:
                df = df[df['repetition'] > 0]
            
            # Get statistics
            total_combos = df['combination_id'].nunique()
            shapes = df['shape'].unique() if 'shape' in df else []
            
            info = f"Total combinations: {total_combos}\n"
            info += f"Shapes found: {', '.join(shapes)}\n"
            info += f"Data points: {len(df)}"
            
            self.info_text.setText(info)
            
            # Set default output directory
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
        plots_per_page = self.plots_spinbox.value()
        
        try:
            self.analysis_thread = TimeSeriesAnalysisThread(
                self.csv_path, output_dir, suffix, plots_per_page
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
            f"Time series analysis complete!\nResults saved to:\n{self.output_dir_input.text()}"
        )


def main():
    app = QApplication(sys.argv)
    window = TimeSeriesGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()