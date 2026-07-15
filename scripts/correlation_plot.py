#!/usr/bin/env python3
"""
Focused correlation analysis script for PSM2 distance vs velocity
Generates comprehensive correlation plots for each combination
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QGroupBox, QTextEdit,
    QProgressBar, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class CorrelationAnalysisThread(QThread):
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
        analyzer = CorrelationAnalyzer(self.csv_path, self.output_dir, 
                                      self.suffix, self.plots_per_page)
        analyzer.run_analysis(self.progress_update.emit)
        self.analysis_complete.emit()


class CorrelationAnalyzer:
    """Core correlation analysis class"""
    
    def __init__(self, csv_path, output_dir, suffix, plots_per_page=4):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        self.plots_per_page = plots_per_page
        
        # Load data
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
                    # Find pull-off start using acceleration
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
                footpedal_changes = combo_df['footpedal_count'].diff() > 0
                post_engagement_changes = footpedal_changes & (combo_df['timestamp'] > engagement_time)
                
                if post_engagement_changes.any():
                    end_idx = post_engagement_changes.idxmax()
                    end_time = combo_df.loc[end_idx, 'timestamp'] - 0.1
                else:
                    end_time = combo_df['timestamp'].max()
        else:
            end_time = post_engagement_df['timestamp'].max()
        
        segment = combo_df[(combo_df['timestamp'] >= engagement_time) & 
                          (combo_df['timestamp'] <= end_time)].copy()
        
        return segment
    
    def calculate_all_correlations(self, distance, velocity):
        """Calculate various correlation metrics"""
        # Clean data
        valid_mask = ~(np.isnan(distance) | np.isnan(velocity))
        dist_clean = distance[valid_mask]
        vel_clean = velocity[valid_mask]
        
        if len(dist_clean) < 10:
            return {}
        
        results = {}
        
        # 1. Direct Pearson correlation
        results['pearson'] = np.corrcoef(dist_clean, vel_clean)[0, 1]
        
        # 2. Spearman correlation (rank-based, robust to outliers)
        results['spearman'] = stats.spearmanr(dist_clean, vel_clean)[0]
        
        # 3. Find optimal lag using cross-correlation
        dist_norm = (dist_clean - np.mean(dist_clean)) / (np.std(dist_clean) + 1e-10)
        vel_norm = (vel_clean - np.mean(vel_clean)) / (np.std(vel_clean) + 1e-10)
        
        correlation = signal.correlate(vel_norm, dist_norm, mode='same')
        lags = signal.correlation_lags(len(vel_norm), len(dist_norm), mode='same')
        
        max_corr_idx = np.argmax(np.abs(correlation))
        optimal_lag = lags[max_corr_idx]
        
        # Calculate correlation at optimal lag
        if 0 < optimal_lag < len(dist_clean) - 1:
            lagged_dist = dist_clean[:-optimal_lag]
            lagged_vel = vel_clean[optimal_lag:]
            min_len = min(len(lagged_dist), len(lagged_vel))
            results['optimal_lag'] = optimal_lag
            results['lag_correlation'] = np.corrcoef(lagged_dist[:min_len], lagged_vel[:min_len])[0, 1]
        else:
            results['optimal_lag'] = 0
            results['lag_correlation'] = results['pearson']
        
        # 4. Derivative correlation (velocity ~ d(error)/dt)
        if len(dist_clean) > 1:
            dist_derivative = np.diff(dist_clean)
            vel_for_deriv = vel_clean[1:]
            results['derivative'] = np.corrcoef(dist_derivative, vel_for_deriv)[0, 1]
        
        # 5. Quadratic correlation
        results['quadratic'] = np.corrcoef(dist_clean**2, vel_clean)[0, 1]
        
        # 6. Inverse correlation
        results['inverse'] = np.corrcoef(dist_clean, -vel_clean)[0, 1]
        
        return results
    
    def plot_correlation_analysis(self, combo_id, data, axes):
        """Create comprehensive correlation plots for one combination"""
        
        # Get data
        time = data['timestamp'] - data['timestamp'].iloc[0]
        distance_mm = data['psm2_dist'].values * 1000
        velocity_mm_s = data['psm2_velocity_magnitude'].values * 1000
        
        # Calculate correlations
        correlations = self.calculate_all_correlations(distance_mm, velocity_mm_s)
        
        # Get metadata
        shape = data['shape'].iloc[0] if 'shape' in data else 'unknown'
        size = data['size'].iloc[0] if 'size' in data else 0
        scale = data['scale'].iloc[0] if 'scale' in data else 1
        rep = data['repetition'].iloc[0] if 'repetition' in data else 1
        
        # Panel 1: Time series
        ax1 = axes[0]
        color1 = 'tab:red'
        ax1.plot(time, distance_mm, color=color1, linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Distance (mm)', color=color1, fontsize=8)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=7)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        color2 = 'tab:blue'
        ax1_twin.plot(time, velocity_mm_s, color=color2, linewidth=1.5, alpha=0.8)
        ax1_twin.set_ylabel('Velocity (mm/s)', color=color2, fontsize=8)
        ax1_twin.tick_params(axis='y', labelcolor=color2, labelsize=7)
        
        title = f"ID {combo_id} | {shape} | Size: {size:.3f}m | Scale: {scale:.1f} | Rep: {rep}"
        ax1.set_title(title, fontsize=9, pad=5)
        
        # Panel 2: Direct scatter plot
        ax2 = axes[1]
        valid = ~(np.isnan(distance_mm) | np.isnan(velocity_mm_s))
        colors = np.arange(np.sum(valid))
        scatter = ax2.scatter(distance_mm[valid], velocity_mm_s[valid], 
                            c=colors, cmap='viridis', s=2, alpha=0.6)
        
        # Add trend line
        if np.sum(valid) > 10:
            z = np.polyfit(distance_mm[valid], velocity_mm_s[valid], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(distance_mm[valid].min(), distance_mm[valid].max(), 100)
            ax2.plot(x_trend, p(x_trend), "r-", alpha=0.5, linewidth=1.5, 
                    label=f'Linear fit (r={correlations.get("pearson", 0):.3f})')
        
        ax2.set_xlabel('Distance (mm)', fontsize=8)
        ax2.set_ylabel('Velocity (mm/s)', fontsize=8)
        ax2.set_title(f'Direct Correlation: r={correlations.get("pearson", 0):.3f}', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=7)
        ax2.legend(fontsize=6)
        
        # Panel 3: Lagged correlation
        ax3 = axes[2]
        optimal_lag = correlations.get('optimal_lag', 0)
        if optimal_lag > 0 and optimal_lag < len(distance_mm) - 1:
            ax3.scatter(distance_mm[:-optimal_lag], velocity_mm_s[optimal_lag:], 
                       c=colors[:len(distance_mm)-optimal_lag], cmap='viridis', s=2, alpha=0.6)
            ax3.set_title(f'Lagged (lag={optimal_lag}): r={correlations.get("lag_correlation", 0):.3f}', 
                         fontsize=8)
        else:
            ax3.scatter(distance_mm[valid], velocity_mm_s[valid], 
                       c=colors, cmap='viridis', s=2, alpha=0.6)
            ax3.set_title('No significant lag found', fontsize=8)
        
        ax3.set_xlabel('Distance (mm)', fontsize=8)
        ax3.set_ylabel('Velocity (mm/s)', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=7)
        
        # Panel 4: Derivative correlation
        ax4 = axes[3]
        if len(distance_mm) > 1:
            dist_derivative = np.diff(distance_mm) * 10  # Scale for visibility
            vel_deriv = velocity_mm_s[1:]
            valid_deriv = ~(np.isnan(dist_derivative) | np.isnan(vel_deriv))
            colors_deriv = np.arange(np.sum(valid_deriv))
            ax4.scatter(dist_derivative[valid_deriv], vel_deriv[valid_deriv], 
                       c=colors_deriv, cmap='viridis', s=2, alpha=0.6)
            ax4.set_xlabel('d(Distance)/dt × 10', fontsize=8)
            ax4.set_ylabel('Velocity (mm/s)', fontsize=8)
            ax4.set_title(f'Derivative: r={correlations.get("derivative", 0):.3f}', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax4.set_title('Derivative correlation', fontsize=8)
        
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', labelsize=7)
        
        # Panel 5: Rolling correlation over time
        ax5 = axes[4]
        window_size = min(50, len(distance_mm) // 4)
        if window_size > 10:
            rolling_corr = []
            time_points = []
            
            for i in range(window_size, len(distance_mm) - window_size):
                window_dist = distance_mm[i-window_size:i+window_size]
                window_vel = velocity_mm_s[i-window_size:i+window_size]
                
                valid_window = ~(np.isnan(window_dist) | np.isnan(window_vel))
                if np.sum(valid_window) > 10:
                    corr = np.corrcoef(window_dist[valid_window], 
                                      window_vel[valid_window])[0, 1]
                    rolling_corr.append(corr)
                    time_points.append(time.iloc[i])
            
            if rolling_corr:
                ax5.plot(time_points, rolling_corr, 'g-', linewidth=1.5)
                ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax5.axhline(y=correlations.get('pearson', 0), color='r', 
                           linestyle='--', alpha=0.5, label='Overall')
                ax5.set_ylim(-1, 1)
                ax5.legend(fontsize=6, loc='upper right')
        
        ax5.set_xlabel('Time (s)', fontsize=8)
        ax5.set_ylabel('Correlation', fontsize=8)
        ax5.set_title(f'Rolling Correlation (window={window_size*2})', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', labelsize=7)
        
        # Panel 6: Correlation summary bar chart
        ax6 = axes[5]
        corr_types = ['Pearson', 'Spearman', 'Lagged', 'Derivative', 'Quadratic', 'Inverse']
        corr_values = [
            correlations.get('pearson', 0),
            correlations.get('spearman', 0),
            correlations.get('lag_correlation', 0),
            correlations.get('derivative', 0),
            correlations.get('quadratic', 0),
            correlations.get('inverse', 0)
        ]
        
        colors_bar = ['green' if abs(v) > 0.5 else 'yellow' if abs(v) > 0.3 else 'red' 
                     for v in corr_values]
        bars = ax6.bar(corr_types, corr_values, color=colors_bar, alpha=0.7)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax6.set_ylabel('Correlation', fontsize=8)
        ax6.set_title('Correlation Summary', fontsize=8)
        ax6.set_ylim(-1, 1)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.tick_params(axis='both', labelsize=6)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, corr_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=6)
    
    def run_analysis(self, progress_callback):
        """Run correlation analysis for all combinations"""
        
        # Get valid combinations
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
        
        n_combos = len(valid_combinations)
        n_pages = (n_combos + self.plots_per_page - 1) // self.plots_per_page
        
        progress_callback(20, f"Generating correlation plots for {n_combos} combinations...")
        
        # Generate plots page by page
        for page_idx in range(n_pages):
            start_idx = page_idx * self.plots_per_page
            end_idx = min(start_idx + self.plots_per_page, n_combos)
            page_combos = valid_combinations[start_idx:end_idx]
            n_plots = len(page_combos)
            
            # Create figure with 6 subplots per combination
            fig = plt.figure(figsize=(18, 4*n_plots))
            
            for plot_idx, combo_id in enumerate(page_combos):
                # Create 6 subplots for this combination
                axes = []
                for col in range(6):
                    ax = plt.subplot(n_plots, 6, plot_idx*6 + col + 1)
                    axes.append(ax)
                
                # Plot correlation analysis
                self.plot_correlation_analysis(combo_id, combo_data_dict[combo_id], axes)
            
            page_num = page_idx + 1
            fig.suptitle(f'Velocity-Distance Correlation Analysis - Page {page_num}/{n_pages}', 
                        fontsize=14, y=1.002)
            plt.tight_layout(rect=[0, 0, 1, 0.995])
            
            output_path = os.path.join(self.output_dir, 
                                      f'correlation_analysis_{self.suffix}_page{page_num}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            progress = 20 + int((page_idx + 1) / n_pages * 70)
            progress_callback(progress, f"Saved page {page_num}/{n_pages}")
        
        # Generate summary matrix
        progress_callback(95, "Generating correlation matrix...")
        self.generate_correlation_matrix(combo_data_dict)
        
        progress_callback(100, "Analysis complete!")
    
    def generate_correlation_matrix(self, combo_data_dict):
        """Generate summary matrix of all correlations"""
        
        # Collect all correlations
        all_correlations = []
        for combo_id, data in combo_data_dict.items():
            distance_mm = data['psm2_dist'].values * 1000
            velocity_mm_s = data['psm2_velocity_magnitude'].values * 1000
            
            correlations = self.calculate_all_correlations(distance_mm, velocity_mm_s)
            correlations['combo_id'] = combo_id
            correlations['shape'] = data['shape'].iloc[0] if 'shape' in data else 'unknown'
            correlations['scale'] = data['scale'].iloc[0] if 'scale' in data else 1
            
            all_correlations.append(correlations)
        
        df = pd.DataFrame(all_correlations)
        
        # Create matrix plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Correlation heatmap
        corr_cols = ['pearson', 'spearman', 'lag_correlation', 'derivative', 'quadratic', 'inverse']
        corr_matrix = df[corr_cols].values
        
        im = ax1.imshow(corr_matrix.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([f"ID {row['combo_id']}" for _, row in df.iterrows()], 
                            rotation=90, fontsize=6)
        ax1.set_yticks(range(len(corr_cols)))
        ax1.set_yticklabels(corr_cols, fontsize=8)
        ax1.set_title('Correlation Matrix: All Combinations', fontsize=10)
        
        plt.colorbar(im, ax=ax1, label='Correlation coefficient')
        
        # Summary statistics by shape
        shape_means = df.groupby('shape')[corr_cols].mean()
        shape_means.plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Shape', fontsize=10)
        ax2.set_ylabel('Mean Correlation', fontsize=10)
        ax2.set_title('Average Correlations by Shape', fontsize=10)
        ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Correlation Analysis Summary', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'correlation_matrix_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


class CorrelationGUI(QMainWindow):
    """GUI for correlation analysis"""
    
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.output_dir = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Velocity-Distance Correlation Analysis Tool")
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
        plots_layout.addWidget(QLabel("Combinations per page:"))
        self.plots_spinbox = QSpinBox()
        self.plots_spinbox.setMinimum(1)
        self.plots_spinbox.setMaximum(6)
        self.plots_spinbox.setValue(3)
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
        self.analyze_btn = QPushButton("Run Correlation Analysis")
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
        plots_per_page = self.plots_spinbox.value()
        
        try:
            self.analysis_thread = CorrelationAnalysisThread(
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
            f"Correlation analysis complete!\nResults saved to:\n{self.output_dir_input.text()}"
        )


def main():
    app = QApplication(sys.argv)
    window = CorrelationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()