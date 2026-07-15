#!/usr/bin/env python3
"""
GUI for comprehensive trajectory heatmap analysis with consistent color mapping
Dark colors = high values, Light colors = low values for ALL metrics
Complete implementation with all features - FIXED LEGEND POSITIONING
"""

import sys
import os
import pandas as pd
import numpy as np

# Set matplotlib backend before importing pyplot to avoid GUI threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for threading safety

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Added for consistent colorbar positioning
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QCheckBox,
    QListWidget, QListWidgetItem, QGroupBox, QTextEdit,
    QProgressBar, QMessageBox, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class AnalysisThread(QThread):
    """Thread for running analysis without blocking GUI"""
    
    progress_update = pyqtSignal(int, str)
    analysis_complete = pyqtSignal()
    
    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer
        
    def run(self):
        self.analyzer.run_analysis(self.progress_update)
        self.analysis_complete.emit()


class TrajectoryAnalyzer:
    """Core analysis class"""
    
    def __init__(self, csv_path, output_dir, suffix, selected_shapes):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.suffix = suffix
        self.selected_shapes = selected_shapes
        
        # Load and filter data
        self.df = pd.read_csv(csv_path)
        self._filter_data()
        
    def _filter_data(self):
        """Filter data based on selected shapes and remove invalid entries"""
        # Remove invalid shapes
        if 'shape' in self.df.columns:
            self.df = self.df[self.df['shape'].notna()]
            self.df = self.df[~self.df['shape'].isin(['', 'none', 'None', 'dot'])]
            
            # Keep only selected shapes
            self.df = self.df[self.df['shape'].isin(self.selected_shapes)]
        
        # Remove practice trials (repetition=0)
        if 'repetition' in self.df.columns:
            self.df = self.df[self.df['repetition'] > 0]
    
    def _segment_combination_data(self, combo_df):
        """Segment data to engaged tracking period using new protocol:
        - Start: approach to < 5mm
        - Stop: backtrace to last valley before reaching 2cm pull-off
        """
        combo_df = combo_df.sort_values('timestamp').copy()
        
        # Find engagement point (PSM2 < 5mm)
        engagement_mask = combo_df['psm2_dist'] < 0.003  # 5mm threshold
        if not engagement_mask.any():
            return pd.DataFrame()
        
        engagement_idx = engagement_mask.idxmax()
        engagement_time = combo_df.loc[engagement_idx, 'timestamp']
        
        # Find pull-off by detecting when error reaches 2cm, then backtrace to last valley
        post_engagement_df = combo_df[combo_df['timestamp'] > engagement_time].copy()
        
        if len(post_engagement_df) > 10:
            # Find when error exceeds 2cm (definite pull-off)
            pulloff_mask = post_engagement_df['psm2_dist'] > 0.02
            
            if pulloff_mask.any():
                # Get index where 2cm is reached
                pulloff_idx = pulloff_mask.idxmax()
                
                # Get data from engagement to 2cm point
                tracking_segment = post_engagement_df.loc[:pulloff_idx].copy()
                
                # Find valleys (local minima) in the error signal
                # Smooth the signal slightly to avoid noise-induced valleys
                smoothed_error = tracking_segment['psm2_dist'].rolling(window=5, min_periods=1, center=True).mean()
                
                # Find local minima by checking where derivative changes from negative to positive
                error_diff = smoothed_error.diff()
                
                # A valley is where the difference goes from negative to positive
                valleys = (error_diff.shift(1) < 0) & (error_diff > 0)
                
                # Also consider points where error is below a threshold as potential valleys
                low_error_points = tracking_segment['psm2_dist'] < 0.005  # 8mm threshold
                
                # Combine both conditions
                valley_candidates = valleys | low_error_points
                
                if valley_candidates.any():
                    # Find the last valley before pull-off
                    valley_indices = tracking_segment[valley_candidates].index
                    last_valley_idx = valley_indices[-1]
                    end_time = tracking_segment.loc[last_valley_idx, 'timestamp']
                else:
                    # No clear valley found, use a point 80% of the way to pulloff
                    segment_before_pulloff = tracking_segment.iloc[:-10] if len(tracking_segment) > 20 else tracking_segment.iloc[:-5]
                    if len(segment_before_pulloff) > 0:
                        end_time = segment_before_pulloff['timestamp'].iloc[-1]
                    else:
                        end_time = tracking_segment['timestamp'].iloc[len(tracking_segment)//2]
            else:
                # No 2cm pulloff detected, use footpedal as fallback
                end_time = self._get_footpedal_end_time(combo_df, engagement_time)
        else:
            # Not enough data for analysis, use footpedal
            end_time = self._get_footpedal_end_time(combo_df, engagement_time)
        
        # Get segment between engagement and the last valley before pull-off
        segment = combo_df[(combo_df['timestamp'] >= engagement_time) & 
                          (combo_df['timestamp'] <= end_time)].copy()
        
        return segment
    
    def _get_footpedal_end_time(self, combo_df, engagement_time):
        """Helper function to get end time based on footpedal press"""
        footpedal_changes = combo_df['footpedal_count'].diff() > 0
        post_engagement_changes = footpedal_changes & (combo_df['timestamp'] > engagement_time)
        
        if post_engagement_changes.any():
            end_idx = post_engagement_changes.idxmax()
            # End slightly before footpedal press to avoid including pull-off motion
            return combo_df.loc[end_idx, 'timestamp'] - 0.1  # 100ms before footpedal
        else:
            return combo_df['timestamp'].max()
    
    def _generate_reference_trajectory(self, shape, size, num_points=200):
        """Generate reference trajectory points"""
        if shape == 'circle':
            radius = size / (2 * np.pi)
            theta = np.linspace(0, 2*np.pi, num_points)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
        elif shape == 'triangle':
            side = size / 3
            radius = side / np.sqrt(3)
            
            vertices = np.array([
                [0, radius],
                [-radius * np.sqrt(3)/2, -radius/2],
                [radius * np.sqrt(3)/2, -radius/2],
                [0, radius]
            ])
            
            points = []
            points_per_edge = num_points // 3
            for i in range(3):
                start = vertices[i]
                end = vertices[i + 1]
                t = np.linspace(0, 1, points_per_edge, endpoint=False)
                edge_points = start + t[:, np.newaxis] * (end - start)
                points.extend(edge_points)
            
            points = np.array(points)
            x = points[:, 0]
            y = points[:, 1]
        else:
            # For other shapes (line, figure8)
            if shape == 'line':
                x = np.linspace(-size/2, size/2, num_points)
                y = np.zeros(num_points)
            elif shape == 'figure8':
                a = size / 5.244
                t = np.linspace(0, 2*np.pi, num_points)
                x = a * np.sin(t) / (1 + np.cos(t)**2)
                y = a * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2)
            else:
                return None
            
        return np.column_stack([x, y])
    
    def _find_closest_trajectory_points(self, data, ref_trajectory):
        """Map data points to closest trajectory points"""
        x_data = data['psm2_x_phantom'].values
        y_data = data['psm2_y_phantom'].values
        
        closest_indices = []
        for i in range(len(x_data)):
            if np.isnan(x_data[i]) or np.isnan(y_data[i]):
                closest_indices.append(-1)
                continue
            
            point = np.array([x_data[i], y_data[i]])
            distances = np.linalg.norm(ref_trajectory - point, axis=1)
            closest_idx = np.argmin(distances)
            closest_indices.append(closest_idx)
        
        return np.array(closest_indices)
    
    def _create_reference_heatmap(self, data, ref_trajectory, closest_indices, ax, 
                                  value_col, title, cmap_name, cbar_label, error_plot=False):
        """Create reference trajectory heatmap with consistent coloring and positioning"""
        
        # Get values
        if error_plot:
            values = data[value_col].values * 1000  # Convert to mm
        else:
            values = data[value_col].values
        
        # Aggregate values at trajectory points
        traj_values = np.zeros(len(ref_trajectory))
        traj_counts = np.zeros(len(ref_trajectory))
        
        for i, closest_idx in enumerate(closest_indices):
            if closest_idx >= 0 and not np.isnan(values[i]):
                traj_values[closest_idx] += values[i]
                traj_counts[closest_idx] += 1
        
        # Average
        traj_mean_values = np.full(len(ref_trajectory), np.nan)
        mask = traj_counts > 0
        traj_mean_values[mask] = traj_values[mask] / traj_counts[mask]
        
        # Interpolate missing values
        valid_mask = ~np.isnan(traj_mean_values)
        if np.sum(valid_mask) > 2:
            indices = np.arange(len(traj_mean_values))
            f = interp1d(indices[valid_mask], traj_mean_values[valid_mask], 
                        kind='linear', fill_value='extrapolate')
            traj_mean_values = f(indices)
        
        # Create line segments
        points = ref_trajectory.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # CONSISTENT COLOR MAPPING: Dark = High, Light = Low
        if error_plot:
            actual_cmap = 'Reds'
        else:
            actual_cmap = cmap_name
        
        norm = Normalize(vmin=np.nanpercentile(traj_mean_values, 5), 
                        vmax=np.nanpercentile(traj_mean_values, 95))
        lc = LineCollection(segments, cmap=actual_cmap, norm=norm)
        
        segment_values = (traj_mean_values[:-1] + traj_mean_values[1:]) / 2
        lc.set_array(segment_values)
        
        # Line width based on magnitude
        widths = 2 + 6 * (segment_values - np.nanmin(segment_values)) / (np.nanmax(segment_values) - np.nanmin(segment_values) + 1e-6)
        lc.set_linewidths(widths)
        
        line = ax.add_collection(lc)
        
        # Set limits
        margin = 0.005
        ax.set_xlim(ref_trajectory[:, 0].min() - margin, ref_trajectory[:, 0].max() + margin)
        ax.set_ylim(ref_trajectory[:, 1].min() - margin, ref_trajectory[:, 1].max() + margin)
        
        # Formatting - IMPROVED
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # IMPROVED: Consistent colorbar positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(line, cax=cax)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Statistics
        valid_values = traj_mean_values[~np.isnan(traj_mean_values)]
        if len(valid_values) > 0:
            if error_plot:
                stats_text = f"μ={np.mean(valid_values):.1f}mm\nσ={np.std(valid_values):.1f}mm"
            else:
                stats_text = f"μ={np.mean(valid_values):.2f}m/s\nσ={np.std(valid_values):.2f}m/s"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_reference_heatmap_mtm(self, data, ref_trajectory, closest_indices, ax, 
                                      value_col, title, cmap_name, cbar_label):
        """Create heatmap on reference trajectory using MTM values mapped by PSM position"""
        
        values = data[value_col].values
        
        # Aggregate MTM values at each PSM trajectory point
        traj_values = np.zeros(len(ref_trajectory))
        traj_counts = np.zeros(len(ref_trajectory))
        
        for i, closest_idx in enumerate(closest_indices):
            if closest_idx >= 0 and not np.isnan(values[i]):
                traj_values[closest_idx] += values[i]
                traj_counts[closest_idx] += 1
        
        # Average
        traj_mean_values = np.full(len(ref_trajectory), np.nan)
        mask = traj_counts > 0
        traj_mean_values[mask] = traj_values[mask] / traj_counts[mask]
        
        # Interpolate
        valid_mask = ~np.isnan(traj_mean_values)
        if np.sum(valid_mask) > 2:
            indices = np.arange(len(traj_mean_values))
            f = interp1d(indices[valid_mask], traj_mean_values[valid_mask], 
                        kind='linear', fill_value='extrapolate')
            traj_mean_values = f(indices)
        
        # Create visualization
        points = ref_trajectory.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=np.nanpercentile(traj_mean_values, 5), 
                        vmax=np.nanpercentile(traj_mean_values, 95))
        lc = LineCollection(segments, cmap=cmap_name, norm=norm)
        
        segment_values = (traj_mean_values[:-1] + traj_mean_values[1:]) / 2
        lc.set_array(segment_values)
        
        widths = 2 + 6 * (segment_values - np.nanmin(segment_values)) / (np.nanmax(segment_values) - np.nanmin(segment_values) + 1e-6)
        lc.set_linewidths(widths)
        
        line = ax.add_collection(lc)
        
        margin = 0.005
        ax.set_xlim(ref_trajectory[:, 0].min() - margin, ref_trajectory[:, 0].max() + margin)
        ax.set_ylim(ref_trajectory[:, 1].min() - margin, ref_trajectory[:, 1].max() + margin)
        
        # IMPROVED formatting
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # IMPROVED: Consistent colorbar positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(line, cax=cax)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        valid_values = traj_mean_values[~np.isnan(traj_mean_values)]
        if len(valid_values) > 0:
            stats_text = f"μ={np.mean(valid_values):.2f}m/s"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_actual_trajectory_heatmap(self, data, ax, value_col, title, 
                                          cmap_name, cbar_label, ref_trajectory=None, 
                                          error_plot=False, alpha=0.8):
        """Create actual path heatmap with consistent coloring and positioning"""
        
        x = data['psm2_x_phantom'].values
        y = data['psm2_y_phantom'].values
        
        if error_plot:
            values = data[value_col].values * 1000
            actual_cmap = 'Reds'
        else:
            values = data[value_col].values
            actual_cmap = cmap_name
        
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(values))
        x = x[valid_mask]
        y = y[valid_mask]
        values = values[valid_mask]
        
        if len(x) < 2:
            return
        
        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=np.percentile(values, 5), vmax=np.percentile(values, 95))
        lc = LineCollection(segments, cmap=actual_cmap, norm=norm, alpha=alpha)
        
        segment_values = (values[:-1] + values[1:]) / 2
        lc.set_array(segment_values)
        
        widths = 0.5 + 3 * (segment_values - segment_values.min()) / (segment_values.max() - segment_values.min() + 1e-6)
        lc.set_linewidths(widths)
        
        line = ax.add_collection(lc)
        
        # Reference trajectory
        if ref_trajectory is not None:
            ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], 
                   'gray', linewidth=0.5, alpha=0.3, linestyle='--')
        
        # Start/end markers
        ax.plot(x[0], y[0], 'go', markersize=4, alpha=alpha, zorder=5)
        ax.plot(x[-1], y[-1], 'ro', markersize=4, alpha=alpha, zorder=5)
        
        margin = 0.005
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(y.min() - margin, y.max() + margin)
        
        # IMPROVED formatting
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # IMPROVED: Consistent colorbar positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(line, cax=cax)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    def _create_mtm_trajectory_heatmap(self, data, ax, value_col, title, cmap_name, cbar_label, alpha=0.8):
        """Create heatmap of actual MTM path with consistent positioning"""
        
        x = data['mtmr_x'].values
        y = data['mtmr_y'].values
        values = data[value_col].values
        
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(values))
        x = x[valid_mask]
        y = y[valid_mask]
        values = values[valid_mask]
        
        if len(x) < 2:
            return
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=np.percentile(values, 5), vmax=np.percentile(values, 95))
        lc = LineCollection(segments, cmap=cmap_name, norm=norm, alpha=alpha)
        
        segment_values = (values[:-1] + values[1:]) / 2
        lc.set_array(segment_values)
        
        widths = 0.5 + 3 * (segment_values - segment_values.min()) / (segment_values.max() - segment_values.min() + 1e-6)
        lc.set_linewidths(widths)
        
        line = ax.add_collection(lc)
        
        ax.plot(x[0], y[0], 'go', markersize=4, alpha=alpha, zorder=5)
        ax.plot(x[-1], y[-1], 'ro', markersize=4, alpha=alpha, zorder=5)
        
        margin = 0.01
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(y.min() - margin, y.max() + margin)
        
        # IMPROVED formatting
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # IMPROVED: Consistent colorbar positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(line, cax=cax)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    def _plot_averaged_reference(self, ax, ref_trajectory, values, title, cmap, label, error_plot=False):
        """Plot averaged values on reference trajectory with consistent positioning"""
        # Interpolate NaN values
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) > 2:
            indices = np.arange(len(values))
            f = interp1d(indices[valid_mask], values[valid_mask], 
                        kind='linear', fill_value='extrapolate')
            values = f(indices)
        
        # Create line segments
        points = ref_trajectory.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=np.nanpercentile(values, 5), 
                        vmax=np.nanpercentile(values, 95))
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        
        segment_values = (values[:-1] + values[1:]) / 2
        lc.set_array(segment_values)
        
        widths = 2 + 6 * (segment_values - np.nanmin(segment_values)) / (np.nanmax(segment_values) - np.nanmin(segment_values) + 1e-6)
        lc.set_linewidths(widths)
        
        line = ax.add_collection(lc)
        
        margin = 0.005
        ax.set_xlim(ref_trajectory[:, 0].min() - margin, ref_trajectory[:, 0].max() + margin)
        ax.set_ylim(ref_trajectory[:, 1].min() - margin, ref_trajectory[:, 1].max() + margin)
        
        # IMPROVED formatting
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # IMPROVED: Consistent colorbar positioning
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(line, cax=cax)
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Add statistics
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            if error_plot:
                stats_text = f"μ={np.mean(valid_values):.1f}mm"
            else:
                stats_text = f"μ={np.mean(valid_values):.2f}m/s"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def run_analysis(self, progress_callback):
        """Run complete analysis"""
        
        # 1. Individual combination heatmaps
        progress_callback.emit(10, "Generating individual combination heatmaps...")
        self.generate_individual_heatmaps()
        
        # 2. Averaged heatmaps for repeated combinations
        progress_callback.emit(40, "Generating averaged heatmaps...")
        self.generate_averaged_heatmaps()
        
        # 3. Statistical comparison plots
        progress_callback.emit(70, "Generating statistical plots...")
        self.generate_statistical_plots()
        
        # 4. Reference trajectory heatmaps
        progress_callback.emit(90, "Generating reference heatmaps...")
        self.generate_reference_heatmaps()
        
        progress_callback.emit(100, "Analysis complete!")
    
    def generate_individual_heatmaps(self):
        """Generate heatmaps for each individual combination - FIXED LEGEND POSITIONING"""
        combinations = sorted(self.df['combination_id'].unique())
        valid_combinations = []
        combo_data = {}
        
        for combo_id in combinations:
            combo_df = self.df[self.df['combination_id'] == combo_id]
            segment = self._segment_combination_data(combo_df)
            
            if len(segment) > 50:
                combo_data[combo_id] = {
                    'data': segment,
                    'shape': combo_df['shape'].iloc[0],
                    'size': combo_df['size'].iloc[0],
                    'scale': combo_df['scale'].iloc[0],
                    'repetition': combo_df['repetition'].iloc[0] if 'repetition' in combo_df else 1,
                    'rotation': {
                        'roll': combo_df.get('rotation_roll', pd.Series([0])).iloc[0],
                        'pitch': combo_df.get('rotation_pitch', pd.Series([0])).iloc[0],
                        'yaw': combo_df.get('rotation_yaw', pd.Series([0])).iloc[0]
                    }
                }
                valid_combinations.append(combo_id)
        
        if not valid_combinations:
            return
        
        # FIXED: Better figure sizing and spacing
        n_combos = len(valid_combinations)
        fig = plt.figure(figsize=(24, 5.5 * n_combos))
        
        # FIXED: Adjusted margins and increased column spacing
        gs = gridspec.GridSpec(n_combos, 6, figure=fig, 
                              hspace=0.6,      # Row spacing
                              wspace=0.55,     # Increased column spacing
                              left=0.03,       # Reduced left margin since no row labels
                              right=0.98,
                              top=0.96,
                              bottom=0.04)
        
        for row_idx, combo_id in enumerate(valid_combinations):
            combo_info = combo_data[combo_id]
            data = combo_info['data']
            
            # Create comprehensive attribute label for row
            attr_text = (f"ID {combo_id} | Rep {combo_info['repetition']} | "
                        f"{combo_info['shape']} | "
                        f"Size: {combo_info['size']:.3f}m | "
                        f"Scale: {combo_info['scale']:.1f}")
            
            # Add rotation info if non-zero
            if (combo_info['rotation']['roll'] != 0 or 
                combo_info['rotation']['pitch'] != 0 or 
                combo_info['rotation']['yaw'] != 0):
                attr_text += (f" | Rot: ({combo_info['rotation']['roll']:.0f}, "
                            f"{combo_info['rotation']['pitch']:.0f}, "
                            f"{combo_info['rotation']['yaw']:.0f})°")
            
            ref_trajectory = self._generate_reference_trajectory(combo_info['shape'], combo_info['size'])
            if ref_trajectory is None:
                continue
                
            closest_indices = self._find_closest_trajectory_points(data, ref_trajectory)
            
            # Generate 6 columns of heatmaps
            # Column 0: PSM2 Velocity Reference
            ax0 = fig.add_subplot(gs[row_idx, 0])
            self._create_reference_heatmap(
                data, ref_trajectory, closest_indices, ax0,
                'psm2_velocity_magnitude', 'PSM2 Vel Ref',
                'viridis', 'Velocity (m/s)'
            )
            
            # FIXED: Add attribute text inside the first plot
            ax0.text(0.98, 0.98, attr_text, transform=ax0.transAxes,
                    fontsize=8, fontweight='bold', va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Column 1: PSM2 Velocity Actual
            ax1 = fig.add_subplot(gs[row_idx, 1])
            self._create_actual_trajectory_heatmap(
                data, ax1, 'psm2_velocity_magnitude', 'PSM2 Vel Actual',
                'viridis', 'Velocity (m/s)', ref_trajectory
            )
            
            # Column 2: MTMR Velocity Mapped
            ax2 = fig.add_subplot(gs[row_idx, 2])
            self._create_reference_heatmap_mtm(
                data, ref_trajectory, closest_indices, ax2,
                'mtmr_velocity_magnitude', 'MTMR Vel Mapped',
                'plasma', 'Velocity (m/s)'
            )
            
            # Column 3: MTMR Velocity Actual
            ax3 = fig.add_subplot(gs[row_idx, 3])
            self._create_mtm_trajectory_heatmap(
                data, ax3, 'mtmr_velocity_magnitude', 'MTMR Vel Actual',
                'plasma', 'Velocity (m/s)'
            )
            
            # Column 4: PSM2 Error Reference
            ax4 = fig.add_subplot(gs[row_idx, 4])
            self._create_reference_heatmap(
                data, ref_trajectory, closest_indices, ax4,
                'psm2_dist', 'PSM2 Error Ref',
                'Reds', 'Error (mm)', error_plot=True
            )
            
            # Column 5: PSM2 Error Actual
            ax5 = fig.add_subplot(gs[row_idx, 5])
            self._create_actual_trajectory_heatmap(
                data, ax5, 'psm2_dist', 'PSM2 Error Actual',
                'Reds', 'Error (mm)', ref_trajectory, error_plot=True
            )
        
        plt.suptitle('Individual Combination Analysis', fontsize=16, y=0.99)
        output_path = os.path.join(self.output_dir, f'individual_heatmaps_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_averaged_heatmaps(self):
        """Generate averaged heatmaps for repeated combinations - FIXED LEGEND POSITIONING"""
        # Group by unique attributes to find all repetitions
        unique_attrs = self.df.groupby(['shape', 'size', 'scale']).groups
        
        # Filter to only those with multiple repetitions
        multi_rep_combos = []
        for attrs, indices in unique_attrs.items():
            # Get all unique combination_ids for this attribute set
            combo_ids = self.df.loc[indices, 'combination_id'].unique()
            
            # Check if we have multiple repetitions
            repetitions = self.df.loc[indices, 'repetition'].unique()
            repetitions = [r for r in repetitions if r > 0]
            
            if len(repetitions) > 1:  # Only process if multiple repetitions exist
                multi_rep_combos.append((attrs, combo_ids, repetitions))
        
        if not multi_rep_combos:
            return  # No repeated combinations found
        
        # FIXED: Better figure sizing and spacing
        fig = plt.figure(figsize=(22, 5.5 * len(multi_rep_combos)))
        gs = gridspec.GridSpec(len(multi_rep_combos), 6, figure=fig, 
                              hspace=0.6, wspace=0.55,  # Increased column spacing
                              left=0.03, right=0.98, top=0.96, bottom=0.04)  # Reduced left margin
        
        for row_idx, (attrs, combo_ids, repetitions) in enumerate(multi_rep_combos):
            shape, size, scale = attrs
            
            # Collect data for each repetition
            repetition_data = []
            rep_labels = []
            for combo_id in combo_ids:
                combo_df = self.df[self.df['combination_id'] == combo_id]
                rep_num = combo_df['repetition'].iloc[0] if 'repetition' in combo_df else 1
                segment = self._segment_combination_data(combo_df)
                if len(segment) > 50:
                    repetition_data.append(segment)
                    rep_labels.append(f"Rep {rep_num}")
            
            if not repetition_data:
                continue
            
            ref_trajectory = self._generate_reference_trajectory(shape, size)
            
            # FIXED: Create attribute text
            attr_text = (f"{shape} | Size: {size:.3f}m | Scale: {scale:.1f} | "
                        f"{len(repetition_data)} repetitions")
            
            # Column 0: Averaged PSM2 Velocity on Reference
            ax0 = fig.add_subplot(gs[row_idx, 0])
            all_traj_values = np.zeros(len(ref_trajectory))
            all_traj_counts = np.zeros(len(ref_trajectory))
            
            for data in repetition_data:
                closest = self._find_closest_trajectory_points(data, ref_trajectory)
                values = data['psm2_velocity_magnitude'].values
                for i, closest_idx in enumerate(closest):
                    if closest_idx >= 0 and not np.isnan(values[i]):
                        all_traj_values[closest_idx] += values[i]
                        all_traj_counts[closest_idx] += 1
            
            # Create averaged reference heatmap
            traj_mean_values = np.full(len(ref_trajectory), np.nan)
            mask = all_traj_counts > 0
            traj_mean_values[mask] = all_traj_values[mask] / all_traj_counts[mask]
            self._plot_averaged_reference(ax0, ref_trajectory, traj_mean_values, 
                                         'PSM2 Vel Avg', 'viridis', 'Velocity (m/s)')
            
            # FIXED: Add attribute text inside the first plot
            ax0.text(0.98, 0.98, attr_text, transform=ax0.transAxes,
                    fontsize=8, fontweight='bold', va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            # Column 1: All PSM2 Actual paths overlayed
            ax1 = fig.add_subplot(gs[row_idx, 1])
            for i, (data, label) in enumerate(zip(repetition_data, rep_labels)):
                self._create_actual_trajectory_heatmap(
                    data, ax1, 'psm2_velocity_magnitude', 
                    f'PSM2 Vel All Reps (n={len(repetition_data)})',
                    'viridis', 'Velocity (m/s)', ref_trajectory, 
                    alpha=0.6
                )
            ax1.set_title(f'PSM2 Vel All Reps (n={len(repetition_data)})', fontsize=10, pad=12)
            
            # Column 2: Averaged MTMR Velocity Mapped to Reference
            ax2 = fig.add_subplot(gs[row_idx, 2])
            all_traj_values = np.zeros(len(ref_trajectory))
            all_traj_counts = np.zeros(len(ref_trajectory))
            
            for data in repetition_data:
                closest = self._find_closest_trajectory_points(data, ref_trajectory)
                values = data['mtmr_velocity_magnitude'].values
                for i, closest_idx in enumerate(closest):
                    if closest_idx >= 0 and not np.isnan(values[i]):
                        all_traj_values[closest_idx] += values[i]
                        all_traj_counts[closest_idx] += 1
            
            traj_mean_values = np.full(len(ref_trajectory), np.nan)
            mask = all_traj_counts > 0
            traj_mean_values[mask] = all_traj_values[mask] / all_traj_counts[mask]
            self._plot_averaged_reference(ax2, ref_trajectory, traj_mean_values,
                                         'MTMR Vel Avg', 'plasma', 'Velocity (m/s)')
            
            # Column 3: All MTMR Actual paths overlayed
            ax3 = fig.add_subplot(gs[row_idx, 3])
            for i, data in enumerate(repetition_data):
                self._create_mtm_trajectory_heatmap(
                    data, ax3, 'mtmr_velocity_magnitude',
                    f'MTMR Vel All Reps',
                    'plasma', 'Velocity (m/s)', alpha=0.6
                )
            ax3.set_title(f'MTMR Vel All Reps (n={len(repetition_data)})', fontsize=10, pad=12)
            
            # Column 4: Averaged PSM2 Error on Reference
            ax4 = fig.add_subplot(gs[row_idx, 4])
            all_traj_values = np.zeros(len(ref_trajectory))
            all_traj_counts = np.zeros(len(ref_trajectory))
            
            for data in repetition_data:
                closest = self._find_closest_trajectory_points(data, ref_trajectory)
                values = data['psm2_dist'].values * 1000  # Convert to mm
                for i, closest_idx in enumerate(closest):
                    if closest_idx >= 0 and not np.isnan(values[i]):
                        all_traj_values[closest_idx] += values[i]
                        all_traj_counts[closest_idx] += 1
            
            traj_mean_values = np.full(len(ref_trajectory), np.nan)
            mask = all_traj_counts > 0
            traj_mean_values[mask] = all_traj_values[mask] / all_traj_counts[mask]
            self._plot_averaged_reference(ax4, ref_trajectory, traj_mean_values,
                                         'PSM2 Error Avg', 'Reds', 'Error (mm)', error_plot=True)
            
            # Column 5: All PSM2 Error Actual paths overlayed
            ax5 = fig.add_subplot(gs[row_idx, 5])
            for i, data in enumerate(repetition_data):
                self._create_actual_trajectory_heatmap(
                    data, ax5, 'psm2_dist',
                    f'PSM2 Error All Reps',
                    'Reds', 'Error (mm)', ref_trajectory, 
                    error_plot=True, alpha=0.6
                )
            ax5.set_title(f'PSM2 Error All Reps (n={len(repetition_data)})', fontsize=10, pad=12)
        
        plt.suptitle('Averaged Analysis for Repeated Combinations', fontsize=16, y=0.99)
        output_path = os.path.join(self.output_dir, f'averaged_heatmaps_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_statistical_plots(self):
        """Generate statistical comparison plots - IMPROVED SPACING"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Increased size
        plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Better spacing
        
        # Process data for statistics
        stats_data = []
        for combo_id in self.df['combination_id'].unique():
            combo_df = self.df[self.df['combination_id'] == combo_id]
            segment = self._segment_combination_data(combo_df)
            if len(segment) > 50:
                stats_data.append({
                    'shape': combo_df['shape'].iloc[0],
                    'scale': combo_df['scale'].iloc[0],
                    'psm2_error_mean': segment['psm2_dist'].mean() * 1000,
                    'psm2_velocity_mean': segment['psm2_velocity_magnitude'].mean(),
                    'mtmr_velocity_mean': segment['mtmr_velocity_magnitude'].mean()
                })
        
        if not stats_data:
            return
            
        stats_df = pd.DataFrame(stats_data)
        
        # Plot 1: Tracking Error by Shape
        ax = axes[0, 0]
        shape_errors = stats_df.groupby('shape')['psm2_error_mean'].apply(list)
        ax.boxplot(shape_errors.values, labels=shape_errors.index)
        ax.set_ylabel('PSM2 Error (mm)', fontsize=10)
        ax.set_title('Tracking Error by Shape', fontsize=12, pad=15)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: PSM2 Velocity by Shape
        ax = axes[0, 1]
        shape_velocities = stats_df.groupby('shape')['psm2_velocity_mean'].apply(list)
        ax.boxplot(shape_velocities.values, labels=shape_velocities.index)
        ax.set_ylabel('PSM2 Velocity (m/s)', fontsize=10)
        ax.set_title('PSM2 Velocity by Shape', fontsize=12, pad=15)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Error vs Scale Factor
        ax = axes[1, 0]
        for shape in stats_df['shape'].unique():
            shape_data = stats_df[stats_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['psm2_error_mean'],
                      label=shape, alpha=0.6, s=50)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('PSM2 Error (mm)', fontsize=10)
        ax.set_title('Error vs Scale Factor', fontsize=12, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Velocity Scaling Effectiveness
        ax = axes[1, 1]
        stats_df['velocity_ratio'] = stats_df['psm2_velocity_mean'] / (stats_df['mtmr_velocity_mean'] + 1e-6)
        for shape in stats_df['shape'].unique():
            shape_data = stats_df[stats_df['shape'] == shape]
            ax.scatter(shape_data['scale'], shape_data['velocity_ratio'],
                      label=shape, alpha=0.6, s=50)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Scale Factor', fontsize=10)
        ax.set_ylabel('Velocity Ratio (PSM2/MTMR)', fontsize=10)
        ax.set_title('Velocity Scaling Effectiveness', fontsize=12, pad=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Statistical Analysis', fontsize=16, y=0.95)
        
        output_path = os.path.join(self.output_dir, f'statistical_plots_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_reference_heatmaps(self):
        """Generate simplified reference trajectory heatmaps - IMPROVED SPACING"""
        combinations = sorted(self.df['combination_id'].unique())
        valid_combos = []
        
        for combo_id in combinations:
            combo_df = self.df[self.df['combination_id'] == combo_id]
            segment = self._segment_combination_data(combo_df)
            if len(segment) > 50:
                valid_combos.append({
                    'id': combo_id,
                    'data': segment,
                    'shape': combo_df['shape'].iloc[0],
                    'size': combo_df['size'].iloc[0],
                    'scale': combo_df['scale'].iloc[0]
                })
        
        if not valid_combos:
            return
        
        # IMPROVED: Better spacing
        n_combos = len(valid_combos)
        fig = plt.figure(figsize=(16, 5 * n_combos))
        gs = gridspec.GridSpec(n_combos, 3, figure=fig, 
                              hspace=0.5, wspace=0.35,
                              left=0.08, right=0.98, top=0.96, bottom=0.04)
        
        for row_idx, combo in enumerate(valid_combos):
            ref_trajectory = self._generate_reference_trajectory(combo['shape'], combo['size'])
            if ref_trajectory is None:
                continue
                
            closest_indices = self._find_closest_trajectory_points(combo['data'], ref_trajectory)
            
            # PSM2 Velocity
            ax1 = fig.add_subplot(gs[row_idx, 0])
            self._create_reference_heatmap(
                combo['data'], ref_trajectory, closest_indices, ax1,
                'psm2_velocity_magnitude',
                f"PSM2 Velocity - {combo['shape']} ({combo['size']:.3f}m)",
                'viridis', 'Velocity (m/s)'
            )
            
            # MTMR Velocity
            ax2 = fig.add_subplot(gs[row_idx, 1])
            self._create_reference_heatmap_mtm(
                combo['data'], ref_trajectory, closest_indices, ax2,
                'mtmr_velocity_magnitude',
                'MTMR Velocity Mapped',
                'plasma', 'Velocity (m/s)'
            )
            
            # PSM2 Error
            ax3 = fig.add_subplot(gs[row_idx, 2])
            self._create_reference_heatmap(
                combo['data'], ref_trajectory, closest_indices, ax3,
                'psm2_dist',
                'PSM2 Error',
                'Reds', 'Error (mm)',
                error_plot=True
            )
        
        plt.suptitle('Reference Trajectory Heatmaps', fontsize=16, y=0.99)
        output_path = os.path.join(self.output_dir, f'reference_heatmaps_{self.suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


class HeatmapAnalysisGUI(QMainWindow):
    """GUI for trajectory heatmap analysis"""
    
    def __init__(self):
        super().__init__()
        self.csv_path = None
        self.output_dir = None
        self.df = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Trajectory Heatmap Analysis Tool")
        self.setGeometry(100, 100, 800, 600)
        
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
        
        # Output suffix
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Output File Suffix:"))
        self.suffix_input = QLineEdit("participant1")
        suffix_layout.addWidget(self.suffix_input)
        suffix_layout.addStretch()
        file_layout.addLayout(suffix_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Shape selection
        shape_group = QGroupBox("Shape Selection")
        shape_layout = QVBoxLayout()
        self.shape_list = QListWidget()
        self.shape_list.setSelectionMode(QAbstractItemView.MultiSelection)
        shape_layout.addWidget(self.shape_list)
        shape_group.setLayout(shape_layout)
        layout.addWidget(shape_group)
        
        # Analysis button
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
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
            self.df = pd.read_csv(path)
            self.csv_path = path
            
            # Remove invalid shapes and dot practice (repetition=0)
            if 'shape' in self.df.columns:
                self.df = self.df[self.df['shape'].notna()]
                self.df = self.df[~self.df['shape'].isin(['', 'none', 'None', 'dot'])]
            
            # Remove practice trials (repetition=0)
            if 'repetition' in self.df.columns:
                self.df = self.df[self.df['repetition'] > 0]
            
            # Get statistics
            total_combos = self.df['combination_id'].nunique()
            
            # Find unique combinations and their repetitions correctly
            # Group by attributes and count unique combination_ids (which have different repetition numbers)
            unique_combos = self.df.groupby(['shape', 'size', 'scale'])['combination_id'].nunique()
            num_unique = len(unique_combos)
            
            # For each unique attribute set, find actual repetition numbers
            repetition_info = {}
            for attrs, group in self.df.groupby(['shape', 'size', 'scale']):
                # Get unique repetition numbers for this attribute combination
                reps = group['repetition'].unique()
                reps = sorted([r for r in reps if r > 0])  # Exclude practice (rep=0)
                repetition_info[attrs] = len(reps)
            
            # Get shapes
            shapes = self.df['shape'].unique()
            
            # Display info
            info = f"Total combinations (trials): {total_combos}\n"
            info += f"Unique attribute combinations: {num_unique}\n"
            info += f"Shapes found: {', '.join(shapes)}\n\n"
            info += "Repetitions per unique attribute combination:\n"
            for attrs, rep_count in repetition_info.items():
                shape, size, scale = attrs
                info += f"  {shape}, size={size:.3f}m, scale={scale:.1f}: {rep_count} repetition(s)\n"
            
            self.info_text.setText(info)
            
            # Populate shape list
            self.shape_list.clear()
            for shape in sorted(shapes):
                item = QListWidgetItem(shape)
                self.shape_list.addItem(item)
                item.setSelected(True)  # Select all by default
            
            # Set default output directory
            if not self.output_dir_input.text():
                self.output_dir_input.setText(os.path.dirname(path))
            
            self.check_ready()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
            
    def check_ready(self):
        """Check if ready to run analysis"""
        ready = (self.csv_path is not None and 
                self.output_dir_input.text() and
                self.shape_list.count() > 0)
        self.analyze_btn.setEnabled(ready)
        
    def run_analysis(self):
        """Run the analysis"""
        # Get selected shapes
        selected_shapes = []
        for i in range(self.shape_list.count()):
            item = self.shape_list.item(i)
            if item.isSelected():
                selected_shapes.append(item.text())
        
        if not selected_shapes:
            QMessageBox.warning(self, "Warning", "Please select at least one shape")
            return
        
        # Create output directory
        output_dir = self.output_dir_input.text()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create analyzer
        suffix = self.suffix_input.text() or "output"
        
        try:
            analyzer = TrajectoryAnalyzer(
                self.csv_path, output_dir, suffix, selected_shapes
            )
            
            # Run in thread
            self.analysis_thread = AnalysisThread(analyzer)
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
            f"Analysis complete!\nResults saved to:\n{self.output_dir_input.text()}"
        )


def main():
    app = QApplication(sys.argv)
    window = HeatmapAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()