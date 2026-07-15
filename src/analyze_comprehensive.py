#!/usr/bin/env python3
import os
import json
import argparse
import rosbag
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import seaborn as sns


def extract_comprehensive_data(bag_path, include_initialization=True):
    """Extract comprehensive data from bag file to pandas DataFrame with all fields
    
    Args:
        bag_path: Path to bag file
        include_initialization: If False, exclude the initialization dot pattern
    """
    data_list = []
    
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/comprehensive_data']):
            try:
                data = json.loads(msg.data)
                
                # Filter out initialization dot if requested
                if not include_initialization:
                    if data.get('repetition', 1) == 0:
                        continue
                    if data.get('shape') == 'dot':
                        continue
                    if data.get('is_initialization', False):
                        continue
                
                # Flatten nested structures for CSV export
                flat_data = {
                    'timestamp': data.get('timestamp'),
                    'participant_id': data.get('participant_id'),
                    
                    # Experimental attributes
                    'combination_id': data.get('combination_id'),
                    'shape': data.get('shape'),
                    'size': data.get('size'),
                    'scale': data.get('scale'),
                    'repetition': data.get('repetition'),
                    'current_index': data.get('current_index'),
                    'total_combinations': data.get('total_combinations'),
                    
                    # PSM distances
                    'psm1_dist': data.get('psm1_dist'),
                    'psm2_dist': data.get('psm2_dist'),
                    
                    # PSM1 positions
                    'psm1_x_phantom': data.get('psm1_position_phantom', {}).get('x') if data.get('psm1_position_phantom') else None,
                    'psm1_y_phantom': data.get('psm1_position_phantom', {}).get('y') if data.get('psm1_position_phantom') else None,
                    'psm1_z_phantom': data.get('psm1_position_phantom', {}).get('z') if data.get('psm1_position_phantom') else None,
                    
                    'psm1_x_world': data.get('psm1_position_world', {}).get('x') if data.get('psm1_position_world') else None,
                    'psm1_y_world': data.get('psm1_position_world', {}).get('y') if data.get('psm1_position_world') else None,
                    'psm1_z_world': data.get('psm1_position_world', {}).get('z') if data.get('psm1_position_world') else None,
                    
                    # PSM2 positions
                    'psm2_x_phantom': data.get('psm2_position_phantom', {}).get('x') if data.get('psm2_position_phantom') else None,
                    'psm2_y_phantom': data.get('psm2_position_phantom', {}).get('y') if data.get('psm2_position_phantom') else None,
                    'psm2_z_phantom': data.get('psm2_position_phantom', {}).get('z') if data.get('psm2_position_phantom') else None,
                    
                    'psm2_x_world': data.get('psm2_position_world', {}).get('x') if data.get('psm2_position_world') else None,
                    'psm2_y_world': data.get('psm2_position_world', {}).get('y') if data.get('psm2_position_world') else None,
                    'psm2_z_world': data.get('psm2_position_world', {}).get('z') if data.get('psm2_position_world') else None,
                    
                    # PSM velocities (linear)
                    'psm1_vel_x': data.get('psm1_velocity', {}).get('linear', {}).get('x') if data.get('psm1_velocity') else None,
                    'psm1_vel_y': data.get('psm1_velocity', {}).get('linear', {}).get('y') if data.get('psm1_velocity') else None,
                    'psm1_vel_z': data.get('psm1_velocity', {}).get('linear', {}).get('z') if data.get('psm1_velocity') else None,
                    'psm1_velocity_magnitude': data.get('psm1_velocity_magnitude'),
                    
                    'psm2_vel_x': data.get('psm2_velocity', {}).get('linear', {}).get('x') if data.get('psm2_velocity') else None,
                    'psm2_vel_y': data.get('psm2_velocity', {}).get('linear', {}).get('y') if data.get('psm2_velocity') else None,
                    'psm2_vel_z': data.get('psm2_velocity', {}).get('linear', {}).get('z') if data.get('psm2_velocity') else None,
                    'psm2_velocity_magnitude': data.get('psm2_velocity_magnitude'),
                    
                    # PSM angular velocities
                    'psm1_ang_vel_x': data.get('psm1_velocity', {}).get('angular', {}).get('x') if data.get('psm1_velocity') else None,
                    'psm1_ang_vel_y': data.get('psm1_velocity', {}).get('angular', {}).get('y') if data.get('psm1_velocity') else None,
                    'psm1_ang_vel_z': data.get('psm1_velocity', {}).get('angular', {}).get('z') if data.get('psm1_velocity') else None,
                    
                    'psm2_ang_vel_x': data.get('psm2_velocity', {}).get('angular', {}).get('x') if data.get('psm2_velocity') else None,
                    'psm2_ang_vel_y': data.get('psm2_velocity', {}).get('angular', {}).get('y') if data.get('psm2_velocity') else None,
                    'psm2_ang_vel_z': data.get('psm2_velocity', {}).get('angular', {}).get('z') if data.get('psm2_velocity') else None,
                    
                    # MTML position
                    'mtml_x': data.get('mtml_position', {}).get('x') if data.get('mtml_position') else None,
                    'mtml_y': data.get('mtml_position', {}).get('y') if data.get('mtml_position') else None,
                    'mtml_z': data.get('mtml_position', {}).get('z') if data.get('mtml_position') else None,
                    'mtml_qx': data.get('mtml_position', {}).get('qx') if data.get('mtml_position') else None,
                    'mtml_qy': data.get('mtml_position', {}).get('qy') if data.get('mtml_position') else None,
                    'mtml_qz': data.get('mtml_position', {}).get('qz') if data.get('mtml_position') else None,
                    'mtml_qw': data.get('mtml_position', {}).get('qw') if data.get('mtml_position') else None,
                    
                    # MTML velocity
                    'mtml_vel_x': data.get('mtml_velocity', {}).get('linear', {}).get('x') if data.get('mtml_velocity') else None,
                    'mtml_vel_y': data.get('mtml_velocity', {}).get('linear', {}).get('y') if data.get('mtml_velocity') else None,
                    'mtml_vel_z': data.get('mtml_velocity', {}).get('linear', {}).get('z') if data.get('mtml_velocity') else None,
                    'mtml_velocity_magnitude': data.get('mtml_velocity_magnitude'),
                    
                    'mtml_ang_vel_x': data.get('mtml_velocity', {}).get('angular', {}).get('x') if data.get('mtml_velocity') else None,
                    'mtml_ang_vel_y': data.get('mtml_velocity', {}).get('angular', {}).get('y') if data.get('mtml_velocity') else None,
                    'mtml_ang_vel_z': data.get('mtml_velocity', {}).get('angular', {}).get('z') if data.get('mtml_velocity') else None,
                    
                    # MTMR position
                    'mtmr_x': data.get('mtmr_position', {}).get('x') if data.get('mtmr_position') else None,
                    'mtmr_y': data.get('mtmr_position', {}).get('y') if data.get('mtmr_position') else None,
                    'mtmr_z': data.get('mtmr_position', {}).get('z') if data.get('mtmr_position') else None,
                    'mtmr_qx': data.get('mtmr_position', {}).get('qx') if data.get('mtmr_position') else None,
                    'mtmr_qy': data.get('mtmr_position', {}).get('qy') if data.get('mtmr_position') else None,
                    'mtmr_qz': data.get('mtmr_position', {}).get('qz') if data.get('mtmr_position') else None,
                    'mtmr_qw': data.get('mtmr_position', {}).get('qw') if data.get('mtmr_position') else None,
                    
                    # MTMR velocity
                    'mtmr_vel_x': data.get('mtmr_velocity', {}).get('linear', {}).get('x') if data.get('mtmr_velocity') else None,
                    'mtmr_vel_y': data.get('mtmr_velocity', {}).get('linear', {}).get('y') if data.get('mtmr_velocity') else None,
                    'mtmr_vel_z': data.get('mtmr_velocity', {}).get('linear', {}).get('z') if data.get('mtmr_velocity') else None,
                    'mtmr_velocity_magnitude': data.get('mtmr_velocity_magnitude'),
                    
                    'mtmr_ang_vel_x': data.get('mtmr_velocity', {}).get('angular', {}).get('x') if data.get('mtmr_velocity') else None,
                    'mtmr_ang_vel_y': data.get('mtmr_velocity', {}).get('angular', {}).get('y') if data.get('mtmr_velocity') else None,
                    'mtmr_ang_vel_z': data.get('mtmr_velocity', {}).get('angular', {}).get('z') if data.get('mtmr_velocity') else None,
                    
                    # Events
                    'clutch_count': data.get('clutch_count'),
                    'footpedal_count': data.get('footpedal_count'),
                    
                    # Rotation (if exists)
                    'rotation_roll': data.get('rotation', {}).get('roll') if data.get('rotation') else 0,
                    'rotation_pitch': data.get('rotation', {}).get('pitch') if data.get('rotation') else 0,
                    'rotation_yaw': data.get('rotation', {}).get('yaw') if data.get('rotation') else 0,
                    
                    # Position offset (if exists)
                    'position_x': data.get('position', {}).get('x') if data.get('position') else 0,
                    'position_y': data.get('position', {}).get('y') if data.get('position') else 0,
                    'position_z': data.get('position', {}).get('z') if data.get('position') else 0.04,
                }
                
                data_list.append(flat_data)
            except Exception as e:
                print(f"Error processing data point: {e}")
                pass
                
    return pd.DataFrame(data_list)


def extract_videos(bag_path, output_dir, fps=30):
    """Extract videos from bag file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writers
    video_l = None
    video_r = None
    
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/video/cameraL', '/video/cameraR']):
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
                
            if topic == '/video/cameraL':
                if video_l is None:
                    h, w = image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_l = cv2.VideoWriter(
                        os.path.join(output_dir, 'camera_left.mp4'),
                        fourcc, fps, (w, h)
                    )
                video_l.write(image)
                
            elif topic == '/video/cameraR':
                if video_r is None:
                    h, w = image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_r = cv2.VideoWriter(
                        os.path.join(output_dir, 'camera_right.mp4'),
                        fourcc, fps, (w, h)
                    )
                video_r.write(image)
                
    if video_l:
        video_l.release()
    if video_r:
        video_r.release()
        
    print(f"Videos extracted to {output_dir}")


def generate_plots(df, output_dir, included_init=True):
    """Generate analysis plots with new data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    title_suffix = " (initialization excluded)" if not included_init else ""
    
    # 1. Distance and velocity over time
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # PSM1 distance
    axes[0,0].plot(df['timestamp'], df['psm1_dist'], 'b-', alpha=0.7, linewidth=0.5)
    axes[0,0].set_ylabel('PSM1 Distance (m)')
    axes[0,0].set_title(f'PSM1 Distance from Trajectory{title_suffix}')
    axes[0,0].grid(True, alpha=0.3)
    
    # PSM2 distance
    axes[0,1].plot(df['timestamp'], df['psm2_dist'], 'r-', alpha=0.7, linewidth=0.5)
    axes[0,1].set_ylabel('PSM2 Distance (m)')
    axes[0,1].set_title(f'PSM2 Distance from Trajectory{title_suffix}')
    axes[0,1].grid(True, alpha=0.3)
    
    # PSM1 velocity
    axes[1,0].plot(df['timestamp'], df['psm1_velocity_magnitude'], 'b-', alpha=0.7, linewidth=0.5)
    axes[1,0].set_ylabel('PSM1 Velocity (m/s)')
    axes[1,0].set_title('PSM1 Tool Tip Velocity')
    axes[1,0].grid(True, alpha=0.3)
    
    # PSM2 velocity
    axes[1,1].plot(df['timestamp'], df['psm2_velocity_magnitude'], 'r-', alpha=0.7, linewidth=0.5)
    axes[1,1].set_ylabel('PSM2 Velocity (m/s)')
    axes[1,1].set_title('PSM2 Tool Tip Velocity')
    axes[1,1].grid(True, alpha=0.3)
    
    # MTM velocities
    axes[2,0].plot(df['timestamp'], df['mtml_velocity_magnitude'], 'g-', alpha=0.7, linewidth=0.5)
    axes[2,0].set_ylabel('MTML Velocity (m/s)')
    axes[2,0].set_xlabel('Time (s)')
    axes[2,0].set_title('Left MTM Velocity')
    axes[2,0].grid(True, alpha=0.3)
    
    axes[2,1].plot(df['timestamp'], df['mtmr_velocity_magnitude'], 'm-', alpha=0.7, linewidth=0.5)
    axes[2,1].set_ylabel('MTMR Velocity (m/s)')
    axes[2,1].set_xlabel('Time (s)')
    axes[2,1].set_title('Right MTM Velocity')
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distances_velocities_over_time.png'), dpi=150)
    plt.close()
    
    # 2. Velocity vs Error correlation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSM1
    valid_psm1 = df[(df['psm1_velocity_magnitude'].notna()) & (df['psm1_dist'] >= 0)]
    if len(valid_psm1) > 0:
        axes[0].scatter(valid_psm1['psm1_velocity_magnitude'], valid_psm1['psm1_dist'], 
                       alpha=0.3, s=1, c='blue')
        axes[0].set_xlabel('Velocity (m/s)')
        axes[0].set_ylabel('Distance Error (m)')
        axes[0].set_title('PSM1: Velocity vs Tracking Error')
        axes[0].grid(True, alpha=0.3)
    
    # PSM2
    valid_psm2 = df[(df['psm2_velocity_magnitude'].notna()) & (df['psm2_dist'] >= 0)]
    if len(valid_psm2) > 0:
        axes[1].scatter(valid_psm2['psm2_velocity_magnitude'], valid_psm2['psm2_dist'], 
                       alpha=0.3, s=1, c='red')
        axes[1].set_xlabel('Velocity (m/s)')
        axes[1].set_ylabel('Distance Error (m)')
        axes[1].set_title('PSM2: Velocity vs Tracking Error')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_error_correlation.png'), dpi=150)
    plt.close()
    
    # 3. Performance by combination and scale
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Group by combination
    if 'combination_id' in df.columns:
        combo_stats = df.groupby('combination_id').agg({
            'psm1_dist': ['mean', 'std'],
            'psm2_dist': ['mean', 'std'],
            'psm1_velocity_magnitude': 'mean',
            'psm2_velocity_magnitude': 'mean'
        }).reset_index()
        
        # Distance by combination
        x = combo_stats['combination_id']
        width = 0.35
        
        axes[0,0].bar(x - width/2, combo_stats['psm1_dist']['mean'], width, 
                     yerr=combo_stats['psm1_dist']['std'], label='PSM1', alpha=0.8, color='blue')
        axes[0,0].bar(x + width/2, combo_stats['psm2_dist']['mean'], width,
                     yerr=combo_stats['psm2_dist']['std'], label='PSM2', alpha=0.8, color='red')
        axes[0,0].set_xlabel('Combination ID')
        axes[0,0].set_ylabel('Mean Distance (m)')
        axes[0,0].set_title(f'Mean Distance by Combination{title_suffix}')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Velocity by combination
        axes[0,1].bar(x - width/2, combo_stats['psm1_velocity_magnitude']['mean'], width, 
                     label='PSM1', alpha=0.8, color='blue')
        axes[0,1].bar(x + width/2, combo_stats['psm2_velocity_magnitude']['mean'], width,
                     label='PSM2', alpha=0.8, color='red')
        axes[0,1].set_xlabel('Combination ID')
        axes[0,1].set_ylabel('Mean Velocity (m/s)')
        axes[0,1].set_title('Mean Velocity by Combination')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Performance by shape
    if 'shape' in df.columns:
        shape_df = df[(df['shape'].notna()) & (df['shape'] != 'dot')]
        if len(shape_df) > 0:
            shape_stats = shape_df.groupby('shape').agg({
                'psm1_dist': 'mean',
                'psm2_dist': 'mean',
                'psm1_velocity_magnitude': 'mean',
                'psm2_velocity_magnitude': 'mean'
            })
            
            shape_stats[['psm1_dist', 'psm2_dist']].plot(kind='bar', ax=axes[1,0], rot=45)
            axes[1,0].set_title(f'Mean Distance by Shape{title_suffix}')
            axes[1,0].set_ylabel('Distance (m)')
            axes[1,0].legend(['PSM1', 'PSM2'])
            
            shape_stats[['psm1_velocity_magnitude', 'psm2_velocity_magnitude']].plot(kind='bar', ax=axes[1,1], rot=45)
            axes[1,1].set_title('Mean Velocity by Shape')
            axes[1,1].set_ylabel('Velocity (m/s)')
            axes[1,1].legend(['PSM1', 'PSM2'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_combination_shape.png'), dpi=150)
    plt.close()
    
    # 4. MTM vs PSM comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left side comparison
    valid_left = df[(df['mtml_velocity_magnitude'].notna()) & (df['psm1_velocity_magnitude'].notna())]
    if len(valid_left) > 0:
        axes[0].scatter(valid_left['mtml_velocity_magnitude'], valid_left['psm1_velocity_magnitude'],
                       alpha=0.3, s=1, c='green')
        axes[0].plot([0, valid_left['mtml_velocity_magnitude'].max()], 
                    [0, valid_left['mtml_velocity_magnitude'].max()], 'k--', alpha=0.5)
        axes[0].set_xlabel('MTML Velocity (m/s)')
        axes[0].set_ylabel('PSM1 Velocity (m/s)')
        axes[0].set_title('Left MTM vs PSM1 Velocity')
        axes[0].grid(True, alpha=0.3)
    
    # Right side comparison
    valid_right = df[(df['mtmr_velocity_magnitude'].notna()) & (df['psm2_velocity_magnitude'].notna())]
    if len(valid_right) > 0:
        axes[1].scatter(valid_right['mtmr_velocity_magnitude'], valid_right['psm2_velocity_magnitude'],
                       alpha=0.3, s=1, c='magenta')
        axes[1].plot([0, valid_right['mtmr_velocity_magnitude'].max()], 
                    [0, valid_right['mtmr_velocity_magnitude'].max()], 'k--', alpha=0.5)
        axes[1].set_xlabel('MTMR Velocity (m/s)')
        axes[1].set_ylabel('PSM2 Velocity (m/s)')
        axes[1].set_title('Right MTM vs PSM2 Velocity')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mtm_psm_velocity_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_report(df, output_path, included_init=True):
    """Generate comprehensive analysis report"""
    report = []
    report.append("=" * 60)
    report.append("COMPREHENSIVE DATA ANALYSIS REPORT WITH MTM/PSM TRACKING")
    report.append("=" * 60)
    report.append("")
    
    # Note about initialization
    if not included_init:
        report.append("NOTE: Initialization dot pattern excluded from analysis")
    else:
        report.append("NOTE: Initialization dot pattern included in analysis")
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS")
    report.append("-" * 40)
    report.append(f"Total data points: {len(df)}")
    
    if len(df) > 0:
        report.append(f"Duration: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
        report.append(f"Sampling rate: {len(df) / (df['timestamp'].max() - df['timestamp'].min()):.1f} Hz")
        if 'participant_id' in df.columns and not df['participant_id'].isna().all():
            report.append(f"Participant: {df['participant_id'].iloc[0]}")
        report.append(f"Total combinations: {df['combination_id'].nunique()}")
        
        if 'clutch_count' in df.columns:
            report.append(f"Total clutch events: {df['clutch_count'].max()}")
        if 'footpedal_count' in df.columns:
            report.append(f"Total footpedal events: {df['footpedal_count'].max()}")
    report.append("")
    
    # PSM Tracking Performance
    report.append("PSM TRACKING PERFORMANCE")
    report.append("-" * 40)
    
    valid_psm1 = df[df['psm1_dist'] >= 0]['psm1_dist']
    valid_psm2 = df[df['psm2_dist'] >= 0]['psm2_dist']
    
    if len(valid_psm1) > 0:
        report.append(f"PSM1 Distance Error:")
        report.append(f"  Mean: {valid_psm1.mean():.4f} m")
        report.append(f"  Std:  {valid_psm1.std():.4f} m")
        report.append(f"  Min:  {valid_psm1.min():.4f} m")
        report.append(f"  Max:  {valid_psm1.max():.4f} m")
        report.append(f"  95th percentile: {valid_psm1.quantile(0.95):.4f} m")
        
    if len(valid_psm2) > 0:
        report.append(f"\nPSM2 Distance Error:")
        report.append(f"  Mean: {valid_psm2.mean():.4f} m")
        report.append(f"  Std:  {valid_psm2.std():.4f} m")
        report.append(f"  Min:  {valid_psm2.min():.4f} m")
        report.append(f"  Max:  {valid_psm2.max():.4f} m")
        report.append(f"  95th percentile: {valid_psm2.quantile(0.95):.4f} m")
    report.append("")
    
    # Velocity Statistics
    report.append("VELOCITY STATISTICS")
    report.append("-" * 40)
    
    psm1_vel = df['psm1_velocity_magnitude'].dropna()
    psm2_vel = df['psm2_velocity_magnitude'].dropna()
    mtml_vel = df['mtml_velocity_magnitude'].dropna()
    mtmr_vel = df['mtmr_velocity_magnitude'].dropna()
    
    if len(psm1_vel) > 0:
        report.append(f"PSM1 Tool Tip Velocity:")
        report.append(f"  Mean: {psm1_vel.mean():.4f} m/s")
        report.append(f"  Max:  {psm1_vel.max():.4f} m/s")
        
    if len(psm2_vel) > 0:
        report.append(f"\nPSM2 Tool Tip Velocity:")
        report.append(f"  Mean: {psm2_vel.mean():.4f} m/s")
        report.append(f"  Max:  {psm2_vel.max():.4f} m/s")
        
    if len(mtml_vel) > 0:
        report.append(f"\nMTML Velocity:")
        report.append(f"  Mean: {mtml_vel.mean():.4f} m/s")
        report.append(f"  Max:  {mtml_vel.max():.4f} m/s")
        
    if len(mtmr_vel) > 0:
        report.append(f"\nMTMR Velocity:")
        report.append(f"  Mean: {mtmr_vel.mean():.4f} m/s")
        report.append(f"  Max:  {mtmr_vel.max():.4f} m/s")
    report.append("")
    
    # Velocity-Error Correlation
    report.append("VELOCITY-ERROR CORRELATION")
    report.append("-" * 40)
    
    valid_psm1_corr = df[(df['psm1_velocity_magnitude'].notna()) & (df['psm1_dist'] >= 0)]
    valid_psm2_corr = df[(df['psm2_velocity_magnitude'].notna()) & (df['psm2_dist'] >= 0)]
    
    if len(valid_psm1_corr) > 10:
        corr1 = valid_psm1_corr['psm1_velocity_magnitude'].corr(valid_psm1_corr['psm1_dist'])
        report.append(f"PSM1 velocity-error correlation: {corr1:.3f}")
        
    if len(valid_psm2_corr) > 10:
        corr2 = valid_psm2_corr['psm2_velocity_magnitude'].corr(valid_psm2_corr['psm2_dist'])
        report.append(f"PSM2 velocity-error correlation: {corr2:.3f}")
    report.append("")
    
    # Performance by Shape
    if 'shape' in df.columns:
        report.append("PERFORMANCE BY SHAPE")
        report.append("-" * 40)
        
        shape_df = df[(df['shape'].notna()) & (df['shape'] != 'dot')]
        if len(shape_df) > 0:
            shapes = shape_df['shape'].astype(str).unique()
            shapes = [s for s in shapes if s != 'dot' and s != 'nan']
            
            for shape in sorted(shapes):
                current_shape_df = shape_df[shape_df['shape'].astype(str) == shape]
                shape_psm1 = current_shape_df[current_shape_df['psm1_dist'] >= 0]['psm1_dist']
                shape_psm2 = current_shape_df[current_shape_df['psm2_dist'] >= 0]['psm2_dist']
                shape_psm1_vel = current_shape_df['psm1_velocity_magnitude'].dropna()
                shape_psm2_vel = current_shape_df['psm2_velocity_magnitude'].dropna()
                
                report.append(f"\n{shape.upper()}:")
                report.append(f"  Data points: {len(current_shape_df)}")
                if len(shape_psm1) > 0:
                    report.append(f"  PSM1 Error: {shape_psm1.mean():.4f} m (std: {shape_psm1.std():.4f})")
                if len(shape_psm2) > 0:
                    report.append(f"  PSM2 Error: {shape_psm2.mean():.4f} m (std: {shape_psm2.std():.4f})")
                if len(shape_psm1_vel) > 0:
                    report.append(f"  PSM1 Velocity: {shape_psm1_vel.mean():.4f} m/s")
                if len(shape_psm2_vel) > 0:
                    report.append(f"  PSM2 Velocity: {shape_psm2_vel.mean():.4f} m/s")
    report.append("")
    
    # Performance by Scale
    if 'scale' in df.columns:
        report.append("PERFORMANCE BY SCALE FACTOR")
        report.append("-" * 40)
        
        scale_df = df[df['scale'].notna()]
        if len(scale_df) > 0:
            for scale in sorted(scale_df['scale'].unique()):
                current_scale_df = scale_df[scale_df['scale'] == scale]
                scale_psm1 = current_scale_df[current_scale_df['psm1_dist'] >= 0]['psm1_dist']
                scale_psm2 = current_scale_df[current_scale_df['psm2_dist'] >= 0]['psm2_dist']
                
                report.append(f"\nScale {scale}:")
                report.append(f"  Data points: {len(current_scale_df)}")
                if len(scale_psm1) > 0:
                    report.append(f"  PSM1 Error: {scale_psm1.mean():.4f} m (std: {scale_psm1.std():.4f})")
                if len(scale_psm2) > 0:
                    report.append(f"  PSM2 Error: {scale_psm2.mean():.4f} m (std: {scale_psm2.std():.4f})")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
        
    print(f"Report saved to {output_path}")
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Analyze comprehensive bag file with MTM/PSM data')
    parser.add_argument('--bag', required=True, help='Path to bag file')
    parser.add_argument('--videos', action='store_true', help='Extract videos')
    parser.add_argument('--data', action='store_true', help='Extract data to CSV')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--exclude-init', action='store_true', 
                       help='Exclude initialization dot from analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    bag_dir = os.path.dirname(args.bag)
    bag_name = os.path.splitext(os.path.basename(args.bag))[0]
    output_base = os.path.join(bag_dir, f"{bag_name}_analysis")
    os.makedirs(output_base, exist_ok=True)
    
    # Extract data
    print("Extracting comprehensive data with MTM/PSM tracking...")
    df = extract_comprehensive_data(args.bag, include_initialization=not args.exclude_init)
    
    if args.exclude_init:
        print(f"Extracted {len(df)} data points (excluding initialization)")
    else:
        print(f"Extracted {len(df)} data points (including initialization)")
    
    if len(df) == 0:
        print("No data extracted. Check bag file contents.")
        return
    
    # Save to CSV
    if args.data:
        csv_path = os.path.join(output_base, 'comprehensive_data_with_mtm_psm.csv')
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        print(f"CSV contains {len(df.columns)} columns including MTM/PSM positions and velocities")
    
    # Extract videos
    if args.videos:
        print("Extracting videos...")
        video_dir = os.path.join(output_base, 'videos')
        extract_videos(args.bag, video_dir)
    
    # Generate plots
    if args.plots:
        print("Generating plots...")
        plots_dir = os.path.join(output_base, 'plots')
        generate_plots(df, plots_dir, included_init=not args.exclude_init)
    
    # Generate report
    if args.report:
        print("Generating report...")
        report_path = os.path.join(output_base, 'analysis_report.txt')
        generate_report(df, report_path, included_init=not args.exclude_init)
    
    print(f"\nAnalysis complete! Results saved to: {output_base}")


if __name__ == "__main__":
    main()