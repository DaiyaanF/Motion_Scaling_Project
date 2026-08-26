import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


# ROS2 bags are a directory (metadata.yaml + one or more .db3 sqlite files),
# not a single .bag file like ROS1. bag_path below should point at that
# directory (the one produced by bezier_collector.py's SequentialWriter).
_MSG_TYPES = {
    '/session_metadata': String,
    '/bezier_events': String,
    '/bezier_comprehensive': String,
    '/camera_left': CompressedImage,
    '/camera_right': CompressedImage,
}


class SlidingWindowBagReader:
    """
    Reader for sliding window experiment ROS2 bag files.

    Parses experiment metadata, tracking data, events, and video from recorded
    bag files. Provides export functionality for offline analysis.
    """

    def __init__(self, bag_path):
        self.bag_path = bag_path

        self.metadata = {}
        self.events = []
        self.comprehensive_data = []
        self.trajectory_points = None
        self.video_frame_counts = {'left': 0, 'right': 0}

        self._parse_bag()

    def _open_reader(self):
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)
        return reader

    def _parse_bag(self):
        """Parse all topics from bag file (single pass, cached for repeated use below)."""
        print(f"Reading: {self.bag_path}")

        reader = self._open_reader()
        # Cache raw (topic, deserialized_msg, stamp_ns) so extract_video()/
        # extract_stereo_video() below don't need to reopen+reread the whole bag.
        self._all_records = []

        while reader.has_next():
            topic, data, stamp_ns = reader.read_next()
            msg_type = _MSG_TYPES.get(topic)
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)
            self._all_records.append((topic, msg, stamp_ns))

            if topic == '/session_metadata':
                data_dict = json.loads(msg.data)
                self.metadata.update(data_dict)
                if data_dict.get('event') == 'trajectory_points':
                    self.trajectory_points = np.array(data_dict['points'])

            elif topic == '/bezier_events':
                self.events.append(json.loads(msg.data))

            elif topic == '/bezier_comprehensive':
                self.comprehensive_data.append(json.loads(msg.data))

            elif topic == '/camera_left':
                self.video_frame_counts['left'] += 1
            elif topic == '/camera_right':
                self.video_frame_counts['right'] += 1

        print(f"  Metadata keys: {list(self.metadata.keys())}")
        print(f"  Events: {len(self.events)}")
        print(f"  Comprehensive records: {len(self.comprehensive_data)}")
        if self.trajectory_points is not None:
            print(f"  Trajectory points: {len(self.trajectory_points)}")
        print(f"  Video frames: left={self.video_frame_counts['left']}, right={self.video_frame_counts['right']}")

    def get_summary(self):
        """Get experiment summary dictionary."""
        return {
            'participant_id': self.metadata.get('participant_id'),
            'num_segments': self.metadata.get('num_segments'),
            'total_points': self.metadata.get('total_points'),
            'total_iterations': self.metadata.get('total_iterations'),
            'iterations_completed': self.metadata.get('iterations_completed', 0),
            'quarter_size': self.metadata.get('quarter_size'),
            'fixed_cloud_size': self.metadata.get('fixed_cloud_size'),
            'motion_scaling': self.metadata.get('motion_scaling'),
            'video_frames_left': self.video_frame_counts['left'],
            'video_frames_right': self.video_frame_counts['right'],
        }

    def get_comprehensive_df(self):
        """Convert comprehensive tracking data to DataFrame."""
        if not self.comprehensive_data:
            return pd.DataFrame()

        flat_data = []
        for record in self.comprehensive_data:
            flat = {
                'timestamp': record.get('timestamp'),
                'iteration': record.get('current_iteration'),
                'progress_idx': record.get('progress_idx'),
                'closest_idx': record.get('closest_idx'),
                'completion_pct': record.get('completion_pct'),
                'current_segment': record.get('current_segment'),
                'closest_dist': record.get('closest_dist'),
                'visible_start': record.get('visible_start'),
                'visible_end': record.get('visible_end'),
                'psm2_velocity_magnitude': record.get('psm2_velocity_magnitude'),
            }

            if record.get('psm2_position'):
                flat['psm2_x'] = record['psm2_position'][0]
                flat['psm2_y'] = record['psm2_position'][1]
                flat['psm2_z'] = record['psm2_position'][2]

            if record.get('psm2_position_smoothed'):
                flat['psm2_smooth_x'] = record['psm2_position_smoothed'][0]
                flat['psm2_smooth_y'] = record['psm2_position_smoothed'][1]
                flat['psm2_smooth_z'] = record['psm2_position_smoothed'][2]

            scaling = record.get('motion_scaling', {})
            flat['scale'] = scaling.get('current_scale')

            flat_data.append(flat)

        return pd.DataFrame(flat_data)

    def get_events_df(self):
        """Convert events to DataFrame."""
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame(self.events)

    def extract_video(self, output_dir=None, side='both', as_video=True, fps=10):
        """
        Extract video frames from bag file.

        Args:
            output_dir: Output directory path
            side: 'left', 'right', or 'both'
            as_video: If True, create MP4 file. If False, save individual frames.
            fps: Output video frame rate
        """
        if output_dir is None:
            output_dir = Path(self.bag_path).parent
        output_dir = Path(output_dir)
        base_name = Path(self.bag_path).stem

        cameras = []
        if side in ['left', 'both']:
            cameras.append(('left', '/camera_left'))
        if side in ['right', 'both']:
            cameras.append(('right', '/camera_right'))

        for cam_name, topic in cameras:
            print(f"Extracting {cam_name} camera...")

            frames = []
            timestamps = []

            for topic_name, msg, stamp_ns in self._all_records:
                if topic_name != topic:
                    continue
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frames.append(frame)
                    timestamps.append(stamp_ns / 1e9)

            if not frames:
                print(f"  No frames found for {cam_name}")
                continue

            print(f"  Found {len(frames)} frames")

            if as_video:
                video_path = output_dir / f"{base_name}_{cam_name}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                for frame in frames:
                    writer.write(frame)
                writer.release()
                print(f"  Saved: {video_path}")
            else:
                frames_dir = output_dir / f"{base_name}_{cam_name}_frames"
                frames_dir.mkdir(exist_ok=True)
                for i, (frame, ts) in enumerate(zip(frames, timestamps)):
                    frame_path = frames_dir / f"frame_{i:05d}_{ts:.3f}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                print(f"  Saved {len(frames)} frames to: {frames_dir}")

    def extract_stereo_video(self, output_dir=None, fps=10):
        """
        Extract side-by-side stereo video.

        Args:
            output_dir: Output directory path
            fps: Output video frame rate
        """
        if output_dir is None:
            output_dir = Path(self.bag_path).parent
        output_dir = Path(output_dir)
        base_name = Path(self.bag_path).stem

        print("Extracting stereo video...")

        left_frames = {}
        right_frames = {}

        for topic, msg, stamp_ns in self._all_records:
            if topic not in ('/camera_left', '/camera_right'):
                continue
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                ts = round(stamp_ns / 1e9, 2)
                if topic == '/camera_left':
                    left_frames[ts] = frame
                else:
                    right_frames[ts] = frame

        common_times = sorted(set(left_frames.keys()) & set(right_frames.keys()))
        print(f"  Found {len(common_times)} matched stereo pairs")

        if not common_times:
            print("  No matching stereo pairs found")
            return

        video_path = output_dir / f"{base_name}_stereo.mp4"
        h, w = left_frames[common_times[0]].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w * 2, h))

        for ts in common_times:
            stereo = np.hstack([left_frames[ts], right_frames[ts]])
            writer.write(stereo)

        writer.release()
        print(f"  Saved: {video_path}")

    def save_csv(self, output_dir=None):
        """Export data to CSV files."""
        if output_dir is None:
            output_dir = Path(self.bag_path).parent
        output_dir = Path(output_dir)
        base_name = Path(self.bag_path).stem

        df = self.get_comprehensive_df()
        if not df.empty:
            csv_path = output_dir / f"{base_name}_comprehensive.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

        events_df = self.get_events_df()
        if not events_df.empty:
            csv_path = output_dir / f"{base_name}_events.csv"
            events_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

        meta_path = output_dir / f"{base_name}_metadata.json"
        with open(meta_path, 'w') as f:
            meta_clean = {k: v for k, v in self.metadata.items() if k != 'points'}
            json.dump(meta_clean, f, indent=2)
        print(f"Saved: {meta_path}")

    def plot_progress(self, save_path=None):
        """Generate visualization plots for experiment data."""
        df = self.get_comprehensive_df()
        if df.empty:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        iterations = df['iteration'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(iterations)))

        # Progress over time
        ax = axes[0, 0]
        for i, iteration in enumerate(iterations):
            iter_df = df[df['iteration'] == iteration]
            time = iter_df['timestamp'] - iter_df['timestamp'].iloc[0]
            ax.plot(time, iter_df['completion_pct'] * 100,
                   label=f"Iter {iteration+1} (scale={iter_df['scale'].iloc[0]:.1f}x)",
                   color=colors[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Completion (%)')
        ax.set_title('Progress Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tracking error
        ax = axes[0, 1]
        for i, iteration in enumerate(iterations):
            iter_df = df[df['iteration'] == iteration]
            time = iter_df['timestamp'] - iter_df['timestamp'].iloc[0]
            dist_mm = iter_df['closest_dist'] * 1000
            ax.plot(time, dist_mm, label=f"Iter {iteration+1}", color=colors[i], alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to Trajectory (mm)')
        ax.set_title('Tracking Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity
        ax = axes[1, 0]
        for i, iteration in enumerate(iterations):
            iter_df = df[df['iteration'] == iteration]
            time = iter_df['timestamp'] - iter_df['timestamp'].iloc[0]
            ax.plot(time, iter_df['psm2_velocity_magnitude'],
                   label=f"Iter {iteration+1}", color=colors[i], alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('PSM Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3D trajectory
        ax = axes[1, 1]
        ax.remove()
        ax = fig.add_subplot(2, 2, 4, projection='3d')

        if self.trajectory_points is not None:
            ax.plot(self.trajectory_points[:, 0],
                   self.trajectory_points[:, 1],
                   self.trajectory_points[:, 2],
                   'k-', alpha=0.3, linewidth=1, label='Target')

        for i, iteration in enumerate(iterations):
            iter_df = df[df['iteration'] == iteration]
            if 'psm2_smooth_x' in iter_df.columns:
                ax.plot(iter_df['psm2_smooth_x'],
                       iter_df['psm2_smooth_y'],
                       iter_df['psm2_smooth_z'],
                       color=colors[i], alpha=0.7, label=f"Iter {iteration+1}")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectories')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        else:
            plt.show()

    def analyze_iterations(self):
        """Print analysis summary for each iteration."""
        df = self.get_comprehensive_df()
        events_df = self.get_events_df()

        print("\nIteration Analysis")
        print("-" * 40)

        for iteration in df['iteration'].unique():
            iter_df = df[df['iteration'] == iteration]

            duration = iter_df['timestamp'].iloc[-1] - iter_df['timestamp'].iloc[0]
            mean_dist = iter_df['closest_dist'].mean() * 1000
            max_dist = iter_df['closest_dist'].max() * 1000
            mean_vel = iter_df['psm2_velocity_magnitude'].mean()
            scale = iter_df['scale'].iloc[0] if 'scale' in iter_df else None
            final_progress = iter_df['completion_pct'].iloc[-1] * 100

            print(f"\nIteration {iteration + 1}:")
            print(f"  Scale: {scale:.2f}x" if scale else "  Scale: N/A")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Final progress: {final_progress:.1f}%")
            print(f"  Mean tracking error: {mean_dist:.2f}mm")
            print(f"  Max tracking error: {max_dist:.2f}mm")
            print(f"  Mean velocity: {mean_vel:.4f} m/s")

    def close(self):
        """No-op: rosbag2_py's SequentialReader has no explicit close/handle to release here."""
        pass


def main():
    parser = argparse.ArgumentParser(description='Read and analyze sliding window experiment bags')
    parser.add_argument('bag_path', help='Path to bag directory (ROS2 bag: contains metadata.yaml + .db3)')
    parser.add_argument('--csv', action='store_true', help='Export to CSV')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--analyze', action='store_true', help='Print analysis')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    parser.add_argument('--video', action='store_true', help='Extract video')
    parser.add_argument('--video-side', choices=['left', 'right', 'both', 'stereo'],
                       default='both', help='Camera selection')
    parser.add_argument('--video-fps', type=int, default=10, help='Video FPS')
    parser.add_argument('--frames', action='store_true', help='Save as individual frames')

    args = parser.parse_args()

    reader = SlidingWindowBagReader(args.bag_path)

    print("\nExperiment Summary")
    print("-" * 40)
    summary = reader.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.analyze:
        reader.analyze_iterations()

    if args.csv:
        reader.save_csv(args.output_dir)

    if args.plot:
        save_path = None
        if args.output_dir:
            base_name = Path(args.bag_path).stem
            save_path = Path(args.output_dir) / f"{base_name}_plots.png"
        reader.plot_progress(save_path)

    if args.video:
        if args.video_side == 'stereo':
            reader.extract_stereo_video(args.output_dir, fps=args.video_fps)
        else:
            reader.extract_video(args.output_dir, side=args.video_side,
                               as_video=not args.frames, fps=args.video_fps)

    reader.close()


if __name__ == "__main__":
    main()

"""
Usage (ROS2 version):
    python read_bezier_bag_v2.py <bag_dir> [options]

Note: bag_path is now a directory (ROS2 bag format: metadata.yaml + N .db3
sqlite files), not a single .bag file as in the ROS1 version.

Examples:
    # Print analysis summary
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --analyze

    # Export to CSV
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --csv

    # Generate plots
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --plot

    # Extract video (both cameras)
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --video

    # Extract side-by-side stereo video
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --video --video-side stereo

    # Extract individual frames instead of video
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --video --frames

    # Full analysis with all outputs
    python read_bezier_bag_v2.py ~/data/sliding_20251213_180000 --analyze --csv --plot --video --video-side stereo

Arguments:
    bag_path               Path to ROS2 bag directory (required)
    --analyze              Print iteration analysis summary
    --csv                  Export data to CSV files
    --plot                 Generate visualization plots
    --output-dir           Output directory for exported files
    --video                Extract video from bag
    --video-side           Camera selection: left, right, both, stereo (default: both)
    --video-fps            Output video frame rate (default: 10)
    --frames               Save as individual frames instead of video
"""
