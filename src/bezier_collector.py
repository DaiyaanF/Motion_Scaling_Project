"""
Sliding Window Trajectory Tracking Experiment (ROS 2 / rclpy version)

Implements a trajectory tracking task for studying human adaptation to motion
scaling in teleoperated surgical robotics. Users follow a Bezier curve path
displayed as a sliding window of points around their current
position.

This script provides:
Sliding window visualization with fixed point cloud size
Rate-limited progress updates to prevent visual jumps
Configurable motion scaling ratios across iterations
Records PSM/MTM kinematics, tracking metrics, and stereo video

Dependencies: frame_transform_manager.py, bezier_trajectory_cubic.py

Usage:
    python bezier_collector.py --participant P001 --segments 4 --fixed "0.5,1.0,2.0"

IMPORTANT (ROS2 conversion notes):
- Data is now recorded as a ROS2 bag DIRECTORY (metadata.yaml + .db3), not a
  single .bag file. Use src/read_bezier_bag_v2.py (also converted) to read it.
- The ROS1 version dynamically registered '/ambf/env/World/trajectory' into
  AMBF's global '/ambf/env/World/point_cloud_topics' rosparam so the point
  cloud would render without any ADF changes. ROS2 has no global parameter
  server (parameters are per-node), and AMBF's own ROS2 examples
  (point_cloud_example.py) instead expect the PointCloud object to be
  predefined in the ADF/world file. That dynamic registration has been
  DROPPED here (it was already best-effort/wrapped in try-except in the
  original) -- you must ensure a PointCloud-type body publishing on
  '/ambf/env/World/trajectory' is declared in your world ADF, or add the
  equivalent parameter-service call yourself once you know the AMBF World
  node's exact ROS2 node name.
"""

import time
import json
import numpy as np
import sys
import os
import threading
from datetime import datetime
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.serialization import serialize_message
import rosbag2_py

from std_msgs.msg import String, Float32
from sensor_msgs.msg import PointCloud, CompressedImage
from geometry_msgs.msg import Point32, PoseStamped
from ambf_msgs.msg import RigidBodyState

from frame_transform_manager import get_frame_manager
from bezier_trajectory_cubic import CubicBezierTrajectoryGenerator

# Default trajectory workspace, sized to guarantee the whole box stays inside
# cameraL/cameraR's fixed frustum (ADF/world/world_stereo.yaml: 55 deg vertical
# FOV, camera at z=-0.05 in CameraFrame) so the camera never has to move mid-trial,
# AND pulled back in close to z=-0.1..-0.2 -- the depth range the team's original
# (pre-camera-fit) defaults used, which is presumably where the PSM tooltip
# naturally reaches in CameraFrame-relative coordinates. The previous version of
# this box (z=-0.17..-0.35) was FOV-safe but too deep for the PSM to reach.
# At the box's nearest wall (0.08m in front of the camera, the tightest point
# since the frustum is narrowest there), the vertical FOV allows +/-0.0416m;
# +/-0.03m keeps some margin. x/y reuse that same half-extent for a symmetric
# box. If this still isn't reachable, the fix is to narrow/shift z further based
# on where psm_pos actually sits during a comfortable teleop pose (see the
# 'Progress:'/dist console log at runtime), not to widen the FOV assumption.
DEFAULT_BOUNDING_BOX = {
    'x_min': -0.03, 'x_max': 0.03,
    'y_min': -0.03, 'y_max': 0.03,
    'z_min': -0.25, 'z_max': -0.13,
}
DEFAULT_START_POSITION = [0.0, 0.0, -0.19]  # box center


class MotionScalingController:
    """
    Controls motion scaling ratio during teleoperation.

    Supports two modes:
    - fixed: Cycles through predetermined scaling ratios across iterations
    - adaptive: Adjusts scaling based on tracking error (experimental)
    """

    def __init__(self, mode='fixed', fixed_ratios=None, initial_scale=1.0,
                 adaptation_rate=0.1, window_size=10, logger=None):
        self.mode = mode
        self.fixed_ratios = fixed_ratios or [1.0, 2.0, 3.0]
        self.current_ratio_idx = 0
        self.iteration_count = 0
        self.current_scale = self.fixed_ratios[0] if mode == 'fixed' else initial_scale
        self.initial_scale = initial_scale
        self.adaptation_rate = adaptation_rate
        self.error_history = deque(maxlen=window_size)
        self.last_error = None
        self.logger = logger

    def get_current_scale(self):
        return self.current_scale

    def on_iteration_complete(self):
        if self.mode == 'fixed':
            self.iteration_count += 1
            old_scale = self.current_scale
            self.current_ratio_idx = (self.current_ratio_idx + 1) % len(self.fixed_ratios)
            self.current_scale = self.fixed_ratios[self.current_ratio_idx]
            if self.logger:
                self.logger.info(f"Scale: {old_scale}x -> {self.current_scale}x")

    def update_adaptive(self, error):
        if self.mode != 'adaptive':
            return
        self.error_history.append(error)
        if len(self.error_history) < 2:
            self.last_error = error
            return

        avg_error = np.mean(self.error_history)
        error_rate = avg_error - (self.last_error or avg_error)
        self.last_error = avg_error

        adjustment = -np.clip(error_rate * 200, -1, 1) * self.adaptation_rate * self.current_scale
        new_scale = np.clip(self.current_scale + adjustment,
                           self.current_scale * 0.8, self.current_scale * 1.2)
        new_scale = np.clip(new_scale, 0.1, 5.0)

        if abs(new_scale - self.current_scale) / self.current_scale > 0.01:
            self.current_scale = new_scale

    def get_info(self):
        info = {'mode': self.mode, 'current_scale': float(self.current_scale)}
        if self.mode == 'fixed':
            info.update({
                'fixed_ratios': self.fixed_ratios,
                'current_ratio_idx': self.current_ratio_idx,
                'iteration_count': self.iteration_count
            })
        return info


class SlidingWindowExperiment(Node):
    """
    Trajectory tracking experiment with sliding window visualization.

    The trajectory is pre-sampled into a continuous array of points. Visualization
    reveals the path in quarter-steps: Q1+Q2 are shown at the start, Q2+Q3 once
    Q1 is complete, then Q3+Q4 once Q2 is complete (see _get_display_window).
    Progress updates are rate-limited to prevent visual jumps.

    The point cloud maintains a fixed size to ensure consistent AMBF rendering.
    """

    def __init__(self, participant_id, num_segments=10, scaling_mode='fixed',
                 fixed_ratios=None, initial_scale=1.0, adaptation_rate=0.1,
                 bounding_box=None, start_position=None,
                 max_forward_speed=10, max_backward_speed=3,
                 points_per_segment=200, proximity_threshold=0.003,
                 search_ahead=50, search_behind=20):

        super().__init__('sliding_window_experiment')

        # Configuration
        self.participant_id = participant_id
        self.num_segments = num_segments
        self.arc_length = 0.05
        self.boundary_size = 0.15
        self.bounding_box = bounding_box or DEFAULT_BOUNDING_BOX
        self.start_position = np.array(start_position or DEFAULT_START_POSITION)
        self.video_rate = 10.0

        # Progress parameters
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.points_per_segment = points_per_segment
        self.proximity_threshold = proximity_threshold
        # quarter_size/fixed_cloud_size depend on total_points, set after
        # _generate_trajectory() runs below
        self.quarter_size = 0
        self.fixed_cloud_size = 0

        # Search constraints prevent jumping to distant trajectory points
        self.search_ahead = search_ahead
        self.search_behind = search_behind

        # Controllers
        self.scaling = MotionScalingController(
            scaling_mode, fixed_ratios, initial_scale, adaptation_rate, logger=self.get_logger())
        self.total_iterations = len(fixed_ratios) if scaling_mode == 'fixed' else 1

        # Thread synchronization
        self.bag_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.bag_closed = False

        # Start spinning this node in the background immediately. Everything
        # below (FrameTransformManager's blocking wait, subscriptions, timers)
        # depends on the node actually being spun to receive callbacks, and
        # rclpy's executor picks up entities added to an already-spinning
        # node dynamically, so it's safe to start this before setup finishes.
        self._spin_thread = threading.Thread(target=self._spin_forever, daemon=True)
        self._spin_thread.start()

        # Frame manager for coordinate transforms
        self.get_logger().info("Initializing frame transform manager...")
        self.frame_mgr = get_frame_manager(self, 'CameraFrame', '/ambf/env/phantom')
        if not self.frame_mgr.is_ready():
            self.get_logger().error("Frame manager failed. Check AMBF simulation.")
            sys.exit(1)

        # Trajectory generator
        self.bezier_gen = CubicBezierTrajectoryGenerator(self.boundary_size, self.bounding_box)

        # Trajectory data
        self.all_segments = []
        self.trajectory_points = None
        self.total_points = 0
        self.segment_boundaries = []

        # PSM state
        self.psm_pos = None
        self.psm_pos_world = None
        self.psm_pos_smoothed = None
        self.psm_orientation = None
        self.psm_velocity = None
        self.psm_vel_mag = None
        self.position_buffer = deque(maxlen=5)

        # MTM state
        self.mtm_position = None
        self.mtm_orientation = None

        # Experiment state
        self.progress_idx = 0
        self.closest_idx = 0
        self.closest_dist = float('inf')
        self.init_complete = False
        self.waiting_for_start = False
        self.left_start_area = True
        self.current_iter = 0
        self.last_visible_start = -1
        self.last_visible_end = -1
        self.frames_since_start = 0

        # Timing
        self.last_video_time = {'left': 0, 'right': 0}
        self.last_status_time = 0
        self.last_progress_log_time = 0
        self.last_publish_time = 0

        # Initialize
        self._setup_bag()
        self._setup_ros()
        self._generate_trajectory()
        # Quarter-stepped display window: reveals two adjacent quarters of the
        # full trajectory at a time (Q1+Q2 -> Q2+Q3 -> Q3+Q4). fixed_cloud_size
        # is sized to the largest of those windows (Q3+Q4, which absorbs the
        # remainder from total_points % 4) so every published cloud pads up to
        # the same point count for stable AMBF rendering.
        self.quarter_size = max(1, self.total_points // 4)
        self.fixed_cloud_size = self.total_points - 2 * self.quarter_size
        self._record_metadata()
        self._publish_start_dot()

        # Timers (rclpy timer callbacks take no arguments, unlike rospy.Timer)
        self.create_timer(1.0 / 30.0, self._scale_timer_cb)
        self.create_timer(0.1, self._republish_timer_cb)
        self.create_timer(1.0 / 30.0, self._main_loop)

        self._log_config()

    def _spin_forever(self):
        rclpy.spin(self)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _setup_bag(self):
        """Initialize ROS2 bag for data recording."""
        data_dir = os.path.expanduser(f"~/intel_motion_scaling/bezier_data/{self.participant_id}")
        os.makedirs(data_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ROS2 bags are a directory (metadata.yaml + .db3 files), not a single file.
        self.bag_path = os.path.join(data_dir, f"sliding_{timestamp}")

        self.bag_writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.bag_writer.open(storage_options, converter_options)

        topic_types = {
            '/session_metadata': 'std_msgs/msg/String',
            '/bezier_events': 'std_msgs/msg/String',
            '/bezier_comprehensive': 'std_msgs/msg/String',
            '/camera_left': 'sensor_msgs/msg/CompressedImage',
            '/camera_right': 'sensor_msgs/msg/CompressedImage',
        }
        for i, (name, type_str) in enumerate(topic_types.items()):
            self.bag_writer.create_topic(rosbag2_py.TopicMetadata(
                id=i, name=name, type=type_str, serialization_format='cdr'))

    def _setup_ros(self):
        """Configure ROS publishers and subscribers."""
        # NOTE: the ROS1 version dynamically registered this topic into AMBF's
        # global point_cloud_topics rosparam here. See module docstring --
        # that mechanism doesn't exist in ROS2; ensure the ADF predefines it.

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.traj_pub = self.create_publisher(PointCloud, '/ambf/env/World/trajectory', 10)
        self.traj_size_pub = self.create_publisher(Float32, '/ambf/env/World/trajectory/radius', latched_qos)
        # Red single-point marker at progress_idx -- visual-only "next point to
        # head toward"; does not gate progress (see _update_progress).
        self.target_pub = self.create_publisher(PointCloud, '/ambf/env/World/target_marker', 10)
        self.target_size_pub = self.create_publisher(Float32, '/ambf/env/World/target_marker/radius', latched_qos)
        self.dist_pub = self.create_publisher(Float32, '/ambf/env/psm2/trajectory_dist', 1)
        self.scale_pub = self.create_publisher(Float32, '/ambf/teleop/scale', latched_qos)

        self.create_subscription(RigidBodyState, '/ambf/env/psm2/toolyawlink/State', self._psm_cb, 1)
        self.create_subscription(
            CompressedImage, '/ambf/env/cameras/cameraL/ImageData/compressed',
            lambda m: self._camera_cb(m, 'left'), 1)
        self.create_subscription(
            CompressedImage, '/ambf/env/cameras/cameraR/ImageData/compressed',
            lambda m: self._camera_cb(m, 'right'), 1)

        # Give discovery a moment before checking for MTM topics (ROS2 graph
        # discovery on a just-created node can lag briefly).
        time.sleep(1.0)
        self._setup_mtm()

    def _setup_mtm(self):
        """Subscribe to MTM if available."""
        topics = ['/dvrk/MTMR/position_cartesian_current', '/dvrk/MTML/position_cartesian_current']
        available = [t[0] for t in self.get_topic_names_and_types()]
        for topic in topics:
            if topic in available:
                self.create_subscription(PoseStamped, topic, self._mtm_cb, 1)
                self.get_logger().info(f"MTM: {topic}")
                return
        self.get_logger().info("MTM not found")

    def _generate_trajectory(self):
        """Generate Bezier trajectory and sample into continuous point array."""
        self.get_logger().info("Generating trajectory...")

        last_pt = self.start_position.copy()
        all_points = []
        self.segment_boundaries = [0]

        for i in range(self.num_segments - 1):
            seg = self.bezier_gen.generate_segment(last_pt, self.arc_length)
            seg.compute_metrics()
            self.all_segments.append(seg)

            points = seg.sample_points(self.points_per_segment)
            all_points.append(points[:-1])

            self.segment_boundaries.append(len(all_points[-1]) + self.segment_boundaries[-1])
            self.get_logger().info(f"  Seg {i}: arc={seg.arc_length*1000:.1f}mm, curv_max={seg.curvature_max:.1f}")
            last_pt = seg.P3

        seg = self.bezier_gen.generate_segment(last_pt, self.arc_length, end_point=self.start_position)
        seg.compute_metrics()
        self.all_segments.append(seg)

        points = seg.sample_points(self.points_per_segment)
        all_points.append(points)
        self.get_logger().info(f"  Seg {self.num_segments-1} (return): arc={seg.arc_length*1000:.1f}mm")

        self.trajectory_points = np.vstack(all_points)
        self.total_points = len(self.trajectory_points)

        self.get_logger().info(f"Generated trajectory: {self.num_segments} segments, {self.total_points} points")

    def _record_metadata(self):
        """Record session metadata to bag."""
        self._record_to_bag('session_start', {
            'event': 'session_start',
            'participant_id': self.participant_id,
            'num_segments': self.num_segments,
            'total_iterations': self.total_iterations,
            'total_points': self.total_points,
            'arc_length': self.arc_length,
            'bounding_box': self.bounding_box,
            'start_position': self.start_position.tolist(),
            'quarter_size': self.quarter_size,
            'fixed_cloud_size': self.fixed_cloud_size,
            'max_forward_speed': self.max_forward_speed,
            'max_backward_speed': self.max_backward_speed,
            'search_ahead': self.search_ahead,
            'search_behind': self.search_behind,
            'proximity_threshold': self.proximity_threshold,
            'motion_scaling': self.scaling.get_info(),
        })

        self._record_to_bag('segment_definitions', {
            'event': 'segment_definitions',
            'segment_boundaries': self.segment_boundaries,
            'all_segments': [{
                'id': i,
                'P0': s.P0.tolist(), 'P1': s.P1.tolist(),
                'P2': s.P2.tolist(), 'P3': s.P3.tolist(),
                'arc_length': float(s.arc_length),
                'curvature_max': float(s.curvature_max),
                'curvature_avg': float(s.curvature_avg),
            } for i, s in enumerate(self.all_segments)]
        })

        self._record_to_bag('trajectory_points', {
            'event': 'trajectory_points',
            'points': self.trajectory_points.tolist(),
        })

    def _psm_cb(self, msg):
        """Process PSM state update."""
        pos_ref, pos_world = self.frame_mgr.compute_tool_tip_positions(msg)

        q = msg.pose.orientation
        self.psm_orientation = {'x': q.x, 'y': q.y, 'z': q.z, 'w': q.w}

        l = msg.twist.linear
        self.psm_velocity = {'x': l.x, 'y': l.y, 'z': l.z}
        self.psm_vel_mag = np.linalg.norm([l.x, l.y, l.z])

        self.psm_pos = pos_ref
        self.psm_pos_world = pos_world

        if pos_ref is not None:
            self.position_buffer.append(pos_ref.copy())
            if len(self.position_buffer) >= 3:
                self.psm_pos_smoothed = np.median(list(self.position_buffer), axis=0)
            else:
                self.psm_pos_smoothed = pos_ref.copy()

    def _camera_cb(self, msg, side):
        """Record camera frames at configured rate."""
        now = self._now_sec()
        if now - self.last_video_time[side] >= 1.0 / self.video_rate:
            topic = '/camera_left' if side == 'left' else '/camera_right'
            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            with self.bag_lock:
                if not self.bag_closed:
                    self.bag_writer.write(topic, serialize_message(msg), stamp_ns)
            self.last_video_time[side] = now

    def _mtm_cb(self, msg):
        """Process MTM state update."""
        p, q = msg.pose.position, msg.pose.orientation
        self.mtm_position = {'x': p.x, 'y': p.y, 'z': p.z}
        self.mtm_orientation = {'x': q.x, 'y': q.y, 'z': q.z, 'w': q.w}

    def _scale_timer_cb(self):
        """Publish motion scaling at 30Hz."""
        self.scale_pub.publish(Float32(data=self.scaling.get_current_scale()))

    def _republish_timer_cb(self):
        """Republish trajectory at 10Hz for rendering stability."""
        with self.state_lock:
            if not self.init_complete or self.waiting_for_start:
                return
            progress = self.progress_idx
        self._publish_window(progress, force=True)

    def _main_loop(self):
        """Main experiment loop running at 30Hz."""
        now = self._now_sec()

        if self.psm_pos is None:
            if now - self.last_status_time > 2.0:
                self.get_logger().warn("Waiting for PSM2 data...")
                self.last_status_time = now
            return

        with self.state_lock:
            waiting = self.waiting_for_start
            init = self.init_complete
            progress = self.progress_idx

        dist_to_start = np.linalg.norm(self.psm_pos - self.start_position)

        if waiting:
            if dist_to_start > self.proximity_threshold * 3:
                self.left_start_area = True

            if now - self.last_status_time > 0.5:
                if not self.left_start_area:
                    self.get_logger().info(f"Move away from DOT first... (dist: {dist_to_start*1000:.1f}mm)")
                else:
                    self.get_logger().info(
                        f"Touch DOT for iteration {self.current_iter + 1}/{self.total_iterations} "
                        f"(dist: {dist_to_start*1000:.1f}mm, scale: {self.scaling.get_current_scale()}x)")
                self.last_status_time = now

            if self.left_start_area and dist_to_start < self.proximity_threshold:
                with self.state_lock:
                    self.waiting_for_start = False
                    self.progress_idx = 0
                    self.closest_idx = 0
                    self.frames_since_start = 0

                self.get_logger().info(f"Starting iteration {self.current_iter + 1}/{self.total_iterations}")

                for _ in range(3):
                    self._publish_window(0, force=True)
                    time.sleep(0.03)

                self._record_to_bag('event', {'event': 'iteration_start', 'iteration': self.current_iter})
            return

        if not init:
            if now - self.last_status_time > 0.5:
                self.get_logger().info(f"Touch DOT to begin (dist: {dist_to_start*1000:.1f}mm)")
                self.last_status_time = now

            if dist_to_start < self.proximity_threshold:
                with self.state_lock:
                    self.init_complete = True
                    self.progress_idx = 0
                    self.closest_idx = 0
                    self.frames_since_start = 0

                self.get_logger().info("Experiment started")
                self._publish_window(0, force=True)
                self._record_to_bag('event', {'event': 'experiment_start', 'iteration': 0})
            return

        self.frames_since_start += 1
        new_progress, window_changed = self._update_progress()

        if self.scaling.mode == 'adaptive' and self.closest_dist < float('inf'):
            self.scaling.update_adaptive(self.closest_dist)

        if now - self.last_progress_log_time > 2.0:
            pct = (new_progress / self.total_points) * 100
            seg_idx = self._get_segment_for_point(new_progress)
            search_start = max(0, new_progress - self.search_behind)
            search_end = min(self.total_points, new_progress + self.search_ahead)

            if self.closest_dist > self.proximity_threshold * 3:
                status = "OFF TRACK"
            elif self.closest_idx == search_end - 1:
                status = "AT EDGE (fwd)"
            elif self.closest_idx == search_start:
                status = "AT EDGE (back)"
            else:
                status = "OK"

            self.get_logger().info(
                f"Progress: {new_progress}/{self.total_points} ({pct:.1f}%) | "
                f"Seg {seg_idx} | dist: {self.closest_dist*1000:.2f}mm | {status}")
            self.last_progress_log_time = now

        force_publish = self.frames_since_start < 30
        should_publish = window_changed or force_publish or (self.frames_since_start % 3 == 0)

        if should_publish:
            self._publish_window(new_progress, force=force_publish)

        if new_progress >= self.total_points - 1:
            self._iteration_complete()
            return

        self._record_comprehensive()

    def _update_progress(self):
        """
        Update progress index based on PSM position.

        Searches for closest trajectory point within a constrained window around
        current progress. Rate-limits progress changes to prevent visual jumps.

        Returns:
            tuple: (new_progress_index, window_changed_flag)
        """
        pos = self.psm_pos_smoothed
        if pos is None:
            pos = self.psm_pos
        if pos is None:
            return self.progress_idx, False

        with self.state_lock:
            current_progress = self.progress_idx

        search_start = max(0, current_progress - self.search_behind)
        search_end = min(self.total_points, current_progress + self.search_ahead)

        if search_start >= search_end:
            return current_progress, False

        search_points = self.trajectory_points[search_start:search_end]
        distances = np.linalg.norm(search_points - pos, axis=1)
        local_closest_idx = int(np.argmin(distances))
        self.closest_dist = distances[local_closest_idx]
        self.closest_idx = search_start + local_closest_idx

        self.dist_pub.publish(Float32(data=self.closest_dist))

        if self.closest_dist > self.proximity_threshold * 3:
            return current_progress, False

        with self.state_lock:
            target_progress = self.closest_idx

            if target_progress > current_progress:
                max_step = self.max_forward_speed
                new_progress = min(target_progress, current_progress + max_step)
            elif target_progress < current_progress:
                max_step = self.max_backward_speed
                new_progress = max(target_progress, current_progress - max_step)
            else:
                new_progress = current_progress

            new_progress = max(0, min(self.total_points - 1, new_progress))
            self.progress_idx = new_progress

            old_start, old_end = self._get_display_window(current_progress)
            new_start, new_end = self._get_display_window(new_progress)

            window_changed = (old_start != new_start) or (old_end != new_end)

        return new_progress, window_changed

    def _get_display_window(self, progress):
        """
        Quarter-stepped reveal window: always shows two adjacent quarters of
        the full trajectory. Starts on Q1+Q2; once progress passes into Q2
        (Q1 complete), reveals Q2+Q3; once progress passes into Q3 (Q2
        complete), reveals Q3+Q4 and stays there through Q4 (there's no Q5).
        """
        quarter = min(3, progress // self.quarter_size)
        window_idx = min(quarter, 2)
        start = window_idx * self.quarter_size
        end = self.total_points if window_idx == 2 else start + 2 * self.quarter_size
        return start, end

    def _get_segment_for_point(self, point_idx):
        """Get segment index containing the given point index."""
        for i, boundary in enumerate(self.segment_boundaries[1:], 1):
            if point_idx < boundary:
                return i - 1
        return len(self.segment_boundaries) - 1

    def _publish_window(self, progress, force=False):
        """
        Publish trajectory visualization point cloud.

        Always publishes fixed_cloud_size points for AMBF rendering stability.
        Pads with duplicates of last visible point when window is smaller.
        """
        now = self._now_sec()

        if not force and (now - self.last_publish_time < 0.05):
            return

        visible_start, visible_end = self._get_display_window(progress)

        if visible_end <= visible_start:
            self.get_logger().warn(f"Invalid window: [{visible_start}:{visible_end}], progress={progress}")
            return

        visible_points = self.trajectory_points[visible_start:visible_end]

        if len(visible_points) == 0:
            self.get_logger().warn(f"Empty visible points at progress={progress}")
            return

        pc = PointCloud()
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.header.frame_id = 'CameraFrame'

        for pt in visible_points:
            pc.points.append(Point32(x=pt[0], y=pt[1], z=pt[2]))

        last_pt = visible_points[-1]
        while len(pc.points) < self.fixed_cloud_size:
            pc.points.append(Point32(x=last_pt[0], y=last_pt[1], z=last_pt[2]))

        self.traj_pub.publish(pc)
        self.traj_size_pub.publish(Float32(data=5.0))
        self._publish_target_marker(self.trajectory_points[progress])

        self.last_visible_start = visible_start
        self.last_visible_end = visible_end
        self.last_publish_time = now

    def _publish_target_marker(self, point):
        """Publish the red single-point marker at the given trajectory point."""
        pc = PointCloud()
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.header.frame_id = 'CameraFrame'

        for _ in range(self.fixed_cloud_size):
            pc.points.append(Point32(x=point[0], y=point[1], z=point[2]))

        self.target_pub.publish(pc)
        self.target_size_pub.publish(Float32(data=8.0))

    def _publish_start_dot(self):
        """Publish start position indicator with fixed point cloud size."""
        pc = PointCloud()
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.header.frame_id = 'CameraFrame'

        for _ in range(self.fixed_cloud_size):
            pc.points.append(Point32(
                x=self.start_position[0],
                y=self.start_position[1],
                z=self.start_position[2]
            ))

        self.traj_pub.publish(pc)
        self.traj_size_pub.publish(Float32(data=10.0))
        self._publish_target_marker(self.start_position)

        self.get_logger().info(
            f"Start position: [{self.start_position[0]:.3f}, "
            f"{self.start_position[1]:.3f}, {self.start_position[2]:.3f}]")

        self.last_visible_start = -1
        self.last_visible_end = -1
        self.last_publish_time = self._now_sec()

    def _iteration_complete(self):
        """Handle iteration completion and prepare for next iteration."""
        self._record_to_bag('event', {
            'event': 'iteration_complete',
            'iteration': self.current_iter,
            'final_progress': self.progress_idx,
        })
        self.current_iter += 1
        self.scaling.on_iteration_complete()

        self.get_logger().info(f"Iteration {self.current_iter}/{self.total_iterations} complete")

        if self.current_iter >= self.total_iterations:
            self._record_to_bag('event', {'event': 'experiment_complete'})
            self.get_logger().info("All iterations complete")
            self.shutdown()
            return

        with self.state_lock:
            self.progress_idx = 0
            self.closest_idx = 0
            self.closest_dist = float('inf')
            self.waiting_for_start = True
            self.left_start_area = False
            self.last_visible_start = -1
            self.last_visible_end = -1
            self.frames_since_start = 0

        self._publish_start_dot()
        self.get_logger().info(f"Next: iteration {self.current_iter + 1}, scale {self.scaling.get_current_scale()}x")

    def _record_to_bag(self, topic_suffix, data):
        """Write data to the ROS2 bag."""
        data['timestamp'] = self._now_sec()
        msg = String(data=json.dumps(data))

        topic_map = {
            'session_start': '/session_metadata',
            'segment_definitions': '/session_metadata',
            'trajectory_points': '/session_metadata',
            'session_end': '/session_metadata',
            'event': '/bezier_events',
            'comprehensive': '/bezier_comprehensive',
        }
        topic = topic_map.get(topic_suffix, '/bezier_events')
        stamp_ns = self.get_clock().now().nanoseconds

        with self.bag_lock:
            if not self.bag_closed:
                self.bag_writer.write(topic, serialize_message(msg), stamp_ns)

    def _record_comprehensive(self):
        """Record comprehensive tracking data for current frame."""
        with self.state_lock:
            progress = self.progress_idx

        seg_idx = self._get_segment_for_point(progress)
        completion_pct = progress / self.total_points if self.total_points > 0 else 0
        search_start = max(0, progress - self.search_behind)
        search_end = min(self.total_points, progress + self.search_ahead)
        visible_start, visible_end = self._get_display_window(progress)

        self._record_to_bag('comprehensive', {
            'participant_id': self.participant_id,
            'motion_scaling': self.scaling.get_info(),
            'current_iteration': self.current_iter,
            'progress_idx': progress,
            'closest_idx': self.closest_idx,
            'total_points': self.total_points,
            'completion_pct': completion_pct,
            'current_segment': seg_idx,
            'closest_dist': float(self.closest_dist) if self.closest_dist < float('inf') else None,
            'visible_start': visible_start,
            'visible_end': visible_end,
            'search_start': search_start,
            'search_end': search_end,
            'psm2_position': self.psm_pos.tolist() if self.psm_pos is not None else None,
            'psm2_position_smoothed': self.psm_pos_smoothed.tolist() if self.psm_pos_smoothed is not None else None,
            'psm2_position_world': self.psm_pos_world.tolist() if self.psm_pos_world is not None else None,
            'psm2_orientation': self.psm_orientation,
            'psm2_velocity': self.psm_velocity,
            'psm2_velocity_magnitude': self.psm_vel_mag,
            'mtm_position': self.mtm_position,
            'mtm_orientation': self.mtm_orientation,
        })

    def _log_config(self):
        """Log experiment configuration."""
        self.get_logger().info(f"Participant: {self.participant_id}")
        self.get_logger().info(f"Segments: {self.num_segments}, Points: {self.total_points}")
        self.get_logger().info(f"Iterations: {self.total_iterations}")
        self.get_logger().info(f"Scaling: {self.scaling.mode}, ratios: {self.scaling.fixed_ratios}")
        self.get_logger().info(f"Display window: quarter-stepped, {self.quarter_size} pts/quarter")
        self.get_logger().info(f"Fixed cloud size: {self.fixed_cloud_size} points")
        self.get_logger().info(f"Search window: {self.search_behind} behind, {self.search_ahead} ahead")
        self.get_logger().info(f"Speed limits: fwd={self.max_forward_speed}, back={self.max_backward_speed} pts/frame")

    def shutdown(self):
        """Clean shutdown with data saving."""
        self.get_logger().info("Shutting down...")

        self._record_to_bag('session_end', {
            'event': 'session_end',
            'iterations_completed': self.current_iter,
            'motion_scaling_info': self.scaling.get_info(),
        })

        with self.bag_lock:
            if not self.bag_closed:
                self.bag_closed = True
                if hasattr(self.bag_writer, 'close'):
                    self.bag_writer.close()
                self.get_logger().info(f"Saved: {self.bag_path}")

        rclpy.shutdown()

    def spin(self):
        """Block the calling (main) thread until the background spin thread exits."""
        self._spin_thread.join()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sliding Window Bezier Trajectory Experiment')

    parser.add_argument('--participant', required=True, help='Participant ID')
    parser.add_argument('--segments', type=int, default=4, help='Number of segments')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fixed', type=str, help='Fixed ratios, e.g., "0.5,1.0,2.0"')
    group.add_argument('--adaptive', action='store_true', help='Adaptive scaling')
    parser.add_argument('--initial-scale', type=float, default=1.0)
    parser.add_argument('--adaptation-rate', type=float, default=0.1)

    parser.add_argument('--max-forward-speed', type=int, default=10)
    parser.add_argument('--max-backward-speed', type=int, default=3)
    parser.add_argument('--points-per-segment', type=int, default=200)
    parser.add_argument('--search-ahead', type=int, default=50)
    parser.add_argument('--search-behind', type=int, default=20)

    parser.add_argument('--bounding-box', type=str, help='"x_min,x_max,y_min,y_max,z_min,z_max"')
    parser.add_argument('--start-position', type=str, help='"x,y,z"')
    parser.add_argument('--proximity-threshold', type=float, default=0.003)

    args = parser.parse_args()

    mode = 'fixed' if args.fixed else 'adaptive'
    ratios = [float(x) for x in args.fixed.split(',')] if args.fixed else [1.0]

    bbox = None
    if args.bounding_box:
        p = [float(x) for x in args.bounding_box.split(',')]
        if len(p) != 6:
            print("Error: bounding-box needs 6 values")
            sys.exit(1)
        bbox = {'x_min': p[0], 'x_max': p[1], 'y_min': p[2], 'y_max': p[3], 'z_min': p[4], 'z_max': p[5]}

    start = None
    if args.start_position:
        start = [float(x) for x in args.start_position.split(',')]
        if len(start) != 3:
            print("Error: start-position needs 3 values")
            sys.exit(1)

    rclpy.init()
    try:
        exp = SlidingWindowExperiment(
            participant_id=args.participant,
            num_segments=args.segments,
            scaling_mode=mode,
            fixed_ratios=ratios,
            initial_scale=args.initial_scale,
            adaptation_rate=args.adaptation_rate,
            bounding_box=bbox,
            start_position=start,
            max_forward_speed=args.max_forward_speed,
            max_backward_speed=args.max_backward_speed,
            points_per_segment=args.points_per_segment,
            proximity_threshold=args.proximity_threshold,
            search_ahead=args.search_ahead,
            search_behind=args.search_behind,
        )
        exp.spin()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


"""
Usage:
    python bezier_collector.py --participant <ID> --segments <N> --fixed "<ratios>"

Examples:
    # Single iteration at 1x scaling
    python bezier_collector.py --participant P001 --segments 4 --fixed "1.0"

    # Three iterations with different scaling ratios
    python bezier_collector.py --participant P001 --segments 4 --fixed "0.5,1.0,2.0"

    # Custom closest-point search window (display window is quarter-stepped
    # automatically and not independently configurable, see --segments)
    python bezier_collector.py --participant P001 --segments 4 --fixed "1.0" \
        --search-ahead 75 --search-behind 30

    # Custom workspace bounds (overrides the camera-fitted default box --
    # widening this risks the trajectory leaving the camera's fixed FOV)
    python bezier_collector.py --participant P001 --segments 4 --fixed "1.0" \
        --bounding-box "-0.03,0.03,-0.03,0.03,-0.25,-0.13" \
        --start-position "0,0,-0.19"

Arguments:
    --participant          Participant ID (required)
    --segments             Number of Bezier segments (default: 4). Also sets the
                            quarter-stepped display window (see _get_display_window):
                            the full trajectory is split into quarters and revealed
                            two at a time (Q1+Q2 -> Q2+Q3 -> Q3+Q4) as each quarter
                            is completed -- not independently configurable.
    --fixed                Comma-separated scaling ratios (required unless --adaptive)
    --adaptive             Use adaptive scaling mode
    --bounding-box         Workspace bounds "x_min,x_max,y_min,y_max,z_min,z_max"
                            (default: sized to fit cameraL/cameraR's fixed FOV,
                            see DEFAULT_BOUNDING_BOX -- the trajectory can no
                            longer escape the operator's view)
    --max-forward-speed    Max forward progress per frame (default: 10)
    --max-backward-speed   Max backward progress per frame (default: 3)
    --search-ahead         Search window ahead for closest point (default: 50)
    --search-behind        Search window behind for closest point (default: 20)
    --points-per-segment   Sampling density per segment (default: 200)
    --proximity-threshold  Distance threshold in meters (default: 0.003)
    --start-position       Start position "x,y,z" (default: box center, see
                            DEFAULT_START_POSITION)
"""
