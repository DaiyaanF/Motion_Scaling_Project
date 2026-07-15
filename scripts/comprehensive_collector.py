#!/usr/bin/env python3
import os
import time
import rospy
import rosbag
import json
import numpy as np
import threading
import signal
import sys
from datetime import datetime
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Joy, PointCloud, CompressedImage
from geometry_msgs.msg import PoseStamped, TwistStamped
from ambf_msgs.msg import RigidBodyState
import tf.transformations as tf


class ComprehensiveDataCollector:
    def __init__(self, participant_id="unknown"):
        rospy.init_node('comprehensive_data_collector')
        
        self.participant_id = participant_id
        self.shutdown_flag = False
        self.lock = threading.Lock()
        self.bag_closed = False
        
        # Phantom frame transformation (for PSM data)
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion)
        )
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        self.tool_offset_local = np.array([0, -0.009855, 0])  # Tool tip offset
        
        # Current state
        self.current_attributes = {}
        self.targets = None
        
        # PSM states (tool tip positions in phantom frame, velocities in world frame)
        self.psm1_position = None  # Tool tip in phantom frame
        self.psm1_position_world = None  # Tool tip in world frame (for velocity context)
        self.psm1_velocity = None  # Velocity in world frame
        self.psm1_dist = None
        
        self.psm2_position = None  # Tool tip in phantom frame
        self.psm2_position_world = None  # Tool tip in world frame
        self.psm2_velocity = None  # Velocity in world frame
        self.psm2_dist = None
        
        # MTM states (in their respective base frames)
        self.mtml_position = None  # Left MTM position
        self.mtml_velocity = None  # Left MTM velocity
        self.mtmr_position = None  # Right MTM position
        self.mtmr_velocity = None  # Right MTM velocity
        
        # Event counters
        self.clutch_count = 0
        self.footpedal_count = 0
        self.last_clutch_state = 0
        self.last_footpedal_state = 0
        
        # THROTTLING CONFIGURATION
        # Distance publishing at 20 Hz
        self.last_psm1_publish_time = 0
        self.last_psm2_publish_time = 0
        self.distance_publish_rate = 20.0  # Hz
        self.distance_publish_period = 1.0 / self.distance_publish_rate
        
        # Video recording at 5 Hz
        self.last_video_l_time = 0
        self.last_video_r_time = 0
        self.video_rate = 5.0  # Hz
        self.video_period = 1.0 / self.video_rate
        
        # MTM data recording at 30 Hz
        self.last_mtml_record_time = 0
        self.last_mtmr_record_time = 0
        self.mtm_record_rate = 30.0  # Hz
        self.mtm_record_period = 1.0 / self.mtm_record_rate
        
        rospy.loginfo(f"Throttling configured:")
        rospy.loginfo(f"  - Distance publishing: {self.distance_publish_rate} Hz")
        rospy.loginfo(f"  - Video recording: {self.video_rate} Hz")
        rospy.loginfo(f"  - MTM data recording: {self.mtm_record_rate} Hz")
        
        # Create data directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.expanduser(f"~/intel_motion_scaling/comprehensive_data/{participant_id}")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Open bag file
        self.bag_path = os.path.join(self.data_dir, f"comprehensive_{timestamp}.bag")
        self.bag = rosbag.Bag(self.bag_path, 'w')
        
        # Publishers for real-time monitoring
        self.psm1_dist_pub = rospy.Publisher(
            '/ambf/env/psm1/trajectory_dist', Float32, queue_size=1
        )
        self.psm2_dist_pub = rospy.Publisher(
            '/ambf/env/psm2/trajectory_dist', Float32, queue_size=1
        )
        
        # Core subscribers
        self.attr_sub = rospy.Subscriber(
            '/experiment/attributes', String, self.attributes_callback, queue_size=10
        )
        self.trajectory_sub = rospy.Subscriber(
            '/ambf/env/World/trajectory', PointCloud, self.trajectory_callback, queue_size=1
        )
        
        # PSM subscribers (AMBF) - these provide both position and velocity
        self.psm1_sub = rospy.Subscriber(
            '/ambf/env/psm1/toolyawlink/State', RigidBodyState, self.psm1_callback, queue_size=1
        )
        self.psm2_sub = rospy.Subscriber(
            '/ambf/env/psm2/toolyawlink/State', RigidBodyState, self.psm2_callback, queue_size=1
        )
        
        # MTM subscribers (dVRK) - position and velocity
        self.mtml_cp_sub = rospy.Subscriber(
            '/MTML/measured_cp', PoseStamped, self.mtml_position_callback, queue_size=1
        )
        self.mtml_cv_sub = rospy.Subscriber(
            '/MTML/measured_cv', TwistStamped, self.mtml_velocity_callback, queue_size=1
        )
        self.mtmr_cp_sub = rospy.Subscriber(
            '/MTMR/measured_cp', PoseStamped, self.mtmr_position_callback, queue_size=1
        )
        self.mtmr_cv_sub = rospy.Subscriber(
            '/MTMR/measured_cv', TwistStamped, self.mtmr_velocity_callback, queue_size=1
        )
        
        # Event subscribers
        self.clutch_sub = rospy.Subscriber(
            '/footpedals/clutch', Joy, self.clutch_callback, queue_size=10
        )
        self.footpedal_sub = rospy.Subscriber(
            '/footpedals/camera', Joy, self.footpedal_callback, queue_size=10
        )
        
        # Video subscribers
        self.camera_l_sub = rospy.Subscriber(
            '/ambf/env/cameras/cameraL/ImageData/compressed', 
            CompressedImage, self.camera_l_callback, queue_size=1
        )
        self.camera_r_sub = rospy.Subscriber(
            '/ambf/env/cameras/cameraR/ImageData/compressed', 
            CompressedImage, self.camera_r_callback, queue_size=1
        )
        
        # Data recording timer (30 Hz for comprehensive data to capture dynamics)
        self.timer = rospy.Timer(rospy.Duration(1.0/30.0), self.timer_callback)
        
        # Register shutdown handlers
        rospy.on_shutdown(self.shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        rospy.loginfo(f"Comprehensive data collector initialized")
        rospy.loginfo(f"Participant: {participant_id}")
        rospy.loginfo(f"Bag path: {self.bag_path}")
        rospy.loginfo(f"Recording: MTM positions/velocities, PSM tool tip positions/velocities")
        
    def _signal_handler(self, signum, frame):
        rospy.loginfo(f"Received signal {signum}, shutting down...")
        self.shutdown_flag = True
        self.shutdown()
        rospy.signal_shutdown("User requested shutdown")
        time.sleep(0.5)
        sys.exit(0)
        
    def attributes_callback(self, msg):
        """Update current experimental attributes"""
        try:
            self.current_attributes = json.loads(msg.data)
            
            # Check for experiment completion
            if self.current_attributes.get('experiment_completed', False):
                rospy.loginfo("Experiment completed signal received")
                self.shutdown()
        except json.JSONDecodeError:
            rospy.logwarn("Failed to decode attributes JSON")
            
    def trajectory_callback(self, msg):
        """Update trajectory points (in phantom frame)"""
        if not self.shutdown_flag:
            self.targets = np.array([[p.x, p.y, p.z] for p in msg.points])
            
    def psm1_callback(self, msg):
        """Update PSM1 tool tip position (in phantom frame) and velocity (world frame)"""
        if not self.shutdown_flag:
            # Get tool tip position in both frames
            self.psm1_position, self.psm1_position_world = self._get_tool_tip_positions(msg)
            
            # Extract velocity from twist field (world frame)
            if hasattr(msg, 'twist'):
                l = msg.twist.linear
                a = msg.twist.angular
                self.psm1_velocity = {
                    'linear': {'x': l.x, 'y': l.y, 'z': l.z},
                    'angular': {'x': a.x, 'y': a.y, 'z': a.z}
                }
            
            # Calculate distance to trajectory
            if self.targets is not None and len(self.targets) > 0 and self.psm1_position is not None:
                _, self.psm1_dist, _ = self._find_closest_point(self.targets, self.psm1_position)
                
                # Throttle publishing to GUI at 20 Hz
                current_time = rospy.Time.now().to_sec()
                if current_time - self.last_psm1_publish_time >= self.distance_publish_period:
                    dist_msg = Float32()
                    dist_msg.data = float(self.psm1_dist)
                    self.psm1_dist_pub.publish(dist_msg)
                    self.last_psm1_publish_time = current_time
                    
    def psm2_callback(self, msg):
        """Update PSM2 tool tip position (in phantom frame) and velocity (world frame)"""
        if not self.shutdown_flag:
            # Get tool tip position in both frames
            self.psm2_position, self.psm2_position_world = self._get_tool_tip_positions(msg)
            
            # Extract velocity from twist field (world frame)
            if hasattr(msg, 'twist'):
                l = msg.twist.linear
                a = msg.twist.angular
                self.psm2_velocity = {
                    'linear': {'x': l.x, 'y': l.y, 'z': l.z},
                    'angular': {'x': a.x, 'y': a.y, 'z': a.z}
                }
            
            # Calculate distance to trajectory
            if self.targets is not None and len(self.targets) > 0 and self.psm2_position is not None:
                _, self.psm2_dist, _ = self._find_closest_point(self.targets, self.psm2_position)
                
                # Throttle publishing to GUI at 20 Hz
                current_time = rospy.Time.now().to_sec()
                if current_time - self.last_psm2_publish_time >= self.distance_publish_period:
                    dist_msg = Float32()
                    dist_msg.data = float(self.psm2_dist)
                    self.psm2_dist_pub.publish(dist_msg)
                    self.last_psm2_publish_time = current_time
                    
    def mtml_position_callback(self, msg):
        """Update MTML position (in MTML base frame)"""
        if not self.shutdown_flag:
            p = msg.pose.position
            o = msg.pose.orientation
            self.mtml_position = {
                'x': p.x, 'y': p.y, 'z': p.z,
                'qx': o.x, 'qy': o.y, 'qz': o.z, 'qw': o.w
            }
            
            # Record to bag at throttled rate
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_mtml_record_time >= self.mtm_record_period:
                with self.lock:
                    if not self.bag_closed:
                        self.bag.write('/mtml/position', msg, rospy.Time.now())
                        self.last_mtml_record_time = current_time
            
    def mtml_velocity_callback(self, msg):
        """Update MTML velocity (in MTML base frame)"""
        if not self.shutdown_flag:
            l = msg.twist.linear
            a = msg.twist.angular
            self.mtml_velocity = {
                'linear': {'x': l.x, 'y': l.y, 'z': l.z},
                'angular': {'x': a.x, 'y': a.y, 'z': a.z}
            }
            
            # Record to bag at throttled rate (uses same timer as position)
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_mtml_record_time >= self.mtm_record_period:
                with self.lock:
                    if not self.bag_closed:
                        self.bag.write('/mtml/velocity', msg, rospy.Time.now())
            
    def mtmr_position_callback(self, msg):
        """Update MTMR position (in MTMR base frame)"""
        if not self.shutdown_flag:
            p = msg.pose.position
            o = msg.pose.orientation
            self.mtmr_position = {
                'x': p.x, 'y': p.y, 'z': p.z,
                'qx': o.x, 'qy': o.y, 'qz': o.z, 'qw': o.w
            }
            
            # Record to bag at throttled rate
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_mtmr_record_time >= self.mtm_record_period:
                with self.lock:
                    if not self.bag_closed:
                        self.bag.write('/mtmr/position', msg, rospy.Time.now())
                        self.last_mtmr_record_time = current_time
            
    def mtmr_velocity_callback(self, msg):
        """Update MTMR velocity (in MTMR base frame)"""
        if not self.shutdown_flag:
            l = msg.twist.linear
            a = msg.twist.angular
            self.mtmr_velocity = {
                'linear': {'x': l.x, 'y': l.y, 'z': l.z},
                'angular': {'x': a.x, 'y': a.y, 'z': a.z}
            }
            
            # Record to bag at throttled rate (uses same timer as position)
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_mtmr_record_time >= self.mtm_record_period:
                with self.lock:
                    if not self.bag_closed:
                        self.bag.write('/mtmr/velocity', msg, rospy.Time.now())
                        
    def clutch_callback(self, msg):
        """Record clutch events"""
        if self.shutdown_flag:
            return
        with self.lock:
            if msg.buttons and len(msg.buttons) > 0:
                current_state = msg.buttons[0]
                if current_state == 1 and self.last_clutch_state == 0:
                    self.clutch_count += 1
                    self.bag.write('/events/clutch', msg, rospy.Time.now())
                    rospy.loginfo(f"Clutch event #{self.clutch_count}")
                self.last_clutch_state = current_state
                
    def footpedal_callback(self, msg):
        """Record footpedal events"""
        if self.shutdown_flag:
            return
        with self.lock:
            if msg.buttons and len(msg.buttons) > 0:
                current_state = msg.buttons[0]
                if current_state == 1 and self.last_footpedal_state == 0:
                    self.footpedal_count += 1
                    self.bag.write('/events/camera_footpedal', msg, rospy.Time.now())
                    rospy.loginfo(f"Camera footpedal event #{self.footpedal_count}")
                self.last_footpedal_state = current_state
                
    def camera_l_callback(self, msg):
        """Record left camera images - throttled to 5 Hz"""
        if not self.shutdown_flag:
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_video_l_time >= self.video_period:
                with self.lock:
                    if not self.bag_closed:
                        self.bag.write('/video/cameraL', msg, rospy.Time.now())
                        self.last_video_l_time = current_time
                
    def camera_r_callback(self, msg):
        """Record right camera images - throttled to 5 Hz"""
        if not self.shutdown_flag:
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_video_r_time >= self.video_period:
                with self.lock:
                    if not self.bag_closed:
                        self.bag.write('/video/cameraR', msg, rospy.Time.now())
                        self.last_video_r_time = current_time
                
    def _get_tool_tip_positions(self, msg):
        """Calculate PSM tool tip position in both phantom and world frames"""
        # Get position and orientation from AMBF state
        p = msg.pose.position
        q = msg.pose.orientation
        
        position = np.array([p.x, p.y, p.z])
        quaternion = [q.x, q.y, q.z, q.w]
        
        # Create transformation matrix for tool
        tool_transform = tf.quaternion_matrix(quaternion)
        tool_transform[:3, 3] = position
        
        # Apply tool offset to get tool tip in world frame
        offset_homogeneous = np.append(self.tool_offset_local, 1.0)
        tool_tip_world = tool_transform @ offset_homogeneous
        tool_tip_world_pos = tool_tip_world[:3]
        
        # Transform to phantom frame
        pos_h = np.append(tool_tip_world_pos, 1.0)
        local_h = np.dot(self.phantom_inverse_transform, pos_h)
        tool_tip_phantom = local_h[:3]
        
        return tool_tip_phantom, tool_tip_world_pos
        
    def _find_closest_point(self, targets, source):
        """Find closest point and distance"""
        dists = np.linalg.norm(targets - source, axis=1)
        idx = np.argmin(dists)
        return targets[idx], dists[idx], idx
        
    def timer_callback(self, event):
        """Write comprehensive data entry at 30Hz for better dynamics capture"""
        if not self.shutdown_flag and not self.bag_closed:
            self._write_comprehensive_data()
            
    def _write_comprehensive_data(self):
        """Write current state to bag with all position/velocity data"""
        if self.bag_closed:
            return
            
        data = {
            'timestamp': rospy.Time.now().to_sec(),
            'participant_id': self.participant_id,
            
            # PSM distances from trajectory
            'psm1_dist': float(self.psm1_dist) if self.psm1_dist is not None else -1,
            'psm2_dist': float(self.psm2_dist) if self.psm2_dist is not None else -1,
            
            # PSM1 tool tip positions
            'psm1_position_phantom': {
                'x': float(self.psm1_position[0]),
                'y': float(self.psm1_position[1]),
                'z': float(self.psm1_position[2])
            } if self.psm1_position is not None else None,
            
            'psm1_position_world': {
                'x': float(self.psm1_position_world[0]),
                'y': float(self.psm1_position_world[1]),
                'z': float(self.psm1_position_world[2])
            } if self.psm1_position_world is not None else None,
            
            # PSM2 tool tip positions
            'psm2_position_phantom': {
                'x': float(self.psm2_position[0]),
                'y': float(self.psm2_position[1]),
                'z': float(self.psm2_position[2])
            } if self.psm2_position is not None else None,
            
            'psm2_position_world': {
                'x': float(self.psm2_position_world[0]),
                'y': float(self.psm2_position_world[1]),
                'z': float(self.psm2_position_world[2])
            } if self.psm2_position_world is not None else None,
            
            # PSM velocities (world frame)
            'psm1_velocity': self.psm1_velocity,
            'psm2_velocity': self.psm2_velocity,
            
            # Calculate velocity magnitudes for convenience
            'psm1_velocity_magnitude': np.linalg.norm([
                self.psm1_velocity['linear']['x'],
                self.psm1_velocity['linear']['y'],
                self.psm1_velocity['linear']['z']
            ]) if self.psm1_velocity else None,
            
            'psm2_velocity_magnitude': np.linalg.norm([
                self.psm2_velocity['linear']['x'],
                self.psm2_velocity['linear']['y'],
                self.psm2_velocity['linear']['z']
            ]) if self.psm2_velocity else None,
            
            # MTM positions and velocities (in their base frames)
            'mtml_position': self.mtml_position,
            'mtml_velocity': self.mtml_velocity,
            'mtmr_position': self.mtmr_position,
            'mtmr_velocity': self.mtmr_velocity,
            
            # Calculate MTM velocity magnitudes
            'mtml_velocity_magnitude': np.linalg.norm([
                self.mtml_velocity['linear']['x'],
                self.mtml_velocity['linear']['y'],
                self.mtml_velocity['linear']['z']
            ]) if self.mtml_velocity else None,
            
            'mtmr_velocity_magnitude': np.linalg.norm([
                self.mtmr_velocity['linear']['x'],
                self.mtmr_velocity['linear']['y'],
                self.mtmr_velocity['linear']['z']
            ]) if self.mtmr_velocity else None,
            
            # Event counts
            'clutch_count': self.clutch_count,
            'footpedal_count': self.footpedal_count,
            
            # Frame references for clarity
            'frame_info': {
                'psm_positions_phantom': 'phantom_frame',
                'psm_positions_world': 'world_frame',
                'psm_velocities': 'world_frame',
                'mtm_positions': 'respective_base_frames',
                'mtm_velocities': 'respective_base_frames',
                'trajectory': 'phantom_frame'
            }
        }
        
        # Add all current experimental attributes
        data.update(self.current_attributes)
        
        # Write to bag
        msg = String(data=json.dumps(data))
        with self.lock:
            if not self.bag_closed:
                self.bag.write('/comprehensive_data', msg, rospy.Time.now())
        
        # Periodic status update
        if int(time.time()) % 10 == 0:  # Every 10 seconds
            combo_id = self.current_attributes.get('combination_id', 0)
            total = self.current_attributes.get('total_combinations', 0)
            rospy.loginfo(f"Recording - Combination: {combo_id}/{total}")
            if self.mtml_position and self.mtmr_position:
                rospy.loginfo(f"  MTMs active: L={self.mtml_position is not None}, R={self.mtmr_position is not None}")
            if self.psm1_position is not None and self.psm2_position is not None:
                rospy.loginfo(f"  PSMs tracking: PSM1 dist={self.psm1_dist:.4f}m, PSM2 dist={self.psm2_dist:.4f}m")
            
    def shutdown(self):
        """Clean shutdown"""
        self.shutdown_flag = True
        with self.lock:
            if self.bag and not self.bag_closed:
                try:
                    # Write final metadata entry
                    final_data = {
                        'timestamp': rospy.Time.now().to_sec(),
                        'participant_id': self.participant_id,
                        'session_end': True,
                        'total_clutch_events': self.clutch_count,
                        'total_footpedal_events': self.footpedal_count,
                        'recording_rates': {
                            'video_hz': self.video_rate,
                            'distance_publish_hz': self.distance_publish_rate,
                            'mtm_record_hz': self.mtm_record_rate,
                            'comprehensive_data_hz': 30.0
                        },
                        'frame_references': {
                            'psm_tool_tip_positions_phantom': 'phantom_frame',
                            'psm_tool_tip_positions_world': 'world_frame',
                            'psm_velocities': 'world_frame',
                            'mtm_data': 'respective_base_frames',
                            'trajectory': 'phantom_frame'
                        }
                    }
                    msg = String(data=json.dumps(final_data))
                    self.bag.write('/session_metadata', msg, rospy.Time.now())
                    
                    self.bag.close()
                    self.bag_closed = True
                    rospy.loginfo(f"Bag closed successfully: {self.bag_path}")
                    rospy.loginfo(f"Final stats - Clutch: {self.clutch_count}, Footpedal: {self.footpedal_count}")
                except Exception as e:
                    rospy.logerr(f"Error closing bag: {e}")
                    
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Data Collector with MTM/PSM tracking')
    parser.add_argument('--participant', type=str, required=True, help='Participant ID')
    args = parser.parse_args()
    
    try:
        collector = ComprehensiveDataCollector(participant_id=args.participant)
        collector.spin()
    except rospy.ROSInterruptException:
        pass