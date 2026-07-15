#!/usr/bin/env python3
"""
Visual Proximity Indicator for dVRK trajectory following
Provides real-time visual feedback about tool-trajectory distance
"""

import sys
import rospy
import numpy as np
import tf.transformations as tf
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from ambf_msgs.msg import RigidBodyState
from std_msgs.msg import Float32
import time

# Force unbuffered output for real-time logging
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


class ProximityIndicator:
    def __init__(self):
        rospy.init_node('proximity_indicator')
        
        # Configuration parameters
        self.distance_threshold = 0.01  # 1cm - maximum distance for indicator visibility
        
        # Segment scaling: LINEAR from 1 point (close) to 40 points (at threshold)
        self.min_segment_points = 1      # Minimum points when very close
        self.max_segment_points = 40     # Maximum points at threshold distance
        
        # Fixed size: 4x the trajectory size
        # Trajectory uses 5.0 for normal shapes, so we use 20.0
        self.indicator_point_size = 20.0
        
        self.update_rate = 30  # Hz
        
        # Phantom frame transformation (matching comprehensive_collector.py)
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion)
        )
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        self.tool_offset_local = np.array([0, -0.009855, 0])
        
        # State variables
        self.trajectory_points = None
        self.psm1_tool_tip = None
        self.psm2_tool_tip = None
        self.psm1_closest_point = None
        self.psm2_closest_point = None
        self.psm1_closest_index = None
        self.psm2_closest_index = None
        self.psm1_distance = None
        self.psm2_distance = None
        
        # CRITICAL: Wait for AMBF to be ready
        rospy.sleep(2.0)
        
        # Register topics with AMBF - THIS IS CRITICAL
        self._register_indicator_topics()
        
        # Publishers for proximity indicators - BOTH NEED LATCHING like trajectory
        self.psm1_indicator_pub = rospy.Publisher(
            '/ambf/env/World/psm1_proximity',
            PointCloud, queue_size=10, latch=True
        )
        self.psm2_indicator_pub = rospy.Publisher(
            '/ambf/env/World/psm2_proximity',
            PointCloud, queue_size=10, latch=True
        )
        
        # Publishers for indicator size - MUST USE /radius (not /size) per AMBF spec
        self.psm1_indicator_size_pub = rospy.Publisher(
            '/ambf/env/World/psm1_proximity/radius',
            Float32, queue_size=10, latch=True
        )
        self.psm2_indicator_size_pub = rospy.Publisher(
            '/ambf/env/World/psm2_proximity/radius',
            Float32, queue_size=10, latch=True
        )
        
        # Subscribers
        self.trajectory_sub = rospy.Subscriber(
            '/ambf/env/World/trajectory',
            PointCloud, self.trajectory_callback, queue_size=1
        )
        
        self.psm1_sub = rospy.Subscriber(
            '/ambf/env/psm1/toolyawlink/State',
            RigidBodyState, self.psm1_callback, queue_size=1
        )
        
        self.psm2_sub = rospy.Subscriber(
            '/ambf/env/psm2/toolyawlink/State',
            RigidBodyState, self.psm2_callback, queue_size=1
        )
        
        # Wait a moment for publishers to establish
        rospy.sleep(0.5)
        
        # Publish initial size (will be latched and republished with each point cloud)
        self._publish_initial_sizes()
        
        # Update timer
        self.timer = rospy.Timer(rospy.Duration(1.0/self.update_rate), self.update_indicators)
        
        # Check subscription status after a moment
        rospy.Timer(rospy.Duration(1.0), self.check_subscriptions, oneshot=True)
        
        rospy.loginfo("Proximity Indicator initialized:")
        rospy.loginfo("  Distance threshold: %.1fmm", self.distance_threshold*1000)
        rospy.loginfo("  Segment points: %d (close) to %d (at threshold)", 
                     self.min_segment_points, self.max_segment_points)
        rospy.loginfo("  Indicator size: %.1f (4x trajectory size of 5.0)", self.indicator_point_size)
        rospy.loginfo("  Publishers: BOTH latch=True (matching dynamic_trajectory.py)")
        rospy.loginfo("  Update rate: %d Hz", self.update_rate)
        
    def check_subscriptions(self, event):
        """Check if we're receiving data from topics"""
        if self.trajectory_points is None:
            rospy.logwarn("No trajectory received yet - waiting...")
        else:
            rospy.loginfo("Trajectory active with %d points", len(self.trajectory_points))
            
        if self.psm1_tool_tip is None:
            rospy.logwarn("No PSM1 data received yet")
        if self.psm2_tool_tip is None:
            rospy.logwarn("No PSM2 data received yet")
        
    def _register_indicator_topics(self):
        """Register indicator topics with AMBF - CRITICAL FOR VISIBILITY"""
        topics_param = '/ambf/env/World/point_cloud_topics'
        
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Get existing topics
                pc_topics = rospy.get_param(topics_param, [])
                
                # Add our indicator topics
                new_topics = [
                    '/ambf/env/World/psm1_proximity',
                    '/ambf/env/World/psm2_proximity'
                ]
                
                for topic in new_topics:
                    if topic not in pc_topics:
                        pc_topics.append(topic)
                        rospy.loginfo('Registering %s with AMBF', topic)
                        
                rospy.set_param(topics_param, pc_topics)
                rospy.loginfo('Successfully registered proximity indicator topics')
                return
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    rospy.logwarn("Attempt %d failed: %s, retrying...", attempt + 1, str(e))
                    rospy.sleep(0.5)
                else:
                    rospy.logerr("Could not register topics after %d attempts: %s", 
                                max_attempts, str(e))
    
    def _publish_initial_sizes(self):
        """Publish initial sizes once (latched, so will persist)"""
        size_msg = Float32()
        size_msg.data = self.indicator_point_size
        
        self.psm1_indicator_size_pub.publish(size_msg)
        self.psm2_indicator_size_pub.publish(size_msg)
        
        rospy.loginfo("Published initial indicator size: %.1f (latched)", self.indicator_point_size)
            
    def trajectory_callback(self, msg):
        """Update trajectory points"""
        if len(msg.points) > 0:
            old_count = len(self.trajectory_points) if self.trajectory_points is not None else 0
            self.trajectory_points = np.array([[p.x, p.y, p.z] for p in msg.points])
            
            if old_count != len(self.trajectory_points):
                rospy.loginfo("Trajectory updated: %d points", len(self.trajectory_points))
                
                # Reset closest point calculations
                self.psm1_closest_point = None
                self.psm1_closest_index = None
                self.psm1_distance = None
                self.psm2_closest_point = None
                self.psm2_closest_index = None
                self.psm2_distance = None
        else:
            self.trajectory_points = None
            # Clear indicators
            self.publish_empty_indicator(self.psm1_indicator_pub)
            self.publish_empty_indicator(self.psm2_indicator_pub)
            
    def psm1_callback(self, msg):
        """Update PSM1 tool tip position"""
        self.psm1_tool_tip = self._get_tool_tip_position(msg)
        if self.trajectory_points is not None and self.psm1_tool_tip is not None:
            self.psm1_closest_point, self.psm1_distance, self.psm1_closest_index = \
                self._find_closest_point(self.trajectory_points, self.psm1_tool_tip)
                
    def psm2_callback(self, msg):
        """Update PSM2 tool tip position"""
        self.psm2_tool_tip = self._get_tool_tip_position(msg)
        if self.trajectory_points is not None and self.psm2_tool_tip is not None:
            self.psm2_closest_point, self.psm2_distance, self.psm2_closest_index = \
                self._find_closest_point(self.trajectory_points, self.psm2_tool_tip)
                
    def _get_tool_tip_position(self, msg):
        """Calculate tool tip position in phantom frame"""
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
        
        # Transform to phantom frame
        pos_h = np.append(tool_tip_world[:3], 1.0)
        local_h = np.dot(self.phantom_inverse_transform, pos_h)
        
        return local_h[:3]
        
    def _find_closest_point(self, trajectory, tool_tip):
        """Find closest point on trajectory to tool tip"""
        distances = np.linalg.norm(trajectory - tool_tip, axis=1)
        closest_idx = np.argmin(distances)
        return trajectory[closest_idx], distances[closest_idx], closest_idx
        
    def update_indicators(self, event):
        """Update proximity indicators for both PSMs"""
        # Update PSM1 indicator
        if (self.trajectory_points is not None and 
            self.psm1_tool_tip is not None and 
            self.psm1_distance is not None):
            
            self.publish_indicator(
                self.psm1_distance,
                self.psm1_closest_index,
                self.psm1_indicator_pub,
                self.psm1_indicator_size_pub,
                "PSM1"
            )
            
        # Update PSM2 indicator  
        if (self.trajectory_points is not None and 
            self.psm2_tool_tip is not None and 
            self.psm2_distance is not None):
            
            self.publish_indicator(
                self.psm2_distance,
                self.psm2_closest_index,
                self.psm2_indicator_pub,
                self.psm2_indicator_size_pub,
                "PSM2"
            )
            
    def publish_indicator(self, distance, center_index, pub, size_pub, psm_name):
        """Publish proximity indicator point cloud
        
        LINEAR SCALING:
        Distance  | Num Points
        ---------|------------
        0mm      | 1 point
        5mm      | 20 points
        10mm     | 40 points
        >10mm    | Hidden
        """
        
        # Check if within threshold
        if distance > self.distance_threshold:
            self.publish_empty_indicator(pub)
            return
            
        # Calculate number of points: LINEAR from 1 to 40
        normalized_dist = distance / self.distance_threshold  # 0 to 1
        
        # Linear interpolation
        num_segment_points = int(self.min_segment_points + 
                                (self.max_segment_points - self.min_segment_points) * normalized_dist)
        
        # Ensure at least 1 point
        num_segment_points = max(1, num_segment_points)
        
        num_trajectory_points = len(self.trajectory_points)
        
        # Calculate segment indices (centered at closest point)
        half_segment = num_segment_points // 2
        
        if num_trajectory_points == 1:
            # Special case: single dot trajectory
            # Still scale from 1-40 points, just repeat the same point
            indices = [0] * num_segment_points
        else:
            start_idx = max(0, center_index - half_segment)
            end_idx = min(num_trajectory_points, center_index + half_segment + 1)
            
            # Handle wraparound for closed trajectories
            if self._is_closed_trajectory():
                indices = []
                for i in range(center_index - half_segment, center_index + half_segment + 1):
                    indices.append(i % num_trajectory_points)
            else:
                indices = list(range(start_idx, end_idx))
            
            # Ensure we have the right number of points
            if len(indices) < num_segment_points:
                # Extend by repeating some indices
                while len(indices) < num_segment_points:
                    indices.append(indices[-1])
            elif len(indices) > num_segment_points:
                # Trim to exact count
                indices = indices[:num_segment_points]
            
        # Create point cloud message
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        
        # Add points from trajectory segment with slight vertical offset
        for idx in indices:
            p = Point32()
            p.x = self.trajectory_points[idx][0]
            p.y = self.trajectory_points[idx][1]
            p.z = self.trajectory_points[idx][2] + 0.003  # 3mm vertical offset
            msg.points.append(p)
        
        # Publish point cloud
        pub.publish(msg)
        
        # Publish size with point cloud (latched, so persists even if we don't call again)
        # This matches the pattern in dynamic_trajectory.py
        size_msg = Float32()
        size_msg.data = self.indicator_point_size
        size_pub.publish(size_msg)
        
        # Log occasionally
        current_time = rospy.Time.now().to_sec()
        if int(current_time) % 3 == 0 and (current_time - int(current_time)) < 0.1:
            rospy.loginfo("%s: dist=%.1fmm, points=%d/%d",
                         psm_name, distance*1000, num_segment_points, self.max_segment_points)
        
    def publish_empty_indicator(self, pub):
        """Publish empty point cloud to clear indicator"""
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        
    def _is_closed_trajectory(self):
        """Check if trajectory is closed"""
        if self.trajectory_points is None or len(self.trajectory_points) < 2:
            return False
            
        dist = np.linalg.norm(self.trajectory_points[0] - self.trajectory_points[-1])
        return dist < 0.001  # Within 1mm
        
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        indicator = ProximityIndicator()
        indicator.spin()
    except rospy.ROSInterruptException:
        pass