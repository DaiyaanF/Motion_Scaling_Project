#!/usr/bin/env python3
"""
Dynamic Trajectory Generator WITH Integrated Proximity Indicator
No separate scripts needed - everything in one place!
"""

import rospy
import json
import numpy as np
import math
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import String, Float32
from ambf_msgs.msg import RigidBodyState
import tf.transformations as tf


class DynamicTrajectoryWithProximity:
    def __init__(self):
        rospy.init_node('dynamic_trajectory_generator')
        
        # Trajectory attributes
        self.current_shape = None
        self.current_size = None
        self.current_rotation = None
        self.current_position = None
        self.trajectory_points = None  # Store generated trajectory
        
        # Proximity indicator settings
        self.distance_threshold = 0.02  # 2cm - show indicator within this range
        self.trajectory_size = 5  # Normal trajectory point size
        self.proximity_fade_points = 10  # Number of points to fade on each side
        
        # PSM tool tip tracking
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion)
        )
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        self.tool_offset_local = np.array([0, -0.009855, 0])
        
        self.psm1_tool_tip = None
        self.psm2_tool_tip = None
        
        # Topics
        self.trajectory_topic = '/ambf/env/World/trajectory'
        self.proximity_topic = '/ambf/env/World/trajectory_proximity'  # Overlay on top
        
        # Register topics
        self._register_topics()
        
        # Publishers
        self.trajectory_pub = rospy.Publisher(self.trajectory_topic, PointCloud, queue_size=10, latch=True)
        self.proximity_pub = rospy.Publisher(self.proximity_topic, PointCloud, queue_size=10, latch=True)
        
        # Subscribers
        self.attr_sub = rospy.Subscriber('/experiment/attributes', String, self.attributes_callback)
        self.psm1_sub = rospy.Subscriber('/ambf/env/psm1/toolyawlink/State', RigidBodyState, self.psm1_callback, queue_size=1)
        self.psm2_sub = rospy.Subscriber('/ambf/env/psm2/toolyawlink/State', RigidBodyState, self.psm2_callback, queue_size=1)
        
        # Update timer for proximity indicator (30 Hz)
        self.proximity_timer = rospy.Timer(rospy.Duration(1.0/30.0), self.update_proximity_indicator)
        
        rospy.loginfo("Dynamic Trajectory with Integrated Proximity Indicator initialized")
        rospy.loginfo(f"  Distance threshold: {self.distance_threshold*1000:.1f}mm")
        rospy.loginfo(f"  Spotlight: closest point + {self.proximity_fade_points} points on each side")
        rospy.loginfo(f"  Size range: 1-50 (based on distance/2)")
        
    def _register_topics(self):
        """Register both trajectory and proximity topics"""
        try:
            topics_param = '/ambf/env/World/point_cloud_topics'
            sizes_param = '/ambf/env/World/point_cloud_radii'
            
            pc_topics = rospy.get_param(topics_param, [])
            pc_sizes = rospy.get_param(sizes_param, [])
            
            # Fix alignment
            while len(pc_sizes) < len(pc_topics):
                pc_sizes.append(5)
            
            # Add trajectory if not present
            if self.trajectory_topic not in pc_topics:
                pc_topics.append(self.trajectory_topic)
                pc_sizes.append(self.trajectory_size)
            
            # Add proximity if not present (will be dynamically sized)
            if self.proximity_topic not in pc_topics:
                pc_topics.append(self.proximity_topic)
                pc_sizes.append(20)  # Default middle size
            
            rospy.set_param(topics_param, pc_topics)
            rospy.set_param(sizes_param, pc_sizes)
            
            rospy.loginfo(f"Registered topics with AMBF")
            
        except Exception as e:
            rospy.logwarn(f"Could not register topics: {e}")
    
    def attributes_callback(self, msg):
        """Handle attribute changes"""
        try:
            attributes = json.loads(msg.data)
            
            if attributes.get('experiment_completed', False):
                rospy.loginfo("Experiment completed - clearing trajectory")
                self.clear_trajectory()
                return
            
            shape = attributes.get('shape')
            size = attributes.get('size')
            rotation = attributes.get('rotation')
            position = attributes.get('position')
            
            if (shape != self.current_shape or size != self.current_size or 
                rotation != self.current_rotation or position != self.current_position):
                
                self.current_shape = shape
                self.current_size = size
                self.current_rotation = rotation
                self.current_position = position
                
                self.update_trajectory()
                
        except json.JSONDecodeError:
            rospy.logwarn("Failed to decode attributes JSON")
    
    def psm1_callback(self, msg):
        """Update PSM1 tool tip position"""
        self.psm1_tool_tip = self._get_tool_tip_position(msg)
    
    def psm2_callback(self, msg):
        """Update PSM2 tool tip position"""
        self.psm2_tool_tip = self._get_tool_tip_position(msg)
    
    def _get_tool_tip_position(self, msg):
        """Calculate tool tip position in phantom frame"""
        p = msg.pose.position
        q = msg.pose.orientation
        
        position = np.array([p.x, p.y, p.z])
        quaternion = [q.x, q.y, q.z, q.w]
        
        tool_transform = tf.quaternion_matrix(quaternion)
        tool_transform[:3, 3] = position
        
        offset_homogeneous = np.append(self.tool_offset_local, 1.0)
        tool_tip_world = tool_transform @ offset_homogeneous
        
        pos_h = np.append(tool_tip_world[:3], 1.0)
        local_h = np.dot(self.phantom_inverse_transform, pos_h)
        
        return local_h[:3]
    
    def update_trajectory(self):
        """Generate and publish trajectory"""
        if not self.current_shape:
            return
        
        shape = self.current_shape
        size = self.current_size
        
        # Generate points
        if shape == 'dot':
            points = [(0, 0, 0)]
        elif shape == 'circle':
            points = self.generate_circle(size)
        elif shape == 'figure8':
            points = self.generate_figure8(size)
        elif shape == 'line':
            points = self.generate_line(size)
        elif shape == 'triangle':
            points = self.generate_triangle(size)
        else:
            return
        
        # Apply transformations
        transformed_points = []
        for point in points:
            transformed = self.apply_transform(point[0], point[1], point[2], 
                                              self.current_rotation, self.current_position)
            transformed_points.append(transformed)
        
        # Store for proximity calculation
        self.trajectory_points = np.array(transformed_points)
        
        # Publish trajectory
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        
        for x, y, z in transformed_points:
            p = Point32()
            p.x = x
            p.y = y
            p.z = z
            msg.points.append(p)
        
        self.trajectory_pub.publish(msg)
        
        rospy.loginfo(f"Published trajectory: {shape} with {len(msg.points)} points")
    
    def update_proximity_indicator(self, event):
        """Update proximity spotlight - shows enhanced region near tool tip"""
        if self.trajectory_points is None or len(self.trajectory_points) == 0:
            return
        
        # Find closest point to either PSM (prefer PSM2)
        closest_idx = None
        min_distance = float('inf')
        
        for tool_tip in [self.psm2_tool_tip, self.psm1_tool_tip]:
            if tool_tip is not None:
                distances = np.linalg.norm(self.trajectory_points - tool_tip, axis=1)
                dist = np.min(distances)
                
                if dist < min_distance:
                    min_distance = dist
                    closest_idx = np.argmin(distances)
        
        # If no tool nearby or too far, clear proximity indicator
        if closest_idx is None or min_distance > self.distance_threshold:
            self.clear_proximity()
            return
        
        # Calculate spotlight size at closest point
        # Size = distance/2 (in mm), clamped between 1-50
        distance_mm = min_distance * 1000  # Convert to mm
        closest_point_size = max(1, min(50, distance_mm / 2))
        
        # Create spotlight region (closest point ± fade_points)
        num_trajectory_points = len(self.trajectory_points)
        start_idx = max(0, closest_idx - self.proximity_fade_points)
        end_idx = min(num_trajectory_points, closest_idx + self.proximity_fade_points + 1)
        
        # Generate proximity point cloud with varying sizes
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        
        # For each point in spotlight region
        for idx in range(start_idx, end_idx):
            # Distance from closest point (in number of trajectory points)
            dist_from_closest = abs(idx - closest_idx)
            
            # Linear fade: closest = closest_point_size, fade_points away = trajectory_size
            if dist_from_closest == 0:
                point_size = closest_point_size
            else:
                # Interpolate between closest_point_size and trajectory_size
                fade_ratio = dist_from_closest / self.proximity_fade_points
                point_size = closest_point_size + (self.trajectory_size - closest_point_size) * fade_ratio
            
            # Add point (we'll add it multiple times based on desired size for visibility)
            # Since AMBF uses global size, we duplicate points to make region more visible
            num_duplicates = max(1, int(point_size / 5))  # More duplicates = more visible
            
            for _ in range(num_duplicates):
                p = Point32()
                p.x = self.trajectory_points[idx][0]
                p.y = self.trajectory_points[idx][1]
                p.z = self.trajectory_points[idx][2] + 0.002  # Slight offset for visibility
                msg.points.append(p)
        
        self.proximity_pub.publish(msg)
    
    def clear_proximity(self):
        """Clear proximity indicator"""
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        msg.points = []
        self.proximity_pub.publish(msg)
    
    def clear_trajectory(self):
        """Clear both trajectory and proximity"""
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        msg.points = []
        
        self.trajectory_pub.publish(msg)
        self.proximity_pub.publish(msg)
        
        self.trajectory_points = None
        rospy.loginfo("Trajectory cleared")
    
    # Shape generation methods (same as before)
    def generate_circle(self, circumference):
        points = []
        num_points = 360
        radius = circumference / (2 * math.pi)
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            points.append((x, y, 0))
        return points
    
    def generate_figure8(self, perimeter):
        points = []
        num_points = 500
        a = perimeter / 5.244
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            x = a * math.sin(t) / (1 + math.cos(t)**2)
            y = a * math.sin(t) * math.cos(t) / (1 + math.cos(t)**2)
            points.append((x, y, 0))
        return points
    
    def generate_line(self, length):
        points = []
        num_points = 100
        for i in range(num_points):
            t = i / (num_points - 1)
            x = length * (t - 0.5)
            points.append((x, 0, 0))
        return points
    
    def generate_triangle(self, perimeter):
        points = []
        num_points = 300
        side_length = perimeter / 3
        radius = side_length / math.sqrt(3)
        vertices = [
            (0, radius, 0),
            (-radius * math.sqrt(3)/2, -radius/2, 0),
            (radius * math.sqrt(3)/2, -radius/2, 0)
        ]
        for i in range(num_points):
            t = i / (num_points - 1)
            edge_idx = int(t * 3) % 3
            edge_t = (t * 3) % 1
            v1 = vertices[edge_idx]
            v2 = vertices[(edge_idx + 1) % 3]
            x = v1[0] + edge_t * (v2[0] - v1[0])
            y = v1[1] + edge_t * (v2[1] - v1[1])
            points.append((x, y, 0))
        return points
    
    def apply_transform(self, x, y, z, rotation, position):
        """Apply rotation and translation"""
        roll = math.radians(rotation.get('roll', 0))
        pitch = math.radians(rotation.get('pitch', 0))
        yaw = math.radians(rotation.get('yaw', 0))
        
        R_roll = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        R_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        
        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        point = np.array([x, y, z])
        rotated = np.dot(R, point)
        
        final_x = rotated[0] + position.get('x', 0)
        final_y = rotated[1] + position.get('y', 0)
        final_z = rotated[2] + position.get('z', 0.04)
        
        return (final_x, final_y, final_z)
    
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    generator = DynamicTrajectoryWithProximity()
    generator.spin()