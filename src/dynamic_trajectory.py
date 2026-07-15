#!/usr/bin/env python3
import rospy
import json
import numpy as np
import math
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import String, Float32
from ambf_msgs.msg import RigidBodyState
import tf.transformations as tf
from bezier_trajectory import BezierTrajectoryGenerator

class DynamicTrajectoryGenerator:
    def __init__(self):
        rospy.init_node('dynamic_trajectory_generator')

        # Store only the relevant attributes for comparison
        self.current_shape = None
        self.current_size = None
        self.current_rotation = None
        self.current_position = None
        self.current_attributes = {}

        self.trajectory_topic = '/ambf/env/World/trajectory'

        # Bezier state
        self.bezier_gen = None
        self.bezier_segments = []
        self.current_segment_id = 0
        self.last_endpoint = np.array([0, 0, 0])
        self.completion_threshold = 0.005
        self.bezier_config = {}
        self.psm2_tool_tip = None
        self.bezier_vertices_history = []

        # Phantom frame transformation for PSM position
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion)
        )
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        self.tool_offset_local = np.array([0, -0.009855, 0])

        # Register the topic with AMBF
        self._register_trajectory_topic()

        # Publishers with latch=True
        self.pub = rospy.Publisher(self.trajectory_topic, PointCloud, queue_size=10, latch=True)
        self.size_pub = rospy.Publisher(
            f'{self.trajectory_topic}/radius', Float32, queue_size=10, latch=True
        )
        self.attr_pub = rospy.Publisher('/experiment/attributes', String, queue_size=10, latch=True)

        # Subscriber for attribute changes
        self.attr_sub = rospy.Subscriber(
            '/experiment/attributes', String, self.attributes_callback
        )

        # Subscriber for PSM2 position (for Bezier completion detection)
        self.psm2_sub = rospy.Subscriber(
            '/ambf/env/psm2/toolyawlink/State',
            RigidBodyState, self.psm2_callback, queue_size=1
        )

        rospy.loginfo("Dynamic trajectory generator initialized")
        rospy.loginfo("  Using /radius for point size (AMBF standard)")
        
    def _register_trajectory_topic(self):
        """Register the trajectory topic with AMBF"""
        topics_param = '/ambf/env/World/point_cloud_topics'
        
        try:
            # Get existing topics
            pc_topics = rospy.get_param(topics_param, [])
            
            # Add our topic if not already present
            if self.trajectory_topic not in pc_topics:
                pc_topics.append(self.trajectory_topic)
                rospy.set_param(topics_param, pc_topics)
                rospy.loginfo(f'Registered trajectory topic {self.trajectory_topic} with AMBF')
            else:
                rospy.loginfo(f'Trajectory topic {self.trajectory_topic} already registered with AMBF')
                
        except Exception as e:
            rospy.logwarn(f"Could not register topic with AMBF: {e}")
        
    def attributes_callback(self, msg):
        try:
            attributes = json.loads(msg.data)
            self.current_attributes = attributes

            if attributes.get('experiment_completed', False):
                rospy.loginfo("Experiment completed - clearing trajectory")
                self.clear_trajectory()
                return

            shape = attributes.get('shape')
            size = attributes.get('size')
            rotation = attributes.get('rotation')
            position = attributes.get('position')
            is_initialization = attributes.get('is_initialization', False)

            if shape == 'bezier':
                self.completion_threshold = attributes.get('completion_threshold', 0.005)
                bezier_config = attributes.get('bezier_config', {})
                if bezier_config:
                    self.bezier_config = bezier_config

            if (shape != self.current_shape or
                size != self.current_size or
                rotation != self.current_rotation or
                position != self.current_position):

                if is_initialization:
                    rospy.loginfo("Publishing initialization dot pattern")
                else:
                    rospy.loginfo(f"Attributes changed - updating trajectory to {shape}")

                self.current_shape = shape
                self.current_size = size
                self.current_rotation = rotation
                self.current_position = position

                if shape == 'bezier':
                    self.bezier_segments = []
                    self.current_segment_id = 0
                    self.last_endpoint = np.array([0, 0, 0])
                    self.bezier_vertices_history = []
                    if self.bezier_gen is None:
                        boundary_size = self.bezier_config.get('boundary_size', 0.1)
                        self.bezier_gen = BezierTrajectoryGenerator(boundary_size=boundary_size)

                self.update_trajectory()

        except json.JSONDecodeError:
            rospy.logwarn("Failed to decode attributes JSON")
            
    def update_trajectory(self):
        """Generate and publish new trajectory based on current attributes"""
        if not self.current_shape:
            rospy.logwarn("No shape defined, skipping trajectory update")
            return
            
        shape = self.current_shape
        size = self.current_size
        rotation = self.current_rotation
        position = self.current_position
        
        rospy.loginfo(f"Generating {shape} trajectory with perimeter/length {size:.3f}m")
        
        # Generate points based on shape
        if shape == 'dot':
            points = self.generate_dot()
        elif shape == 'circle':
            points = self.generate_circle(size)
        elif shape == 'figure8':
            points = self.generate_figure8(size)
        elif shape == 'line':
            points = self.generate_line(size)
        elif shape == 'triangle':
            points = self.generate_triangle(size)
        elif shape == 'bezier':
            points = self.generate_bezier(size)
        else:
            rospy.logwarn(f"Unknown shape: {shape}")
            return
            
        # Apply rotation and position
        transformed_points = []
        for point in points:
            transformed = self.apply_transform(
                point[0], point[1], point[2],
                rotation, position
            )
            transformed_points.append(transformed)
            
        # Create and publish point cloud
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        
        for x, y, z in transformed_points:
            p = Point32()
            p.x = x
            p.y = y
            p.z = z
            msg.points.append(p)
            
        # Publish the point cloud
        self.pub.publish(msg)
        
        # Publish radius for AMBF (larger for dot to make it visible)
        radius_msg = Float32()
        if shape == 'dot':
            radius_msg.data = 10.0  # Larger point radius for visibility
        else:
            radius_msg.data = 5.0  # Normal point radius
        self.size_pub.publish(radius_msg)
        
        rospy.loginfo(f"Published trajectory: {shape} with perimeter/length {size:.3f}m ({len(msg.points)} points, radius={radius_msg.data})")
        
    def clear_trajectory(self):
        """Publish empty trajectory to clear display"""
        msg = PointCloud()
        msg.header.frame_id = 'phantomBODY phantom'
        msg.header.stamp = rospy.Time.now()
        msg.points = []
        
        self.pub.publish(msg)
        rospy.loginfo("Trajectory cleared")
        
    def generate_dot(self):
        """Generate a single dot point"""
        # Single point at origin
        return [(0, 0, 0)]
        
    def generate_circle(self, circumference):
        """Generate circle points
        
        Args:
            circumference: Total circumference of the circle in meters
        """
        points = []
        num_points = 360
        
        # Convert circumference to radius: C = 2πr => r = C/(2π)
        radius = circumference / (2 * math.pi)
        
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            points.append((x, y, 0))
            
        return points
        
    def generate_figure8(self, perimeter):
        """Generate figure-8 points
        
        Args:
            perimeter: Approximate total perimeter of the figure-8 in meters
        """
        points = []
        num_points = 500
        
        # The arc length of a lemniscate with parameter a is approximately 5.244a
        # So to get desired perimeter: a = perimeter / 5.244
        # This is an approximation - exact calculation requires elliptic integrals
        a = perimeter / 5.244
        
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            x = a * math.sin(t) / (1 + math.cos(t)**2)
            y = a * math.sin(t) * math.cos(t) / (1 + math.cos(t)**2)
            points.append((x, y, 0))
            
        return points
        
    def generate_line(self, length):
        """Generate line points
        
        Args:
            length: Total length of the line in meters
        """
        points = []
        num_points = 100
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = length * (t - 0.5)
            points.append((x, 0, 0))
            
        return points
        
    def generate_triangle(self, perimeter):
        """Generate triangle points
        
        Args:
            perimeter: Total perimeter of the equilateral triangle in meters
        """
        points = []
        num_points = 300
        
        # For equilateral triangle: perimeter = 3 * side_length
        side_length = perimeter / 3
        
        # For equilateral triangle: circumradius = side_length / sqrt(3)
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
        
    def generate_bezier(self, arc_length):
        if len(self.bezier_segments) == 0:
            segments_to_generate = self.bezier_config.get('segments_visible', 2)
            for _ in range(segments_to_generate):
                segment = self.bezier_gen.generate_segment(
                    self.last_endpoint,
                    arc_length,
                    curvature_range=None
                )
                self.bezier_segments.append(segment)
                self.last_endpoint = segment.P2

                segment_rotation = self._calculate_segment_rotation(segment)
                self.bezier_vertices_history.append({
                    'P0': {'x': float(segment.P0[0]), 'y': float(segment.P0[1]), 'z': float(segment.P0[2])},
                    'P1': {'x': float(segment.P1[0]), 'y': float(segment.P1[1]), 'z': float(segment.P1[2])},
                    'P2': {'x': float(segment.P2[0]), 'y': float(segment.P2[1]), 'z': float(segment.P2[2])},
                    'rotation': segment_rotation
                })

            if len(self.bezier_segments) > 0:
                self._publish_bezier_attributes(self.bezier_segments[0])

        all_points = []
        for segment in self.bezier_segments:
            segment_points = segment.sample_points(num_points=100)
            all_points.extend(segment_points)

        return all_points

    def psm2_callback(self, msg):
        if self.current_shape == 'bezier' and len(self.bezier_segments) > 0:
            psm_pos = self._get_tool_tip_position(msg)
            self.psm2_tool_tip = psm_pos

            first_segment = self.bezier_segments[0]
            if self.bezier_gen.check_completion(psm_pos, first_segment.P2, self.completion_threshold):
                self._on_segment_completed()

    def _get_tool_tip_position(self, msg):
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

    def _on_segment_completed(self):
        self.bezier_segments.pop(0)
        self.current_segment_id += 1
        rospy.loginfo(f"Segment {self.current_segment_id} completed")

        new_segment = self.bezier_gen.generate_segment(
            self.last_endpoint,
            self.current_size,
            curvature_range=None
        )
        self.bezier_segments.append(new_segment)
        self.last_endpoint = new_segment.P2

        segment_rotation = self._calculate_segment_rotation(new_segment)
        self.bezier_vertices_history.append({
            'P0': {'x': float(new_segment.P0[0]), 'y': float(new_segment.P0[1]), 'z': float(new_segment.P0[2])},
            'P1': {'x': float(new_segment.P1[0]), 'y': float(new_segment.P1[1]), 'z': float(new_segment.P1[2])},
            'P2': {'x': float(new_segment.P2[0]), 'y': float(new_segment.P2[1]), 'z': float(new_segment.P2[2])},
            'rotation': segment_rotation
        })

        self.update_trajectory()
        self._publish_bezier_attributes(self.bezier_segments[0])

    def _publish_bezier_attributes(self, segment):
        updated_attributes = dict(self.current_attributes)
        updated_attributes.update({
            'segment_id': self.current_segment_id,
            'total_segments_completed': self.current_segment_id,
            'curvature_max': float(segment.curvature_max),
            'curvature_avg': float(segment.curvature_avg),
            'bezier_vertices': {
                'P0': {'x': float(segment.P0[0]), 'y': float(segment.P0[1]), 'z': float(segment.P0[2])},
                'P1': {'x': float(segment.P1[0]), 'y': float(segment.P1[1]), 'z': float(segment.P1[2])},
                'P2': {'x': float(segment.P2[0]), 'y': float(segment.P2[1]), 'z': float(segment.P2[2])}
            },
            'bezier_vertices_history': self.bezier_vertices_history,
            'arc_length': float(segment.arc_length),
            'euclidean_distance': float(segment.euclidean_distance),
            'arc_ratio': float(segment.arc_ratio)
        })

        msg = String(data=json.dumps(updated_attributes))
        self.attr_pub.publish(msg)

    def _calculate_segment_rotation(self, segment):
        tangent_start = 2 * (segment.P1 - segment.P0)
        tangent_norm = np.linalg.norm(tangent_start)

        if tangent_norm < 1e-6:
            return {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

        tangent_unit = tangent_start / tangent_norm

        yaw = math.atan2(tangent_unit[1], tangent_unit[0])
        pitch = math.asin(-tangent_unit[2])

        return {
            'roll': 0.0,
            'pitch': float(math.degrees(pitch)),
            'yaw': float(math.degrees(yaw))
        }

    def apply_transform(self, x, y, z, rotation, position):
        roll = math.radians(rotation.get('roll', 0))
        pitch = math.radians(rotation.get('pitch', 0))
        yaw = math.radians(rotation.get('yaw', 0))

        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])

        R_pitch = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        R_yaw = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

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
    generator = DynamicTrajectoryGenerator()
    generator.spin()