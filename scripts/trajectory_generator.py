#!/usr/bin/env python3
import rospy
import argparse
import numpy as np
import math
import signal
import sys
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32, String

class TrajectoryGenerator:
    def __init__(self, pattern_type, **kwargs):
        """
        Initialize the trajectory generator
        
        Args:
            pattern_type (str): Type of pattern ('circle', 'figure8', 'line', 'triangle')
            **kwargs: Pattern-specific parameters
        """
        self.pattern_type = pattern_type.lower()
        self.params = self._set_default_params()
        self.params.update(kwargs)
        
        # ROS setup
        rospy.init_node(f'trajectory_generator_{self.pattern_type}')
        
        # Single topic configuration - all patterns use same topic
        self.base_topic = '/ambf/env/World/trajectory'
        self.size_topic = f'{self.base_topic}/size'
        self.pattern_info_topic = f'{self.base_topic}/pattern_info'
        
        # Register topic with AMBF
        self._register_topic()
        
        # Publishers
        self.pub = rospy.Publisher(self.base_topic, PointCloud, queue_size=10, latch=True)
        self.size_pub = rospy.Publisher(self.size_topic, Float32, queue_size=10, latch=True)
        
        # Pattern info publisher to indicate current pattern
        self.pattern_pub = rospy.Publisher(self.pattern_info_topic, String, queue_size=10, latch=True)
        
        # Generate the trajectory
        self.points = self._generate_trajectory()
        self._create_pointcloud_message()
        
        # Flag for clean shutdown
        self.shutdown_requested = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"Generated {self.pattern_type} trajectory with {len(self.points)} points")
        print(f"Parameters: {self.params}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self._clear_trajectory()
        rospy.signal_shutdown("Graceful shutdown")
        sys.exit(0)
    
    def _clear_trajectory(self):
        """Publish an empty point cloud to clear the trajectory"""
        empty_msg = PointCloud()
        empty_msg.header.frame_id = self.params['frame_id']
        empty_msg.header.stamp = rospy.Time.now()
        
        # Publish empty point cloud multiple times to ensure it's received
        for _ in range(5):
            self.pub.publish(empty_msg)
            rospy.sleep(0.1)
        
        print(f"Cleared {self.pattern_type} trajectory from display")
    
    def _set_default_params(self):
        """Set default parameters for all patterns"""
        return {
            'radius': 0.03,          # For circle, figure8, triangle
            'length': 0.06,          # For line
            'pos_x': 0.0,           # Position offsets
            'pos_y': 0.0,
            'pos_z': 0.02,
            'roll': 0.0,            # Orientation (degrees)
            'pitch': 0.0,
            'yaw': 0.0,
            'point_size': 5.0,      # Visual point size
            'num_points': None,     # Auto-calculate based on pattern
            'frame_id': 'phantomBODY phantom'
        }
    
    def _register_topic(self):
        """Register the single trajectory topic with AMBF point cloud system"""
        topics_param = '/ambf/env/World/point_cloud_topics'
        pc_topics = rospy.get_param(topics_param, [])
        
        if self.base_topic not in pc_topics:
            pc_topics.append(self.base_topic)
            rospy.set_param(topics_param, pc_topics)
            print(f'Registered unified trajectory topic {self.base_topic} with AMBF')
        
        print(f'Using unified trajectory topic: {self.base_topic}')
        print(f'Current pattern: {self.pattern_type}')
        print('NOTE: Starting this pattern will automatically replace any previous pattern')
    
    def _apply_transform(self, x, y, z=0):
        """Apply rotation and translation to a point"""
        # Convert angles to radians
        roll_rad = math.radians(self.params['roll'])
        pitch_rad = math.radians(self.params['pitch'])
        yaw_rad = math.radians(self.params['yaw'])
        
        # Create rotation matrices
        R_roll = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])
        
        R_pitch = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])
        
        R_yaw = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations
        R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
        
        # Apply rotation
        point = np.array([x, y, z])
        rotated_point = np.dot(R, point)
        
        # Add position offset
        final_x = rotated_point[0] + self.params['pos_x']
        final_y = rotated_point[1] + self.params['pos_y']
        final_z = rotated_point[2] + self.params['pos_z']
        
        return final_x, final_y, final_z
    
    def _generate_circle(self):
        """Generate circle trajectory points"""
        num_points = self.params['num_points'] or 3600
        radius = self.params['radius']
        points = []
        
        for i in range(num_points):
            t = 2 * math.pi * i / num_points
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            x_final, y_final, z_final = self._apply_transform(x, y, 0)
            points.append((x_final, y_final, z_final))
        
        return points
    
    def _generate_figure8(self):
        """Generate figure-8 (lemniscate) trajectory points"""
        num_points = self.params['num_points'] or 18000
        radius = self.params['radius']
        
        # Calculate arc length parameterization for even spacing
        total_length = 0
        step_size = 0.01
        parameter_values = []
        accumulated_lengths = []
        
        t = 0
        prev_x, prev_y = self._calculate_figure8_point(0, radius)
        
        while t <= 2 * np.pi:
            x, y = self._calculate_figure8_point(t, radius)
            
            if t > 0:
                segment_length = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                total_length += segment_length
            
            parameter_values.append(t)
            accumulated_lengths.append(total_length)
            
            prev_x, prev_y = x, y
            t += step_size
        
        # Normalize accumulated lengths
        if total_length > 0:
            accumulated_lengths = [l/total_length for l in accumulated_lengths]
        
        # Generate evenly spaced points
        points = []
        for i in range(num_points):
            target_length = i / (num_points - 1)
            closest_idx = min(range(len(accumulated_lengths)), 
                            key=lambda i: abs(accumulated_lengths[i] - target_length))
            
            t = parameter_values[closest_idx]
            x, y = self._calculate_figure8_point(t, radius)
            x_final, y_final, z_final = self._apply_transform(x, y, 0)
            points.append((x_final, y_final, z_final))
        
        return points
    
    def _calculate_figure8_point(self, t, radius):
        """Calculate figure-8 coordinates at parameter t"""
        x = radius * math.sin(t) / (1 + math.cos(t)**2)
        y = radius * math.sin(t) * math.cos(t) / (1 + math.cos(t)**2)
        return x, y
    
    def _generate_line(self):
        """Generate line trajectory points"""
        num_points = self.params['num_points'] or 1000
        length = self.params['length']
        points = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            x = length * (t - 0.5)  # Centered line
            y = 0.0
            x_final, y_final, z_final = self._apply_transform(x, y, 0)
            points.append((x_final, y_final, z_final))
        
        return points
    
    def _generate_triangle(self):
        """Generate equilateral triangle trajectory points"""
        num_points = self.params['num_points'] or 18000
        radius = self.params['radius']  # Circumradius
        points = []
        
        # Equilateral triangle vertices
        vertex1 = np.array([0, radius, 0])
        vertex2 = np.array([-radius * math.sqrt(3)/2, -radius/2, 0])
        vertex3 = np.array([radius * math.sqrt(3)/2, -radius/2, 0])
        vertices = [vertex1, vertex2, vertex3]
        
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Determine which edge and position along edge
            if t <= 1/3:
                s = t * 3
                edge_start, edge_end = vertices[0], vertices[1]
            elif t <= 2/3:
                s = (t - 1/3) * 3
                edge_start, edge_end = vertices[1], vertices[2]
            else:
                s = (t - 2/3) * 3
                edge_start, edge_end = vertices[2], vertices[0]
            
            # Interpolate along edge
            point = edge_start + s * (edge_end - edge_start)
            x_final, y_final, z_final = self._apply_transform(point[0], point[1], 0)
            points.append((x_final, y_final, z_final))
        
        return points
    
    def _generate_trajectory(self):
        """Generate trajectory based on pattern type"""
        if self.pattern_type == 'circle':
            return self._generate_circle()
        elif self.pattern_type == 'figure8':
            return self._generate_figure8()
        elif self.pattern_type == 'line':
            return self._generate_line()
        elif self.pattern_type == 'triangle':
            return self._generate_triangle()
        elif self.pattern_type == 'clear':
            return []  # Empty trajectory for clearing
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
    
    def _create_pointcloud_message(self):
        """Create ROS PointCloud message"""
        self.msg = PointCloud()
        self.msg.header.frame_id = self.params['frame_id']
        
        # Add all points to the message
        for x, y, z in self.points:
            point = Point32()
            point.x = x
            point.y = y
            point.z = z
            self.msg.points.append(point)
        
        # Size message
        self.size_msg = Float32()
        self.size_msg.data = self.params['point_size']
        
        # Pattern info message
        self.pattern_msg = String()
        pattern_info = {
            'pattern': self.pattern_type,
            'params': {k: v for k, v in self.params.items() if k not in ['frame_id']}
        }
        self.pattern_msg.data = str(pattern_info)
    
    def publish_trajectory(self):
        """Publish the trajectory once and keep the node alive"""
        # Small delay to ensure AMBF is ready
        rospy.sleep(0.5)
        
        if self.pattern_type == 'clear':
            print("Publishing empty trajectory to clear display...")
        else:
            print(f"Publishing {self.pattern_type} trajectory on unified topic: {self.base_topic}")
            print("This pattern will replace any previously active pattern.")
        
        print("Press Ctrl+C to stop...")
        
        # Publish pattern info once at start (latched)
        self.pattern_pub.publish(self.pattern_msg)
        
        # Publish the trajectory ONCE
        self.msg.header.stamp = rospy.Time.now()
        self.pub.publish(self.msg)
        self.size_pub.publish(self.size_msg)
        
        print(f"Trajectory published. Node will remain active until stopped.")
        
        # Keep the node alive but don't republish
        # Use a slow rate just to check for shutdown
        rate = rospy.Rate(1)  # 1 Hz - just for checking shutdown
        
        try:
            while not rospy.is_shutdown() and not self.shutdown_requested:
                # Just sleep and check for shutdown
                # Don't republish the trajectory
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            # Clear trajectory on exit (unless we're already the clear pattern)
            if self.pattern_type != 'clear':
                self._clear_trajectory()
            print(f"Trajectory generator for {self.pattern_type} stopped.")
    
    def save_trajectory_points(self, filename=None):
        """Save trajectory points to a file for analysis"""
        if filename is None:
            filename = f"{self.pattern_type}_trajectory.txt"
        
        with open(filename, 'w') as f:
            f.write(f"# {self.pattern_type.upper()} Trajectory Points\n")
            f.write(f"# Parameters: {self.params}\n")
            f.write("# x, y, z\n")
            for x, y, z in self.points:
                f.write(f"{x:.6f}, {y:.6f}, {z:.6f}\n")
        
        print(f"Trajectory points saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Unified Trajectory Generator for dVRK User Studies')
    
    # Pattern selection
    parser.add_argument('--pattern', type=str, required=True,
                       choices=['circle', 'figure8', 'line', 'triangle', 'clear'],
                       help='Type of trajectory pattern to generate (use "clear" to remove trajectory)')
    
    # Size parameters
    parser.add_argument('--radius', type=float, default=0.03,
                       help='Radius for circle, figure8, triangle (default: 0.03)')
    parser.add_argument('--length', type=float, default=0.06,
                       help='Length for line pattern (default: 0.06)')
    
    # Position parameters
    parser.add_argument('--pos_x', type=float, default=0.0,
                       help='X position offset (default: 0.0)')
    parser.add_argument('--pos_y', type=float, default=0.0,
                       help='Y position offset (default: 0.0)')
    parser.add_argument('--pos_z', type=float, default=0.02,
                       help='Z position offset (default: 0.02)')
    
    # Orientation parameters
    parser.add_argument('--roll', type=float, default=0.0,
                       help='Roll angle in degrees (default: 0.0)')
    parser.add_argument('--pitch', type=float, default=0.0,
                       help='Pitch angle in degrees (default: 0.0)')
    parser.add_argument('--yaw', type=float, default=0.0,
                       help='Yaw angle in degrees (default: 0.0)')
    
    # Visual parameters
    parser.add_argument('--point_size', type=float, default=5.0,
                       help='Visual point size (default: 5.0)')
    parser.add_argument('--num_points', type=int, default=None,
                       help='Number of points (auto-calculated if not specified)')
    
    # Presets for common orientations
    parser.add_argument('--preset', type=str, 
                       choices=['horizontal', 'vertical_yz', 'vertical_xz', 'diagonal'],
                       help='Use preset orientation')
    
    # Utility options
    parser.add_argument('--save_points', action='store_true',
                       help='Save trajectory points to file')
    parser.add_argument('--frame_id', type=str, default='phantomBODY phantom',
                       help='ROS frame ID (default: phantomBODY phantom)')
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset:
        if args.preset == 'vertical_yz':
            args.pitch = 90
        elif args.preset == 'vertical_xz':
            args.roll = 90
        elif args.preset == 'diagonal':
            args.pitch = 45
            args.yaw = 45
    
    # Create parameter dictionary
    params = {
        'radius': args.radius,
        'length': args.length,
        'pos_x': args.pos_x,
        'pos_y': args.pos_y,
        'pos_z': args.pos_z,
        'roll': args.roll,
        'pitch': args.pitch,
        'yaw': args.yaw,
        'point_size': args.point_size,
        'num_points': args.num_points,
        'frame_id': args.frame_id
    }
    
    try:
        # Create trajectory generator
        generator = TrajectoryGenerator(args.pattern, **params)
        
        # Save points if requested
        if args.save_points:
            generator.save_trajectory_points()
        
        # Start publishing
        generator.publish_trajectory()
        
    except rospy.ROSInterruptException:
        print("Trajectory generation stopped by user")
    except KeyboardInterrupt:
        print("\nShutting down trajectory generator...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()