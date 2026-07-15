import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32
import numpy as np
import math
import time
import argparse

def calculate_figure8_3d(t, roll=0, pitch=0, yaw=0, figure8_radius=0.06, position=[0, 0, 0.02]):
    """
    Calculate 3D figure-8 coordinates with rotation and position offset
    """
    # Classic lemniscate (figure-8) formula
    x_base = figure8_radius * math.sin(t) / (1 + math.cos(t)**2)
    y_base = figure8_radius * math.sin(t) * math.cos(t) / (1 + math.cos(t)**2)
    z_base = 0.0
    
    # Convert angles to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
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
    base_point = np.array([x_base, y_base, z_base])
    rotated_point = np.dot(R, base_point)
    
    # Add position offset
    final_x = rotated_point[0] + position[0]
    final_y = rotated_point[1] + position[1]
    final_z = rotated_point[2] + position[2]
    
    return final_x, final_y, final_z

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a 3D rotatable figure-8 trajectory')
    parser.add_argument('--roll', type=float, default=0,
                        help='Roll angle in degrees (rotation around x-axis)')
    parser.add_argument('--pitch', type=float, default=0,
                        help='Pitch angle in degrees (rotation around y-axis)')
    parser.add_argument('--yaw', type=float, default=0,
                        help='Yaw angle in degrees (rotation around z-axis)')
    parser.add_argument('--radius', type=float, default=0.06,
                        help='Radius of the figure-8')
    parser.add_argument('--pos_x', type=float, default=0.0,
                        help='X position offset')
    parser.add_argument('--pos_y', type=float, default=0.0,
                        help='Y position offset')
    parser.add_argument('--pos_z', type=float, default=0.02,
                        help='Z position offset')
    parser.add_argument('--preset', type=str, choices=['horizontal', 'vertical', 'tilted'], 
                        help='Use a preset orientation')
    
    args = parser.parse_args()
    
    # Apply presets if specified
    if args.preset:
        if args.preset == 'vertical':
            args.pitch = 90
        elif args.preset == 'tilted':
            args.pitch = 45

    # ROS initialization
    rospy.init_node('static_figure8_pc')

    # AMBF setup
    topics_names_param = '/ambf/env/World/point_cloud_topics'
    figure_8_topic = '/ambf/env/World/figure_8'
    figure_8_radius_topic = f'{figure_8_topic}/radius'

    pc_topics = rospy.get_param(topics_names_param, [])
    print('Existing Topics AMBF is listening to for Point Cloud:')
    print(pc_topics)

    pc_topics.append(figure_8_topic)
    rospy.set_param(topics_names_param, pc_topics)
    print(f'Adding topic {figure_8_topic} via the ROS Param server')

    pub = rospy.Publisher(figure_8_topic, PointCloud, queue_size=10)
    size_pub = rospy.Publisher(figure_8_radius_topic, Float32, queue_size=10)

    msg = PointCloud()
    msg.header.frame_id = 'phantomBODY phantom'

    # PC size
    size_msg = Float32()
    size_msg.data = 8.0
    figure8_radius = args.radius
    position = [args.pos_x, args.pos_y, args.pos_z]
    num_points = 18000

    for i in range(num_points):
        msg.points.append(Point32())

    print(f"Generating figure-8 with:")
    print(f"  Radius: {figure8_radius}")
    print(f"  Position: {position}")
    print(f"  Orientation: Roll={args.roll}°, Pitch={args.pitch}°, Yaw={args.yaw}°")

    # Create an evenly distributed figure-8 curve using arc length parameterization
    total_length = 0
    step_size = 0.01
    parameter_values = []
    accumulated_lengths = []

    # Calculate total length and sample points
    t = 0
    prev_x, prev_y, prev_z = calculate_figure8_3d(0, args.roll, args.pitch, args.yaw, figure8_radius, position)

    while t <= 2*np.pi:
        x, y, z = calculate_figure8_3d(t, args.roll, args.pitch, args.yaw, figure8_radius, position)
        
        if t > 0:
            segment_length = math.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)
            total_length += segment_length
        
        parameter_values.append(t)
        accumulated_lengths.append(total_length)
        
        prev_x, prev_y, prev_z = x, y, z
        t += step_size

    # Normalize accumulated lengths
    if total_length > 0:
        accumulated_lengths = [l/total_length for l in accumulated_lengths]

    # Generate evenly spaced points along the curve
    point_idx = 0
    for i in range(num_points):
        target_length = i / (num_points - 1)
        
        closest_idx = min(range(len(accumulated_lengths)), 
                         key=lambda i: abs(accumulated_lengths[i] - target_length))
        
        t = parameter_values[closest_idx]
        x, y, z = calculate_figure8_3d(t, args.roll, args.pitch, args.yaw, figure8_radius, position)
        
        msg.points[point_idx].x = x
        msg.points[point_idx].y = y
        msg.points[point_idx].z = z
        point_idx += 1
        if point_idx >= num_points:
            break

    # Fill any remaining points
    for i in range(point_idx, num_points):
        msg.points[i].x = msg.points[point_idx-1].x
        msg.points[i].y = msg.points[point_idx-1].y
        msg.points[i].z = msg.points[point_idx-1].z

    # Main loop
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        size_pub.publish(size_msg)
        
        rate.sleep()