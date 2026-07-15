import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32
import numpy as np
import math
import time
import argparse

def calculate_circle_3d(t, roll=0, pitch=0, yaw=0, radius=0.05, position=[0, 0, 0.02]):
    """
    Calculate 3D circle coordinates with rotation and position offset
    """
    # Base circle in x-y plane, centered at origin
    x_base = radius * math.cos(t)
    y_base = radius * math.sin(t)
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
    parser = argparse.ArgumentParser(description='Generate a 3D rotatable circle trajectory')
    parser.add_argument('--roll', type=float, default=0,
                        help='Roll angle in degrees (rotation around x-axis)')
    parser.add_argument('--pitch', type=float, default=0,
                        help='Pitch angle in degrees (rotation around y-axis)')
    parser.add_argument('--yaw', type=float, default=0,
                        help='Yaw angle in degrees (rotation around z-axis)')
    parser.add_argument('--radius', type=float, default=0.01,
                        help='Radius of the circle')
    parser.add_argument('--pos_x', type=float, default=0.0,
                        help='X position offset')
    parser.add_argument('--pos_y', type=float, default=0.0,
                        help='Y position offset')
    parser.add_argument('--pos_z', type=float, default=0.02,
                        help='Z position offset')
    parser.add_argument('--preset', type=str, choices=['horizontal', 'vertical_yz', 'vertical_xz'], 
                        help='Use a preset orientation')
    
    args = parser.parse_args()
    
    # Apply presets if specified
    if args.preset:
        if args.preset == 'vertical_yz':
            args.pitch = 90  # Rotate to yz-plane
        elif args.preset == 'vertical_xz':
            args.roll = 90   # Rotate to xz-plane

    # ROS initialization
    rospy.init_node('static_circle_pc')

    # AMBF setup
    topics_names_param = '/ambf/env/World/point_cloud_topics'
    circle_topic = '/ambf/env/World/circle'
    circle_radius_topic = f'{circle_topic}/radius'

    pc_topics = rospy.get_param(topics_names_param, [])
    print('Existing Topics AMBF is listening to for Point Cloud:')
    print(pc_topics)

    pc_topics.append(circle_topic)
    rospy.set_param(topics_names_param, pc_topics)
    print(f'Adding topic {circle_topic} via the ROS Param server')

    pub = rospy.Publisher(circle_topic, PointCloud, queue_size=10)
    size_pub = rospy.Publisher(circle_radius_topic, Float32, queue_size=10)

    msg = PointCloud()
    msg.header.frame_id = 'phantomBODY phantom'

    # PC size
    size_msg = Float32()
    size_msg.data = 5.0
    circle_radius = args.radius
    position = [args.pos_x, args.pos_y, args.pos_z]
    num_points = 3600  # Good resolution for smooth circle

    for i in range(num_points):
        msg.points.append(Point32())

    print(f"Generating circle with:")
    print(f"  Radius: {circle_radius}")
    print(f"  Position: {position}")
    print(f"  Orientation: Roll={args.roll}°, Pitch={args.pitch}°, Yaw={args.yaw}°")

    # Generate evenly spaced points along the circle
    for i in range(num_points):
        t = 2 * math.pi * i / num_points  # Parameter from 0 to 2π
        x, y, z = calculate_circle_3d(t, args.roll, args.pitch, args.yaw, circle_radius, position)
        
        msg.points[i].x = x
        msg.points[i].y = y
        msg.points[i].z = z

    # Main loop
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        size_pub.publish(size_msg)
        
        rate.sleep()