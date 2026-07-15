import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32
import numpy as np
import math
import time

# Function to calculate triangle coordinates at parameter t (0 to 1)
def calculate_triangle(t):
    # Equilateral triangle vertices (can be modified for different triangle shapes)
    # Vertex coordinates in local frame
    vertex1 = np.array([0, triangle_radius, 0])  # Top vertex
    vertex2 = np.array([-triangle_radius * math.sqrt(3)/2, -triangle_radius/2, 0])  # Bottom left
    vertex3 = np.array([triangle_radius * math.sqrt(3)/2, -triangle_radius/2, 0])   # Bottom right
    
    # Normalize t to [0, 1] range
    t = t % 1.0
    
    if t <= 1/3:  # First edge: vertex1 to vertex2
        s = t * 3  # Scale to [0, 1] for this edge
        point = vertex1 + s * (vertex2 - vertex1)
    elif t <= 2/3:  # Second edge: vertex2 to vertex3
        s = (t - 1/3) * 3  # Scale to [0, 1] for this edge
        point = vertex2 + s * (vertex3 - vertex2)
    else:  # Third edge: vertex3 to vertex1
        s = (t - 2/3) * 3  # Scale to [0, 1] for this edge
        point = vertex3 + s * (vertex1 - vertex3)
    
    return point[0], point[1]


# ROS
rospy.init_node('static_triangle_pc')

# AMBF will have a default PC listener at '/ambf/env/World/point_cloud'
topics_names_param = '/ambf/env/World/point_cloud_topics'
triangle_topic = '/ambf/env/World/triangle'
triangle_radius_topic = f'{triangle_topic}/radius'

pc_topics = rospy.get_param(topics_names_param, [])
print('Existing Topics AMBF is listening to for Point Cloud:')
print(pc_topics)

pc_topics.append(triangle_topic)
rospy.set_param(topics_names_param, pc_topics)
print(f'Adding topic {triangle_topic} via the ROS Param server')

pub = rospy.Publisher(triangle_topic, PointCloud, queue_size=10)
size_pub = rospy.Publisher(triangle_radius_topic, Float32, queue_size=10)

msg = PointCloud()
msg.header.frame_id = 'phantomBODY phantom'

# PC
size_msg = Float32()
size_msg.data = 8.0  # Point size
triangle_radius = 0.03
num_points = 18000

# Initialize points array
for i in range(num_points):
    msg.points.append(Point32())

# Calculate triangle perimeter for even distribution
edge_length = triangle_radius * math.sqrt(3)  # For equilateral triangle
perimeter = 3 * edge_length

# Generate evenly spaced points along the triangle
for i in range(num_points):
    # Parameter t goes from 0 to 1 around the triangle
    t = i / (num_points - 1)
    
    x, y = calculate_triangle(t)
    z = 0.02  # Fixed z position
    
    msg.points[i].x = x
    msg.points[i].y = y
    msg.points[i].z = z


# Main loop
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    # Update timestamp and publish
    msg.header.stamp = rospy.Time.now()
    pub.publish(msg)
    size_pub.publish(size_msg)
    
    rate.sleep()
