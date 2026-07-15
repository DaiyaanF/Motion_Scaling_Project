#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud
from ambf_msgs.msg import RigidBodyState
from geometry_msgs.msg import Point32, Pose, Point
from std_msgs.msg import Float32
import numpy as np
import math
import matplotlib.pyplot as plt
import tf.transformations as tf

class ClosestPointDistPlotter:
    def __init__(self, targets_topic: str, track_1_topic: str, track_2_topic: str):
        # Initialize the ROS node
        rospy.init_node('closest_point_dist_plotter', anonymous=True)
        
        # Set phantom transformation from YAML file first
        # These values should match the phantom position and orientation in phantom.yaml
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        # Convert from roll, pitch, yaw to quaternion
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        # Create transformation matrix
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion)
        )
        # Inverse transform (to convert world coordinates to phantom coordinates)
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        
        # Latest values from each topic
        self.targets = None
        self.track_1 = None
        self.track_2 = None
        
        # For plotting distances
        self.times = []
        self.track_1_dist_data = []
        self.track_2_dist_data = []
        
        # For plotting distance changes
        self.track_1_dist_change = []
        self.track_2_dist_change = []
        
        self.start_time = rospy.get_time()
        
        # Setup matplotlib for interactive real-time plotting
        self.setup_plotting()
        
        # Subscribers for topics - add these AFTER initializing transforms
        self.sub_targets = rospy.Subscriber(targets_topic, PointCloud, self.targets_callback)
        self.sub_track_1 = rospy.Subscriber(track_1_topic, RigidBodyState, self.track_1_callback)
        self.sub_track_2 = rospy.Subscriber(track_2_topic, RigidBodyState, self.track_2_callback)
    
    def targets_callback(self, msg):
        # Store the target points directly as they're already in phantom frame
        self.targets = np.array([[p.x, p.y, p.z] for p in msg.points])

    def track_1_callback(self, msg):
        p = msg.pose.position
        position = np.array([p.x, p.y, p.z])
        # Transform PSM1 position from world frame to phantom frame
        self.track_1 = self.transform_to_phantom_frame(position)

    def track_2_callback(self, msg):
        p = msg.pose.position
        position = np.array([p.x, p.y, p.z])
        # Transform PSM2 position from world frame to phantom frame
        self.track_2 = self.transform_to_phantom_frame(position)
    
    def transform_to_phantom_frame(self, position):
        """Transform a position from world frame to phantom frame"""
        # Convert to homogeneous coordinates
        position_homogeneous = np.append(position, 1.0)
        # Apply inverse transform to get position in phantom's local frame
        local_position_homogeneous = np.dot(self.phantom_inverse_transform, position_homogeneous)
        # Return to 3D coordinates
        return local_position_homogeneous[:3]

    def find_closest_point_and_dist(self, targets, source):        
        # Compute Euclidean distances from each point to the reference point
        distances = np.linalg.norm(targets - source, axis=1)
        
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        
        # Retrieve the closest point and its corresponding distance
        closest_point = targets[min_index]
        min_distance = distances[min_index]
        
        return closest_point, min_distance
    
    def setup_plotting(self):
        # Create figure with two subplots
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax_dist, self.ax_change) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Title for the whole figure
        self.fig.suptitle('PSM Interaction with Figure-8 Tube', fontsize=16)
        
        # Setup for distance plot
        self.ax_dist.set_xlabel('Time (s)')
        self.ax_dist.set_ylabel('Distance (m)')
        self.ax_dist.set_ylim(0, 1.0)
        self.ax_dist.set_title('Distance from Tube')
        self.ax_dist.grid(True)
        
        # Create empty line objects for distance plot
        self.line_psm1_dist, = self.ax_dist.plot([], [], 'r-', label='PSM1 Distance')
        self.line_psm2_dist, = self.ax_dist.plot([], [], 'b-', label='PSM2 Distance')
        self.ax_dist.legend(loc='upper right')
        
        # Setup for distance change plot
        self.ax_change.set_xlabel('Time (s)')
        self.ax_change.set_ylabel('Distance Change Rate (m/s)')
        self.ax_change.set_title('Rate of Change in Distance')
        self.ax_change.grid(True)
        
        # Create empty line objects for distance change plot
        self.line_psm1_change, = self.ax_change.plot([], [], 'r-', label='PSM1 Distance Change')
        self.line_psm2_change, = self.ax_change.plot([], [], 'b-', label='PSM2 Distance Change')
        self.ax_change.legend(loc='upper right')
        
        # Set tight layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)
    
    def update_plot(self):
        # Only update if all required values have been received
        if self.targets is not None and self.track_1 is not None and self.track_2 is not None:
            current_time = rospy.get_time() - self.start_time

            # Find closest points and distances for both PSMs
            closest_point_1, min_dist_1 = self.find_closest_point_and_dist(targets=self.targets, source=self.track_1)
            closest_point_2, min_dist_2 = self.find_closest_point_and_dist(targets=self.targets, source=self.track_2)

            # Append new data points for distances
            self.times.append(current_time)
            self.track_1_dist_data.append(min_dist_1)
            self.track_2_dist_data.append(min_dist_2)
            
            # Calculate and append distance changes (rate of change)
            if len(self.track_1_dist_data) > 1 and len(self.times) > 1:
                # Calculate time difference
                dt = self.times[-1] - self.times[-2]
                if dt > 0:  # Avoid division by zero
                    # Calculate distance difference
                    dd1 = self.track_1_dist_data[-1] - self.track_1_dist_data[-2]
                    dd2 = self.track_2_dist_data[-1] - self.track_2_dist_data[-2]
                    
                    # Calculate rate of change (m/s)
                    rate1 = dd1 / dt
                    rate2 = dd2 / dt
                    
                    self.track_1_dist_change.append(rate1)
                    self.track_2_dist_change.append(rate2)
                else:
                    # If dt is zero (unlikely), use the previous rate or zero
                    if len(self.track_1_dist_change) > 0:
                        self.track_1_dist_change.append(self.track_1_dist_change[-1])
                        self.track_2_dist_change.append(self.track_2_dist_change[-1])
                    else:
                        self.track_1_dist_change.append(0)
                        self.track_2_dist_change.append(0)
            else:
                # For the first point, rate of change is zero
                self.track_1_dist_change.append(0)
                self.track_2_dist_change.append(0)

            print("MIN DIST 1: ", min_dist_1)
            print("MIN DIST 2:", min_dist_2)
            if len(self.track_1_dist_change) > 0:
                print("DIST CHANGE RATE 1:", self.track_1_dist_change[-1])
                print("DIST CHANGE RATE 2:", self.track_2_dist_change[-1])

            # Update the distance plot
            self.line_psm1_dist.set_data(self.times, self.track_1_dist_data)
            self.line_psm2_dist.set_data(self.times, self.track_2_dist_data)
            self.ax_dist.relim()            # Recompute the data limits
            self.ax_dist.autoscale_view()   # Autoscale the view to the new data
            
            # Update the distance change plot
            # Use the same time points but exclude the first point since we don't have change data for it
            change_times = self.times
            self.line_psm1_change.set_data(change_times, self.track_1_dist_change)
            self.line_psm2_change.set_data(change_times, self.track_2_dist_change)
            self.ax_change.relim()          # Recompute the data limits
            self.ax_change.autoscale_view() # Autoscale the view to the new data

            plt.draw()
            plt.pause(0.001)

if __name__ == '__main__':
    try:
        figure_8_topic = '/ambf/env/World/figure_8'
        psm1_toolyawlink_topic = '/ambf/env/psm1/toolyawlink/State'
        psm2_toolyawlink_topic = '/ambf/env/psm2/toolyawlink/State'

        psm_tracker = ClosestPointDistPlotter(figure_8_topic, psm1_toolyawlink_topic, psm2_toolyawlink_topic)
        rate = rospy.Rate(10)  # 10 Hz update rate
        
        # Main loop: update the plot in the main thread
        while not rospy.is_shutdown():
            psm_tracker.update_plot()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
