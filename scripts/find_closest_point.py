#!/usr/bin/env python3
import os
import time
import rospy
import rosbag
import argparse
import threading
import signal
import sys
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud, CompressedImage, Joy
from ambf_msgs.msg import RigidBodyState
from geometry_msgs.msg import Point32, Pose, Point
from std_msgs.msg import Float32, Float32MultiArray, ColorRGBA
import tf.transformations as tf


class ClosestPointDistPlotter:
    def __init__(
        self,
        bag_name: str,
        pattern: str,
        scale_value: float,
        trial_number: int,
        clutch_topic: str,
        targets_topic: str,
        track_1_topic: str,
        track_2_topic: str,
        camera_L_topic: str,
        camera_R_topic: str,
        track_1_dist_topic: str,
        track_2_dist_topic: str,
        gui_mode: bool = False
    ):
        rospy.init_node("closest_point_dist_plotter", anonymous=True)

        self.lock = threading.Lock()
        self.gui_mode = gui_mode
        self.shutdown_flag = False

        # Phantom pose
        self.phantom_position = np.array([0.0564242899, 0.2649616003, 0.7395174503])
        roll, pitch, yaw = 0.0, 0.0, 0.87266469
        self.phantom_quaternion = tf.quaternion_from_euler(roll, pitch, yaw)
        self.phantom_transform = tf.compose_matrix(
            translate=self.phantom_position,
            angles=tf.euler_from_quaternion(self.phantom_quaternion),
        )
        self.phantom_inverse_transform = tf.inverse_matrix(self.phantom_transform)
        self.tool_offset_local = np.array([0, -0.008855, 0])

        # Latest data
        self.targets = None
        self.track_1 = None
        self.track_2 = None
        self.clutch_count = 0
        self.last_clutch_state = 0

        # Time‑series buffers for plotting
        self.times = []
        self.track_1_dist_data = []
        self.track_2_dist_data = []
        self.start_time = rospy.get_time()

        # Bag setup
        py_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # Create folder name using trial number, pattern, and scale
        self.pattern = pattern  # Store the pattern
        self.scale_value = scale_value
        self.trial_number = trial_number

        # Format: trial{number}_{pattern}_scale{scale}
        bag_folder = f"/home/kyu17/intel_motion_scaling/data/{bag_name}"
        os.makedirs(bag_folder, exist_ok=True)
        self.bag_path = f"{bag_folder}/dist_video_clutch.bag"
        self.bag = rosbag.Bag(self.bag_path, "w")
        
        # Register shutdown hook
        rospy.on_shutdown(self.close_bag)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)


        # dynamic color pc
        self.colored_pc_topic = "/ambf/env/World/trajectory_proximity"
        self.colored_pc_pub = rospy.Publisher(
        self.colored_pc_topic, PointCloud, queue_size=10)
        # Register the colored point cloud topic with AMBF
        self._register_colored_pc_topic()
        # Store trajectory points for colored visualization
        self.stored_trajectory = None

        # Distance publishers
        self.psm1_dist_pub = rospy.Publisher(
            track_1_dist_topic, Float32, queue_size=10, latch=True
        )
        self.psm2_dist_pub = rospy.Publisher(
            track_2_dist_topic, Float32, queue_size=10, latch=True
        )

        # Subscribers
        self.sub_targets = rospy.Subscriber(
            targets_topic, PointCloud, self.targets_callback
        )
        self.sub_clutch = rospy.Subscriber(clutch_topic, Joy, self.clutch_callback)
        self.sub_track_1 = rospy.Subscriber(
            track_1_topic, RigidBodyState, self.track_1_callback
        )
        self.sub_track_2 = rospy.Subscriber(
            track_2_topic, RigidBodyState, self.track_2_callback
        )
        self.sub_camera_1 = rospy.Subscriber(
            camera_L_topic, CompressedImage, self.camera_L_callback
        )
        self.sub_camera_2 = rospy.Subscriber(
            camera_R_topic, CompressedImage, self.camera_R_callback
        )

        # Plot setup - use non-interactive backend in GUI mode
        if self.gui_mode:
            matplotlib.use('Agg')  # Non-interactive backend
            self.fig = None
            self.ax = None
            rospy.loginfo("Running in GUI mode - plotting disabled for clean shutdown")
        else:
            self.setup_plotting()
            rospy.loginfo("Running in interactive mode - press Enter to exit")

    def _register_colored_pc_topic(self):
        """Register the colored point cloud topic with AMBF"""
        topics_param = '/ambf/env/World/point_cloud_topics'
        pc_topics = rospy.get_param(topics_param, [])
        
        if self.colored_pc_topic not in pc_topics:
            pc_topics.append(self.colored_pc_topic)
            rospy.set_param(topics_param, pc_topics)
            rospy.loginfo(f'Registered colored point cloud topic {self.colored_pc_topic} with AMBF')
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        rospy.loginfo(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_flag = True
        
        # Close matplotlib if it's open
        if not self.gui_mode and self.fig is not None:
            plt.close('all')
        
        # Trigger ROS shutdown
        rospy.signal_shutdown("Received interrupt signal")
        
        # Exit immediately to prevent force kill
        sys.exit(0)

    # -------------- Callbacks ---------------
    def targets_callback(self, msg):
        if not self.shutdown_flag:
            self.targets = np.array([[p.x, p.y, p.z] for p in msg.points])
            # Store for colored visualization
            self.stored_trajectory = [[p.x, p.y, p.z] for p in msg.points]

    def clutch_callback(self, msg):
        if self.shutdown_flag:
            return
        with self.lock:
            current_state = msg.buttons[0] if msg.buttons else 0
            if current_state == 1 and self.last_clutch_state == 0:
                self.clutch_count += 1
                rospy.loginfo("Clutch event count: %d", self.clutch_count)
                self.bag.write("/footpedal/clutch", msg, rospy.Time.now())
            self.last_clutch_state = current_state

    def track_1_callback(self, msg):
        if not self.shutdown_flag:
            # Get position and orientation from tool yaw link
            p = msg.pose.position
            q = msg.pose.orientation
            
            # Create transformation matrix for tool yaw link
            position = np.array([p.x, p.y, p.z])
            quaternion = [q.x, q.y, q.z, q.w]
            
            tool_transform = tf.quaternion_matrix(quaternion)
            tool_transform[:3, 3] = position
            
            # Apply tool offset in local frame (Y-axis forward for grippers)
            offset_homogeneous = np.append(self.tool_offset_local, 1.0)
            tool_tip_world = tool_transform @ offset_homogeneous
            
            # Transform to phantom frame
            self.track_1 = self.transform_to_phantom_frame(tool_tip_world[:3])

    def track_2_callback(self, msg):
        if not self.shutdown_flag:
            # Get position and orientation from tool yaw link
            p = msg.pose.position
            q = msg.pose.orientation
            
            # Create transformation matrix for tool yaw link
            position = np.array([p.x, p.y, p.z])
            quaternion = [q.x, q.y, q.z, q.w]
            
            tool_transform = tf.quaternion_matrix(quaternion)
            tool_transform[:3, 3] = position
            
            # Apply tool offset in local frame (Y-axis forward for grippers)
            offset_homogeneous = np.append(self.tool_offset_local, 1.0)
            tool_tip_world = tool_transform @ offset_homogeneous
            
            # Transform to phantom frame
            self.track_2 = self.transform_to_phantom_frame(tool_tip_world[:3])

    def camera_L_callback(self, msg):
        if self.shutdown_flag:
            return
        with self.lock:
            self.bag.write("/cameraL/video", msg, rospy.Time.now())

    def camera_R_callback(self, msg):
        if self.shutdown_flag:
            return
        with self.lock:
            self.bag.write("/cameraR/video", msg, rospy.Time.now())

    # -------------- Helpers -----------------
    def transform_to_phantom_frame(self, position):
        pos_h = np.append(position, 1.0)
        local_h = np.dot(self.phantom_inverse_transform, pos_h)
        return local_h[:3]

    def find_closest_point_and_dist(self, targets, source):
        dists = np.linalg.norm(targets - source, axis=1)
        idx = np.argmin(dists)
        return targets[idx], dists[idx], idx  # Now also returns index

    def publish_colored_point_cloud(self, center_idx, distance, psm_id):
        """Publish colored point cloud based on proximity"""
        if self.stored_trajectory is None or distance >= 0.02:
            # Clear colored point cloud if too far
            empty_msg = PointCloud()
            empty_msg.header.frame_id = "phantomBODY phantom"
            empty_msg.header.stamp = rospy.Time.now()
            self.colored_pc_pub.publish(empty_msg)
            return
        
        # Calculate number of points based on distance
        # Linear scaling: 0.02m -> 2000 points, 0.001m -> 100 points
        if distance <= 0.001:
            num_points = 100
        elif distance >= 0.02:
            num_points = 1000
        else:
            # Linear interpolation
            num_points = int(100 + (1000 - 20) * (0.02 - distance) / (0.02 - 0.001))
        
        # Calculate range of points to display
        total_trajectory_points = len(self.stored_trajectory)
        half_points = num_points // 2
        
        # Calculate start and end indices with wrapping for closed trajectories
        start_idx = center_idx - half_points
        end_idx = center_idx + half_points
        
        # Create point cloud message
        colored_msg = PointCloud()
        colored_msg.header.frame_id = "phantomBODY phantom"
        colored_msg.header.stamp = rospy.Time.now()
        
        # Extract points with wrapping
        for i in range(start_idx, end_idx + 1):
            # Wrap around for closed trajectories
            wrapped_idx = i % total_trajectory_points
            if 0 <= wrapped_idx < total_trajectory_points:
                point = Point32()
                point.x = self.stored_trajectory[wrapped_idx][0]
                point.y = self.stored_trajectory[wrapped_idx][1]
                point.z = self.stored_trajectory[wrapped_idx][2]
                colored_msg.points.append(point)
        
        # Publish the colored point cloud
        self.colored_pc_pub.publish(colored_msg)
        
        # Log for debugging (optional, comment out for production)
        rospy.logdebug(f"PSM{psm_id} proximity: {distance:.4f}m, showing {num_points} colored points")

    # -------------- Plotting ----------------
    def setup_plotting(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle("PSM Distance from Trajectory")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Distance (m)")
        self.ax.set_ylim(0, 0.01)
        self.ax.grid(True)
        self.line_psm1_dist, = self.ax.plot([], [], "r-", label="PSM1 Distance")
        self.line_psm2_dist, = self.ax.plot([], [], "b-", label="PSM2 Distance")
        self.ax.legend(loc="upper right")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    def update_plot(self):
        if self.shutdown_flag:
            return
            
        with self.lock:
            if (
                    self.targets is None
                    or self.track_1 is None
                    or self.track_2 is None
            ):
                return

            t_now = rospy.get_time() - self.start_time
            
            # Find closest points and distances with indices
            _, dist1, idx1 = self.find_closest_point_and_dist(self.targets, self.track_1)
            _, dist2, idx2 = self.find_closest_point_and_dist(self.targets, self.track_2)

            # Publish distances
            self.psm1_dist_pub.publish(Float32(data=dist1))
            self.psm2_dist_pub.publish(Float32(data=dist2))
            
            # Determine which PSM is closer and publish colored point cloud
            if dist1 < dist2:
                self.publish_colored_point_cloud(idx1, dist1, 1)
            else:
                self.publish_colored_point_cloud(idx2, dist2, 2)

            # Log & bag
            rospy.loginfo("t=%.2f  PSM1=%.4f  PSM2=%.4f", t_now, dist1, dist2)
            dist_msg = Float32MultiArray(data=[t_now, dist1, dist2])
            self.bag.write("/min_distances", dist_msg, rospy.Time.now())

            # Only update plot if not in GUI mode
            if not self.gui_mode and self.fig is not None:
                self.times.append(t_now)
                self.track_1_dist_data.append(dist1)
                self.track_2_dist_data.append(dist2)
                self.line_psm1_dist.set_data(self.times, self.track_1_dist_data)
                self.line_psm2_dist.set_data(self.times, self.track_2_dist_data)
                self.ax.relim()
                self.ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

    # -------------- Shutdown / util ---------
    def close_bag(self):
        with self.lock:
            if self.bag is not None:
                try:
                    self.bag.close()
                    rospy.loginfo(f"Rosbag closed and indexed successfully: {self.bag_path}")
                except Exception as e:
                    rospy.logerr(f"Error closing bag: {e}")
                finally:
                    self.bag = None

    def listen_for_exit(self):
        try:
            input("Press Enter to exit...\n")
            rospy.signal_shutdown("User requested shutdown.")
        except EOFError:
            # Handle case where stdin is closed
            pass


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Collect closest‑point distances and camera feeds."
        )
        parser.add_argument(
            "bag_name",
            help="Folder name for the bag (e.g. ykc/1)",
        )
        parser.add_argument(
            "--gui-mode",
            action="store_true",
            help="Run in GUI mode (disables interactive keyboard exit and plotting)"
        )

        parser.add_argument("--pattern", help="Trajectory pattern being used", default="unknown")
        parser.add_argument("--scale", type=float, help="Motion scaling factor", default=1.0)
        parser.add_argument("--trial", type=int, help="Trial number", default=1)

        args = parser.parse_args()

        # Topics - Updated to use unified trajectory topic
        trajectory_topic = "/ambf/env/World/trajectory"
        clutch_topic = "/footpedals/clutch"
        psm1_topic = "/ambf/env/psm1/toolyawlink/State"
        psm2_topic = "/ambf/env/psm2/toolyawlink/State"
        camL_topic = "/ambf/env/cameras/cameraL/ImageData/compressed"
        camR_topic = "/ambf/env/cameras/cameraR/ImageData/compressed"
        psm1_dist_topic = "/ambf/env/psm1/trajectory_dist"
        psm2_dist_topic = "/ambf/env/psm2/trajectory_dist"

        tracker = ClosestPointDistPlotter(
            bag_name=args.bag_name,
            pattern=args.pattern,
            scale_value=args.scale,
            trial_number=args.trial,
            clutch_topic=clutch_topic,
            targets_topic=trajectory_topic,
            track_1_topic=psm1_topic,
            track_2_topic=psm2_topic,
            camera_L_topic=camL_topic,
            camera_R_topic=camR_topic,
            track_1_dist_topic=psm1_dist_topic,
            track_2_dist_topic=psm2_dist_topic,
            gui_mode=args.gui_mode  # Pass the GUI mode flag
        )

        # Only start the keyboard listener if NOT in GUI mode
        if not args.gui_mode:
            # Allow keyboard exit in interactive mode
            threading.Thread(target=tracker.listen_for_exit, daemon=True).start()

        rate = rospy.Rate(5)  # 5 Hz
        while not rospy.is_shutdown() and not tracker.shutdown_flag:
            tracker.update_plot()
            rate.sleep()

        # Ensure clean exit
        rospy.loginfo("Main loop ended, performing final cleanup...")
        tracker.close_bag()
        
        # Give a moment for cleanup to complete
        time.sleep(0.1)
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received, shutting down cleanly")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        # Ensure we exit cleanly
        sys.exit(0)