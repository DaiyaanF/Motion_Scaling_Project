import argparse
import cv2
import numpy as np
import os
import rosbag
import re
from pathlib import Path
import csv
import rospy


def sanitize_topic_name(topic):
    """
    Sanitize a topic name to be filesystem-friendly.
    E.g., "/camera/image_raw" becomes "camera_image_raw".
    """
    # Remove leading slash and replace other slashes with underscores.
    return re.sub(r'^/', '', topic).replace('/', '_')


def read_and_save_mp4(bag_file, topic, output_file, fps):
    """
    Extract messages from a specific topic in a ROS bag and save as an MP4 video.

    Parameters:
        bag_file (str): Path to the input ROS bag file.
        topic (str): Topic to extract messages from.
        output_file (str): Path for the output MP4 video file.
        fps (float): Frames per second (FPS) for the output video.
    """
    bag = rosbag.Bag(bag_file, 'r')
    frames = []

    print("Processing topic '{}'...".format(topic))
    
    # Filter messages for this topic
    for t, msg, _ in bag.read_messages(topics=[topic]):
        # Convert the compressed image bytes into a numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)
        # Decode the image (OpenCV decodes to BGR format)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            print("Warning: A message on topic '{}' could not be decoded; skipping.".format(topic))
            continue
        frames.append(image)
    
    bag.close()

    if not frames:
        print("No valid image frames were found for topic '{}'.".format(topic))
        return

    # Determine the frame dimensions from the first frame
    height, width, _ = frames[0].shape
    frame_size = (width, height)

    # Set up the video writer for MP4 format using the 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    print("Writing video for topic '{}' to '{}' at {} FPS...".format(topic, output_file, fps))
    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()
    print("Video for topic '{}' saved successfully as '{}'.".format(topic, output_file))


def read_dist(bag_path, topic):
    """
    Reads distance events from a ROS bag file and returns them as a list of tuples.
    
    Each message on the topic is expected to be a Float32MultiArray with the following format:
        data[0]: Time (in seconds)
        data[1]: Distance from PSM1
        data[2]: Distance from PSM2
    
    Parameters:
        bag_path (str): Path to the ROS bag file.
        topic (str): Topic to read the distance events from.
        
    Returns:
        list of tuples: Each tuple is (time, psm1_distance, psm2_distance).
    """
    distance_events = []
    
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            if len(msg.data) >= 3:
                event_time = msg.data[0]
                dist_psm1 = msg.data[1]
                dist_psm2 = msg.data[2]
                distance_events.append((event_time, dist_psm1, dist_psm2))
            else:
                rospy.logwarn("Message on '%s' has unexpected data length: %d", topic, len(msg.data))
    
    return distance_events


def save_dist_to_csv(bag_path, csv_path, topic):
    """
    Reads distance events from the specified ROS bag and writes them to a CSV file.
    
    The CSV file contains three columns:
        - time: Timestamp when the event occurred (seconds)
        - psm1_distance: Closest distance from PSM1
        - psm2_distance: Closest distance from PSM2
    
    Parameters:
        bag_path (str): Path to the input ROS bag file.
        csv_path (str): Path to the output CSV file.
        topic (str): The ROS topic from which to extract the distance events.
    """
    events = read_dist(bag_path, topic)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(['time', 'psm1_distance', 'psm2_distance'])
        # Write each event row
        for event in events:
            writer.writerow(event)
    
    print("Saved distance events to:", csv_path)


def read_clutch(bag_file, topic):
    """
    Reads a rosbag file and extracts clutch event times and a cumulative clutch count.
    
    Each message on the '/footpedal/clutch' topic is assumed to correspond
    to a clutch event (i.e. a rising edge from 0 to 1).
    
    Parameters:
        bag_file (str): Path to the rosbag file.
        topic (str): Topic to read the clutch events from.
        
    Returns:
        list of tuples: Each tuple contains (event_time, clutch_count).
                        - event_time: The time stamp (in seconds) from the bag for the event.
                        - clutch_count: The cumulative number of clutch events recorded up to that event.
    """
    clutch_events = []
    clutch_count = 0
    
    # Open the bag file in read mode
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate only over messages in the '/footpedal/clutch' topic
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            clutch_count += 1
            # Convert the ROS time to seconds
            event_time = t.to_sec()
            clutch_events.append((event_time, clutch_count))
    
    return clutch_events


def save_clutch_to_csv(bag_path, csv_path, topic):
    """
    Reads clutch events from a ROS bag and saves them to a CSV file.
    
    The CSV will contain two columns:
        - event_time: The timestamp of the event (in seconds)
        - clutch_count: The cumulative number of clutch events recorded
    
    Parameters:
        bag_path (str): Path to the ROS bag file.
        csv_path (str): Path to the output CSV file.
        topic (str): The ROS topic to read the clutch events from.
    """
    events = read_clutch(bag_path, topic)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(['event_time', 'clutch_count'])
        # Write each event row
        for event in events:
            writer.writerow(event)
    
    print("Saved clutch events to:", csv_path)