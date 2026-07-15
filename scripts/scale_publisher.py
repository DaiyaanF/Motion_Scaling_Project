#!/usr/bin/env python3
"""
Simple manual motion scale publisher for dVRK
Publishes a fixed scale value to /ambf/teleop/scale topic
Can be controlled via command line argument or modified in real-time
"""

import rospy
import argparse
from std_msgs.msg import Float32


def main():
    parser = argparse.ArgumentParser(
        description="Publish a fixed motion scaling factor for dVRK teleoperation"
    )
    
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Motion scaling factor (default: 1.0, range: 0.01-1.5)"
    )
    
    parser.add_argument(
        "--rate", type=int, default=10,
        help="Publishing rate in Hz (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Validate scale
    scale = max(0.01, min(1.5, args.scale))
    if scale != args.scale:
        print(f"Scale value clamped to {scale}")
    
    try:
        rospy.init_node("manual_motion_scaler", anonymous=False)
        
        # Publisher with latching
        scale_pub = rospy.Publisher(
            "/ambf/teleop/scale", 
            Float32, 
            queue_size=1,
            latch=True
        )
        
        # Create message
        scale_msg = Float32()
        scale_msg.data = scale
        
        rospy.loginfo(f"Publishing motion scale: {scale} at {args.rate} Hz")
        
        # Publishing loop
        rate = rospy.Rate(args.rate)
        while not rospy.is_shutdown():
            scale_pub.publish(scale_msg)
            rate.sleep()
            
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()