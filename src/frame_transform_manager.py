"""
Frame Transform Manager (ROS 2 / rclpy version)

Handles coordinate transformations between AMBF simulation frames for the
dVRK surgical robot. Computes tool tip positions in both camera frame
(for visualization) and world frame (for analysis).

Subscribes to the CameraFrame pose and provides transform utilities for
converting PSM end-effector positions between reference frames.

Note: this class does NOT spin the node itself. The caller (e.g. a
rclpy.Node running its own executor in a background thread) must keep the
node spinning for `_frame_callback` to actually fire while this class's
constructor polls and waits for the first frame.
"""

import time
import threading
import numpy as np
import transformations as tft  # pip package, drop-in for ROS1's tf.transformations
from ambf_msgs.msg import RigidBodyState


def _xyzw_to_wxyz(q_xyzw):
    """transformations.py (unlike ROS1's tf.transformations) expects [w,x,y,z]."""
    x, y, z, w = q_xyzw
    return [w, x, y, z]


class FrameTransformManager:
    """
    Manages frame transformations by subscribing to AMBF rigid body states.
    Provides transformation from world frame to CameraFrame (phantom frame).
    """

    def __init__(self, node, reference_frame_name='CameraFrame', namespace='/ambf/env/phantom'):
        """
        Args:
            node: an rclpy Node (used for logging, clock, and creating the subscription).
                  Must be spun (directly or via an executor) elsewhere for callbacks to fire.
            reference_frame_name: Name of the reference frame (default: 'CameraFrame')
            namespace: AMBF namespace (default: '/ambf/env/phantom')
        """
        self.node = node
        self.reference_frame_name = reference_frame_name
        self.namespace = namespace

        # Frame transform data
        self.frame_position = None
        self.frame_quaternion = None
        self.frame_transform = None
        self.frame_inverse_transform = None

        # Tool offset (from PSM toolyawlink to tool tip)
        self.tool_offset_local = np.array([0, -0.009855, 0])

        # Lock for thread safety
        self.lock = threading.Lock()

        # Subscribe to reference frame
        self.frame_topic = f"{namespace}/{reference_frame_name}/State"
        self.frame_sub = node.create_subscription(
            RigidBodyState,
            self.frame_topic,
            self._frame_callback,
            1
        )

        node.get_logger().info(f"FrameTransformManager: Subscribing to {self.frame_topic}")

        # Wait for first frame update
        self._wait_for_frame(timeout=5.0)

    def _frame_callback(self, msg):
        """Callback for reference frame updates"""
        with self.lock:
            p = msg.pose.position
            q = msg.pose.orientation

            self.frame_position = np.array([p.x, p.y, p.z])
            self.frame_quaternion = np.array([q.x, q.y, q.z, q.w])

            # Compute transformation matrices
            q_wxyz = _xyzw_to_wxyz(self.frame_quaternion)
            angles = tft.euler_from_quaternion(q_wxyz)
            self.frame_transform = tft.compose_matrix(
                translate=self.frame_position,
                angles=angles
            )
            self.frame_inverse_transform = tft.inverse_matrix(self.frame_transform)

    def _wait_for_frame(self, timeout=5.0):
        """Wait for first frame update (polls; relies on the node being spun elsewhere)."""
        self.node.get_logger().info(f"Waiting for {self.reference_frame_name} frame data...")
        start_time = time.time()

        while True:
            if self.frame_transform is not None:
                self.node.get_logger().info(f"OK: {self.reference_frame_name} frame received")
                self.node.get_logger().info(
                    f"  Position: [{self.frame_position[0]:.4f}, "
                    f"{self.frame_position[1]:.4f}, "
                    f"{self.frame_position[2]:.4f}]"
                )
                return True

            if time.time() - start_time > timeout:
                self.node.get_logger().warn(f"Timeout waiting for {self.reference_frame_name} frame")
                self.node.get_logger().warn(f"Make sure {self.frame_topic} is publishing")
                return False

            time.sleep(0.1)

    def world_to_reference(self, point_world):
        """
        Transform point from world frame to reference frame

        Args:
            point_world: Point in world frame (3D numpy array or list)

        Returns:
            Point in reference frame (3D numpy array)
        """
        with self.lock:
            if self.frame_inverse_transform is None:
                self.node.get_logger().warn("Reference frame not available yet")
                return np.array(point_world)

            point_world = np.array(point_world)
            point_h = np.append(point_world, 1.0)
            local_h = np.dot(self.frame_inverse_transform, point_h)
            return local_h[:3]

    def reference_to_world(self, point_reference):
        """
        Transform point from reference frame to world frame

        Args:
            point_reference: Point in reference frame (3D numpy array or list)

        Returns:
            Point in world frame (3D numpy array)
        """
        with self.lock:
            if self.frame_transform is None:
                self.node.get_logger().warn("Reference frame not available yet")
                return np.array(point_reference)

            point_reference = np.array(point_reference)
            point_h = np.append(point_reference, 1.0)
            world_h = np.dot(self.frame_transform, point_h)
            return world_h[:3]

    def compute_tool_tip_positions(self, psm_state_msg):
        """
        Compute tool tip position in both world and reference frames from PSM state

        Args:
            psm_state_msg: RigidBodyState message from PSM toolyawlink

        Returns:
            tuple: (tool_tip_reference, tool_tip_world)
                - tool_tip_reference: Position in reference frame (phantom frame)
                - tool_tip_world: Position in world frame
        """
        # Extract pose from message
        p = psm_state_msg.pose.position
        q = psm_state_msg.pose.orientation

        position = np.array([p.x, p.y, p.z])
        quaternion_xyzw = [q.x, q.y, q.z, q.w]

        # Create transformation matrix for PSM
        tool_transform = tft.quaternion_matrix(_xyzw_to_wxyz(quaternion_xyzw))
        tool_transform[:3, 3] = position

        # Apply tool offset to get tool tip in world frame
        offset_homogeneous = np.append(self.tool_offset_local, 1.0)
        tool_tip_world = tool_transform @ offset_homogeneous
        tool_tip_world_pos = tool_tip_world[:3]

        # Transform to reference frame
        tool_tip_reference = self.world_to_reference(tool_tip_world_pos)

        return tool_tip_reference, tool_tip_world_pos

    def get_frame_info(self):
        """
        Get current frame information

        Returns:
            dict: Frame information including position, quaternion, and transform matrices
        """
        with self.lock:
            if self.frame_position is None:
                return None

            return {
                'name': self.reference_frame_name,
                'position': self.frame_position.copy(),
                'quaternion': self.frame_quaternion.copy(),
                'transform': self.frame_transform.copy(),
                'inverse_transform': self.frame_inverse_transform.copy()
            }

    def is_ready(self):
        """Check if frame transform is ready"""
        with self.lock:
            return self.frame_transform is not None


# Singleton instance for easy access
_frame_manager_instance = None
_frame_manager_lock = threading.Lock()


def get_frame_manager(node, reference_frame_name='CameraFrame', namespace='/ambf/env/phantom'):
    """
    Get or create singleton FrameTransformManager instance

    Args:
        node: an rclpy Node, spun elsewhere (see class docstring)
        reference_frame_name: Name of the reference frame (default: 'CameraFrame')
        namespace: AMBF namespace (default: '/ambf/env/phantom')

    Returns:
        FrameTransformManager instance
    """
    global _frame_manager_instance

    with _frame_manager_lock:
        if _frame_manager_instance is None:
            _frame_manager_instance = FrameTransformManager(
                node,
                reference_frame_name=reference_frame_name,
                namespace=namespace
            )
        return _frame_manager_instance


if __name__ == "__main__":
    import rclpy

    rclpy.init()
    test_node = rclpy.create_node('frame_transform_test')

    # Spin in a background thread so the subscription callback can fire
    # while _wait_for_frame() polls in the main thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(test_node,), daemon=True)
    spin_thread.start()

    # Create frame manager
    frame_mgr = FrameTransformManager(test_node, reference_frame_name='CameraFrame')

    # Get frame info
    info = frame_mgr.get_frame_info()
    if info:
        print("\n=== Frame Information ===")
        print(f"Frame: {info['name']}")
        print(f"Position: {info['position']}")
        print(f"Quaternion: {info['quaternion']}")

    # Test transformation
    test_point_world = np.array([0.0, 0.0, 0.0])
    test_point_ref = frame_mgr.world_to_reference(test_point_world)

    print(f"\nTest transformation:")
    print(f"World point: {test_point_world}")
    print(f"Reference frame point: {test_point_ref}")

    # Test reverse
    test_point_world_back = frame_mgr.reference_to_world(test_point_ref)
    print(f"Back to world: {test_point_world_back}")

    rclpy.shutdown()
