import rospy
import math
from std_msgs.msg import Float32


def compute_scaling(distance: float) -> float:
    """Return the motion‑scaling factor from a distance value."""
    return math.exp(-750.0 * distance) + 0.01


class MotionScaler:
    """
    Subscribes to `distance_topic`, publishes scale factor on `scaling_topic`.
    """

    def __init__(self, distance_topic: str, scaling_topic: str):
        # Publisher
        self.scaling_pub = rospy.Publisher(
            scaling_topic, Float32, queue_size=10, latch=True
        )
        # Subscriber
        self.distance_sub = rospy.Subscriber(
            distance_topic, Float32, self.distance_callback, queue_size=10
        )
        rospy.loginfo(
            f"[motion_scaler] listening on '{distance_topic}', "
            f"publishing to '{scaling_topic}'."
        )

    def distance_callback(self, msg: Float32):
        factor = compute_scaling(msg.data)
        self.scaling_pub.publish(Float32(data=factor))
        rospy.logdebug(f"distance: {msg.data:.6f}  →  factor: {factor:.6f}")


# --------------------------------------------------------------------------- #
#  Everything below happens when the script is invoked (no separate function) #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        rospy.init_node("motion_scaler", anonymous=False)

        # ── Define topics here ─────────
        psm1_dist_topic = "/ambf/env/psm1/figure_dist"
        psm2_dist_topic = "/ambf/env/psm2/figure_dist"
        factor_topic = "/ambf/teleop/scale"
        # --------------------------------------------------------------------

        MotionScaler(psm2_dist_topic, factor_topic)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
