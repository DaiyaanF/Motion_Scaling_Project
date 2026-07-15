import rospy
import json
import numpy as np
import itertools
import threading
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Joy

class AttributeManager:
    def __init__(self, config_file=None):
        rospy.init_node('attribute_manager')
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'shapes': ['circle', 'line'],
                'size_ranges': {
                    'circle': {'min': 0.01, 'max': 0.03, 'step': 0.02},
                    'line': {'min': 0.02, 'max': 0.04, 'step': 0.02}
                },
                'scaling_factors': [0.1, 1.25],
                'rotations': [{'roll': 0, 'pitch': 0, 'yaw': 0}],
                'positions': [{'x': 0, 'y': 0, 'z': 0.04}],
                'repetitions': 1
            }
        
        # Generate all combinations with dot initialization
        self.combinations = self._generate_combinations_with_dot()
        self.current_index = 0
        self.current_combination = self.combinations[0] if self.combinations else None
        self.transition_in_progress = False
        self.experiment_completed = False
        
        # Publishers
        self.attr_pub = rospy.Publisher('/experiment/attributes', String, queue_size=10, latch=True)
        self.scale_pub = rospy.Publisher('/ambf/teleop/scale', Float32, queue_size=10, latch=True)
        
        # Subscriber for footpedal
        self.footpedal_sub = rospy.Subscriber('/footpedals/camera', Joy, self.footpedal_callback)
        
        # Wait a moment for other nodes to initialize
        rospy.sleep(0.5)
        
        # Publish initial state (the dot)
        self.publish_current_attributes()
        
        # Publishing timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        rospy.loginfo(f"AttributeManager initialized with {len(self.combinations)} combinations (including dot)")
        
    def _generate_combinations_with_dot(self):
        """Generate all combinations with dot as first pattern"""
        combinations = []
        
        # ALWAYS start with dot pattern - fixed parameters
        dot_combo = {
            'combination_id': 1,
            'repetition': 0,  # Special marker for initialization pattern
            'shape': 'dot',
            'size': 0.005,  # 5mm dot
            'scale': 1.0,
            'rotation': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'position': {'x': 0, 'y': 0, 'z': 0.04}
        }
        combinations.append(dot_combo)
        
        # Generate normal combinations
        normal_combinations = []
        combination_id = 2  # Start from 2 since dot is 1
        
        for rep in range(self.config['repetitions']):
            for shape in self.config['shapes']:
                size_range = self.config['size_ranges'].get(shape, {'min': 0.02, 'max': 0.05, 'step': 0.01})
                
                # Fixed: Use np.linspace instead of np.arange to avoid floating point issues
                min_size = size_range['min']
                max_size = size_range['max']
                step_size = size_range['step']
                num_steps = int(round((max_size - min_size) / step_size)) + 1
                sizes = np.linspace(min_size, max_size, num_steps)
                
                # Log the sizes for debugging
                rospy.loginfo(f"Shape {shape}: generating {len(sizes)} sizes: {sizes}")
                
                for size in sizes:
                    for scale in self.config['scaling_factors']:
                        for rotation in self.config['rotations']:
                            for position in self.config['positions']:
                                combo = {
                                    'combination_id': combination_id,
                                    'repetition': rep + 1,
                                    'shape': shape,
                                    'size': float(size),
                                    'scale': float(scale),
                                    'rotation': rotation,
                                    'position': position
                                }
                                normal_combinations.append(combo)
                                combination_id += 1
        
        # Randomize only the normal combinations
        np.random.shuffle(normal_combinations)
        
        # Add randomized combinations after the dot
        combinations.extend(normal_combinations)
        
        # Reassign combination IDs to maintain sequence
        for i, combo in enumerate(combinations):
            combo['combination_id'] = i + 1
            
        return combinations
    
    def footpedal_callback(self, msg):
        """Handle camera footpedal press"""
        with self.lock:
            if self.experiment_completed:
                rospy.loginfo("Experiment already completed - ignoring footpedal")
                return
                
            if self.transition_in_progress:
                return
                
            if msg.buttons and msg.buttons[0] == 1:  # Camera button pressed
                self.transition_in_progress = True
                rospy.loginfo("Camera footpedal pressed - transitioning in 1 second")
                rospy.Timer(rospy.Duration(1.0), self.delayed_transition, oneshot=True)
    
    def delayed_transition(self, event):
        """Execute transition after delay"""
        self.next_combination()
        
    def next_combination(self):
        """Move to the next combination"""
        with self.lock:
            # If we're at or beyond the last combination, mark as complete
            if self.current_index >= len(self.combinations) - 1:
                if not self.experiment_completed:
                    rospy.loginfo("All combinations completed! Experiment finished.")
                    self.experiment_completed = True
                    
                    # Publish completion status
                    completion_msg = {
                        'experiment_completed': True,
                        'total_combinations': len(self.combinations),
                        'timestamp': rospy.Time.now().to_sec()
                    }
                    self.attr_pub.publish(String(data=json.dumps(completion_msg)))
                self.transition_in_progress = False
                return
                
            # Normal transition to next combination
            self.current_index = self.current_index + 1
            self.current_combination = self.combinations[self.current_index]
            self.transition_in_progress = False
            
            # Log transition (special message for leaving dot pattern)
            if self.current_index == 1:
                rospy.loginfo("Leaving initialization dot pattern, starting actual experiment")
            
            rospy.loginfo(f"Switched to combination {self.current_index + 1}/{len(self.combinations)}: "
                         f"{self.current_combination['shape']} - "
                         f"size: {self.current_combination['size']:.3f} - "
                         f"scale: {self.current_combination['scale']:.2f}")
    
    def publish_current_attributes(self):
        """Publish current combination attributes"""
        if self.current_combination and not self.experiment_completed:
            # Add metadata
            data = self.current_combination.copy()
            data['current_index'] = self.current_index
            data['total_combinations'] = len(self.combinations)
            data['timestamp'] = rospy.Time.now().to_sec()
            data['is_initialization'] = (self.current_index == 0)  # Mark if this is the dot
            
            # Publish attributes
            self.attr_pub.publish(String(data=json.dumps(data)))
            
            # Publish scale
            scale_msg = Float32()
            scale_msg.data = self.current_combination['scale']
            self.scale_pub.publish(scale_msg)
    
    def timer_callback(self, event):
        """Periodic publishing"""
        self.publish_current_attributes()
    
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Attribute Manager for dVRK User Study')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()
    
    try:
        manager = AttributeManager(config_file=args.config)
        manager.spin()
    except rospy.ROSInterruptException:
        pass