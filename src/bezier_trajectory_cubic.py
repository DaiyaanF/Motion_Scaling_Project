"""
Cubic Bezier Trajectory Generator

Generates smooth 3D trajectories using cubic Bezier curves for trajectory
tracking experiments. Provides control over arc length, curvature, and
workspace bounds. 
"""

import numpy as np
from scipy.optimize import minimize_scalar


class CubicBezierSegment:
    def __init__(self, P0, P1, P2, P3):
        self.P0 = np.array(P0)
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.P3 = np.array(P3)
        self.arc_length = None
        self.curvature_max = None
        self.curvature_avg = None
        self.euclidean_distance = None
        self.arc_ratio = None

    def compute_metrics(self, num_samples=1000):
        self.arc_length = self._calculate_arc_length(num_samples)
        self.curvature_max, self.curvature_avg = self._calculate_curvature(num_samples)
        self.euclidean_distance = np.linalg.norm(self.P3 - self.P0)
        self.arc_ratio = self.arc_length / self.euclidean_distance if self.euclidean_distance > 0 else 1.0

    def _calculate_arc_length(self, num_samples):
        t_values = np.linspace(0, 1, num_samples)
        derivatives = self._bezier_derivative(t_values)
        speeds = np.linalg.norm(derivatives, axis=1)
        arc_length = np.trapz(speeds, t_values)
        return arc_length

    def _calculate_curvature(self, num_samples):
        t_values = np.linspace(0.01, 0.99, num_samples)
        
        first_deriv = self._bezier_derivative(t_values)
        second_deriv = self._bezier_second_derivative(t_values)
        
        cross_products = np.cross(first_deriv, second_deriv)
        cross_norms = np.linalg.norm(cross_products, axis=1)
        first_deriv_norms = np.linalg.norm(first_deriv, axis=1)
        
        curvatures = cross_norms / (first_deriv_norms ** 3 + 1e-10)
        
        return np.max(curvatures), np.mean(curvatures)

    def _bezier_derivative(self, t):
        t = np.atleast_1d(t)
        derivatives = np.zeros((len(t), 3))
        for i, t_val in enumerate(t):
            derivatives[i] = (
                3 * (1 - t_val)**2 * (self.P1 - self.P0) +
                6 * (1 - t_val) * t_val * (self.P2 - self.P1) +
                3 * t_val**2 * (self.P3 - self.P2)
            )
        return derivatives

    def _bezier_second_derivative(self, t):
        t = np.atleast_1d(t)
        second_derivs = np.zeros((len(t), 3))
        for i, t_val in enumerate(t):
            second_derivs[i] = (
                6 * (1 - t_val) * (self.P2 - 2*self.P1 + self.P0) +
                6 * t_val * (self.P3 - 2*self.P2 + self.P1)
            )
        return second_derivs

    def sample_points(self, num_points=100):
        t_values = np.linspace(0, 1, num_points)
        points = []
        for t in t_values:
            point = (
                (1 - t)**3 * self.P0 +
                3 * (1 - t)**2 * t * self.P1 +
                3 * (1 - t) * t**2 * self.P2 +
                t**3 * self.P3
            )
            points.append(tuple(point))
        return points
    
    def get_curvature_at_t(self, t_values):
        """Per-point curvature at the given parametric t values (vectorized).
        Pass the same t_values used for sample_points() to get a curvature
        array aligned index-for-index with the sampled points."""
        t_values = np.atleast_1d(np.clip(t_values, 0.001, 0.999))
        first_deriv = self._bezier_derivative(t_values)
        second_deriv = self._bezier_second_derivative(t_values)
        cross_products = np.cross(first_deriv, second_deriv)
        cross_norms = np.linalg.norm(cross_products, axis=1)
        first_deriv_norms = np.linalg.norm(first_deriv, axis=1)
        return cross_norms / (first_deriv_norms ** 3 + 1e-10)

    def get_point_at_t(self, t):
        return (
            (1 - t)**3 * self.P0 +
            3 * (1 - t)**2 * t * self.P1 +
            3 * (1 - t) * t**2 * self.P2 +
            t**3 * self.P3
        )


class CubicBezierTrajectoryGenerator:
    def __init__(self, boundary_size=0.1, bounding_box=None, max_attempts=100):
        """
        Args:
            boundary_size: Local randomization range for control points
            bounding_box: Overall workspace bounds dict with keys:
                         'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
                         If None, only local boundary_size is used
            max_attempts: Max generation attempts before fallback
        """
        self.boundary_size = boundary_size
        self.bounding_box = bounding_box
        self.max_attempts = max_attempts

    def generate_segment(self, start_point, target_arc_length, end_point=None, curvature_range=None):
        start_point = np.array(start_point)
        
        for attempt in range(self.max_attempts):
            if end_point is None:
                P1, P2, P3 = self._generate_random_points(start_point)
            else:
                P1, P2 = self._generate_intermediate_points(start_point, np.array(end_point))
                P3 = np.array(end_point)
            
            scale_factor = self._find_scale_for_arc_length(start_point, P1, P2, P3, target_arc_length)
            
            if scale_factor is None:
                continue
            
            direction_P1 = P1 - start_point
            direction_P2 = P2 - start_point
            direction_P3 = P3 - start_point
            
            P1_scaled = start_point + direction_P1 * scale_factor
            P2_scaled = start_point + direction_P2 * scale_factor
            P3_scaled = start_point + direction_P3 * scale_factor
            
            # Check local bounds
            if not self._within_bounds_relative(P1_scaled, start_point) or \
               not self._within_bounds_relative(P2_scaled, start_point) or \
               not self._within_bounds_relative(P3_scaled, start_point):
                continue
            
            # Check global bounding box
            if self.bounding_box is not None:
                segment_temp = CubicBezierSegment(start_point, P1_scaled, P2_scaled, P3_scaled)
                if not self._segment_within_bounding_box(segment_temp):
                    continue
            
            segment = CubicBezierSegment(start_point, P1_scaled, P2_scaled, P3_scaled)
            segment.compute_metrics()
            
            if abs(segment.arc_length - target_arc_length) < 0.001:
                if curvature_range is not None:
                    if curvature_range[0] <= segment.curvature_max <= curvature_range[1]:
                        return segment
                else:
                    return segment
        
        # Fallback: generate simple segment
        angle = np.random.uniform(0, 2*np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), np.random.uniform(-0.2, 0.2)])
        direction = direction / np.linalg.norm(direction)
        
        P1 = start_point + direction * target_arc_length * 0.2
        P2 = start_point + direction * target_arc_length * 0.6
        P3 = start_point + direction * target_arc_length
        
        # Clamp to bounding box if defined
        if self.bounding_box is not None:
            P1 = self._clamp_to_bounding_box(P1)
            P2 = self._clamp_to_bounding_box(P2)
            P3 = self._clamp_to_bounding_box(P3)
        
        segment = CubicBezierSegment(start_point, P1, P2, P3)
        segment.compute_metrics()
        return segment

    def _generate_random_points(self, start_point):
        half_bound = self.boundary_size / 2
        
        offset_P1 = np.random.uniform(-half_bound, half_bound, 3)
        offset_P2 = np.random.uniform(-half_bound, half_bound, 3)
        offset_P3 = np.random.uniform(-half_bound, half_bound, 3)
        
        P1 = start_point + offset_P1
        P2 = start_point + offset_P2
        P3 = start_point + offset_P3
        
        return P1, P2, P3
    
    def _generate_intermediate_points(self, P0, P3):
        direction = P3 - P0
        length = np.linalg.norm(direction)
        
        perp1 = np.random.uniform(-length*0.3, length*0.3, 3)
        perp2 = np.random.uniform(-length*0.3, length*0.3, 3)
        
        P1 = P0 + direction * 0.33 + perp1
        P2 = P0 + direction * 0.67 + perp2
        
        return P1, P2

    def _find_scale_for_arc_length(self, P0, P1, P2, P3, target_length):
        def arc_length_diff(scale):
            P1_s = P0 + (P1 - P0) * scale
            P2_s = P0 + (P2 - P0) * scale
            P3_s = P0 + (P3 - P0) * scale
            temp_segment = CubicBezierSegment(P0, P1_s, P2_s, P3_s)
            current_length = temp_segment._calculate_arc_length(100)
            return abs(current_length - target_length)
        
        result = minimize_scalar(arc_length_diff, bounds=(0.1, 5.0), method='bounded')
        
        if result.fun < 0.001:
            return result.x
        return None

    def _within_bounds_relative(self, point, reference_point):
        relative_pos = point - reference_point
        half_bound = self.boundary_size / 2
        return np.all(np.abs(relative_pos) <= half_bound)
    
    def _point_within_bounding_box(self, point):
        if self.bounding_box is None:
            return True
        bb = self.bounding_box
        return (bb['x_min'] <= point[0] <= bb['x_max'] and
                bb['y_min'] <= point[1] <= bb['y_max'] and
                bb['z_min'] <= point[2] <= bb['z_max'])
    
    def _segment_within_bounding_box(self, segment, num_samples=50):
        if self.bounding_box is None:
            return True
        
        # Check control points
        for p in [segment.P0, segment.P1, segment.P2, segment.P3]:
            if not self._point_within_bounding_box(p):
                return False
        
        # Check sampled points along curve
        points = segment.sample_points(num_samples)
        for p in points:
            if not self._point_within_bounding_box(np.array(p)):
                return False
        
        return True
    
    def _clamp_to_bounding_box(self, point):
        if self.bounding_box is None:
            return point
        bb = self.bounding_box
        return np.array([
            np.clip(point[0], bb['x_min'], bb['x_max']),
            np.clip(point[1], bb['y_min'], bb['y_max']),
            np.clip(point[2], bb['z_min'], bb['z_max'])
        ])


class PathCoverageTracker:
    """Track path coverage using bins evenly spaced by ARC LENGTH along the Bezier segment.
    
    Each bin represents a section of the path. A bin is marked visited when the PSM
    comes close to ANY point within that bin's section of the path, not just the center.
    """
    
    def __init__(self, segment, num_bins=20, coverage_threshold=0.85, visit_threshold=0.003):
        """
        Args:
            segment: CubicBezierSegment to track
            num_bins: Number of bins to divide the segment into
            coverage_threshold: Fraction of bins that must be visited (0.0-1.0)
            visit_threshold: Distance threshold to consider a bin visited (meters)
        """
        self.segment = segment
        self.num_bins = num_bins
        self.coverage_threshold = coverage_threshold
        self.visit_threshold = visit_threshold
        
        # Pre-compute points for each bin (multiple points per bin for better coverage detection)
        self.points_per_bin = 10  # Check 10 points per bin instead of just center
        self.bin_points, self.point_to_bin = self._compute_arc_length_bins()
        
        # Track which bins have been visited
        self.bins_visited = np.zeros(num_bins, dtype=bool)
        
        # For display: bin center points
        self.bin_centers = self._compute_bin_centers()
    
    def _compute_arc_length_bins(self):
        """Compute multiple points per bin, evenly spaced by arc length"""
        total_points = self.num_bins * self.points_per_bin
        num_samples = 500
        t_values = np.linspace(0, 1, num_samples)
        
        # Get all sampled points
        sampled_points = np.array([self.segment.get_point_at_t(t) for t in t_values])
        
        # Compute cumulative arc length
        diffs = np.diff(sampled_points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.zeros(num_samples)
        cumulative_length[1:] = np.cumsum(segment_lengths)
        total_length = cumulative_length[-1]
        
        # Generate evenly spaced points along arc length
        bin_points = []
        point_to_bin = []  # Maps each point to its bin index
        
        for i in range(total_points):
            # Target arc length for this point
            target_length = (i + 0.5) / total_points * total_length
            
            # Find the index where cumulative length crosses target
            idx = np.searchsorted(cumulative_length, target_length)
            idx = min(idx, num_samples - 1)
            
            # Linear interpolation for accuracy
            if idx > 0 and idx < num_samples:
                len_before = cumulative_length[idx - 1]
                len_after = cumulative_length[idx]
                if len_after > len_before:
                    alpha = (target_length - len_before) / (len_after - len_before)
                    point = sampled_points[idx - 1] * (1 - alpha) + sampled_points[idx] * alpha
                else:
                    point = sampled_points[idx]
            else:
                point = sampled_points[idx]
            
            bin_points.append(point)
            point_to_bin.append(i // self.points_per_bin)  # Which bin this point belongs to
        
        return np.array(bin_points), np.array(point_to_bin)
    
    def _compute_bin_centers(self):
        """Compute center point of each bin for display purposes"""
        centers = []
        for bin_idx in range(self.num_bins):
            # Get all points in this bin
            mask = self.point_to_bin == bin_idx
            bin_pts = self.bin_points[mask]
            if len(bin_pts) > 0:
                centers.append(np.mean(bin_pts, axis=0))
            else:
                # Fallback: use t-parameter
                t = (bin_idx + 0.5) / self.num_bins
                centers.append(self.segment.get_point_at_t(t))
        return np.array(centers)
        
    def update(self, psm_pos):
        """Update coverage based on current PSM position.
        
        Checks distance to ALL points in each bin, not just center.
        If PSM is close to ANY point in a bin, that bin is marked visited.
        """
        psm_pos = np.array(psm_pos)
        
        # Calculate distance to all points
        distances = np.linalg.norm(self.bin_points - psm_pos, axis=1)
        
        # Find which points are within threshold
        close_points = distances < self.visit_threshold
        
        # Mark bins as visited if ANY of their points are close
        for point_idx in np.where(close_points)[0]:
            bin_idx = self.point_to_bin[point_idx]
            self.bins_visited[bin_idx] = True
        
    def get_coverage(self):
        """Return fraction of bins visited (0.0 to 1.0)"""
        return np.sum(self.bins_visited) / self.num_bins
    
    def get_visited_count(self):
        """Return number of bins visited"""
        return np.sum(self.bins_visited)
    
    def is_complete(self):
        """Check if coverage threshold met"""
        return self.get_coverage() >= self.coverage_threshold
    
    def get_unvisited_bin_centers(self):
        """Return center points of unvisited bins"""
        return self.bin_centers[~self.bins_visited]
    
    def get_visited_bin_centers(self):
        """Return center points of visited bins"""
        return self.bin_centers[self.bins_visited]
    
    def reset(self):
        """Reset for new iteration"""
        self.bins_visited = np.zeros(self.num_bins, dtype=bool)
