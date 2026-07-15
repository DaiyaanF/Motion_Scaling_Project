#!/usr/bin/env python3

import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


class BezierSegment:
    def __init__(self, P0, P1, P2):
        self.P0 = np.array(P0)
        self.P1 = np.array(P1)
        self.P2 = np.array(P2)
        self.arc_length = None
        self.curvature_max = None
        self.curvature_avg = None
        self.euclidean_distance = None
        self.arc_ratio = None

    def compute_metrics(self, num_samples=1000):
        self.arc_length = self._calculate_arc_length(num_samples)
        self.curvature_max, self.curvature_avg = self._calculate_curvature(num_samples)
        self.euclidean_distance = np.linalg.norm(self.P2 - self.P0)
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
            derivatives[i] = 2 * (1 - t_val) * (self.P1 - self.P0) + 2 * t_val * (self.P2 - self.P1)
        return derivatives

    def _bezier_second_derivative(self, t):
        return 2 * (self.P2 - 2 * self.P1 + self.P0)

    def sample_points(self, num_points=100):
        t_values = np.linspace(0, 1, num_points)
        points = []
        for t in t_values:
            point = (1 - t)**2 * self.P0 + 2 * (1 - t) * t * self.P1 + t**2 * self.P2
            points.append(tuple(point))
        return points


class BezierTrajectoryGenerator:
    def __init__(self, boundary_size=0.1, max_attempts=100):
        self.boundary_size = boundary_size
        self.max_attempts = max_attempts

    def generate_segment(self, start_point, target_arc_length, curvature_range=None):
        start_point = np.array(start_point)

        for attempt in range(self.max_attempts):
            P1, P2 = self._generate_random_points(start_point)

            scale_factor = self._find_scale_for_arc_length(start_point, P1, P2, target_arc_length)

            if scale_factor is None:
                continue

            direction = P2 - start_point
            P2_scaled = start_point + direction * scale_factor
            direction_p1 = P1 - start_point
            P1_scaled = start_point + direction_p1 * scale_factor

            if not self._within_bounds(P1_scaled) or not self._within_bounds(P2_scaled):
                continue

            segment = BezierSegment(start_point, P1_scaled, P2_scaled)
            segment.compute_metrics()

            if abs(segment.arc_length - target_arc_length) < 0.001:
                if curvature_range is not None:
                    if curvature_range[0] <= segment.curvature_max <= curvature_range[1]:
                        return segment
                else:
                    return segment

        P1 = start_point + np.array([0.01, 0.01, 0])
        P2 = start_point + np.array([target_arc_length, 0, 0])
        segment = BezierSegment(start_point, P1, P2)
        segment.compute_metrics()
        return segment

    def _generate_random_points(self, start_point):
        half_bound = self.boundary_size / 2

        P1 = np.random.uniform(-half_bound, half_bound, 3)
        P2 = np.random.uniform(-half_bound, half_bound, 3)

        return P1, P2

    def _find_scale_for_arc_length(self, P0, P1, P2, target_length):
        def arc_length_diff(scale):
            P1_scaled = P0 + (P1 - P0) * scale
            P2_scaled = P0 + (P2 - P0) * scale
            temp_segment = BezierSegment(P0, P1_scaled, P2_scaled)
            current_length = temp_segment._calculate_arc_length(100)
            return abs(current_length - target_length)

        result = minimize_scalar(arc_length_diff, bounds=(0.1, 3.0), method='bounded')

        if result.fun < 0.001:
            return result.x
        return None

    def _within_bounds(self, point):
        half_bound = self.boundary_size / 2
        return np.all(np.abs(point) <= half_bound)

    def check_completion(self, psm_pos, segment_end, threshold):
        distance = np.linalg.norm(psm_pos - segment_end)
        return distance < threshold
