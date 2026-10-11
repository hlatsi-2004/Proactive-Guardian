"""
Feature extraction module for Proactive Guardian.
Computes the 8-feature vector from pose keypoints and hazard zones.
"""

import numpy as np


class FeatureExtractor:
    """Computes the 8-feature vector from pose keypoints and hazard zones."""

    def __init__(self, hazard_zones):
        """
        Args:
            hazard_zones: dict of zone_name -> (x_min, y_min, x_max, y_max) in pixel coords
        """
        self.hazard_zones = hazard_zones
        self.prev_hip = None

    def compute_features(self, keypoints, frame_shape):
        """
        Compute the 8-feature vector.

        Features:
            1. distance_to_hazard
            2. velocity_toward_zone
            3. posture_class
            4. time_to_contact
            5. hip_height_trend
            6. head_zone_alignment
            7. object_proximity
            8. raw_joint_angles
        """
        if keypoints is None:
            return None

        h, w = frame_shape[:2]

        hip = keypoints.get('LEFT_HIP') or keypoints.get('RIGHT_HIP')
        head = keypoints.get('NOSE')
        if hip is None or head is None:
            return None

        hip_px = np.array([hip['x'] * w, hip['y'] * h])
        head_px = np.array([head['x'] * w, head['y'] * h])

        # 1. Distance to nearest hazard
        min_dist = float('inf')
        nearest_zone = None
        for zone_name, (x_min, y_min, x_max, y_max) in self.hazard_zones.items():
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            d = np.linalg.norm(hip_px - np.array([cx, cy]))
            if d < min_dist:
                min_dist = d
                nearest_zone = zone_name

        # 2. Velocity toward zone
        velocity = 0.0
        if self.prev_hip is not None:
            velocity = np.linalg.norm(hip_px - self.prev_hip)
        self.prev_hip = hip_px

        # 3. Posture class
        posture = self._classify_posture(keypoints)

        # 4. Time to contact
        time_to_contact = min_dist / velocity if velocity > 1e-3 else float('inf')

        # 5. Hip height trend
        hip_height_trend = hip_px[1] / h

        # 6. Head-zone alignment
        head_zone_alignment = abs(head_px[1] - hip_px[1]) / h

        # 7. Object proximity (placeholder — thermal sensor input)
        object_proximity = 0.0

        # 8. Raw joint angles
        joint_angles = self._compute_joint_angles(keypoints)

        return np.array([
            min_dist,
            velocity,
            posture,
            time_to_contact if time_to_contact != float('inf') else 9999,
            hip_height_trend,
            head_zone_alignment,
            object_proximity,
            joint_angles
        ])

    def _classify_posture(self, keypoints):
        """Returns: 0=standing, 1=walking, 2=crouching, 3=climbing."""
        hip = keypoints.get('LEFT_HIP')
        knee = keypoints.get('LEFT_KNEE')
        if hip is None or knee is None:
            return 0
        delta_y = abs(hip['y'] - knee['y'])
        if delta_y < 0.05:
            return 2  # crouching
        elif delta_y > 0.15:
            return 3  # climbing
        return 0

    def _compute_joint_angles(self, keypoints):
        """Simple proxy for joint angle magnitude."""
        return 0.0


if __name__ == "__main__":
    zones = {
        'stove': (400, 200, 600, 400),
        'stairs': (100, 500, 300, 700),
        'pool': (700, 600, 900, 800)
    }
    extractor = FeatureExtractor(zones)
    print("Feature extractor initialised with zones:", list(zones.keys()))
