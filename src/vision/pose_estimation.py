python
"""
Pose estimation module for Proactive Guardian.
Uses MediaPipe to extract 17 keypoints from a live camera feed.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """Extracts pose keypoints from video frames using MediaPipe."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_keypoints(self, frame):
        """
        Extract keypoints from a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            dict: keypoint name -> (x, y, z, visibility) or None if no person detected
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        keypoints = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(idx).name
            keypoints[name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        return keypoints

    def draw_skeleton(self, frame, keypoints):
        """Draw the detected skeleton on the frame for visual debugging."""
        if keypoints is None:
            return frame

        h, w, _ = frame.shape
        for name, kp in keypoints.items():
            if kp['visibility'] > 0.5:
                cx, cy = int(kp['x'] * w), int(kp['y'] * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        return frame

    def close(self):
        self.pose.close()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = estimator.extract_keypoints(frame)
        if keypoints:
            frame = estimator.draw_skeleton(frame, keypoints)
            cv2.putText(frame, "Person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Proactive Guardian - Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    estimator.close()
    cv2.destroyAllWindows()
