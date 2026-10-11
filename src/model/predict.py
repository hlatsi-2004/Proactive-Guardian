"""
Inference script for Proactive Guardian.
Runs the full pipeline: camera -> pose -> features -> model -> smoothing -> alert.
"""

import cv2
import numpy as np
import lightgbm as lgb
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vision.pose_estimation import PoseEstimator
from features.extract_features import FeatureExtractor
from smoothing.temporal_filter import TemporalFilter
from alerts.proactivity_engine import ProactivityEngine


HAZARD_ZONES = {
    'stove':  (400, 200, 600, 400),
    'stairs': (100, 500, 300, 700),
    'pool':   (700, 600, 900, 800),
}


def run_pipeline(camera_index=0, model_path="src/model/lightgbm_model.txt"):
    estimator = PoseEstimator()
    extractor = FeatureExtractor(HAZARD_ZONES)
    smoother = TemporalFilter()
    engine = ProactivityEngine()

    model = None
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        print("Model loaded.")
    else:
        print("Warning: model not found. Running in simulation mode.")

    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = estimator.extract_keypoints(frame)

        if keypoints:
            features = extractor.compute_features(keypoints, frame.shape)

            if features is not None and model is not None:
                raw_pred = int(np.argmax(model.predict([features])) + 1)
            else:
                raw_pred = 1

            smoothed = smoother.update(raw_pred)
            decision = engine.decide(smoothed)

            cv2.putText(frame, f"Risk: L{smoothed} ({decision['name']})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if engine.should_alert(smoothed):
                cv2.putText(frame, f"ALERT: {decision['voice']}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Proactive Guardian", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    estimator.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()
