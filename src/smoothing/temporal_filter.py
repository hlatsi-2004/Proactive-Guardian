"""
Temporal smoothing for Proactive Guardian.
Applies 5-frame majority vote + 1.5s hysteresis to prevent flicker.
"""

from collections import deque
import time


class TemporalFilter:
    """Smooths risk level predictions over time."""

    def __init__(self, window_size=5, hysteresis_seconds=1.5):
        self.window_size = window_size
        self.hysteresis_seconds = hysteresis_seconds
        self.window = deque(maxlen=window_size)
        self.current_level = 1
        self.pending_level = None
        self.pending_since = None

    def update(self, predicted_level):
        self.window.append(predicted_level)
        majority = max(set(self.window), key=self.window.count)

        if majority == self.current_level:
            self.pending_level = None
            self.pending_since = None
            return self.current_level

        if majority != self.pending_level:
            self.pending_level = majority
            self.pending_since = time.time()
            return self.current_level

        elapsed = time.time() - self.pending_since
        if elapsed >= self.hysteresis_seconds:
            self.current_level = majority
            self.pending_level = None
            self.pending_since = None

        return self.current_level


if __name__ == "__main__":
    f = TemporalFilter()
    test_sequence = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1]
    for pred in test_sequence:
        smoothed = f.update(pred)
        print(f"Predicted: {pred} -> Smoothed: {smoothed}")
