"""
Basic unit tests for Proactive Guardian.
Run with: python -m pytest tests/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.smoothing.temporal_filter import TemporalFilter
from src.alerts.proactivity_engine import ProactivityEngine


def test_temporal_filter_holds_level():
    f = TemporalFilter(window_size=5, hysteresis_seconds=0.1)
    f.update(3)
    f.update(3)
    f.update(3)
    f.update(3)
    f.update(3)
    assert f.current_level in (1, 3)


def test_proactivity_engine_no_alert_below_3():
    engine = ProactivityEngine()
    assert engine.should_alert(1) is False
    assert engine.should_alert(2) is False
    assert engine.should_alert(3) is True
    assert engine.should_alert(5) is True


def test_proactivity_engine_returns_correct_voice():
    engine = ProactivityEngine()
    assert engine.decide(1)['voice'] is None
    assert engine.decide(3)['voice'] == 'Liam, hot.'
    assert engine.decide(5)['voice'] == 'Liam, stop!'


if __name__ == "__main__":
    test_temporal_filter_holds_level()
    test_proactivity_engine_no_alert_below_3()
    test_proactivity_engine_returns_correct_voice()
    print("All tests passed.")
