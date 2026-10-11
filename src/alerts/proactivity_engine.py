"""
Proactivity engine for Proactive Guardian.
Decides whether to Notify, Suggest, or Intervene based on risk level.
"""


class ProactivityEngine:
    """Maps risk levels to alert actions."""

    LEVELS = {
        1: {'name': 'Safe',      'action': 'None',      'led': 'off',   'voice': None},
        2: {'name': 'Watching',  'action': 'Log',       'led': 'off',   'voice': None},
        3: {'name': 'Caution',   'action': 'Suggest',   'led': 'amber', 'voice': 'Liam, hot.'},
        4: {'name': 'Warning',   'action': 'Intervene', 'led': 'amber', 'voice': 'Liam, move back.'},
        5: {'name': 'Critical',  'action': 'Intervene', 'led': 'red',   'voice': 'Liam, stop!'},
    }

    def decide(self, risk_level):
        return self.LEVELS.get(risk_level, self.LEVELS[1])

    def should_alert(self, risk_level):
        return risk_level >= 3


if __name__ == "__main__":
    engine = ProactivityEngine()
    for level in range(1, 6):
        decision = engine.decide(level)
        alert = "ALERT" if engine.should_alert(level) else "silent"
        print(f"L{level} -> {decision['name']:10s} | {alert:6s} | LED: {decision['led']:5s} | Voice: {decision['voice']}")
