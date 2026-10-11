
# Proactive Guardian

A context-aware safety system that predicts child hazards before they happen.


## What It Does

Instead of alerting after an accident, Proactive Guardian watches posture, movement, and context to predict danger — then sends graded alerts only when they matter.

## Hazards Detected

- Stove
- Stairs
- Pool

## How It Works

Camera → Pose → Features → LightGBM → Smoothing → Alert

## Results

| Metric | Value |
| :--- | :--- |
| Detection rate | 94% |
| False alarm rate | ~5% |
| Early warning | 2.6 s before danger |
| Trust score | 4.2 / 5.0 |
| Privacy score | 4.6 / 5.0 |

## Setup

    git clone https://github.com/hlatsi-2004/Proactive-Guardian.git
    pip install -r requirements.txt

## Author

Mamorobela Mahlatsi Suprise (202400251)
Sol Plaatje University
Supervisor: Dr. F. Matsebula

## License

MIT
