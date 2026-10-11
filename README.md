# Proactive Guardian

A context-aware, multi-hazard safety system for child injury prevention in the home.

## Live Demo

Try the interactive simulation: **[Proactive Guardian Demo](https://hlatsi-2004.github.io/Proactive-Guardian/)**

The demo lets you:
- Switch between three hazard scenarios (stove, stairs, pool)
- Watch the child move toward danger
- See the risk level escalate from L1 (Safe) to L5 (Critical)
- Hear graded voice alerts (calm, firm, critical)
- View the event timeline and privacy controls

## Overview

Proactive Guardian perceives danger the way a human would—interpreting posture, movement, and environmental context to predict hazards before they occur. Unlike reactive monitoring systems, it delivers graded alerts matched to actual risk, reducing alarm fatigue and building caregiver trust.

## Features

- Pose estimation using MediaPipe (17 keypoints)
- 8-feature vector capturing distance, velocity, posture, and context
- LightGBM classifier predicting risk levels L1–L5
- Temporal smoothing (5-frame majority vote + 1.5s hysteresis)
- Graded alerts: Notify / Suggest / Intervene
- Local-first processing — no cloud, no video storage

## Pipeline

Camera → Pose Estimation → Feature Extraction → LightGBM → Temporal Smoothing → Proactivity Engine → Alert

## Hazard Scenarios

- Stove proximity
- Stair access
- Pool approach

## Results

| Metric | Value |
| :--- | :--- |
| Detection rate | 94% |
| False alarm rate | ~5% |
| Early-warning lead time | 2.6 s |
| Inference latency | <8 ms/frame (Raspberry Pi 4) |
| Trust score | 4.2 / 5.0 |
| Privacy confidence | 4.6 / 5.0 |

## Installation

    git clone https://github.com/hlatsi-2004/Proactive-Guardian.git
    cd Proactive-Guardian
    pip install -r requirements.txt

## Usage

    python src/model/predict.py --camera 0

## Project Structure

    Proactive-Guardian/
    ├── demo/                  ← Interactive prototype
    ├── docs/                  ← Report and figures
    ├── src/
    │   ├── vision/            ← Pose estimation
    │   ├── features/          ← Feature extraction
    │   ├── model/             ← Training and inference
    │   ├── smoothing/         ← Temporal filter
    │   └── alerts/            ← Proactivity engine
    ├── notebooks/             ← Analysis notebook
    └── tests/                 ← Unit tests

## Author

**Mamorobela Mahlatsi Suprise**
Student ID: 202400251
Sol Plaatje University — Department of Computer Science & Information Technology

## Supervisor

Dr. F. Matsebula

## License

MIT License
