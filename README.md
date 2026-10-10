# Proactive Guardian

A context-aware, multi-hazard safety system for child injury prevention in the home.

## Overview

Proactive Guardian perceives danger the way a human would—interpreting posture, movement, and environmental context to predict hazards before they occur. Unlike reactive monitoring systems, it delivers graded alerts matched to actual risk, reducing alarm fatigue and building caregiver trust.

## Features

- **Pose estimation** using MediaPipe (17 keypoints)
- **8-feature vector** capturing distance, velocity, posture, and context
- **LightGBM classifier** predicting risk levels L1–L5
- **Temporal smoothing** (5-frame majority vote + 1.5s hysteresis)
- **Graded alerts**: Notify / Suggest / Intervene
- **Local-first processing** — no cloud, no video storage

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

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/proactive-guardian.git
cd proactive-guardian
pip install -r requirements.txt
