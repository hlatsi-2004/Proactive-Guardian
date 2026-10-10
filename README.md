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
<img width="940" height="193" alt="image" src="https://github.com/user-attachments/assets/f704b277-7960-429b-b02d-1293fa5523ef" />
