﻿# Solar Panel Angle Optimization with Q-Learning

This project uses Reinforcement Learning (Q-Learning) to optimize solar panel tilt angles for maximum energy output. By dynamically adjusting angles to track the sun, it enhances energy efficiency. The custom environment simulates real-world solar tracking, making it ideal for renewable energy applications.

The Q-Learning algorithm learns to optimize solar panel tilt angles (0°-90°, 1° steps) to maximize energy output by tracking the sun. Using an epsilon-greedy policy, it updates a Q-table based on rewards (cosine of angle difference). Assumptions: static sun angle (e.g., 45°), discrete actions (±3°,±5°), and no motor energy costs. The custom environment simulates a single panel in a noise-free setting, ideal for initial RL experiments.
