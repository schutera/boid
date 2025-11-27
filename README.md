# Boid 3D Simulation with Deep Q-Learning

## Overview
This project simulates 3D boid flocks in Python, with reinforcement learning (DQN) agents controlling leader boids to keep their flock in designated quadrants. The simulation uses matplotlib for visualization and PyTorch for DQN.

## Research Idea
- Training by reducing the number of leaders (but on the way there training a single decentralized AI model) / taming the flock.
- Think about spiralling behavior to control a flock.

## Features
- Multiple flocks, each with a leader boid controlled by a DQN agent
- Flocks can be assigned to specific quadrants in the 3D space
- Reward shaping: leaders are rewarded for keeping their flock in the quadrant, moving toward the quadrant, and maintaining flock cohesion
- Toggleable predator and leader logic
- Visual quadrant indicators and leader auras
- Console logging of reward components, loss, and buffer size

## Training Process
1. **State Space:**
   - For each leader: up to 5 nearest neighbors (relative position and velocity)
   - Quadrant indicator (integer 1-8)
   - State vector length: 31
2. **Action Space:**
   - 7 discrete actions: move in ±x, ±y, ±z, or no move
3. **Reward Calculation:**
   - Fraction of flock in assigned quadrant
   - Additional reward for boids moving toward the quadrant
   - Cohesion reward for keeping flock close together
   - Total reward = quadrant_reward + 0.2 * move_toward_reward + 0.2 * cohesion
4. **DQN Training:**
   - Experience replay buffer (default size: 10,000)
   - Batch size: 64
   - Epsilon-greedy exploration
   - Loss: mean squared error between predicted and target Q-values
   - Training starts when buffer is full

## Parameters
- `width, height, depth`: Size of the 3D simulation space
- `num_boids`: Number of boids per flock
- `num_flocks`: Number of flocks
- `visual_range`: Range for boid interactions
- `speed_limit`: Maximum boid speed
- `margin`: Boundary repulsion margin
- `turn_factor`: Strength of boundary repulsion
- `centering_factor`: Cohesion strength
- `avoid_factor_intra`: Avoidance within flock
- `avoid_factor_inter`: Avoidance between flocks
- `matching_factor_intra`: Alignment within flock
- `matching_factor_inter`: Alignment between flocks
- `avoid_factor_predator`: Predator avoidance strength
- `enable_predators`: Toggle predator logic
- `enable_leader`: Toggle leader DQN logic
- `num_predators`: Number of predators
- `predator_speed`: Predator speed
- `flock_colors`: Colors for each flock
- `color_predator`: Predator color
- `min_distance`: Minimum distance for avoidance
- DQN agent parameters: learning rate, gamma, epsilon, buffer size, batch size

## Visualization
- Each flock is shown in a unique color
- Leader boids have a transparent aura
- Assigned quadrant is shown as a colored wireframe box
- Boid traces and predator visualization

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulation:
   ```bash
   python boid3d_simulation.py
   ```
3. Adjust parameters in `boid3d_simulation.py` as needed

## Files
- `boid3d_simulation.py`: Main simulation and training logic
- `dqn_agent.py`: DQN agent implementation
- `requirements.txt`: Python dependencies
- `.gitignore`: Standard ignores for Python projects

## Notes
- Training is long-term; reward shaping helps propagate learning
- You can tune reward weights, DQN parameters, and simulation settings for different behaviors
- Console output shows leader rewards, loss, and buffer size for monitoring




## Further Reading: Key Research / Papers on Training a Leader

Here are several representative works and themes combining reinforcement learning (RL) with leader–follower flocking or swarm control:

- **A Continuous Actor-Critic Reinforcement Learning Approach to Flocking with Fixed-Wing UAVs**  
    Chang Wang, Chao Yan, Xiaojia Xiang & Han Zhou (2019)  
    - Uses actor-critic RL for leader/follower flocking in continuous state/action spaces  
    - Introduces CACER algorithm with double prioritized experience replay  
    - Tested in simulation and hardware-in-the-loop  
    - Proceedings of Machine Learning Research

- **PPO-Exp: Keeping Fixed-Wing UAV Formation with Deep Reinforcement Learning**  
    - One RL-based intelligent leader, followers without intelligence chips  
    - Uses a variant of Proximal Policy Optimization (PPO) for exploration/estimation trade-off  
    - Designs low-communication protocol between leader and followers  
    - MDPI

- **Using Reinforcement Learning to Herd a Robotic Swarm to a Target Distribution**  
    Kakish, Elamvazhuthi & Berman (2020)  
    - Leader agent learns to herd swarm to desired spatial distribution via RL  
    - Uses mean-field models; leader policy depends on population distribution  
    - Validated in simulation and with physical robots  
    - arXiv

- **Hiding Leader's Identity in Leader-Follower Navigation through Multi-Agent RL**  
    Deka, Luo, Li, Lewis & Sycara (2021)  
    - Leader trained via multi-agent RL to camouflage among followers  
    - Combines Graph Neural Networks (GNNs) and adversarial training  
    - Relevant for stealth/security in leader-based swarms  
    - arXiv

- **A Hierarchical RL Framework for Multi-UAV Combat Using Leader–Follower Strategy**  
    Pang, He, Mohamed, Lin, Zhang & Hao (2025)  
    - Hierarchical multi-level RL for leader–follower UAVs in combat  
    - Leader-Follower Multi-Agent PPO (LFMAPPO) assigns explicit roles  
    - Shows improved cooperation and strategy in adversarial settings  
    - arXiv, Pure

- **Resilient Autonomous Control of Distributed Multi-agent Systems**  
    Moghadam & Modares (2017)  
    - RL-based control for leader–follower systems under attacks/uncertainties  
    - Leader is non-autonomous; uses off-policy RL and observer-based control  
    - Protocol includes trust/confidence values for resilience  
    - arXiv

- **Graph Neural Network based Deep RL for Multi-Agent Leader-Follower Flocking**  
    (2024)  
    - Combines GNNs and deep RL for leader–follower flocking  
    - GNNs model local interaction graphs; RL learns leader policy leveraging graph structure  
    - OUCI
