# Mars Rover SLAM Simulation

![Mars Rover SLAM Simulation](https://img.shields.io/badge/Mars%20Rover-SLAM%20Simulation-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Reinforcement Learning](https://img.shields.io/badge/ML-Reinforcement%20Learning-green)

A sophisticated 2D simulation environment for training and evaluating SLAM (Simultaneous Localization and Mapping) algorithms using Reinforcement Learning. This project provides a virtual Mars-like environment where a rover agent learns to navigate, map unknown terrain, and avoid obstacles efficiently.

## 🚀 Features

- **Interactive SLAM Simulation**: Real-time visualization of rover movement, sensor readings, and map construction
- **RL Agent Training**: Implements Q-learning with advanced optimizations for autonomous rover navigation
- **Dark Mode UI**: Modern dark-themed interface with intuitive controls and metrics
- **Comprehensive Evaluation Metrics**: Real-time performance tracking for mapping accuracy, exploration coverage, and path efficiency
- **Sensor Simulation**: Realistic sensor modeling with customizable noise parameters
- **Obstacle Generation**: Procedurally generated environments with random obstacles
- **Streamlit Dashboard**: Interactive UI for training, simulation control, and performance visualization

## 📋 Project Structure

```
SLAM_RL_ANAV/
├── app.py              # Main Streamlit application
├── environment.py      # SLAM environment simulation
├── agent.py            # Reinforcement learning agent implementation
└── requirements.txt    # Required Python packages
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Rushi-Sh/SLAM_RL_ANAV.git
cd SLAM_RL_ANAV
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 🎮 Usage Guide

### Training the Agent

1. Configure environment parameters in the sidebar (Map Size, Render Scale)
2. Adjust agent parameters (Learning Rate, Discount Factor)
3. Set training parameters (Number of Episodes, Batch Size)
4. Click "🧠 Train Agent" to begin training
5. Watch the training progress with real-time metrics

### Running the Simulation

1. After training is complete, click "▶️ Start Simulation" to see the agent navigate autonomously
2. Use the "🔄 Reset Simulation" button to start a new environment
3. Control simulation speed using the slider in the sidebar
4. Try manual control with the Forward, Left, and Right buttons for testing

### Analyzing Performance

1. Navigate to the "Evaluation Metrics" tab to see detailed performance analysis
2. Review mapping precision, exploration coverage, and path efficiency
3. Compare the actual environment with the SLAM-generated map
4. Track metrics over time with the performance charts
5. Download evaluation data for offline analysis

## 🧠 Technical Details

### Environment Implementation

The `environment.py` file implements a customized OpenAI Gym environment that simulates a Mars rover. Key components include:

- **Observation Space**: 8-directional sensor readings + rover position + orientation
- **Action Space**: Forward movement, left turn, right turn
- **Reward System**: Rewards for exploration, mapping accuracy, and efficient movement
- **Collision Detection**: Accurate collision model with safety margins
- **Sensor Model**: Realistic distance sensors with configurable noise

### Agent Implementation

The `agent.py` file implements an enhanced Q-learning agent with several optimizations:

- **Prioritized Experience Replay**: Focuses learning on important experiences
- **Double Q-learning**: Reduces overestimation bias
- **Adaptive Learning Rate**: Adjusts learning based on state visit frequency
- **Optimistic Initialization**: Encourages exploration of unvisited states
- **Epsilon Decay**: Gradually transitions from exploration to exploitation

### Evaluation Metrics

The simulation tracks several key performance indicators:

- **Mapping Precision**: Accuracy of obstacle detection in the environment
- **Exploration Coverage**: Percentage of the environment that has been explored
- **Path Efficiency**: Ratio of unique cells visited to total movement
- **Visual Comparison**: Side-by-side visualization of ground truth vs. SLAM map

## 📈 Performance Benchmarks

| Metric | Poor | Average | Good | Excellent |
|--------|------|---------|------|-----------|
| Mapping Precision | <40% | 40-60% | 60-80% | >80% |
| Exploration Coverage | <30% | 30-50% | 50-70% | >70% |
| Path Efficiency | <30% | 30-50% | 50-70% | >70% |

## 🔧 Configuration Options

The simulation offers various configuration options:

- **Map Size**: Controls the dimensions of the environment (10-50)
- **Render Scale**: Adjusts visualization size (10-30)
- **Learning Rate**: Controls how quickly the agent learns (0.0001-0.01)
- **Discount Factor**: Determines importance of future rewards (0.8-0.999)
- **Simulation Speed**: Adjusts visualization speed for better analysis

## 🔍 Future Enhancements

Potential areas for project expansion:

- Multi-agent SLAM simulation
- 3D environment representation
- Additional sensor types (camera, lidar)
- Implementation of other RL algorithms (DQN, PPO)
- Online training mode with human feedback
- More complex terrain types and obstacles

## 📚 References

- Reinforcement Learning: An Introduction by Sutton & Barto
- Simultaneous Localization and Mapping: A Survey of Current Trends (Cadena et al.)
- OpenAI Gym Documentation
- Streamlit Documentation

