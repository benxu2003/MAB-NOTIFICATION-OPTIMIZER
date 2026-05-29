# Multi-Armed Bandit for Mobile Notification Optimization

This repository implements a comprehensive **Multi-Armed Bandit (MAB)** framework designed to dynamically optimize mobile push notification Click-Through Rates (CTR). The system leverages various MAB algorithms including Thompson Sampling, Epsilon-Greedy, and UCB1 to balance exploration and exploitation for optimal notification personalization.

## 🎯 Problem Statement

Mobile app notifications are crucial for user engagement, but determining the most effective notification content remains challenging. Traditional A/B testing approaches are static and may waste resources on suboptimal variants. This project implements dynamic optimization using Multi-Armed Bandits to continuously learn and adapt to user preferences in real-time.

## ✨ Features

* **Multiple Bandit Algorithms**: Implementation of Thompson Sampling, Epsilon-Greedy, and UCB1 algorithms for comparison and optimal selection.
* **Synthetic Environment**: Simulated notification environment with configurable true conversion rates for different notification variants.
* **Real Data Evaluation**: Support for off-policy evaluation using historical marketing data via Replay Evaluation.
* **Comprehensive Visualization**: Detailed plots showing convergence, CTR trends, traffic allocation, and algorithm comparison.
* **Modular Architecture**: Clean, decoupled code structure following industry best practices.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mab-notification-optimizer.git
   cd mab-notification-optimizer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

Execute the main script to run all experiments:

```bash
python main.py
```

The script will:
1. Run synthetic simulations with Thompson Sampling
2. Compare different bandit algorithms
3. Evaluate on real data if available (optional)

## 📊 Output

All visualizations and results are saved in the `results/` folder with timestamps:
- `synthetic_simulation_YYYYMMDD_HHMMSS.png`: Comprehensive analysis of Thompson Sampling performance
- `algorithm_comparison_YYYYMMDD_HHMMSS.png`: Side-by-side comparison of different MAB algorithms

## 🏗️ Architecture