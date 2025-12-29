# RL Learning Tool

A web-based tool for exploring reinforcement learning algorithms and environments.

## What's Inside

**5 Environments:**
- GridWorld - navigate to goal avoiding obstacles
- FrozenLake - slippery ice with holes
- CliffWalking - walk along cliff edge
- MountainCar - build momentum to reach flag
- CartPole - balance pole on cart

**7 Algorithms:**
- Q-Learning (off-policy)
- SARSA (on-policy)
- Monte Carlo (first-visit)
- Value Iteration
- Policy Iteration
- TD(0)
- N-step TD

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
cd rl_tool
python app.py
```

Open browser to: `http://localhost:5000`

## How to Use

1. Pick an environment from the dropdown
2. Click "Initialize" to set it up
3. Choose an algorithm
4. Adjust parameters if needed (gamma, alpha, epsilon, etc.)
5. Click "Train Agent" and wait for it to learn
6. Click "Run Episode" to see the trained agent in action

You can also use manual control with arrow keys to explore the environment yourself.

## Project Structure

```
rl_tool/
├── app.py              - flask backend
├── algorithms.py       - rl algorithm implementations
├── environments.py     - environment definitions
├── templates/
│   └── index.html      - main page
└── static/
    ├── style.css       - styling
    └── script.js       - frontend logic
```

## Notes

- Training can take a while depending on the algorithm and environment
- MountainCar and CartPole need more episodes to learn properly
- Use higher epsilon for better exploration in sparse reward environments
- Model-based algorithms (PI/VI) work better on smaller discrete environments
- Model-free algorithms (Q-Learning, SARSA, MC) work on all environments

## Requirements

- Python 3.8+
- Flask
- NumPy
- Gymnasium

See `requirements.txt` for specific versions.
