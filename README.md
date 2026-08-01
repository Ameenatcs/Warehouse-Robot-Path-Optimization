# Warehouse Robot Path Optimization Using Q-Learning

A reinforcement-learning simulation that trains a warehouse robot to navigate from a start location to a pickup point and then to a drop-off point while avoiding shelves in a 20 x 20 grid.

This group academic project was completed as part of the M.Sc. Artificial Intelligence programme at Jonkoping University, Sweden.

## Problem

Warehouse robots must find efficient routes while avoiding obstacles and adapting to changes in the environment. This project models the warehouse as a grid and uses tabular Q-learning to learn navigation policies through trial and error.

The simulation contains:

- A configurable start position
- A fixed pickup point
- A configurable drop-off position
- Shelf cells that act as obstacles
- Four possible actions: up, right, down and left

## Q-Learning Approach

The agent updates its action values using the Bellman equation:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
```

The main hyperparameters are:

| Parameter | Value |
|---|---:|
| Learning rate (`alpha`) | 0.20 |
| Discount factor (`gamma`) | 0.95 |
| Initial exploration (`epsilon`) | 1.00 |
| Final exploration (`epsilon`) | 0.05 |
| Episodes per route phase | 1,500 |

Exploration decreases linearly during training using an epsilon-greedy policy.

## Reward Design

| Event | Reward |
|---|---:|
| Every movement step | -1 |
| Collision with a wall or shelf | -10 additional penalty |
| Reaching the goal | +50 |
| Moving closer to the goal | `0.2 x change in Manhattan distance` |

## Training Pipeline

Training is performed in two phases:

1. Start position to pickup point
2. Pickup point to drop-off position

The interface displays the learned routes after training. Users can also shuffle the shelf layout, which resets the Q-table and allows the agent to learn in a modified environment.

## Features

- Tabular Q-learning implementation written with NumPy
- Linear epsilon decay
- Reward shaping based on Manhattan distance
- Dynamically generated shelf layouts
- Collision handling for shelves and grid boundaries
- Tkinter graphical interface
- Learned-path overlays
- Q-value and visit-count heatmaps
- Episode-reward and Q-value convergence plots

## Verified Example Run

Using the default random seeds, layout and coordinates, a test run successfully reached both goals:

| Route | Learned path length |
|---|---:|
| Start `(1, 1)` to pickup `(0, 6)` | 7 grid positions |
| Pickup `(0, 6)` to drop-off `(18, 18)` | 31 grid positions |

The combined route required 36 movements. Results may vary if the layout, coordinates, random seed, hyperparameters or reward settings are changed.

## Repository Structure

```text
Warehouse-Robot-Path-Optimization/
|-- warehouse_q_learning.py
|-- background.jpg
|-- warehouse.png
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Requirements

- Python 3.9 or newer
- Tkinter, normally included with standard Python installations

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

On Linux, Tkinter may need to be installed separately using the operating system's package manager.

## How to Run

1. Download or clone this repository.
2. Open a terminal in the repository folder.
3. Install the dependencies.
4. Run:

```bash
python warehouse_q_learning.py
```

In the application:

1. Enter valid start and drop-off coordinates.
2. Select **Train Start to Pickup to Drop**.
3. Inspect the learned path, reward plot and convergence diagnostic.
4. Use **Show Q Heatmap** or **Show Visit Heat** to explore the learned values.
5. Select **Shuffle Layout** to generate a different obstacle arrangement and retrain.

## Technology Stack

- Python
- NumPy
- Tkinter
- Matplotlib
- Pillow
- Reinforcement learning
- Q-learning

## Current Limitations

- Tabular Q-learning does not scale efficiently to large or continuous state spaces.
- The environment assumes perfect position information and does not model sensor noise.
- The pickup location and reward parameters are configured in the source code.
- Training occurs in the interface thread, so the GUI may be temporarily unresponsive.
- The project models a simulation and is not connected to physical robot hardware.
- Repeated runs can produce different results when seeds or layouts are changed.

## Future Improvements

- Use separate Q-tables or goal-aware state representations for each route phase
- Add repeatable evaluation across multiple random seeds
- Compare Q-learning with A*, Dijkstra's algorithm and Deep Q-Networks
- Add cycle detection and explicit success metrics for generated paths
- Move training to a background thread to keep the interface responsive
- Add multi-robot coordination and sensor-noise simulation

## Authors

- Ameena Thanzoor
- Jobsy Johnson
- Manish Doddamane Nagaraju
- Noufa Haneefa

## References

1. Watkins, C. J. C. H., and Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279-292.
2. Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
