# Warehouse Robot Path Optimization using Q-Learning  

###  Project Overview  
This project implements a **Reinforcement Learning (RL)** approach for optimizing warehouse robot navigation using the **Q-learning algorithm**.  
The system trains an autonomous robot to efficiently navigate a **20×20 warehouse grid**, picking up and dropping off items while **avoiding obstacles (shelves)**.  

A **Tkinter GUI** visualizes the warehouse layout, robot movements, and learning progress in real-time.  
The program supports dynamic layouts, Q-value heatmaps, and visualizations of learned paths.

---

##  Objectives  
- Enable a robot to learn **optimal paths** in a grid-based warehouse.  
- Avoid obstacles while minimizing travel distance and time.  
- Demonstrate **adaptive learning** under changing warehouse layouts.  
- Visualize training, reward convergence, and Q-value evolution.  

---

##  Core Concept — Q-Learning  

The robot (agent) interacts with its environment (warehouse grid), learns through **trial and error**, and updates its **Q-table** based on the **Bellman equation**:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]
\]

Where:  
- **α (alpha)** – Learning rate (0.2)  
- **γ (gamma)** – Discount factor (0.95)  
- **ε (epsilon)** – Exploration probability (decays from 1.0 → 0.05)  

Rewards are defined as:  
| Action | Reward |
|:--|:--|
| Step | -1 |
| Collision with obstacle/wall | -10 |
| Goal reached | +50 |
| Distance improvement | +0.2 × (Δ Manhattan distance) |

---

##  Features  
-  Q-learning with linear epsilon decay  
-  20×20 customizable grid environment  
-  Dynamic obstacle generation and shuffling  
-  Two-phase training: **Start→Pickup** and **Pickup→Drop**  
-  Tkinter GUI with:  
  - Obstacle visualization (shelves)  
  - Q-value and visit-count heatmaps  
  - Path overlays for trained routes  
- Matplotlib plots for:  
  - Episode reward progression  
  - Q-value convergence diagnostics  

---

## Project Structure  
```
warehouse_r1_images.py    # Main code file
background.jpg            # Optional background image (in same folder)
warehouse.png             # Shelf/obstacle icon (in same folder)
```

---

##  Dependencies  
Make sure the following Python packages are installed:

```bash
pip install numpy matplotlib pillow
```

(`tkinter` comes pre-installed with Python on most systems.)

---

##  How to Run  
1. **Ensure all required files** (`warehouse_r1_images.py`, images) are in the same folder.  
2. Open a terminal in that directory.  
3. Run the program:  
   ```bash
   python warehouse_r1_images.py
   ```
4. Use the GUI to:
   - Set **Start** and **Drop** coordinates.  
   - Click **Train Start→Pickup→Drop** to begin learning.  
   - Toggle **Show Q Heatmap** or **Show Visit Heat** for visualization.  
   - Use **Shuffle Layout** to randomize shelves and retrain.  
   - Use **Clear Overlays** to reset paths.

---

##  Output Visualization  
- **Per-episode reward curve:** Shows improvement in robot performance.  
- **Q-value convergence plot:** Displays stabilization of learning.  
- **Grid GUI:** Highlights shelves, start (S), pickup (P), and drop (D) points with color-coded paths.  
- **Heatmaps:**  
  - **Q Heatmap:** Indicates high-value routes.  
  - **Visit Heat:** Shows exploration density during training.  

---

##  Experiment Setup  
- Grid size: **20×20**  
- Start: `(1, 1)`  
- Pickup: `(0, 6)`  
- Drop: `(18, 18)`  
- Episodes per phase: **1500**  
- Exploration: ε decays from **1.0 → 0.05**  
- Step limit per episode: `6 * GRID_W * GRID_H // 5`  

---

##  Results Summary  
| Metric | Observation |
|:--|:--|
| Average Reward (final 100 episodes) | ~+45 to +50 |
| Path length after convergence | ~65 steps (down from ~250) |
| Collisions | Nearly 0 after training |
| Convergence point | Around episode 1200 |

Robot successfully learns **efficient, collision-free navigation**, and adapts well when layouts change.

---

##  Limitations  
- Tabular Q-learning scales poorly for large or continuous environments.  
- Training can be time-consuming (thousands of episodes).  
- Reward parameters are fixed and may need tuning.  
- Simulation assumes perfect location sensing (no sensor noise).  

---

##  Future Enhancements  
- Implement **Deep Q-Networks (DQN)** for scalability.  
- Add **multi-agent coordination** for multiple robots.  
- Include **sensor noise modeling** and partial observability.  
- Integrate with **real-world robot control systems**.  

---

##  Authors  
- **Ameena Thanzoor**  
- **Jobsy Johnson**  
- **Manish Doddamane Nagaraju**  
- **Noufa Haneefa**  

---

##  References  
1. Watkins, C.J.C.H., & Dayan, P. (1992). *Technical Note: Q-Learning.* *Machine Learning*, 8(3–4), 279–292.  
2. Sutton, R.S., & Barto, A.G. (2020). *Reinforcement Learning: An Introduction (2nd ed.).* MIT Press.  
3. Peyas et al. (2021). *Autonomous Warehouse Robot using Deep Q-Learning.* IEEE TENCON.  
4. Li et al. (2024). *Deep RL-based Obstacle Avoidance for Robot Movement in Warehouse Environments.* ICCASIT.
