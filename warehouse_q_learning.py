
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageEnhance

# Configuration
GRID_W, GRID_H = 20, 20
CELL_SIZE = 28  
PICK_UP_POINT = (0, 6)  

# RL rewards
STEP_PENALTY = -1.0
OBSTACLE_PENALTY = -10.0
GOAL_REWARD = +50.0

# Learning hyperparameters
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] 
N_ACTIONS = len(ACTIONS)
ALPHA = 0.2
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPISODES_1 = 1500  
EPISODES_2 = 1500  
MAX_STEPS_PER_EPISODE = 6 * GRID_W * GRID_H // 5 

# Layout knobs
LAYOUT_SEED = 42           
EXTRA_BLOCKS_PC = 0.12     

# Images (must be in the same folder as this script)
BACKGROUND_FILE = "background.jpg"
SHELF_FILE = "warehouse.png"
# Visual tweaks
GRID_LINE = "#2A3C55"      
BG_DARKEN = 0.72           
SHELF_MARGIN = 0.25            

RNG = np.random.default_rng(0)

#  Layout generation
def in_bounds(x, y): return 0 <= x < GRID_W and 0 <= y < GRID_H

def generate_obstacles(seed=LAYOUT_SEED, extra_pc=EXTRA_BLOCKS_PC, keep=frozenset()):
    """
    Build shelves as:
      - Base aisles (columns) with a tiny jitter to avoid uniformity
      - Extra random blocks sprinkled between lanes
    We preserve "service lanes" (every 4th row and every 5th column) and any cells in `keep`.
    """
    rng = np.random.default_rng(seed)
    obstacles = set()

    # Base aisles
    for col_i, x in enumerate(range(3, GRID_W - 3, 5)):
        jitter = int(rng.integers(-1, 2))  # -1, 0, +1
        xx = int(np.clip(x + jitter, 1, GRID_W - 2))
        for y in range(2, GRID_H - 2):
            if y % 4 != 0:  # leave cross passages
                if (xx, y) not in keep:
                    obstacles.add((xx, y))

    # Service lanes kept free 
    lane_rows = {r for r in range(0, GRID_H, 4)}
    lane_cols = {c for c in range(0, GRID_W, 5)}

    # Candidates for extra shelves
    candidates = []
    for y in range(GRID_H):
        for x in range(GRID_W):
            if (x, y) in obstacles:       
                continue
            if x in lane_cols or y in lane_rows:
                continue                   
            if (x, y) in keep:
                continue                   
            candidates.append((x, y))

    extra_n = int(extra_pc * GRID_W * GRID_H)
    extra_n = min(extra_n, len(candidates))
    rng.shuffle(candidates)
    for (x, y) in candidates[:extra_n]:
        obstacles.add((x, y))

    # never block reserved cells
    for k in keep:
        if k in obstacles:
            obstacles.remove(k)
    return obstacles

# Global obstacle map
OBSTACLES = generate_obstacles(seed=LAYOUT_SEED, keep={PICK_UP_POINT})

def is_free(x, y): return in_bounds(x, y) and (x, y) not in OBSTACLES

def manhattan(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def linear_decay(e0, e1, step, total):
    step = np.clip(step, 0, total)
    return float(e0 + (e1 - e0) * (step / total))

#  Q-Learning core 
def state_index(x, y): return y * GRID_W + x

def choose_action(q, state, epsilon):
    if RNG.random() < epsilon:
        return int(RNG.integers(N_ACTIONS))
    return int(np.argmax(q[state]))

def get_next_state(x, y, a):
    dx, dy = ACTIONS[a]
    nx, ny = x + dx, y + dy
    if not in_bounds(nx, ny) or (nx, ny) in OBSTACLES:
        # collision → stay and penalize
        return x, y, True
    return nx, ny, False

def train_q_learning(q, start, goal, episodes, visit_counts=None, dense_shaping=True):
    """
    Train IN-PLACE on q-table.
    Returns: rewards_hist, deltas_hist (max |ΔQ| per episode)
    """
    rewards_hist, deltas_hist = [], []

    for ep in range(episodes):
        epsilon = linear_decay(EPS_START, EPS_END, ep, episodes - 1)
        x, y = start
        steps = 0
        total_reward = 0.0
        prev_q = q.copy()

        if visit_counts is not None:
            visit_counts[:, :] = 0

        while steps < MAX_STEPS_PER_EPISODE:
            s = state_index(x, y)
            a = choose_action(q, s, epsilon)
            nx, ny, collided = get_next_state(x, y, a)

            reward = STEP_PENALTY
            if collided and (nx == x and ny == y):
                reward += OBSTACLE_PENALTY

            if dense_shaping:
                reward += 0.2 * (manhattan((x, y), goal) - manhattan((nx, ny), goal))

            done = False
            if (nx, ny) == goal:
                reward += GOAL_REWARD
                done = True

            s2 = state_index(nx, ny)
            target = reward if done else reward + GAMMA * np.max(q[s2])
            q[s, a] = (1 - ALPHA) * q[s, a] + ALPHA * target

            total_reward += reward
            x, y = nx, ny
            steps += 1

            if visit_counts is not None:
                visit_counts[y, x] += 1

            if done:
                break

        rewards_hist.append(total_reward)
        deltas_hist.append(float(np.max(np.abs(q - prev_q))))

    return rewards_hist, deltas_hist

def greedy_path(q, start, goal, max_steps=2000):
    path = [start]
    x, y = start
    for _ in range(max_steps):
        if (x, y) == goal:
            break
        s = state_index(x, y)
        a = int(np.argmax(q[s]))
        nx, ny, _ = get_next_state(x, y, a)
        if (nx, ny) == (x, y):  # stuck
            break
        path.append((nx, ny))
        x, y = nx, ny
    return path

#  Tkinter UI 
class App:
    def __init__(self, root):
        self.root = root
        root.title("Warehouse Q-Learning — 20x20")

        self.canvas_w = GRID_W * CELL_SIZE
        self.canvas_h = GRID_H * CELL_SIZE
        self.canvas = tk.Canvas(root, width=self.canvas_w, height=self.canvas_h, bg="#0E2540", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=40, padx=10, pady=10)

        frm = tk.Frame(root)
        frm.grid(row=0, column=1, sticky="nw", padx=10)

        tk.Label(frm, text="Start (x,y)").grid(row=0, column=0, sticky="w")
        self.start_x = tk.Entry(frm, width=4); self.start_y = tk.Entry(frm, width=4)
        self.start_x.insert(0, "1"); self.start_y.insert(0, "1")
        self.start_x.grid(row=0, column=1); self.start_y.grid(row=0, column=2)

        tk.Label(frm, text="Drop (x,y)").grid(row=1, column=0, sticky="w")
        self.drop_x = tk.Entry(frm, width=4); self.drop_y = tk.Entry(frm, width=4)
        self.drop_x.insert(0, "18"); self.drop_y.insert(0, "18")
        self.drop_x.grid(row=1, column=1); self.drop_y.grid(row=1, column=2)

        self.btn_train = tk.Button(frm, text="Train Start→Pickup→Drop", command=self.train_pipeline)
        self.btn_train.grid(row=2, column=0, columnspan=3, pady=8)

        self.btn_clear = tk.Button(frm, text="Clear Overlays", command=self.clear_overlays)
        self.btn_clear.grid(row=3, column=0, columnspan=3)

        self.btn_shuffle = tk.Button(frm, text="Shuffle Layout", command=self.shuffle_layout)
        self.btn_shuffle.grid(row=4, column=0, columnspan=3, pady=4)

        self.var_qheat = tk.IntVar(value=1)
        self.var_visit = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="Show Q Heatmap", variable=self.var_qheat, command=self.redraw)\
            .grid(row=5, column=0, columnspan=3, sticky="w")
        tk.Checkbutton(frm, text="Show Visit Heat", variable=self.var_visit, command=self.redraw)\
            .grid(row=6, column=0, columnspan=3, sticky="w")

        self.status = tk.Label(frm, text="", fg="gray")
        self.status.grid(row=7, column=0, columnspan=3, sticky="w")

        # Q-learning state
        self.q_table = np.zeros((GRID_W * GRID_H, N_ACTIONS), dtype=np.float32)
        self.visit_counts = np.zeros((GRID_H, GRID_W), dtype=np.int32)
        self.paths = []

        # Load images (with graceful fallback)
        self._load_assets()

        # Initial draw
        self.draw_grid()

    # Assets
    def _load_assets(self):
        # Background
        self.bg_img_tk = None
        if os.path.exists(BACKGROUND_FILE):
            bg = Image.open(BACKGROUND_FILE).convert("RGB").resize((self.canvas_w, self.canvas_h), Image.LANCZOS)
            if BG_DARKEN != 1.0:
                enh = ImageEnhance.Brightness(bg)
                bg = enh.enhance(BG_DARKEN)
            self.bg_img_tk = ImageTk.PhotoImage(bg)

        # Shelf icon
        self.shelf_img_tk = None
        if os.path.exists(SHELF_FILE):
            shelf_raw = Image.open(SHELF_FILE).convert("RGBA")
            # Fit inside the cell while keeping aspect ratio
            max_w = CELL_SIZE - 2 * SHELF_MARGIN
            max_h = CELL_SIZE - 2 * SHELF_MARGIN
            w, h = shelf_raw.size
            scale = min(max_w / w, max_h / h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            shelf = shelf_raw.resize(new_size, Image.LANCZOS)
            self.shelf_img_tk = ImageTk.PhotoImage(shelf)

    #UI helpers
    def clear_overlays(self):
        self.paths.clear()
        self.visit_counts[:, :] = 0
        self.redraw()

    def shuffle_layout(self):
        """Randomize shelves, reset Q-table and overlays, keep pickup free."""
        global OBSTACLES, LAYOUT_SEED
        LAYOUT_SEED += 1
        OBSTACLES = generate_obstacles(seed=LAYOUT_SEED, keep={PICK_UP_POINT})
        # reset agent knowledge since the environment changed
        self.q_table[:, :] = 0.0
        self.paths.clear()
        self.visit_counts[:, :] = 0
        self.status.config(text=f"Layout shuffled (seed {LAYOUT_SEED}).")
        self.redraw()

    def parse_xy(self, x_str, y_str, name):
        try:
            x, y = int(x_str), int(y_str)
        except ValueError:
            raise ValueError(f"{name} must be integers")
        if not is_free(x, y):
            raise ValueError(f"{name} ({x},{y}) is invalid or blocked")
        return (x, y)

    # Training 
    def train_pipeline(self):
        try:
            s = self.parse_xy(self.start_x.get(), self.start_y.get(), "Start")
            d = self.parse_xy(self.drop_x.get(), self.drop_y.get(), "Drop")
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        if not is_free(*PICK_UP_POINT):
            messagebox.showerror("Config error", f"Pickup point {PICK_UP_POINT} is blocked. Click Shuffle again or edit config.")
            return

        # Leg 1: Start -> Pickup
        r1, d1 = train_q_learning(self.q_table, s, PICK_UP_POINT, EPISODES_1, visit_counts=self.visit_counts)
        path1 = greedy_path(self.q_table, s, PICK_UP_POINT)

        # Leg 2: Pickup -> Drop (continue same Q)
        r2, d2 = train_q_learning(self.q_table, PICK_UP_POINT, d, EPISODES_2, visit_counts=self.visit_counts)
        path2 = greedy_path(self.q_table, PICK_UP_POINT, d)

        self.paths = [path1, path2]

        # Plots
        plt.figure()
        plt.plot(r1, label="Start→Pickup")
        plt.plot(range(len(r1), len(r1) + len(r2)), r2, label="Pickup→Drop")
        plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.title("Per-episode Reward")
        plt.legend(); plt.tight_layout()

        plt.figure()
        plt.plot(d1, label="Start→Pickup")
        plt.plot(range(len(d1), len(d1) + len(d2)), d2, label="Pickup→Drop")
        plt.xlabel("Episode"); plt.ylabel("Max |ΔQ|"); plt.title("Convergence Diagnostic")
        plt.legend(); plt.tight_layout()

        self.status.config(text=f"Trained {EPISODES_1 + EPISODES_2} eps. Path lengths: {len(path1)} + {len(path2)}")
        self.redraw()
        plt.show()

    # Drawing 
    def redraw(self):
        self.draw_grid()
        if self.var_qheat.get():
            self.draw_q_heatmap()
        if self.var_visit.get():
            self.draw_visit_heat()
        self.draw_paths()

    def draw_grid(self):
        self.canvas.delete("all")

        # Background image (or solid fill)
        if self.bg_img_tk:
            self.canvas.create_image(0, 0, image=self.bg_img_tk, anchor="nw")
        else:
            self.canvas.create_rectangle(0, 0, self.canvas_w, self.canvas_h, fill="#0F3057", outline="")

        # Cells: shelves as images, others transparent (background shows through)
        for y in range(GRID_H):
            for x in range(GRID_W):
                x0, y0 = x * CELL_SIZE, y * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE

                if (x, y) in OBSTACLES and self.shelf_img_tk:
                    # center the shelf icon within the cell
                    iw = self.shelf_img_tk.width()
                    ih = self.shelf_img_tk.height()
                    cx, cy = x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2
                    self.canvas.create_image(cx, cy, image=self.shelf_img_tk)
                elif (x, y) in OBSTACLES:
                    # fallback rectangle if image missing
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="#424A57", outline="#2A3039")

                # thin grid overlay for readability
                self.canvas.create_rectangle(x0, y0, x1, y1, outline=GRID_LINE)

        # pickup marker
        px, py = PICK_UP_POINT
        self._draw_cell_marker(px, py, color="#2A9D8F", text="P")

    def _draw_cell_marker(self, x, y, color="#F4A261", text=None):
        x0, y0 = x * CELL_SIZE, y * CELL_SIZE
        x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
        pad = 6
        self.canvas.create_oval(x0 + pad, y0 + pad, x1 - pad, y1 - pad, fill=color, outline="white", width=2)
        if text:
            self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=text, fill="white",
                                    font=("TkDefaultFont", 9, "bold"))

    def draw_paths(self):
        colors = ["#E76F51", "#00B2FF"]
        labels = [("S", "P"), ("P", "D")]
        for idx, path in enumerate(self.paths):
            if not path:
                continue
            color = colors[idx % len(colors)]
            self._draw_cell_marker(*path[0], color=color, text=labels[idx][0])
            self._draw_cell_marker(*path[-1], color=color, text=labels[idx][1])
            for i in range(1, len(path)):
                x0, y0 = path[i - 1]
                x1, y1 = path[i]
                cx0, cy0 = x0 * CELL_SIZE + CELL_SIZE // 2, y0 * CELL_SIZE + CELL_SIZE // 2
                cx1, cy1 = x1 * CELL_SIZE + CELL_SIZE // 2, y1 * CELL_SIZE + CELL_SIZE // 2
                self.canvas.create_line(cx0, cy0, cx1, cy1, fill=color, width=3)
                # step markers (sparse)
                if i % max(1, (len(path) // 20)) == 0 or i == len(path) - 1:
                    self.canvas.create_text(cx1, cy1, text=str(i), fill="white",
                                            font=("TkDefaultFont", 8, "bold"))

    def draw_q_heatmap(self):
        # V(s) = max_a Q(s,a)
        vs = np.max(self.q_table, axis=1).reshape((GRID_H, GRID_W))
        vmin, vmax = vs.min(), vs.max()
        denom = (vmax - vmin) if vmax > vmin else 1.0
        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x, y) in OBSTACLES:
                    continue
                val = (vs[y, x] - vmin) / denom
                x0, y0 = x * CELL_SIZE, y * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                shade = "#3CB371" if val > 0 else "#B22222"
                stipples = ["gray75", "gray50", "gray25", "gray12"]
                idx = int(val * (len(stipples) - 1) + 0.0001)
                idx = max(0, min(idx, len(stipples) - 1))
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=shade, stipple=stipples[idx], outline="")

    def draw_visit_heat(self):
        vc = self.visit_counts
        if vc.max() == 0:
            return
        vmax = vc.max()
        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x, y) in OBSTACLES:
                    continue
                val = vc[y, x] / vmax
                x0, y0 = x * CELL_SIZE, y * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
                stipples = ["gray75", "gray50", "gray25", "gray12"]
                idx = int(val * (len(stipples) - 1) + 0.0001)
                idx = max(0, min(idx, len(stipples) - 1))
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="", fill="#1D3557", stipple=stipples[idx])

# Entrypoint 
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
