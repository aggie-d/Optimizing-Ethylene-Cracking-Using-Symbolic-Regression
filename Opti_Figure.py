import numpy as np
import matplotlib.pyplot as plt

# Create a figure with three subplots to tell the story of the methodology
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# --- Panel 1: Methodology Overview (Logic Flow) ---
axs[0].axis('off')
axs[0].set_title('1. Methodology Overview', fontsize=14, fontweight='bold', pad=20)

box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='navy', alpha=0.8)

# Flowchart elements: Data -> Optimization -> Equation -> Boundary
axs[0].text(0.5, 0.9, r"Reactor Parameters ($x$)", ha='center', va='center', bbox=box_props)
axs[0].annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85), arrowprops=dict(arrowstyle='->', lw=1.5))

axs[0].text(0.5, 0.65, r"PySR Optimization" + "\n" + r"$\min \sum \mathcal{L}(\sigma(F(x)), y)$", 
            ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e6f3ff', edgecolor='blue'))
axs[0].annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.6), arrowprops=dict(arrowstyle='->', lw=1.5))

axs[0].text(0.5, 0.4, r"Optimal Equation: $F(x)$", ha='center', va='center', bbox=box_props)
axs[0].annotate('', xy=(0.5, 0.25), xytext=(0.5, 0.35), arrowprops=dict(arrowstyle='->', lw=1.5))

axs[0].text(0.5, 0.15, r"Feasibility Boundary" + "\n" + r"$\sigma(F(x)) \geq 0.5$", 
            ha='center', va='center', bbox=box_props)

# --- Panel 2: Sigmoid Loss Function Utility ---
# This explains WHY the sigmoid is used: to map raw output to a 0-1 probability
x_sig = np.linspace(-6, 6, 100)
y_sig = 1 / (1 + np.exp(-x_sig))

axs[1].plot(x_sig, y_sig, 'b-', lw=3, label=r'$\sigma(F(x))$')
axs[1].axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Threshold (0.5)')
axs[1].axvline(0, color='black', lw=1)

# Highlight classification regions
axs[1].fill_between(x_sig, 0.5, 1, where=(x_sig >= 0), color='green', alpha=0.1)
axs[1].fill_between(x_sig, 0, 0.5, where=(x_sig <= 0), color='red', alpha=0.1)

axs[1].text(3, 0.75, "Feasible (1)", color='green', fontweight='bold', ha='center')
axs[1].text(-3, 0.25, "Infeasible (0)", color='red', fontweight='bold', ha='center')

axs[1].set_title('1. Mapping to Probability', fontsize=14, fontweight='bold', pad=15)
axs[1].set_xlabel(r'Model Output $F(x)$', fontsize=12)
axs[1].set_ylabel(r'Probability $P(y=1)$', fontsize=12)
axs[1].grid(True, alpha=0.3)
axs[1].legend()

# --- Panel 3: Boundary Mapping (Parameter Space) ---
# Illustrates the physical result of the classification boundary
np.random.seed(42)
temp = np.random.uniform(300, 800, 100)
pressure = np.random.uniform(1, 50, 100)

# Define a hypothetical non-linear boundary equation F(x) = 0
def f_boundary(t, p):
    return (t - 550)**2 / 20000 + (p - 25)**2 / 400 - 1

z = f_boundary(temp, pressure)
feasible = z < 0

axs[2].scatter(temp[feasible], pressure[feasible], c='green', marker='o', label='Feasible', alpha=0.6)
axs[2].scatter(temp[~feasible], pressure[~feasible], c='red', marker='x', label='Infeasible', alpha=0.6)

# Create a contour line for the decision boundary where F(x) = 0
t_grid, p_grid = np.meshgrid(np.linspace(300, 800, 100), np.linspace(1, 50, 100))
z_grid = f_boundary(t_grid, p_grid)
axs[2].set_xticklabels([])
axs[2].set_yticklabels([])
axs[2].contour(t_grid, p_grid, z_grid, levels=[0], colors='black', linestyles='-', linewidths=2)

axs[2].set_title('2. Final Boundary Mapping', fontsize=14, fontweight='bold', pad=15)
axs[2].set_xlabel('X', fontsize=12)
axs[2].set_ylabel('Y', fontsize=12)
axs[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig('methodology_optimization_viz.png', dpi=300)