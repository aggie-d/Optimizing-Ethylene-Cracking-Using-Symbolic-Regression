import numpy as np
import matplotlib.pyplot as plt

def plot_parabolic_feasibility_with_points():
    # Define range for the boundary curve
    x_curve = np.linspace(0, 4, 400)
    y_curve = 16 - x_curve**2
    
    # Generate 100 random points in the viewport (x: 0-5, y: 0-20)
    np.random.seed(42)  # Fixed seed for consistent results
    num_points = 100
    x_random = np.random.uniform(0, 5, num_points)
    y_random = np.random.uniform(0, 20, num_points)
    
    # Logic for feasibility: 
    # 1. x must be within the boundary range [0, 4]
    # 2. y must be below or on the curve y = 16 - x^2
    feasible_mask = (x_random <= 4) & (y_random <= (16 - x_random**2))
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the boundary line and shade the region
    plt.plot(x_curve, y_curve, label=r'Boundary: $y = 16 - x^2$', color='black', lw=2, linestyle='--')
    plt.fill_between(x_curve, 0, y_curve, color='teal', alpha=0.1, label='Feasible Region')
    
    # Scatter plot the points based on feasibility
    plt.scatter(x_random[feasible_mask], y_random[feasible_mask], 
                color='green', label='Feasible (Inside)', zorder=5, s=40)
    plt.scatter(x_random[~feasible_mask], y_random[~feasible_mask], 
                color='red', label='Infeasible (Outside)', zorder=5, s=40)
    
    # Axis and Plot styling
    plt.xlim(0, 5)
    plt.ylim(0, 22)
    plt.axhline(0, color='black', lw=1.5)
    plt.axvline(0, color='black', lw=1.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Example Feasibility Boundary')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_parabolic_feasibility_with_points()