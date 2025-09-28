import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Parameters ---
x = np.array([0.0, 0.0])  # Trajectory point
a = np.array([1.0, 0.0])  # Observed acceleration direction
epsilon = 0.1

# --- Force Residual Score Function ---
def force_residual_score(p, x, a, epsilon=0.1):
    vec = p - x
    norm_vec = np.linalg.norm(vec) + epsilon
    f_hat = vec / norm_vec
    residual = f_hat - a
    return np.linalg.norm(residual)

# --- Grid for p values ---
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

# --- Evaluate score on grid ---
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        p = np.array([X[i, j], Y[i, j]])
        Z[i, j] = force_residual_score(p, x, a, epsilon)

# --- Plotting ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Add trajectory point and 3D acceleration vector
z0 = force_residual_score(x, x, a)
ax.scatter(x[0], x[1], z0, c='red', s=50, label='x (trajectory point)')
ax.quiver(x[0], x[1], z0, a[0], a[1], 0, length=0.8, color='blue', linewidth=2, label='a (acceleration)')

# Labels and view
ax.set_title('3D Surface of Force Residual Score')
ax.set_xlabel('p_x')
ax.set_ylabel('p_y')
ax.set_zlabel('Score')
ax.view_init(elev=35, azim=135)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.legend()
plt.tight_layout()
plt.show()
