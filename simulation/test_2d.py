import numpy as np
import matplotlib.pyplot as plt
from force_point_fitter import ForcePointFitter

# --- Trajectory generation ---
def generate_trajectory_with_single_affordance(
    affordance=np.array([2.0, 3.0]),
    T=100,
    dt=0.1,
    initial_position=np.array([0.0, 0.0]),
    force_noise_std=0.1,
    seed=12
):
    np.random.seed(seed)

    x = np.zeros((T, 2))  # position
    v = np.zeros((T, 2))  # velocity
    a = np.zeros((T, 2))  # acceleration

    x[0] = initial_position
    v[0] = np.random.uniform(-0.5, 0.5, size=2)  # random initial velocity

    def force_func(vec):
        direction = vec / (np.linalg.norm(vec) + 1e-8)
        magnitude = np.random.uniform(0, 1)
        return magnitude * direction

    for t in range(1, T):
        vec_to_force = affordance - x[t - 1]
        force = force_func(vec_to_force)


        force += np.random.normal(0, force_noise_std, size=2)
        a[t - 1] = force
        v[t] = v[t - 1] + a[t - 1] * dt
        x[t] = x[t - 1] + v[t] * dt

    # Final acceleration estimate
    v = np.gradient(x, dt, axis=0)
    a = np.gradient(v, dt, axis=0)

    return x, a, affordance

# --- Main script ---
if __name__ == '__main__':
    # Generate trajectory with a single affordance point
    x, a, true_a = generate_trajectory_with_single_affordance()

    # Fit a single force point
    fitter = ForcePointFitter(x, a)
    p, score, p_path = fitter.fit_one_point_weighted()

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(x[:, 0], x[:, 1], label='Trajectory', linewidth=2)
    plt.quiver(x[:, 0], x[:, 1], a[:, 0], a[:, 1], color='gray', scale=10, width=0.003, label='Acceleration')

    plt.scatter(*true_a, c='orange', s=150, marker='*', label='True Affordance')
    plt.scatter(*p, c='red', s=100, label='Fitted Point')

    # Plot dashed optimization path
    plt.plot(p_path[:, 0], p_path[:, 1], 'r--', linewidth=1, label='Optimization Path')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Single Force Point Fitting (Weighted Cosine Score = {score:.3f})')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
