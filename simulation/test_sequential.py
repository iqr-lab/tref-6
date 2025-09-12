import numpy as np
import matplotlib.pyplot as plt
from force_point_fitter import ForcePointFitter

# --- Trajectory generation ---
# seed 60 failure case: straight line, parallel
# seed 61 failure case: too far away
def generate_trajectory_with_switching_affordances(
    affordance_a=np.array([2.0, 3.0]),
    affordance_b=np.array([7.0, -2.0]),
    T=100,
    dt=0.1,
    initial_position=np.array([0.0, 0.0]),
    force_noise_std=0.1,
    seed=47
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
        if t < T // 2:
            vec_to_force = (affordance_a - x[t - 1])
        else:
            vec_to_force = affordance_b - x[t - 1]

        force = force_func(vec_to_force)

        force += np.random.normal(0, force_noise_std, size=2)
        a[t - 1] = force
        v[t] = v[t - 1] + a[t - 1] * dt
        x[t] = x[t - 1] + v[t] * dt

    # Final acceleration estimate
    v = np.gradient(x, dt, axis=0)
    a = np.gradient(v, dt, axis=0)

    return x, a, affordance_a, affordance_b

# --- Main script ---
if __name__ == '__main__':
    # Generate trajectory
    x, a, true_a, true_b = generate_trajectory_with_switching_affordances()

    # Fit force points
    fitter = ForcePointFitter(x, a)
    # Fit force points and get paths
    p1, p2, score, p1_path, p2_path = fitter.fit_two_independent_points_weighted(split_ratio=0.5)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(x[:, 0], x[:, 1], label='Trajectory', linewidth=2)
    plt.quiver(x[:, 0], x[:, 1], a[:, 0], a[:, 1], color='gray', scale=10, width=0.003, label='Acceleration')

    plt.scatter(*true_a, c='orange', s=150, marker='*', label='True Affordance A')
    plt.scatter(*true_b, c='purple', s=150, marker='*', label='True Affordance B')

    plt.scatter(*p1, c='red', s=100, label='Fitted Point 1 (0-50%)')
    plt.scatter(*p2, c='blue', s=100, label='Fitted Point 2 (50-100%)')

    # Plot dashed optimization paths
    plt.plot(p1_path[:, 0], p1_path[:, 1], 'r--', linewidth=1, label='P1 Update Path')
    plt.plot(p2_path[:, 0], p2_path[:, 1], 'b--', linewidth=1, label='P2 Update Path')

    plt.axvline(x=x[len(x) // 2, 0], color='gray', linestyle='--', label='Split at 50%')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Fitted Force Points (Weighted Cosine Score = {score:.3f})')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()