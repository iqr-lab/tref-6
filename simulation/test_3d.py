import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from force_point_fitter_3d import ForcePointFitter3D
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D





# --- 3D Trajectory generation ---
def generate_trajectory_with_single_affordance_3d(
    affordance=None,
    T=100,
    dt=0.1,
    initial_position=np.array([0.0, 0.0, 0.0]),
    force_noise_std=0.5,
    seed=380,
    affordance_range=([-5, 5], [-5, 5], [-5, 5])
):
    np.random.seed(seed)

    if affordance is None:
        # Randomly generate the ground-truth affordance within the range
        affordance = np.array([
            np.random.uniform(*affordance_range[0]),
            np.random.uniform(*affordance_range[1]),
            np.random.uniform(*affordance_range[2]),
        ])

    x = np.zeros((T, 3))  # position
    v = np.zeros((T, 3))  # velocity
    a = np.zeros((T, 3))  # acceleration

    x[0] = initial_position
    v[0] = np.random.uniform(-0.5, 0.5, size=3)  # random initial velocity

    def force_func(vec):
        direction = vec / (np.linalg.norm(vec) + 1e-8)
        magnitude = np.random.uniform(0, 1) * np.linalg.norm(vec)
        return magnitude * direction + magnitude * np.random.normal(0, force_noise_std, size=3)

    for t in range(1, T):
        vec_to_force = affordance - x[t - 1]
        force = force_func(vec_to_force)

        a[t - 1] = force
        v[t] = v[t - 1] + a[t - 1] * dt
        x[t] = x[t - 1] + v[t] * dt

    # Final acceleration estimate
    v = np.gradient(x, dt, axis=0)
    a = np.gradient(v, dt, axis=0)

    return x, a, affordance

def generate_rotational_trajectory_with_affordance_3d(
    affordance=np.array([0.0, 0.0, 0.0]),  # Hinge point
    radius=1.0,
    T=10,
    dt=0.1,
    initial_angle_deg=0,
    initial_angular_velocity=1.5,  # radians/sec
    force_strength=5.0,
    damping=0.05,
    axis=np.array([0, 0, 1]),  # Y-axis (vertical) rotation
    seed=42,
):
    np.random.seed(seed)
    axis = axis / np.linalg.norm(axis)

    # Get an orthogonal vector to define the door plane
    if np.allclose(axis, [0, 0, 1]):
        radial_vector = np.array([1, 0, 0]) * radius
    else:
        radial_vector = np.cross(axis, [0, 0, 1])
        radial_vector = radial_vector / np.linalg.norm(radial_vector) * radius

    theta = np.radians(initial_angle_deg)
    omega = initial_angular_velocity

    x = np.zeros((T, 3))  # position
    v = np.zeros((T, 3))  # linear velocity
    a = np.zeros((T, 3))  # linear acceleration

    # Initial position and velocity from angle and angular velocity
    def get_rotated_vector(angle):
        return (
            radial_vector * np.cos(angle) +
            np.cross(axis, radial_vector) * np.sin(angle) +
            axis * np.dot(axis, radial_vector) * (1 - np.cos(angle))
        )

    x[0] = affordance + get_rotated_vector(theta)
    v[0] = np.cross(axis, x[0] - affordance) * omega  # Tangential velocity

    for t in range(1, T):
        rel_pos = x[t - 1] - affordance
        tangential_dir = np.cross(axis, rel_pos)
        tangential_dir /= np.linalg.norm(tangential_dir) + 1e-8

        # Centripetal-like force pulling toward the hinge
        force = -force_strength * rel_pos + 0.0 * tangential_dir  # optionally add torque-like push
        force -= damping * v[t - 1]  # damping

        a[t - 1] = force
        v[t] = v[t - 1] + a[t - 1] * dt
        x[t] = x[t - 1] + v[t] * dt

    # Recalculate acceleration with better estimates
    v = np.gradient(x, dt, axis=0)
    a = np.gradient(v, dt, axis=0)

    return x, a, affordance

def evaluate_method_over_seeds(method='force_residual', n_seeds=10, visualize_last=True):
    errors = []
    all_data = [] 
    for seed in range(n_seeds):
        x, a, true_a = generate_trajectory_with_single_affordance_3d(seed=seed)
        fitter = ForcePointFitter3D(x, a)
        p, score, p_path, used_method = fitter.fit_one_point(method=method)
        error = np.linalg.norm(p - true_a)
        errors.append(error)

        all_data.append((error, x, a, true_a, p, p_path, method, score))
    

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    if visualize_last:
        # Visualize the worst round (highest error)
        worst_idx = np.argmax(errors)
        error, x, a, true_a, p, p_path, method, score = all_data[worst_idx]
        print(f"\n Best Seed (Seed {worst_idx}): Error = {error:.4f}")
        visualize_run(x, a, true_a, p, p_path, method, score, error)

    print(f"\nMethod: {method}")
    print(f"Average Error over {n_seeds} seeds: {mean_error:.4f} ± {std_error:.4f}")
    return mean_error, std_error

def evaluate_method_over_T(method='force_residual', seed=47, T_max = 100, visualize_Ts = []):
    errors = []
    T_list = list(range(1, T_max + 1))
    x_full, a_full, true_a = generate_trajectory_with_single_affordance_3d(seed=seed, T=T_max)
    for T in T_list:
        x = x_full[:T]
        a = a_full[:T]

        fitter = ForcePointFitter3D(x, a)
        p, score, p_path, used_method = fitter.fit_one_point(method=method)
        error = np.linalg.norm(p - true_a)
        errors.append(error)
        if T in visualize_Ts:
            print(f"\nT = {T}, Error = {error:.4f}")
            visualize_run(x, a, true_a, p, p_path, method, score, error)
    

    # Plotting T vs Error
    plt.figure(figsize=(8, 5))
    plt.plot(T_list, errors, label=f'{method}')
    plt.xlabel('Trajectory Length (T)')
    plt.ylabel('Inference Error')
    plt.title(f'Error vs. Trajectory Length (Seed {seed})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nMethod: {method}")
    return errors


def visualize_run(x, a, true_a, p, p_path, method, score, error):
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(x[:, 0], x[:, 1], x[:, 2], color='#1f77b4', linewidth=2.5, label='Trajectory')

    # Start and End
    ax.scatter(*x[0], color='blue', s=80, label='Start', edgecolors='k')
    ax.scatter(*x[-1], color='green', s=80, label='End', edgecolors='k')

    # Acceleration vectors
    ax.quiver(
        x[:, 0], x[:, 1], x[:, 2],
        a[:, 0], a[:, 1], a[:, 2],
        length=0.3, normalize=False,
        color='gray', alpha=0.5, linewidth=0.5
    )

    # True affordance and fitted point
    ax.scatter(*true_a, color='orange', s=150, marker='*', edgecolors='k', label='True Influence Point')
    ax.scatter(*p, color='red', s=120, edgecolors='k', label='Fitted Point')

    # Optimization path
    if p_path is not None:
        ax.plot(p_path[:, 0], p_path[:, 1], p_path[:, 2], 'r--', linewidth=2, label='Optimization Path')

    # Axis labels
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)

    # Adjust 3D viewing angle
    ax.view_init(elev=20, azim=-60)

    # Legend outside the plot
    #ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Make it a bit tighter
    plt.tight_layout()

    plt.show()





def evaluate_mean_std_over_T(method='force_residual', T_max=100, n_seeds=50):
    T_list = list(range(1, T_max + 1))
    error_matrix = []

    for seed in tqdm(range(n_seeds), desc="Running seeds"):
        x_full, a_full, true_a = generate_trajectory_with_single_affordance_3d(seed=seed, T=T_max)
        errors = []

        for T in tqdm(T_list, desc="Running T"):
            x = x_full[:T]
            a = a_full[:T]

            fitter = ForcePointFitter3D(x, a)
            p, score, p_path, _ = fitter.fit_one_point(method=method)
            error = np.linalg.norm(p - true_a)
            errors.append(error)

        error_matrix.append(errors)

    error_matrix = np.array(error_matrix)  # shape: (n_seeds, T_max)
    mean_errors = np.mean(error_matrix, axis=0)
    std_errors = np.std(error_matrix, axis=0)

    # Plot
    fig = plt.figure(figsize=(10, 6), dpi=300, constrained_layout=True)  # <-- Set dpi=300

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 14,        # general text
        'axes.labelsize': 14,   # axes label
        'xtick.labelsize': 12,  # x-axis ticks
        'ytick.labelsize': 12,  # y-axis ticks
        'legend.fontsize': 12,  # legend font
    })

    # Plot
    plt.plot(T_list, mean_errors, label=f'{method} (mean)', color='#D36000')
    plt.fill_between(
        T_list,
        mean_errors - std_errors,
        mean_errors + std_errors,
        color='#FEB83E',
        alpha=0.2,
        label='±1 std'
    )

    plt.xlabel('Trajectory Length (T)')
    plt.ylabel('Inference Error')

    # No title! (delete plt.title)

    plt.grid(False)
    plt.legend()

    #plt.tight_layout()
    plt.show()

    return T_list, mean_errors, std_errors, error_matrix

# --- Main script ---
if __name__ == '__main__':
    #x, a, true_a = generate_trajectory_with_single_affordance_3d()

    # x, a, true_a = generate_rotational_trajectory_with_affordance_3d()

    # # Fit a single force point
    #fitter = ForcePointFitter3D(x, a)
    # p, score, p_path, method = fitter.fit_one_point(method='force_residual') # or 'force_residual', 'cosine', 'quadratic', 'hybrid'
    # error = np.linalg.norm(p - true_a)
    # # --- 3D Interactive Visualization ---
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # Step 1: Show trajectory with start and end points
    # ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', linewidth=2)
    # ax.scatter(*x[0], c='blue', s=100, label='Start')
    # ax.scatter(*x[-1], c='green', s=100, label='End')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Trajectory')
    # ax.legend()
    # plt.tight_layout()
    # plt.draw()
    # print("Step 1: Trajectory shown. Click to show acceleration vectors.")
    # plt.waitforbuttonpress()

    # # Step 2: Show acceleration vectors
    # ax.quiver(x[:, 0], x[:, 1], x[:, 2], a[:, 0], a[:, 1], a[:, 2],
    #           length=0.1, normalize=True, color='gray', label='Acceleration')
    # ax.set_title('Trajectory with Acceleration Vectors')
    # ax.legend()
    # plt.draw()
    # print("Step 2: Acceleration shown. Click to show ground truth affordance.")
    # plt.waitforbuttonpress()

    # # Step 3: Show ground truth affordance point
    # ax.scatter(*true_a, c='orange', s=150, marker='*', label='True Affordance')
    # ax.set_title('Add Ground Truth Affordance')
    # ax.legend()
    # plt.draw()
    # print("Step 3: Ground truth affordance shown. Click to show fitted point.")
    # plt.waitforbuttonpress()

    # # Step 4: Show fitted point and optimization path
    # ax.scatter(*p, c='red', s=100, label='Fitted Point')
    # ax.plot(p_path[:, 0], p_path[:, 1], p_path[:, 2], 'r--', linewidth=1, label='Optimization Path')
    # ax.set_title(f'Fitted Point ({method}, Score = {score:.3f})'), Error (distance to GT): {error:.4f}')
    # ax.legend()
    # plt.draw()
    # print("Step 4: Fitted point and path shown. Done!")
    # plt.show()
    
    # Choose method: 'force_residual', 'cosine', 'quadratic', 'hybrid', 'inverse_dynamics'
    evaluate_method_over_seeds(method='force_residual', n_seeds=50)
    #evaluate_mean_std_over_T(method="force_residual")

