import numpy as np

class ForcePointFitter3D:
    def __init__(self, x, a, dt=0.1):
        """
        Parameters:
        - x: trajectory positions, shape (T, 3)
        - a: trajectory accelerations, shape (T, 3)
        - dt: time step
        """
        self.x = x
        self.a = a
        self.T = len(x)
        self.dt = dt

    def adam_optimize(self, score_fn, grad_fn, initial_p, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8,
                                  iterations=2000, noise_every=40, tol=1e-5, patience=30):
        p = initial_p.copy()
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        path = [p.copy()]

        best_score = -np.inf
        stagnation_counter = 0

        for t in range(1, iterations + 1):
            grad = grad_fn(p, self.x, self.a)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p += lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % noise_every == 0:
                p += np.random.normal(0, 0.05, size=p.shape)

            path.append(p.copy())

            #score = self._abs_cosine_similarity(p, x, a)
            #score = self._force_residual_score(p, x, a)
            score = score_fn(p, self.x, self.a)
            if score > best_score + tol:
                best_score = score
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= patience:
                break

        return np.array(path)
    # --- Score Functions ---
    def _force_residual_score(self, p, x, a):
        vectors = p - x
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        predicted_force = vectors / (norms + 1e-6)
        residuals = predicted_force - a
        return -np.mean(np.linalg.norm(residuals, axis=1))

    def _abs_cosine_similarity(self, p, x, a):
        vectors = x - p
        dot = np.einsum('ij,ij->i', vectors, a)
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(a, axis=1)
        return np.mean(np.abs(dot / (norms + 1e-8)))

    def _quadratic_residual_score(self, p, x, a):
        predicted = p - x
        residuals = predicted - a
        return -np.mean(np.linalg.norm(residuals, axis=1)**2)
    
    def _hybrid_residual_score(self, p, x, a):
        dists = np.linalg.norm(p - x, axis=1, keepdims=True)
        magnitudes = dists
        predicted = (p - x) * magnitudes / (dists + 1e-6)
        residuals = predicted - a
        return -np.mean(np.linalg.norm(residuals, axis=1))
    
    # --- gradient functions ---
    def _gradient(self, score_fn, p, x, a, eps=1e-4):
        grad = np.zeros_like(p)
        for i in range(len(p)):
            p_eps1 = p.copy()
            p_eps2 = p.copy()
            p_eps1[i] += eps
            p_eps2[i] -= eps
            grad[i] = (score_fn(p_eps1, x, a) - score_fn(p_eps2, x, a)) / (2 * eps)
        return grad

    # --- initialization ---
    def highest_accel_initialization(self, x, a, k=5, noise_std=0.1):
        k = int(k)
        norms = np.linalg.norm(a, axis=1)
        top_k_indices = np.argsort(norms)[-k:]
        center = np.mean(x[top_k_indices], axis=0)
        return center + np.random.normal(0, noise_std, size=3)

    def fit_one_point(self, method='force_residual', lr=0.05, steps=200):
        initial_p = self.highest_accel_initialization(self.x, self.a)
        score_fn = {
            'force_residual': self._force_residual_score,
            'cosine': self._abs_cosine_similarity,
            'quadratic': self._quadratic_residual_score,
            'hybrid': self._hybrid_residual_score
        }[method]
        grad_fn = lambda p, x, a: self._gradient(score_fn, p, x, a)
        p_path = self.adam_optimize(score_fn, grad_fn, initial_p, lr=lr, iterations=steps)
        p_final = p_path[-1]
        final_score = score_fn(p_final, self.x, self.a)
        return p_final, final_score, p_path, method
