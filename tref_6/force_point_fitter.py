import numpy as np

class ForcePointFitter:
    def __init__(self, x, a, dt=0.1):
        """
        Parameters:
        - x: trajectory positions, shape (T, 2)
        - a: trajectory accelerations, shape (T, 2)
        - dt: time step
        """
        self.x = x
        self.a = a
        self.T = len(x)
        self.dt = dt

    def adam_optimize_abs_single(self, x, a, initial_p, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, iterations=2000, noise_every=40, tol=1e-5, patience=30, pull=True):
        p = initial_p.copy()
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        path = [p.copy()]

        best_score = -np.inf
        stagnation_counter = 0

        for t in range(1, iterations + 1):
            #grad = self._gradient_abs_cosine_single(p, x, a, pull=pull)
            grad = self._gradient_force_residual_score(p, x, a)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p += lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % noise_every == 0:
                p += np.random.normal(0, 0.05, size=p.shape)

            path.append(p.copy())

            # Compute score for early stopping check
            #score = self._abs_cosine_similarity_weighted(p, x, a, pull=pull)
            score = self._force_residual_score(p, x, a)
            if score > best_score + tol:
                best_score = score
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= patience:
                print(f"Early stopping at step {t}, score={score:.4f}")
                break

        return np.array(path)

    def _force_residual_score(self, p, x, a):
        vectors = p - x
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        predicted_force = vectors / (norms + 1e-6)
        residuals = predicted_force - a
        return -np.mean(np.linalg.norm(residuals, axis=1))

    def _gradient_force_residual_score(self, p, x, a, eps=1e-4):
        grad = np.zeros_like(p)
        for i in range(len(p)):
            p_eps1 = p.copy()
            p_eps2 = p.copy()
            p_eps1[i] += eps
            p_eps2[i] -= eps
            grad[i] = (
                self._force_residual_score(p_eps1, x, a) -
                self._force_residual_score(p_eps2, x, a)
            ) / (2 * eps)
        return grad

    # def _gradient_abs_cosine_single(self, p, x, a, eps=1e-4, pull=True):
    #     grad = np.zeros_like(p)
    #     for i in range(len(p)):
    #         p_eps1 = p.copy()
    #         p_eps2 = p.copy()
    #         p_eps1[i] += eps
    #         p_eps2[i] -= eps
    #         grad[i] = (
    #             self._abs_cosine_similarity_weighted(p_eps1, x, a) -
    #             self._abs_cosine_similarity_weighted(p_eps2, x, a)
    #         ) / (2 * eps)
    #     return grad

    # def _abs_cosine_similarity_weighted(self, p, x, a, decay_radius = 1, pull=True):
    #     vectors = x - p
    #     dot = np.einsum('ij,ij->i', vectors, a)
    #     norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(a, axis=1)
    #     weights = np.linalg.norm(a, axis=1)
    #     distances = np.linalg.norm(vectors, axis=1)
    #     decay = 2 - np.exp(-distances / decay_radius)
    #     decay = 1
    #     #return np.sum(weights * np.abs(dot / (norms + 1e-8))) / (np.sum(weights) + 1e-8)
    #     if pull:
    #         return -np.sum(weights * (dot / (norms + 1e-8)) * decay)  / (np.sum(weights) + 1e-8)
    #     else:
    #         return np.sum(weights * (dot / (norms + 1e-8)) * decay) / (np.sum(weights) + 1e-8)


    def fit_two_independent_points_weighted(self, split_ratio=0.5, lr=0.05, steps=2000):
        """
        Splits the trajectory into two segments and fits a point to each.
        Returns (p1, p2, weighted_score)
        """
        split = int(self.T * split_ratio)
        x1, a1 = self.x[:split], self.a[:split]
        x2, a2 = self.x[split:], self.a[split:]

        init_p1 = self.highest_accel_initialization(x1, a1)
        init_p2 = self.highest_accel_initialization(x2, a2)

        p1_path = self.adam_optimize_abs_single(x1, a1, initial_p=init_p1, lr=lr, iterations=steps)
        p2_path = self.adam_optimize_abs_single(x2, a2, initial_p=init_p2, lr=lr, iterations=steps)
        p1 = p1_path[-1]
        p2 = p2_path[-1]


        score1 = self._force_residual_score(p1, x1, a1)
        score2 = self._force_residual_score(p2, x2, a2)
        score = (score1 + score2) / 2

        return p1, p2, score, p1_path, p2_path



        # # Fit each point separately
        # p1_path = self.adam_optimize_abs_single(x1, a1, initial_p=init_p1, lr=lr, iterations=steps, pull=True)
        # p2_path = self.adam_optimize_abs_single(x2, a2, initial_p=init_p2, lr=lr, iterations=steps, pull=True)

        # p1 = p1_path[-1]
        # p2 = p2_path[-1]

        # # Compute weighted score
        # vectors1 = x1 - p1
        # vectors2 = x2 - p2
        # dot1 = np.einsum('ij,ij->i', vectors1, a1)
        # dot2 = np.einsum('ij,ij->i', vectors2, a2)
        # norm1 = np.linalg.norm(vectors1, axis=1) * np.linalg.norm(a1, axis=1)
        # norm2 = np.linalg.norm(vectors2, axis=1) * np.linalg.norm(a2, axis=1)
        # weight1 = np.linalg.norm(a1, axis=1)
        # weight2 = np.linalg.norm(a2, axis=1)

        # score1 = weight1 * np.abs(dot1 / (norm1 + 1e-8))
        # score2 = weight2 * np.abs(dot2 / (norm2 + 1e-8))
        # score = (np.sum(score1) + np.sum(score2)) / (np.sum(weight1) + np.sum(weight2) + 1e-8)
        # # score1 = np.abs(dot1 / (norm1 + 1e-8))
        # # score2 = np.abs(dot2 / (norm2 + 1e-8))
        # # score = (np.sum(score1) + np.sum(score2)) / (len(score1) + len(score2))
        # return p1, p2, score, p1_path, p2_path

    def fit_one_point_weighted(self, lr=0.05, steps=2000):
        """
        Fit a single force application point over the entire trajectory
        using force-weighted cosine similarity (not absolute).
        """
        #initial_p = np.random.uniform(-2, 8, size=2)
        initial_p = self.highest_accel_initialization(self.x, self.a)
        p_path = self.adam_optimize_abs_single(self.x, self.a, initial_p, lr=lr, iterations=steps)
        p_final = p_path[-1]

        score = self._force_residual_score(p_final, self.x, self.a)
        # p_path = self.adam_optimize_abs_single(self.x, self.a, initial_p, lr=lr, iterations=steps)
        # p_final = p_path[-1]

        # Compute weighted cosine similarity score (not abs)
        # vectors = self.x - p_final
        # dot = np.einsum('ij,ij->i', vectors, self.a)
        # norm = np.linalg.norm(vectors, axis=1) * np.linalg.norm(self.a, axis=1)
        # weights = np.linalg.norm(self.a, axis=1)

        # cosine = dot / (norm + 1e-8)  # keep sign
        # weighted_score = np.sum(weights * cosine) / (np.sum(weights) + 1e-8)
        #unweighted_score = np.mean(cosine)

        return p_final, score, p_path

    @staticmethod
    def highest_accel_initialization(x, a, k=5, noise_std=0.1):
        """
        Initialize near the average of the top-k highest-acceleration frames.
        Optionally add small Gaussian noise.
        """
        k = int(k)
        norms = np.linalg.norm(a, axis=1)
        top_k_indices = np.argsort(norms)[-k:]
        center = np.mean(x[top_k_indices], axis=0)
        return center + np.random.normal(0, noise_std, size=2)
