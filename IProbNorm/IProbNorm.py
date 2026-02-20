from .utils import clamp, normal_pdf, truncated_normal_bin_pmf, fit_truncated_normal_per_column

import numpy as np
import torch

class IProbNorm:
    """
        IProbNorm is an interval-based model that addresses target uncertainty by discretizing values into intervals and
        employing attention and truncated normal distributions with trainable sigma values.

        Args:
            random_state (int): Seed for random number generation.
            weight_dim (int): Second dimensionality of W_k and W_q tensors in alpha_net.
            device (string): model device

            gauss_kernel (bool): Use Gaussian kernel instead of dot-product for attention computation.
            tau (float): Kernel width parameter for Gaussian attention (controls attention spread).
            tau_learnable (bool): Make tau learnable (optimize kernel width during training).

            k (int): Determines the amount by which the loss calculation interval is expanded on each side of the original interval
            normalized_loss_sigma (float): The value of sigma for the normal distribution of weights in the main loss component

            lr (float): Learning rate for the optimizer.
            beta1 (float): Beta1 parameter for the Adam optimizer.
            beta2 (float): Beta2 parameter for the Adam optimizer.
            lr_scheduler (bool): Whether to use a learning rate scheduler.

            batch_size (int): Count of examples in each test batch
            batch_query_key_rate (float): Query to Key size ratio when training a model

            reg_L2_alpha (float): L2-regularization coefficient for AlphaNet weights
            reg_L_sigma (float): regularization coefficient for trainable sigma

            min_sigma (float): lower bound of sigma value

            debug(bool): Print logs during training
        """


    def __init__(self,
                 # model parameters
                 random_state: int = 42,
                 weight_dim: int = 10,
                 device: str = "cpu",

                 gauss_kernel: bool = False,
                 tau: float = 0.5,
                 tau_learnable: bool = False,

                 k: int = 0,
                 normalized_loss_sigma: float = None,

                 # optimization parameters
                 lr: float = 0.1,
                 lr2: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 lr_scheduler: bool = False,
                 batch_size: int = 50,
                 batch_query_key_rate: float = 0.1,

                 reg_L2_alpha: float = 0.0001,
                 reg_L_sigma: float = 0.001,
                 min_sigma=0.001,

                 debug: bool = False,
                 ):

        self.random_state = random_state
        self.weight_dim = weight_dim
        self.device = device

        self.interval_bounds = None
        self.num_intervals = None

        self._interval_intersections = None
        self._interval_intersections_val = None
        self._alpha_net = None
        self._alpha_net_best = None

        self.Keys = None
        self.S_norm = None
        self.S_norm_best = None

        self._S_omega_params = None
        self._S_mu_params = None
        self._S_omega_params_best = None
        self._S_mu_params_best = None
        self.min_sigma = torch.tensor(min_sigma, dtype=torch.float32, device=self.device)

        self.M = 1

        self.lr = lr
        self.lr2 = lr2
        self.betas = (beta1, beta2)
        self.k = k
        self.nls = normalized_loss_sigma

        self.best_val_loss = None
        self.best_epoch = None

        self.best_val_loss_pi = None
        self.best_epoch_pi = None

        self.gauss_kernel = gauss_kernel
        self.tau = tau
        self.tau_learnable = tau_learnable

        self.lr_scheduler = lr_scheduler

        if tau_learnable:
            self.tau = torch.nn.Parameter(torch.tensor(tau))

        self.batch_size = batch_size
        self.batch_query_key_rate = batch_query_key_rate

        self._S_pi_logits_best = None
        self._S_pi_logits = None
        self.M_pi = None

        self.reg_L2_alpha = reg_L2_alpha
        self.reg_L_sigma = reg_L_sigma
        self.debug = debug

    def _create_optimizer(self, learn_sigma=False, learn_mu=False, learn_attention=True):
        param_groups = []

        lr = self.lr2

        if learn_attention:
            lr = self.lr
            if not self.gauss_kernel:
                param_groups.append({"params": self._alpha_net.parameters(), "weight_decay": 0})
            elif self.tau_learnable:
                param_groups.append({"params": [self.tau], "weight_decay": 0})

        if learn_mu:
            param_groups.append({"params": [self._S_mu_params], "weight_decay": 0})
        if learn_sigma:
            param_groups.append({"params": [self._S_omega_params], "weight_decay": 0})

        return torch.optim.Adam(param_groups, betas=self.betas, lr=lr)

    def _create_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5, last_epoch=-1)

    def _scale_y(self, y):
        return (y - self.y_bounds[0])/(self.y_bounds[1] - self.y_bounds[0])

    def _get_interval_bounds(self, y):
        self.y_bounds = [np.min(y), np.max(y)]
        self.y_range = np.max(y) - np.min(y)
        self.interval_bounds = np.unique(y)
        self.interval_bounds.sort()
        self.num_intervals = len(self.interval_bounds)

    def _get_interval_intersections(self, y):
        bounds = np.array(self.interval_bounds)
        y_lows = y[:, 0]
        y_highs = y[:, 1]
        N = y.shape[0]

        start_idxs = np.digitize(y_lows, bounds) - 1
        end_idxs = np.digitize(y_highs, bounds) - 1

        touch_high = y_highs == bounds[end_idxs]
        end_idxs = np.where(touch_high & (start_idxs != end_idxs), end_idxs - 1, end_idxs)

        no_intersection = start_idxs > end_idxs

        result = np.zeros((N, 2), dtype=int)
        result[:, 0] = start_idxs
        result[:, 1] = end_idxs

        result[no_intersection, :] = -1

        return result

    def compute_BrierScore(self, P, indicies=None, is_train=True, sum_vector=None):
        if len(P.shape) == 2:
            P = P.unsqueeze(-1)
        intersections = None
        if is_train:
            intersections = torch.tensor(self._interval_intersections, dtype=torch.int, device=P.device)
        else:
            intersections = torch.tensor(self._interval_intersections_val, dtype=torch.int, device=P.device)

        N, K, M = P.shape

        L = torch.zeros(size=[N, 1, M], dtype=torch.float32, device=P.device)

        for idx in range(N):
            i = idx
            if sum_vector is None:
                if not indicies is None:
                    i = indicies[idx]
                start_idx, end_idx = intersections[i]
                start_idx = clamp(start_idx, 0, self.num_intervals)
                end_idx = clamp(end_idx, 0, self.num_intervals)

                sums = torch.sum(P[idx, start_idx:end_idx + 1, :], dim=0)
            else:
                sums = sum_vector[i]
            L[idx, 0, :] = sums-1

        L = torch.pow(L, 2).sum()/(M*N)

        return L

    def compute_LogLoss(self, P, indicies=None, is_train=True, sum_vector=None):
        if len(P.shape) == 2:
            P = P.unsqueeze(-1)
        intersections = None
        if is_train:
            intersections = torch.tensor(self._interval_intersections, dtype=torch.int, device=P.device)
        else:
            intersections = torch.tensor(self._interval_intersections_val, dtype=torch.int, device=P.device)

        N, K, M = P.shape

        L = torch.zeros(size=[N, 1, M], dtype=torch.float32, device=P.device)

        for idx in range(N):
            i = idx
            if sum_vector is None:
                if not indicies is None:
                    i = indicies[idx]
                start_idx, end_idx = intersections[i]
                start_idx = clamp(start_idx, 0, self.num_intervals)
                end_idx = clamp(end_idx, 0, self.num_intervals)
                sums = torch.sum(P[idx, start_idx:end_idx + 1, :], dim=0)
            else:
                sums = sum_vector[i]
            L[idx, 0, :] = -torch.log(torch.maximum(sums, torch.tensor(10e-20, dtype=torch.float32, device="cpu")))

        L = L.sum()/(M*N)

        return L

    def compute_L_main(self, P, k=0, indicies=None, is_train=True):
        if len(P.shape) == 2:
            P = P.unsqueeze(-1)
        intersections = None
        if is_train:
            intersections = torch.tensor(self._interval_intersections, dtype=torch.int, device=P.device)
        else:
            intersections = torch.tensor(self._interval_intersections_val, dtype=torch.int, device=P.device)

        N, K, M = P.shape

        L = torch.zeros(size=[N, 1, M], dtype=torch.float32, device=P.device)

        for idx in range(N):
            i = idx
            if not indicies is None:
                i = indicies[idx]
            start_idx, end_idx = intersections[i]

            start_idx = clamp(start_idx-k, 0, self.num_intervals)
            end_idx = clamp(end_idx+k, 0, self.num_intervals)

            sums = None
            if self.nls is None:
                sums = torch.maximum(torch.sum(P[idx, start_idx:end_idx + 1, :], dim=0), torch.tensor(10e-20, dtype=torch.float32, device="cpu"))
            else:
                i_width = end_idx - start_idx

                i_X = np.array(range(-i_width//2, i_width//2+1))
                normal_weights = torch.Tensor(normal_pdf(i_X, self.nls), device=P.device)
                sums = torch.maximum(torch.einsum('i,ij->j', normal_weights, P[idx, start_idx:end_idx + 1, :]), torch.tensor(10e-20, dtype=torch.float32, device="cpu"))

            L[idx, 0, :] = -torch.log(sums)

        L = L.sum()/(M*N)

        return L

    def compute_L_sigmas(self, omega_matrix):
        sigma_matrix = self.compute_sigmas(omega_matrix)
        L = -torch.maximum(sigma_matrix, torch.tensor(10e-20, dtype=torch.float32, device="cpu")).log().mean()

        return L * self.reg_L_sigma

    def compute_sigmas(self, omega_matrix):
        return torch.pow(omega_matrix, 2) + self.min_sigma

    def compute_L2(self):
        if self.gauss_kernel:
            L2 = 0
            if self.tau_learnable:
                L2 = 0.025 / (self.tau + 1e-6)
        else:
            if hasattr(self._alpha_net, 'W'):
                L2 = self.reg_L2_alpha * (torch.sum(torch.pow(self._alpha_net.W, 2)))
            else:
                L2 = self.reg_L2_alpha * (torch.sum(torch.pow(self._alpha_net.W_k, 2))+torch.sum(torch.pow(self._alpha_net.W_q, 2)))
        return L2

    def compute_L(self, P, k=0, indicies=None, is_train=True):
        if len(P.shape) == 2:
            P = P.unsqueeze(-1)
        loss = self.compute_L_main(P, k, indicies, is_train)

        return loss + self.compute_L2()

    def _generate_norm_matrix(self, M):
        """
        :param M: Число генераций
        :return:
        """
        intersections = self._interval_intersections
        N = intersections.shape[0]
        S = np.zeros((N, 2, M), dtype=float)

        for i in range(N):
            start_idx, end_idx = intersections[i]

            if start_idx == -1 or start_idx > end_idx:
                continue

            b_left, b_right = self.interval_bounds[start_idx]/self.y_range, self.interval_bounds[end_idx]/self.y_range
            width = b_right - b_left

            S[i, 0, :] = b_left + width/2
            S[i, 1, :] = np.random.uniform(width/6, width/2, M)
        return torch.tensor(S, dtype=torch.float32, device=self.device)

    def _unwrap_norm_matrix(self, matrix, omega_unpack=True):
        intersections = self._interval_intersections
        N = intersections.shape[0]
        L = len(self.interval_bounds) + 1

        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(-1)

        M = matrix.shape[2]

        S = torch.zeros((N, L, M), device=self.device)

        mu = matrix[:, 0, :]
        sigma = matrix[:, 1, :]
        if omega_unpack:
            sigma = self.compute_sigmas(sigma)

        for i in range(N):
            start_idx, end_idx = intersections[i]

            if start_idx == -1 or start_idx > end_idx:
                continue

            S[i, :-1, :] = truncated_normal_bin_pmf(
                interval_bounds=self.interval_bounds,
                mu=mu[i,:]*self.y_range,
                sigma=sigma[i,:]*self.y_range,
                left_bound=self.interval_bounds[start_idx],
                right_bound=self.interval_bounds[end_idx]
            )

        return S

    def _gauss_weights(self, keys, query, is_train=False):
        diff = query.unsqueeze(1) - keys.unsqueeze(0)
        distance = torch.norm(diff, dim=2)
        weights = -(distance ** 2) / self.tau
        weights_max = torch.amax(weights, dim=1, keepdim=True)
        weights_shifted = weights - weights_max
        kernel = torch.exp(weights_shifted)
        kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-12)

        if is_train and self.tau_learnable:
            if self.debug:
                print(f"tau = {self.tau.data}")

        return kernel

    def _is_learnable(self):
        return not self.gauss_kernel or self.tau_learnable

    def predict(self, X_new, approximate=False):
        X = X_new
        if type(X) == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32, device=self.device)

        if not self.gauss_kernel:
            A = self._alpha_net.forward(self.Keys, X)
        else:
            A = self._gauss_weights(self.Keys, X)

        probs = torch.zeros([X.shape[0], self.num_intervals + 1])
        if hasattr(self, 'S_pi'):
            S_pi = self._unwrap_norm_matrix(self.S_pi).squeeze().detach().clone()
            probs = torch.einsum('kb,bt->kt', A, S_pi)
        else:
            S = self._unwrap_norm_matrix(self.S_norm, omega_unpack=False).detach().clone()
            for j in range(self.num_intervals + 1):
                for m in range(self.M):
                    probs[:, j] += torch.matmul(A, S[:, j, m])/self.M

        if approximate:
            mu, sigma = fit_truncated_normal_per_column(probs, self.interval_bounds)
            return [mu, sigma]

        return probs

    def get_alpha_net(self):
        return self._alpha_net