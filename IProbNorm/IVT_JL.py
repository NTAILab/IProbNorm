from .IProbNorm import IProbNorm

import numpy as np
import torch
from .alpha_net import AlphaNet
import sys
import copy

class IVT_JL(IProbNorm):

    def fit(self, x : np.array, y : np.array, x_val : np.array, y_val : np.array, epochs):
        """
        Train the interval-based model using the joint learning approach.

        Parameters
        ----------
        x : np.array
            Training feature matrix of shape (n_samples, n_features).
        y : np.array
            Training target intervals of shape (n_samples, 2), where:
            - y[:, 0] contains lower bounds of event intervals
            - y[:, 1] contains upper bounds of event intervals
        x_val : np.array
            Validation feature matrix of shape (n_val_samples, n_features).
        y_val : np.array
            Validation target intervals of shape (n_val_samples, 2).
        epochs : int
            Number of training epochs (complete passes through training data).
        """

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        k = self.k
        self.best_val_loss = sys.float_info.max
        self.best_epoch = 0

        self.X_val = x_val
        self.X_train = x
        self.Y_train = y
        self.Y_val = y_val

        self._get_interval_bounds(y)

        self._interval_intersections = self._get_interval_intersections(y)
        self._interval_intersections_val = self._get_interval_intersections(y_val)

        self.S_norm = self._generate_norm_matrix(M=1000)
        self._S_omega_params = torch.nn.Parameter(torch.mean(torch.sqrt(self.S_norm[:, 1].detach().clone()), dim=1))
        self._S_mu_params = torch.nn.Parameter(torch.mean(self.S_norm[:, 0].detach().clone(), dim=1))
        self._S_omega_params_best = self._S_omega_params.clone()
        self._S_mu_params_best = self._S_mu_params.clone()

        optimizer = None
        optimizer2 = None
        scheduler = None
        scheduler2 = None
        if self._is_learnable():
            if not self.gauss_kernel:
                self._alpha_net = AlphaNet(input_dim=x.shape[1], weight_dim=self.weight_dim).to(self.device)
            optimizer = self._create_optimizer(learn_sigma=False, learn_mu=False, learn_attention=True)
            optimizer2 = self._create_optimizer(learn_sigma=True, learn_mu=False, learn_attention=False)
            scheduler = self._create_scheduler(optimizer) if self.lr_scheduler else None
            scheduler2 = self._create_scheduler(optimizer2) if self.lr_scheduler else None

        self.Keys = torch.tensor(x, dtype=torch.float32, device=self.device)

        self.losses_train, self.losses_val = [], []
        self.brier_train, self.brier_val = [], []
        self.log_losses_train, self.log_losses_val = [], []

        for epoch in range(epochs):
            indices = torch.randperm(x.shape[0], device="cpu")

            log_str = f"Epoch {epoch}: "

            # === Train ===

            self.losses_train.append(0)
            self.brier_train.append(0)
            self.log_losses_train.append(0)

            batches = 0.0

            query_size = int(x.shape[0] * self.batch_query_key_rate)
            query_indices = torch.split(indices, query_size)

            for query_num, query_idx in enumerate(query_indices):
                key_idx = torch.ones(x.shape[0], dtype=torch.bool)
                key_idx[query_idx] = False

                if self._is_learnable():
                    optimizer.zero_grad()
                    optimizer2.zero_grad()

                S_params = torch.stack([self._S_mu_params, self._S_omega_params], dim=1)
                S_all = self._unwrap_norm_matrix(S_params).squeeze()

                Query_train = torch.tensor(x[query_idx], dtype=torch.float32, device=self.device)
                Key_train = torch.tensor(x[key_idx], dtype=torch.float32, device=self.device)

                A = None
                S = S_all[key_idx, :]

                if not self.gauss_kernel:
                    self._alpha_net.train()
                    A = self._alpha_net.forward(Key_train, Query_train)
                else:
                    A = self._gauss_weights(Key_train, Query_train, is_train=True)

                probs = torch.einsum('kb,bt->kt', A, S)

                loss_train_batch = self.compute_L(P=probs, k=k, is_train=True, indicies=query_idx) + self.compute_L_sigmas(self._S_omega_params)
                score_brier_batch = float(self.compute_BrierScore(P=probs, is_train=True, indicies=query_idx).item())
                log_losses_batch = float(self.compute_LogLoss(P=probs, is_train=True, indicies=query_idx).item())

                self.losses_train[-1] += float(loss_train_batch.item())
                self.brier_train[-1] += float(score_brier_batch)
                self.log_losses_train[-1] += float(log_losses_batch)
                batches += 1

                if self._is_learnable():
                    loss_train_batch.backward()
                    optimizer.step()
                    optimizer2.step()

                    if self.lr_scheduler:
                        scheduler.step()
                        scheduler2.step()

            self.losses_train[-1] /= batches
            self.brier_train[-1] /= batches
            self.log_losses_train[-1] /= batches

            log_str += f"train={self.losses_train[-1]} "
            log_str += f"train_br={self.brier_train[-1]} "

            # === Test ===

            if x_val is not None:
                with torch.no_grad():
                    S_params = torch.stack([self._S_mu_params, self._S_omega_params], dim=1)
                    S_all = self._unwrap_norm_matrix(S_params).squeeze()

                    indices = torch.arange(x_val.shape[0], device="cpu")
                    batch_indices = torch.split(indices, self.batch_size)

                    self.losses_val.append(0)
                    self.brier_val.append(0)
                    self.log_losses_val.append(0)

                    batches = 0

                    for query_num, query_idx in enumerate(batch_indices):
                        Query_val = torch.tensor(x_val[query_idx], dtype=torch.float32, device=self.device)

                        if not self.gauss_kernel:
                            self._alpha_net.eval()
                            A = self._alpha_net.forward(self.Keys, Query_val)
                        else:
                            A = self._gauss_weights(self.Keys, Query_val)

                        probs = torch.einsum('kb,bt->kt', A, S_all)

                        loss = float(self.compute_L(probs, k, is_train=False, indicies=query_idx) + self.compute_L_sigmas(self._S_omega_params))
                        score_brier = float(
                            self.compute_BrierScore(P=probs, is_train=False, indicies=query_idx).item())
                        log_loss = float(self.compute_LogLoss(P=probs, is_train=False, indicies=query_idx).item())

                        self.losses_val[-1] += loss
                        self.brier_val[-1] += score_brier
                        self.log_losses_val[-1] += log_loss
                        batches += 1.0

                    self.losses_val[-1] /= batches
                    self.brier_val[-1] /= batches
                    self.log_losses_val[-1] /= batches

                    log_str += f"val={self.losses_val[-1]} "
                    log_str += f"val_br={self.brier_val[-1]} "

                    if self.losses_val[-1] < self.best_val_loss:
                        self._S_mu_params_best = self._S_mu_params.clone()
                        self._S_omega_params_best = self._S_omega_params.clone()
                        self.best_val_loss = self.losses_val[-1]
                        self.best_epoch = epoch
                        self._alpha_net_best = copy.deepcopy(self._alpha_net)

            if self.debug:
                print(log_str)

        self.S_pi = torch.stack([self._S_mu_params_best, self._S_omega_params_best], dim=1)
        if not self.gauss_kernel:
            if self._alpha_net_best:
                self._alpha_net = copy.deepcopy(self._alpha_net_best)