import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pickle
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.exceptions import ConvergenceWarning

def _suggest_num_blocks(input_length, kernel_size=3, dilation_base=2):
    extra_needed = max(0.0, float(input_length - kernel_size))
    if extra_needed <= 0:
        return 1
    required_sum = extra_needed / (kernel_size - 1)
    if dilation_base == 1:
        return max(1, int(np.ceil(required_sum)))
    covered_sum = 0.0
    dilation = 1.0
    blocks = 0
    while covered_sum < required_sum:
        covered_sum += dilation
        dilation *= dilation_base
        blocks += 1
    return max(1, blocks)
def dialtedCNN(input_length, hidden_channels=64, kernel_size=3, dilation_base=2, num_blocks=None):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd to keep sequence length fixed.")
    if dilation_base < 1:
        raise ValueError("dilation_base must be >= 1.")
    if num_blocks is None:
        num_blocks = _suggest_num_blocks(
            input_length=input_length,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
        )
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1.")
    class DilatedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_blocks = num_blocks
            stem_pad = (kernel_size - 1) // 2
            self.stem = nn.Conv1d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=stem_pad,
                padding_mode="circular",
            )
            self.blocks = nn.ModuleList()
            for i in range(num_blocks):
                dilation = dilation_base ** i
                pad = dilation * (kernel_size - 1) // 2
                self.blocks.append(
                    nn.Conv1d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=pad,
                        padding_mode="circular",
                    )
                )
            self.readout = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        def forward(self, x):
            if x.dim() == 2:
                h = x.unsqueeze(1)
            elif x.dim() == 3 and x.size(1) == 1:
                h = x
            else:
                raise ValueError("Expected x shape [batch, L] or [batch, 1, L].")
            h = F.silu(self.stem(h))
            for conv in self.blocks:
                h = h + F.silu(conv(h))
            out = self.readout(h).squeeze(1)
            return torch.tanh(out)
    return DilatedCNN()

class dilatedCNNModel:
    def __init__(
        self,
        hidden_channels=64,
        kernel_size=3,
        dilation_base=2,
        num_blocks=None,
        lr=1e-3,
        batch_size=256,
        epochs=20,
        augment_z2=True,
        seed=0,
        device=None,
        verbose=1,
    ):
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.num_blocks = num_blocks
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.augment_z2 = augment_z2
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.model_ = None
        self.optimizer_ = None
        self.history_ = None
        self.input_length_ = None
        self.device_ = None
    def get_params(self, deep=True):
        return {
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "dilation_base": self.dilation_base,
            "num_blocks": self.num_blocks,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "augment_z2": self.augment_z2,
            "seed": self.seed,
            "device": self.device,
            "verbose": self.verbose,
        }
    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self
    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    def _to_numpy_2d(self, X, name="X"):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"{name} must have shape (N, L), got {X_arr.shape}")
        return X_arr
    def _resolve_device(self):
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    def _build_model(self, input_length):
        self.input_length_ = int(input_length)
        self.model_ = dialtedCNN(
            input_length=self.input_length_,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            num_blocks=self.num_blocks,
        ).to(self.device_)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
    def fit(self, X, y, val_data=None, val_split=0.0, return_history=False, plot=False):
        self._set_seed()
        self.device_ = self._resolve_device()
        X_arr = self._to_numpy_2d(X, "X")
        y_arr = self._to_numpy_2d(y, "y")
        if X_arr.shape != y_arr.shape:
            raise ValueError(f"Shape mismatch: X{X_arr.shape} vs y{y_arr.shape}")
        N, L = X_arr.shape
        self._build_model(L)
        if val_data is not None:
            X_val_arr = self._to_numpy_2d(val_data[0], "X_val")
            y_val_arr = self._to_numpy_2d(val_data[1], "y_val")
            if X_val_arr.shape != y_val_arr.shape:
                raise ValueError(f"Validation shape mismatch: X_val{X_val_arr.shape} vs y_val{y_val_arr.shape}")
            if X_val_arr.shape[1] != L:
                raise ValueError("Training and validation must use the same system size L.")
            X_train_arr, y_train_arr = X_arr, y_arr
            has_val = True
        else:
            has_val = val_split > 0.0
            if has_val:
                if not (0.0 < val_split < 1.0):
                    raise ValueError("val_split must be in (0, 1).")
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(N)
                n_train = int((1.0 - val_split) * N)
                n_train = max(1, min(n_train, N - 1))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                X_train_arr, y_train_arr = X_arr[train_idx], y_arr[train_idx]
                X_val_arr, y_val_arr = X_arr[val_idx], y_arr[val_idx]
            else:
                X_train_arr, y_train_arr = X_arr, y_arr
                X_val_arr, y_val_arr = None, None
        X_train = torch.from_numpy(X_train_arr)
        y_train = torch.from_numpy(y_train_arr)
        gen = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=gen,
        )
        if has_val:
            X_val = torch.from_numpy(X_val_arr)
            y_val = torch.from_numpy(y_val_arr)
            val_loader = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            val_loader = None
        self.history_ = {
            "train_loss": [],
            "train_mse": [],
            "val_mse": [],
        }
        eps = 1e-6
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_count = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                if self.augment_z2:
                    xb = torch.cat([xb, -xb], dim=0)
                    yb = torch.cat([yb, -yb], dim=0)
                pred_m = self.model_(xb)
                pred_p = ((pred_m + 1.0) * 0.5).clamp(eps, 1.0 - eps)
                target_p = (yb + 1.0) * 0.5
                loss = F.binary_cross_entropy(pred_p, target_p)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                n_elem = yb.numel()
                train_loss_sum += loss.item() * n_elem
                train_mse_sum += F.mse_loss(pred_m, yb, reduction="sum").item()
                train_count += n_elem
            train_loss = train_loss_sum / train_count
            train_mse = train_mse_sum / train_count
            self.history_["train_loss"].append(train_loss)
            self.history_["train_mse"].append(train_mse)
            if val_loader is not None:
                self.model_.eval()
                val_mse_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        pred_m = self.model_(xb)
                        val_mse_sum += F.mse_loss(pred_m, yb, reduction="sum").item()
                        val_count += yb.numel()
                val_mse = val_mse_sum / val_count
            else:
                val_mse = None
            self.history_["val_mse"].append(val_mse)
            if self.verbose:
                if val_mse is None:
                    print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f}")
                else:
                    print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(self.history_["train_loss"], label="train_loss")
            if any(v is not None for v in self.history_["val_mse"]):
                val_curve = [np.nan if v is None else v for v in self.history_["val_mse"]]
                plt.plot(val_curve, label="val_mse")
            plt.xlabel("epoch")
            plt.legend()
            plt.tight_layout()
            plt.show()
        if return_history:
            return self, self.history_
        return self
    def _predict_magnetization_array(self, X):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_arr = self._to_numpy_2d(X, "X")
        if X_arr.shape[1] != self.input_length_:
            raise ValueError(f"Expected input length {self.input_length_}, got {X_arr.shape[1]}")
        X_tensor = torch.from_numpy(X_arr)
        loader = DataLoader(X_tensor, batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device_)
                pm = self.model_(xb)
                preds.append(pm.cpu().numpy())
        return np.concatenate(preds, axis=0)
    def predict_magnetization(self, X):
        return self._predict_magnetization_array(X)
    def predict_proba(self, X):
        m = self._predict_magnetization_array(X)
        return np.clip((m + 1.0) * 0.5, 0.0, 1.0)
    def predict(self, X):
        m = self._predict_magnetization_array(X)
        return np.where(m >= 0.0, 1.0, -1.0).astype(np.float32)
    def score(self, X, y, metric="neg_mse"):
        y_arr = self._to_numpy_2d(y, "y")
        m = self._predict_magnetization_array(X)
        if y_arr.shape != m.shape:
            raise ValueError(f"Shape mismatch: predictions{m.shape} vs y{y_arr.shape}")
        mse = float(np.mean((m - y_arr) ** 2))
        if metric == "neg_mse":
            return -mse
        if metric == "mse":
            return mse
        if metric == "accuracy":
            y_pred = np.where(m >= 0.0, 1.0, -1.0)
            return float(np.mean(y_pred == y_arr))
        raise ValueError("metric must be one of {'neg_mse', 'mse', 'accuracy'}")
    def total_parameters(self):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return int(sum(p.numel() for p in self.model_.parameters()))
    def trainable_parameters(self):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return int(sum(p.numel() for p in self.model_.parameters() if p.requires_grad))
    def parameter_counts(self):
        return {
            "total": self.total_parameters(),
            "trainable": self.trainable_parameters(),
        }
    def save(self, path, extra=None):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        checkpoint = {
            "params": self.get_params(),
            "input_length": self.input_length_,
            "model_state_dict": self.model_.state_dict(),
            "history": self.history_,
            "extra": extra,
        }
        torch.save(checkpoint, path)
    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint["params"])
        model.device_ = model._resolve_device()
        model._build_model(checkpoint["input_length"])
        model.model_.load_state_dict(checkpoint["model_state_dict"])
        model.history_ = checkpoint.get("history")
        return model

class MLPModel:
    def __init__(
        self,
        hidden_dims=(256, 256),
        dropout=0.0,
        lr=1e-3,
        batch_size=256,
        epochs=20,
        augment_z2=True,
        seed=0,
        device=None,
        verbose=1,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.augment_z2 = augment_z2
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.model_ = None
        self.optimizer_ = None
        self.history_ = None
        self.input_length_ = None
        self.device_ = None
    def get_params(self, deep=True):
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "augment_z2": self.augment_z2,
            "seed": self.seed,
            "device": self.device,
            "verbose": self.verbose,
        }
    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self
    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    def _to_numpy_2d(self, X, name="X"):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"{name} must have shape (N, L), got {X_arr.shape}")
        return X_arr
    def _resolve_device(self):
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    def _build_model(self, input_length):
        hidden_dims = self.hidden_dims
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        hidden_dims = tuple(int(h) for h in hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")
        class _MLP(nn.Module):
            def __init__(self, in_dim, h_dims, dropout):
                super().__init__()
                layers = []
                prev = in_dim
                for h in h_dims:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.SiLU())
                    if dropout > 0.0:
                        layers.append(nn.Dropout(dropout))
                    prev = h
                layers.append(nn.Linear(prev, in_dim))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return torch.tanh(self.net(x))
        self.input_length_ = int(input_length)
        self.model_ = _MLP(self.input_length_, hidden_dims, float(self.dropout)).to(self.device_)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
    def fit(self, X, y, val_data=None, val_split=0.0, return_history=False, plot=False):
        self._set_seed()
        self.device_ = self._resolve_device()
        X_arr = self._to_numpy_2d(X, "X")
        y_arr = self._to_numpy_2d(y, "y")
        if X_arr.shape != y_arr.shape:
            raise ValueError(f"Shape mismatch: X{X_arr.shape} vs y{y_arr.shape}")
        N, L = X_arr.shape
        self._build_model(L)
        if val_data is not None:
            X_val_arr = self._to_numpy_2d(val_data[0], "X_val")
            y_val_arr = self._to_numpy_2d(val_data[1], "y_val")
            if X_val_arr.shape != y_val_arr.shape:
                raise ValueError(f"Validation shape mismatch: X_val{X_val_arr.shape} vs y_val{y_val_arr.shape}")
            if X_val_arr.shape[1] != L:
                raise ValueError("Training and validation must use the same system size L.")
            X_train_arr, y_train_arr = X_arr, y_arr
            has_val = True
        else:
            has_val = val_split > 0.0
            if has_val:
                if not (0.0 < val_split < 1.0):
                    raise ValueError("val_split must be in (0, 1).")
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(N)
                n_train = int((1.0 - val_split) * N)
                n_train = max(1, min(n_train, N - 1))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                X_train_arr, y_train_arr = X_arr[train_idx], y_arr[train_idx]
                X_val_arr, y_val_arr = X_arr[val_idx], y_arr[val_idx]
            else:
                X_train_arr, y_train_arr = X_arr, y_arr
                X_val_arr, y_val_arr = None, None
        X_train = torch.from_numpy(X_train_arr)
        y_train = torch.from_numpy(y_train_arr)
        gen = torch.Generator().manual_seed(self.seed)
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=gen,
        )
        if has_val:
            X_val = torch.from_numpy(X_val_arr)
            y_val = torch.from_numpy(y_val_arr)
            val_loader = DataLoader(
                TensorDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            val_loader = None
        self.history_ = {"train_loss": [], "train_mse": [], "val_mse": []}
        eps = 1e-6
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_count = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                if self.augment_z2:
                    xb = torch.cat([xb, -xb], dim=0)
                    yb = torch.cat([yb, -yb], dim=0)
                pred_m = self.model_(xb)
                pred_p = ((pred_m + 1.0) * 0.5).clamp(eps, 1.0 - eps)
                target_p = (yb + 1.0) * 0.5
                loss = F.binary_cross_entropy(pred_p, target_p)
                self.optimizer_.zero_grad()
                loss.backward()
                self.optimizer_.step()
                n_elem = yb.numel()
                train_loss_sum += loss.item() * n_elem
                train_mse_sum += F.mse_loss(pred_m, yb, reduction="sum").item()
                train_count += n_elem
            train_loss = train_loss_sum / train_count
            train_mse = train_mse_sum / train_count
            self.history_["train_loss"].append(train_loss)
            self.history_["train_mse"].append(train_mse)
            if val_loader is not None:
                self.model_.eval()
                val_mse_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_)
                        yb = yb.to(self.device_)
                        pred_m = self.model_(xb)
                        val_mse_sum += F.mse_loss(pred_m, yb, reduction="sum").item()
                        val_count += yb.numel()
                val_mse = val_mse_sum / val_count
            else:
                val_mse = None
            self.history_["val_mse"].append(val_mse)
            if self.verbose:
                if val_mse is None:
                    print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f}")
                else:
                    print(f"Epoch {epoch:03d}/{self.epochs} | Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(self.history_["train_loss"], label="train_loss")
            if any(v is not None for v in self.history_["val_mse"]):
                val_curve = [np.nan if v is None else v for v in self.history_["val_mse"]]
                plt.plot(val_curve, label="val_mse")
            plt.xlabel("epoch")
            plt.legend()
            plt.tight_layout()
            plt.show()
        if return_history:
            return self, self.history_
        return self
    def _predict_magnetization_array(self, X):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_arr = self._to_numpy_2d(X, "X")
        if X_arr.shape[1] != self.input_length_:
            raise ValueError(f"Expected input length {self.input_length_}, got {X_arr.shape[1]}")
        X_tensor = torch.from_numpy(X_arr)
        loader = DataLoader(X_tensor, batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device_)
                pm = self.model_(xb)
                preds.append(pm.cpu().numpy())
        return np.concatenate(preds, axis=0)
    def predict_magnetization(self, X):
        return self._predict_magnetization_array(X)
    def predict_proba(self, X):
        m = self._predict_magnetization_array(X)
        return np.clip((m + 1.0) * 0.5, 0.0, 1.0)
    def predict(self, X):
        m = self._predict_magnetization_array(X)
        return np.where(m >= 0.0, 1.0, -1.0).astype(np.float32)
    def score(self, X, y, metric="neg_mse"):
        y_arr = self._to_numpy_2d(y, "y")
        m = self._predict_magnetization_array(X)
        if y_arr.shape != m.shape:
            raise ValueError(f"Shape mismatch: predictions{m.shape} vs y{y_arr.shape}")
        mse = float(np.mean((m - y_arr) ** 2))
        if metric == "neg_mse":
            return -mse
        if metric == "mse":
            return mse
        if metric == "accuracy":
            y_pred = np.where(m >= 0.0, 1.0, -1.0)
            return float(np.mean(y_pred == y_arr))
        raise ValueError("metric must be one of {'neg_mse', 'mse', 'accuracy'}")
    def total_parameters(self):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return int(sum(p.numel() for p in self.model_.parameters()))
    def trainable_parameters(self):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return int(sum(p.numel() for p in self.model_.parameters() if p.requires_grad))
    def parameter_counts(self):
        return {
            "total": self.total_parameters(),
            "trainable": self.trainable_parameters(),
        }
    def save(self, path, extra=None):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        checkpoint = {
            "params": self.get_params(),
            "input_length": self.input_length_,
            "model_state_dict": self.model_.state_dict(),
            "history": self.history_,
            "extra": extra,
        }
        torch.save(checkpoint, path)
    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint["params"])
        model.device_ = model._resolve_device()
        model._build_model(checkpoint["input_length"])
        model.model_.load_state_dict(checkpoint["model_state_dict"])
        model.history_ = checkpoint.get("history")
        return model


class myLogisticRegressionModel:
    def __init__(
        self,
        solver="lbfgs",
        C=1.0,
        max_iter=400,
        fit_intercept=True,
        pbc_augment=False,
        pbc_max_shifts=None,
        pbc_shared=False,
        augment_z2=False,
        seed=0,
        verbose=1,
    ):
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.pbc_augment = pbc_augment
        self.pbc_max_shifts = pbc_max_shifts
        self.pbc_shared = pbc_shared
        self.augment_z2 = augment_z2
        self.seed = seed
        self.verbose = verbose
        self.models_ = None
        self.shared_model_ = None
        self.input_length_ = None
        self.history_ = None
    def get_params(self, deep=True):
        return {
            "solver": self.solver,
            "C": self.C,
            "max_iter": self.max_iter,
            "fit_intercept": self.fit_intercept,
            "pbc_augment": self.pbc_augment,
            "pbc_max_shifts": self.pbc_max_shifts,
            "pbc_shared": self.pbc_shared,
            "augment_z2": self.augment_z2,
            "seed": self.seed,
            "verbose": self.verbose,
        }
    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self
    def _to_numpy_2d(self, X, name="X"):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"{name} must have shape (N, L), got {X_arr.shape}")
        return X_arr
    def _shift_list(self, L, force_full=False):
        if force_full:
            if self.pbc_max_shifts is None or int(self.pbc_max_shifts) >= L:
                return list(range(L))
            m = int(self.pbc_max_shifts)
            if m < 1:
                raise ValueError("pbc_max_shifts must be >= 1 when provided")
            shifts = np.linspace(0, L - 1, num=m, dtype=int).tolist()
            return list(dict.fromkeys(shifts))
        if not self.pbc_augment:
            return [0]
        if self.pbc_max_shifts is None or int(self.pbc_max_shifts) >= L:
            return list(range(L))
        m = int(self.pbc_max_shifts)
        if m < 1:
            raise ValueError("pbc_max_shifts must be >= 1 when provided")
        shifts = np.linspace(0, L - 1, num=m, dtype=int).tolist()
        return list(dict.fromkeys(shifts))
    def _augment_with_shifts(self, X, y):
        L = X.shape[1]
        shifts = self._shift_list(L, force_full=False)
        if len(shifts) == 1 and shifts[0] == 0:
            return X, y
        X_aug = [np.roll(X, s, axis=1) for s in shifts]
        y_aug = [np.roll(y, s, axis=1) for s in shifts]
        return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)
    def _build_shared_dataset(self, X, y):
        L = X.shape[1]
        shifts = self._shift_list(L, force_full=True)
        X_list = []
        y_list = []
        for s in shifts:
            X_list.append(np.roll(X, -s, axis=1))
            y_list.append(y[:, s])
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
    def fit(self, X, y, val_data=None, val_split=0.0, return_history=False, plot=False):
        np.random.seed(self.seed)
        X_arr = self._to_numpy_2d(X, "X")
        y_arr = self._to_numpy_2d(y, "y")
        if X_arr.shape != y_arr.shape:
            raise ValueError(f"Shape mismatch: X{X_arr.shape} vs y{y_arr.shape}")
        N, L = X_arr.shape
        self.input_length_ = int(L)
        if val_data is not None:
            X_val_arr = self._to_numpy_2d(val_data[0], "X_val")
            y_val_arr = self._to_numpy_2d(val_data[1], "y_val")
            if X_val_arr.shape != y_val_arr.shape:
                raise ValueError(f"Validation shape mismatch: X_val{X_val_arr.shape} vs y_val{y_val_arr.shape}")
            if X_val_arr.shape[1] != L:
                raise ValueError("Training and validation must use the same system size L.")
            X_train_arr, y_train_arr = X_arr, y_arr
            has_val = True
        else:
            has_val = val_split > 0.0
            if has_val:
                if not (0.0 < val_split < 1.0):
                    raise ValueError("val_split must be in (0, 1).")
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(N)
                n_train = int((1.0 - val_split) * N)
                n_train = max(1, min(n_train, N - 1))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                X_train_arr, y_train_arr = X_arr[train_idx], y_arr[train_idx]
                X_val_arr, y_val_arr = X_arr[val_idx], y_arr[val_idx]
            else:
                X_train_arr, y_train_arr = X_arr, y_arr
                X_val_arr, y_val_arr = None, None

        if self.augment_z2:
            X_train_arr = np.concatenate([X_train_arr, -X_train_arr], axis=0)
            y_train_arr = np.concatenate([y_train_arr, -y_train_arr], axis=0)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if self.pbc_shared:
            X_shared, y_shared = self._build_shared_dataset(X_train_arr, y_train_arr)
            y_shared_bin = ((y_shared + 1.0) * 0.5).astype(np.int8)
            if y_shared_bin.min() == y_shared_bin.max():
                self.shared_model_ = float(y_shared_bin[0])
            else:
                clf = LogisticRegression(
                    solver=self.solver,
                    C=self.C,
                    max_iter=self.max_iter,
                    fit_intercept=self.fit_intercept,
                    random_state=self.seed,
                )
                clf.fit(X_shared, y_shared_bin)
                self.shared_model_ = clf
            self.models_ = None
        else:
            X_train_aug, y_train_aug = self._augment_with_shifts(X_train_arr, y_train_arr)
            y_train_bin = ((y_train_aug + 1.0) * 0.5).astype(np.int8)
            self.models_ = []
            self.shared_model_ = None
            for site in range(L):
                y_site = y_train_bin[:, site]
                if y_site.min() == y_site.max():
                    self.models_.append(float(y_site[0]))
                    continue
                clf = LogisticRegression(
                    solver=self.solver,
                    C=self.C,
                    max_iter=self.max_iter,
                    fit_intercept=self.fit_intercept,
                    random_state=self.seed,
                )
                clf.fit(X_train_aug, y_site)
                self.models_.append(clf)
        train_mse = self.score(X_train_arr, y_train_arr, metric="mse")
        val_mse = self.score(X_val_arr, y_val_arr, metric="mse") if has_val else None
        self.history_ = {
            "train_loss": [None],
            "train_mse": [train_mse],
            "val_mse": [val_mse],
        }
        if self.verbose:
            mode = "shared-PBC" if self.pbc_shared else "sitewise"
            if val_mse is None:
                print(f"LogReg({mode}) | Train MSE: {train_mse:.6f}")
            else:
                print(f"LogReg({mode}) | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        if plot and has_val:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 3))
            plt.bar(["train", "val"], [train_mse, val_mse])
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.show()
        if return_history:
            return self, self.history_
        return self
    def _predict_prob_plus(self, X):
        X_arr = self._to_numpy_2d(X, "X")
        if X_arr.shape[1] != self.input_length_:
            raise ValueError(f"Expected input length {self.input_length_}, got {X_arr.shape[1]}")
        N = X_arr.shape[0]
        L = self.input_length_
        p_plus = np.empty((N, L), dtype=np.float32)
        if self.pbc_shared:
            if self.shared_model_ is None:
                raise RuntimeError("Model is not fitted. Call fit() first.")
            for site in range(L):
                X_shift = np.roll(X_arr, -site, axis=1)
                if isinstance(self.shared_model_, float):
                    p_plus[:, site] = self.shared_model_
                else:
                    p_plus[:, site] = self.shared_model_.predict_proba(X_shift)[:, 1]
            return p_plus
        if self.models_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        for site, model_site in enumerate(self.models_):
            if isinstance(model_site, float):
                p_plus[:, site] = model_site
            else:
                p_plus[:, site] = model_site.predict_proba(X_arr)[:, 1]
        return p_plus
    def predict_proba(self, X):
        return self._predict_prob_plus(X)
    def predict_magnetization(self, X):
        return 2.0 * self._predict_prob_plus(X) - 1.0
    def predict(self, X):
        m = self.predict_magnetization(X)
        return np.where(m >= 0.0, 1.0, -1.0).astype(np.float32)
    def score(self, X, y, metric="neg_mse"):
        y_arr = self._to_numpy_2d(y, "y")
        m = self.predict_magnetization(X)
        if y_arr.shape != m.shape:
            raise ValueError(f"Shape mismatch: predictions{m.shape} vs y{y_arr.shape}")
        mse = float(np.mean((m - y_arr) ** 2))
        if metric == "neg_mse":
            return -mse
        if metric == "mse":
            return mse
        if metric == "accuracy":
            y_pred = np.where(m >= 0.0, 1.0, -1.0)
            return float(np.mean(y_pred == y_arr))
        raise ValueError("metric must be one of {'neg_mse', 'mse', 'accuracy'}")
    def total_parameters(self):
        if self.pbc_shared:
            if self.shared_model_ is None:
                raise RuntimeError("Model is not fitted. Call fit() first.")
            if isinstance(self.shared_model_, float):
                return 1
            return int(self.shared_model_.coef_.size + self.shared_model_.intercept_.size)
        if self.models_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        total = 0
        for m in self.models_:
            if isinstance(m, float):
                total += 1
            else:
                total += int(m.coef_.size + m.intercept_.size)
        return total
    def trainable_parameters(self):
        return self.total_parameters()
    def parameter_counts(self):
        return {
            "total": self.total_parameters(),
            "trainable": self.trainable_parameters(),
        }
    def save(self, path, extra=None):
        if self.models_ is None and self.shared_model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        checkpoint = {
            "params": self.get_params(),
            "input_length": self.input_length_,
            "models": self.models_,
            "shared_model": self.shared_model_,
            "history": self.history_,
            "extra": extra,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        model = cls(**checkpoint["params"])
        model.input_length_ = checkpoint["input_length"]
        model.models_ = checkpoint.get("models")
        model.shared_model_ = checkpoint.get("shared_model")
        model.history_ = checkpoint.get("history")
        return model

class myLinearRegressionModel:
    def __init__(
        self,
        fit_intercept=True,
        n_jobs=None,
        pbc_augment=False,
        pbc_max_shifts=None,
        pbc_shared=False,
        augment_z2=False,
        seed=0,
        verbose=1,
    ):
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.pbc_augment = pbc_augment
        self.pbc_max_shifts = pbc_max_shifts
        self.pbc_shared = pbc_shared
        self.augment_z2 = augment_z2
        self.seed = seed
        self.verbose = verbose
        self.model_ = None
        self.shared_model_ = None
        self.input_length_ = None
        self.history_ = None
    def get_params(self, deep=True):
        return {
            "fit_intercept": self.fit_intercept,
            "n_jobs": self.n_jobs,
            "pbc_augment": self.pbc_augment,
            "pbc_max_shifts": self.pbc_max_shifts,
            "pbc_shared": self.pbc_shared,
            "augment_z2": self.augment_z2,
            "seed": self.seed,
            "verbose": self.verbose,
        }
    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self
    def _to_numpy_2d(self, X, name="X"):
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"{name} must have shape (N, L), got {X_arr.shape}")
        return X_arr
    def _shift_list(self, L, force_full=False):
        if force_full:
            if self.pbc_max_shifts is None or int(self.pbc_max_shifts) >= L:
                return list(range(L))
            m = int(self.pbc_max_shifts)
            if m < 1:
                raise ValueError("pbc_max_shifts must be >= 1 when provided")
            shifts = np.linspace(0, L - 1, num=m, dtype=int).tolist()
            return list(dict.fromkeys(shifts))
        if not self.pbc_augment:
            return [0]
        if self.pbc_max_shifts is None or int(self.pbc_max_shifts) >= L:
            return list(range(L))
        m = int(self.pbc_max_shifts)
        if m < 1:
            raise ValueError("pbc_max_shifts must be >= 1 when provided")
        shifts = np.linspace(0, L - 1, num=m, dtype=int).tolist()
        return list(dict.fromkeys(shifts))
    def _augment_with_shifts(self, X, y):
        L = X.shape[1]
        shifts = self._shift_list(L, force_full=False)
        if len(shifts) == 1 and shifts[0] == 0:
            return X, y
        X_aug = [np.roll(X, s, axis=1) for s in shifts]
        y_aug = [np.roll(y, s, axis=1) for s in shifts]
        return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)
    def _build_shared_dataset(self, X, y):
        L = X.shape[1]
        shifts = self._shift_list(L, force_full=True)
        X_list = []
        y_list = []
        for s in shifts:
            X_list.append(np.roll(X, -s, axis=1))
            y_list.append(y[:, s])
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
    def fit(self, X, y, val_data=None, val_split=0.0, return_history=False, plot=False):
        np.random.seed(self.seed)
        X_arr = self._to_numpy_2d(X, "X")
        y_arr = self._to_numpy_2d(y, "y")
        if X_arr.shape != y_arr.shape:
            raise ValueError(f"Shape mismatch: X{X_arr.shape} vs y{y_arr.shape}")
        N, L = X_arr.shape
        self.input_length_ = int(L)
        if val_data is not None:
            X_val_arr = self._to_numpy_2d(val_data[0], "X_val")
            y_val_arr = self._to_numpy_2d(val_data[1], "y_val")
            if X_val_arr.shape != y_val_arr.shape:
                raise ValueError(f"Validation shape mismatch: X_val{X_val_arr.shape} vs y_val{y_val_arr.shape}")
            if X_val_arr.shape[1] != L:
                raise ValueError("Training and validation must use the same system size L.")
            X_train_arr, y_train_arr = X_arr, y_arr
            has_val = True
        else:
            has_val = val_split > 0.0
            if has_val:
                if not (0.0 < val_split < 1.0):
                    raise ValueError("val_split must be in (0, 1).")
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(N)
                n_train = int((1.0 - val_split) * N)
                n_train = max(1, min(n_train, N - 1))
                train_idx = perm[:n_train]
                val_idx = perm[n_train:]
                X_train_arr, y_train_arr = X_arr[train_idx], y_arr[train_idx]
                X_val_arr, y_val_arr = X_arr[val_idx], y_arr[val_idx]
            else:
                X_train_arr, y_train_arr = X_arr, y_arr
                X_val_arr, y_val_arr = None, None
        if self.augment_z2:
            X_train_arr = np.concatenate([X_train_arr, -X_train_arr], axis=0)
            y_train_arr = np.concatenate([y_train_arr, -y_train_arr], axis=0)
        if self.pbc_shared:
            X_shared, y_shared = self._build_shared_dataset(X_train_arr, y_train_arr)
            model = LinearRegression(
                fit_intercept=self.fit_intercept,
                n_jobs=self.n_jobs,
            )
            model.fit(X_shared, y_shared)
            self.shared_model_ = model
            self.model_ = None
        else:
            X_train_aug, y_train_aug = self._augment_with_shifts(X_train_arr, y_train_arr)
            model = LinearRegression(
                fit_intercept=self.fit_intercept,
                n_jobs=self.n_jobs,
            )
            model.fit(X_train_aug, y_train_aug)
            self.model_ = model
            self.shared_model_ = None
        train_mse = self.score(X_train_arr, y_train_arr, metric="mse")
        val_mse = self.score(X_val_arr, y_val_arr, metric="mse") if has_val else None
        self.history_ = {
            "train_loss": [None],
            "train_mse": [train_mse],
            "val_mse": [val_mse],
        }
        if self.verbose:
            mode = "shared-PBC" if self.pbc_shared else "sitewise"
            if val_mse is None:
                print(f"LinReg({mode}) | Train MSE: {train_mse:.6f}")
            else:
                print(f"LinReg({mode}) | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        if plot and has_val:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 3))
            plt.bar(["train", "val"], [train_mse, val_mse])
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.show()
        if return_history:
            return self, self.history_
        return self
    def predict_magnetization(self, X):
        X_arr = self._to_numpy_2d(X, "X")
        if X_arr.shape[1] != self.input_length_:
            raise ValueError(f"Expected input length {self.input_length_}, got {X_arr.shape[1]}")
        L = self.input_length_
        if self.pbc_shared:
            if self.shared_model_ is None:
                raise RuntimeError("Model is not fitted. Call fit() first.")
            pred = np.empty((X_arr.shape[0], L), dtype=np.float64)
            for site in range(L):
                X_shift = np.roll(X_arr, -site, axis=1)
                pred[:, site] = self.shared_model_.predict(X_shift)
            return np.clip(pred, -1.0, 1.0)
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        pred = self.model_.predict(X_arr)
        return np.clip(pred, -1.0, 1.0)
    def predict_proba(self, X):
        m = self.predict_magnetization(X)
        return np.clip((m + 1.0) * 0.5, 0.0, 1.0)
    def predict(self, X):
        m = self.predict_magnetization(X)
        return np.where(m >= 0.0, 1.0, -1.0).astype(np.float32)
    def score(self, X, y, metric="neg_mse"):
        y_arr = self._to_numpy_2d(y, "y")
        m = self.predict_magnetization(X)
        if y_arr.shape != m.shape:
            raise ValueError(f"Shape mismatch: predictions{m.shape} vs y{y_arr.shape}")
        mse = float(np.mean((m - y_arr) ** 2))
        if metric == "neg_mse":
            return -mse
        if metric == "mse":
            return mse
        if metric == "accuracy":
            y_pred = np.where(m >= 0.0, 1.0, -1.0)
            return float(np.mean(y_pred == y_arr))
        raise ValueError("metric must be one of {'neg_mse', 'mse', 'accuracy'}")
    def total_parameters(self):
        if self.pbc_shared:
            if self.shared_model_ is None:
                raise RuntimeError("Model is not fitted. Call fit() first.")
            coef = np.asarray(self.shared_model_.coef_)
            intercept = np.asarray(self.shared_model_.intercept_)
            return int(coef.size + intercept.size)
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        coef = np.asarray(self.model_.coef_)
        intercept = np.asarray(self.model_.intercept_)
        return int(coef.size + intercept.size)
    def trainable_parameters(self):
        return self.total_parameters()
    def parameter_counts(self):
        return {
            "total": self.total_parameters(),
            "trainable": self.trainable_parameters(),
        }
    def save(self, path, extra=None):
        if self.model_ is None and self.shared_model_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        checkpoint = {
            "params": self.get_params(),
            "input_length": self.input_length_,
            "model": self.model_,
            "shared_model": self.shared_model_,
            "history": self.history_,
            "extra": extra,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        model = cls(**checkpoint["params"])
        model.input_length_ = checkpoint["input_length"]
        model.model_ = checkpoint.get("model")
        model.shared_model_ = checkpoint.get("shared_model")
        model.history_ = checkpoint.get("history")
        return model
