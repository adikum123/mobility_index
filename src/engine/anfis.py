import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anfis_toolbox import ANFISRegressor
from anfis_toolbox.logging_config import enable_training_logs
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..database.model_registry import ModelRegistry
from ..database.test_run_registry import TestRunRegistry
from ..database.train_run_registry import TrainRunRegistry
from ..engine.matrices_processor import (
    DISTANCE_MATRIX_FILENAME,
    MatricesProcessor,
    interval_journey_counts_matrix_filename,
    interval_time_matrix_filename,
)

INDEX4_ARRAY_FILENAME = "index4_array.npz"


def _log_training_history_table(history: dict) -> None:
    """Print epoch train losses to stdout (always visible; does not rely on logger setup)."""
    train = history.get("train") or []
    print("\n--- Per-epoch training loss (train split only) ---")
    for i, tl in enumerate(train):
        print(f"  epoch {i + 1:>4}/{len(train)}  train_loss={tl:.8f}")
    if train:
        print(f"  summary: train_loss first={train[0]:.8f}  last={train[-1]:.8f}\n")


class ANFIS:

    def __init__(
        self,
        num_indices: int,
        num_epochs: int,
        learning_rate: float,
        membership_functions: str,
        time_interval: int,
        loss_function: str,
        batch_size: int | None = 256,
        *,
        optimizer: str = "hybrid",
        optimizer_params: dict | None = None,
        shuffle: bool = True,
        index4_mode: str = None,
        num_experts: int | None = None,
        n_mfs: int = 3,
    ):
        assert num_indices in [3, 4], "Number of indices must be 3 or 4"
        assert isinstance(time_interval, int) and time_interval in range(
            0, 8
        ), "Time interval must be an integer and in the range of 0 to 7"
        self.num_indices = num_indices
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.membership_functions = membership_functions
        self.time_interval = time_interval
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.shuffle = shuffle
        self.model = ANFISRegressor(
            n_mfs=n_mfs,
            mf_type=membership_functions,
            optimizer=optimizer,
            loss=loss_function,
            learning_rate=learning_rate,
            batch_size=batch_size,
            shuffle=shuffle,
            epochs=num_epochs,
            verbose=True,
        )
        self.base_dir = Path(__file__).parents[2]
        self.data_path = self.base_dir / "data" / "output"
        self.plots_path = self.base_dir / "plots"
        self.models_dir = self.base_dir / "models"
        self.num_experts = num_experts

        # discretize normalized [0,1] values into low/mid/high bins
        self.journey_count_cats = ("Mali", "Srednji", "Veliki")
        self.duration_cats = ("Kratko", "Srednje", "Dugo")
        self.distance_cats = ("Kratka", "Srednja", "Velika")
        self.index4_cats = ("Niska", "Umjerena", "Visoka")

        # map column names to English snake_case
        self.column_map = {
            "Broj putovanja": "journey_count_category",
            "Trajanje": "duration_category",
            "Udaljenost": "distance_category",
            "Ocjena": "rating",
        }
        if self.num_indices == 4:
            self.column_map["Dostupnost"] = "availability_category"

        _VALID_INDEX4_MODES = ("average", "destination")
        assert num_indices == 3 or (
            num_indices == 4 and index4_mode in _VALID_INDEX4_MODES
        ), f"If 4 indices are used, index4_mode must be one of {_VALID_INDEX4_MODES}"
        self.index4_mode = index4_mode

        self.model_id = f"anfis_i{num_indices}_t{time_interval}"
        self.config = {
            "num_indices": num_indices,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "membership_functions": membership_functions,
            "time_interval": time_interval,
            "loss_function": loss_function,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "optimizer_params": optimizer_params,
            "shuffle": shuffle,
            "index4_mode": index4_mode,
        }
        model_path = str(
            self.models_dir / f"anfis_model_time_interval_{time_interval}.pkl"
        )
        self.model_registry = ModelRegistry()
        self.train_run_registry = TrainRunRegistry()
        self.test_run_registry = TestRunRegistry()
        self.model_path = model_path
        self.model_registry.register_model(
            model_id=self.model_id,
            path=model_path,
            config=self.config,
        )

    @staticmethod
    def _discretize(value: float) -> int:
        if value < 1 / 3:
            return 0
        if value < 2 / 3:
            return 1
        return 2

    def _expand_index4(self, values: np.ndarray) -> np.ndarray:
        """Expand a 1-D per-station array into an n x n matrix.

        The expansion strategy is controlled by ``self.index4_mode``:
        - "average":     matrix[i,j] = (v[i] + v[j]) / 2
        - "destination": matrix[i,j] = v[j]
        """
        v = values
        if self.index4_mode == "average":
            return (v[:, None] + v[None, :]) / 2
        if self.index4_mode == "destination":
            return np.broadcast_to(v[None, :], (len(v), len(v))).copy()
        raise ValueError(f"Unknown index4_mode: {self.index4_mode!r}")

    def _get_X(self) -> np.ndarray:
        distance_matrix_path = (
            self.data_path / "distance_matrices" / DISTANCE_MATRIX_FILENAME
        )
        dm, ref_switch_ids = MatricesProcessor.load_matrix_npz(distance_matrix_path)
        distance_matrix_flat = dm.flatten()

        journey_count_matrix_path = (
            self.data_path
            / "journey_counts_matrices"
            / interval_journey_counts_matrix_filename(self.time_interval)
        )
        jm, j_sids = MatricesProcessor.load_matrix_npz(journey_count_matrix_path)
        if not np.array_equal(ref_switch_ids, j_sids):
            raise ValueError(
                "switch_ids in journey count matrix do not match distance matrix"
            )
        journey_count_matrix_flat = jm.flatten()

        time_matrix_path = (
            self.data_path
            / "time_matrices"
            / interval_time_matrix_filename(self.time_interval)
        )
        tm, t_sids = MatricesProcessor.load_matrix_npz(time_matrix_path)
        if not np.array_equal(ref_switch_ids, t_sids):
            raise ValueError("switch_ids in time matrix do not match distance matrix")
        time_matrix_flat = tm.flatten()

        if self.num_indices == 3:
            # stack flattened matrices as columns
            X = np.column_stack(
                (
                    # add 1e-6 for numerical stability
                    journey_count_matrix_flat + 1e-6,
                    time_matrix_flat + 1e-6,
                    distance_matrix_flat + 1e-6,
                )
            ).astype(float)
            return X

        index4_array_path = self.data_path / "index4" / INDEX4_ARRAY_FILENAME
        with np.load(index4_array_path) as idx4:
            index4_sids = np.asarray(idx4["switch_ids"], dtype=np.int64)
            index4_values = np.asarray(idx4["scores"], dtype=np.float64)
        if not np.array_equal(ref_switch_ids, index4_sids):
            raise ValueError("switch_ids in index4 array do not match distance matrix")
        index4_matrix = self._expand_index4(index4_values).flatten()

        # stack flattened matrices as columns
        X = np.column_stack(
            (
                # add 1e-6 for numerical stability
                journey_count_matrix_flat + 1e-6,
                time_matrix_flat + 1e-6,
                distance_matrix_flat + 1e-6,
                index4_matrix + 1e-6,
            )
        )
        return X

    def _get_lookup(self) -> list[dict[tuple[str, ...], float]]:
        """Build one lookup dict per expert sheet.

        Returns a list of dicts, each mapping category tuples to normalized
        ratings from that expert.
        """
        if self.num_indices == 3:
            mapper_path = (
                Path(__file__).parents[2]
                / "data"
                / "mappers"
                / "anketa_3_indikatora.xlsx"
            )
            key_cols = (
                "journey_count_category",
                "duration_category",
                "distance_category",
            )
        else:
            mapper_path = (
                Path(__file__).parents[2]
                / "data"
                / "mappers"
                / "anketa_4_indikatora.xlsx"
            )
            key_cols = (
                "journey_count_category",
                "duration_category",
                "distance_category",
                "availability_category",
            )

        all_sheets = pd.read_excel(mapper_path, sheet_name=None)
        rating_min = 1
        rating_max = 6

        lookups: list[dict[tuple[str, ...], float]] = []
        for idx, mapper_df in enumerate(all_sheets.values()):
            if self.num_experts is not None and idx >= self.num_experts:
                break
            mapper_df = mapper_df.drop(columns=["Br"], errors="ignore")
            mapper_df = mapper_df.rename(columns=self.column_map)
            lookup = {}
            for _, row in mapper_df.iterrows():
                key = tuple(str(row[c]).strip() for c in key_cols)
                r = float(row["rating"])
                lookup[key] = (r - rating_min) / (rating_max - rating_min)
            lookups.append(lookup)

        return lookups

    def _get_Y(self, X: np.ndarray, lookup: dict[tuple[str, ...], float]) -> np.ndarray:
        if self.num_indices == 3:
            return np.array(
                [
                    lookup[
                        (
                            self.journey_count_cats[ANFIS._discretize(X[i, 0])],
                            self.duration_cats[ANFIS._discretize(X[i, 1])],
                            self.distance_cats[ANFIS._discretize(X[i, 2])],
                        )
                    ]
                    for i in range(len(X))
                ],
                dtype=float,
            )
        return np.array(
            [
                lookup[
                    (
                        self.journey_count_cats[ANFIS._discretize(X[i, 0])],
                        self.duration_cats[ANFIS._discretize(X[i, 1])],
                        self.distance_cats[ANFIS._discretize(X[i, 2])],
                        self.index4_cats[ANFIS._discretize(X[i, 3])],
                    )
                ]
                for i in range(len(X))
            ],
            dtype=float,
        )

    def set_data(self) -> None:
        X_base = self._get_X()
        lookups = self._get_lookup()
        num_experts = len(lookups) if self.num_experts is None else self.num_experts
        print(f"Loading ratings from {num_experts} experts (data augmentation)")

        self.X = np.tile(X_base, (num_experts, 1))
        self.Y = np.concatenate([self._get_Y(X_base, lookup) for lookup in lookups])

    def set_train_val_data(self, random_state: int = 42):
        """Split 80% train / 20% held-out. Only the train split is used in ``fit``;
        the held-out split is for metrics in ``test()`` and is not used during training.
        """
        X_train, X_val, Y_train, Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=random_state
        )
        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val

    def train(self) -> None:
        start_time = time.time()

        # data
        print("Preparing data...")
        self.set_data()
        print(f"Data shape: X={self.X.shape}, Y={self.Y.shape}")

        # split data into training and validation sets
        self.set_train_val_data()
        print(
            f"Training samples: {len(self.X_train)}, Validation samples: {len(self.X_val)}"
        )

        yt, yv = self.Y_train, self.Y_val
        print(
            "\n--- Target (Y) stats ---\n"
            f"  train (used for fit): min={yt.min():.6f} max={yt.max():.6f} "
            f"mean={yt.mean():.6f} std={yt.std():.6f}\n"
            f"  held-out (for evaluation in test() only): min={yv.min():.6f} max={yv.max():.6f} "
            f"mean={yv.mean():.6f} std={yv.std():.6f}\n"
        )

        # So toolbox + hybrid trainer INFO lines appear (nested loggers propagate to root)
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            force=True,
        )
        enable_training_logs()

        # Train on train split only; held-out data is not passed to fit (see test()).
        print("Starting ANFIS training...")
        self.model.fit(self.X_train, self.Y_train, verbose=True)
        print("Training completed successfully")

        end_time = time.time()
        print(
            f"Training time: {end_time - start_time} seconds for {self.num_epochs} epochs"
        )

        history = self.model.training_history_
        _log_training_history_table(history)

        # save model file
        self.model_registry.save_model_file(
            model=self.model,
            path=self.model_path,
        )

        # extract plots from the training history
        train_loss = history["train"]
        print("Plotting training progress...")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            range(1, len(train_loss) + 1),
            train_loss,
            label="Training loss (train split)",
        )
        ax.set_title(
            f"ANFIS Training Loss Over Epochs (time interval {self.time_interval})"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        fig.savefig(
            self.plots_path / f"training_loss_time_interval_{self.time_interval}.png"
        )
        plt.close(fig)

        train_duration = end_time - start_time
        train_config = {
            **self.config,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val),
            "data_shape_X": list(self.X.shape),
            "data_shape_Y": list(self.Y.shape),
            "num_experts": self.num_experts,
        }
        train_metrics = {
            "final_loss": train_loss[-1] if train_loss else None,
            "loss_history": train_loss,
            "train_duration_seconds": train_duration,
        }
        self.train_run_registry.save_train_run(
            model_id=self.model_id,
            train_config=train_config,
            metrics=train_metrics,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def test(self):
        """Evaluate on the held-out split (not used during ``fit``)."""
        metrics = {}

        # evaluate metrics
        Y_pred = self.predict(self.X_val)
        mse = mean_squared_error(self.Y_val, Y_pred)
        r2 = r2_score(self.Y_val, Y_pred)

        # print metrics
        print("Test Set Evaluation:")
        print(f"R²: {r2:.8f} for time interval {self.time_interval}")
        print(f"MSE: {mse:.8f} for time interval {self.time_interval}")
        print(f"RMSE: {np.sqrt(mse):.8f} for time interval {self.time_interval}")

        # plot predictions vs true ratings
        rng = np.random.default_rng(42 + self.time_interval)
        unique_y = np.unique(self.Y_val)
        idx_list = []
        for y_val in unique_y:
            mask = self.Y_val == y_val
            indices = np.where(mask)[0]
            n_sample = min(10, len(indices))
            chosen = rng.choice(indices, size=n_sample, replace=False)
            idx_list.extend(chosen)
        idx = np.array(idx_list)
        idx = idx[np.argsort(self.Y_val[idx])]  # sort by y_true for clearer plot
        y_test_rand = self.Y_val[idx]
        y_pred_rand = Y_pred[idx]
        x_samples = np.arange(len(idx))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_samples, y_test_rand, "o-", label="y_true", markersize=4, alpha=0.8)
        ax.plot(x_samples, y_pred_rand, "s-", label="y_pred", markersize=4, alpha=0.8)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Rating")
        ax.set_title(f"ANFIS: y_true vs y_pred (time interval {self.time_interval})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            self.plots_path
            / f"predictions_vs_true_ratings_time_interval_{self.time_interval}.png"
        )
        plt.close(fig)

        # save test run metrics
        metrics = {
            "r2": r2,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
        self.test_run_registry.save_test_run(
            model_id=self.model_id,
            metrics=metrics,
        )
        return metrics
