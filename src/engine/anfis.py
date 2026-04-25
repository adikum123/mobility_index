import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anfis_toolbox import ANFISRegressor
from anfis_toolbox.logging_config import enable_training_logs
from sklearn.metrics import mean_squared_error, r2_score

from ..database.model_registry import ModelRegistry
from ..database.test_run_registry import TestRunRegistry
from ..database.train_run_registry import TrainRunRegistry
from ..engine.matrices_processor import (
    DISTANCE_MATRIX_FILENAME,
    MatricesProcessor,
    interval_journey_counts_matrix_filename,
    interval_time_matrix_filename,
)
from .plotter import plot_anfis_training_loss

INDEX4_ARRAY_FILENAME = "index4_array.npz"


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
        num_train_experts: int,
        num_val_experts: int,
        num_test_experts: int,
        optimizer: str = "hybrid",
        optimizer_params: dict | None = None,
        shuffle: bool = True,
        index4_mode: str = None,
        n_mfs: int = 3,
    ):
        _init_params = {k: v for k, v in locals().items() if k != "self"}
        assert num_indices in [3, 4], "Number of indices must be 3 or 4"
        assert isinstance(time_interval, int) and time_interval in range(
            0, 8
        ), "Time interval must be an integer and in the range of 0 to 7"
        for name, n in (
            ("num_train_experts", num_train_experts),
            ("num_val_experts", num_val_experts),
            ("num_test_experts", num_test_experts),
        ):
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"{name} must be a positive int")
        self.num_train_experts = num_train_experts
        self.num_val_experts = num_val_experts
        self.num_test_experts = num_test_experts
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
        self.config = dict(_init_params)
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

    def _mapper_path_and_key_cols(
        self,
    ) -> tuple[Path, tuple[str, ...]]:
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
        return mapper_path, key_cols

    def _build_lookups_from_excel_sheets(
        self,
        all_sheets: dict,
        n: int,
        key_cols: tuple[str, ...],
    ) -> list[dict[tuple[str, ...], float]]:
        """Build ``n`` lookup dicts from the first ``n`` worksheet tabs in file order."""
        if n > len(all_sheets):
            raise ValueError(
                f"Need {n} expert sheet(s) in order; file has {len(all_sheets)}"
            )
        rating_min = 1
        rating_max = 6
        lookups: list[dict[tuple[str, ...], float]] = []
        for idx, mapper_df in enumerate(all_sheets.values()):
            if idx >= n:
                break
            mapper_df = mapper_df.drop(columns=["Br"], errors="ignore")
            mapper_df = mapper_df.rename(columns=self.column_map)
            lookup: dict[tuple[str, ...], float] = {}
            for _, row in mapper_df.iterrows():
                key = tuple(str(row[c]).strip() for c in key_cols)
                r = float(row["rating"])
                lookup[key] = (r - rating_min) / (rating_max - rating_min)
            lookups.append(lookup)
        return lookups

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

    def _stack_xy_for_lookups(
        self, X_base: np.ndarray, lookups: list[dict[tuple[str, ...], float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not lookups:
            return (
                np.empty((0, X_base.shape[1]), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        return (
            np.tile(X_base, (len(lookups), 1)),
            np.concatenate([self._get_Y(X_base, lu) for lu in lookups]),
        )

    def _set_data_expert_split(self) -> None:
        """First ``n_train`` sheets → train, next ``n_val`` → val, then ``n_test`` → test."""
        nt, nv, nte = (
            self.num_train_experts,
            self.num_val_experts,
            self.num_test_experts,
        )
        n_total = nt + nv + nte
        X_base = self._get_X()
        mp, kc = self._mapper_path_and_key_cols()
        sheets = pd.read_excel(mp, sheet_name=None)
        all_lu = self._build_lookups_from_excel_sheets(sheets, n_total, kc)
        l_tr = all_lu[0:nt]
        l_va = all_lu[nt : nt + nv]
        l_te = all_lu[nt + nv : nt + nv + nte]
        self.X_train, self.Y_train = self._stack_xy_for_lookups(X_base, l_tr)
        self.X_val, self.Y_val = self._stack_xy_for_lookups(X_base, l_va)
        self.X_test, self.Y_test = self._stack_xy_for_lookups(X_base, l_te)
        names = list(sheets.keys())[:n_total]
        tr_nm, va_nm, te_nm = names[:nt], names[nt : nt + nv], names[nt + nv : n_total]
        print(
            f"Expert split (workbook order): {nt} train → {len(self.X_train)} samples "
            f"({tr_nm!r}), {nv} val → {len(self.X_val)} ({va_nm!r}), "
            f"{nte} test → {len(self.X_test)} ({te_nm!r})"
        )

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

    def train(self) -> None:
        # start
        start_time = time.time()

        # load data and print stats
        print("Preparing data...")
        self._set_data_expert_split()
        print(
            f"train: {len(self.X_train)} samples  |  val: {len(self.X_val)}  |  test: {len(self.X_test)}"
        )
        yt, yv, yte = self.Y_train, self.Y_val, self.Y_test
        print(
            "\n--- Target (Y) stats ---\n"
            f"  train: min={yt.min():.6f} max={yt.max():.6f} mean={yt.mean():.6f} std={yt.std():.6f}\n"
            f"  val:   min={yv.min():.6f} max={yv.max():.6f} mean={yv.mean():.6f} std={yv.std():.6f}\n"
            f"  test:  min={yte.min():.6f} max={yte.max():.6f} mean={yte.mean():.6f} std={yte.std():.6f}  "
        )

        # set logging level
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            force=True,
        )
        enable_training_logs()

        # train and save model
        self.model.fit(
            self.X_train,
            self.Y_train,
            validation_data=(self.X_val, self.Y_val),
            validation_frequency=1,
            verbose=True,
        )
        history = self.model.training_history_
        self.model_registry.save_model_file(
            model=self.model,
            path=self.model_path,
        )

        # print training completion
        print("Training completed successfully")
        end_time = time.time()
        print(
            f"Wall time: {end_time - start_time:.1f} s; configured epochs: {self.num_epochs}"
        )

        # plot training progress
        train_loss = list(history["train"])
        val_losses = list(history["val"])
        print("Plotting training progress...")
        plot_anfis_training_loss(
            train_loss,
            val_losses,
            time_interval=self.time_interval,
            output_path=self.plots_path
            / f"training_loss_time_interval_{self.time_interval}.png",
            val_legend_label="Val loss (validation expert group)",
        )

        # save to database
        train_duration = end_time - start_time
        train_metrics = {
            "final_loss": float(val_losses[-1]),
            "loss_history": train_loss,
            "val_loss_history": val_losses,
            "train_duration_seconds": train_duration,
        }
        self.train_run_registry.save_train_run(
            model_id=self.model_id,
            train_config=self.config,
            metrics=train_metrics,
        )

    def test(self) -> dict:
        """Evaluate on test set"""
        x_ev, y_ev = self.X_test, self.Y_test
        y_pred = self.model.predict(x_ev)
        mse = mean_squared_error(y_ev, y_pred)
        r2 = r2_score(y_ev, y_pred)

        print(f"R²:   {r2:.8f}")
        print(f"MSE:  {mse:.8f}")
        print(f"RMSE: {np.sqrt(mse):.8f}")

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
