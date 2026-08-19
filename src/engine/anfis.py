import logging
import time
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from anfis_toolbox import ANFISRegressor
from anfis_toolbox.logging_config import enable_training_logs
from sklearn.metrics import mean_squared_error, r2_score

from ..database.model_registry import ModelRegistry
from ..database.test_run_registry import TestRunRegistry
from ..database.train_run_registry import TrainRunRegistry
from .plotter import plot_anfis_training_loss


class ANFIS:

    def __init__(
        self,
        num_indices: int,
        num_epochs: int,
        learning_rate: float,
        membership_functions: str,
        loss_function: str,
        sheet_dir: str,
        batch_size: int | None = 256,
        overlap: float = 0.5,
        margin: float = 0.1, 
        *,
        optimizer: str = "hybrid",
        shuffle: bool = True,
        n_mfs: int = 3,
    ):
        _init_params = {k: v for k, v in list(locals().items()) if k != "self"}
        assert num_indices in (3, 4), "Number of indices must be 3 or 4"
        self.num_indices = num_indices
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.membership_functions = membership_functions
        self.optimizer = optimizer
        self.shuffle = shuffle
        self.rating_max = 6
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
            overlap=overlap,
            margin=margin,
            random_state=random.randint(0, 100)
        )
        project_root = Path(__file__).parents[2]
        self.plots_path = project_root / "plots"
        self.models_dir = project_root / "models"

        self.column_map = {
            "Broj putovanja": "journey_count_category",
            "Trajanje": "duration_category",
            "Udaljenost": "distance_category",
            "Ocjena": "rating",
        }
        if self.num_indices == 4:
            self.column_map["Dostupnost"] = "availability_category"

        self.model_id = uuid.uuid4().hex
        self.config = dict(_init_params)
        self.model_path = self.models_dir / f"{self.model_id}.pkl"
        self.model_registry = ModelRegistry()
        self.train_run_registry = TrainRunRegistry()
        self.test_run_registry = TestRunRegistry()
        self.train_run_id: str | None = None
        self.model_registry.register_model(
            model_id=self.model_id,
            path=str(self.model_path),
            config=self.config,
        )
        self.sheet_dir = sheet_dir

    # Text labels used in the survey Excel for each indicator column.
    # Each label maps to a normalised [0, 1] value: low=0.0, mid=0.5, high=1.0.
    # Rating (Ocjena) is an integer 1–6 and is normalised separately as (r-1)/(max-1).
    _INDICATOR_CODES: dict[str, float] = {
        "Mali": 0.0,
        "Srednji": 0.5,
        "Veliki": 1.0,
        "Kratko": 0.0,
        "Srednje": 0.5,
        "Dugo": 1.0,
        "Kratka": 0.0,
        "Srednja": 0.5,
        "Velika": 1.0,
        "Niska": 0.0,
        "Umjerena": 0.5,
        "Visoka": 1.0,
    }

    def _survey_path(self) -> Path:
        filename = (
            "anketa_3_indikatora_27_kombinacija.xlsx"
            if self.num_indices == 3
            else "anketa_4_indikatora_81_kombinacija.xlsx"
        )
        return Path(__file__).parents[2] / "data" / "mappers" / self.sheet_dir / filename

    def _load_survey_pairs(
        self, sheet_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the requested survey sheets and return (X, Y) pairs."""
        input_cols = (
            ["journey_count_category", "duration_category", "distance_category"]
            if self.num_indices == 3
            else [
                "journey_count_category",
                "duration_category",
                "distance_category",
                "availability_category",
            ]
        )

        all_sheets: dict = pd.read_excel(self._survey_path(), sheet_name=None)

        rows_X: list[list[float]] = []
        rows_Y: list[float] = []

        for name in sheet_names:
            if name not in all_sheets:
                raise ValueError(
                    f"Sheet '{name}' not found in survey file. "
                    f"Available sheets: {list(all_sheets.keys())}"
                )
            df = all_sheets[name]
            df = df.drop(columns=["Br"], errors="ignore")
            df = df.rename(columns=self.column_map)

            for _, row in df.iterrows():
                x_row = [
                    self._INDICATOR_CODES[str(row[col]).strip()] for col in input_cols
                ]
                y_val = (float(row["rating"]) - 1) / (self.rating_max - 1)
                rows_X.append(x_row)
                rows_Y.append(y_val)

        return np.array(rows_X, dtype=np.float64), np.array(rows_Y, dtype=np.float64)

    def _set_data_expert_split(self) -> None:
        """Split survey sheets into train / val / test (70% / 15% / 15%)."""
        all_sheets: dict = pd.read_excel(self._survey_path(), sheet_name=None)
        sheet_names = list(all_sheets.keys())
        n = len(sheet_names)
        i_train = int(0.7 * n)
        i_val = i_train + int(0.15 * n)
        tr_names = sheet_names[:i_train]
        va_names = sheet_names[i_train:i_val]
        te_names = sheet_names[i_val:]
        self.train_sheet_names = tr_names
        self.val_sheet_names = va_names
        self.test_sheet_names = te_names

        self.X_train, self.Y_train = self._load_survey_pairs(tr_names)
        self.X_val, self.Y_val = self._load_survey_pairs(va_names)
        self.X_test, self.Y_test = self._load_survey_pairs(te_names)

        nt, nv, nte = len(tr_names), len(va_names), len(te_names)
        print(
            f"Survey split (70/15/15): {nt} train experts → {len(self.X_train)} samples "
            f"({tr_names!r}),  {nv} val → {len(self.X_val)} ({va_names!r}),  "
            f"{nte} test → {len(self.X_test)} ({te_names!r})"
        )

    def train(self) -> None:
        start_time = time.time()

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

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            force=True,
        )
        enable_training_logs()

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

        print("Training completed successfully")
        end_time = time.time()
        print(
            f"Wall time: {end_time - start_time:.1f} s; configured epochs: {self.num_epochs}"
        )

        train_loss = list(history["train"])
        val_losses = list(history["val"])
        print("Plotting training progress...")
        plot_anfis_training_loss(
            train_loss,
            val_losses,
            title=f"ANFIS loss ({self.num_indices} indicators)",
            output_path=self.plots_path / f"{self.train_run_id or self.model_id}.png",
            val_legend_label="Val loss",
        )

        train_duration = end_time - start_time
        train_metrics = {
            "final_loss": float(val_losses[-1]),
            "loss_history": train_loss,
            "val_loss_history": val_losses,
            "train_duration_seconds": train_duration,
        }
        self.train_run_id = self.train_run_registry.save_train_run(
            model_id=self.model_id,
            train_config=self.config,
            metrics=train_metrics,
        )

    def test(self) -> dict:
        """Evaluate on test set and persist metrics to test_run."""
        x_ev, y_ev = self.X_test, self.Y_test
        y_pred = self.model.predict(x_ev)
        mse = mean_squared_error(y_ev, y_pred)
        r2 = r2_score(y_ev, y_pred)

        print(f"R²:   {r2:.8f}")
        print(f"MSE:  {mse:.8f}")
        print(f"RMSE: {np.sqrt(mse):.8f}")

        metrics = {
            "r2": r2,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
        }
        if self.train_run_id is None:
            raise RuntimeError("No train run id; call train() before test().")
        self.test_run_registry.save_test_run(
            run_id=self.train_run_id,
            model_id=self.model_id,
            metrics=metrics,
        )
        return metrics
