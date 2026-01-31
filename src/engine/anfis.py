from pathlib import Path

import anfis_toolbox as atb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class ANFIS:

    def __init__(
        self,
        num_indices: int,
        num_epochs: int,
        learning_rate: float,
        membership_functions: str,
        optimizer: str,
        time_interval: int,
    ):
        assert num_indices in [3, 4], "Number of indices must be 3 or 4"
        assert isinstance(time_interval, int) and time_interval in range(
            0, 8
        ), "Time interval must be an integer and in the range of 0 to 7"
        self.num_indices = num_indices
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.membership_functions = membership_functions
        self.optimizer = optimizer
        self.time_interval = time_interval
        self.model = atb.ANFISRegressor(
            n_mfs=3,
            epochs=num_epochs,
            learning_rate=learning_rate,
            mf_type=membership_functions,
            optimizer=optimizer,
        )
        self.base_dir = Path(__file__).parents[2]
        self.data_path = self.base_dir / "data" / "output"
        self.plots_path = self.base_dir / "plots"
        self.models_dir = self.base_dir / "models"

        # discretize normalized [0,1] values into low/mid/high bins
        self.journey_count_cats = ("Mali", "Srednji", "Veliki")
        self.duration_cats = ("Kratko", "Srednje", "Dugo")
        self.distance_cats = ("Kratka", "Srednja", "Velika")

        # map column names to English snake_case
        self.column_map = {
            "Broj putovanja": "journey_count_category",
            "Trajanje": "duration_category",
            "Udaljenost": "distance_category",
            "Ocjena": "rating",
        }

    @staticmethod
    def _discretize(value: float) -> int:
        if value < 1 / 3:
            return 0
        if value < 2 / 3:
            return 1
        return 2

    def _get_X(self) -> np.ndarray:
        if self.num_indices == 3:
            # load distance matrix
            distance_matrix_path = (
                self.data_path / "distance_matrices" / "distance_matrix.xlsx"
            )
            df = pd.read_excel(distance_matrix_path, index_col=0)
            distance_matrix_flat = df.to_numpy().flatten()

            # load journey count matrix
            journey_count_matrix_path = (
                self.data_path
                / "journey_counts_matrices"
                / f"interval_{self.time_interval}_journey_counts_matrix.xlsx"
            )
            df = pd.read_excel(journey_count_matrix_path, index_col=0)
            journey_count_matrix_flat = df.to_numpy().flatten()

            # load time matrix
            time_matrix_path = (
                self.data_path
                / "time_matrices"
                / f"interval_{self.time_interval}_time_matrix.xlsx"
            )
            df = pd.read_excel(time_matrix_path, index_col=0)
            time_matrix_flat = df.to_numpy().flatten()

            # stack flattened matrices as columns
            X = np.column_stack(
                (
                    journey_count_matrix_flat + 1e-6,
                    time_matrix_flat + 1e-6,
                    distance_matrix_flat + 1e-6,
                )
            )
            return X

    def _get_lookup(self) -> dict[tuple[str, str, str], float]:
        if self.num_indices == 3:
            mapper_path = (
                Path(__file__).parents[2]
                / "data"
                / "mappers"
                / "anketa_3_indikatora.xlsx"
            )
            mapper_df = pd.read_excel(mapper_path)
            mapper_df = mapper_df.drop(columns=["Br"], errors="ignore")
            mapper_df = mapper_df.rename(columns=self.column_map)

            # normalize rating to [0, 1] and store as float
            rating_min = 1
            rating_max = 6
            lookup = {}
            for _, row in mapper_df.iterrows():
                key = (
                    str(row["journey_count_category"]).strip(),
                    str(row["duration_category"]).strip(),
                    str(row["distance_category"]).strip(),
                )
                r = float(row["rating"])
                lookup[key] = (
                    (r - rating_min) / (rating_max - rating_min)
                    if rating_max > rating_min
                    else 0.0
                )
            return lookup

    def _get_Y(
        self, X: np.ndarray, lookup: dict[tuple[str, str, str], float]
    ) -> np.ndarray:
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

    def set_data(
        self,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        X = self._get_X()
        lookup = self._get_lookup()
        Y = self._get_Y(X, lookup)
        self.X = X
        self.Y = Y

    def get_train_val_data(self, random_state: int = 42):
        """
        Returns a train-validation split using 60%-20% of the original data.
        """
        # First split: 80% train+val, 20% test (we’ll discard test here)
        X_train_val, _, Y_train_val, _ = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=random_state
        )
        # Second split: 75% train, 25% val → effectively 60% train, 20% val
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_val,
            Y_train_val,
            test_size=0.25,
            random_state=random_state,
        )
        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        return X_train, X_val, Y_train, Y_val

    def get_test_data(self, random_state: int = 42):
        """
        Returns a hold-out test set using 20% of the original data.
        """
        _, X_test, _, Y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=random_state
        )
        print(f"Test samples: {len(X_test)}")
        return X_test, Y_test

    def train(self) -> None:
        # data
        print("Preparing data...")
        self.set_data()
        print(f"Data shape: X={self.X.shape}, Y={self.Y.shape}")

        # split data into training and validation sets
        X_train, X_val, Y_train, Y_val = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42
        )
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # training
        print("Starting ANFIS training...")
        self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val))
        print("Training completed successfully")

        # save model
        self.model.save(self.models_dir / "anfis_model.pkl")
        print(f"Model saved to {self.models_dir / 'anfis_model.pkl'}")

        # extract plots from the training history
        history = self.model.training_history_
        train_loss = history["train"]
        val_loss = history["val"]
        print("Plotting training progress...")
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("ANFIS Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_path / "training_loss.png")

    def test(self):
        """
        Evaluate the model on the test set.
        """
        # get data
        X_test, Y_test = self.get_test_data()

        # evaluate metrics
        Y_pred = self.model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        # print metrics
        print("Test Set Evaluation:")
        print(f"R²: {r2:.4f}")
        print(f"MSE: {mse:.4f}")

        # X = randomly selected sample indices, Y = y_true and y_pred
        n = len(Y_test)
        n_plot = min(500, n)
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=n_plot, replace=False)
        y_test_rand = Y_test[idx]
        y_pred_rand = Y_pred[idx]
        x_samples = np.arange(n_plot)
        plt.figure(figsize=(12, 5))
        plt.plot(x_samples, y_test_rand, "o-", label="y_true", markersize=4, alpha=0.8)
        plt.plot(x_samples, y_pred_rand, "s-", label="y_pred", markersize=4, alpha=0.8)
        plt.xlabel("Sample")
        plt.ylabel("Rating")
        plt.title("ANFIS: y_true vs y_pred (random sample)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_path / "predictions_vs_true_ratings.png")
