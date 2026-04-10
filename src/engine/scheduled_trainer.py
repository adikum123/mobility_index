import math

import numpy as np
from anfis_toolbox.optim.hybrid import HybridTrainer


class ScheduledHybridTrainer(HybridTrainer):
    """HybridTrainer with per-epoch learning rate scheduling.

    Supported schedules:
    - ``"cosine"``:  smooth cosine annealing from *initial_lr* to *min_lr*
    - ``"step"``:    halves every ⅓ of total epochs
    - ``"linear"``:  linear decay from *initial_lr* to *min_lr*
    """

    _VALID_SCHEDULES = ("cosine", "step", "linear")

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        verbose: bool = False,
        schedule: str = "cosine",
        min_lr: float = 1e-5,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate, epochs=epochs, verbose=verbose, **kwargs
        )
        if schedule not in self._VALID_SCHEDULES:
            raise ValueError(
                f"Unknown schedule {schedule!r}, must be one of {self._VALID_SCHEDULES}"
            )
        self.initial_lr = learning_rate
        self.min_lr = min_lr
        self.schedule = schedule

    def _get_lr(self, epoch: int, total_epochs: int) -> float:
        if self.schedule == "cosine":
            cos_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
            return self.min_lr + (self.initial_lr - self.min_lr) * cos_decay
        if self.schedule == "step":
            step = max(total_epochs // 3, 1)
            factor = 0.5 ** (epoch // step)
            return max(self.initial_lr * factor, self.min_lr)
        if self.schedule == "linear":
            alpha = 1 - epoch / total_epochs
            return max(self.initial_lr * alpha, self.min_lr)
        return self.initial_lr

    def fit(self, model, X, y, *, validation_data=None, validation_frequency=1):
        if validation_frequency < 1:
            raise ValueError("validation_frequency must be >= 1")

        X_train, y_train = self._prepare_training_data(model, X, y)
        state = self.init_state(model, X_train, y_train)  # noqa: E1111

        prepared_val = None
        if validation_data is not None:
            prepared_val = self._prepare_validation_data(model, *validation_data)

        epochs = int(getattr(self, "epochs", 1))
        batch_size = getattr(self, "batch_size", None)
        shuffle = bool(getattr(self, "shuffle", True))
        verbose = bool(getattr(self, "verbose", False))

        train_history: list[float] = []
        val_history: list[float | None] = [] if prepared_val is not None else []
        n_samples = X_train.shape[0]

        for epoch_idx in range(epochs):
            self.learning_rate = self._get_lr(epoch_idx, epochs)

            epoch_losses: list[float] = []
            if batch_size is None:
                loss, state = self.train_step(model, X_train, y_train, state)
                epoch_losses.append(float(loss))
            else:
                indices = np.arange(n_samples)
                if shuffle:
                    np.random.shuffle(indices)
                for start in range(0, n_samples, batch_size):
                    batch_idx = indices[start : start + batch_size]
                    loss, state = self.train_step(
                        model, X_train[batch_idx], y_train[batch_idx], state
                    )
                    epoch_losses.append(float(loss))

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            train_history.append(epoch_loss)

            val_loss: float | None = None
            if prepared_val is not None:
                if (epoch_idx + 1) % validation_frequency == 0:
                    X_val, y_val = prepared_val
                    val_loss = float(self.compute_loss(model, X_val, y_val))
                val_history.append(val_loss)

            self._log_epoch(epoch_idx, epoch_loss, val_loss, verbose)

        result: dict = {"train": train_history}
        if prepared_val is not None:
            result["val"] = val_history
        return result
