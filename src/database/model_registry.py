from pathlib import Path

from anfis_toolbox import ANFISRegressor

from .database import ANFISModel, init_database


class ModelRegistry:

    def __init__(self):
        self.db = init_database()

    def register_model(
        self,
        model_id: str,
        path: str,
        config: dict,
    ):
        """Insert or update the model record in the database."""
        ANFISModel.insert(  # pylint: disable=no-value-for-parameter
            model_id=model_id,
            path=path,
            config=config,
        ).on_conflict_replace().execute()

    def save_model_file(
        self,
        model: ANFISRegressor,
        path: str | Path,
    ):
        """Persist the trained model weights to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
