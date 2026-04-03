from pathlib import Path

from anfis_toolbox import ANFISRegressor

from .database import ANFISModel, init_database

MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"


class ModelRegistry:

    def __init__(self):
        self.db = init_database()

    def save_model(
        self,
        model: ANFISRegressor,
        model_id: str,
        train_config: dict[str, str],
        metrics: dict[str:str],
    ):
        """Saves model checkpoint and saves it to the database"""
        path = MODELS_ROOT / f"{model_id}.pkl"
        ANFISModel.get_or_create(
            model_id=model_id, train_config=train_config, metrics=metrics
        )
        model.save(path)
