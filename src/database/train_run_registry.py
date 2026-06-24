import uuid

from .database import TrainRun, init_database


class TrainRunRegistry:

    def __init__(self):
        self.db = init_database()

    def save_train_run(
        self,
        model_id: str,
        train_config: dict,
        metrics: dict,
    ) -> str:
        run_id = uuid.uuid4().hex
        TrainRun.insert(  # pylint: disable=no-value-for-parameter
            run_id=run_id,
            model_id=model_id,
            train_config=train_config,
            metrics=metrics,
        ).execute()
        return run_id
