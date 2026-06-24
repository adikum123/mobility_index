from .database import TestRun, init_database


class TestRunRegistry:

    def __init__(self):
        self.db = init_database()

    def save_test_run(
        self,
        run_id: str,
        model_id: str,
        metrics: dict,
    ) -> None:
        TestRun.insert(  # pylint: disable=no-value-for-parameter
            run_id=run_id,
            model_id=model_id,
            metrics=metrics,
        ).execute()
