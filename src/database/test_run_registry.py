import uuid

from .database import TestRun, init_database


class TestRunRegistry:

    def __init__(self):
        self.db = init_database()

    def save_test_run(
        self,
        model_id: str,
        metrics: dict,
    ):
        TestRun.insert(  # pylint: disable=no-value-for-parameter
            test_run_id=uuid.uuid4().hex,
            model_id=model_id,
            metrics=metrics,
        ).execute()
