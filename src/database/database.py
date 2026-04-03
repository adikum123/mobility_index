"""
Database models for tracking embedders, detectors, train runs, and metrics.

Uses Peewee ORM with SQLite for persistent storage.
"""

from datetime import datetime
from pathlib import Path
from tkinter import CASCADE

from peewee import (
    CharField,
    CompositeKey,
    DateTimeField,
    ForeignKeyField,
    Model,
    SqliteDatabase,
)
from playhouse.sqlite_ext import JSONField

# Database will be initialized lazily
_db = SqliteDatabase(None)


def get_default_db_path() -> Path:
    """Get the path to the default SQLite database file."""
    project_root = Path(__file__).resolve().parents[2]
    db_dir = project_root / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "mobility_index.db"


def get_database() -> SqliteDatabase:
    """Get the database instance, initializing if needed.

    Args:
        db_path: Optional path to a SQLite database file.
                 If None, uses the default path (models/patchable_wm/registry.db).
                 If a different path is provided than the currently connected one,
                 the connection is re-initialized to point to the new file.
    """
    db_path = get_default_db_path()
    resolved = str(db_path) if db_path is not None else str(get_default_db_path())

    if _db.database is None:
        if not _db.is_closed():
            _db.close()
        _db.init(resolved)
        _current_db_path = resolved

    return _db


class BaseModel(Model):
    """Base model with database binding."""

    class Meta:
        database = _db


class ANFISModel(BaseModel):
    model_id: CharField(primary_key=True)
    path: CharField
    updated_at: DateTimeField(default=datetime.now)
    config: JSONField

    class Meta:

        table_name = "anfis_model"


class TrainRun(BaseModel):
    train_run_id: CharField
    model_id: ForeignKeyField(ANFISModel, on_delete=CASCADE)
    train_config: JSONField
    metrics: JSONField
    updated_at: DateTimeField(default=datetime.now)

    class Meta:

        table_name = "train_run"
        primary_key = CompositeKey("train_run_id", "model_id")


class TestRun(BaseModel):
    test_run_id: CharField
    model_id: ForeignKeyField(ANFISModel, on_delete=CASCADE)
    metrics: JSONField
    updated_at: DateTimeField(default=datetime.now)

    class Meta:

        table_name = "test_run"
        primary_key = CompositeKey("model_id", "test_run_id")


ALL_MODELS = [ANFISModel, TrainRun, TestRun]


def init_database() -> SqliteDatabase:
    """Initialize the database and create tables if they don't exist."""
    db = get_database()
    db.connect(reuse_if_open=True)
    db.create_tables(ALL_MODELS, safe=True)
    return db


def close_database() -> None:
    """Close the database connection."""
    if not _db.is_closed():
        _db.close()
