"""SQLAlchemy persistence layer (SQLite by default).

A single ``Database`` instance owns the engine and session factory. ORM models
(the fine-tuned-model registry and the job store) subclass :class:`Base`. SQLite
is configured for multi-threaded access because training runs in a background
thread while the UI thread reads state.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from llmstudio.core.utils.logging import get_logger

log = get_logger("utils.db")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Database:
    """Owns a SQLAlchemy engine + session factory and creates tables on demand."""

    def __init__(self, url_or_path: Union[str, Path], *, echo: bool = False) -> None:
        if isinstance(url_or_path, Path) or "://" not in str(url_or_path):
            path = Path(url_or_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{path}"
        else:
            url = str(url_or_path)

        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            url,
            echo=echo,
            future=True,
            connect_args=connect_args,
        )
        if url.startswith("sqlite"):
            # WAL lets the UI keep reading while the training thread writes
            # metrics/checkpoints, and busy_timeout avoids spurious "database is
            # locked" errors under concurrent access.
            @event.listens_for(self.engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self._created = False
        self._lock = threading.Lock()

    def create_all(self) -> None:
        """Create tables for all registered ORM models (idempotent)."""
        with self._lock:
            if not self._created:
                # Import models so they are registered on Base.metadata before
                # create_all. Local import avoids a circular import at module load.
                from llmstudio.core.models import registry as _registry  # noqa: F401
                from llmstudio.core.training import job as _job  # noqa: F401

                Base.metadata.create_all(self.engine)
                self._created = True
                log.debug("Database tables ensured at %s", self.engine.url)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional session scope: commit on success, rollback on error."""
        self.create_all()
        sess = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()


# ---------------------------------------------------------------------------
# Process-wide singleton keyed to the configured registry DB path.
# ---------------------------------------------------------------------------
_DB: Optional[Database] = None
_DB_LOCK = threading.Lock()


def get_database(db_path: Optional[Union[str, Path]] = None) -> Database:
    """Return the shared :class:`Database`, creating it from settings if needed."""
    global _DB
    if db_path is not None:
        return Database(db_path)
    if _DB is None:
        with _DB_LOCK:
            if _DB is None:
                from llmstudio.config import get_settings

                path = get_settings().resolved_paths.registry_db
                _DB = Database(path)
    return _DB


def reset_database() -> None:
    """Drop the cached singleton (tests / after a path change)."""
    global _DB
    with _DB_LOCK:
        _DB = None
