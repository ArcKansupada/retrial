"""SQLite storage: one local file, tree-structured.

A fork is a new session row, so original sessions are never mutated.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

from .errors import AmbiguousSha, NotFound, SchemaVersionError
from .sha import compute_sha
from .types import JSON, EditProvenance, Session, SessionStatus, Step, StepType

DEFAULT_DIR = ".retrial"
DB_NAME = "sessions.db"

#: Points every command at one store, overriding upward search.
ENV_VAR = "RETRIAL_DB"

#: Bump when the tables change, and add a migration in `_migrate`. Stored in the file via
#: `PRAGMA user_version`.
#:
#: v1: sessions, steps and their indexes. Written before the marker existed, so v0 with tables
#: present means v1.
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    name              TEXT,
    parent_session_id TEXT,
    parent_sha        TEXT,
    forked_at_step    INTEGER,
    edit_json         TEXT,
    created_at        REAL NOT NULL,
    status            TEXT NOT NULL,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS steps (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sha          TEXT NOT NULL UNIQUE,
    session_id   TEXT NOT NULL,
    step_number  INTEGER NOT NULL,
    step_type    TEXT NOT NULL,
    input_json   TEXT NOT NULL,
    output_json  TEXT NOT NULL,
    tokens_used  INTEGER,
    cost_usd     REAL,
    duration_ms  REAL,
    created_at   REAL NOT NULL,
    UNIQUE (session_id, step_number),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_steps_sha ON steps(sha);
CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id, step_number);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
"""


def default_db_path(root: str | None = None) -> str:
    """Where a store rooted at `root` (default: cwd) is created.

    To find an existing one, use `resolve_db_path`.
    """
    return os.path.join(root or os.getcwd(), DEFAULT_DIR, DB_NAME)


def find_db_path(start: str | None = None) -> str | None:
    """The nearest existing store at or above `start`, or None."""
    current = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(current, DEFAULT_DIR, DB_NAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:  # filesystem root
            return None
        current = parent


def resolve_db_path(start: str | None = None) -> str:
    """Which database a command should use, in precedence order.

    1. `RETRIAL_DB`. 2. The nearest existing store at or above `start`. 3.
    `start`/.retrial/sessions.db, created here.

    An explicit `--db` outranks all three.
    """
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return from_env
    return find_db_path(start) or default_db_path(start)


def schema_version(conn: sqlite3.Connection) -> int:
    """The schema version stamped into the file itself."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _apply_schema(conn: sqlite3.Connection, path: str) -> None:
    """Bring a database up to SCHEMA_VERSION, or refuse to touch it."""
    found = schema_version(conn)
    initialized = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        is not None
    )

    if found > SCHEMA_VERSION:
        # Written by a newer retrial. Refuse rather than risk corrupting it.
        raise SchemaVersionError(path, found, SCHEMA_VERSION)

    if initialized and found == 0:
        # Written by retrial 0.1.0, before the marker existed. That layout is v1.
        _stamp(conn, SCHEMA_VERSION)
        return

    if initialized and found < SCHEMA_VERSION:
        _migrate(conn, path, found)
        return

    conn.executescript(SCHEMA)
    _stamp(conn, SCHEMA_VERSION)


def _migrate(conn: sqlite3.Connection, path: str, found: int) -> None:
    """Upgrade an older database in place. Empty at v1; the next bump goes here."""
    raise SchemaVersionError(path, found, SCHEMA_VERSION)


def _stamp(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA takes no bound parameters. `version` is an int constant, never user input.
    conn.execute(f"PRAGMA user_version = {int(version)}")
    conn.commit()


class Store:
    """A connection to one database. Safe to share between threads.

    Every method that touches the connection holds `_lock`, across each statement and its
    commit.
    """

    path: str
    conn: sqlite3.Connection

    def __init__(self, path: str | None = None) -> None:
        self.path = path or default_db_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # Reentrant: get_step() takes the lock and then calls resolve_sha().
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Every step is committed as it happens, so a crashed run keeps its trace. WAL with
        # synchronous=NORMAL makes that cheap; only an OS crash can lose the last commits.
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")

        try:
            _apply_schema(self.conn, self.path)
        except BaseException:
            # Close the connection if opening fails; on Windows an orphaned handle locks the
            # file.
            self.conn.close()
            raise

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Several writes that land together, or not at all.

        Holds the lock for the whole block.
        """
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- sessions ---------------------------------------------------------

    def create_session(
        self,
        name: str | None,
        parent_session_id: str | None = None,
        parent_sha: str | None = None,
        forked_at_step: int | None = None,
        edit: EditProvenance | None = None,
        status: SessionStatus = "running",
    ) -> str:
        session_id = "s_" + uuid.uuid4().hex[:10]
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions (id, name, parent_session_id, parent_sha, "
                "forked_at_step, edit_json, created_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    name,
                    parent_session_id,
                    parent_sha,
                    forked_at_step,
                    json.dumps(edit) if edit is not None else None,
                    time.time(),
                    status,
                ),
            )
            self.conn.commit()
        return session_id

    def set_status(self, session_id: str, status: SessionStatus) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET status = ? WHERE id = ?", (status, session_id)
            )
            self.conn.commit()

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise NotFound(f"no session {session_id!r}")
        return cast(Session, dict(row))

    def list_sessions(self) -> list[Session]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM sessions ORDER BY created_at ASC"
            ).fetchall()
        return [cast(Session, dict(r)) for r in rows]

    # -- steps ------------------------------------------------------------

    def add_step(
        self,
        session_id: str,
        step_number: int | None,
        step_type: StepType,
        input_obj: JSON,
        output_obj: JSON,
        tokens_used: int | None = None,
        cost_usd: float | None = None,
        duration_ms: float | None = None,
    ) -> str:
        """Append a step. `step_number=None` allocates the next one atomically.

        Pass None unless reconstructing a specific numbering.
        """
        from .serialize import canonical_json

        with self._lock:
            if step_number is None:
                step_number = self._next_step_number(session_id)
            sha = compute_sha(session_id, step_number, step_type, input_obj, output_obj)
            self.conn.execute(
                "INSERT INTO steps (sha, session_id, step_number, step_type, "
                "input_json, output_json, tokens_used, cost_usd, duration_ms, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sha,
                    session_id,
                    step_number,
                    step_type,
                    canonical_json(input_obj),
                    canonical_json(output_obj),
                    tokens_used,
                    cost_usd,
                    duration_ms,
                    time.time(),
                ),
            )
            self.conn.commit()
        return sha

    def steps_for(self, session_id: str) -> list[Step]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM steps WHERE session_id = ? ORDER BY step_number ASC",
                (session_id,),
            ).fetchall()
        return [_step(r) for r in rows]

    def next_step_number(self, session_id: str) -> int:
        """The number the next step would get. Advisory only - see `add_step`."""
        with self._lock:
            return self._next_step_number(session_id)

    def _next_step_number(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(step_number) AS n FROM steps WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return 0 if row["n"] is None else row["n"] + 1

    def resolve_sha(self, prefix: str) -> str:
        """Resolve a (possibly abbreviated) SHA to exactly one step."""
        prefix = prefix.strip().lower()
        if not prefix:
            raise NotFound("empty SHA")
        with self._lock:
            rows = self.conn.execute(
                "SELECT sha FROM steps WHERE sha LIKE ? || '%'", (prefix,)
            ).fetchall()
        if not rows:
            raise NotFound(f"no step matching SHA {prefix!r}")
        if len(rows) > 1:
            raise AmbiguousSha(prefix, [r["sha"] for r in rows])
        return rows[0]["sha"]

    def get_step(self, sha_or_prefix: str) -> Step:
        with self._lock:
            sha = self.resolve_sha(sha_or_prefix)
            row = self.conn.execute(
                "SELECT * FROM steps WHERE sha = ?", (sha,)
            ).fetchone()
        return _step(row)


def _step(row: sqlite3.Row) -> Step:
    d = dict(row)
    d["input"] = json.loads(d.pop("input_json"))
    d["output"] = json.loads(d.pop("output_json"))
    return cast(Step, d)
