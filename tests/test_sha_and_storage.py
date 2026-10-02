import sqlite3

import pytest

from retrial.errors import (
    AmbiguousSha,
    NotFound,
    ReplayIntegrityError,
    RetrialError,
    SchemaVersionError,
)
from retrial.serialize import canonical_json, to_jsonable
from retrial.sha import compute_sha
from retrial.storage import SCHEMA_VERSION, Store


def test_sha_is_stable_across_dict_ordering():
    """Key order must not change a step's identity."""
    a = compute_sha("s_1", 0, "model_call", {"b": 1, "a": 2}, {"z": 1, "y": 2})
    b = compute_sha("s_1", 0, "model_call", {"a": 2, "b": 1}, {"y": 2, "z": 1})
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"session_id": "s_2"},
        {"step_number": 1},
        {"step_type": "tool_call"},
        {"input_obj": {"a": 99}},
        {"output_obj": {"z": 99}},
    ],
)
def test_every_component_changes_the_sha(kwargs):
    base = dict(
        session_id="s_1",
        step_number=0,
        step_type="model_call",
        input_obj={"a": 1},
        output_obj={"z": 1},
    )
    assert compute_sha(**base) != compute_sha(**{**base, **kwargs})


def test_canonical_json_is_deterministic():
    assert canonical_json({"b": [3, 1], "a": {"d": 1, "c": 2}}) == (
        '{"a":{"c":2,"d":1},"b":[3,1]}'
    )


def test_serializer_handles_sdk_shaped_objects():
    class Pydanticish:
        def model_dump(self, mode=None):
            return {"stop_reason": "end_turn", "content": [{"type": "text"}]}

    class SdkIsh:
        def to_dict(self):
            return {"ok": True}

    assert to_jsonable(Pydanticish())["stop_reason"] == "end_turn"
    assert to_jsonable(SdkIsh()) == {"ok": True}


def test_serializer_refuses_rather_than_storing_a_lossy_shadow():
    """repr() would make the recording a description, not a recording."""

    class Opaque:
        __slots__ = ()

    with pytest.raises(ReplayIntegrityError, match="cannot serialize"):
        to_jsonable({"resp": Opaque()})


def test_serializer_refuses_cycles():
    d = {}
    d["self"] = d
    with pytest.raises(ReplayIntegrityError, match="circular"):
        to_jsonable(d)


# --- storage ---------------------------------------------------------------


def test_sha_prefix_resolves_like_git(store):
    sid = store.create_session("x")
    sha = store.add_step(sid, 0, "model_call", {"a": 1}, {"b": 2})
    assert store.resolve_sha(sha[:7]) == sha
    assert store.get_step(sha[:4])["sha"] == sha


def test_ambiguous_prefix_errors_instead_of_guessing(store):
    """§11's collision decision: error and ask for a longer prefix."""
    sid = store.create_session("x")
    shas = [store.add_step(sid, i, "model_call", {"i": i}, {}) for i in range(50)]

    # Find a 1-char prefix shared by at least two steps.
    prefix = next(
        p for p in "0123456789abcdef" if sum(s.startswith(p) for s in shas) > 1
    )
    with pytest.raises(AmbiguousSha, match="ambiguous"):
        store.resolve_sha(prefix)


def test_unknown_sha_errors(store):
    with pytest.raises(NotFound):
        store.resolve_sha("deadbeef")


def test_fork_is_a_new_row_not_a_mutation(store):
    parent = store.create_session("orig")
    sha = store.add_step(parent, 0, "tool_call", [], [])
    child = store.create_session(
        "fork", parent_session_id=parent, parent_sha=sha, forked_at_step=0
    )
    assert store.get_session(parent)["parent_session_id"] is None
    assert store.get_session(child)["parent_sha"] == sha
    assert len(store.list_sessions()) == 2


# -- schema versioning --------------------------------------------------------
#
# The file states which schema wrote it; an unreadable version is refused at open.


def _user_version(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_a_new_database_is_stamped_with_the_schema_version(tmp_path):
    path = str(tmp_path / "new.db")
    Store(path).close()
    assert _user_version(path) == SCHEMA_VERSION


def test_reopening_keeps_the_stamp_and_the_data(tmp_path):
    path = str(tmp_path / "reopen.db")
    with Store(path) as store:
        session_id = store.create_session(name="first")

    with Store(path) as store:
        assert store.get_session(session_id)["name"] == "first"
    assert _user_version(path) == SCHEMA_VERSION


def test_a_newer_schema_is_refused_rather_than_opened(tmp_path):
    path = str(tmp_path / "future.db")
    Store(path).close()

    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError) as excinfo:
        Store(path)

    message = str(excinfo.value)
    assert "newer retrial" in message
    assert f"v{SCHEMA_VERSION + 1}" in message
    assert "pip install -U retrial" in message


def test_refusing_a_newer_schema_does_not_leave_the_file_locked(tmp_path):
    """A failed open must close its connection, or Windows keeps the file locked."""
    path = str(tmp_path / "locked.db")
    Store(path).close()
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.commit()
    conn.close()

    for _ in range(3):
        with pytest.raises(SchemaVersionError):
            Store(path)

    # Still readable and writable by anyone else, i.e. nothing was left open.
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    conn.close()
    Store(path).close()


def test_a_pre_versioning_database_is_adopted_not_refused(tmp_path):
    """retrial 0.1.0 wrote v1 tables with no stamp. That layout is v1."""
    path = str(tmp_path / "legacy.db")
    with Store(path) as store:
        session_id = store.create_session(name="recorded-before-versioning")
        store.add_step(session_id, 0, "model_call", {"messages": []}, {"ok": True})

    # Reproduce a 0.1.0 file: correct tables, no stamp.
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    assert _user_version(path) == 0

    with Store(path) as store:
        assert store.get_session(session_id)["name"] == "recorded-before-versioning"
        assert len(store.steps_for(session_id)) == 1
    assert _user_version(path) == SCHEMA_VERSION


def test_schema_version_error_is_a_retrial_error(tmp_path):
    """So the CLI renders it as a message rather than a traceback."""
    assert issubclass(SchemaVersionError, RetrialError)
