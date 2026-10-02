"""Which database a command uses.

Discovery searches upward, like git. These cover the search, its precedence against $RETRIAL_DB
and --db, and `init`, which must not search.
"""

import importlib

import pytest
from click.testing import CliRunner
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import record
from retrial.cli import cli
from retrial.storage import (
    ENV_VAR,
    Store,
    default_db_path,
    find_db_path,
    resolve_db_path,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """No inherited RETRIAL_DB, and no store cached from another test."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    # importlib, because `retrial.record` the attribute is the decorator.
    record_module = importlib.import_module("retrial.record")

    def drop():
        for store in record_module._default_stores.values():
            store.close()
        record_module._default_stores.clear()

    drop()
    yield
    drop()


def make_store(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".retrial" / "sessions.db"
    Store(str(path)).close()
    return path


def run(*args):
    return CliRunner().invoke(cli, list(args), obj={})


# -- the search ---------------------------------------------------------------


def test_finds_a_store_in_an_ancestor_directory(tmp_path, monkeypatch):
    db = make_store(tmp_path / "proj")
    deep = tmp_path / "proj" / "a" / "b" / "c"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert find_db_path() == str(db)
    assert resolve_db_path() == str(db)


def test_the_nearest_store_wins(tmp_path, monkeypatch):
    make_store(tmp_path / "outer")
    inner_db = make_store(tmp_path / "outer" / "inner")
    monkeypatch.chdir(tmp_path / "outer" / "inner")

    assert find_db_path() == str(inner_db)


def test_no_store_anywhere_falls_back_to_creating_one_here(tmp_path, monkeypatch):
    """Unchanged first-run behaviour: a bare directory still just works."""
    monkeypatch.chdir(tmp_path)

    assert find_db_path() is None
    assert resolve_db_path() == default_db_path()


def test_the_search_stops_at_the_filesystem_root(tmp_path, monkeypatch):
    """It must terminate, not loop, when nothing is ever found."""
    deep = tmp_path / "x" / "y" / "z"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert find_db_path() is None


# -- precedence ---------------------------------------------------------------


def test_env_var_outranks_the_search(tmp_path, monkeypatch):
    make_store(tmp_path / "proj")
    elsewhere = make_store(tmp_path / "elsewhere")
    monkeypatch.chdir(tmp_path / "proj")
    monkeypatch.setenv(ENV_VAR, str(elsewhere))

    assert resolve_db_path() == str(elsewhere)


def test_db_flag_outranks_the_env_var(tmp_path, monkeypatch):
    flagged = make_store(tmp_path / "flagged")
    env = make_store(tmp_path / "env")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(env))

    with Store(str(flagged)) as store:
        store.create_session(name="only-in-flagged")

    result = run("--db", str(flagged), "list")
    assert result.exit_code == 0, result.output
    assert "only-in-flagged" in result.output


# -- end to end: the actual bug -----------------------------------------------


def test_cli_from_a_subdirectory_reads_the_project_store(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    db = make_store(root)
    with Store(str(db)) as store:
        store.create_session(name="recorded-at-the-root")

    sub = root / "services" / "booking"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    result = run("list")
    assert result.exit_code == 0, result.output
    assert "recorded-at-the-root" in result.output
    # The old behaviour: a second empty store, and "No sessions recorded yet."
    assert not (sub / ".retrial").exists()


def test_recording_from_a_subdirectory_writes_to_the_project_store(
    tmp_path, monkeypatch, opening
):
    root = tmp_path / "proj"
    db = make_store(root)
    sub = root / "services"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    agent = record(session_name="launched-from-a-subdirectory")(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    assert not (sub / ".retrial").exists()
    with Store(str(db)) as store:
        assert [s["name"] for s in store.list_sessions()] == [
            "launched-from-a-subdirectory"
        ]
        assert store.steps_for(store.list_sessions()[0]["id"])


def test_the_cli_sees_what_a_subdirectory_run_recorded(tmp_path, monkeypatch, opening):
    """The halves must agree - that is the whole point of the change."""
    root = tmp_path / "proj"
    make_store(root)
    sub = root / "deep" / "nested"
    sub.mkdir(parents=True)

    monkeypatch.chdir(sub)
    agent = record(session_name="round-trip")(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    monkeypatch.chdir(root)
    assert "round-trip" in run("list").output


# -- init is the exception ----------------------------------------------------


def test_init_creates_here_even_when_a_store_exists_above(tmp_path, monkeypatch):
    """`git init` always makes a repo here. So does this."""
    make_store(tmp_path / "proj")
    sub = tmp_path / "proj" / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)

    result = run("init")
    assert result.exit_code == 0, result.output
    assert (sub / ".retrial" / "sessions.db").is_file()


def test_init_warns_when_it_shadows_a_store_above(tmp_path, monkeypatch):
    """Otherwise the symptom is "my sessions vanished", far from the cause."""
    outer = make_store(tmp_path / "proj")
    sub = tmp_path / "proj" / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)

    output = run("init").output
    assert "note:" in output
    assert str(outer) in output


def test_init_does_not_warn_when_there_is_nothing_above(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "note:" not in run("init").output


def test_init_is_idempotent_and_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("init")
    assert "Already initialized" in run("init").output
