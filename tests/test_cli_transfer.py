"""`retrial export` and `retrial import` at the command line.

An export is data, so it must not go through the console-degrading `echo()`, and diagnostics go
to stderr.
"""

import io
import json
import sys

import pytest
from click.testing import CliRunner
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import fork, record
from retrial.cli import cli, write_data
from retrial.storage import Store

PRICE_999 = {
    "op": "replace",
    "path": "/output/0/content",
    "value": json.dumps({"flight_price": 999}),
}


@pytest.fixture
def project(tmp_path, opening, monkeypatch):
    """A store on disk with a root run and a fork, plus its path."""
    monkeypatch.delenv("RETRIAL_DB", raising=False)
    db = tmp_path / "src" / ".retrial" / "sessions.db"
    store = Store(str(db))
    agent = record(session_name="root-run", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))
    root = agent.last_session_id
    tool = next(s for s in store.steps_for(root) if s["step_type"] == "tool_call")
    forked = fork(
        from_sha=tool["sha"],
        edit=PRICE_999,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )
    store.close()
    return {"db": str(db), "root": root, "fork": forked, "tmp": tmp_path}


def run(*args, **kwargs):
    return CliRunner().invoke(cli, list(args), obj={}, **kwargs)


def export_to(project, *args):
    out = project["tmp"] / "trace.jsonl"
    result = run("--db", project["db"], "export", *args, "-o", str(out))
    assert result.exit_code == 0, result.output
    return out


# -- export ---------------------------------------------------------------------


def test_export_writes_parseable_jsonl(project):
    path = export_to(project, project["fork"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["kind"] == "header"
    assert {r["id"] for r in rows if r["kind"] == "session"} == {
        project["root"],
        project["fork"],
    }


def test_export_to_stdout_is_only_the_file(project):
    """Nothing else may land on stdout, or the pipe is corrupted."""
    result = run("--db", project["db"], "export", project["root"])

    assert result.exit_code == 0
    for line in result.output.splitlines():
        json.loads(line)  # every single line, no exceptions


def test_export_all(project):
    path = export_to(project, "--all")
    ids = {
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "session"
    }
    assert ids == {project["root"], project["fork"]}


def test_export_needs_a_target(project):
    result = run("--db", project["db"], "export")
    assert result.exit_code != 0
    assert "name a session" in result.output


def test_export_refuses_ids_and_all_together(project):
    result = run("--db", project["db"], "export", project["root"], "--all")
    assert result.exit_code != 0
    assert "not both" in result.output


def test_an_unknown_session_is_an_error_not_a_traceback(project):
    result = run("--db", project["db"], "export", "s_nosuchid01")
    assert result.exit_code == 1
    assert result.output.startswith("error:")


def test_no_ancestors_warns_on_stderr_not_stdout(project):
    """The note must not end up inside the file."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--db", project["db"], "export", project["fork"], "--no-ancestors"],
        obj={},
    )

    assert result.exit_code == 0
    stdout = result.stdout
    for line in stdout.splitlines():
        json.loads(line)
    assert "note:" in result.stderr
    assert "note:" not in stdout


# -- the encoding trap ----------------------------------------------------------


def test_export_writes_utf8_whatever_the_console_is(monkeypatch):
    """An export is data: a degraded character would break the step's sha."""
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252", errors="replace")
    monkeypatch.setattr(sys, "stdout", console)

    write_data(['{"text": "✈ booked \U0001f600"}\n'], None)
    console.flush()

    written = raw.getvalue()
    assert written.decode("utf-8") == '{"text": "✈ booked \U0001f600"}\n'
    assert b"?" not in written


def test_a_trace_with_non_ascii_survives_the_round_trip(tmp_path, opening, monkeypatch):
    """End to end, because the sha is what makes this checkable at all."""
    monkeypatch.delenv("RETRIAL_DB", raising=False)
    source_db = tmp_path / "a" / ".retrial" / "sessions.db"
    store = Store(str(source_db))
    session = store.create_session(name="unicode-run")
    store.add_step(
        session,
        None,
        "model_call",
        {"messages": [{"role": "user", "content": "Book me a flight ✈"}]},
        {"content": [{"type": "text", "text": "Booked \U0001f600 - €450"}]},
    )
    original = store.steps_for(session)[0]["sha"]
    store.close()

    path = tmp_path / "trace.jsonl"
    assert run("--db", str(source_db), "export", session, "-o", str(path)).exit_code == 0

    target_db = tmp_path / "b" / ".retrial" / "sessions.db"
    result = run("--db", str(target_db), "import", str(path))
    assert result.exit_code == 0, result.output

    with Store(str(target_db)) as landed:
        step = landed.steps_for(session)[0]
        assert step["sha"] == original
        assert "\U0001f600" in step["output"]["content"][0]["text"]


# -- import ---------------------------------------------------------------------


def test_import_reads_a_file(project, tmp_path):
    path = export_to(project, "--all")
    target = tmp_path / "dest" / ".retrial" / "sessions.db"

    result = run("--db", str(target), "import", str(path))

    assert result.exit_code == 0, result.output
    assert "2 session(s)" in result.output
    with Store(str(target)) as store:
        assert len(store.list_sessions()) == 2


def test_import_reads_stdin(project, tmp_path):
    path = export_to(project, "--all")
    target = tmp_path / "dest" / ".retrial" / "sessions.db"

    result = run(
        "--db", str(target), "import", "-", input=path.read_text(encoding="utf-8")
    )

    assert result.exit_code == 0, result.output
    with Store(str(target)) as store:
        assert len(store.list_sessions()) == 2


def test_a_file_with_a_utf8_bom_still_imports(project, tmp_path):
    """A byte-order mark added by Windows tooling must be ignored."""
    path = export_to(project, "--all")
    with open(path, "rb") as handle:
        body = handle.read()
    bom_path = tmp_path / "with_bom.jsonl"
    bom_path.write_bytes(b"\xef\xbb\xbf" + body)
    target = tmp_path / "dest" / ".retrial" / "sessions.db"

    result = run("--db", str(target), "import", str(bom_path))

    assert result.exit_code == 0, result.output
    with Store(str(target)) as store:
        assert len(store.list_sessions()) == 2


def test_a_bom_on_stdin_still_imports(project, tmp_path):
    path = export_to(project, "--all")
    with open(path, "rb") as handle:
        piped = ("\ufeff" + handle.read().decode("utf-8"))
    target = tmp_path / "dest" / ".retrial" / "sessions.db"

    result = run("--db", str(target), "import", "-", input=piped)

    assert result.exit_code == 0, result.output
    with Store(str(target)) as store:
        assert len(store.list_sessions()) == 2


def test_importing_twice_says_nothing_changed(project, tmp_path):
    path = export_to(project, "--all")
    target = tmp_path / "dest" / ".retrial" / "sessions.db"
    run("--db", str(target), "import", str(path))

    result = run("--db", str(target), "import", str(path))

    assert result.exit_code == 0
    assert "Already up to date" in result.output


def test_a_tampered_file_is_refused_with_a_message(project, tmp_path):
    path = export_to(project, "--all")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    victim = next(r for r in rows if r["kind"] == "step")
    victim["output"] = {"content": "tampered"}
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    target = tmp_path / "dest" / ".retrial" / "sessions.db"

    result = run("--db", str(target), "import", str(path))

    assert result.exit_code == 1
    assert "does not match its content" in result.output
    with Store(str(target)) as store:
        assert store.list_sessions() == []


def test_the_error_names_the_file_and_line(project, tmp_path):
    path = export_to(project, "--all")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[2] = "{ not json\n"
    path.write_text("".join(lines), encoding="utf-8")
    target = tmp_path / "dest" / ".retrial" / "sessions.db"

    result = run("--db", str(target), "import", str(path))

    assert result.exit_code == 1
    assert f"{path}:3" in result.output


def test_an_imported_trace_is_visible_to_the_other_commands(project, tmp_path):
    """The point of importing: `list`, `log` and `diff` all work on it."""
    path = export_to(project, "--all")
    target = tmp_path / "dest" / ".retrial" / "sessions.db"
    run("--db", str(target), "import", str(path))

    listed = run("--db", str(target), "list")
    assert "root-run" in listed.output

    diffed = run("--db", str(target), "diff", project["root"], project["fork"])
    assert diffed.exit_code == 0, diffed.output
