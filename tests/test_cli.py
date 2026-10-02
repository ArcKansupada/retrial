"""CLI tests, driven through click's runner against the real commands."""

import json

import pytest
from click.testing import CliRunner

from retrial.cli import _glyphs, cli


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / ".retrial" / "sessions.db")
    runner = CliRunner()

    def run(*args, expect_ok=True):
        result = runner.invoke(cli, ["--db", db, *args], obj={})
        if expect_ok and result.exit_code != 0:
            raise AssertionError(
                f"`retrial {' '.join(args)}` exited {result.exit_code}\n"
                f"{result.output}\n{result.exception!r}"
            )
        return result

    return run, db, tmp_path


def write_agent(tmp_path, module="toy_agent"):
    """An agent module the CLI can import and call with only `messages`."""
    (tmp_path / f"{module}.py").write_text(
        '''
import json
from retrial import record

def call_model(messages, tools=None):
    last = messages[-1]
    if isinstance(last["content"], str):
        return {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "price", "input": {}}]}
    results = [b for b in last["content"] if b.get("type") == "tool_result"]
    price = json.loads(results[-1]["content"])["price"]
    verdict = "cheap" if price <= 500 else "expensive"
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": verdict}]}

def execute_tools(response):
    return [{"type": "tool_result", "tool_use_id": b["id"],
             "content": json.dumps({"price": 450})}
            for b in response["content"] if b.get("type") == "tool_use"]

@record(session_name="toy")
def run_agent(messages, tools=None, call_model=call_model, execute_tools=execute_tools):
    while True:
        r = call_model(messages, tools)
        messages.append({"role": "assistant", "content": r["content"]})
        if r["stop_reason"] != "tool_use":
            return r
        messages.append({"role": "user", "content": execute_tools(response=r)})

def not_an_agent(messages):
    pass
''',
        encoding="utf-8",
    )


def record_a_run(db, tmp_path, module="toy_agent"):
    import subprocess
    import sys

    write_agent(tmp_path, module)
    script = (
        f"import {module}, retrial.storage as st;"
        f"st.default_db_path = lambda root=None: {db!r};"
        f"{module}.run_agent([{{'role':'user','content':'hi'}}])"
    )
    # Run in a subprocess with cwd set, since the default store resolves lazily.
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )


def test_init_creates_the_store(env):
    run, db, _ = env
    result = run("init")
    assert "Initialized empty retrial store" in result.output
    assert run("init").output.startswith("Already initialized")


def test_list_on_an_empty_store(env):
    run, _, _ = env
    run("init")
    assert "No sessions recorded yet" in run("list").output


def test_glyphs_fall_back_to_ascii_on_a_cp1252_console(monkeypatch):
    """Box-drawing characters must not crash a cp1252 console."""

    class Cp1252Stdout:
        encoding = "cp1252"

    monkeypatch.setattr("retrial.cli.sys.stdout", Cp1252Stdout())
    g = _glyphs()
    assert g == {"last": "`-- ", "mid": "|-- ", "pipe": "|   "}
    for value in g.values():
        value.encode("cp1252")  # must not raise


def test_glyphs_use_box_drawing_when_supported(monkeypatch):
    class Utf8Stdout:
        encoding = "utf-8"

    monkeypatch.setattr("retrial.cli.sys.stdout", Utf8Stdout())
    assert _glyphs()["last"] == "└── "


def test_log_show_and_fork_end_to_end(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)

    listed = run("list")
    assert "toy" in listed.output
    session_id = listed.output.split()[0]

    logged = run("log", session_id)
    assert "model_call" in logged.output
    assert "tool_call" in logged.output
    assert "ran price" in logged.output

    tool_line = next(line for line in logged.output.splitlines() if "tool_call" in line)
    sha = tool_line.split()[0]

    shown = run("show", sha)
    assert "tool_call" in shown.output
    assert "tool_result" in shown.output
    # The tool result's content is a JSON string, so it shows escaped.
    assert r'{\"price\": 450}' in shown.output

    edit = tmp_path / "edit.json"
    edit.write_text(
        json.dumps(
            {"op": "replace", "path": "/output/0/content", "value": '{"price": 9000}'}
        ),
        encoding="utf-8",
    )

    forked = run(
        "fork", sha, "--agent", "toy_agent:run_agent", "--edit-file", str(edit)
    )
    assert "Forked into session" in forked.output
    fork_id = forked.output.split()[-1].strip()

    fork_log = run("log", fork_id)
    assert f"forked from {sha}" in fork_log.output
    assert "replace /output/0/content" in fork_log.output

    tree = run("list")
    assert fork_id in tree.output
    assert "forked from" in tree.output


def test_fork_rejects_an_undecorated_agent(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    sha = next(
        line.split()[0]
        for line in run("log", run("list").output.split()[0]).output.splitlines()
        if "tool_call" in line
    )

    result = run(
        "fork", sha, "--agent", "toy_agent:not_an_agent", expect_ok=False
    )
    assert result.exit_code != 0
    assert "not decorated with @record" in result.output


def test_fork_reports_a_bad_agent_spec(env):
    run, _, _ = env
    run("init")
    for spec, message in [
        ("no_colon", "expected MODULE:FUNCTION"),
        ("nope.nothing:x", "could not import"),
    ]:
        result = run("fork", "abc", "--agent", spec, expect_ok=False)
        assert message in result.output


def test_diff_end_to_end(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    session_id = run("list").output.split()[0]
    logged = run("log", session_id)
    sha = next(line.split()[0] for line in logged.output.splitlines() if "tool_call" in line)

    edit = tmp_path / "edit.json"
    edit.write_text(
        json.dumps(
            {"op": "replace", "path": "/output/0/content", "value": '{"price": 9000}'}
        ),
        encoding="utf-8",
    )
    forked = run("fork", sha, "--agent", "toy_agent:run_agent", "--edit-file", str(edit))
    fork_id = forked.output.split()[-1].strip()

    result = run("diff", session_id, fork_id)
    assert f"common ancestor: {session_id}" in result.output
    assert "shared prefix" in result.output
    assert f"diverged at {sha}" in result.output
    assert "replace /output/0/content" in result.output
    assert "cheap" in result.output  # A's final answer
    assert "expensive" in result.output  # B's
    assert "[replayed]" in result.output and "[live]" in result.output


def test_diff_of_a_session_with_itself_is_clean(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    session_id = run("list").output.split()[0]
    result = run("diff", session_id, session_id)
    assert "Trajectories are identical." in result.output


def test_diff_full_flag_expands_the_shared_prefix(env):
    """Without --full the shared prefix collapses to a count. Needs a real divergence."""
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    a = run("list").output.split()[0]
    sha = next(
        line.split()[0] for line in run("log", a).output.splitlines() if "tool_call" in line
    )

    edit = tmp_path / "edit.json"
    edit.write_text(
        json.dumps(
            {"op": "replace", "path": "/output/0/content", "value": '{"price": 9000}'}
        ),
        encoding="utf-8",
    )
    b = run(
        "fork", sha, "--agent", "toy_agent:run_agent", "--edit-file", str(edit)
    ).output.split()[-1].strip()

    summary = run("diff", a, b)
    assert "= 1 shared step(s)" in summary.output

    full = run("diff", a, b, "--full")
    assert "= 1 shared step(s)" not in full.output
    assert "model_call" in full.output


def test_bisect_end_to_end(env):
    """A failing run, localized through the real CLI."""
    run, db, tmp_path = env
    run("init")

    (tmp_path / "flaky_agent.py").write_text(
        '''
import json, os, pathlib
from retrial import record

STATE = pathlib.Path(__file__).parent / "outage.flag"

def call_model(messages, tools=None):
    last = messages[-1]
    if isinstance(last["content"], str):
        return {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "book", "input": {}}]}
    payload = json.loads(
        [b for b in last["content"] if b.get("type") == "tool_result"][-1]["content"])
    text = "Failed to book." if "error" in payload else "Confirmed booking."
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}

def execute_tools(response):
    down = STATE.exists()
    return [{"type": "tool_result", "tool_use_id": b["id"],
             "content": json.dumps({"error": "down"} if down else {"ok": True})}
            for b in response["content"] if b.get("type") == "tool_use"]

@record(session_name="flaky")
def run_agent(messages, tools=None, call_model=call_model, execute_tools=execute_tools):
    while True:
        r = call_model(messages, tools)
        messages.append({"role": "assistant", "content": r["content"]})
        if r["stop_reason"] != "tool_use":
            return r
        messages.append({"role": "user", "content": execute_tools(r)})
''',
        encoding="utf-8",
    )

    import subprocess
    import sys

    # Record a run while the "service" is down.
    (tmp_path / "outage.flag").touch()
    subprocess.run(
        [sys.executable, "-c",
         "import flaky_agent; flaky_agent.run_agent([{'role':'user','content':'book'}])"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )
    # The outage is over. Re-execution can now recover, so bisect has a boundary.
    (tmp_path / "outage.flag").unlink()

    session_id = run("list").output.split()[0]
    result = run(
        "bisect", session_id,
        "--check", "output contains 'Confirmed'",
        "--agent", "flaky_agent:run_agent",
    )

    assert "probe step" in result.output
    assert "recovered" in result.output
    assert "still broken" in result.output
    assert "First step that could not recover" in result.output
    assert "re-execution(s)" in result.output


def test_bisect_refuses_a_passing_run(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    session_id = run("list").output.split()[0]

    result = run(
        "bisect", session_id,
        "--check", "output contains 'cheap'",  # the run already says this
        "--agent", "toy_agent:run_agent",
        expect_ok=False,
    )
    assert result.exit_code != 0


def test_bisect_rejects_an_unparseable_check(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    session_id = run("list").output.split()[0]

    result = run(
        "bisect", session_id, "--check", "gibberish",
        "--agent", "toy_agent:run_agent", expect_ok=False,
    )
    assert result.exit_code != 0


def test_unknown_sha_is_a_clean_error(env):
    run, db, tmp_path = env
    run("init")
    record_a_run(db, tmp_path)
    result = run("show", "ffffff", expect_ok=False)
    assert result.exit_code != 0


def _record_and_get(run, db, tmp_path):
    run("init")
    record_a_run(db, tmp_path)
    return run("list").output.split()[0]


def test_ablate_end_to_end(env):
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)

    result = run(
        "ablate", session_id,
        "--check", "output contains 'cheap'",
        "--agent", "toy_agent:run_agent",
    )
    assert "ablating" in result.output
    assert "re-execution(s)" in result.output
    assert "baseline: check passes" in result.output
    # The asymmetry must be stated, not left for the reader to assume.
    assert "signal is asymmetric" in result.output


def test_ablate_refuses_a_failing_baseline_and_points_at_bisect(env):
    """The dual-tool guard, through the real CLI."""
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)

    result = run(
        "ablate", session_id,
        "--check", "output contains 'never appears'",
        "--agent", "toy_agent:run_agent",
        expect_ok=False,
    )
    assert result.exit_code != 0
    assert "no good outcome to ablate" in result.output
    assert "retrial bisect" in result.output


def test_sweep_end_to_end(env):
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)
    sha = next(
        line.split()[0] for line in run("log", session_id).output.splitlines()
        if "tool_call" in line
    )

    values = tmp_path / "values.json"
    values.write_text(
        json.dumps([json.dumps({"price": p}) for p in (100, 400, 800, 9000)]),
        encoding="utf-8",
    )

    result = run(
        "sweep", sha,
        "--values-file", str(values),
        "--check", "output contains 'cheap'",
        "--agent", "toy_agent:run_agent",
    )
    assert "over 4 value(s)" in result.output
    assert "PASS" in result.output and "FAIL" in result.output
    # toy_agent's rule is price <= 500, so the flip is between 400 and 800.
    assert "Threshold: the check flips between" in result.output


def test_sweep_samples_reports_rates(env):
    """toy_agent is deterministic, so the rates are 3/3 and 0/3."""
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)
    sha = next(
        line.split()[0] for line in run("log", session_id).output.splitlines()
        if "tool_call" in line
    )

    values = tmp_path / "values.json"
    values.write_text(
        json.dumps([json.dumps({"price": p}) for p in (400, 800)]), encoding="utf-8"
    )

    result = run(
        "sweep", sha,
        "--values-file", str(values),
        "--check", "output contains 'cheap'",
        "--samples", "3",
        "--agent", "toy_agent:run_agent",
    )
    assert "over 2 value(s) x 3 samples" in result.output
    assert "PASS 3/3" in result.output
    assert "FAIL 0/3" in result.output
    assert "6 re-execution(s)" in result.output
    assert "pass rate 3/3 -> 0/3" in result.output


def test_sweep_rejects_zero_samples(env):
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)
    sha = next(
        line.split()[0] for line in run("log", session_id).output.splitlines()
        if "tool_call" in line
    )
    values = tmp_path / "values.json"
    values.write_text(json.dumps([json.dumps({"price": 100})]), encoding="utf-8")

    result = run(
        "sweep", sha, "--values-file", str(values), "--samples", "0",
        "--agent", "toy_agent:run_agent", expect_ok=False,
    )
    assert result.exit_code != 0


def test_sweep_without_a_check_reports_answers(env):
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)
    sha = next(
        line.split()[0] for line in run("log", session_id).output.splitlines()
        if "tool_call" in line
    )
    values = tmp_path / "values.json"
    values.write_text(json.dumps([json.dumps({"price": 100})]), encoding="utf-8")

    result = run(
        "sweep", sha, "--values-file", str(values), "--agent", "toy_agent:run_agent"
    )
    assert "cheap" in result.output
    assert "Threshold" not in result.output


def test_sweep_rejects_a_values_file_that_is_not_a_list(env):
    run, db, tmp_path = env
    session_id = _record_and_get(run, db, tmp_path)
    sha = next(
        line.split()[0] for line in run("log", session_id).output.splitlines()
        if "tool_call" in line
    )
    values = tmp_path / "values.json"
    values.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    result = run(
        "sweep", sha, "--values-file", str(values),
        "--agent", "toy_agent:run_agent", expect_ok=False,
    )
    assert result.exit_code != 0
    assert "expected a JSON array" in result.output


# -- entry points -------------------------------------------------------------
#
# `--version` and `python -m` are both promised elsewhere.


def test_version_flag_reports_the_package_version():
    from retrial import __version__

    result = CliRunner().invoke(cli, ["--version"], obj={})
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "retrial" in result.output


def test_version_flag_has_a_short_form():
    result = CliRunner().invoke(cli, ["-V"], obj={})
    assert result.exit_code == 0


def test_python_dash_m_retrial_is_a_working_entry_point():
    """`python -m retrial` must reach the same CLI as the console script."""
    import subprocess
    import sys
    from pathlib import Path

    from retrial import __version__

    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "retrial", "--version"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert __version__ in proc.stdout


def test_package_ships_the_pep561_typed_marker():
    """py.typed is what makes the annotations visible to a consumer.

    Asserted against the installed package.
    """
    from pathlib import Path

    import retrial

    assert (Path(retrial.__file__).parent / "py.typed").is_file()


def test_public_types_are_importable_from_the_top_level():
    """Annotating your own code must not require a private-looking submodule."""
    from retrial import DiffResult, Session, Step, TrajectoryEntry

    assert set(Step.__annotations__) >= {"sha", "step_type", "input", "output"}
    assert "origin" in TrajectoryEntry.__annotations__
    assert "parent_session_id" in Session.__annotations__
    assert "divergence" in DiffResult.__annotations__


def test_a_newer_database_is_refused_as_a_message_not_a_traceback(env):
    """The refusal has to be legible, or it just looks like retrial crashed."""
    import sqlite3

    from retrial.storage import SCHEMA_VERSION

    run, db, tmp_path = env
    run("init")

    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    result = run("list", expect_ok=False)
    assert result.exit_code == 1
    assert result.output.startswith("error: ")
    assert "newer retrial" in result.output
    assert "Traceback" not in result.output
