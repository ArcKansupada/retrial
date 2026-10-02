"""Deciding what an import would do, before anything is written.

`validate` produces a plan and refuses whenever the file and the store disagree. Nothing here
writes.
"""

import json

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import fork, record
from retrial.errors import ExportFormatError, SchemaVersionError
from retrial.portable import dump_line, parse_document
from retrial.storage import Store
from retrial.transfer import export, validate

PRICE_999 = {
    "op": "replace",
    "path": "/output/0/content",
    "value": json.dumps({"flight_price": 999}),
}


@pytest.fixture
def source(tmp_path, opening):
    """A store holding a root run and a fork of it."""
    store = Store(str(tmp_path / "source" / ".retrial" / "sessions.db"))
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
    yield {"store": store, "root": root, "fork": forked}
    store.close()


@pytest.fixture
def empty(tmp_path):
    store = Store(str(tmp_path / "target" / ".retrial" / "sessions.db"))
    yield store
    store.close()


def lines_of(source, *args, **kwargs):
    return list(export(source["store"], *args, **kwargs))


def doc(lines):
    return parse_document(lines)


def edited(lines, index, **changes):
    """One row altered - how a file gets corrupted, and how to test for it."""
    rows = [json.loads(line) for line in lines]
    rows[index].update(changes)
    return [dump_line(r) for r in rows]


# -- the happy paths ------------------------------------------------------------


def test_a_fresh_import_plans_every_row(source, empty):
    plan = validate(empty, doc(lines_of(source)))

    assert [s["id"] for s in plan.new_sessions] == [source["root"], source["fork"]]
    assert len(plan.new_steps) == sum(
        len(source["store"].steps_for(s)) for s in (source["root"], source["fork"])
    )
    assert plan.skipped_steps == []
    assert not plan.is_empty


def test_re_importing_the_same_file_plans_nothing(source, empty):
    """Idempotence: the file coming back changes nothing the second time."""
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])

    plan = validate(empty, doc(lines))
    assert plan.new_sessions == [] and plan.new_steps == []
    assert plan.is_empty


def test_an_overlapping_file_plans_only_what_is_new(source, empty):
    """The main flow: your exported root comes back alongside someone's fork of it."""
    _apply(empty, validate(empty, doc(lines_of(source, [source["root"]]))),
           source["store"])

    plan = validate(empty, doc(lines_of(source)))
    assert [s["id"] for s in plan.new_sessions] == [source["fork"]]
    assert plan.skipped_sessions == [source["root"]]
    assert {s["session_id"] for s in plan.new_steps} == {source["fork"]}


def _apply(store, plan, origin):
    """A stand-in for the writer, so these tests can set up a second import."""
    for session in plan.new_sessions:
        store.conn.execute(
            "INSERT INTO sessions (id, name, parent_session_id, parent_sha, "
            "forked_at_step, edit_json, created_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session["id"],
                session["name"],
                session["parent_session_id"],
                session["parent_sha"],
                session["forked_at_step"],
                json.dumps(session["edit"]) if session["edit"] else None,
                session["created_at"],
                session["status"],
            ),
        )
    for step in plan.new_steps:
        store.add_step(
            step["session_id"],
            step["step_number"],
            step["step_type"],
            step["input"],
            step["output"],
        )
    store.conn.commit()


# -- integrity of the content ---------------------------------------------------


def test_a_step_whose_content_was_altered_is_refused(source, empty):
    """The sha is recomputed, not trusted."""
    lines = lines_of(source)
    step_index = next(
        i for i, line in enumerate(lines) if json.loads(line)["kind"] == "step"
    )
    tampered = edited(lines, step_index, output={"content": "something else"})

    with pytest.raises(ExportFormatError, match="does not match its content"):
        validate(empty, doc(tampered))


def test_the_altered_step_is_named_by_line(source, empty):
    lines = lines_of(source)
    step_index = next(
        i for i, line in enumerate(lines) if json.loads(line)["kind"] == "step"
    )
    tampered = edited(lines, step_index, input={"messages": []})

    with pytest.raises(ExportFormatError) as excinfo:
        validate(empty, doc(tampered))
    assert excinfo.value.line == step_index + 1


def test_an_altered_step_number_is_caught_by_the_sha(source, empty):
    """step_number is inside the hash, so renumbering is tamper too.

    Uses the last step: renumbering another trips the parse-time ordering rule first.
    """
    lines = lines_of(source)
    last_step = max(
        i for i, line in enumerate(lines) if json.loads(line)["kind"] == "step"
    )
    with pytest.raises(ExportFormatError, match="does not match its content"):
        validate(empty, doc(edited(lines, last_step, step_number=99)))


# -- referential integrity ------------------------------------------------------


def test_a_fork_whose_parent_is_nowhere_is_refused(source, empty):
    """--no-ancestors into a store that does not have the parent."""
    lines = lines_of(source, [source["fork"]], ancestors=False)

    with pytest.raises(ExportFormatError, match="neither in this file nor"):
        validate(empty, doc(lines))


def test_a_fork_whose_parent_is_already_here_is_accepted(source, empty):
    """The same file, into a store that already has the parent."""
    _apply(empty, validate(empty, doc(lines_of(source, [source["root"]]))),
           source["store"])

    plan = validate(empty, doc(lines_of(source, [source["fork"]], ancestors=False)))
    assert [s["id"] for s in plan.new_sessions] == [source["fork"]]


# -- two runs claiming one id ---------------------------------------------------


def test_a_session_differing_in_content_is_refused(source, empty):
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])

    with pytest.raises(ExportFormatError, match="different name"):
        validate(empty, doc(edited(lines, 1, name="a-different-run")))


def test_the_refusal_names_every_differing_field(source, empty):
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])
    changed = edited(lines, 1, name="other", created_at=1.0)

    with pytest.raises(ExportFormatError) as excinfo:
        validate(empty, doc(changed))
    assert "name" in str(excinfo.value) and "created_at" in str(excinfo.value)


def test_the_refusal_points_at_a_separate_store(source, empty):
    """The message has to name the escape, because there is one."""
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])

    with pytest.raises(ExportFormatError, match="--db"):
        validate(empty, doc(edited(lines, 1, name="other")))


def test_a_different_step_in_the_same_slot_is_refused(source, empty):
    """Same session, same step number, different content."""
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])

    # A real step from the fork, relabelled onto the root's slot 0, with a recomputed sha.
    from retrial.sha import compute_sha

    rows = [json.loads(line) for line in lines]
    victim = next(r for r in rows if r["kind"] == "step")
    victim["output"] = {"content": [{"type": "text", "text": "different"}]}
    victim["sha"] = compute_sha(
        victim["session_id"], victim["step_number"], victim["step_type"],
        victim["input"], victim["output"],
    )

    with pytest.raises(ExportFormatError, match="already has a different step"):
        validate(empty, doc([dump_line(r) for r in rows]))


# -- status is the one permitted change -----------------------------------------


def test_a_running_session_may_finish(source, empty):
    """Export a run in progress, export it again once it completes."""
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])
    empty.set_status(source["root"], "running")

    plan = validate(empty, doc(lines))
    assert plan.status_updates == [(source["root"], "complete")]
    assert not plan.is_empty


def test_a_finished_session_may_not_change_its_outcome(source, empty):
    """complete -> failed is two different runs, not an update."""
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])

    with pytest.raises(ExportFormatError, match="does not change"):
        validate(empty, doc(edited(lines, 1, status="failed")))


def test_a_finished_session_may_not_go_back_to_running(source, empty):
    lines = lines_of(source)
    _apply(empty, validate(empty, doc(lines)), source["store"])

    with pytest.raises(ExportFormatError, match="does not change"):
        validate(empty, doc(edited(lines, 1, status="running")))


# -- versions -------------------------------------------------------------------


def test_a_newer_format_declaring_nothing_required_is_read_with_a_warning(
    source, empty
):
    """Unrecognized fields that nobody declared required are inert."""
    lines = edited(lines_of(source), 0, format=99)
    plan = validate(empty, doc(lines))

    assert plan.new_sessions
    assert any("newer than this retrial" in w for w in plan.warnings)


def test_a_newer_format_requiring_an_unknown_feature_is_refused(source, empty):
    """The other half of the bargain: a producer can say "this one matters"."""
    lines = edited(lines_of(source), 0, format=99, requires=["encrypted-steps"])

    with pytest.raises(ExportFormatError, match="encrypted-steps"):
        validate(empty, doc(lines))


def test_that_refusal_says_how_to_fix_it(source, empty):
    lines = edited(lines_of(source), 0, format=99, requires=["encrypted-steps"])

    with pytest.raises(ExportFormatError, match="pip install -U retrial"):
        validate(empty, doc(lines))


def test_an_older_format_with_no_translator_is_refused_as_a_bug(source, empty):
    """Every format retrial has shipped should be readable."""
    lines = edited(lines_of(source), 0, format=0)

    with pytest.raises(ExportFormatError, match="this is a bug"):
        validate(empty, doc(lines))


def test_a_newer_schema_is_refused(source, empty):
    """Rows from a schema this store cannot hold."""
    lines = edited(lines_of(source), 0, schema=99)

    with pytest.raises(SchemaVersionError, match="newer retrial"):
        validate(empty, doc(lines))


# -- nothing is written ---------------------------------------------------------


def test_validation_never_writes(source, empty):
    """Stated directly, because everything above depends on it."""
    before = (empty.list_sessions(), empty.conn.execute(
        "SELECT COUNT(*) AS n FROM steps").fetchone()["n"])

    validate(empty, doc(lines_of(source)))
    with pytest.raises(ExportFormatError):
        validate(empty, doc(lines_of(source, [source["fork"]], ancestors=False)))

    after = (empty.list_sessions(), empty.conn.execute(
        "SELECT COUNT(*) AS n FROM steps").fetchone()["n"])
    assert before == after
