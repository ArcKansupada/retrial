"""Importing a trace, and what survives the trip.

The key test is the last: export a fork, import it into an empty store, and check `diff` and
`trajectory` agree on both sides. The rest guard against duplicating, half-importing, or re-
dating a trace.
"""

import json

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import diff, fork, record, trajectory
from retrial.errors import ExportFormatError
from retrial.portable import dump_line, parse_document
from retrial.storage import Store
from retrial.transfer import export, import_

PRICE_999 = {
    "op": "replace",
    "path": "/output/0/content",
    "value": json.dumps({"flight_price": 999}),
}


@pytest.fixture
def source(tmp_path, opening):
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
def target(tmp_path):
    store = Store(str(tmp_path / "target" / ".retrial" / "sessions.db"))
    yield store
    store.close()


def lines_of(source, *args, **kwargs):
    return list(export(source["store"], *args, **kwargs))


def contents(store):
    """Everything that could differ, in a comparable shape."""
    return (
        [dict(s) for s in store.list_sessions()],
        [
            dict(step)
            for s in store.list_sessions()
            for step in store.steps_for(s["id"])
        ],
    )


# -- the trip -------------------------------------------------------------------


def test_a_trace_arrives_whole(source, target):
    result = import_(target, lines_of(source))

    assert result.sessions_added == 2
    assert result.steps_added == sum(
        len(source["store"].steps_for(s)) for s in (source["root"], source["fork"])
    )
    assert {s["id"] for s in target.list_sessions()} == {
        source["root"],
        source["fork"],
    }


def test_shas_survive_the_trip(source, target):
    """Session ids are preserved, so a step keeps the same handle on both machines."""
    import_(target, lines_of(source))

    for session_id in (source["root"], source["fork"]):
        assert [s["sha"] for s in target.steps_for(session_id)] == [
            s["sha"] for s in source["store"].steps_for(session_id)
        ]


def test_a_sha_prefix_still_resolves_after_import(source, target):
    """Which is what makes "look at 9f2c3d" mean the same thing in a reply."""
    import_(target, lines_of(source))
    sha = source["store"].steps_for(source["root"])[0]["sha"]

    assert target.get_step(sha[:7])["sha"] == sha


def test_timestamps_are_when_it_happened_not_when_it_arrived(source, target):
    """Otherwise a store fills with runs all claiming the import date."""
    import_(target, lines_of(source))

    for session_id in (source["root"], source["fork"]):
        assert [s["created_at"] for s in target.steps_for(session_id)] == [
            s["created_at"] for s in source["store"].steps_for(session_id)
        ]
    assert (
        target.get_session(source["root"])["created_at"]
        == source["store"].get_session(source["root"])["created_at"]
    )


def test_step_content_is_byte_identical(source, target):
    import_(target, lines_of(source))

    for session_id in (source["root"], source["fork"]):
        here = target.steps_for(session_id)
        there = source["store"].steps_for(session_id)
        for a, b in zip(here, there, strict=True):
            assert a["input"] == b["input"]
            assert a["output"] == b["output"]


def test_fork_provenance_survives(source, target):
    """Without it the fork is just another root and diff has nothing to say."""
    import_(target, lines_of(source))
    forked = target.get_session(source["fork"])

    original = source["store"].get_session(source["fork"])
    assert forked["parent_session_id"] == original["parent_session_id"]
    assert forked["parent_sha"] == original["parent_sha"]
    assert forked["forked_at_step"] == original["forked_at_step"]
    assert json.loads(forked["edit_json"]) == json.loads(original["edit_json"])


# -- running it twice -----------------------------------------------------------


def test_importing_the_same_file_twice_changes_nothing(source, target):
    lines = lines_of(source)
    import_(target, lines)
    before = contents(target)

    result = import_(target, lines)

    assert result.changed_nothing
    assert result.sessions_skipped == 2
    assert contents(target) == before


def test_a_later_export_appends_only_what_is_new(source, target):
    """You send a root; they fork it and send the file back."""
    import_(target, lines_of(source, [source["root"]]))
    result = import_(target, lines_of(source))

    assert result.sessions_added == 1
    assert result.sessions_skipped == 1
    assert {s["id"] for s in target.list_sessions()} == {
        source["root"],
        source["fork"],
    }


def test_a_run_that_finished_after_export_is_updated(source, target):
    lines = lines_of(source)
    import_(target, lines)
    target.set_status(source["root"], "running")

    result = import_(target, lines)

    assert result.status_updated == 1
    assert target.get_session(source["root"])["status"] == "complete"


# -- all of it, or none of it ---------------------------------------------------


def test_a_rejected_file_leaves_the_store_untouched(source, target):
    """Rejected on validation, before the transaction opens."""
    import_(target, lines_of(source, [source["root"]]))
    before = contents(target)

    rows = [json.loads(line) for line in lines_of(source)]
    victim = next(r for r in rows if r["kind"] == "step")
    victim["output"] = {"content": "tampered"}

    with pytest.raises(ExportFormatError):
        import_(target, [dump_line(r) for r in rows])

    assert contents(target) == before


def test_a_write_failing_partway_rolls_back(source, target, monkeypatch):
    """The transaction itself, with a failure injected past validation."""
    import retrial.transfer as transfer

    real = transfer._insert_step
    calls = {"n": 0}

    def explode(conn, step):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("disk full")
        return real(conn, step)

    monkeypatch.setattr(transfer, "_insert_step", explode)
    before = contents(target)

    with pytest.raises(RuntimeError, match="disk full"):
        import_(target, lines_of(source))

    assert contents(target) == before
    assert target.list_sessions() == []


def test_the_store_still_works_after_a_rolled_back_import(source, target, monkeypatch):
    """A rollback that left the connection wedged would be its own bug."""
    import retrial.transfer as transfer

    def explode(conn, step):
        raise RuntimeError("disk full")

    monkeypatch.setattr(transfer, "_insert_step", explode)
    with pytest.raises(RuntimeError):
        import_(target, lines_of(source))

    monkeypatch.undo()
    result = import_(target, lines_of(source))
    assert result.sessions_added == 2


# -- what the trace can still do ------------------------------------------------


def test_diff_and_trajectory_agree_on_both_sides(source, target):
    """The actual promise: the imported trace works, not just parses."""
    import_(target, lines_of(source))

    here = trajectory(target, source["fork"])
    there = trajectory(source["store"], source["fork"])
    assert [e["sha"] for e in here] == [e["sha"] for e in there]
    assert [e["origin"] for e in here] == [e["origin"] for e in there]

    mine = diff(target, source["root"], source["fork"])
    theirs = diff(source["store"], source["root"], source["fork"])
    assert mine["identical"] == theirs["identical"]
    assert mine["divergence"] == theirs["divergence"]
    assert mine["final"] == theirs["final"]
    def shape(blocks):
        return [
            (b["tag"], [e["sha"] for e in b["a"]], [e["sha"] for e in b["b"]])
            for b in blocks
        ]

    assert shape(mine["blocks"]) == shape(theirs["blocks"])


def test_an_imported_fork_can_be_forked_again(source, target, opening):
    """It arrives as a first-class session, not a read-only artifact."""
    import_(target, lines_of(source))
    agent = record(session_name="downstream", store=target)(raw_agent)
    tool = next(
        s for s in target.steps_for(source["fork"]) if s["step_type"] == "tool_call"
    )

    again = fork(
        from_sha=tool["sha"],
        edit={**PRICE_999, "value": json.dumps({"flight_price": 250})},
        agent=agent,
        store=target,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    assert target.get_session(again)["parent_session_id"] == source["fork"]
    assert trajectory(target, again)


# -- accepting a parsed document ------------------------------------------------


def test_a_parsed_document_can_be_imported_directly(source, target):
    """So a caller that already validated does not parse twice."""
    result = import_(target, parse_document(lines_of(source)))
    assert result.sessions_added == 2
