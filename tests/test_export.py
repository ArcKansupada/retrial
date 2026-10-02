"""Exporting a trace so it still works where it lands.

The key test: a fork travels with its ancestors, or it cannot be diffed or walked.
"""

import json

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import fork, record
from retrial.errors import ExportFormatError, NotFound
from retrial.portable import parse_document
from retrial.transfer import export

PRICE_999 = {
    "op": "replace",
    "path": "/output/0/content",
    "value": json.dumps({"flight_price": 999}),
}


def tool_step(store, session_id):
    return next(s for s in store.steps_for(session_id) if s["step_type"] == "tool_call")


def make_fork(store, agent, session_id, edit=PRICE_999):
    return fork(
        from_sha=tool_step(store, session_id)["sha"],
        edit=edit,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )


@pytest.fixture
def recorded(store, opening):
    """A root run, and a fork of it - the shape worth sending someone."""
    agent = record(session_name="root-run", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))
    root = agent.last_session_id
    return {"root": root, "fork": make_fork(store, agent, root), "agent": agent}


def read(store, *args, **kwargs):
    return parse_document(list(export(store, *args, **kwargs)))


# -- what travels ---------------------------------------------------------------


def test_a_fork_travels_with_its_ancestors(store, recorded):
    """Without the parent the trace cannot be diffed or walked."""
    _, sessions, _ = read(store, [recorded["fork"]])

    assert [s["id"] for s in sessions] == [recorded["root"], recorded["fork"]]


def test_ancestors_come_first(store, recorded):
    """The format guarantees this ordering, so export must produce it."""
    _, sessions, _ = read(store, [recorded["fork"]])
    position = {s["id"]: i for i, s in enumerate(sessions)}

    for session in sessions:
        parent = session["parent_session_id"]
        if parent is not None:
            assert position[parent] < position[session["id"]]


def test_descendants_are_not_included(store, recorded):
    """Exporting a root must not hand over every experiment run on top of it."""
    _, sessions, _ = read(store, [recorded["root"]])

    assert [s["id"] for s in sessions] == [recorded["root"]]


def test_no_ancestors_sends_the_session_alone(store, recorded):
    _, sessions, _ = read(store, [recorded["fork"]], ancestors=False)

    assert [s["id"] for s in sessions] == [recorded["fork"]]
    # Provenance is kept even without the parent, so the receiver can look for it.
    assert sessions[0]["parent_session_id"] == recorded["root"]


def test_a_no_ancestors_export_is_still_a_parseable_file(store, recorded):
    """It names a parent it does not contain, which the format allows."""
    lines = list(export(store, [recorded["fork"]], ancestors=False))
    parse_document(lines)  # must not raise


def test_exporting_everything(store, recorded):
    _, sessions, steps = read(store)

    assert {s["id"] for s in sessions} == {recorded["root"], recorded["fork"]}
    assert steps


def test_the_same_session_asked_for_twice_appears_once(store, recorded):
    _, sessions, _ = read(store, [recorded["fork"], recorded["fork"]])

    assert len(sessions) == len({s["id"] for s in sessions})


def test_two_sessions_sharing_an_ancestor_emit_it_once(store, recorded):
    second = make_fork(
        store,
        recorded["agent"],
        recorded["root"],
        edit={**PRICE_999, "value": json.dumps({"flight_price": 1450})},
    )

    _, sessions, _ = read(store, [recorded["fork"], second])
    ids = [s["id"] for s in sessions]
    assert ids.count(recorded["root"]) == 1
    assert set(ids) == {recorded["root"], recorded["fork"], second}


# -- what the rows carry --------------------------------------------------------


def test_every_step_travels_with_its_session(store, recorded):
    _, sessions, steps = read(store, [recorded["fork"]])

    by_session = {s["id"]: 0 for s in sessions}
    for step in steps:
        by_session[step["session_id"]] += 1
    for session_id, count in by_session.items():
        assert count == len(store.steps_for(session_id))


def test_step_content_survives_verbatim(store, recorded):
    """A recording that changed in transit would not be a recording."""
    _, _, steps = read(store, [recorded["root"]])
    stored = store.steps_for(recorded["root"])

    # strict: a length mismatch means steps went missing.
    for row, original in zip(steps, stored, strict=True):
        assert row["sha"] == original["sha"]
        assert row["input"] == original["input"]
        assert row["output"] == original["output"]
        assert row["created_at"] == original["created_at"]


def test_the_forks_edit_provenance_survives(store, recorded):
    """`retrial log` shows WHAT changed, not just where. That must travel."""
    _, sessions, _ = read(store, [recorded["fork"]])
    forked = next(s for s in sessions if s["id"] == recorded["fork"])

    assert forked["edit"] is not None
    assert forked["parent_sha"] is not None
    assert forked["forked_at_step"] is not None


def test_the_header_reports_the_schema_it_came_from(store, recorded):
    header, _, _ = read(store, [recorded["root"]])

    assert header["format"] == 1
    assert header["schema"] == 1
    assert header["requires"] == []


def test_output_is_one_json_object_per_line(store, recorded):
    for line in export(store, [recorded["fork"]]):
        assert line.endswith("\n")
        assert isinstance(json.loads(line), dict)


# -- refusals -------------------------------------------------------------------


def test_an_unknown_session_is_refused_before_any_output(store, recorded):
    """A bad id raises from the `export()` call itself, before any output."""
    with pytest.raises(NotFound):
        export(store, ["s_nosuchid01"])


def test_a_parent_missing_from_the_store_is_refused(store, recorded):
    """The trace would not be usable where it landed, so say so here."""
    store.conn.execute("PRAGMA foreign_keys = OFF")
    store.conn.execute("DELETE FROM sessions WHERE id = ?", (recorded["root"],))
    store.conn.commit()

    with pytest.raises(ExportFormatError, match="not in this store"):
        list(export(store, [recorded["fork"]]))


def test_that_session_can_still_be_sent_alone(store, recorded):
    """The refusal above names this escape, so it has to work."""
    store.conn.execute("PRAGMA foreign_keys = OFF")
    store.conn.execute("DELETE FROM sessions WHERE id = ?", (recorded["root"],))
    store.conn.commit()

    _, sessions, _ = read(store, [recorded["fork"]], ancestors=False)
    assert [s["id"] for s in sessions] == [recorded["fork"]]


def test_a_cyclic_parent_chain_is_refused(store, recorded):
    """Only reachable through a damaged store, but it must not hang."""
    store.conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        (recorded["fork"], recorded["root"]),
    )
    store.conn.commit()

    with pytest.raises(ExportFormatError, match="cyclic parent chain"):
        list(export(store, [recorded["fork"]]))
