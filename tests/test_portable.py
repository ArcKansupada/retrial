"""The export file format on its own, with no store involved.

Covers the two ordering guarantees (header first, sessions before what references them) and the
field checks.
"""

import json

import pytest

from retrial.errors import ExportFormatError
from retrial.portable import (
    FORMAT_VERSION,
    dump_document,
    dump_line,
    header_row,
    parse_document,
    parse_line,
    session_row,
    step_row,
)

# -- fixtures as literals, so a format change shows up here as an edit ---------


def a_session(session_id="s_root00001", parent=None, **over):
    row = {
        "kind": "session",
        "id": session_id,
        "name": "booking-agent",
        "parent_session_id": parent,
        "parent_sha": "9f2c" * 16 if parent else None,
        "forked_at_step": 2 if parent else None,
        "edit": None,
        "created_at": 1753660000.0,
        "status": "complete",
    }
    row.update(over)
    return row


def a_step(session_id="s_root00001", number=0, **over):
    row = {
        "kind": "step",
        "sha": f"{number:064x}",
        "session_id": session_id,
        "step_number": number,
        "step_type": "model_call",
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "output": {"content": [{"type": "text", "text": "hello"}]},
        "tokens_used": 812,
        "cost_usd": 0.0041,
        "duration_ms": 1830.4,
        "created_at": 1753660001.0,
    }
    row.update(over)
    return row


def document(*rows, header=None):
    lines = [dump_line(header or header_row(schema=1))]
    lines += [dump_line(r) for r in rows]
    return lines


# -- round trip ----------------------------------------------------------------


def test_a_row_survives_dump_and_parse():
    for row in (header_row(schema=1), a_session(), a_step()):
        assert parse_line(dump_line(row), 1) == row


def test_dump_line_is_one_line():
    line = dump_line(a_step())
    assert line.endswith("\n")
    assert line.count("\n") == 1


def test_two_dumps_of_the_same_row_are_byte_identical():
    """So re-exporting an unchanged trace is an empty diff, not noise."""
    assert dump_line(a_session()) == dump_line(dict(reversed(list(a_session().items()))))


def test_a_whole_document_round_trips():
    rows = [a_session(), a_session("s_fork00002", parent="s_root00001"),
            a_step(number=0), a_step(number=1),
            a_step("s_fork00002", number=2)]
    header, sessions, steps = parse_document(document(*rows))

    assert header["format"] == FORMAT_VERSION
    assert [s["id"] for s in sessions] == ["s_root00001", "s_fork00002"]
    assert [s["step_number"] for s in steps] == [0, 1, 2]


def test_dump_document_orders_header_sessions_steps():
    lines = list(dump_document(header_row(schema=1), [a_session()], [a_step()]))
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["header", "session", "step"]


def test_blank_lines_are_skipped():
    """A file that gained a trailing newline in transit is the same file."""
    lines = document(a_session(), a_step())
    lines.insert(1, "\n")
    lines.append("")
    header, sessions, steps = parse_document(lines)
    assert len(sessions) == 1 and len(steps) == 1


# -- the store's records convert ------------------------------------------------


def test_session_row_parses_edit_json():
    """An escaped JSON string inside JSON defeats reading the file."""
    stored = {
        "id": "s_root00001",
        "name": "x",
        "parent_session_id": None,
        "parent_sha": None,
        "forked_at_step": None,
        "edit_json": json.dumps({"type": "patch", "ops": [{"op": "replace"}]}),
        "created_at": 1.0,
        "status": "complete",
    }
    assert session_row(stored)["edit"] == {"type": "patch", "ops": [{"op": "replace"}]}


def test_session_row_handles_no_edit():
    stored = dict(a_session(), edit_json=None)
    del stored["kind"], stored["edit"]
    assert session_row(stored)["edit"] is None


def test_step_row_drops_the_local_rowid():
    """`id` is an autoincrement rowid - meaningless in another store."""
    stored = dict(a_step(), id=17)
    del stored["kind"]
    assert "id" not in step_row(stored)


# -- ordering is enforced, not assumed -----------------------------------------


def test_the_header_must_be_first():
    with pytest.raises(ExportFormatError, match="expected a header on line 1"):
        parse_document([dump_line(a_session())])


def test_a_second_header_is_refused():
    lines = document(a_session())
    lines.append(dump_line(header_row(schema=1)))
    with pytest.raises(ExportFormatError, match="second header"):
        parse_document(lines)


def test_a_session_after_a_step_is_refused():
    """Sessions first is what lets an importer create them in one pass."""
    with pytest.raises(ExportFormatError, match="session row after a step row"):
        parse_document(document(a_session(), a_step(), a_session("s_late000002")))


def test_a_parent_defined_after_its_fork_is_refused():
    with pytest.raises(ExportFormatError, match="defined later in the file"):
        parse_document(
            document(a_session("s_fork00002", parent="s_root00001"), a_session())
        )


def test_a_parent_absent_from_the_file_is_allowed():
    """`--no-ancestors` produces this; whether the parent exists is import's question."""
    header, sessions, steps = parse_document(
        document(a_session("s_fork00002", parent="s_root00001"))
    )
    assert sessions[0]["parent_session_id"] == "s_root00001"


def test_a_duplicate_session_is_refused():
    with pytest.raises(ExportFormatError, match="appears twice"):
        parse_document(document(a_session(), a_session()))


def test_a_step_for_an_undeclared_session_is_refused():
    with pytest.raises(ExportFormatError, match="never defines"):
        parse_document(document(a_session(), a_step("s_ghost00003")))


def test_step_numbers_must_advance():
    # Distinct shas, or the duplicate-sha check fires first.
    with pytest.raises(ExportFormatError, match="does not advance"):
        parse_document(
            document(
                a_session(), a_step(number=1), a_step(number=1, sha="f" * 64)
            )
        )


def test_a_repeated_step_is_refused():
    """Two rows sharing a sha are the same step written twice."""
    with pytest.raises(ExportFormatError, match="appears twice"):
        parse_document(document(a_session(), a_step(number=0), a_step(number=0)))


def test_a_sessions_steps_must_be_contiguous():
    """Interleaved blocks would make a streaming importer buffer everything."""
    rows = [a_session(), a_session("s_fork00002", parent="s_root00001"),
            a_step(number=0), a_step("s_fork00002", number=1), a_step(number=2)]
    with pytest.raises(ExportFormatError, match="steps in more than one block"):
        parse_document(document(*rows))


# -- malformed rows ------------------------------------------------------------


def test_the_line_number_is_reported():
    """One bad line in a thousand needs a location, not just a description."""
    lines = document(a_session(), a_step(), a_step(number=1))
    lines[2] = "{not json\n"
    with pytest.raises(ExportFormatError) as excinfo:
        parse_document(lines, path="trace.jsonl")

    assert excinfo.value.line == 3
    assert "trace.jsonl:3" in str(excinfo.value)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ExportFormatError, match="unknown row kind"):
        parse_line(json.dumps({"kind": "comment"}) + "\n", 1)


def test_a_row_without_a_kind_is_refused():
    with pytest.raises(ExportFormatError, match="no 'kind' field"):
        parse_line(json.dumps({"id": "s_x"}) + "\n", 1)


def test_a_non_object_line_is_refused():
    with pytest.raises(ExportFormatError, match="expected a JSON object"):
        parse_line("[1, 2, 3]\n", 1)


def test_a_missing_field_is_named():
    row = a_step()
    del row["output"]
    with pytest.raises(ExportFormatError, match="missing output"):
        parse_line(dump_line(row), 1)


def test_a_wrong_type_is_refused_at_the_boundary():
    """A string step_number would otherwise sort lexicographically later on."""
    with pytest.raises(ExportFormatError, match="step_number must be int"):
        parse_line(dump_line(a_step(step_number="1")), 1)


def test_a_bool_is_not_an_int():
    """True is an int to Python; in a step_number it is a malformed file."""
    with pytest.raises(ExportFormatError, match="must be an int, got a bool"):
        parse_line(dump_line(a_step(step_number=True)), 1)


def test_an_integral_float_is_accepted():
    """JSON has one number type: 1753660001 and ...0.0 are the same value."""
    row = parse_line(dump_line(a_step(created_at=1753660001)), 1)
    assert row["created_at"] == 1753660001.0
    assert isinstance(row["created_at"], float)


def test_a_null_in_a_required_field_is_refused():
    with pytest.raises(ExportFormatError, match="sha must not be null"):
        parse_line(dump_line(a_step(sha=None)), 1)


def test_an_optional_field_may_be_null():
    """Cost is None when unpriced - never an estimate. That must survive."""
    row = parse_line(dump_line(a_step(cost_usd=None, tokens_used=None)), 1)
    assert row["cost_usd"] is None


def test_requires_must_be_a_list_of_strings():
    header = dict(header_row(schema=1), requires="everything")
    with pytest.raises(ExportFormatError, match="list of strings"):
        parse_line(dump_line(header), 1)


def test_an_empty_file_is_refused():
    with pytest.raises(ExportFormatError, match="empty"):
        parse_document([])


def test_a_file_of_only_blank_lines_is_refused():
    with pytest.raises(ExportFormatError, match="no header row"):
        parse_document(["\n", "  \n"])
