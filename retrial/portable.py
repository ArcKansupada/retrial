"""The on-disk export format: reading and writing rows.

JSONL, one object per line, discriminated by `kind`. This module knows the format only, not the
store.

Two ordering rules are guaranteed and enforced by `parse_document`: the header is line 1, and
every session appears before any step or fork that references it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

from .errors import ExportFormatError
from .types import (
    ExportHeader,
    ExportRow,
    ExportSession,
    ExportStep,
    Session,
    Step,
)

#: Version of the file layout. Bump when the envelope changes and add a translator.
FORMAT_VERSION = 1

_KINDS = ("header", "session", "step")

_SESSION_FIELDS = (
    "id",
    "name",
    "parent_session_id",
    "parent_sha",
    "forked_at_step",
    "edit",
    "created_at",
    "status",
)

_STEP_FIELDS = (
    "sha",
    "session_id",
    "step_number",
    "step_type",
    "input",
    "output",
    "tokens_used",
    "cost_usd",
    "duration_ms",
    "created_at",
)


# -- writing -------------------------------------------------------------------


def header_row(schema: int, requires: Iterable[str] = ()) -> ExportHeader:
    from . import __version__

    return {
        "kind": "header",
        "format": FORMAT_VERSION,
        "schema": schema,
        "requires": list(requires),
        "exported_at": time.time(),
        "retrial": __version__,
    }


def session_row(session: Session) -> ExportSession:
    """A stored session as a portable row. `edit_json` becomes a parsed `edit`."""
    raw = session["edit_json"]
    return {
        "kind": "session",
        "id": session["id"],
        "name": session["name"],
        "parent_session_id": session["parent_session_id"],
        "parent_sha": session["parent_sha"],
        "forked_at_step": session["forked_at_step"],
        "edit": json.loads(raw) if raw else None,
        "created_at": session["created_at"],
        "status": session["status"],
    }


def step_row(step: Step) -> ExportStep:
    """A stored step as a portable row. `id` is dropped - see types.ExportStep."""
    return {
        "kind": "step",
        "sha": step["sha"],
        "session_id": step["session_id"],
        "step_number": step["step_number"],
        "step_type": step["step_type"],
        "input": step["input"],
        "output": step["output"],
        "tokens_used": step["tokens_used"],
        "cost_usd": step["cost_usd"],
        "duration_ms": step["duration_ms"],
        "created_at": step["created_at"],
    }


def dump_line(row: ExportRow) -> str:
    """One row as one line, newline included.

    `sort_keys` makes two exports of the same trace byte-identical.
    """
    return json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"


def dump_document(
    header: ExportHeader,
    sessions: Iterable[ExportSession],
    steps: Iterable[ExportStep],
) -> Iterator[str]:
    """Header, then sessions, then steps. Lazy, so a large store streams."""
    yield dump_line(header)
    for session in sessions:
        yield dump_line(session)
    for step in steps:
        yield dump_line(step)


# -- reading -------------------------------------------------------------------


@dataclass
class Document:
    """A parsed export file. Unpacks as `header, sessions, steps`.

    Also records which line each row came from, for later validation errors.
    """

    header: ExportHeader
    sessions: list[ExportSession]
    steps: list[ExportStep]
    #: Session id or step sha -> line number. Both are unique within a file.
    lines: dict[str, int] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.header, self.sessions, self.steps))

    def line_of(self, key: str) -> int | None:
        return self.lines.get(key)


def parse_line(text: str, line: int, path: str | None = None) -> ExportRow:
    """One line into a row, type-checking each field, or refuse with the line number."""

    def bad(message: str) -> ExportFormatError:
        return ExportFormatError(message, line=line, path=path)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise bad(f"not valid JSON ({exc.msg})") from exc

    if not isinstance(obj, dict):
        raise bad(f"expected a JSON object, got {type(obj).__name__}")

    kind = obj.get("kind")
    if kind not in _KINDS:
        raise bad(
            f"unknown row kind {kind!r}; expected one of {', '.join(_KINDS)}"
            if kind is not None
            else "row has no 'kind' field"
        )

    if kind == "header":
        _require(obj, ("format", "schema"), bad)
        _check_type(obj, "format", int, bad)
        _check_type(obj, "schema", int, bad)
        obj.setdefault("requires", [])
        obj.setdefault("exported_at", 0.0)
        obj.setdefault("retrial", "unknown")
        if not isinstance(obj["requires"], list) or not all(
            isinstance(r, str) for r in obj["requires"]
        ):
            raise bad("'requires' must be a list of strings")
        return cast(ExportHeader, obj)

    if kind == "session":
        _require(obj, _SESSION_FIELDS, bad)
        _check_type(obj, "id", str, bad)
        _check_type(obj, "name", str, bad, optional=True)
        _check_type(obj, "parent_session_id", str, bad, optional=True)
        _check_type(obj, "parent_sha", str, bad, optional=True)
        _check_type(obj, "forked_at_step", int, bad, optional=True)
        _check_type(obj, "created_at", float, bad)
        _check_type(obj, "status", str, bad)
        return cast(ExportSession, obj)

    _require(obj, _STEP_FIELDS, bad)
    _check_type(obj, "sha", str, bad)
    _check_type(obj, "session_id", str, bad)
    _check_type(obj, "step_number", int, bad)
    _check_type(obj, "step_type", str, bad)
    _check_type(obj, "tokens_used", int, bad, optional=True)
    _check_type(obj, "cost_usd", float, bad, optional=True)
    _check_type(obj, "duration_ms", float, bad, optional=True)
    _check_type(obj, "created_at", float, bad)
    return cast(ExportStep, obj)


def parse_document(lines: Iterable[str], path: str | None = None) -> Document:
    """A whole file into its three parts, with the ordering rules enforced.

    Blank lines are skipped. Does not check shas or touch a store; see `import`.
    """
    header: ExportHeader | None = None
    sessions: list[ExportSession] = []
    steps: list[ExportStep] = []
    seen_sessions: set[str] = set()
    seen_shas: set[str] = set()
    session_lines: list[int] = []
    row_lines: dict[str, int] = {}
    number = 0

    for number, text in enumerate(lines, start=1):
        if not text.strip():
            continue

        def bad(message: str, _n: int = number) -> ExportFormatError:
            return ExportFormatError(message, line=_n, path=path)

        row = parse_line(text, number, path)
        kind = row["kind"]

        if header is None:
            if kind != "header":
                raise bad(f"expected a header on line 1, got a {kind} row")
            header = cast(ExportHeader, row)
            continue

        if kind == "header":
            raise bad("a second header row; an export file has exactly one")

        if kind == "session":
            if steps:
                # Sessions come first so an importer can create them before any step.
                raise bad("a session row after a step row; sessions come first")
            session = cast(ExportSession, row)
            if session["id"] in seen_sessions:
                raise bad(f"session {session['id']} appears twice")
            seen_sessions.add(session["id"])
            sessions.append(session)
            session_lines.append(number)
            row_lines[session["id"]] = number
            continue

        step = cast(ExportStep, row)
        if step["sha"] in seen_shas:
            # Two rows sharing a sha are the same step written twice.
            raise bad(f"step {step['sha'][:12]} appears twice")
        if step["session_id"] not in seen_sessions:
            raise bad(
                f"step {step['sha'][:12]} belongs to session {step['session_id']}, "
                "which this file never defines"
            )
        if steps:
            previous = steps[-1]
            same = previous["session_id"] == step["session_id"]
            if same and step["step_number"] <= previous["step_number"]:
                raise bad(
                    f"step_number {step['step_number']} does not advance on "
                    f"{previous['step_number']} within session {step['session_id']}"
                )
            if not same and any(s["session_id"] == step["session_id"] for s in steps):
                raise bad(
                    f"session {step['session_id']} has steps in more than one "
                    "block; a session's steps must be contiguous"
                )
        steps.append(step)
        seen_shas.add(step["sha"])
        row_lines[step["sha"]] = number

    if header is None:
        raise ExportFormatError(
            "file is empty" if number == 0 else "file has no header row",
            path=path,
        )

    # Deferred: a file exported with --no-ancestors may name parents it does not contain.
    position = {s["id"]: i for i, s in enumerate(sessions)}
    for index, session in enumerate(sessions):
        parent = session["parent_session_id"]
        if parent is not None and position.get(parent, -1) > index:
            raise ExportFormatError(
                f"session {session['id']} names parent {parent}, which is defined "
                "later in the file; ancestors must come first",
                line=session_lines[index],
                path=path,
            )
    return Document(header, sessions, steps, row_lines)


# -- field checks --------------------------------------------------------------


#: Builds the error for the line being parsed.
_Bad = Callable[[str], ExportFormatError]


def _require(obj: dict[str, Any], fields: Iterable[str], bad: _Bad) -> None:
    missing = [f for f in fields if f not in obj]
    if missing:
        raise bad(f"{obj['kind']} row is missing {', '.join(missing)}")


def _check_type(
    obj: dict[str, Any], field: str, expected: type, bad: _Bad, optional: bool = False
) -> None:
    value = obj.get(field)
    if value is None:
        if optional:
            return
        raise bad(f"{field} must not be null")
    # A bool is an int to Python; reject it as a step_number.
    if expected is int and isinstance(value, bool):
        raise bad(f"{field} must be an int, got a bool")
    # JSON has one number type, so accept an integral float.
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        obj[field] = float(value)
        return
    if not isinstance(value, expected):
        raise bad(f"{field} must be {expected.__name__}, got {type(value).__name__}")
