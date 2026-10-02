"""Moving traces between stores.

`portable.py` knows the file format; this knows the store. Ancestors travel with an exported
session by default, so it arrives forkable and diffable.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from .errors import ExportFormatError, NotFound, SchemaVersionError
from .portable import (
    FORMAT_VERSION,
    Document,
    dump_line,
    header_row,
    parse_document,
    session_row,
    step_row,
)
from .serialize import canonical_json
from .sha import compute_sha
from .storage import SCHEMA_VERSION, Store, schema_version
from .types import ExportSession, ExportStep, Session, SessionStatus


def export(
    store: Store,
    session_ids: Sequence[str] | None = None,
    ancestors: bool = True,
    requires: Iterable[str] = (),
) -> Iterator[str]:
    """Emit `session_ids` (default: the whole store) as export-file lines.

    Lazy, but ids are validated eagerly. Ancestors are included unless `ancestors=False`;
    descendants never are.
    """
    selected = _select(store, session_ids, ancestors)

    def lines() -> Iterator[str]:
        yield dump_line(
            header_row(schema=schema_version(store.conn), requires=requires)
        )
        for session in selected:
            yield dump_line(session_row(session))
        # Read each session's steps only when its block is reached.
        for session in selected:
            for step in store.steps_for(session["id"]):
                yield dump_line(step_row(step))

    return lines()


def _select(
    store: Store, session_ids: Sequence[str] | None, ancestors: bool
) -> list[Session]:
    """The sessions to emit, parents always before children."""
    if session_ids is None:
        wanted = [s["id"] for s in store.list_sessions()]
        ancestors = True  # a whole-store export is closed by definition
    else:
        wanted = list(dict.fromkeys(session_ids))

    # Eagerly, so a bad id raises from `export()` itself, before any output.
    emitted: dict[str, Session] = {}
    for session_id in wanted:
        for session in _chain(store, session_id, ancestors):
            emitted.setdefault(session["id"], session)
    return list(emitted.values())


# -- import: the validation pass ------------------------------------------------
#
# Nothing here writes. It decides what would happen and refuses if any of it is wrong.

#: Features an importer must understand by name. Empty at format v1.
KNOWN_FEATURES: frozenset[str] = frozenset()

#: One entry per shipped format version, mapping it forward. Empty at v1.
_TRANSLATORS: dict[int, Callable[[Document], Document]] = {}

#: Terminal states. A session may advance into one on import, never out.
_TERMINAL: frozenset[str] = frozenset({"complete", "failed"})


@dataclass
class ImportPlan:
    """What an import would do. Produced by `validate`, applied by `import_`."""

    new_sessions: list[ExportSession] = field(default_factory=list)
    new_steps: list[ExportStep] = field(default_factory=list)
    #: Already present with identical content.
    skipped_sessions: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    #: (session_id, new status) for runs that finished after they were sent.
    status_updates: list[tuple[str, SessionStatus]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.new_sessions or self.new_steps or self.status_updates)


def validate(store: Store, document: Document, path: str | None = None) -> ImportPlan:
    """Decide what importing `document` into `store` would do, or refuse.

    Refuses rather than merges whenever the file and the store disagree.
    """
    document, warnings = _translate(document, path)
    plan = ImportPlan(warnings=warnings)
    _check_schema(store, document, path)

    known_sessions = {s["id"] for s in document.sessions}
    for session in document.sessions:
        _plan_session(store, session, document, plan, path)

    # One read per session: its steps are contiguous in the file.
    local_steps: dict[str, dict[int, str]] = {}
    for step in document.steps:
        session_id = step["session_id"]
        if session_id not in local_steps:
            local_steps[session_id] = {
                s["step_number"]: s["sha"] for s in store.steps_for(session_id)
            }
        _plan_step(step, local_steps[session_id], document, plan, path)

    _check_parents(store, document, known_sessions, path)
    return plan


def _bad(
    message: str, document: Document, key: str | None = None, path: str | None = None
) -> ExportFormatError:
    return ExportFormatError(
        message, line=document.line_of(key) if key else None, path=path
    )


def _translate(document: Document, path: str | None) -> tuple[Document, list[str]]:
    """Bring an older file up to FORMAT_VERSION, or refuse an unreadable one.

    Forward only. A newer file is accepted only if it `requires` nothing unknown.
    """
    found = document.header["format"]

    if found > FORMAT_VERSION:
        unknown = sorted(set(document.header["requires"]) - KNOWN_FEATURES)
        if unknown:
            raise ExportFormatError(
                f"file is format v{found} and requires {', '.join(unknown)}, which "
                f"this retrial (format v{FORMAT_VERSION}) does not understand. "
                "Upgrade with `pip install -U retrial`.",
                line=1,
                path=path,
            )
        return document, [
            f"file is format v{found}, newer than this retrial's v{FORMAT_VERSION}. "
            "It declares no features this version must understand, so it is being "
            "read anyway - anything unrecognized in it is ignored."
        ]

    while found < FORMAT_VERSION:
        translate = _TRANSLATORS.get(found)
        if translate is None:
            raise ExportFormatError(
                f"file is format v{found}, and this retrial has no translation for "
                "it. Every format retrial has shipped should be readable; this is "
                "a bug.",
                line=1,
                path=path,
            )
        document = translate(document)
        found = document.header["format"]

    return document, []


def _check_schema(store: Store, document: Document, path: str | None) -> None:
    """The rows themselves must come from a schema this store can hold."""
    found = document.header["schema"]
    if found > SCHEMA_VERSION:
        raise SchemaVersionError(path or "<export>", found, SCHEMA_VERSION)
    if found > schema_version(store.conn):
        # Cannot happen through the CLI, since opening a Store upgrades it.
        raise ExportFormatError(
            f"file carries schema v{found} but this store is at "
            f"v{schema_version(store.conn)}",
            path=path,
        )


def _plan_session(
    store: Store,
    session: ExportSession,
    document: Document,
    plan: ImportPlan,
    path: str | None,
) -> None:
    try:
        existing = store.get_session(session["id"])
    except NotFound:
        plan.new_sessions.append(session)
        return

    # Present already, usually because this session is coming home.
    differing = _disagreements(existing, session)
    if differing:
        raise _bad(
            f"session {session['id']} already exists here with a different "
            f"{', '.join(differing)}. These are two different runs claiming one "
            "id; importing would merge them. Import into a separate store with "
            "--db instead.",
            document,
            session["id"],
            path,
        )

    if existing["status"] != session["status"]:
        # A run may finish after it was exported. Advancing is an update; anything else is not.
        if existing["status"] == "running" and session["status"] in _TERMINAL:
            plan.status_updates.append((session["id"], session["status"]))
        else:
            raise _bad(
                f"session {session['id']} is {existing['status']} here but "
                f"{session['status']} in the file; a finished run does not change "
                "its outcome",
                document,
                session["id"],
                path,
            )

    plan.skipped_sessions.append(session["id"])


def _disagreements(existing: Session, incoming: ExportSession) -> list[str]:
    """Fields that must agree. Status is excluded - see `_plan_session`."""
    local_edit = json.loads(existing["edit_json"]) if existing["edit_json"] else None
    pairs = {
        "name": (existing["name"], incoming["name"]),
        "parent_session_id": (
            existing["parent_session_id"],
            incoming["parent_session_id"],
        ),
        "parent_sha": (existing["parent_sha"], incoming["parent_sha"]),
        "forked_at_step": (existing["forked_at_step"], incoming["forked_at_step"]),
        "edit": (local_edit, incoming["edit"]),
        "created_at": (existing["created_at"], incoming["created_at"]),
    }
    return [name for name, (a, b) in pairs.items() if a != b]


def _plan_step(
    step: ExportStep,
    local: dict[int, str],
    document: Document,
    plan: ImportPlan,
    path: str | None,
) -> None:
    # Check the sha first: recomputing it proves the content is what was exported.
    recomputed = compute_sha(
        step["session_id"],
        step["step_number"],
        step["step_type"],
        step["input"],
        step["output"],
    )
    if recomputed != step["sha"]:
        raise _bad(
            f"step {step['sha'][:12]} does not match its content (computed "
            f"{recomputed[:12]}). The file has been altered since it was "
            "exported, and a trace that changed in transit is not a recording.",
            document,
            step["sha"],
            path,
        )

    existing = local.get(step["step_number"])
    if existing is None:
        plan.new_steps.append(step)
    elif existing == step["sha"]:
        plan.skipped_steps.append(step["sha"])
    else:
        raise _bad(
            f"session {step['session_id']} already has a different step "
            f"{step['step_number']} here ({existing[:12]}, not "
            f"{step['sha'][:12]}). Import into a separate store with --db.",
            document,
            step["sha"],
            path,
        )


def _check_parents(
    store: Store, document: Document, in_file: set[str], path: str | None
) -> None:
    """Every named parent must exist, in the file or already in the store."""
    for session in document.sessions:
        parent = session["parent_session_id"]
        if parent is None or parent in in_file:
            continue
        try:
            store.get_session(parent)
        except NotFound as exc:
            raise _bad(
                f"session {session['id']} names parent {parent}, which is neither "
                "in this file nor in this store. Export the parent too - without "
                "it the fork cannot be diffed or replayed.",
                document,
                session["id"],
                path,
            ) from exc


# -- import: the write pass -----------------------------------------------------


@dataclass
class ImportResult:
    """What an import did. Every count is a row, not a file."""

    sessions_added: int = 0
    steps_added: int = 0
    #: Already present with identical content - the file coming home.
    sessions_skipped: int = 0
    steps_skipped: int = 0
    status_updated: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def changed_nothing(self) -> bool:
        return not (self.sessions_added or self.steps_added or self.status_updated)


def import_(
    store: Store, source: Iterable[str] | Document, path: str | None = None
) -> ImportResult:
    """Read an export into `store`. All of it, or none of it.

    `source` is the file's lines or a parsed `Document`. Everything is validated first, then
    written in one transaction.
    """
    document = source if isinstance(source, Document) else parse_document(source, path)
    plan = validate(store, document, path)

    result = ImportResult(
        sessions_skipped=len(plan.skipped_sessions),
        steps_skipped=len(plan.skipped_steps),
        warnings=list(plan.warnings),
    )
    if plan.is_empty:
        return result

    with store.transaction() as conn:
        # Sessions first and ancestors before forks, so foreign keys resolve as rows land.
        for session in plan.new_sessions:
            _insert_session(conn, session)
            result.sessions_added += 1
        for step in plan.new_steps:
            _insert_step(conn, step)
            result.steps_added += 1
        for session_id, status in plan.status_updates:
            conn.execute(
                "UPDATE sessions SET status = ? WHERE id = ?", (status, session_id)
            )
            result.status_updated += 1

    return result


def _insert_session(conn: sqlite3.Connection, session: ExportSession) -> None:
    conn.execute(
        "INSERT INTO sessions (id, name, parent_session_id, parent_sha, "
        "forked_at_step, edit_json, created_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session["id"],
            session["name"],
            session["parent_session_id"],
            session["parent_sha"],
            session["forked_at_step"],
            json.dumps(session["edit"]) if session["edit"] is not None else None,
            session["created_at"],
            session["status"],
        ),
    )


def _insert_step(conn: sqlite3.Connection, step: ExportStep) -> None:
    """Written directly, not through `Store.add_step`, which commits per row and mints a fresh sha
    and created_at.
    """
    conn.execute(
        "INSERT INTO steps (sha, session_id, step_number, step_type, "
        "input_json, output_json, tokens_used, cost_usd, duration_ms, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            step["sha"],
            step["session_id"],
            step["step_number"],
            step["step_type"],
            canonical_json(step["input"]),
            canonical_json(step["output"]),
            step["tokens_used"],
            step["cost_usd"],
            step["duration_ms"],
            step["created_at"],
        ),
    )


def _chain(store: Store, session_id: str, ancestors: bool) -> list[Session]:
    """A session and its ancestors, root first."""
    session = store.get_session(session_id)
    if not ancestors:
        return [session]

    chain = [session]
    seen = {session_id}
    while True:
        parent_id = chain[-1]["parent_session_id"]
        if parent_id is None:
            break
        if parent_id in seen:
            # fork() writes a parent before its child, so a cycle means the store is damaged.
            raise ExportFormatError(
                f"session {session_id} has a cyclic parent chain through "
                f"{parent_id}; the store is inconsistent and cannot be exported"
            )
        try:
            chain.append(store.get_session(parent_id))
        except NotFound as exc:
            raise ExportFormatError(
                f"session {chain[-1]['id']} names parent {parent_id}, which is not "
                "in this store. The trace is incomplete and would not be usable "
                "where it landed - export the parent too, or pass ancestors=False "
                "to send this session alone."
            ) from exc
        seen.add(parent_id)

    chain.reverse()
    return chain
