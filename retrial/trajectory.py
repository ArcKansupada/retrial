"""Materializing a full trajectory from root to tip.

A fork stores only the steps it re-executed; the prefix lives in its parent, so the full path is
assembled by walking the parent chain. Each entry is tagged:

    origin="replayed"  came from an ancestor; no model call was made
    origin="live"      this session re-executed it
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

from .types import JSON, EditProvenance, Origin, Session, Step, TrajectoryEntry

if TYPE_CHECKING:
    from .storage import Store


def trajectory(store: Store, session_id: str) -> list[TrajectoryEntry]:
    """The full path of steps from the root run through to this session's tip."""
    return _walk(store, session_id, set())


def _walk(store: Store, session_id: str, seen: set[str]) -> list[TrajectoryEntry]:
    if session_id in seen:
        raise ValueError(f"cycle in session ancestry at {session_id}")
    seen = seen | {session_id}

    session = store.get_session(session_id)
    own = [_entry(s, origin="live") for s in store.steps_for(session_id)]

    parent_id = session["parent_session_id"]
    if not parent_id:
        return own

    ancestors = _walk(store, parent_id, seen)
    cut = _index_of(ancestors, parent_id, session["forked_at_step"])
    if cut is None:
        raise ValueError(
            f"session {session_id} forked from step {session['forked_at_step']} of "
            f"{parent_id}, but that step is not in the parent's trajectory"
        )

    forked_from = ancestors[cut]
    for entry in ancestors[: cut + 1]:
        entry["origin"] = "replayed"

    if forked_from["step_type"] == "tool_call":
        # The fork resumed after this tool call with an edited output. Show what the fork saw.
        forked_from = cast(
            TrajectoryEntry,
            dict(
                forked_from,
                output=_as_seen_by(store, session_id, forked_from),
                edited=True,
                edit=_edit_of(session),
            ),
        )
        prefix = ancestors[:cut] + [forked_from]
    else:
        # A model_call fork re-runs that call, so the parent's version is not on the path.
        prefix = ancestors[:cut]
        if own:
            own[0] = cast(
                TrajectoryEntry,
                dict(own[0], edited=True, edit=_edit_of(session)),
            )

    return prefix + own


def _as_seen_by(store: Store, session_id: str, forked_from: TrajectoryEntry) -> JSON:
    """Recover the tool output the fork actually resumed with.

    Read from the fork's first recorded model input, which works for callback edits too.
    """
    own = store.steps_for(session_id)
    if not own or own[0]["step_type"] != "model_call":
        return forked_from["output"]

    seed = own[0]["input"].get("messages", [])
    blocks = {
        block["tool_use_id"]: block
        for message in seed
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and "tool_use_id" in block
    }
    return [
        copy.deepcopy(blocks.get(entry.get("tool_use_id"), entry))
        for entry in forked_from["output"]
    ]


def _edit_of(session: Session) -> EditProvenance | None:
    import json

    if not session["edit_json"]:
        return None
    try:
        return cast(EditProvenance, json.loads(session["edit_json"]))
    except ValueError:
        return None


def _index_of(
    entries: list[TrajectoryEntry], session_id: str, step_number: int | None
) -> int | None:
    for i, entry in enumerate(entries):
        if entry["session_id"] == session_id and entry["step_number"] == step_number:
            return i
    return None


def _entry(step: Step, origin: Origin) -> TrajectoryEntry:
    return {
        "sha": step["sha"],
        "session_id": step["session_id"],
        "step_number": step["step_number"],
        "step_type": step["step_type"],
        "input": step["input"],
        "output": step["output"],
        "tokens_used": step.get("tokens_used"),
        "cost_usd": step.get("cost_usd"),
        "duration_ms": step.get("duration_ms"),
        "origin": origin,
        "edited": False,
        "edit": None,
    }
