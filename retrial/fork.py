"""The replay / fork engine.

A fork re-enters the live agent loop. The message state after a tool call is the exact input to
the next model call, which the recorder captured, so a fork reads that array back, patches the
one recorded fact, verifies the patch landed, and hands it to the user's function. An
unverifiable patch raises.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from typing import Any

from .errors import IntegrationError, ReplayIntegrityError
from .patch import normalize_edit
from .record import _is_async, _pending
from .storage import Store
from .types import JSON, Agent, Edit, Step


def _running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _run_agent(agent: Agent, *args: Any, **kwargs: Any) -> Any:
    """Invoke the agent, awaiting it if it is async.

    An async agent runs under asyncio.run in a fresh loop.
    """
    if _is_async(agent):
        return asyncio.run(agent(*args, **kwargs))
    return agent(*args, **kwargs)


def fork(
    from_sha: str,
    edit: Edit = None,
    agent: Agent | None = None,
    store: Store | None = None,
    name: str | None = None,
    agent_args: Sequence[Any] = (),
    **agent_kwargs: Any,
) -> str:
    """Fork a recorded run at `from_sha` and resume it live.

        fork(
            from_sha="a1b2c3d",
            edit={"op": "replace", "path": "/output/0/content", "value": "..."},
            agent=run_agent,
        )

    `agent` must be the @record-decorated function from the original run and take the message
    state as its first argument. Returns the new session id; the original is never mutated.
    """
    if agent is None:
        raise IntegrationError(
            "fork() needs the agent function to re-execute. Real re-execution "
            "means calling your loop again - there is nothing to run without it."
        )

    # asyncio.run cannot nest, so refuse to fork an async agent from inside a running loop.
    if _is_async(agent) and _running_loop():
        raise IntegrationError(
            "fork() cannot drive an async agent from inside a running event loop: "
            "it re-executes via asyncio.run, which a running loop forbids. Call "
            "fork from synchronous code (the CLI does), or await a native async "
            "fork API - which is not shipped yet."
        )

    owns_store = store is None
    store = store or Store()
    try:
        step = store.get_step(from_sha)
        session = store.get_session(step["session_id"])

        apply_edit, provenance = normalize_edit(edit)
        edited = apply_edit(copy.deepcopy(_public(step)))
        if not isinstance(edited, dict):
            raise ReplayIntegrityError(
                f"edit callback returned {type(edited).__name__}; it must return "
                "the step dict"
            )

        seed = _seed_messages(store, step, edited)

        fork_session_id = store.create_session(
            name=name or f"{session['name']}-fork",
            parent_session_id=step["session_id"],
            parent_sha=step["sha"],
            forked_at_step=step["step_number"],
            edit=provenance,
        )

        # The decorator reads this so the run records into the fork's session.
        token = _pending.set({"store": store, "session_id": fork_session_id})
        try:
            _run_agent(agent, seed, *agent_args, **agent_kwargs)
        finally:
            _pending.reset(token)

        return fork_session_id
    finally:
        if owns_store:
            store.close()


def _public(step: Step) -> dict[str, Any]:
    return {
        "sha": step["sha"],
        "step_number": step["step_number"],
        "type": step["step_type"],
        "input": step["input"],
        "output": step["output"],
    }


def _seed_messages(store: Store, step: Step, edited: dict[str, Any]) -> list[JSON]:
    if step["step_type"] == "model_call":
        # The recorded input is the state at this point: patch it directly.
        messages = edited.get("input", {}).get("messages")
        if not isinstance(messages, list):
            raise ReplayIntegrityError(
                "edit removed or corrupted /input/messages; a fork needs a "
                "message list to resume from"
            )
        return copy.deepcopy(messages)

    if step["step_type"] != "tool_call":
        raise ReplayIntegrityError(
            f"cannot fork a {step['step_type']!r} step; retrial v1 forks "
            "model_call and tool_call steps"
        )

    return _splice_tool_output(store, step, edited)


def _splice_tool_output(store: Store, step: Step, edited: dict[str, Any]) -> list[JSON]:
    following = _next_model_call(store, step)
    if following is None:
        raise ReplayIntegrityError(
            f"cannot fork step {step['sha'][:7]}: it is the last recorded step of "
            "session {}, so the message state after it was never observed. This "
            "happens when a run crashed mid-loop - a completed loop always calls "
            "the model again after a tool call.".format(step["session_id"])
        )

    seed = copy.deepcopy(following["input"]["messages"])
    recorded = step["output"]
    new = edited.get("output")

    if not isinstance(recorded, list) or not isinstance(new, list):
        raise ReplayIntegrityError(
            "a tool_call step's output must be a list of tool results; got "
            f"recorded={type(recorded).__name__}, edited={type(new).__name__}"
        )

    recorded_by_id = _by_tool_use_id(recorded, "recorded")
    edited_by_id = _by_tool_use_id(new, "edited")

    added = set(edited_by_id) - set(recorded_by_id)
    if added:
        raise ReplayIntegrityError(
            f"edit introduced tool results that were never recorded ({sorted(added)}). "
            "retrial can only substitute facts the original run actually produced - "
            "there is no place in the recorded history to put a new one."
        )
    dropped = set(recorded_by_id) - set(edited_by_id)
    if dropped:
        raise ReplayIntegrityError(
            f"edit removed tool results {sorted(dropped)}; the recorded history "
            "still references them, so the resumed loop would see a dangling "
            "tool_use with no result."
        )

    patched = 0
    for tool_use_id, recorded_entry in recorded_by_id.items():
        block = _find_result_block(seed, tool_use_id)
        if block is None:
            raise ReplayIntegrityError(
                f"tool result {tool_use_id!r} was recorded but does not appear in "
                "the message history your loop built. retrial refuses to guess "
                "where it went - the replay would no longer be what happened."
            )
        _verify_verbatim(block, recorded_entry, tool_use_id)
        for key, value in edited_by_id[tool_use_id].items():
            if recorded_entry.get(key) != value:
                block[key] = value
        patched += 1

    if patched != len(recorded_by_id):
        raise ReplayIntegrityError(
            f"spliced {patched} of {len(recorded_by_id)} tool results"
        )
    return seed


def _by_tool_use_id(results: list[JSON], label: str) -> dict[str, JSON]:
    out: dict[str, JSON] = {}
    for entry in results:
        if not isinstance(entry, dict) or "tool_use_id" not in entry:
            raise ReplayIntegrityError(
                f"{label} tool result {entry!r} has no 'tool_use_id'. retrial "
                "matches results back into the message history by that id; "
                "without it the splice cannot be verified."
            )
        out[entry["tool_use_id"]] = entry
    return out


def _find_result_block(messages: list[JSON], tool_use_id: str) -> JSON:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("tool_use_id") == tool_use_id:
                return block
    return None


def _verify_verbatim(block: JSON, recorded_entry: JSON, tool_use_id: str) -> None:
    """The recorded output must appear in the history exactly as recorded."""
    for key, value in recorded_entry.items():
        if key not in block:
            raise ReplayIntegrityError(
                f"tool result {tool_use_id!r}: recorded field {key!r} is missing "
                "from the message history; your loop transformed the result "
                "before appending it, so retrial cannot patch it faithfully."
            )
        if block[key] != value:
            raise ReplayIntegrityError(
                f"tool result {tool_use_id!r}: recorded {key!r} does not match "
                "what is in the message history. Your loop transformed the result "
                "between executing the tool and appending it. retrial patches the "
                "history verbatim, so it cannot honestly apply your edit here."
            )


def _next_model_call(store: Store, step: Step) -> Step | None:
    for candidate in store.steps_for(step["session_id"]):
        if (
            candidate["step_number"] > step["step_number"]
            and candidate["step_type"] == "model_call"
        ):
            return candidate
    return None
