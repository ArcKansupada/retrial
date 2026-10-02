"""The fork `edit` API: a structured patch, or a callback.

A patch is data, so it is stored on the fork's session and shown by `retrial log`. A callback
covers edits that depend on the step's content, but cannot be replayed from the record. Patch
paths are JSON Pointers rooted at the step.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from .errors import RetrialError
from .types import JSON, Edit, EditProvenance, Patch

_MISSING = object()

#: What a normalized edit does: take the step dict, return the edited one.
EditFn = Callable[[dict[str, Any]], dict[str, Any]]


class PatchError(RetrialError):
    pass


def parse_pointer(pointer: str) -> list[str]:
    if pointer in ("", "/"):
        return []
    if not pointer.startswith("/"):
        raise PatchError(
            f"path {pointer!r} must be a JSON Pointer starting with '/' "
            "(e.g. /output/0/content)"
        )
    return [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]


def _descend(doc: JSON, tokens: list[str], pointer: str) -> JSON:
    """Walk to the container holding the final token."""
    node = doc
    for i, token in enumerate(tokens[:-1]):
        node = _child(node, token, "/".join(tokens[: i + 1]), pointer)
    return node


def _child(node: JSON, token: str, so_far: str, pointer: str) -> JSON:
    if isinstance(node, list):
        index = _as_index(token, pointer)
        if index >= len(node):
            raise PatchError(
                f"path {pointer!r}: index {index} is out of range at /{so_far} "
                f"(length {len(node)})"
            )
        return node[index]
    if isinstance(node, dict):
        if token not in node:
            raise PatchError(
                f"path {pointer!r}: no key {token!r} at /{so_far}. "
                f"Available: {sorted(node)}"
            )
        return node[token]
    raise PatchError(
        f"path {pointer!r}: cannot descend into {type(node).__name__} at /{so_far}"
    )


def _as_index(token: str, pointer: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise PatchError(
            f"path {pointer!r}: {token!r} is not a valid list index"
        ) from None


def apply_patch(doc: JSON, patch: Patch) -> JSON:
    """Apply one op or a list of ops. Returns a new document; never mutates."""
    ops: list[Any] = [patch] if isinstance(patch, dict) else list(patch)
    result = copy.deepcopy(doc)
    for op in ops:
        result = _apply_one(result, op)
    return result


def _apply_one(doc: JSON, op: Any) -> JSON:
    if not isinstance(op, dict):
        raise PatchError(f"each patch op must be an object, got {type(op).__name__}")
    kind = op.get("op")
    if kind not in ("replace", "add", "remove"):
        raise PatchError(
            f"unsupported op {kind!r}; retrial supports replace, add, remove"
        )
    pointer = op.get("path")
    if not isinstance(pointer, str):
        raise PatchError("every patch op needs a string 'path'")

    tokens = parse_pointer(pointer)
    if not tokens:
        raise PatchError("cannot patch the step root; target a field like /output")

    value = op.get("value", _MISSING)
    if kind in ("replace", "add") and value is _MISSING:
        raise PatchError(f"op {kind!r} at {pointer!r} needs a 'value'")

    parent = _descend(doc, tokens, pointer)
    last = tokens[-1]

    if isinstance(parent, list):
        if kind == "add" and last == "-":
            parent.append(value)
            return doc
        index = _as_index(last, pointer)
        if kind == "add":
            if index > len(parent):
                raise PatchError(f"path {pointer!r}: index {index} past end of list")
            parent.insert(index, value)
        elif index >= len(parent):
            raise PatchError(
                f"path {pointer!r}: index {index} out of range (length {len(parent)})"
            )
        elif kind == "replace":
            parent[index] = value
        else:
            del parent[index]
        return doc

    if isinstance(parent, dict):
        if kind == "replace" and last not in parent:
            raise PatchError(
                f"path {pointer!r}: no existing key {last!r} to replace. "
                "Use op 'add' to create it."
            )
        if kind == "remove":
            if last not in parent:
                raise PatchError(f"path {pointer!r}: no key {last!r} to remove")
            del parent[last]
        else:
            parent[last] = value
        return doc

    raise PatchError(
        f"path {pointer!r}: cannot set a field on {type(parent).__name__}"
    )


def normalize_edit(edit: Edit) -> tuple[EditFn, EditProvenance | None]:
    """Return (apply_fn, provenance) for either edit shape.

    `provenance` is stored on the fork's session: the patch itself, or a marker that a callback
    ran.
    """
    if edit is None:
        return (lambda step: copy.deepcopy(step)), None

    if isinstance(edit, (dict, list)):
        patch = copy.deepcopy(edit)
        return (lambda step: apply_patch(step, patch)), {"type": "patch", "patch": patch}

    if callable(edit):
        name = getattr(edit, "__name__", repr(edit))
        return edit, {
            "type": "callback",
            "repr": name,
            "note": "callback edits do not round-trip from this record",
        }

    raise PatchError(
        f"edit must be a patch object, a list of patch ops, or a callable - "
        f"got {type(edit).__name__}"
    )
