"""Turning live objects into stable JSON.

Serialization must be deterministic, since step SHAs hash it, and lossless, since forks replay
it. Anything that cannot round-trip raises.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from .errors import ReplayIntegrityError
from .types import JSON

_PRIMITIVES = (str, int, float, bool, type(None))


def to_jsonable(obj: Any, _path: str = "", _seen: frozenset[int] | None = None) -> JSON:
    """Convert an arbitrary object into JSON-safe primitives.

    Handles Pydantic models, SDK objects with `to_dict`, dataclasses, namedtuples and
    containers.
    """
    if _seen is None:
        _seen = frozenset()

    if isinstance(obj, _PRIMITIVES):
        return obj

    # A cyclic history cannot be replayed. Refuse.
    marker = id(obj)
    if marker in _seen:
        raise ReplayIntegrityError(
            f"circular reference at {_path or '<root>'}; cannot record a state "
            "that references itself"
        )
    _seen = _seen | {marker}

    if isinstance(obj, dict):
        out: dict[str, JSON] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ReplayIntegrityError(
                    f"non-string dict key {key!r} at {_path or '<root>'}; JSON "
                    "cannot represent it, so the replay would not round-trip"
                )
            out[key] = to_jsonable(value, f"{_path}/{key}", _seen)
        return out

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, f"{_path}/{i}", _seen) for i, v in enumerate(obj)]

    # Pydantic v2 (the Anthropic SDK's response models).
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"), _path, _seen)

    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict(), _path, _seen)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), _path, _seen)

    if hasattr(obj, "_asdict"):  # namedtuple
        return to_jsonable(obj._asdict(), _path, _seen)

    if hasattr(obj, "__dict__"):
        public = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        if public:
            return to_jsonable(public, _path, _seen)

    raise ReplayIntegrityError(
        f"cannot serialize {type(obj).__name__} at {_path or '<root>'}. retrial "
        "records exact state so replays are faithful; storing repr() here would "
        "make the recording a description rather than a recording."
    )


def canonical_json(obj: JSON) -> str:
    """Deterministic JSON, for storage and step SHAs. `sort_keys` keeps a SHA stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
