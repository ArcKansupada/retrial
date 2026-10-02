"""Content-addressed step identity.

A step SHA hashes (session id, step number, input, output). Full SHAs are stored; prefixes
resolve like git's, and an ambiguous one is an error.
"""

from __future__ import annotations

import hashlib

from .serialize import canonical_json
from .types import JSON

SHORT = 7


def compute_sha(
    session_id: str,
    step_number: int,
    step_type: str,
    input_obj: JSON,
    output_obj: JSON,
) -> str:
    payload = canonical_json(
        {
            "session_id": session_id,
            "step_number": step_number,
            "step_type": step_type,
            "input": input_obj,
            "output": output_obj,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def short(sha: str) -> str:
    return sha[:SHORT]
