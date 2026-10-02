"""Ablation and sweep - forking many times over one axis and comparing.

    ablate  varies the STEP   - perturb each fact in turn, see which matter
    sweep   varies the VALUE  - substitute N values at one step, find a threshold

Outcomes are compared with a check, not by text, because a model rewords itself on every run.

The signal is asymmetric. A check that did NOT flip soundly rules the fact out. A check that DID
flip only suggests it matters: the agent may be reacting to the perturbation itself.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from .bisect import describe_check, forkable_steps, parse_check
from .diff import final_answer
from .errors import RetrialError
from .fork import fork
from .pricing import trajectory_cost
from .trajectory import trajectory
from .types import (
    JSON,
    AblateProbe,
    AblateResult,
    Agent,
    Check,
    Edit,
    Patch,
    Step,
    SweepBoundary,
    SweepProbe,
    SweepResult,
    SweepRun,
)

if TYPE_CHECKING:
    from .storage import Store

# A "not available" stand-in. Valid JSON, since tool results usually are.
UNAVAILABLE = json.dumps({"error": "data unavailable"})


def _probe(
    store: Store,
    sha: str,
    edit: Edit,
    agent: Agent,
    agent_args: Sequence[Any],
    agent_kwargs: dict[str, Any],
    name: str,
    # Not CheckFunction: a caller's plain callable has no `.expression`.
    check: Callable[[str | None], bool] | None,
) -> dict[str, Any]:
    """Fork once and re-execute. A failure is returned as an outcome, not raised."""
    try:
        fork_id = fork(
            from_sha=sha,
            edit=edit,
            agent=agent,
            store=store,
            name=name,
            agent_args=agent_args,
            **agent_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - a probe's failure is data
        return {
            "session_id": None,
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}",
            "passed": None,
        }

    answer = final_answer(trajectory(store, fork_id))
    return {
        "session_id": fork_id,
        "answer": answer,
        "error": None,
        "passed": check(answer) if check else None,
    }


def _default_perturbation(step: Step) -> Patch:
    """Blank every result this step produced, one op per result."""
    return [
        {"op": "replace", "path": f"/output/{i}/content", "value": UNAVAILABLE}
        for i in range(len(step["output"]))
    ]


def ablate(
    store: Store,
    session_id: str,
    check: Check,
    agent: Agent,
    perturbation: Patch | Callable[[Step], Patch] | None = None,
    agent_args: Sequence[Any] = (),
    on_probe: Callable[[AblateProbe], None] | None = None,
    **agent_kwargs: Any,
) -> AblateResult:
    """Which recorded facts is this run's outcome load-bearing on?

    Perturbs each tool_call's output in turn and reports whether the check flipped.
    `perturbation` is a patch, a callable returning one, or None to blank the results.
    """
    if isinstance(check, str):
        check = parse_check(check)

    baseline = final_answer(trajectory(store, session_id))
    baseline_passed = check(baseline)
    if not baseline_passed:
        # With a failing baseline every probe would read NOT load-bearing. Use bisect.
        raise RetrialError(
            "the check does not pass on the original run, so there is no good "
            "outcome to ablate. Ablation asks which facts a SUCCESSFUL run "
            "depended on.\n"
            f"  final answer was: {baseline!r}\n"
            "Either the check is wrong for this run, or the run failed -- in "
            "which case you want `retrial bisect`, which localizes which step "
            "made a failure inevitable."
        )

    baseline_trajectory = trajectory(store, session_id)
    baseline_cost: float | None
    baseline_cost, baseline_unpriced = trajectory_cost(baseline_trajectory)
    if baseline_unpriced:
        # An unknown model prices as None; report no cost rather than a partial one.
        baseline_cost = None

    candidates = [
        s for s in forkable_steps(store, session_id) if s["step_type"] == "tool_call"
    ]
    if not candidates:
        raise RetrialError(
            f"session {session_id} has no forkable tool_call steps to ablate. "
            "Ablation perturbs recorded facts, and this run produced none that "
            "can be resumed from."
        )

    probes: list[AblateProbe] = []
    for step in candidates:
        if perturbation is None:
            edit: Patch = _default_perturbation(step)
        elif callable(perturbation):
            edit = perturbation(step)
        else:
            edit = perturbation

        raw = _probe(
            store,
            step["sha"],
            edit,
            agent,
            agent_args,
            agent_kwargs,
            f"ablate-step-{step['step_number']}",
            check,
        )
        probe = cast(AblateProbe, raw)
        raw.update(
            {
                "sha": step["sha"],
                "step_number": step["step_number"],
                "tools": [b.get("name") for b in step["input"] if isinstance(b, dict)],
                "flipped": (
                    None if probe["passed"] is None else probe["passed"] != baseline_passed
                ),
            }
        )
        probe["cost_usd"], probe["unpriced"] = (
            trajectory_cost(trajectory(store, probe["session_id"]))
            if probe["session_id"]
            else (None, 0)
        )
        # What deleting this fact would do to the cost, from recorded costs.
        probe["cost_delta"] = (
            None
            if probe["cost_usd"] is None or baseline_cost is None
            else probe["cost_usd"] - baseline_cost
        )
        probes.append(probe)
        if on_probe:
            on_probe(probe)

    return {
        "session_id": session_id,
        "check": describe_check(check),
        "baseline_answer": baseline,
        "baseline_passed": baseline_passed,
        "baseline_cost": baseline_cost,
        # Not load-bearing AND cheaper without it: delete the tool call.
        "deletable": [
            p
            for p in probes
            if p["flipped"] is False
            and p["cost_delta"] is not None
            and p["cost_delta"] < 0
        ],
        "probes": probes,
        "re_executions": len(probes),
        # Only this one is a sound conclusion. See the module docstring.
        "not_load_bearing": [p for p in probes if p["flipped"] is False],
        "possibly_load_bearing": [p for p in probes if p["flipped"] is True],
        "inconclusive": [p for p in probes if p["flipped"] is None],
    }


def sweep(
    store: Store,
    from_sha: str,
    values: Sequence[JSON],
    agent: Agent,
    path: str = "/output/0/content",
    check: Check | None = None,
    agent_args: Sequence[Any] = (),
    samples: int = 1,
    on_probe: Callable[[SweepProbe], None] | None = None,
    **agent_kwargs: Any,
) -> SweepResult:
    """Substitute N values at one step and compare the outcomes.

    `check` is optional; without one each value's answer is returned verbatim. `samples` re-
    executes each value that many times and reports `pass_rate`. Costs len(values) * samples re-
    executions.
    """
    if isinstance(check, str):
        check = parse_check(check)
    if not values:
        raise RetrialError("sweep needs at least one value")
    if samples < 1:
        raise RetrialError("samples must be at least 1")

    step = store.get_step(from_sha)

    probes: list[SweepProbe] = []
    for index, value in enumerate(values):
        edit: Edit = {"op": "replace", "path": path, "value": value}
        runs = [
            cast(
                SweepRun,
                _probe(
                    store,
                    step["sha"],
                    edit,
                    agent,
                    agent_args,
                    agent_kwargs,
                    f"sweep-{index}" if samples == 1 else f"sweep-{index}-{sample}",
                    check,
                ),
            )
            for sample in range(samples)
        ]
        probe = _aggregate(value, runs, scored=check is not None)
        probes.append(probe)
        if on_probe:
            on_probe(probe)

    return {
        "sha": step["sha"],
        "step_number": step["step_number"],
        "path": path,
        "check": describe_check(check) if check else None,
        "probes": probes,
        "re_executions": sum(len(p["runs"]) for p in probes),
        "boundaries": _boundaries(probes) if check else [],
        "samples": samples,
    }


def _aggregate(value: JSON, runs: list[SweepRun], scored: bool) -> SweepProbe:
    """Collapse one value's re-executions into a single probe.

    The verdict is the majority (bisect's `samples` requires unanimity). A tie counts as not
    passing.
    """
    answered = [r for r in runs if r["error"] is None]
    passes = sum(1 for r in answered if r["passed"]) if scored and answered else None
    pass_rate = passes / len(answered) if passes is not None else None
    passed = None if pass_rate is None else pass_rate > 0.5

    # Show a run that agrees with the verdict; fall back to the first.
    representative = next(
        (r for r in answered if r["passed"] is passed),
        answered[0] if answered else runs[0],
    )
    return {
        "session_id": representative["session_id"],
        "answer": representative["answer"],
        # Report an error only when no run produced an answer.
        "error": None if answered else runs[0]["error"],
        "passed": passed,
        "value": value,
        "runs": runs,
        "evaluated": len(answered),
        "passes": passes,
        "pass_rate": pass_rate,
    }


def _boundaries(probes: Sequence[SweepProbe]) -> list[SweepBoundary]:
    """Adjacent value pairs where the check flipped, in the order the values were given."""
    out: list[SweepBoundary] = []
    for earlier, later in itertools.pairwise(probes):
        if earlier["passed"] is None or later["passed"] is None:
            continue
        if earlier["passed"] != later["passed"]:
            out.append({"from": earlier, "to": later})
    return out
