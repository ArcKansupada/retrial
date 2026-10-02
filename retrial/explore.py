"""Ablation and sweep - forking many times over one axis and comparing.

Both are the same move as bisect: fork, re-execute for real, evaluate a check.

    ablate  varies the STEP    - perturb each fact in turn, see which matter
    sweep   varies the VALUE   - substitute N values at one step, find a threshold

A check rather than comparing answers, because a real model rewords itself on
every run: text equality would measure non-determinism, not causation.

A check collapses the wording, but not the sampling - a model near a decision
boundary genuinely answers differently run to run. `sweep(samples=N)` re-runs
each value N times and reports the rate, so a threshold is a crossing of two
rates instead of a crossing of two coin flips.

The signal is asymmetric, and callers are told so:

  check did NOT flip -> soundly NOT load-bearing. The answer survived without
                        the fact. A real conclusion.
  check DID flip     -> only POSSIBLY load-bearing. The agent may be reacting
                        to the perturbation itself (a tool that now errors)
                        rather than to the value it lost.

So ablation rules facts out rigorously and rules them in suggestively - still
more than attribution could claim, because this is intervention, not inference.
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

# A neutral "this fact was not available" stand-in. Valid JSON, because tool
# result content is conventionally a JSON string: an agent parsing it should
# see a well-formed object, not a syntax error.
UNAVAILABLE = json.dumps({"error": "data unavailable"})


def _probe(
    store: Store,
    sha: str,
    edit: Edit,
    agent: Agent,
    agent_args: Sequence[Any],
    agent_kwargs: dict[str, Any],
    name: str,
    # Not CheckFunction: a check reaching here may be the caller's own plain
    # callable, which carries no `.expression`. Only the parsed kind does.
    check: Callable[[str | None], bool] | None,
) -> dict[str, Any]:
    """Fork once and re-execute. A failure is an outcome, not a crash: many
    probes run, so one error is reported alongside the rest, not raised."""
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
    """Blank every result this step produced.

    One op per result rather than a fixed `/output/0/content`, so a step with
    parallel tool calls is fully ablated, not just its first result.
    """
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

    Perturbs each tool_call's output in turn, re-executes, and reports whether
    the check flipped. `perturbation` may be a patch, a callable taking the step
    and returning a patch, or None for the default (blank the results).
    """
    if isinstance(check, str):
        check = parse_check(check)

    baseline = final_answer(trajectory(store, session_id))
    baseline_passed = check(baseline)
    if not baseline_passed:
        # With a failing baseline every probe reports "outcome held" and gets
        # labelled NOT load-bearing - reassuring, and vacuous: the run was
        # broken the whole time. Bisect is the tool for that case.
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
        # An unknown model prices as None, and summing that as zero would make
        # every delta a lie. Report no cost at all rather than a partial one.
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
        # What deleting this fact would do to the bill. Measured, not modelled:
        # both figures are the real recorded cost of a real trajectory.
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

    Finds thresholds: at what fare does it stop booking? `check` is optional -
    without one you get each value's answer verbatim, which is what you want
    when reading the results rather than automating over them.

    `samples` re-executes each value that many times. With one sample a
    threshold is the crossing of two coin flips; with several it is a crossing
    of two rates, and `pass_rate` per value is the curve a caller can fit.
    Costs len(values) * samples real re-executions.
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

    The verdict is the MAJORITY, where bisect's `samples` requires unanimity.
    The asymmetry is deliberate: one reproduction proves a failure is reachable
    from a step, so bisect is right to treat it as decisive, but a sweep asks
    where behaviour *typically* flips - a fare the agent books four times in
    five belongs on the booking side of the boundary, not the refusing side.

    A tie is not a majority, so the boundary lands on the first value where
    passing runs actually outnumber failing ones.
    """
    answered = [r for r in runs if r["error"] is None]
    passes = sum(1 for r in answered if r["passed"]) if scored and answered else None
    pass_rate = passes / len(answered) if passes is not None else None
    passed = None if pass_rate is None else pass_rate > 0.5

    # Show a run that agrees with the verdict, so a reader never sees a
    # booking confirmation next to FAIL. Falls back to the first run when
    # nothing was scored, or when every one of them errored.
    representative = next(
        (r for r in answered if r["passed"] is passed),
        answered[0] if answered else runs[0],
    )
    return {
        "session_id": representative["session_id"],
        "answer": representative["answer"],
        # One bad sample out of five is noise, not a failed probe. Only report
        # an error when there is no answer at all behind this value.
        "error": None if answered else runs[0]["error"],
        "passed": passed,
        "value": value,
        "runs": runs,
        "evaluated": len(answered),
        "passes": passes,
        "pass_rate": pass_rate,
    }


def _boundaries(probes: Sequence[SweepProbe]) -> list[SweepBoundary]:
    """Adjacent value pairs where the check flipped - the thresholds.

    Reported in the order the values were given, so a caller sweeping an
    ordered range reads them as the crossing points.
    """
    out: list[SweepBoundary] = []
    for earlier, later in itertools.pairwise(probes):
        if earlier["passed"] is None or later["passed"] is None:
            continue
        if earlier["passed"] != later["passed"]:
            out.append({"from": earlier, "to": later})
    return out
