"""CLI surface. SHA prefix matching applies throughout, same as git."""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Iterable
from typing import Any

import click

from . import __version__
from .bisect import bisect as bisect_session
from .diff import diff as diff_sessions
from .errors import RetrialError
from .explore import ablate as ablate_session
from .explore import sweep as sweep_session
from .fork import fork as fork_session
from .pricing import cost_of_step, fmt, trajectory_cost
from .regress import recorded_runs
from .regress import rerun as rerun_sessions
from .sha import short
from .storage import (
    DEFAULT_DIR,
    Store,
    default_db_path,
    find_db_path,
    resolve_db_path,
)
from .trajectory import trajectory
from .transfer import export as export_sessions
from .transfer import import_ as import_document
from .types import (
    AblateProbe,
    AblateResult,
    Agent,
    BisectProbe,
    BisectResult,
    DiffResult,
    EditProvenance,
    RerunOutcome,
    RerunResult,
    Session,
    Step,
    SweepProbe,
    SweepResult,
    TrajectoryEntry,
)


def _store(ctx: click.Context) -> Store:
    return Store(ctx.obj["db"])


def _console_encoding() -> str:
    return getattr(sys.stdout, "encoding", None) or "utf-8"


def echo(text: object = "") -> None:
    """click.echo that degrades a character the console cannot encode.

    Model output contains emoji, and a Windows console defaults to cp1252.
    """
    text = str(text)
    encoding = _console_encoding()
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    click.echo(text)


def warn(text: object = "") -> None:
    """Diagnostics, on stderr, so a piped export stays a clean file."""
    text = str(text)
    encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    click.echo(text, err=True)


def write_data(lines: Iterable[str], destination: str | None) -> None:
    """Write export lines as UTF-8, whatever the console encoding.

    Not `echo()`: a degraded character would change the step's content and its sha.
    """
    if destination is not None:
        with open(destination, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(lines)
        return

    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:  # a text-only stream, e.g. under some test harnesses
        for line in lines:
            sys.stdout.write(line)
        return
    for line in lines:
        stream.write(line.encode("utf-8"))
    stream.flush()


def read_data(source: str) -> list[str]:
    """Read an export file as UTF-8. `-` means stdin.

    Decoded as utf-8-sig, so a byte-order mark added by Windows tooling is ignored.
    """
    if source == "-":
        stream = getattr(sys.stdin, "buffer", None)
        text = (
            stream.read().decode("utf-8-sig")
            if stream is not None
            else sys.stdin.read().lstrip("\ufeff")
        )
        return text.splitlines(keepends=True)
    with open(source, encoding="utf-8-sig") as handle:
        return handle.readlines()


def _glyphs() -> dict[str, str]:
    """Tree glyphs the terminal can encode: ASCII where box-drawing characters are missing."""
    try:
        "└── ├── │   ".encode(_console_encoding())
    except (UnicodeEncodeError, LookupError):
        return {"last": "`-- ", "mid": "|-- ", "pipe": "|   "}
    return {"last": "└── ", "mid": "├── ", "pipe": "│   "}


class RetrialGroup(click.Group):
    """Renders retrial's deliberate errors as messages, not tracebacks.

    On the group, so it applies however the CLI is entered.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except RetrialError as exc:
            echo(f"error: {exc}")
            ctx.exit(1)


@click.group(cls=RetrialGroup)
# From the package, not importlib.metadata, so a source checkout answers `--version`.
@click.version_option(__version__, "-V", "--version", prog_name="retrial")
@click.option(
    "--db",
    default=None,
    help=(
        "Path to the sessions database. Defaults to the nearest .retrial/ at or "
        "above the current directory, then $RETRIAL_DB, then ./.retrial/."
    ),
)
@click.pass_context
def cli(ctx: click.Context, db: str | None) -> None:
    """retrial - git for agent trajectories."""
    ctx.ensure_object(dict)
    # Both are kept: `init` creates a store here; every other command searches upward.
    ctx.obj["db_flag"] = db
    ctx.obj["db"] = db or resolve_db_path()


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Create .retrial/ and the sqlite database in the current directory.

    \b
    Example:
      retrial init
    """
    # Not ctx.obj["db"], which searches upward and would reopen a parent's store.
    path = ctx.obj["db_flag"] or default_db_path()
    existed = os.path.exists(path)
    shadowed = None if existed else find_db_path()

    Store(path).close()

    if existed:
        echo(f"Already initialized: {path}")
        return

    echo(f"Initialized empty retrial store: {path}")
    if shadowed:
        # Legal, but rarely intended. Warn here.
        echo(
            f"note: a store already exists above this directory at {shadowed}\n"
            "      commands run here will now use the new one. Delete this "
            f"{DEFAULT_DIR}/ to go back to it."
        )


@cli.command(name="list")
@click.pass_context
def list_sessions(ctx: click.Context) -> None:
    """List sessions as a tree.

    \b
    Example:
      retrial list
    """
    with _store(ctx) as store:
        sessions = store.list_sessions()
        if not sessions:
            echo("No sessions recorded yet.")
            return

        children: dict[str, list[Session]] = {}
        roots: list[Session] = []
        for s in sessions:
            if s["parent_session_id"]:
                children.setdefault(s["parent_session_id"], []).append(s)
            else:
                roots.append(s)

        g = _glyphs()

        def render(
            session: Session,
            prefix: str = "",
            is_last: bool = True,
            is_root: bool = False,
        ) -> None:
            if is_root:
                echo(_session_line(store, session))
                branch_prefix = ""
            else:
                connector = g["last"] if is_last else g["mid"]
                echo(prefix + connector + _session_line(store, session))
                branch_prefix = prefix + ("    " if is_last else g["pipe"])
            kids = children.get(session["id"], [])
            for i, kid in enumerate(kids):
                render(kid, branch_prefix, i == len(kids) - 1)

        for root in roots:
            render(root, is_root=True)


def _session_line(store: Store, session: Session) -> str:
    own = len(store.steps_for(session["id"]))
    if session["parent_session_id"]:
        # A fork's own steps are only its re-executed suffix. Report both counts.
        total = len(trajectory(store, session["id"]))
        count = f"{total} steps, {own} new"
    else:
        count = f"{own} steps"
    line = f"{session['id']}  {session['name']}  ({count}, {session['status']})"
    if session["parent_sha"]:
        line += f"  forked from {short(session['parent_sha'])}"
        edit = _edit_summary(session["edit_json"])
        if edit:
            line += f" [{edit}]"
    return line


def _edit_summary(edit_json: str | None) -> str | None:
    """Show WHAT changed, not just where. This is why the patch is stored."""
    if not edit_json:
        return None
    try:
        edit = json.loads(edit_json)
    except ValueError:
        return None
    if edit.get("type") == "callback":
        return f"callback {edit.get('repr', '?')} - not reproducible from record"
    patch = edit.get("patch")
    ops = [patch] if isinstance(patch, dict) else (patch or [])
    return ", ".join(
        f"{op.get('op')} {op.get('path')}"
        + (f" = {json.dumps(op['value'])[:40]}" if "value" in op else "")
        for op in ops
    )


@cli.command()
@click.argument("session_id")
@click.pass_context
def log(ctx: click.Context, session_id: str) -> None:
    """Step-by-step history of a session, one SHA per step.

    \b
    Example:
      retrial log s_a8d4f64945
    """
    with _store(ctx) as store:
        session = store.get_session(session_id)
        steps = store.steps_for(session_id)

        echo(f"session {session['id']}  ({session['name']})")
        if session["parent_sha"]:
            echo(
                f"  forked from {short(session['parent_sha'])} "
                f"(session {session['parent_session_id']}, step {session['forked_at_step']})"
            )
            edit = _edit_summary(session["edit_json"])
            if edit:
                echo(f"  edit: {edit}")
        echo(f"  status: {session['status']}")
        echo()

        if not steps:
            echo("  (no steps)")
            return

        for step in steps:
            bits = []
            if step["duration_ms"] is not None:
                bits.append(f"{step['duration_ms']:.0f}ms")
            if step["tokens_used"] is not None:
                bits.append(f"{step['tokens_used']} tok")
            suffix = f"  ({', '.join(bits)})" if bits else ""
            echo(
                f"  {short(step['sha'])}  step {step['step_number']}  "
                f"{step['step_type']}{suffix}"
            )
            echo(f"            {_summarize(step)}")


def _summarize(step: Step | TrajectoryEntry) -> str:
    if step["step_type"] == "model_call":
        messages = step["input"].get("messages", [])
        out = step["output"]
        stop = out.get("stop_reason") if isinstance(out, dict) else None
        content = out.get("content") if isinstance(out, dict) else None
        # str(), because a content block need not carry a 'type'.
        kinds = (
            [str(b.get("type", "?")) for b in content if isinstance(b, dict)]
            if isinstance(content, list)
            else []
        )
        return f"{len(messages)} messages in -> {stop or '?'} [{', '.join(kinds)}]"
    names = [
        str(b["name"])
        for b in step["input"]
        if isinstance(b, dict) and b.get("name")
    ] if isinstance(step["input"], list) else []
    return f"ran {', '.join(names) or '?'}"


@cli.command()
@click.argument("sha")
@click.pass_context
def show(ctx: click.Context, sha: str) -> None:
    """Full detail on one step, by SHA (prefix ok).

    \b
    Example:
      retrial show 4f0c1e2
    """
    with _store(ctx) as store:
        step = store.get_step(sha)
        echo(f"sha        {step['sha']}")
        echo(f"session    {step['session_id']}")
        echo(f"step       {step['step_number']}  ({step['step_type']})")
        if step["duration_ms"] is not None:
            echo(f"duration   {step['duration_ms']:.1f}ms")
        if step["tokens_used"] is not None:
            echo(f"tokens     {step['tokens_used']}")
        echo("\ninput")
        echo(_indent(json.dumps(step["input"], indent=2)))
        echo("\noutput")
        echo(_indent(json.dumps(step["output"], indent=2)))


@cli.command()
@click.argument("session_ids", nargs=-1)
@click.option("--all", "everything", is_flag=True, help="Export the whole store.")
@click.option(
    "--no-ancestors",
    is_flag=True,
    help="Send only the named sessions. The result is not independently usable.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    metavar="FILE",
    help="Write here instead of stdout.",
)
@click.pass_context
def export(
    ctx: click.Context,
    session_ids: tuple[str, ...],
    everything: bool,
    no_ancestors: bool,
    output: str | None,
) -> None:
    """Write sessions to a portable file, ancestors included.

    Ancestors travel by default because a fork without its parents cannot be
    diffed or replayed - the two things you would send it for. Descendants
    never travel: exporting a root does not hand over the experiments run on
    top of it.

    \b
    Examples:
      retrial export s_ab12cd34ef > trace.jsonl
      retrial export --all -o backup.jsonl
    """
    if everything and session_ids:
        raise click.UsageError("give session ids or --all, not both.")
    if not everything and not session_ids:
        raise click.UsageError("name a session to export, or pass --all.")

    with _store(ctx) as store:
        lines = export_sessions(
            store,
            None if everything else list(session_ids),
            ancestors=not no_ancestors,
        )
        # Diagnostics go to stderr, so a piped export is only the file.
        if no_ancestors:
            warn(
                "note: --no-ancestors, so any parent is named but not included. "
                "This file only imports into a store that already has them."
            )
        write_data(lines, output)
        if output:
            warn(f"Wrote {output}")


@cli.command(name="import")
@click.argument("source", type=click.Path())
@click.pass_context
def import_cmd(ctx: click.Context, source: str) -> None:
    """Read an export back into this store. All of it, or none of it.

    Every step's SHA is recomputed and checked, so a file altered in transit
    is refused rather than trusted. Sessions already here with identical
    content are skipped, which makes re-importing the same file a no-op.

    \b
    Examples:
      retrial import trace.jsonl
      cat trace.jsonl | retrial import -
    """
    with _store(ctx) as store:
        result = import_document(store, read_data(source), path=None if source == "-" else source)

    for warning in result.warnings:
        warn(f"warning: {warning}")

    if result.changed_nothing:
        echo("Already up to date - nothing in that file is new here.")
        return

    parts = [
        f"{result.sessions_added} session(s)",
        f"{result.steps_added} step(s)",
    ]
    echo(f"Imported {', '.join(parts)}.")
    if result.status_updated:
        echo(f"Updated the status of {result.status_updated} session(s).")
    if result.sessions_skipped or result.steps_skipped:
        echo(
            f"Skipped {result.sessions_skipped} session(s) and "
            f"{result.steps_skipped} step(s) already present."
        )


def _indent(text: str, by: str = "  ") -> str:
    return "\n".join(by + line for line in text.splitlines())


@cli.command()
@click.argument("sha")
@click.option(
    "--agent",
    required=True,
    metavar="MODULE:FUNCTION",
    help="The @record-decorated agent to re-execute, e.g. myapp.agent:run_agent. "
    "It must be callable with just the seeded message list.",
)
@click.option(
    "--edit-file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON patch to apply to the step, e.g. "
    '{"op": "replace", "path": "/output/0/content", "value": "..."}',
)
@click.option("--name", default=None, help="Name for the new forked session.")
@click.pass_context
def fork(
    ctx: click.Context,
    sha: str,
    agent: str,
    edit_file: str | None,
    name: str | None,
) -> None:
    """Fork from a step SHA and re-execute the agent for real.

    \b
    Example:
      retrial fork a1b2c3d --agent myapp:agent --edit-file edit.json
    """
    edit = None
    if edit_file:
        with open(edit_file) as fh:
            edit = json.load(fh)

    target = _load_agent(agent)
    with _store(ctx) as store:
        session_id = fork_session(
            from_sha=sha, edit=edit, agent=target, store=store, name=name
        )
        echo(f"Forked into session {session_id}")
        echo(f"  retrial log {session_id}")


@cli.command()
@click.argument("session_a")
@click.argument("session_b")
@click.option("--full", is_flag=True, help="Show every shared step, not a summary.")
@click.pass_context
def diff(ctx: click.Context, session_a: str, session_b: str, full: bool) -> None:
    """Compare two trajectories and show where they diverged.

    \b
    Example:
      retrial diff s_a8d4f64945 s_3f9c02ab1e
    """
    with _store(ctx) as store:
        result = diff_sessions(store, session_a, session_b)
        _render_diff(result, full)


def _render_diff(result: DiffResult, full: bool) -> None:
    a, b = result["a"], result["b"]
    echo("comparing")
    echo(f"  A  {a['id']}  {a['session']['name']}  ({len(a['steps'])} steps)")
    echo(f"  B  {b['id']}  {b['session']['name']}  ({len(b['steps'])} steps)")

    ancestor = result["common_ancestor"]
    echo(
        f"\ncommon ancestor: {ancestor}"
        if ancestor
        else "\ncommon ancestor: none (independent runs)"
    )

    if result["identical"]:
        echo("\nTrajectories are identical.")
        return

    shared = result["shared_prefix"]
    if shared:
        echo(
            f"shared prefix:   {len(shared)} step(s), through {short(shared[-1]['sha'])}"
        )
    else:
        echo("shared prefix:   none - diverged from the first step")

    divergence = result["divergence"]
    if divergence:
        echo(f"\ndiverged at {short(divergence['sha'])}")
        if divergence["edit"]:
            echo(f"  cause: {_edit_phrase(divergence['edit'])}")

    echo()
    for block in result["blocks"]:
        if block["tag"] == "equal" and not full:
            if block["a"]:
                echo(f"  = {len(block['a'])} shared step(s)")
            continue
        if block["tag"] == "equal":
            for entry in block["a"]:
                echo(f"  = {_diff_line(entry)}")
            continue
        for entry in block["a"]:
            echo(f"  - A  {_diff_line(entry)}")
        for entry in block["b"]:
            echo(f"  + B  {_diff_line(entry)}")

    echo("\nfinal answer")
    echo(f"  A  {result['final']['a']}")
    echo(f"  B  {result['final']['b']}")

    if any(e.get("edited") for blk in result["blocks"] for e in blk["a"] + blk["b"]):
        echo(
            "\n  * output was substituted by the fork's edit; the SHA still names "
            "the original recorded step"
        )
    echo(
        "  [replayed] played back from an ancestor, no model call made"
        "\n  [live]     this session actually re-executed it"
    )


def _edit_phrase(edit: EditProvenance) -> str:
    if edit.get("type") == "callback":
        return (
            f"callback {edit.get('repr', '?')} - effect is recorded, "
            "but the edit itself does not round-trip"
        )
    patch = edit.get("patch")
    if isinstance(patch, dict):
        ops: list[Any] = [patch]
    elif isinstance(patch, list):
        ops = patch
    else:
        ops = []
    return ", ".join(
        f"{op.get('op')} {op.get('path')}"
        + (f" = {json.dumps(op['value'])[:60]}" if "value" in op else "")
        for op in ops
    )


def _diff_line(entry: TrajectoryEntry) -> str:
    mark = "*" if entry.get("edited") else " "
    origin = "replayed" if entry["origin"] == "replayed" else "live"
    return (
        f"{short(entry['sha'])}{mark} {entry['step_type']:<10} "
        f"[{origin}]  {_summarize(entry)}"
    )


@cli.command()
@click.argument("session_id")
@click.option(
    "--check",
    required=True,
    metavar="EXPR",
    help="What a GOOD run looks like, e.g. \"output contains 'confirmed'\". "
    "Also: output not contains '...', output matches '<regex>'.",
)
@click.option(
    "--agent",
    required=True,
    metavar="MODULE:FUNCTION",
    help="The @record-decorated agent to re-execute.",
)
@click.option(
    "--samples",
    default=1,
    show_default=True,
    help="Re-probe each step N times and require unanimity before calling it "
    "recovered. Trades API calls for confidence against model non-determinism.",
)
@click.pass_context
def bisect(
    ctx: click.Context, session_id: str, check: str, agent: str, samples: int
) -> None:
    """Find the earliest step from which the agent could no longer recover.

    Each probe forks the run and re-executes for real, so this costs real model
    calls - roughly log2(steps) of them, times --samples.

    \b
    Example:
      retrial bisect s_cc6b0dc420 --agent myapp:agent --check "output contains 'confirmed'"
    """
    target = _load_agent(agent)
    with _store(ctx) as store:
        echo(f"bisecting {session_id} against: {check}\n")

        def report(probe: BisectProbe) -> None:
            verdict = "recovered" if probe["passed"] else "still broken"
            echo(
                f"  probe step {probe['step_number']} ({short(probe['sha'])}): "
                f"{verdict}  -> {probe['answer']!r}"
            )

        result = bisect_session(
            store, session_id, check, agent=target, samples=samples, on_probe=report
        )
        _render_bisect(result)


def _render_bisect(result: BisectResult) -> None:
    echo(
        f"\n{result['re_executions']} re-execution(s) over "
        f"{len(result['candidates'])} candidate step(s)."
    )

    culprit = result["culprit"]
    # bisect sets `unreproducible` to exactly `culprit is None`.
    if result["unreproducible"] or culprit is None:
        echo(
            "\nNo culprit: the agent recovered from every step probed, even "
            "though the original run failed. The failure is not reproducible by "
            "replay - likely genuine model non-determinism rather than a step "
            "that went wrong. Try --samples to probe harder."
        )
        return

    if result["inherent"]:
        echo(
            "\nNo single step is to blame: the failure reproduces even when the "
            "run is re-executed from the very first step. It is inherent to the "
            "prompt, tools, or task rather than to something that went wrong "
            "partway through."
        )
        return

    echo(f"\nFirst step that could not recover: {short(culprit['sha'])}")
    echo(f"  step {culprit['step_number']}  {culprit['step_type']}")
    echo(f"  {_summarize(culprit)}")
    echo(f"\n  retrial show {short(culprit['sha'])}")
    echo(
        "\nEverything before this step recovered on re-execution; from here on "
        "the failure reproduces. Note that binary search assumes that boundary "
        "is clean - with a real model it may be fuzzy, so check the probes above."
    )


@cli.command()
@click.argument("session_id")
@click.option("--check", required=True, metavar="EXPR", help="What a good run looks like.")
@click.option("--agent", required=True, metavar="MODULE:FUNCTION")
@click.pass_context
def ablate(ctx: click.Context, session_id: str, check: str, agent: str) -> None:
    """Which recorded facts is this run's outcome load-bearing on?

    Perturbs each tool result in turn, re-executes for real, and reports
    whether the check flipped. Costs one re-execution per tool_call step.

    \b
    Example:
      retrial ablate s_a8d4f64945 --agent myapp:agent --check "output contains 'QX7R2M'"
    """
    target = _load_agent(agent)
    with _store(ctx) as store:
        echo(f"ablating {session_id} against: {check}\n")

        def report(probe: AblateProbe) -> None:
            if probe["error"]:
                verdict = f"could not probe ({probe['error']})"
            elif probe["flipped"]:
                verdict = "outcome FLIPPED   -> possibly load-bearing"
            else:
                verdict = "outcome held      -> not load-bearing"
            tools = ", ".join(t for t in probe["tools"] if t) or "?"
            echo(f"  step {probe['step_number']} ({short(probe['sha'])}) {tools}: {verdict}")

        result = ablate_session(
            store, session_id, check, agent=target, on_probe=report
        )
        _render_ablate(result)


def _render_ablate(result: AblateResult) -> None:
    echo(f"\n{result['re_executions']} re-execution(s).")
    echo(f"baseline: check {'passes' if result['baseline_passed'] else 'fails'}")

    sound = result["not_load_bearing"]
    if sound:
        echo("\nNot load-bearing (the outcome survived without these facts):")
        for probe in sound:
            echo(f"  {short(probe['sha'])}  step {probe['step_number']}  "
                 f"{', '.join(t for t in probe['tools'] if t)}")

    maybe = result["possibly_load_bearing"]
    if maybe:
        echo("\nPossibly load-bearing (the outcome flipped without these):")
        for probe in maybe:
            echo(f"  {short(probe['sha'])}  step {probe['step_number']}  "
                 f"{', '.join(t for t in probe['tools'] if t)}")

    for probe in result["inconclusive"]:
        echo(f"\nInconclusive at step {probe['step_number']}: {probe['error']}")

    echo(
        "\nThe signal is asymmetric. 'Not load-bearing' is a sound conclusion - "
        "the run reached the same outcome without that fact. 'Possibly' is "
        "weaker: the agent may be reacting to the perturbation itself rather "
        "than to the value it lost."
    )


@cli.command()
@click.argument("sha")
@click.option(
    "--values-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON array of values to substitute, one fork each.",
)
@click.option(
    "--path",
    default="/output/0/content",
    show_default=True,
    help="JSON Pointer into the step to substitute at.",
)
@click.option("--check", default=None, metavar="EXPR", help="Optional; finds thresholds.")
@click.option(
    "--samples",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Re-executions per value. >1 reports a pass rate; the majority wins.",
)
@click.option("--agent", required=True, metavar="MODULE:FUNCTION")
@click.pass_context
def sweep(
    ctx: click.Context,
    sha: str,
    values_file: str,
    path: str,
    check: str | None,
    samples: int,
    agent: str,
) -> None:
    """Substitute N values at one step and compare the outcomes.

    Costs one real re-execution per value, times --samples.

    \b
    Example:
      retrial sweep eec0274 --agent myapp:agent --values-file fares.json
      retrial sweep eec0274 --agent myapp:agent --values-file fares.json \\
          --check "output contains 'Confirmed'" --samples 5
    """
    with open(values_file) as fh:
        values = json.load(fh)
    if not isinstance(values, list):
        raise click.BadParameter(
            f"expected a JSON array of values, got {type(values).__name__}",
            param_hint="--values-file",
        )

    target = _load_agent(agent)
    with _store(ctx) as store:
        each = f" x {samples} samples" if samples > 1 else ""
        echo(f"sweeping {short(sha)} at {path} over {len(values)} value(s){each}\n")

        def report(probe: SweepProbe) -> None:
            if probe["error"]:
                outcome = f"error: {probe['error']}"
            elif probe["passed"] is None:
                outcome = repr(probe["answer"])
            else:
                verdict = "PASS" if probe["passed"] else "FAIL"
                if samples > 1:
                    verdict += f" {probe['passes']}/{probe['evaluated']}"
                outcome = f"{verdict} -> {probe['answer']!r}"
            echo(f"  {json.dumps(probe['value'])[:44]}: {outcome}")

        result = sweep_session(
            store,
            sha,
            values,
            agent=target,
            path=path,
            check=check,
            samples=samples,
            on_probe=report,
        )
        _render_sweep(result)


def _render_sweep(result: SweepResult) -> None:
    echo(f"\n{result['re_executions']} re-execution(s).")
    for boundary in result["boundaries"]:
        before, after = boundary["from"], boundary["to"]
        echo(
            f"\nThreshold: the check flips between "
            f"{json.dumps(before['value'])[:40]} and {json.dumps(after['value'])[:40]}"
        )
        if result["samples"] > 1:
            # The rates either side show how sharp the crossing is.
            echo(
                f"  pass rate {_rate(before)} -> {_rate(after)} "
                f"over {result['samples']} samples per value"
            )
    if result["check"] and not result["boundaries"]:
        echo("\nNo threshold: the check gave the same verdict for every value.")


def _rate(probe: SweepProbe) -> str:
    return f"{probe['passes']}/{probe['evaluated']}"


@cli.command()
@click.argument("session_id")
@click.pass_context
def cost(ctx: click.Context, session_id: str) -> None:
    """Cost and token breakdown per step.

    \b
    Example:
      retrial cost s_3ca5a1df75
    """
    with _store(ctx) as store:
        entries = trajectory(store, session_id)
        total, unpriced = trajectory_cost(entries)

        echo(f"session {session_id}\n")
        echo(f"  {'step':>4}  {'sha':<8} {'ms':>7}  {'tokens':>7}  {'cost':>10}")
        for entry in entries:
            if entry["step_type"] != "model_call":
                continue
            ms = entry["duration_ms"] or 0
            echo(
                f"  {entry['step_number']:>4}  {short(entry['sha']):<8} {ms:>7.0f}  "
                f"{entry['tokens_used'] or 0:>7}  {fmt(cost_of_step(entry)):>10}"
            )

        echo(f"\n  total: {fmt(total) if not unpriced else 'partly unpriced'}")
        if unpriced:
            echo(
                f"  {unpriced} model call(s) could not be priced (unknown model or "
                "no usage reported). retrial reports no total rather than a "
                "partial one you might act on."
            )


@cli.command()
@click.option("--check", required=True, metavar="EXPR", help="What a good run looks like.")
@click.option("--agent", required=True, metavar="MODULE:FUNCTION")
@click.option(
    "--from",
    "where",
    type=click.Choice(["first", "last"]),
    default="first",
    show_default=True,
    help="Resume at the first forkable step (thorough, full re-run) or the last "
    "tool call (cheapest, but usually vacuous - the decisions are already made).",
)
@click.option(
    "--at-tool",
    default=None,
    metavar="NAME",
    help="Resume at this tool's call instead, pinning everything before it. Use "
    "this to test the specific decision your change affects - it is the option "
    "that is both cheap and meaningful.",
)
@click.pass_context
def rerun(
    ctx: click.Context, check: str, agent: str, where: str, at_tool: str | None
) -> None:
    """Re-execute every recorded run against your current code.

    Your recorded runs are the regression suite - nobody writes test cases.
    Edit your prompt or swap your model, then run this: it reports which
    recorded outcomes your change broke.

    \b
    Example:
      retrial rerun --agent myapp:agent --check "output contains 'QX7R2M'"
    """
    target = _load_agent(agent)
    with _store(ctx) as store:
        runs = recorded_runs(store)
        echo(
            f"re-executing {len(runs)} recorded run(s) against current code\n"
            f"  check:      {check}\n"
            f"  resuming at: {where} forkable step\n"
        )

        def report(result: RerunOutcome) -> None:
            marks = {
                "regressed": "REGRESSED",
                "fixed": "fixed",
                "still passing": "ok",
                "still failing": "still failing",
            }
            echo(
                f"  {result['session_id']}  {marks.get(result['verdict'], result['verdict']):<13}"
                f" {result['model_calls']} call(s)"
            )

        result = rerun_sessions(
            store, check, agent=target, where=where, at_tool=at_tool, on_result=report
        )
        _render_rerun(result)


def _render_rerun(result: RerunResult) -> None:
    total = len(result["results"])
    echo(
        f"\n{total} run(s), {result['model_calls']} model call(s)"
        + (f", {fmt(result['cost_usd'])}" if result["cost_usd"] is not None else "")
    )
    echo(f"  still passing : {len([r for r in result['unchanged'] if r['before']])}")
    echo(f"  REGRESSED     : {len(result['regressed'])}")
    echo(f"  fixed         : {len(result['fixed'])}")
    if result["errored"]:
        echo(f"  errored       : {len(result['errored'])}")

    for regression in result["regressed"]:
        echo(
            f"\nREGRESSED  {regression['session_id']}  "
            f"(resumed at {short(regression['resumed_at'])})"
        )
        echo(f"  was: {regression['before_answer']}")
        echo(f"  now: {regression['answer']}")
        echo(f"  retrial diff {regression['session_id']} {regression['fork_id']}")

    if result["regressed"]:
        # Exit non-zero so CI fails on a regression without extra plumbing.
        raise SystemExit(1)


def _load_agent(spec: str) -> Agent:
    if ":" not in spec:
        raise click.BadParameter(
            f"expected MODULE:FUNCTION, got {spec!r} (e.g. myapp.agent:run_agent)",
            param_hint="--agent",
        )
    module_name, _, attr = spec.partition(":")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise click.BadParameter(
            f"could not import {module_name!r}: {exc}", param_hint="--agent"
        ) from None
    try:
        target = getattr(module, attr)
    except AttributeError:
        raise click.BadParameter(
            f"{module_name!r} has no attribute {attr!r}", param_hint="--agent"
        ) from None
    if not getattr(target, "__retrial_agent__", False):
        raise click.BadParameter(
            f"{spec} is not decorated with @record, so its re-execution would not "
            "be recorded.",
            param_hint="--agent",
        )
    return target


def main() -> None:
    # RetrialError is handled by RetrialGroup.
    cli(obj={})


if __name__ == "__main__":
    main()
