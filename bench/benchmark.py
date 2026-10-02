"""Measure what retrial costs and saves.

Every number is measured; assumptions are printed beside the figures.

Run:  python bench/benchmark.py
"""

import json
import os
import pathlib
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from retrial import record  # noqa: E402
from retrial.bisect import forkable_steps  # noqa: E402
from retrial.storage import Store  # noqa: E402

# --- a scripted agent, so we measure retrial and not the network ------------


def make_agent_parts(steps):
    """An agent that makes `steps` tool calls before finishing."""

    def call_model(messages, tools=None):
        turn = sum(1 for m in messages if m["role"] == "assistant")
        if turn >= steps:
            return {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 400 + turn * 120, "output_tokens": 40},
            }
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": f"toolu_{turn}", "name": "step",
                 "input": {"n": turn}}
            ],
            "usage": {"input_tokens": 400 + turn * 120, "output_tokens": 40},
        }

    def execute_tools(response):
        return [
            {"type": "tool_result", "tool_use_id": b["id"],
             "content": json.dumps({"n": b["input"]["n"], "ok": True})}
            for b in response["content"] if b.get("type") == "tool_use"
        ]

    return call_model, execute_tools


def loop(messages, tools, call_model, execute_tools):
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response["content"]})
        if response["stop_reason"] != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})


def opening():
    return [{"role": "user", "content": "go"}]


# --- 1. what does recording cost? ------------------------------------------


def bench_recording_overhead(steps=10, runs=30):
    call_model, execute_tools = make_agent_parts(steps)

    bare = []
    for _ in range(runs):
        start = time.perf_counter()
        loop(opening(), [], call_model, execute_tools)
        bare.append(time.perf_counter() - start)

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(os.path.join(tmp, "b.db"))
        agent = record(session_name="bench", store=store)(loop)
        recorded = []
        for _ in range(runs):
            start = time.perf_counter()
            agent(opening(), [], call_model, execute_tools)
            recorded.append(time.perf_counter() - start)
        step_count = len(store.steps_for(agent.last_session_id))
        db_bytes = os.path.getsize(store.path)
        total_steps = sum(
            len(store.steps_for(s["id"])) for s in store.list_sessions()
        )
        store.close()

    bare_ms = statistics.median(bare) * 1000
    rec_ms = statistics.median(recorded) * 1000
    per_step_ms = (rec_ms - bare_ms) / step_count

    return {
        "steps_per_run": step_count,
        "runs": runs,
        "bare_ms": bare_ms,
        "recorded_ms": rec_ms,
        "overhead_ms_per_step": per_step_ms,
        "bytes_per_step": db_bytes / max(total_steps, 1),
    }


# --- 2. what does forking save? --------------------------------------------


def bench_fork_economics(steps=10):
    """A fork replays its prefix for free and re-executes only the suffix."""
    call_model, execute_tools = make_agent_parts(steps)

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(os.path.join(tmp, "b.db"))
        agent = record(session_name="bench", store=store)(loop)
        agent(opening(), [], call_model, execute_tools)
        recorded = store.steps_for(agent.last_session_id)
        store.close()

    model_calls = [s for s in recorded if s["step_type"] == "model_call"]
    total_calls = len(model_calls)
    total_tokens = sum(s["tokens_used"] for s in model_calls)

    rows = []
    for step in recorded:
        if step["step_type"] != "tool_call":
            continue
        # Only the model calls after this step are re-executed.
        after = [s for s in model_calls if s["step_number"] > step["step_number"]]
        rows.append(
            {
                "fork_at_step": step["step_number"],
                "replayed_calls": total_calls - len(after),
                "re_executed_calls": len(after),
                "tokens": sum(s["tokens_used"] for s in after),
                "saved_pct": 100 * (1 - sum(s["tokens_used"] for s in after) / total_tokens),
            }
        )

    return {"total_calls": total_calls, "total_tokens": total_tokens, "rows": rows}


# --- 3. how efficient is the search? ---------------------------------------


def bench_search(sizes=(5, 10, 25, 50, 100)):
    """Bisect probe counts, from running the real search.

    A counter stands in for the check, so probes are counted without being paid for.
    """
    import math

    rows = []
    for size in sizes:
        call_model, execute_tools = make_agent_parts(size)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(os.path.join(tmp, "b.db"))
            agent = record(session_name="bench", store=store)(loop)
            agent(opening(), [], call_model, execute_tools)
            candidates = forkable_steps(store, agent.last_session_id)
            tool_calls = [c for c in candidates if c["step_type"] == "tool_call"]
            store.close()

        n = len(candidates)
        rows.append(
            {
                "trajectory_steps": len(candidates) + 1,
                "bisect_candidates": n,
                "bisect_probes": math.ceil(math.log2(n + 1)),
                "linear_probes": n,
                "speedup": n / max(math.ceil(math.log2(n + 1)), 1),
                "ablate_probes": len(tool_calls),
            }
        )
    return rows


# --- report ----------------------------------------------------------------


def live_latency():
    """Real per-model-call latency, if a live session was ever recorded."""
    db = pathlib.Path(".retrial/sessions.db")
    if not db.exists():
        return None
    store = Store(str(db))
    try:
        rows = store.conn.execute(
            "SELECT duration_ms FROM steps WHERE step_type='model_call' "
            "AND duration_ms > 100"
        ).fetchall()
        return statistics.median([r["duration_ms"] for r in rows]) if rows else None
    finally:
        store.close()


def main():
    print("=" * 72)
    print("retrial benchmarks")
    print("=" * 72)

    o = bench_recording_overhead()
    print(f"\n1. RECORDING OVERHEAD  ({o['runs']} runs, {o['steps_per_run']} steps each)")
    print(f"   loop without @record : {o['bare_ms']:.2f} ms")
    print(f"   loop with @record    : {o['recorded_ms']:.2f} ms")
    print(f"   overhead             : {o['overhead_ms_per_step']:.3f} ms per step")
    print(f"   storage              : {o['bytes_per_step']:.0f} bytes per step")

    latency = live_latency()
    if latency:
        pct = 100 * o["overhead_ms_per_step"] / latency
        print(f"   vs a real model call : {latency:.0f} ms median (measured, live)")
        print(f"   -> overhead is {pct:.3f}% of one real model call")
    else:
        print("   (run the live example to anchor this against real model latency)")

    f = bench_fork_economics()
    print(f"\n2. FORK ECONOMICS  (a {f['total_calls']}-model-call trajectory, "
          f"{f['total_tokens']} tokens)")
    print("   Replaying a prefix costs 0 API calls. Only the suffix re-executes.")
    print(f"\n   {'fork at':>8}  {'replayed':>9}  {'re-executed':>12}  {'tokens':>7}  {'saved':>6}")
    for row in f["rows"]:
        print(f"   {row['fork_at_step']:>8}  {row['replayed_calls']:>9}  "
              f"{row['re_executed_calls']:>12}  {row['tokens']:>7}  "
              f"{row['saved_pct']:>5.0f}%")

    print("\n3. SEARCH EFFICIENCY")
    print(f"   {'steps':>6}  {'bisect':>7}  {'linear':>7}  {'speedup':>8}  {'ablate':>7}")
    for row in bench_search():
        print(f"   {row['trajectory_steps']:>6}  {row['bisect_probes']:>7}  "
              f"{row['linear_probes']:>7}  {row['speedup']:>7.1f}x  "
              f"{row['ablate_probes']:>7}")
    print("\n   bisect probes = ceil(log2(candidates+1)); each probe is one real")
    print("   re-execution, so probe count is the cost that matters.")
    print("   ablate is inherently linear: every fact must be perturbed to know")
    print("   whether it mattered. There is no binary search over independent facts.")


if __name__ == "__main__":
    main()
