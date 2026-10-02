# retrial

**Git for agent trajectories.** Branch, diff, and bisect LLM agent runs — backed by real re-execution, not static logs.

When an agent does something wrong, most debugging is scrolling through logs. retrial lets you go back to step N, change one fact — a tool result, a retrieved doc, an intermediate decision — and re-run your real agent from there to see what it *actually* would have done differently. With the real model, not a guess. Validated end to end against live `claude-opus-4-8`.

## Why it's different

- **Real re-execution, not log branching.** A fork re-enters your live agent loop and makes real model calls from the fork point onward. It does not relabel stored JSON and call it a branch.
- **Zero-export integration.** A decorator on your loop, not a JSON schema to hand-build.
- **Narrow and deep.** One thing — branch/diff/bisect by real execution — done well.
- **Python-native, local-first.** No account, no telemetry, no dashboard. One SQLite file in `.retrial/`.

## Install

```bash
pip install retrial
```

## Integrate

retrial needs two things from your loop: the function that calls the model, and the function that runs tool calls. You pass both in — no monkey-patching of your SDK, so every recorded step traces back to a line you wrote.

```python
from retrial import record

@record(session_name="booking-agent")
def run_agent(messages, tools=TOOLS, call_model=call_model, execute_tools=execute_tools):
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})
```

**The one rule: `messages` must be a parameter**, not a list created inside the function. That's what lets a fork seed your loop with edited history and get genuine re-execution. Give the other parameters defaults to fork from the CLI. Then just run it; steps log as the loop runs, with no export step. Async agents work too — decorate the `async def` and `await` it as usual; `fork`, `bisect`, `ablate`, `sweep`, and `rerun` re-execute async agents from the CLI or any synchronous caller.

A bundled example under `examples/` forks, diffs, and bisects with no API key.

## Any model, including local ones

retrial never imports an LLM SDK — it intercepts the `call_model` you pass in, so it has no opinion about who you call. Adapters translate a provider's wire format to and from retrial's canonical shape, which is what keeps `fork`, `diff`, `bisect`, and `cost` working identically no matter what produced the run.

```python
from retrial import openai_adapter, gemini_adapter

call_model = openai_adapter(model="gpt-5")
call_model = gemini_adapter(model="gemini-2.5-pro")

# ...or the same agent against a model on your own machine
call_model = openai_adapter(
    model="llama3.1",
    base_url="http://localhost:11434/v1",   # ollama, vllm, llama.cpp, LM Studio
    api_key="unused",
)
```

That's the only line that changes. Your loop, your tools, and every command downstream stay exactly as they were — `examples/local_model_agent.py` is the booking agent run this way, end to end, with no key and no spend.

SDKs are optional extras (`pip install 'retrial[openai]'`, `'retrial[gemini]'`), imported lazily by the adapter that needs one; a plain install still pulls in nothing but `click`. Writing an adapter for anything else — an internal gateway, a runtime that doesn't exist yet — means returning a `ModelResponse` from `call_model`.

Cost is never guessed. An unrecognized model records as `unpriced` rather than as a plausible number; `register_prices` is how you fill that in, and `FREE` is how you say a model runs on your hardware:

```python
from retrial import register_prices, FREE

register_prices({"llama3.1": FREE, "my-finetune": (0.50, 1.50)})
```

## Example: fork, then diff

Every step gets a content-hash SHA, addressable by a short prefix like git:

```bash
$ retrial log s_a8d4f64945
  4f0c1e2  step 0  model_call  (312ms, 450 tok)
  a1b2c3d  step 1  tool_call   ran search_flight
  9e77b10  step 2  model_call  (288ms, 544 tok)
```

Fork step 1 with one fact changed — the edit is a JSON patch — and the real model decides again from there:

```bash
$ retrial fork a1b2c3d --agent examples.booking_agent:run_agent --edit-file edit.json
Forked into session s_3f9c02ab1e
```

```bash
$ retrial diff s_a8d4f64945 s_3f9c02ab1e
diverged at d66697c
  cause: replace /output/0/content = flight_price 1200

  - A  book_flight    model_call
  + B  check_budget   model_call

final answer
  A  Confirmed: AUS-SFO booked for $450.
  B  That's over the $600 limit. I need approval before booking.
```

The fork called `check_budget`, a tool the original never touched — the kind of divergence only real re-execution produces. The original is never mutated; a fork is a new session, so you can branch the same step as many times as you like.

## Also

Same machinery — a fork plus a check — pointed at different questions:

- **`bisect`** — which step doomed a *failed* run? Binary search over resume points, about log2(steps) re-executions.
- **`ablate`** — which recorded facts did a *good* run actually need? Perturb each and see if the outcome flips. Causal, not heuristic.
- **`sweep`** — fork one step across many values to find a decision threshold in the model's behavior. `--samples N` re-runs each value N times and reports a pass rate, so a threshold is a crossing of two rates rather than of two coin flips.
- **`rerun`** — re-execute every recorded run against your current code. Your traces are the regression suite; it exits non-zero on a regression, so CI fails without extra plumbing.
- **`cost`** — token and dollar breakdown per step. An unknown model prices as `unpriced`, never as a guess.

Bisect and ablate are duals and each refuses the other's job: bisect wants a run that failed, ablate a run that worked.

## What retrial refuses to do

The whole product rests on the replay being exactly what happened, so retrial raises rather than guesses when it can't verify that:

- Your loop **transforms a tool result** before appending it — the patch would land on a value you never saw.
- An edit **invents or drops** a tool result the run never produced.
- A run **crashed mid-loop** — the message state after its trailing tool call was never observed.
- A **sync** agent is handed an **async** `call_model` or `execute_tools` — a sync loop can't await it, so retrial would record an un-awaited coroutine. (Async *agents* are fine; this is only the mismatch.)
- The database was written by a **newer retrial**, or an imported step's **content doesn't match its SHA**.

A wrong replay would be worse than no replay. Merge was cut for the same reason: two forks are competing hypotheses, and answers don't merge.

## CLI

```
retrial init                    create .retrial/ + sqlite db
retrial list                    sessions, tree view
retrial log <session>           step-by-step history, SHA per step
retrial show <sha>              full detail on one step
retrial fork <sha> --agent M:F --edit-file e.json
retrial diff <a> <b>            --full to expand shared steps
retrial bisect <session> --check EXPR --agent M:F
retrial ablate <session> --check EXPR --agent M:F
retrial sweep <sha> --values-file v.json --agent M:F [--samples N]
retrial rerun --check EXPR --agent M:F
retrial cost <session>
retrial export <session> > trace.jsonl
retrial import trace.jsonl
```

SHA prefix matching applies throughout. The store is found by searching upward for `.retrial/`, the way git finds `.git/`; override with `--db` or `RETRIAL_DB`.

## Python API

```python
from retrial import record, fork, diff, bisect, ablate, sweep, rerun, trajectory, Store
from retrial import export, import_
from retrial import openai_adapter, gemini_adapter, ModelResponse, tool_result, tool_uses
from retrial import register_prices, FREE
```

Every function returns a plain dict, so results stay JSON-shaped and printable. The shapes are declared in `retrial/types.py` and shipped with `py.typed`, so a typo in a key is a type error, not a `KeyError` at 3am.

## License

MIT
