# Changelog

All notable changes to retrial are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until 1.0, the **public API is the names exported from `retrial/__init__.py`**
plus the CLI. The dict shapes those functions return are now declared in
`retrial/types.py`; before 1.0 they may **gain** keys in a minor release, but an
existing key will not silently change meaning.

## [Unreleased]

### Added
- **`sweep(samples=N)` / `retrial sweep --samples N`.** A real model near a
  decision boundary answers differently run to run, so a single probe per value
  measures a coin flip rather than a threshold. `--samples` re-executes each
  value N times and reports `passes` / `evaluated` / `pass_rate` per value, so a
  crossing is a crossing of two rates and the rates on either side say how sharp
  it is - `5/5 -> 0/5` is a decision, `3/5 -> 2/5` is the agent being unsure
  twice. The verdict is the **majority**, deliberately unlike `bisect`'s
  `samples`, which requires unanimity: one reproduction proves a failure is
  reachable from a step, but a fare the agent books four times in five belongs
  on the booking side of the boundary. A tie counts as not passing, so a
  boundary never moves between identical runs. A value whose probes partly
  errored is scored on the samples that survived; only a value with no answer at
  all is reported as an error.

### Changed
- `SweepProbe` gains `runs`, `evaluated`, `passes`, and `pass_rate`; `SweepResult`
  gains `samples`. Additive, per the pre-1.0 policy above. `session_id` and
  `answer` on a probe are now the first run whose verdict agrees with `passed`,
  so a booking confirmation is never printed next to `FAIL`; with the default
  `samples=1` there is one run and the values are unchanged.
- `sweep`'s `re_executions` now counts every fork (`len(values) * samples`),
  not values.

## [0.2.0] - 2026-08-07

### Added
- **retrial works with any model now, local ones included.** `retrial.providers`
  adds adapters that translate a provider's wire format to and from retrial's
  canonical trajectory shape, so `fork`, `diff`, `bisect`, `ablate`, `sweep`,
  `rerun`, and `cost` are provider-indifferent without a single branch on who
  was called - they only ever see canonical messages. `openai_adapter` takes a
  `base_url`, which is all it takes to run a recorded agent against Ollama,
  vLLM, llama.cpp, LM Studio, or OpenRouter; `gemini_adapter` covers
  google-genai, synthesizing the tool-call id Gemini does not issue but a fork
  splices on. Tools are declared once in canonical form and translated per
  provider, so switching models never edits your tool list. The SDKs are
  optional extras (`retrial[openai]`, `retrial[gemini]`) imported lazily inside
  each adapter - a plain install still pulls in nothing but `click`. Writing an
  adapter for anything else means returning a `ModelResponse` from `call_model`.
  `examples/local_model_agent.py` is the booking agent run end to end against a
  local model, with no key and no spend.
- **Prices are extensible: `register_prices` and `FREE`.** The built-in table
  cannot know about your fine-tune, your gateway's model names, or a model
  newer than this release, and retrial still refuses to guess at any of them.
  `register_prices({"my-finetune": (0.50, 1.50)})` fills the gap, and
  `register_prices({"llama3.1": FREE})` says a model runs on your own hardware -
  so a local run reports `$0.00000`, which is a fact, instead of `unpriced`,
  which is merely the absence of one. An unregistered model is `unpriced` as
  before.

### Fixed
- **A dated model snapshot now prices as its longest matching base model.**
  `normalize()` took the first prefix match in table order, so with both
  `gpt-5` and `gpt-5-mini` known, `gpt-5-mini-2026-01` could price at the
  expensive model's rate. It could not bite while the table held only Claude
  models, and it would have bitten immediately as soon as it didn't.
- **Cached tokens are no longer billed twice through an adapter.** OpenAI's
  `prompt_tokens` and Gemini's `promptTokenCount` include the cached prefix,
  while retrial (following Anthropic) prices `input_tokens` and
  `cache_read_input_tokens` separately. Both adapters subtract, so a cached
  token is charged at the cache rate only.

## [0.1.4] - 2026-08-07

### Added
- **Async agents are recorded, no longer refused.** `@record` now wraps an
  `async def` agent (and async `call_model` / `execute_tools`): the wrapper
  awaits the body, so a session is stamped `complete`/`failed` only after the
  loop actually runs, and the interception points await the real call so the
  response is recorded, not an un-awaited coroutine. The fork handoff moved from
  a `threading.local` to a `contextvars.ContextVar`, which stays correct under
  both threads and `asyncio.gather` (a thread-local would let concurrent async
  forks cross-record into each other's sessions). A *sync* agent handed an async
  interception point is still refused. Re-execution drives async agents too:
  `fork` (and the `bisect`/`ablate`/`sweep`/`rerun` built on it) run them with
  `asyncio.run` from any synchronous caller, including the CLI. The one refusal
  left is forking from inside a running event loop, where `asyncio.run` cannot
  nest - a native async fork API is future work.
- **Every CLI command's `--help` now carries a usage example.** All 13 commands
  (`init`, `list`, `log`, `show`, `fork`, `diff`, `bisect`, `ablate`, `sweep`,
  `rerun`, `cost`, `export`, `import`) show one realistic invocation, so the
  reference is right where you are when you run `<command> --help`.

## [0.1.3] - 2026-08-01

### Changed
- **README rewritten and pointed at PyPI.** `pip install retrial` now that the
  package is published (was `pip install -e .`). Cut to a focused overview - one
  worked fork/diff example, the other commands a line each. Removed all in-page
  hyperlinks; relative ones never resolved on PyPI anyway.

### Fixed
- **Dollar amounts no longer render as inline math.** They now appear only in
  code blocks, never in prose. PyPI renders Markdown through comrak, which reads
  `$...$` as math and - unlike GitHub - decodes an `&#36;` entity before
  scanning, so neither the 0.1.1 backslash escape nor the 0.1.2 entity fixed it
  there. Keeping the amounts out of prose does, on every renderer.

## [0.1.2] - 2026-08-01

### Fixed
- **README dollar amounts, take two.** The `\$` escape in 0.1.1 fixed GitHub
  but not PyPI: PyPI now renders Markdown through comrak, which reads `$...$` as
  inline math and does not treat a backslash-escaped `\$` as a way out - so a
  sentence like "a &#36;450 ... &#36;1,450 fare" still collapsed into a math span there.
  Replaced the escapes with the HTML entity `&#36;`. This fixed GitHub but not
  PyPI - see 0.1.3. Documentation only; no code change.

## [0.1.1] - 2026-08-01

### Fixed
- **README rendering on PyPI and GitHub.** Documentation only; no code change.
  Relative links (`retrial-design-doc.md`, `examples/README.md`) 404'd on PyPI,
  which does not rewrite relative paths to the repo the way GitHub does - made
  them absolute. And prose dollar amounts formed accidental inline-math spans:
  two dollar signs in a sentence ("a &#36;450 ... &#36;1,450 fare") are read as math
  delimiters, lifting the text between them out of the prose in an italic math
  font. Escaped the prose dollar signs (`\$`), which render as a literal `$` and
  cannot open a math span; dollar signs inside code fences are untouched.

## [0.1.0] - 2026-07-31

First working version: record, fork, diff, bisect, ablate, sweep, rerun, and
cost, all backed by real re-execution against a live model. Validated against
`claude-opus-4-8`.

### Added
- **`retrial export` / `retrial import`**, and `export` / `import_` in the
  Python API. A session and its ancestors travel as a JSONL file — the
  collaboration story with no server in it, and the only lossless way to keep a
  trace before deleting `.retrial/`. A fork carries its ancestors (a fork
  without its parents can't be diffed or replayed); descendants stay behind.
  Import is idempotent and incremental: every step's SHA is recomputed and
  checked, so a file altered in transit is refused rather than trusted; matching
  content is skipped; the whole read is one transaction, so a file rejected on
  its last line leaves the store untouched. Session IDs are preserved, so a SHA
  stays a shared handle across machines — two runs claiming one ID is a refusal,
  not a silent merge. Adds `ExportFormatError`.
- **The store is found by searching upward**, the way git finds `.git/`. A
  command run in a subdirectory now uses the project's store instead of quietly
  creating a second, empty one beside itself — see *Fixed* below. `RETRIAL_DB`
  points a whole shell or CI job at one store; `--db` still outranks everything.
  `retrial init` is the deliberate exception: like `git init` it always creates
  here, and now prints a `note:` if that shadows a store further up.
- **Schema versioning.** The database now records which schema wrote it
  (`PRAGMA user_version`), and a file written by a newer retrial is refused at
  open time with a `SchemaVersionError` naming both versions, instead of being
  opened and misbehaving later. Databases from earlier builds carry no stamp; that
  layout *is* v1, so they are adopted and labelled rather than rejected. There is
  a `_migrate` hook for the next bump.
- **Type hints throughout, and a `py.typed` marker.** The package is fully
  annotated and checked with mypy under `disallow_untyped_defs`. `retrial/types.py`
  declares every shape the public API returns — `Step`, `Session`,
  `TrajectoryEntry`, `DiffResult`, `BisectResult`, `AblateResult`, `SweepResult`,
  `RerunResult`, and the patch/check/agent types — all re-exported from the top
  level. The records are still plain dicts; a TypedDict is a dict at runtime, so
  nothing about their behaviour changed.
- `python -m retrial` as an entry point, matching the `retrial` console script.
- `retrial --version` (`-V`).
- Packaging metadata for PyPI: SPDX license, `LICENSE` file, authors, project
  URLs, classifiers, and keywords.
- CI across Python 3.10–3.13 on Linux, plus Windows and macOS at the ends of
  the range — the Windows jobs cover the cp1252 console-encoding paths.
- Tag-triggered PyPI publishing via OIDC trusted publishing.

### Changed
- **Renamed from `retrail` to `retrial`.** The package, the `retrial` command,
  the `.retrial/` store directory, and the `RETRIAL_*` environment variables all
  move together. Done before the first release on purpose: nothing is published,
  so this is a rename rather than a deprecation shim plus a squatted name kept
  alive forever. Nobody has a `.retrail/` to migrate except the author, who
  renames the directory by hand; the SQLite file itself is unchanged, since the
  old name was never written into it.
- Minimum Python is now 3.10. The previous `>=3.9` floor was declared but never
  tested, and 3.9 reached end of life in October 2025.
- The package version is single-sourced from `retrial/__init__.py`; the
  duplicate in `pyproject.toml` is gone.

### Fixed
- **A `Store` can be shared between threads.** Recording two runs at once — a
  web service, a harness evaluating prompts in parallel — used to fail as a raw
  `sqlite3.ProgrammingError` several frames deep, because the connection refused
  cross-thread use. Every method now holds a reentrant lock spanning the
  statement *and* its commit, so `check_same_thread=False` is safe. Three
  narrower races went with it: the process-wide store cache was a check-then-set
  that opened one connection per thread and orphaned all but one (measured: 8
  threads, 8 connections, 7 leaked); `fork()`'s handoff to the decorator was a
  module-level dict, so two concurrent forks could record one agent's steps
  under the other's session; and step numbers were read in one statement and
  written in another, which two writers on one session lose to a `UNIQUE`
  violation. `Store.add_step` now accepts `step_number=None` and allocates
  inside the lock — pass None unless you are reconstructing a specific
  numbering.
- **An async agent is refused instead of mis-recorded.** `@record` on an
  `async def` raised only by accident and only sometimes: calling the function
  returns a coroutine without running the body, so the session was stamped
  `complete` before the loop had taken a step, and the handler that marks a
  crashed run `failed` never saw the failure — it happened later, inside the
  event loop. An async agent that raised partway through was stored as a
  successful run. It is now an `IntegrationError` at decoration time, so the
  problem surfaces on import rather than after a run that looked fine. An async
  `call_model` or `execute_tools` inside a *sync* agent is refused at call time
  for the same reason; it previously failed further away, as "cannot serialize
  coroutine" from the serializer.
- **A run recorded from a subdirectory is no longer invisible.** `.retrial/` was
  resolved against the current working directory and nothing else, so `retrial
  log` one level down created a fresh database and truthfully reported no
  sessions while the real trace sat untouched one directory up. Recording split
  the same way — an agent launched from a subdirectory wrote somewhere the CLI
  would never look — and the two halves could disagree without either ever
  raising.
