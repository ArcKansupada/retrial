"""Cost accounting and the regression runner."""

import json

import pytest

from retrial import record
from retrial.errors import RetrialError
from retrial.pricing import PRICES, cost_of, fmt, normalize, trajectory_cost
from retrial.regress import recorded_runs, rerun

# --- pricing ---------------------------------------------------------------


def response(model, **usage):
    return {"model": model, "usage": usage}


def test_cost_of_a_known_model():
    # Opus 4.8: $5/1M in, $25/1M out.
    cost = cost_of(response("claude-opus-4-8", input_tokens=1_000_000, output_tokens=0))
    assert cost == pytest.approx(5.00)
    cost = cost_of(response("claude-opus-4-8", input_tokens=0, output_tokens=1_000_000))
    assert cost == pytest.approx(25.00)


def test_cache_tokens_are_priced_at_their_own_rates():
    """Cache reads are ~0.1x and writes ~1.25x, and must be priced."""
    read = cost_of(
        response("claude-opus-4-8", cache_read_input_tokens=1_000_000, output_tokens=0)
    )
    write = cost_of(
        response("claude-opus-4-8", cache_creation_input_tokens=1_000_000, output_tokens=0)
    )
    assert read == pytest.approx(0.50)   # 5.00 * 0.1
    assert write == pytest.approx(6.25)  # 5.00 * 1.25


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("anthropic.claude-opus-4-8", "claude-opus-4-8"),      # Bedrock
        ("us.anthropic.claude-sonnet-5", "claude-sonnet-5"),   # regional Bedrock
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),     # dated snapshot
        ("gpt-4", None),
        ("claude-from-the-future", None),
        (None, None),
        (42, None),
    ],
)
def test_model_normalization(model, expected):
    assert normalize(model) == expected


def test_an_unknown_model_prices_as_none_not_a_guess():
    """A silently wrong cost is worse than a missing one - you'd act on it."""
    assert cost_of(response("gpt-4", input_tokens=1000, output_tokens=100)) is None
    assert cost_of(response("claude-not-real", input_tokens=1000)) is None


def test_a_response_without_usage_prices_as_none():
    assert cost_of({"model": "claude-opus-4-8"}) is None
    assert cost_of({"model": "claude-opus-4-8", "usage": "nonsense"}) is None
    assert cost_of("not a response") is None


def test_every_price_table_entry_is_a_sane_pair():
    for model, (input_price, output_price) in PRICES.items():
        assert input_price > 0 and output_price > 0, model
        assert output_price > input_price, f"{model}: output should cost more"


def test_trajectory_cost_reports_unpriced_calls_rather_than_hiding_them():
    """Summing None as zero would make a partial total look complete."""
    entries = [
        {"step_type": "model_call", "cost_usd": 0.01},
        {"step_type": "model_call", "cost_usd": None},   # unknown model
        {"step_type": "tool_call", "cost_usd": None},    # tools cost nothing
        {"step_type": "model_call", "cost_usd": 0.02},
    ]
    total, unpriced = trajectory_cost(entries)
    assert total == pytest.approx(0.03)
    assert unpriced == 1


def test_fmt():
    assert fmt(None) == "unpriced"
    assert fmt(0.000123) == "$0.00012"
    assert fmt(1.5) == "$1.5000"


# --- the recorded corpus ---------------------------------------------------


def call_model(messages, tools=None):
    last = messages[-1]
    if isinstance(last["content"], str):
        return {
            "stop_reason": "tool_use",
            "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "id": "toolu_01", "name": "price",
                         "input": {}}],
            "usage": {"input_tokens": 400, "output_tokens": 40},
        }
    payload = json.loads(
        [b for b in last["content"] if b.get("type") == "tool_result"][-1]["content"]
    )
    text = "cheap" if payload.get("price", 0) <= LIMIT["value"] else "expensive"
    return {
        "stop_reason": "end_turn",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 520, "output_tokens": 12},
    }


# The "config" this agent runs under. Editing it stands in for a prompt change.
LIMIT = {"value": 500}


def execute_tools(response):
    return [
        {"type": "tool_result", "tool_use_id": b["id"],
         "content": json.dumps({"price": 450})}
        for b in response["content"] if b.get("type") == "tool_use"
    ]


def loop(messages, tools, call_model, execute_tools):
    while True:
        r = call_model(messages, tools)
        messages.append({"role": "assistant", "content": r["content"]})
        if r["stop_reason"] != "tool_use":
            return r
        messages.append({"role": "user", "content": execute_tools(r)})


DEPS = ([], call_model, execute_tools)
CHECK = "output contains 'cheap'"


@pytest.fixture
def agent(store):
    return record(session_name="pricing", store=store)(loop)


@pytest.fixture(autouse=True)
def reset_limit():
    LIMIT["value"] = 500
    yield
    LIMIT["value"] = 500


@pytest.fixture
def corpus(store, agent):
    ids = []
    for _ in range(3):
        agent([{"role": "user", "content": "how much?"}], *DEPS)
        ids.append(agent.last_session_id)
    return ids


def test_cost_is_recorded_per_model_call(store, agent, corpus):
    steps = [
        s for s in store.steps_for(corpus[0]) if s["step_type"] == "model_call"
    ]
    assert all(s["cost_usd"] is not None and s["cost_usd"] > 0 for s in steps)
    # 400 in + 40 out at Opus 4.8 rates.
    assert steps[0]["cost_usd"] == pytest.approx((400 * 5 + 40 * 25) / 1e6)


def test_tool_calls_have_no_cost(store, agent, corpus):
    tools = [s for s in store.steps_for(corpus[0]) if s["step_type"] == "tool_call"]
    assert tools and all(s["cost_usd"] is None for s in tools)


# --- rerun -----------------------------------------------------------------


def test_recorded_runs_excludes_forks_and_probes(store, agent, corpus):
    """Otherwise every bisect probe becomes a test case."""
    from retrial import fork

    step = next(s for s in store.steps_for(corpus[0]) if s["step_type"] == "tool_call")
    fork(from_sha=step["sha"], agent=agent, store=store, agent_args=DEPS)

    assert len(store.list_sessions()) == 4
    assert sorted(recorded_runs(store)) == sorted(corpus)


def test_rerun_reports_all_passing_when_nothing_changed(store, agent, corpus):
    result = rerun(store, CHECK, agent=agent, agent_args=DEPS)
    assert len(result["results"]) == 3
    assert result["regressed"] == []
    assert len(result["unchanged"]) == 3


def test_rerun_detects_a_regression_from_a_config_change(store, agent, corpus):
    """The headline: change the config, and recorded runs catch the break."""
    LIMIT["value"] = 100  # the "prompt change": now $450 is expensive

    result = rerun(store, CHECK, agent=agent, agent_args=DEPS)
    assert len(result["regressed"]) == 3
    regression = result["regressed"][0]
    assert regression["before"] is True and regression["after"] is False
    assert regression["before_answer"] == "cheap"
    assert regression["answer"] == "expensive"


def test_rerun_detects_a_fix(store, agent, corpus):
    LIMIT["value"] = 100
    rerun(store, CHECK, agent=agent, agent_args=DEPS)  # regressed
    LIMIT["value"] = 500
    result = rerun(store, CHECK, agent=agent, agent_args=DEPS)
    assert result["regressed"] == []


def test_resuming_at_last_costs_about_one_model_call_per_case(store, agent, corpus):
    """Pinned history, one decision re-tested: a 2-call trajectory re-executes 1 call."""
    result = rerun(store, CHECK, agent=agent, where="last", agent_args=DEPS)
    assert result["model_calls"] == 3  # one per case
    assert all(r["model_calls"] == 1 for r in result["results"])


def test_resuming_at_first_re_runs_everything(store, agent, corpus):
    result = rerun(store, CHECK, agent=agent, where="first", agent_args=DEPS)
    assert result["model_calls"] == 6  # two per case
    assert all(r["model_calls"] == 2 for r in result["results"])


def test_rerun_costs_are_reported(store, agent, corpus):
    result = rerun(store, CHECK, agent=agent, where="last", agent_args=DEPS)
    assert result["cost_usd"] is not None and result["cost_usd"] > 0
    # Only the re-executed suffix is billed; the replayed prefix is free.
    assert result["cost_usd"] == pytest.approx(3 * (520 * 5 + 12 * 25) / 1e6)


def test_a_case_that_errors_does_not_stop_the_suite(store, agent, corpus):
    def explode(messages, tools=None):
        raise RuntimeError("model down")

    result = rerun(store, CHECK, agent=agent, agent_args=([], explode, execute_tools))
    assert len(result["errored"]) == 3
    assert all("model down" in r["error"] for r in result["errored"])
    assert result["regressed"] == []


def test_rerun_refuses_an_empty_corpus(store, agent):
    with pytest.raises(RetrialError, match="no recorded runs"):
        rerun(store, CHECK, agent=agent, agent_args=DEPS)


def test_rerun_rejects_a_bad_from_value(store, agent, corpus):
    with pytest.raises(RetrialError, match="must be 'first' or 'last'"):
        rerun(store, CHECK, agent=agent, where="middle", agent_args=DEPS)


def test_a_trace_recorded_before_cost_tracking_is_still_priced():
    """Old traces must not be stranded as 'unpriced'.

    A step with cost_usd=None still carries model and usage, so the cost is recoverable.
    """
    from retrial.pricing import cost_of_step

    old = {
        "step_type": "model_call",
        "cost_usd": None,  # recorded before pricing existed
        "output": {"model": "claude-opus-4-8",
                   "usage": {"input_tokens": 1_000_000, "output_tokens": 0}},
    }
    assert cost_of_step(old) == pytest.approx(5.00)


def test_a_stored_cost_is_never_overwritten_by_todays_prices():
    """Re-pricing history would quietly rewrite what you actually paid."""
    from retrial.pricing import cost_of_step

    entry = {
        "step_type": "model_call",
        "cost_usd": 999.0,  # whatever it cost back then
        "output": {"model": "claude-opus-4-8",
                   "usage": {"input_tokens": 1_000_000, "output_tokens": 0}},
    }
    assert cost_of_step(entry) == 999.0


def test_tool_steps_are_never_priced():
    from retrial.pricing import cost_of_step

    assert cost_of_step({"step_type": "tool_call", "cost_usd": None, "output": []}) is None


# --- where you resume decides whether the test is worth anything ------------


def two_tool_call_model(messages, tools=None):
    """search -> decide -> confirm. The budget decision lives at `search`."""
    last = messages[-1]
    if isinstance(last["content"], str):
        return _t("tool_use", [{"type": "tool_use", "id": "toolu_s", "name": "search",
                                "input": {}}])
    results = [b for b in last["content"] if b.get("type") == "tool_result"]
    latest = results[-1]
    payload = json.loads(latest["content"])

    if latest["tool_use_id"] == "toolu_s":
        if payload["price"] > LIMIT["value"]:
            return _t("end_turn", [{"type": "text", "text": "too expensive, declining"}])
        return _t("tool_use", [{"type": "tool_use", "id": "toolu_c", "name": "confirm",
                                "input": {}}])
    # By here the decision is made; this turn only reports the code.
    return _t("end_turn", [{"type": "text", "text": f"cheap - code {payload['code']}"}])


def _t(stop, content):
    return {"stop_reason": stop, "model": "claude-opus-4-8", "content": content,
            "usage": {"input_tokens": 400, "output_tokens": 20}}


def two_tool_executor(response):
    out = []
    for b in response["content"]:
        if b.get("type") != "tool_use":
            continue
        payload = {"price": 450} if b["name"] == "search" else {"code": "QX7R2M"}
        out.append({"type": "tool_result", "tool_use_id": b["id"],
                    "content": json.dumps(payload)})
    return out


TWO_DEPS = ([], two_tool_call_model, two_tool_executor)


@pytest.fixture
def two_tool_corpus(store, agent):
    agent([{"role": "user", "content": "book it"}], *TWO_DEPS)
    return agent.last_session_id


def test_resuming_at_last_misses_a_regression_it_should_catch(store, agent, two_tool_corpus):
    """The trap: `--from last` resumes after the booking already happened, so the suite says "still
    passing". That is why `first` is the default.
    """
    LIMIT["value"] = 100  # the config change: $450 is now too expensive

    vacuous = rerun(store, CHECK, agent=agent, where="last", agent_args=TWO_DEPS)
    assert vacuous["regressed"] == []          # <- misses it
    assert vacuous["model_calls"] == 1         # <- and looks cheap doing so


def test_resuming_at_the_deciding_tool_catches_it_in_one_call(store, agent, two_tool_corpus):
    """--at-tool is the option that is both cheap AND meaningful."""
    LIMIT["value"] = 100

    result = rerun(store, CHECK, agent=agent, at_tool="search", agent_args=TWO_DEPS)
    assert len(result["regressed"]) == 1
    assert result["regressed"][0]["answer"] == "too expensive, declining"
    assert result["model_calls"] == 1  # same cost as the vacuous probe above
    assert result["resumed_from"] == "tool:search"


def test_the_default_catches_it_too(store, agent, two_tool_corpus):
    LIMIT["value"] = 100
    result = rerun(store, CHECK, agent=agent, agent_args=TWO_DEPS)
    assert result["resumed_from"] == "first"
    assert len(result["regressed"]) == 1


def test_at_tool_skips_runs_that_never_called_that_tool(store, agent, two_tool_corpus):
    result = rerun(store, CHECK, agent=agent, at_tool="nonexistent", agent_args=TWO_DEPS)
    assert len(result["errored"]) == 1
    assert "no forkable 'nonexistent' tool call" in result["errored"][0]["error"]
