"""Ablation and sweep.

The agent searches a fare, checks the budget, then books:

    search_flight  -> the fare; the booking decision turns on it.
    check_budget   -> advisory only; this agent ignores the verdict.
    book_flight    -> produces the confirmation the check looks for.

So `check_budget` is not load-bearing, and ablation must say so.
"""

import json

import pytest

from retrial import record
from retrial.errors import RetrialError
from retrial.explore import UNAVAILABLE, ablate, sweep

BUDGET = 600


def call_model(messages, tools=None):
    last = messages[-1]
    if isinstance(last["content"], str):
        return _reply("toolu_search", "search_flight", {"route": "AUS-SFO"})

    results = [b for b in last["content"] if b.get("type") == "tool_result"]
    latest = results[-1]
    payload = json.loads(latest["content"])

    if latest["tool_use_id"] == "toolu_search":
        if "error" in payload:
            return _text("I couldn't get a fare.")
        return _reply("toolu_budget", "check_budget", {"amount": payload["fare"]})

    if latest["tool_use_id"] == "toolu_budget":
        # Ignores the budget verdict and re-reads the fare, so check_budget is not load-bearing.
        fare = _fare_in(messages)
        if fare is None or fare > BUDGET:
            return _text(f"Fare of ${fare} is over budget. Not booking.")
        return _reply("toolu_book", "book_flight", {"amount": fare})

    if latest["tool_use_id"] == "toolu_book":
        if "error" in payload:
            return _text("Booking failed.")
        return _text(f"Confirmed: booked for ${payload['amount']}.")

    raise AssertionError(f"no scripted reply for {latest}")


def _fare_in(messages):
    for message in messages:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("tool_use_id") == "toolu_search":
                payload = json.loads(block["content"])
                return payload.get("fare")
    return None


def _reply(tool_id, name, args):
    return {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": args}],
    }


def _text(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def execute_tools(response):
    out = []
    for block in response["content"]:
        if block.get("type") != "tool_use":
            continue
        name = block["name"]
        if name == "search_flight":
            payload = {"fare": 450, "carrier": "Southwest"}
        elif name == "check_budget":
            payload = {"approved": True, "limit": BUDGET}
        else:
            payload = {"confirmation": "QX7R2M", "amount": block["input"]["amount"]}
        out.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": json.dumps(payload),
            }
        )
    return out


def loop(messages, tools, call_model, execute_tools):
    while True:
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response["content"]})
        if response["stop_reason"] != "tool_use":
            return response
        messages.append({"role": "user", "content": execute_tools(response)})


DEPS = ([], call_model, execute_tools)
CHECK = "output contains 'Confirmed'"


@pytest.fixture
def agent(store):
    return record(session_name="booking", store=store)(loop)


@pytest.fixture
def recorded(store, agent, opening):
    response = agent(opening, *DEPS)
    assert response["content"][-1]["text"] == "Confirmed: booked for $450."
    return agent.last_session_id


def search_sha(store, session_id):
    for step in store.steps_for(session_id):
        if step["step_type"] == "tool_call" and step["input"][0]["name"] == "search_flight":
            return step["sha"]
    raise AssertionError("no search_flight step")


# --- ablate ----------------------------------------------------------------


def test_ablation_separates_facts_that_mattered_from_tools_that_merely_ran(
    store, agent, recorded
):
    """The headline. check_budget ran, but the outcome did not depend on it."""
    result = ablate(store, recorded, CHECK, agent=agent, agent_args=DEPS)

    assert result["baseline_passed"] is True
    verdicts = {
        p["tools"][0]: p["flipped"] for p in result["probes"] if not p["error"]
    }
    assert verdicts == {
        "search_flight": True,   # blank the fare -> no booking
        "check_budget": False,   # blank the verdict -> books anyway
        "book_flight": True,     # blank the confirmation -> nothing to confirm
    }

    sound = [p["tools"][0] for p in result["not_load_bearing"]]
    assert sound == ["check_budget"]


def test_one_re_execution_per_tool_call(store, agent, recorded):
    result = ablate(store, recorded, CHECK, agent=agent, agent_args=DEPS)
    assert result["re_executions"] == 3
    assert len(result["probes"]) == 3


def test_every_probe_is_a_recorded_session(store, agent, recorded):
    """Auditable, like bisect's. You can retrial log any of them."""
    before = len(store.list_sessions())
    result = ablate(store, recorded, CHECK, agent=agent, agent_args=DEPS)

    assert len(store.list_sessions()) == before + result["re_executions"]
    for probe in result["probes"]:
        session = store.get_session(probe["session_id"])
        assert session["parent_session_id"] == recorded
        assert json.loads(session["edit_json"])["type"] == "patch"


def test_the_default_perturbation_blanks_every_result_in_a_step(store, agent, recorded):
    """A step with parallel tool calls must be fully ablated, not just its first result."""
    from retrial.explore import _default_perturbation

    step = {"output": [{"content": "a"}, {"content": "b"}, {"content": "c"}]}
    patch = _default_perturbation(step)
    assert [op["path"] for op in patch] == [
        "/output/0/content",
        "/output/1/content",
        "/output/2/content",
    ]
    assert all(op["value"] == UNAVAILABLE for op in patch)


def test_a_custom_perturbation_can_be_a_callable(store, agent, recorded):
    """The escape hatch, same shape as the fork edit API."""

    def halve_the_fare(step):
        payload = json.loads(step["output"][0]["content"])
        if "fare" in payload:
            payload["fare"] = payload["fare"] * 10  # 4500: way over budget
        return [
            {
                "op": "replace",
                "path": "/output/0/content",
                "value": json.dumps(payload),
            }
        ]

    result = ablate(
        store, recorded, CHECK, agent=agent, perturbation=halve_the_fare, agent_args=DEPS
    )
    by_tool = {p["tools"][0]: p for p in result["probes"]}
    assert by_tool["search_flight"]["flipped"] is True
    assert "over budget" in by_tool["search_flight"]["answer"]


def test_a_probe_that_breaks_the_agent_is_reported_not_raised(store, agent, recorded):
    """One bad probe must not destroy the other results."""

    def corrupt(step):
        return [{"op": "replace", "path": "/output/0/content", "value": "not json"}]

    result = ablate(
        store, recorded, CHECK, agent=agent, perturbation=corrupt, agent_args=DEPS
    )
    assert result["inconclusive"], "expected the JSON parse to break the agent"
    assert all(p["flipped"] is None for p in result["inconclusive"])
    assert all("JSONDecodeError" in p["error"] for p in result["inconclusive"])


def test_ablate_refuses_a_session_with_no_tool_calls(store, opening):
    """A run with no recorded facts has nothing to ablate.

    The check must pass here, or the baseline guard fires first.
    """

    def no_tools(messages, tools, call_model, execute_tools):
        return call_model(messages, tools)

    agent = record(session_name="chat", store=store)(no_tools)
    agent(opening, [], lambda m, t: _text("hello"), execute_tools)

    with pytest.raises(RetrialError, match="no forkable tool_call steps"):
        ablate(
            store, agent.last_session_id, "output contains 'hello'",
            agent=agent, agent_args=DEPS,
        )


def test_on_probe_streams_progress(store, agent, recorded):
    seen = []
    ablate(store, recorded, CHECK, agent=agent, agent_args=DEPS, on_probe=seen.append)
    assert [p["tools"][0] for p in seen] == [
        "search_flight",
        "check_budget",
        "book_flight",
    ]


# --- sweep -----------------------------------------------------------------


def fares(*amounts):
    return [json.dumps({"fare": a, "carrier": "Southwest"}) for a in amounts]


def test_sweep_finds_the_threshold(store, agent, recorded):
    """At what fare does it stop booking? The agent's rule is > $600."""
    sha = search_sha(store, recorded)
    result = sweep(
        store, sha, fares(300, 500, 700, 900), agent=agent, check=CHECK, agent_args=DEPS
    )

    assert [p["passed"] for p in result["probes"]] == [True, True, False, False]
    assert len(result["boundaries"]) == 1
    crossing = result["boundaries"][0]
    assert json.loads(crossing["from"]["value"])["fare"] == 500
    assert json.loads(crossing["to"]["value"])["fare"] == 700


def test_sweep_reports_the_exact_boundary_when_values_bracket_it(store, agent, recorded):
    sha = search_sha(store, recorded)
    result = sweep(
        store, sha, fares(599, 600, 601), agent=agent, check=CHECK, agent_args=DEPS
    )
    # The rule is `> 600`, so 600 books and 601 does not.
    assert [p["passed"] for p in result["probes"]] == [True, True, False]
    crossing = result["boundaries"][0]
    assert json.loads(crossing["from"]["value"])["fare"] == 600
    assert json.loads(crossing["to"]["value"])["fare"] == 601


def test_sweep_without_a_check_just_reports_answers(store, agent, recorded):
    sha = search_sha(store, recorded)
    result = sweep(store, sha, fares(300, 900), agent=agent, agent_args=DEPS)

    assert result["check"] is None
    assert result["boundaries"] == []
    assert all(p["passed"] is None for p in result["probes"])
    assert "Confirmed: booked for $300." == result["probes"][0]["answer"]
    assert "over budget" in result["probes"][1]["answer"]


def test_sweep_with_no_flip_reports_no_threshold(store, agent, recorded):
    sha = search_sha(store, recorded)
    result = sweep(
        store, sha, fares(100, 200, 300), agent=agent, check=CHECK, agent_args=DEPS
    )
    assert all(p["passed"] for p in result["probes"])
    assert result["boundaries"] == []


def test_sweep_costs_one_re_execution_per_value(store, agent, recorded):
    sha = search_sha(store, recorded)
    result = sweep(store, sha, fares(100, 200, 300), agent=agent, agent_args=DEPS)
    assert result["re_executions"] == 3


def test_sweep_needs_values(store, agent, recorded):
    with pytest.raises(RetrialError, match="at least one value"):
        sweep(store, search_sha(store, recorded), [], agent=agent, agent_args=DEPS)


def test_sweep_defaults_to_one_sample_per_value(store, agent, recorded):
    """A single sample still reports a rate."""
    sha = search_sha(store, recorded)
    result = sweep(store, sha, fares(300, 900), agent=agent, check=CHECK, agent_args=DEPS)

    assert result["samples"] == 1
    assert [p["pass_rate"] for p in result["probes"]] == [1.0, 0.0]
    assert all(len(p["runs"]) == 1 for p in result["probes"])
    assert all(p["evaluated"] == 1 for p in result["probes"])


# --- sweep with samples ----------------------------------------------------
#
# The stand-in's wobble is scripted, so the expected rates are exact.


def flaky(*decisions):
    """A call_model that books or refuses per a fixed list, one entry per re-execution.

    "boom" raises, standing in for a failed probe.
    """
    verdicts = iter(decisions)

    def call_model(messages, tools=None):
        last = messages[-1]
        if isinstance(last["content"], str):
            return _reply("toolu_search", "search_flight", {"route": "AUS-SFO"})

        latest = [b for b in last["content"] if b.get("type") == "tool_result"][-1]
        payload = json.loads(latest["content"])

        if latest["tool_use_id"] == "toolu_search":
            return _reply("toolu_budget", "check_budget", {"amount": payload["fare"]})

        if latest["tool_use_id"] == "toolu_budget":
            verdict = next(verdicts)
            if verdict == "boom":
                raise RuntimeError("model unavailable")
            fare = _fare_in(messages)
            if verdict:
                return _reply("toolu_book", "book_flight", {"amount": fare})
            return _text(f"Fare of ${fare} is over budget. Not booking.")

        if latest["tool_use_id"] == "toolu_book":
            return _text(f"Confirmed: booked for ${payload['amount']}.")

        raise AssertionError(f"no scripted reply for {latest}")

    return ([], call_model, execute_tools)


def test_sweep_samples_each_value_and_reports_the_rate(store, agent, recorded):
    """The headline: a threshold becomes a crossing of two rates."""
    sha = search_sha(store, recorded)
    result = sweep(
        store,
        sha,
        fares(590, 610),
        agent=agent,
        check=CHECK,
        samples=3,
        # value 590 books 2 of 3; value 610 books 1 of 3.
        agent_args=flaky(False, True, True, True, False, False),
    )

    assert result["samples"] == 3
    assert result["re_executions"] == 6
    assert [p["passes"] for p in result["probes"]] == [2, 1]
    assert [p["pass_rate"] for p in result["probes"]] == [2 / 3, 1 / 3]
    # Majority, not unanimity: 2-of-3 books, so 590 is on the booking side.
    assert [p["passed"] for p in result["probes"]] == [True, False]
    assert len(result["boundaries"]) == 1


def test_sweep_samples_are_separate_re_executions(store, agent, recorded):
    sha = search_sha(store, recorded)
    result = sweep(
        store, sha, fares(500), agent=agent, samples=3, agent_args=flaky(True, True, True)
    )

    probe = result["probes"][0]
    assert len(probe["runs"]) == 3
    assert len({r["session_id"] for r in probe["runs"]}) == 3


def test_sweep_shows_an_answer_that_agrees_with_the_verdict(store, agent, recorded):
    """The first run of each value dissents, so the probe must not just show run 0."""
    sha = search_sha(store, recorded)
    result = sweep(
        store,
        sha,
        fares(590, 610),
        agent=agent,
        check=CHECK,
        samples=3,
        agent_args=flaky(False, True, True, True, False, False),
    )

    booked, refused = result["probes"]
    assert booked["passed"] is True
    assert "Confirmed" in booked["answer"]
    assert refused["passed"] is False
    assert "Not booking" in refused["answer"]


def test_sweep_counts_a_tie_as_not_passing(store, agent, recorded):
    """A tie is not a majority, and must be deterministic."""
    sha = search_sha(store, recorded)
    result = sweep(
        store,
        sha,
        fares(600),
        agent=agent,
        check=CHECK,
        samples=2,
        agent_args=flaky(True, False),
    )

    probe = result["probes"][0]
    assert probe["pass_rate"] == 0.5
    assert probe["passed"] is False


def test_sweep_scores_a_value_on_the_samples_that_survived(store, agent, recorded):
    """One failed probe out of three is noise, not a verdict about the value."""
    sha = search_sha(store, recorded)
    result = sweep(
        store,
        sha,
        fares(500),
        agent=agent,
        check=CHECK,
        samples=3,
        agent_args=flaky("boom", True, True),
    )

    probe = result["probes"][0]
    assert probe["evaluated"] == 2
    assert probe["passes"] == 2
    assert probe["pass_rate"] == 1.0
    assert probe["passed"] is True
    # The probe as a whole is not an error: two runs did answer.
    assert probe["error"] is None
    assert probe["runs"][0]["error"].startswith("RuntimeError")


def test_sweep_reports_a_value_whose_every_sample_failed(store, agent, recorded):
    """No answer behind this value, so no rate, and not a 0/0."""
    sha = search_sha(store, recorded)
    result = sweep(
        store,
        sha,
        fares(500),
        agent=agent,
        check=CHECK,
        samples=2,
        agent_args=flaky("boom", "boom"),
    )

    probe = result["probes"][0]
    assert probe["evaluated"] == 0
    assert probe["passes"] is None
    assert probe["pass_rate"] is None
    assert probe["passed"] is None
    assert "model unavailable" in probe["error"]


def test_sweep_without_a_check_reports_no_rate(store, agent, recorded):
    sha = search_sha(store, recorded)
    result = sweep(
        store, sha, fares(500), agent=agent, samples=2, agent_args=flaky(True, False)
    )

    probe = result["probes"][0]
    assert probe["evaluated"] == 2
    assert probe["passes"] is None
    assert probe["pass_rate"] is None
    assert probe["passed"] is None


def test_sweep_needs_at_least_one_sample(store, agent, recorded):
    with pytest.raises(RetrialError, match="at least 1"):
        sweep(
            store,
            search_sha(store, recorded),
            fares(500),
            agent=agent,
            samples=0,
            agent_args=DEPS,
        )


def test_ablate_refuses_when_the_baseline_check_fails(store, agent, recorded):
    """Ablate and bisect are duals; each must refuse the other's job.

    With a failing baseline every probe would read NOT load-bearing.
    """
    with pytest.raises(RetrialError, match="no good outcome to ablate") as exc:
        ablate(
            store, recorded, "output contains 'never appears'",
            agent=agent, agent_args=DEPS,
        )
    # It must point at the tool that does handle a failing run.
    assert "retrial bisect" in str(exc.value)
    # And show the answer, so a mistyped check is obvious immediately.
    assert "Confirmed: booked for $450." in str(exc.value)


def test_ablate_and_bisect_refuse_opposite_baselines(store, agent, recorded, opening):
    """The duality asserted directly: exactly one tool takes the job."""
    from retrial.bisect import bisect

    # This run PASSES the check: ablate accepts it, bisect refuses it.
    ablate(store, recorded, CHECK, agent=agent, agent_args=DEPS)
    with pytest.raises(RetrialError, match="already passes"):
        bisect(store, recorded, CHECK, agent=agent, agent_args=DEPS)
