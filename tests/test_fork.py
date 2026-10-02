"""Forking is real re-execution.

`test_fork_matches_counterfactual` is the milestone-0 assertion as a regression test.
"""

import json

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import fork, record
from retrial.errors import IntegrationError, ReplayIntegrityError


def build_agent(store):
    return record(session_name="booking", store=store)(raw_agent)


def run_original(store, opening, price=450):
    # Return the session id explicitly: `agent.last_session_id` moves on each run.
    agent = build_agent(store)
    response = agent(opening, TOOLS, fake_model, make_executor(price))
    return agent, agent.last_session_id, response


def tool_step(store, session_id):
    return next(
        s for s in store.steps_for(session_id) if s["step_type"] == "tool_call"
    )


PRICE_999 = {
    "op": "replace",
    "path": "/output/0/content",
    "value": json.dumps({"flight_price": 999}),
}


def final_text(store, session_id):
    steps = store.steps_for(session_id)
    last_model = [s for s in steps if s["step_type"] == "model_call"][-1]
    return last_model["output"]["content"][-1]["text"]


def test_fork_matches_counterfactual(store, opening):
    """Fork == a from-scratch run in a world where the edit was always true."""
    agent, original_id, original = run_original(store, opening)
    assert original.content[-1]["text"] == "Booked for $450."

    step = tool_step(store, original_id)
    fork_id = fork(
        from_sha=step["sha"],
        edit=PRICE_999,
        agent=agent,
        store=store,
        # The world still returns $450; only the spliced fact changed.
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    counterfactual = raw_agent(
        [{"role": "user", "content": "Book me AUS to SFO."}],
        TOOLS,
        fake_model,
        make_executor(999),
    )

    assert final_text(store, fork_id) == counterfactual.content[-1]["text"]
    assert final_text(store, fork_id) == "Over budget. Need approval first."


def tools_run_in(store, session_id):
    return [
        b["name"]
        for s in store.steps_for(session_id)
        if s["step_type"] == "tool_call"
        for b in s["input"]
    ]


def test_fork_diverges_structurally_from_the_original(store, opening):
    """Relabeling stored JSON could never produce a step that never happened."""
    agent, original_id, _ = run_original(store, opening)
    assert tools_run_in(store, original_id) == ["search_flight"]

    step = tool_step(store, original_id)
    fork_id = fork(
        from_sha=step["sha"],
        edit=PRICE_999,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    # check_budget is only reachable because of the edit.
    assert tools_run_in(store, fork_id) == ["check_budget"]


def test_a_fork_stores_only_the_re_executed_suffix(store, opening):
    """The replayed prefix is not duplicated into the fork's session.

    A fork's own steps begin at the resume point; the rest stay in the parent.
    """
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)

    fork_id = fork(
        from_sha=step["sha"],
        edit=PRICE_999,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    session = store.get_session(fork_id)
    fork_steps = store.steps_for(fork_id)

    # Numbered from zero, and none re-record the prefix.
    assert [s["step_number"] for s in fork_steps] == [0, 1, 2]
    assert not any(s["sha"] in {p["sha"] for p in store.steps_for(original_id)}
                   for s in fork_steps)

    # Parent prefix + fork suffix, longer because of the extra tool call.
    prefix_len = session["forked_at_step"] + 1
    trajectory = prefix_len + len(fork_steps)
    assert trajectory > len(store.steps_for(original_id))


def test_fork_leaves_the_original_session_intact(store, opening):
    agent, original_id, _ = run_original(store, opening)
    before = store.steps_for(original_id)
    before_shas = [s["sha"] for s in before]

    step = tool_step(store, original_id)
    fork(
        from_sha=step["sha"],
        edit=PRICE_999,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    after = store.steps_for(original_id)
    assert [s["sha"] for s in after] == before_shas
    assert after == before


def test_fork_records_provenance(store, opening):
    """`retrial log` must be able to show where AND what."""
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)

    fork_id = fork(
        from_sha=step["sha"],
        edit=PRICE_999,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    session = store.get_session(fork_id)
    assert session["parent_session_id"] == original_id
    assert session["parent_sha"] == step["sha"]
    assert session["forked_at_step"] == step["step_number"]
    assert json.loads(session["edit_json"]) == {"type": "patch", "patch": PRICE_999}


def test_fork_by_sha_prefix(store, opening):
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)

    fork_id = fork(
        from_sha=step["sha"][:7],
        edit=PRICE_999,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )
    assert store.get_session(fork_id)["parent_sha"] == step["sha"]


def test_callback_edit_works_and_is_recorded_as_not_reproducible(store, opening):
    """The escape hatch: an edit that depends on the recorded content."""
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)

    def triple_the_price(step_dict):
        payload = json.loads(step_dict["output"][0]["content"])
        payload["flight_price"] *= 3  # 450 -> 1350, which no patch could know
        step_dict["output"][0]["content"] = json.dumps(payload)
        return step_dict

    fork_id = fork(
        from_sha=step["sha"],
        edit=triple_the_price,
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )

    assert final_text(store, fork_id) == "Over budget. Need approval first."
    recorded = json.loads(store.get_session(fork_id)["edit_json"])
    assert recorded["type"] == "callback"
    assert recorded["repr"] == "triple_the_price"
    assert "not round-trip" in recorded["note"]


def test_fork_a_model_call_reruns_the_decision(store, opening):
    """Forking a model_call patches the history the model saw."""
    agent, original_id, _ = run_original(store, opening)
    steps = store.steps_for(original_id)
    second_model_call = [s for s in steps if s["step_type"] == "model_call"][1]

    fork_id = fork(
        from_sha=second_model_call["sha"],
        edit={
            "op": "replace",
            "path": "/input/messages/2/content/0/content",
            "value": json.dumps({"flight_price": 999}),
        },
        agent=agent,
        store=store,
        agent_args=(TOOLS, fake_model, make_executor(450)),
    )
    assert final_text(store, fork_id) == "Over budget. Need approval first."


# --- refusing to guess ------------------------------------------------------


def test_refuses_when_the_loop_transformed_the_tool_result(store, opening):
    """If the recorded output is not in the history verbatim, refuse."""

    def mangling_agent(messages, tools, call_model, execute_tools):
        while True:
            response = call_model(messages, tools)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return response
            results = execute_tools(response)
            # A trailing space: valid JSON, but no longer the recorded value.
            rewritten = [dict(r, content=r["content"] + " ") for r in results]
            messages.append({"role": "user", "content": rewritten})

    agent = record(session_name="mangling", store=store)(mangling_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))
    original_id = agent.last_session_id

    step = tool_step(store, original_id)
    with pytest.raises(ReplayIntegrityError, match="transformed the result"):
        fork(
            from_sha=step["sha"],
            edit=PRICE_999,
            agent=agent,
            store=store,
            agent_args=(TOOLS, fake_model, make_executor(450)),
        )


def test_refuses_to_fork_a_step_whose_successor_state_was_never_seen(store, opening):
    """A crashed run's trailing tool_call has no state observed after it."""

    def crashing_agent(messages, tools, call_model, execute_tools):
        response = call_model(messages, tools)
        messages.append({"role": "assistant", "content": response.content})
        execute_tools(response)
        raise RuntimeError("boom")

    agent = record(session_name="crashing", store=store)(crashing_agent)
    with pytest.raises(RuntimeError):
        agent(opening, TOOLS, fake_model, make_executor(450))
    assert store.get_session(agent.last_session_id)["status"] == "failed"

    step = tool_step(store, agent.last_session_id)
    with pytest.raises(ReplayIntegrityError, match="never observed"):
        fork(
            from_sha=step["sha"],
            edit=PRICE_999,
            agent=agent,
            store=store,
            agent_args=(TOOLS, fake_model, make_executor(450)),
        )


def test_refuses_an_edit_that_invents_a_tool_result(store, opening):
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)

    def invent(step_dict):
        step_dict["output"].append({"tool_use_id": "toolu_99", "content": "{}"})
        return step_dict

    with pytest.raises(ReplayIntegrityError, match="never recorded"):
        fork(
            from_sha=step["sha"],
            edit=invent,
            agent=agent,
            store=store,
            agent_args=(TOOLS, fake_model, make_executor(450)),
        )


def test_refuses_an_edit_that_drops_a_tool_result(store, opening):
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)

    with pytest.raises(ReplayIntegrityError, match="dangling"):
        fork(
            from_sha=step["sha"],
            edit={"op": "remove", "path": "/output/0"},
            agent=agent,
            store=store,
            agent_args=(TOOLS, fake_model, make_executor(450)),
        )


def test_fork_without_an_agent_is_an_error(store, opening):
    agent, original_id, _ = run_original(store, opening)
    step = tool_step(store, original_id)
    with pytest.raises(IntegrationError, match="nothing to run"):
        fork(from_sha=step["sha"], edit=PRICE_999, store=store)
