"""Provider adapters.

`test_fork_matches_counterfactual_*` re-run the milestone assertion from `test_fork.py` through
a real adapter driving a fake SDK client. No network and no key.
"""

import json
from types import SimpleNamespace

import pytest
from conftest import raw_agent

from retrial import fork, record
from retrial.pricing import FREE, PRICES, cost_of, normalize, register_prices
from retrial.providers import ModelResponse, tool_result, tool_uses
from retrial.providers.gemini import GeminiAdapter, to_gemini_contents
from retrial.providers.gemini import to_usage as gemini_usage
from retrial.providers.openai import (
    OpenAIAdapter,
    to_openai_messages,
    to_openai_tool,
)
from retrial.providers.openai import (
    to_usage as openai_usage,
)
from retrial.serialize import to_jsonable

TOOLS = [
    {
        "name": "search_flight",
        "description": "Look up a fare.",
        "input_schema": {
            "type": "object",
            "properties": {"route": {"type": "string"}},
            "required": ["route"],
        },
    },
    {"name": "check_budget", "description": "Check the budget.", "input_schema": {}},
]


# --- fake SDK clients ------------------------------------------------------
#
# Each answers in its provider's wire shape and branches on the tool result.


class FakeOpenAI:
    """Enough of `client.chat.completions.create` to drive the loop."""

    def __init__(self, as_objects=False):
        self.as_objects = as_objects
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        payload = _openai_reply(request["messages"])
        return _objectify(payload) if self.as_objects else payload


def _openai_reply(messages):
    last = messages[-1]

    if last["role"] == "user":
        return _openai_payload(
            text="Looking up that flight.",
            calls=[("call_01", "search_flight", {"route": "AUS-SFO"})],
            finish="tool_calls",
        )

    if last["role"] == "tool":
        result = json.loads(last["content"])
        if last["tool_call_id"] == "call_01":
            price = result["flight_price"]
            if price <= 500:
                return _openai_payload(text=f"Booked for ${price}.", finish="stop")
            return _openai_payload(
                text=f"${price} is over budget.",
                calls=[("call_02", "check_budget", {"amount": price})],
                finish="tool_calls",
            )
        return _openai_payload(text="Over budget. Need approval first.", finish="stop")

    raise AssertionError(f"FakeOpenAI has no policy for: {last}")


def _openai_payload(text=None, calls=(), finish="stop"):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for call_id, name, args in calls
        ]
    return {
        "id": "chatcmpl-x",
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class FakeGemini:
    """Enough of `client.models.generate_content` to drive the loop."""

    def __init__(self):
        self.requests = []
        self.models = SimpleNamespace(generate_content=self._create)

    def _create(self, **request):
        self.requests.append(request)
        return _gemini_reply(request["contents"])


def _gemini_reply(contents):
    parts = contents[-1]["parts"]
    responses = [p["function_response"] for p in parts if "function_response" in p]

    if not responses:
        return _gemini_payload(
            text="Looking up that flight.",
            calls=[("search_flight", {"route": "AUS-SFO"})],
        )

    answer = responses[-1]["response"]
    if responses[-1]["name"] == "search_flight":
        price = answer["flight_price"]
        if price <= 500:
            return _gemini_payload(text=f"Booked for ${price}.")
        return _gemini_payload(
            text=f"${price} is over budget.",
            calls=[("check_budget", {"amount": price})],
        )
    return _gemini_payload(text="Over budget. Need approval first.")


def _gemini_payload(text=None, calls=()):
    parts = []
    if text:
        parts.append({"text": text})
    for name, args in calls:
        parts.append({"function_call": {"name": name, "args": args}})
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finish_reason": "STOP",
            }
        ],
        "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 20},
        "model_version": "gemini-test",
    }


def _objectify(value):
    """Turn a captured payload into attribute-access objects, as an SDK returns."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _objectify(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_objectify(v) for v in value]
    return value


# --- the shared agent ------------------------------------------------------


def make_executor(flight_price):
    """The world, in canonical blocks. Identical for every provider."""

    def execute_tools(response):
        out = []
        for block in tool_uses(response):
            if block["name"] == "search_flight":
                payload = {"flight_price": flight_price}
            else:
                payload = {"approved": False}
            out.append(tool_result(block["id"], payload))
        return out

    return execute_tools


def run_and_fork(store, adapter, price=450, edited=999):
    """Record a run through `adapter`, then fork its tool_call step."""
    agent = record(session_name="booking", store=store)(raw_agent)
    opening = [{"role": "user", "content": "Book me AUS to SFO."}]
    agent(opening, TOOLS, adapter, make_executor(price))
    original_id = agent.last_session_id

    step = next(
        s for s in store.steps_for(original_id) if s["step_type"] == "tool_call"
    )
    fork_id = fork(
        from_sha=step["sha"],
        edit={
            "op": "replace",
            "path": "/output/0/content",
            "value": json.dumps({"flight_price": edited}),
        },
        agent=agent,
        store=store,
        # The world still returns the original price; only the spliced fact changed.
        agent_args=(TOOLS, adapter, make_executor(price)),
    )
    return original_id, fork_id


def final_text(store, session_id):
    steps = store.steps_for(session_id)
    last_model = [s for s in steps if s["step_type"] == "model_call"][-1]
    return last_model["output"]["content"][-1]["text"]


# --- the tests that matter -------------------------------------------------


@pytest.mark.parametrize("as_objects", [False, True], ids=["dicts", "sdk-objects"])
def test_fork_matches_counterfactual_openai(store, as_objects):
    """The milestone assertion, run through the OpenAI adapter.

    Covers both response shapes: SDK objects and plain dicts.
    """
    adapter = OpenAIAdapter("gpt-test", client=FakeOpenAI(as_objects), system="Book flights.")
    _, fork_id = run_and_fork(store, adapter)

    counterfactual = raw_agent(
        [{"role": "user", "content": "Book me AUS to SFO."}],
        TOOLS,
        OpenAIAdapter("gpt-test", client=FakeOpenAI(as_objects)),
        make_executor(999),
    )

    assert final_text(store, fork_id) == counterfactual.content[-1]["text"]
    assert final_text(store, fork_id) == "Over budget. Need approval first."


def test_fork_matches_counterfactual_gemini(store):
    """The same assertion over a wire format that agrees with nobody."""
    adapter = GeminiAdapter("gemini-test", client=FakeGemini(), system="Book flights.")
    _, fork_id = run_and_fork(store, adapter)

    counterfactual = raw_agent(
        [{"role": "user", "content": "Book me AUS to SFO."}],
        TOOLS,
        GeminiAdapter("gemini-test", client=FakeGemini()),
        make_executor(999),
    )

    assert final_text(store, fork_id) == counterfactual.content[-1]["text"]
    assert final_text(store, fork_id) == "Over budget. Need approval first."


def test_recorded_shape_is_identical_across_providers(store):
    """A step recorded through either adapter has the same shape.

    The same keys in the same places, not the same text.
    """
    openai_id, _ = run_and_fork(store, OpenAIAdapter("gpt-test", client=FakeOpenAI()))
    gemini_id, _ = run_and_fork(store, GeminiAdapter("gemini-test", client=FakeGemini()))

    def shape(session_id):
        step = next(
            s for s in store.steps_for(session_id) if s["step_type"] == "model_call"
        )
        out = step["output"]
        return sorted(out), sorted(out["usage"]), [b["type"] for b in out["content"]]

    assert shape(openai_id) == shape(gemini_id)


def test_response_records_canonically_and_keeps_raw_out_of_the_trace():
    """`to_dict` decides what lands in the trace - `raw` is not part of it."""
    response = ModelResponse(
        content=[{"type": "text", "text": "hi"}],
        stop_reason="end_turn",
        usage={"input_tokens": 3, "output_tokens": 1},
        model="gpt-test",
        raw=object(),  # unserializable on purpose
    )
    assert to_jsonable(response) == {
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 1},
        "model": "gpt-test",
    }
    assert response.text == "hi"


# --- translators, pure and offline -----------------------------------------


def test_openai_messages_split_tool_results_into_their_own_turns():
    canonical = [
        {"role": "user", "content": "Book it."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking."},
                {"type": "tool_use", "id": "call_01", "name": "search_flight",
                 "input": {"route": "AUS-SFO"}},
            ],
        },
        {"role": "user", "content": [tool_result("call_01", {"flight_price": 450})]},
    ]
    assert to_openai_messages(canonical, system="S") == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Book it."},
        {
            "role": "assistant",
            "content": "Looking.",
            "tool_calls": [
                {
                    "id": "call_01",
                    "type": "function",
                    "function": {
                        "name": "search_flight",
                        "arguments": '{"route": "AUS-SFO"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_01", "content": '{"flight_price": 450}'},
    ]


def test_openai_drops_reasoning_blocks_from_the_request():
    """Anthropic's signed thinking blocks must not be sent to another provider."""
    canonical = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "...", "signature": "sig"},
                {"type": "text", "text": "Done."},
            ],
        }
    ]
    assert to_openai_messages(canonical) == [{"role": "assistant", "content": "Done."}]


def test_openai_tool_schema_translates_and_passes_native_through():
    assert to_openai_tool(TOOLS[0]) == {
        "type": "function",
        "function": {
            "name": "search_flight",
            "description": "Look up a fare.",
            "parameters": TOOLS[0]["input_schema"],
        },
    }
    native = {"type": "function", "function": {"name": "x", "parameters": {}}}
    assert to_openai_tool(native) is native


def test_openai_tool_use_survives_malformed_arguments():
    """A small model emitting bad JSON is recorded, not invented away."""
    adapter = OpenAIAdapter("gpt-test")
    payload = _openai_payload(finish="tool_calls")
    payload["choices"][0]["message"]["tool_calls"] = [
        {"id": "call_01", "type": "function",
         "function": {"name": "search_flight", "arguments": "{route: AUS-"}}
    ]
    block = adapter.from_response(payload).content[0]
    assert block["input"] == {"_unparsed_arguments": "{route: AUS-"}


def test_stop_reason_reflects_tool_calls_even_when_the_server_says_stop():
    """Some compatible servers mislabel a tool turn; the loop must not stall."""
    adapter = OpenAIAdapter("gpt-test")
    payload = _openai_payload(
        calls=[("call_01", "search_flight", {})], finish="stop"
    )
    assert adapter.from_response(payload).stop_reason == "tool_use"


def test_gemini_contents_use_model_role_and_function_parts():
    canonical = [
        {"role": "user", "content": "Book it."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "search_flight_0", "name": "search_flight",
                 "input": {"route": "AUS-SFO"}},
            ],
        },
        {"role": "user", "content": [tool_result("search_flight_0", {"flight_price": 450})]},
    ]
    assert to_gemini_contents(canonical) == [
        {"role": "user", "parts": [{"text": "Book it."}]},
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "search_flight", "args": {"route": "AUS-SFO"}}}
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "search_flight",
                        "response": {"flight_price": 450},
                    }
                }
            ],
        },
    ]


def test_gemini_synthesizes_a_tool_use_id_so_the_step_can_be_forked():
    """Gemini issues no call id, and a fork splices by one."""
    adapter = GeminiAdapter("gemini-test")
    response = adapter.from_response(
        _gemini_payload(calls=[("search_flight", {}), ("search_flight", {})])
    )
    ids = [b["id"] for b in response.content if b["type"] == "tool_use"]
    assert ids == ["search_flight_0", "search_flight_1"]
    assert len(set(ids)) == 2


# --- usage and cost --------------------------------------------------------


def test_openai_usage_does_not_bill_cached_tokens_twice():
    """`prompt_tokens` includes the cached part; retrial prices them apart."""
    assert openai_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
    ) == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_input_tokens": 800,
    }


def test_gemini_usage_does_not_bill_cached_tokens_twice():
    assert gemini_usage(
        {
            "prompt_token_count": 1000,
            "candidates_token_count": 50,
            "cached_content_token_count": 800,
        }
    ) == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_input_tokens": 800,
    }


def test_gemini_usage_reads_camel_case_too():
    """A recorded trace stores the SDK's camelCase spelling."""
    assert gemini_usage({"promptTokenCount": 30, "candidatesTokenCount": 5}) == {
        "input_tokens": 30,
        "output_tokens": 5,
    }


class TestPricing:
    """Registration is the supported way to price anything not built in."""

    @pytest.fixture(autouse=True)
    def restore_table(self):
        original = dict(PRICES)
        yield
        PRICES.clear()
        PRICES.update(original)

    def test_unknown_model_stays_unpriced(self):
        assert cost_of({"model": "my-finetune", "usage": {"input_tokens": 1000}}) is None

    def test_registered_model_prices(self):
        register_prices({"my-finetune": (1.0, 2.0)})
        cost = cost_of(
            {"model": "my-finetune", "usage": {"input_tokens": 1_000_000,
                                               "output_tokens": 1_000_000}}
        )
        assert cost == pytest.approx(3.0)

    def test_local_model_is_free_not_unpriced(self):
        """The distinction the whole registry exists for: $0 is a fact."""
        register_prices({"llama3.1:70b": FREE})
        cost = cost_of(
            {"model": "llama3.1:70b", "usage": {"input_tokens": 5_000_000}}
        )
        assert cost == 0.0
        assert cost is not None

    def test_routing_prefixes_resolve_to_the_model(self):
        register_prices({"gpt-test": (1.0, 1.0)})
        for name in ("openai/gpt-test", "gpt-test-2026-01-01"):
            assert normalize(name) == "gpt-test"
        register_prices({"gemini-test": (1.0, 1.0)})
        assert normalize("models/gemini-test") == "gemini-test"

    def test_longest_prefix_wins(self):
        """`gpt-5-mini-2026-01` must not price as `gpt-5`."""
        register_prices({"gpt-5": (10.0, 30.0), "gpt-5-mini": (1.0, 3.0)})
        assert normalize("gpt-5-mini-2026-01") == "gpt-5-mini"

    @pytest.mark.parametrize(
        "bad",
        [{"m": (1.0,)}, {"m": "free"}, {"m": (-1.0, 2.0)}, {"": (1.0, 2.0)}],
        ids=["short-tuple", "not-a-tuple", "negative", "empty-name"],
    )
    def test_bad_registration_is_refused(self, bad):
        with pytest.raises(ValueError):
            register_prices(bad)
