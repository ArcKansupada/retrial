"""A synchronous agent still refuses an async interception point.

A sync loop never awaits, so it would record an un-awaited coroutine.
"""

import asyncio

import pytest
from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import record
from retrial.errors import IntegrationError


def test_an_async_call_model_is_refused(store, opening):
    """The sync-agent variant: it would record an un-awaited coroutine."""

    async def async_model(messages, tools):
        return None

    agent = record(session_name="sync-agent", store=store)(raw_agent)
    with pytest.raises(IntegrationError) as excinfo:
        agent(opening, TOOLS, async_model, make_executor(450))

    assert "async" in str(excinfo.value)
    assert "call_model" in str(excinfo.value)


def test_an_async_execute_tools_is_refused(store, opening):
    async def async_tools(response):
        return []

    agent = record(session_name="sync-agent", store=store)(raw_agent)
    with pytest.raises(IntegrationError) as excinfo:
        agent(opening, TOOLS, fake_model, async_tools)

    assert "async" in str(excinfo.value)
    assert "execute_tools" in str(excinfo.value)


def test_refusing_an_async_call_model_leaves_no_session_behind(store, opening):
    """Same rule as the not-callable check: reject before a row exists."""

    async def async_model(messages, tools):
        return None

    agent = record(session_name="sync-agent", store=store)(raw_agent)
    with pytest.raises(IntegrationError):
        agent(opening, TOOLS, async_model, make_executor(450))

    assert store.list_sessions() == []


def test_an_async_callable_object_is_refused_too(store, opening):
    """An object with `async def __call__` is not itself a coroutine function."""

    class AsyncClient:
        async def __call__(self, messages, tools):
            return None

    agent = record(session_name="sync-agent", store=store)(raw_agent)
    with pytest.raises(IntegrationError) as excinfo:
        agent(opening, TOOLS, AsyncClient(), make_executor(450))

    assert "async" in str(excinfo.value)


def test_a_sync_agent_is_unaffected(store, opening):
    """The guard must not cost the supported case anything."""
    agent = record(session_name="sync-agent", store=store)(raw_agent)
    agent(opening, TOOLS, fake_model, make_executor(450))

    sessions = store.list_sessions()
    assert [s["status"] for s in sessions] == ["complete"]
    assert store.steps_for(sessions[0]["id"])


def test_the_wrapper_never_returns_a_coroutine(store, opening):
    """A session must not be marked complete before its work happened."""
    agent = record(session_name="sync-agent", store=store)(raw_agent)
    result = agent(opening, TOOLS, fake_model, make_executor(450))
    assert not asyncio.iscoroutine(result)
