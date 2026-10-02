"""The recording layer.

A decorator wraps the user's loop. The model-calling and tool-executing functions are passed in
explicitly, not monkey-patched, so every recorded step traces back to a line the user wrote.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import inspect
import threading
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

from .errors import IntegrationError
from .pricing import cost_of
from .serialize import to_jsonable
from .storage import Store
from .types import JSON

P = ParamSpec("P")
R = TypeVar("R")


def _is_async(obj: object) -> bool:
    """True for anything that returns a coroutine when called.

    Covers a class with an `async def __call__`, which `iscoroutinefunction` misses.
    """
    if inspect.iscoroutinefunction(obj):
        return True
    if not callable(obj) or inspect.isroutine(obj):
        return False
    # A callable always has type(obj).__call__, so this cannot raise.
    return inspect.iscoroutinefunction(type(obj).__call__)

# Handoff from fork() to the decorator: the session the next recorded run belongs to. A
# ContextVar, so concurrent forks under asyncio or threads each see their own.
_pending: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "retrial_pending", default=None
)

# One Store per database per process, shared across threads.
_default_stores: dict[str, Store] = {}

# Guards the cache: check-then-insert is not atomic across threads.
_stores_lock = threading.Lock()


def _default_store() -> Store:
    from .storage import resolve_db_path

    # Searches upward, so a run from a subdirectory records into the project's store.
    path = resolve_db_path()
    with _stores_lock:
        store = _default_stores.get(path)
        if store is None:
            store = _default_stores[path] = Store(path)
        return store


def record(
    session_name: str | None = None,
    store: Store | None = None,
    model_arg: str = "call_model",
    tools_arg: str = "execute_tools",
    messages_arg: str = "messages",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record every step of a raw agent loop.

        @record(session_name="booking-agent")
        def run_agent(messages, tools, call_model, execute_tools):
            while True:
                response = call_model(messages, tools)
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    return response
                messages.append({"role": "user", "content": execute_tools(response)})

    `messages` must be a parameter: that is what lets a fork seed the loop with edited history.
    The decorated function keeps its signature and gains `last_session_id` and
    `__retrial_agent__`.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)
        for name in (messages_arg, model_arg, tools_arg):
            if name not in signature.parameters:
                raise IntegrationError(
                    f"@record expected {fn.__name__!r} to take a {name!r} parameter "
                    f"but its signature is {signature}. retrial intercepts the model "
                    "call and the tool executor by name; pass them in explicitly "
                    "(or set the *_arg options on @record)."
                )

        agent_is_async = _is_async(fn)

        def setup(
            args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[inspect.BoundArguments, Store, str]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate before touching the store, so a rejected call leaves no empty session.
            for name in (model_arg, tools_arg):
                value = bound.arguments[name]
                if not callable(value):
                    raise IntegrationError(
                        f"{fn.__name__}() got {name}={value!r}, which is not "
                        f"callable. @record wraps {name} before your function body "
                        "runs, so it must already be a callable when the agent is "
                        "invoked - it cannot be built lazily inside the body. Give "
                        "it a real function as its default and do any lazy setup "
                        "(client construction, auth) on first call inside it."
                    )
                # A sync agent cannot await an async interception point. Refuse it here, by
                # name.
                if not agent_is_async and _is_async(value):
                    raise IntegrationError(
                        f"{fn.__name__}() is a synchronous agent but got an async "
                        f"{name}. A sync loop cannot await it, so retrial would "
                        "record an un-awaited coroutine instead of the response. "
                        "Make the agent async (retrial records async agents now), "
                        f"or pass a synchronous {name} (for example one that calls "
                        "asyncio.run internally)."
                    )

            # Consume the fork handoff exactly once.
            ctx = _pending.get()
            _pending.set(None)
            active_store = ctx["store"] if ctx else (store or _default_store())
            session_id = (
                ctx["session_id"]
                if ctx
                else active_store.create_session(name=session_name or fn.__name__)
            )

            recorder = _Recorder(active_store, session_id)
            bound.arguments[model_arg] = recorder.wrap_model(bound.arguments[model_arg])
            bound.arguments[tools_arg] = recorder.wrap_tools(bound.arguments[tools_arg])
            return bound, active_store, session_id

        if agent_is_async:

            @functools.wraps(fn)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                bound, active_store, session_id = setup(args, kwargs)
                wrapper.last_session_id = session_id  # type: ignore[attr-defined]
                try:
                    # fn is async here, so its result is awaitable; R cannot express that.
                    result = await fn(*bound.args, **bound.kwargs)  # type: ignore[misc]
                except BaseException:
                    active_store.set_status(session_id, "failed")
                    raise
                active_store.set_status(session_id, "complete")
                return result

        else:

            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                bound, active_store, session_id = setup(args, kwargs)
                wrapper.last_session_id = session_id  # type: ignore[attr-defined]
                try:
                    result = fn(*bound.args, **bound.kwargs)
                except BaseException:
                    # Keep every step a crashed run recorded, and mark why it stopped.
                    active_store.set_status(session_id, "failed")
                    raise
                active_store.set_status(session_id, "complete")
                return result

        wrapper.last_session_id = None  # type: ignore[attr-defined]
        wrapper.__retrial_agent__ = True  # type: ignore[attr-defined]
        return cast("Callable[P, R]", wrapper)

    return decorator


class _Recorder:
    def __init__(self, store: Store, session_id: str) -> None:
        self.store = store
        self.session_id = session_id

        # Step numbers are allocated inside the store's lock; see Store.add_step.

    def wrap_model(self, call_model: Callable[..., Any]) -> Callable[..., Any]:
        # Snapshot the history verbatim before the model sees it. This is what a fork replays.
        if _is_async(call_model):

            @functools.wraps(call_model)
            async def wrapped_async(messages: Any, *args: Any, **kwargs: Any) -> Any:
                snapshot = to_jsonable(copy.deepcopy(list(messages)))
                started = time.perf_counter()
                response = await call_model(messages, *args, **kwargs)
                self._record_model_call(snapshot, started, response, args)
                return response

            return wrapped_async

        @functools.wraps(call_model)
        def wrapped(messages: Any, *args: Any, **kwargs: Any) -> Any:
            snapshot = to_jsonable(copy.deepcopy(list(messages)))
            started = time.perf_counter()
            response = call_model(messages, *args, **kwargs)
            self._record_model_call(snapshot, started, response, args)
            return response

        return wrapped

    def _record_model_call(
        self, snapshot: JSON, started: float, response: Any, args: tuple[Any, ...]
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        serialized = to_jsonable(response)
        tokens, cost = _usage(serialized)
        self.store.add_step(
            self.session_id,
            None,
            "model_call",
            {"messages": snapshot, "extra": to_jsonable(list(args))},
            serialized,
            tokens_used=tokens,
            cost_usd=cost,
            duration_ms=elapsed_ms,
        )

    def wrap_tools(self, execute_tools: Callable[..., Any]) -> Callable[..., Any]:
        if _is_async(execute_tools):

            @functools.wraps(execute_tools)
            async def wrapped_async(response: Any, *args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                results = await execute_tools(response, *args, **kwargs)
                self._record_tool_call(started, response, results)
                return results

            return wrapped_async

        @functools.wraps(execute_tools)
        def wrapped(response: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            results = execute_tools(response, *args, **kwargs)
            self._record_tool_call(started, response, results)
            return results

        return wrapped

    def _record_tool_call(self, started: float, response: Any, results: Any) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.store.add_step(
            self.session_id,
            None,
            "tool_call",
            _tool_uses(to_jsonable(response)),
            to_jsonable(results),
            duration_ms=elapsed_ms,
        )


def _tool_uses(serialized_response: JSON) -> JSON:
    content = serialized_response.get("content") if isinstance(
        serialized_response, dict
    ) else None
    if not isinstance(content, list):
        return serialized_response
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def _usage(serialized_response: JSON) -> tuple[int | None, float | None]:
    """Best-effort token and cost accounting.

    Cost is None whenever it cannot be known exactly; never an estimate.
    """
    if not isinstance(serialized_response, dict):
        return None, None
    usage = serialized_response.get("usage")
    if not isinstance(usage, dict):
        return None, None

    tokens = None
    parts = [usage.get("input_tokens"), usage.get("output_tokens")]
    if any(isinstance(p, int) for p in parts):
        tokens = sum(p for p in parts if isinstance(p, int))
    return tokens, cost_of(serialized_response)
