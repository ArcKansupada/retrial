"""A Store is shared between threads, not one per thread.

These run real threads, with enough concurrency that unguarded code fails reliably.
"""

import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from conftest import TOOLS, fake_model, make_executor, raw_agent

from retrial import record

THREADS = 8


def gather(fn, count=THREADS):
    """Run `fn(i)` on `count` threads, all released at once.

    The barrier guarantees the calls overlap.
    """
    barrier = threading.Barrier(count)

    def body(i):
        barrier.wait()
        return fn(i)

    with ThreadPoolExecutor(count) as pool:
        return list(pool.map(body, range(count)))


# -- the connection ------------------------------------------------------------


def test_a_store_opened_on_one_thread_works_on_another(store):
    """This raised sqlite3.ProgrammingError before check_same_thread=False."""
    with ThreadPoolExecutor(1) as pool:
        session_id = pool.submit(store.create_session, "from-another-thread").result()

    assert store.get_session(session_id)["name"] == "from-another-thread"


def test_concurrent_writers_all_land(store):
    """Every session is written, and the store is left consistent."""
    ids = gather(lambda i: store.create_session(name=f"session-{i}"))

    assert len(set(ids)) == THREADS
    assert sorted(s["name"] for s in store.list_sessions()) == sorted(
        f"session-{i}" for i in range(THREADS)
    )


def test_concurrent_readers_and_writers_do_not_collide(store):
    seed = store.create_session(name="seed")

    def mixed(i):
        if i % 2:
            return store.create_session(name=f"writer-{i}")
        return store.get_session(seed)["name"]

    results = gather(mixed)
    assert "seed" in results


def test_steps_recorded_concurrently_into_one_session_all_survive(
    store, monkeypatch
):
    """The read-then-write window that `next_step_number` leaves open.

    add_step allocates inside the lock instead. The delay widens the window so the bug is
    certain, not lucky.
    """
    real = type(store)._next_step_number

    def slow(self, session_id):
        n = real(self, session_id)
        time.sleep(0.01)
        return n

    monkeypatch.setattr(type(store), "_next_step_number", slow)

    session_id = store.create_session(name="one-session-many-writers")
    gather(lambda i: store.add_step(session_id, None, "model_call", {"i": i}, {}))

    steps = store.steps_for(session_id)
    assert len(steps) == THREADS
    assert [s["step_number"] for s in steps] == list(range(THREADS))


def test_an_explicit_step_number_is_still_honoured(store):
    """Passing a number is what fork's reconstruction path relies on."""
    session_id = store.create_session(name="explicit")
    store.add_step(session_id, 7, "model_call", {}, {})

    assert [s["step_number"] for s in store.steps_for(session_id)] == [7]


# -- the process-wide cache ----------------------------------------------------


def test_the_default_store_is_created_exactly_once(tmp_path, monkeypatch):
    """Unguarded, 8 threads opened 8 connections and orphaned 7 of them."""
    monkeypatch.delenv("RETRIAL_DB", raising=False)
    monkeypatch.chdir(tmp_path)
    record_module = importlib.import_module("retrial.record")
    record_module._default_stores.clear()

    try:
        stores = gather(lambda i: record_module._default_store())
        assert len({id(s) for s in stores}) == 1
        assert len(record_module._default_stores) == 1
        # The object callers got is the one that is cached, not a leaked twin.
        assert stores[0] is next(iter(record_module._default_stores.values()))
    finally:
        for s in record_module._default_stores.values():
            s.close()
        record_module._default_stores.clear()


# -- fork's handoff ------------------------------------------------------------


def test_pending_context_does_not_leak_between_threads(store):
    """`_pending` is how fork tells the decorator which session to record into.

    As a ContextVar, a value set on one thread is invisible to the others.
    """
    record_module = importlib.import_module("retrial.record")
    sessions = [store.create_session(name=f"fork-{i}") for i in range(THREADS)]

    def claim(i):
        record_module._pending.set({"store": store, "session_id": sessions[i]})
        # Yield, so every thread has published before any thread consumes.
        threading.Event().wait(0.01)
        ctx = record_module._pending.get()
        return ctx["session_id"]

    assert gather(claim) == sessions


def test_pending_set_on_one_thread_is_invisible_to_another(store):
    record_module = importlib.import_module("retrial.record")
    token = record_module._pending.set({"store": store, "session_id": "s_main"})
    try:
        with ThreadPoolExecutor(1) as pool:
            seen = pool.submit(record_module._pending.get).result()
        assert seen is None
    finally:
        record_module._pending.reset(token)


# -- end to end ----------------------------------------------------------------


def test_two_agents_recording_at_once_get_separate_intact_sessions(store, opening):
    """The case that motivated all of this."""
    agent = record(store=store)(raw_agent)

    def run(i):
        agent(list(opening), TOOLS, fake_model, make_executor(450))

    gather(run, count=4)

    sessions = store.list_sessions()
    assert len(sessions) == 4
    assert {s["status"] for s in sessions} == {"complete"}
    for session in sessions:
        steps = store.steps_for(session["id"])
        assert steps, "a session recorded no steps"
        numbers = [s["step_number"] for s in steps]
        assert numbers == list(range(len(numbers))), "steps interleaved across sessions"
