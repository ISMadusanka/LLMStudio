"""Tests for the in-process event bus."""

from __future__ import annotations

from llmstudio.core.utils.events import LOG, METRIC, EventBus


def test_history_and_metrics_frame():
    bus = EventBus()
    bus.log("job1", "started")
    bus.metric("job1", 1, {"loss": 2.0})
    bus.metric("job1", 2, {"loss": 1.5})

    logs = bus.history("job1", kinds={LOG})
    assert len(logs) == 1 and logs[0].data["message"] == "started"

    frame = bus.metrics_frame("job1")
    assert [m["loss"] for m in frame] == [2.0, 1.5]
    assert [m["step"] for m in frame] == [1, 2]


def test_subscription_receives_live_events():
    bus = EventBus()
    sub = bus.subscribe("job2")
    bus.metric("job2", 1, {"loss": 0.9})
    sub.close()  # poison pill ends the stream

    received = [e for e in sub.stream(timeout=0.2)]
    assert any(e.kind == METRIC for e in received)


def test_isolation_between_jobs():
    bus = EventBus()
    bus.log("a", "x")
    assert bus.history("b") == []
