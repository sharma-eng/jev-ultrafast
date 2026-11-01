"""A page or browser failure fails one call, never the service. Offline: no Chrome, no paid API."""

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest

import jev_ultrafast.browser as chrome
import jev_ultrafast.mcp_server as server
from jev_ultrafast import model


class FakeCDP:
    """Answers CDP like Chrome would, with optional stalls per method."""

    def __init__(self, stall=(), pages=()):
        self.stall, self.calls, self.closed = set(stall), [], []
        self.pages = list(pages)

    def __call__(self, method, session_id=None, _response_timeout=5.0, **params):
        self.calls.append(method)
        if method in self.stall:
            raise TimeoutError(f"{method} timed out")
        if method == "Target.createTarget":
            self.pages.append("tab-new")
            return {"targetId": "tab-new"}
        if method == "Target.attachToTarget":
            return {"sessionId": "session-1"}
        if method == "Target.closeTarget":
            self.closed.append(params["targetId"])
            return {}
        if method == "Target.getTargets":
            return {"targetInfos": [{"targetId": t, "type": "page"} for t in self.pages]}
        if method == "Runtime.evaluate":
            return {"result": {"value": True}}
        if method == "Page.handleJavaScriptDialog":
            raise RuntimeError("No dialog is showing")  # what the daemon relays from Chrome
        return {}


@pytest.fixture
def fake(monkeypatch):
    cdp = FakeCDP()
    monkeypatch.setattr(chrome, "cdp", cdp)
    monkeypatch.setattr(chrome, "ready_browser", lambda: None)
    chrome.ACTIVE.clear()
    return cdp


def test_a_stuck_page_raises_unresponsive_not_timeout(fake):
    fake.stall.add("Runtime.evaluate")
    b = chrome.Browser.__new__(chrome.Browser)
    b.session, b.target, b.deadline = "session-1", "tab-new", time.monotonic()
    with pytest.raises(chrome.Unresponsive) as raised:
        b.evaluate("1")
    assert not isinstance(raised.value, TimeoutError)


def test_a_busy_page_is_waited_out_while_time_remains(fake):
    stalls = iter([True, True, False])
    page = {"url": "https://shop.test/", "title": "Shop", "text": "ok", "actions": [], "scroll": {}}

    def cdp(method, **_):
        if method == "Runtime.evaluate" and next(stalls):
            raise TimeoutError("busy")
        if method == "Page.handleJavaScriptDialog":
            raise RuntimeError("No dialog is showing")
        return {"result": {"value": page}}

    chrome.cdp = cdp
    b = chrome.Browser.__new__(chrome.Browser)
    b.session, b.target, b.deadline, b.after_input = "session-1", "tab-new", time.monotonic() + 10, None
    assert b.observe(screenshot=False)["title"] == "Shop"
    with pytest.raises(chrome.StalePage, match="busy"):  # a busy read before an action re-observes
        stalls = iter([True])
        b.evaluate("1")


def test_an_input_that_times_out_stops_the_task_instead_of_retrying(fake):
    fake.stall.add("Input.dispatchMouseEvent")
    target = {"result": {"value": {"x": 5, "y": 5}}}
    chrome.cdp = lambda method, **kw: target if method == "Runtime.evaluate" else fake(method, **kw)
    action = {"id": "e1", "kind": "click", "node": 1}
    with pytest.raises(chrome.Unresponsive):
        chrome.browser_operation({"operation": "act", "session": "s", "action": action})
    assert fake.calls.count("Input.dispatchMouseEvent") == 1


def test_a_dismissed_dialog_means_observe_again(fake):
    calls = iter([TimeoutError("blocked by alert"), {}, {}])

    def cdp(method, **_):
        result = next(calls) if method in ("Runtime.evaluate", "Page.handleJavaScriptDialog") else {}
        if isinstance(result, Exception):
            raise result
        return result

    chrome.cdp = cdp
    b = chrome.Browser.__new__(chrome.Browser)
    b.session, b.target, b.deadline = "session-1", "tab-new", time.monotonic() + 10
    with pytest.raises(chrome.StalePage):
        b.evaluate("document.title")


def test_failed_open_closes_its_tab(fake):
    fake.stall.add("Target.attachToTarget")
    with pytest.raises(chrome.Unresponsive):
        chrome.Browser("https://example.com", deadline=time.monotonic() + 5)
    assert fake.closed == ["tab-new"]
    assert not chrome.ACTIVE


def test_slow_navigation_waits_for_the_document_instead_of_failing(fake):
    fake.stall.add("Page.navigate")
    b = chrome.Browser("https://example.com", deadline=time.monotonic() + 5)
    assert b.target == "tab-new" and "tab-new" in chrome.ACTIVE
    b.close()
    assert fake.closed == ["tab-new"] and not chrome.ACTIVE


def test_sweep_closes_only_tabs_no_task_owns(fake):
    fake.pages = ["leaked-1", "owned", "leaked-2"]
    chrome.ACTIVE.add("owned")
    assert chrome.sweep_tabs() == 2
    assert fake.closed == ["leaked-1", "leaked-2"]


def test_task_failure_returns_partial_page_and_closes_the_tab(monkeypatch):
    browser = Mock(salvage=Mock(return_value={"url": "https://shop.test/", "title": "Shop", "text": "Price $5"}))
    monkeypatch.setattr(server.chrome, "Browser", Mock(return_value=browser))
    monkeypatch.setattr(server, "Agent", Mock(side_effect=chrome.Unresponsive("The page stopped responding.")))
    exits = Mock()
    monkeypatch.setattr(server.os, "_exit", exits)
    result = server.run_task("https://shop.test/", "Read the price")
    assert result["status"] == "page_unresponsive"
    assert result["page_text"] == "Price $5" and "still available" in result["error"]
    browser.close.assert_called_once()
    exits.assert_not_called()


def test_a_crash_mid_task_keeps_the_steps_already_taken(monkeypatch):
    agent = Mock()
    agent.snapshot.return_value = {"status": "ready", "page": {"url": "u", "title": "t", "text": "seen"},
                                   "history": [{"action": "Search"}]}
    agent.command.side_effect = RuntimeError("Browser connection is down.")
    browser = Mock()
    monkeypatch.setattr(server.chrome, "Browser", Mock(return_value=browser))
    monkeypatch.setattr(server, "Agent", Mock(return_value=agent))
    result = server.run_task("https://shop.test/", "Find it")
    assert result["status"] == "error" and result["steps"] == ["Search"] and result["page_text"] == "seen"
    browser.close.assert_called_once()


def test_parallel_calls_each_get_a_tab_and_extra_ones_are_told_busy(monkeypatch):
    gate = threading.Event()

    def slow_task(url, goal, progress):
        gate.wait(5)
        return {"status": "done", "url": url}

    monkeypatch.setattr(server, "run_task", slow_task)
    monkeypatch.setattr(server, "SLOTS", threading.BoundedSemaphore(2))
    monkeypatch.setattr(server, "MAX_TASKS", 2)
    monkeypatch.setattr(server, "QUEUE_SECONDS", 0.2)

    async def scenario():
        calls = [asyncio.create_task(server.browse_web(f"https://site{i}.test", "Read it")) for i in range(3)]
        await asyncio.sleep(0.5)
        gate.set()
        return await asyncio.gather(*calls)

    results = asyncio.run(scenario())
    assert sorted(r["status"] for r in results) == ["busy", "done", "done"]


def test_a_call_that_outlives_its_budget_answers_without_exiting(monkeypatch):
    release = threading.Event()

    def stuck_task(url, goal, progress):
        progress["state"] = {"page": {"url": url, "title": "Slow", "text": "partial"}, "history": []}
        release.wait(5)
        return {"status": "done"}

    exits = Mock()
    monkeypatch.setattr(server.os, "_exit", exits)
    monkeypatch.setattr(server, "run_task", stuck_task)
    monkeypatch.setattr(server, "BUDGET_SECONDS", 0.1)
    monkeypatch.setattr(server, "QUEUE_SECONDS", 0.1)
    monkeypatch.setattr(server, "SLACK_SECONDS", 0.1)
    try:
        result = asyncio.run(server.browse_web("https://slow.test", "Read it"))
    finally:
        release.set()
    assert result["status"] == "out_of_time" and result["page_text"] == "partial"
    exits.assert_not_called()


@pytest.mark.parametrize(
    ("answers", "processes", "stops", "launches"),
    [
        ([True], [], 0, 0),  # healthy: leave it alone
        ([False], [], 0, 1),  # not running: launch
        ([None, None, None], [(1, 10, True)], 1, 1),  # listening but stuck: stop, then launch
        ([False], [(1, 10, True)], 1, 1),  # alive without its debug port: stop, then launch
        ([None, True], [(1, 10, True)], 0, 0),  # one slow answer is not a hang
    ],
)
def test_chrome_is_relaunched_only_when_it_is_down_or_stuck(monkeypatch, answers, processes, stops, launches):
    monkeypatch.setenv("BU_CDP_URL", "http://127.0.0.1:9333")
    monkeypatch.setattr(chrome.Path, "exists", lambda _p: True)
    replies = iter(answers)
    monkeypatch.setattr(chrome, "chrome_answers", lambda timeout=1.0: next(replies))
    monkeypatch.setattr(chrome, "chrome_processes", lambda: processes)
    monkeypatch.setattr(chrome.time, "sleep", lambda _s: None)
    stop, launch = Mock(), Mock()
    monkeypatch.setattr(chrome, "stop_chrome", stop)
    monkeypatch.setattr(chrome, "launch_chrome", launch)
    assert chrome.ensure_chrome() is bool(launches)
    assert stop.call_count == stops and launch.call_count == launches


def test_model_calls_never_outlive_the_task(monkeypatch):
    post = Mock()
    monkeypatch.setattr(model.CLIENT, "post", post)
    model.LIMIT.deadline = time.monotonic() + 0.2
    try:
        with pytest.raises(RuntimeError, match="Out of time"):
            time.sleep(0.3)
            model.post_json("https://model.test", "key", {})
        post.assert_not_called()
        model.LIMIT.deadline = time.monotonic() + 3
        post.return_value = Mock(status_code=200, is_error=False, json=Mock(return_value={"ok": 1}))
        model.post_json("https://model.test", "key", {})
        assert post.call_args.kwargs["timeout"] <= 3
    finally:
        model.LIMIT.deadline = None


def test_the_monitor_survives_anything_it_finds(monkeypatch):
    monkeypatch.setattr(server, "maintain", Mock(side_effect=[RuntimeError("Chrome gone"), KeyboardInterrupt]))
    monkeypatch.setattr(server.time, "sleep", lambda _s: None)
    with pytest.raises(KeyboardInterrupt):
        server.monitor()
    assert server.maintain.call_count == 2
