"""Observed actions through Browser Harness; one CDP session, no per-step subprocess.

A web page can stall, pop a dialog, crash its renderer or take Chrome down with it. None of
that may stop the service: every browser call is bounded, a stuck page fails only its own
task, its tab is always closed, and the dedicated Chrome is relaunched when it stops answering.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from browser_harness.admin import ensure_daemon, restart_daemon
from browser_harness.helpers import cdp

# Atomically read visible content and controls, preserving actual DOM node identity.
READ_STATE = Path(__file__).with_name("snapshot.js").read_text()
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"
CALL_SECONDS = 5.0  # One CDP round trip; a page that takes longer is treated as stuck.
NO_DIALOGS = "window.alert=()=>{};window.confirm=()=>false;window.prompt=()=>null;"


class StalePage(ValueError):
    """A decision no longer refers to the observed page."""


class Unresponsive(RuntimeError):
    """The page or browser stopped answering. Only this task fails; never retry its last input.

    Deliberately not a TimeoutError, so no caller can mistake one stuck page for a hung service.
    """


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Studio runs headless: without these, macOS and Chrome throttle or freeze the hidden tabs
# this agent drives, and Chrome's own hang dialog can block a page indefinitely.
CHROME_FLAGS = (
    "--no-first-run",
    "--no-default-browser-check",
    "--no-startup-window",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-hang-monitor",
    "--disable-session-crashed-bubble",
    "--hide-crash-restore-bubble",
    "--disable-features=Translate,MediaRouter,OptimizationHints,HeavyAdIntervention,IntensiveWakeUpThrottling",
)
# Serialises (re)launching Chrome and the daemon with routine maintenance.
MAINTENANCE = threading.RLock()
# Tabs owned by running tasks. Every other page in the dedicated profile is a leftover.
ACTIVE = set()
ACTIVE_LOCK = threading.Lock()


def profile_dir():
    return Path(os.environ.get("JEV_CHROME_PROFILE", "~/.config/browser-harness/jev-chrome")).expanduser()


def endpoint():
    return os.environ.get("BU_CDP_URL", "").rstrip("/")


def chrome_answers(timeout=1.0):
    """True when the dedicated Chrome answers; False when nothing listens; None when it is stuck."""
    try:
        urllib.request.urlopen(endpoint() + "/json/version", timeout=timeout).read()
        return True
    except ConnectionRefusedError:
        return False
    except OSError as error:
        reason = getattr(error, "reason", error)
        return False if isinstance(reason, ConnectionRefusedError) else None


def chrome_processes():
    """(pid, rss_kb, is_main) for every process of the dedicated profile."""
    marker = f"--user-data-dir={profile_dir()}"
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,rss=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and marker in parts[2] and parts[1].isdigit():
            found.append((int(parts[0]), int(parts[1]), "--type=" not in parts[2]))
    return found


def stop_chrome():
    """Stop the dedicated Chrome (only that profile); it is relaunched on demand."""
    mains = [pid for pid, _, main in chrome_processes() if main]
    for pid in mains:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(main for _, _, main in chrome_processes()):
        time.sleep(0.2)
    for pid, _, _ in chrome_processes():
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def launch_chrome():
    port = endpoint().rsplit(":", 1)[-1]
    subprocess.Popen(
        [CHROME, f"--user-data-dir={profile_dir()}", f"--remote-debugging-port={port}", *CHROME_FLAGS],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    deadline = time.monotonic() + 20
    while chrome_answers() is not True:
        if time.monotonic() > deadline:
            raise Unresponsive("Dedicated Chrome did not start; no action executed.")
        time.sleep(0.2)


def ensure_chrome():
    """With BU_CDP_URL set, make sure the dedicated Chrome answers. Returns True if it was (re)started."""
    if not endpoint() or not Path(CHROME).exists():
        return False
    with MAINTENANCE:
        state = chrome_answers()
        for _ in range(2):  # A busy Chrome can miss one probe; three misses in a row means stuck.
            if state is not None:
                break
            time.sleep(0.5)
            state = chrome_answers(timeout=2.0)
        if state is True:
            return False
        if state is None or any(main for _, _, main in chrome_processes()):
            stop_chrome()  # Listening but stuck, or alive without its debug port.
        launch_chrome()
        return True


def ready_browser():
    """Chrome answering and a daemon connected to it, restarting whichever is broken."""
    with MAINTENANCE:
        if ensure_chrome():
            restart_daemon()  # a daemon from the previous Chrome holds a dead connection
        ensure_daemon()


def raw(method, timeout=CALL_SECONDS, **params):
    """One browser-level CDP call that can only fail with Unresponsive or RuntimeError."""
    try:
        return cdp(method, _response_timeout=timeout, **params)
    except TimeoutError:
        raise Unresponsive(f"Browser did not answer {method}.") from None
    except OSError:
        raise Unresponsive("Browser connection is down.") from None


def page_targets():
    return [t for t in raw("Target.getTargets").get("targetInfos", []) if t.get("type") == "page"]


def sweep_tabs(keep=()):
    """Close pages no running task owns. Returns how many were closed."""
    with ACTIVE_LOCK:
        owned = set(ACTIVE) | set(keep)
    closed = 0
    for target in page_targets():
        if target["targetId"] not in owned:
            try:
                raw("Target.closeTarget", targetId=target["targetId"])
                closed += 1
            except RuntimeError:
                pass
    return closed


class Browser:
    def __init__(self, url, deadline=None):
        self.target = self.session = self.after_input = None
        self.deadline = deadline or time.monotonic() + 30
        ready_browser()
        try:
            self.target = raw("Target.createTarget", url="about:blank", background=True)["targetId"]
        except RuntimeError:
            # Chrome restarted under a live daemon: reconnect once and retry.
            with MAINTENANCE:
                restart_daemon()
                ensure_daemon()
            self.target = raw("Target.createTarget", url="about:blank", background=True)["targetId"]
        with ACTIVE_LOCK:
            ACTIVE.add(self.target)
        try:
            self.session = raw("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
            # Chrome tracks a page's alert/confirm dialogs only with Page enabled; rescue() needs that.
            self.call("Page.enable")
            # Tasks are read-only, so a page's dialogs only ever block it: answer them as "cancel"
            # before the page's own scripts run. rescue() still covers anything this misses.
            self.call("Page.addScriptToEvaluateOnNewDocument", source=NO_DIALOGS)
            self.call("Emulation.setDeviceMetricsOverride", width=1120, height=780, deviceScaleFactor=1, mobile=False)
            # Keep rAF/menus rendering in an owned background tab, without activating the user's Chrome tab.
            self.call("Emulation.setFocusEmulationEnabled", enabled=True)
            try:
                self.call("Page.navigate", url=url, timeout=min(10.0, self.remaining()))
            except Unresponsive:
                pass  # A slow server: keep waiting for the document below, within the task's budget.
            deadline = min(time.monotonic() + 15, self.deadline)
            while time.monotonic() < deadline:
                try:
                    loaded = self.evaluate("document.readyState === 'complete' && location.href !== 'about:blank'")
                except (StalePage, Unresponsive):
                    loaded = False
                if loaded:
                    break
                time.sleep(0.05)
        except BaseException:
            self.close()  # Never leave a tab behind, whatever failed.
            raise

    def remaining(self):
        return max(1.0, self.deadline - time.monotonic())

    def call(self, method, timeout=CALL_SECONDS, **params):
        try:
            return cdp(method, session_id=self.session, _response_timeout=min(timeout, self.remaining()), **params)
        except TimeoutError:
            raise Unresponsive(f"The page stopped responding ({method}).") from None
        except OSError:
            raise Unresponsive("Browser connection is down.") from None

    def rescue(self):
        """Dismiss a JavaScript dialog if one is what blocks the page. True if one was open.

        Chrome answers this from the browser process, so it is quick even while the page is busy.
        """
        try:
            self.call("Page.handleJavaScriptDialog", timeout=2.0, accept=False)
            return True
        except RuntimeError:
            return False

    def busy_wait_left(self):
        """Seconds a read may still wait for a busy page before the task gives up."""
        return self.deadline - time.monotonic() - 0.5

    def salvage(self):
        """Whatever the tab shows right now, for a partial answer. Never raises."""
        if not self.session:
            return {}
        try:
            value = self.call(
                "Runtime.evaluate", timeout=2.0, returnByValue=True,
                expression="({url: location.href, title: document.title, text: document.body?.innerText ?? ''})",
            ).get("result", {}).get("value")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def evaluate(self, expression):
        try:
            response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        except Unresponsive:
            # A read is safe to repeat. Clear a dialog if that is what blocked it; otherwise the
            # page is busy (heavy sites block for seconds while loading), so observe again.
            if self.rescue():
                raise StalePage("A page dialog was dismissed. Observe again.") from None
            if self.busy_wait_left() > 0:
                raise StalePage("The page is busy. Observe again.") from None
            raise
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot=True):
        if getattr(self, "after_input", None):
            action, self.after_input = self.after_input, None
            # This is read-only and happens after execution was logged, even if navigation interrupts it.
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const autocomplete=action.kind==='fill' && field?.getAttribute('role')==='combobox';
                      let frames=0, stopped=false;
                      const finish=()=>{stopped=true;resolve()};
                      setTimeout(finish,autocomplete ? 200 : 50);
                      const ready=()=>{
                        if (stopped) return;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        if (++frames>=2 && (!autocomplete || options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        }))) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    timeout=2.0,
                    awaitPromise=True,
                    returnByValue=True,
                )
            except RuntimeError:
                pass  # Includes Unresponsive: the settle wait is best-effort and read-only.
        stale = 0
        while True:
            try:
                return browser_operation(
                    {"operation": "observe", "session": self.session, "screenshot": screenshot,
                     "timeout": min(CALL_SECONDS, self.remaining())}
                )
            except StalePage:
                stale += 1
                if stale == 10:
                    raise
                time.sleep(0.02)
            except Unresponsive:
                # Reading is safe to repeat: wait out a busy page for as long as the task allows.
                if not self.rescue() and self.busy_wait_left() <= 0:
                    raise

    def same_page(self, page):
        """Same document, URL, scroll, viewport and form values; ignores carousels and lazy content."""
        return self.evaluate("(() => window.__jevFast?.pageKey() ?? null)()") == page["page_key"]

    def fresh(self, page, action=None):
        if action is not None and action["kind"] not in {"click", "select", "fill"}:
            return self.same_page(page)
        if action is not None:
            node = action["node"]
            if type(node) is not int:
                return False
            current = self.evaluate(
                "(() => { const c=window.__jevFast; "
                f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
            )
            return current == [page["page_key"], page["guards"].get(str(node))]
        return self.evaluate(MARKER) == page["marker"]

    def act(self, action, page, text=None):
        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")
        if action["kind"] == "wait":
            time.sleep(0.1)
        # A timeout here raises Unresponsive, which stops the task: the input may have landed.
        result = browser_operation({"operation": "act", "session": self.session, "action": action, "text": text,
                                    "timeout": min(CALL_SECONDS, self.remaining())})
        self.after_input = action if action["kind"] != "wait" else None
        return result

    def close(self):
        target, self.target = self.target, None
        if target:
            try:
                # Browser-level, so it works even when the page's renderer is stuck.
                raw("Target.closeTarget", timeout=3.0, targetId=target)
            except Exception:
                pass  # Tab or browser already gone; the sweep removes anything left.
            finally:
                with ACTIVE_LOCK:
                    ACTIVE.discard(target)


def fingerprint(state):
    content = {k: state[k] for k in ("url", "text", "actions", "scroll")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def browser_operation(request):
    operation = request["operation"]
    session = request["session"]
    timeout = request.get("timeout", CALL_SECONDS)

    def call(method, **params):
        try:
            return cdp(method, session_id=session, _response_timeout=timeout, **params)
        except TimeoutError:
            raise Unresponsive(f"The page stopped responding ({method}).") from None
        except OSError:
            raise Unresponsive("Browser connection is down.") from None

    def evaluate(expression):
        result = call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            if operation == "act" and request["action"]["kind"] == "select":
                raise RuntimeError("Dropdown execution was interrupted; inspect before retrying.")
            raise StalePage("Document changed during evaluation")
        return result.get("result", {}).get("value")

    if operation == "act":
        action = request["action"]
        kind = action["kind"]
        if kind == "scroll":
            call("Input.dispatchMouseEvent", type="mouseWheel", x=550, y=650, deltaX=0, deltaY=action["delta"])
        elif kind != "wait":
            if type(action["node"]) is not int:
                raise ValueError("Invalid observed node")
            # Code-owned node IDs refer to actual observed elements, never model-generated selectors.
            target = evaluate("""(action => {
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{bubbles:true}));
                e.dispatchEvent(new Event('change',{bubbles:true}));
              }
              return {x,y};
            })(""" + json.dumps(action) + ")")
            if target is None:
                if kind == "select":
                    raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
                raise StalePage("Target changed or is covered. Observe again.")
            if kind != "select":
                x, y = target["x"], target["y"]
                for event in ("mousePressed", "mouseReleased"):
                    call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
                if kind == "fill":
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyDown",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                        commands=["selectAll"],
                    )
                    call(
                        "Input.dispatchKeyEvent",
                        type="keyUp",
                        key="a",
                        code="KeyA",
                        modifiers=4 if sys.platform == "darwin" else 2,
                    )
                    call("Input.insertText", text=request["text"])
        return {"executed": action["id"]}

    info = evaluate(READ_STATE)
    if info is None:
        raise StalePage("Document is navigating")
    info["fingerprint"] = fingerprint(info)
    if request.get("screenshot", True):
        info["screenshot"] = call("Page.captureScreenshot", format="jpeg", quality=72)["data"]
    return info
