#!/usr/bin/env python
"""Run pymmcore-plus tests against pymmcore-proxy's RemoteMMCore.

This script:
1. Locates pymmcore-plus test files (sibling repo or pip show)
2. Creates a temp directory with symlinked test files
3. Writes a conftest.py that provides proxy fixtures and skip markers
4. Runs pytest — incompatible tests are skipped with explanatory reasons

Usage:
    python scripts/run_compat_tests.py [pytest-args...]

Examples:
    python scripts/run_compat_tests.py -v
    python scripts/run_compat_tests.py -v -k test_mda_waiting
    python scripts/run_compat_tests.py -x --tb=short
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------- locate pymmcore-plus tests ----------

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SIBLING = REPO_ROOT.parent / "pymmcore-plus" / "tests"


def _find_pmp_tests() -> Path:
    if SIBLING.is_dir():
        return SIBLING
    # fallback: pip show
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "-f", "pymmcore-plus"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "Location:" in line:
                loc = Path(line.split(":", 1)[1].strip())
                candidate = loc / "pymmcore_plus" / ".." / ".." / "tests"
                if candidate.is_dir():
                    return candidate.resolve()
    except Exception:
        pass
    raise FileNotFoundError(
        "Cannot find pymmcore-plus tests. "
        "Place pymmcore-plus repo next to pymmcore-proxy, or pip install it."
    )


# ---------- test files to run ----------

COMPAT_FILES = [
    "test_events.py",
    "test_mda.py",
]

# ---------- incompatible tests with reasons ----------
# Every skipped test gets a reason explaining *why* it's incompatible.

SKIP_TESTS: dict[str, str] = {
    # --- test_events.py ---
    # Uses devicePropertyChanged("Camera", "Gain") callable signal filter — proxy
    # signals are plain psygnal.Signal, not the filtered CMMCorePlus variant.
    "test_device_property_events": "devicePropertyChanged callable filter not supported",
    # These check isinstance(signal, PSignalInstance) — proxy uses plain psygnal.Signal
    "test_event_signatures": "isinstance(signal, PSignalInstance) check on proxy signals",
    "test_deprecated_event_signatures": "isinstance + pytest.warns(FutureWarning) on proxy",
    # These test CMMCorePlus/signaler internals, not proxy-relevant
    "test_signal_backend_selection": "tests CMMCorePlus signal backend selection internals",
    "test_events_protocols": "tests signaler class internals, not proxy-relevant",
    # --- test_mda.py ---
    # Requires qtbot (Qt event loop)
    "test_mda_failures": "requires qtbot + patch.object on remote engine",
    "test_autofocus": "requires qtbot + mock_fullfocus (can't monkeypatch remote)",
    "test_autofocus_relative_z_plan": "requires mock_fullfocus + weakref on proxy",
    "test_autofocus_retries": "requires mock_fullfocus_failure fixture",
    "test_set_mda_fov": "requires qtbot",
    "test_mda_iterable_of_events": "requires qtbot",
    # These use MagicMock(wraps=engine) + set_engine() + qtbot — can't mock remote engine
    "test_runner_cancel": "MagicMock wrapping of remote engine + qtbot",
    "test_runner_pause": "MagicMock wrapping of remote engine + qtbot",
    # Requires passing local Python objects (MagicMock engines) to server
    "test_engine_protocol": "can't pass local MyEngine to remote server",
    "test_queue_mda": "can't pass MagicMock(wraps=engine) to remote server",
    # Uses weakref on output handlers — doesn't work through proxy
    "test_get_handlers": "weakref on proxy output handlers not supported",
    # caplog won't capture server-side logs
    "test_mda_no_device": "caplog can't capture server-side log messages",
    # Uses _warn_focus_dir.cache_clear() (server-side) + pytest.warns for
    # server-side warnings
    "test_restore_initial_state": "pytest.warns can't capture server-side warnings",
    "test_restore_initial_state_enabled_by_default":
        "setFocusDirection with enum arg + shared session-scoped core state",
}


# ---------- conftest content ----------

CONFTEST = '''\
"""Compat conftest — provides proxy fixtures for pymmcore-plus tests.

Replaces pymmcore-plus's own conftest. The ``core`` fixture yields a
RemoteMMCore connected to a ProxyServer wrapping CMMCorePlus.

Tests that are known to be incompatible with the proxy are auto-skipped
with a reason explaining the incompatibility.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Iterator

import httpx
import pytest
import uvicorn

from pymmcore_proxy import ProxyServer, RemoteMMCore

# --- Incompatible tests (injected by run_compat_tests.py) ---
SKIP_TESTS = {skip_tests_repr}


class _AutoFlushCore:
    """Wraps RemoteMMCore so every public method call flushes signals.

    This makes signal delivery appear synchronous, which is required by
    pymmcore-plus tests that assert mock.assert_called() immediately
    after a method call.
    """

    def __init__(self, client):
        object.__setattr__(self, "_client", client)

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if callable(attr) and not name.startswith("_"):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                try:
                    self._client.flush_signals()
                except Exception:
                    pass
                return result
            return wrapper
        return attr

    def __setattr__(self, name, value):
        setattr(self._client, name, value)


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests listed in SKIP_TESTS with their reason."""
    for item in items:
        test_name = item.originalname or item.name
        if test_name in SKIP_TESTS:
            item.add_marker(pytest.mark.skip(reason=f"proxy: {SKIP_TESTS[test_name]}"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def _demo_core():
    """A CMMCorePlus instance with the DemoCamera configuration loaded."""
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    core.loadSystemConfiguration("MMConfig_Demo.cfg")
    core.mda.engine.use_hardware_sequencing = False
    return core


@pytest.fixture(scope="session")
def _server_url(_demo_core):
    """Start a ProxyServer in a background thread, yield its base URL."""
    port = _free_port()
    proxy = ProxyServer(_demo_core, port=port)

    config = uvicorn.Config(
        proxy.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Proxy server failed to start")

    yield url

    server.should_exit = True
    thread.join(timeout=5.0)


@pytest.fixture()
def core(_server_url) -> Iterator[Any]:
    """A RemoteMMCore client connected to the test server.

    This replaces pymmcore-plus\\'s core fixture.  Wrapped in
    _AutoFlushCore so signals appear synchronous.
    """
    client = RemoteMMCore(
        _server_url, connect_signals=True, timeout=60.0,
    )
    # Give the signal listener a moment to connect
    time.sleep(0.3)
    # Reset hardware state so tests are isolated
    client.loadSystemConfiguration("MMConfig_Demo.cfg")
    yield _AutoFlushCore(client)
    client.close()


@pytest.fixture
def mock_fullfocus():
    """Skip — can\\'t monkeypatch remote objects."""
    pytest.skip("proxy: can\\'t monkeypatch remote objects")


@pytest.fixture
def mock_fullfocus_failure():
    """Skip — can\\'t monkeypatch remote objects."""
    pytest.skip("proxy: can\\'t monkeypatch remote objects")


@pytest.fixture
def caplog(caplog):
    """Provide caplog — redirect pymmcore-plus logger to it."""
    from pymmcore_plus._logger import logger
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
'''


# ---------- main ----------


def main() -> int:
    pmp_tests = _find_pmp_tests()
    print(f"Found pymmcore-plus tests at: {pmp_tests}")

    with tempfile.TemporaryDirectory(prefix="proxy_compat_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Write our conftest with the skip list injected
        conftest_content = CONFTEST.replace(
            "{skip_tests_repr}", repr(SKIP_TESTS)
        )
        (tmpdir / "conftest.py").write_text(conftest_content)

        # Copy the local_config.cfg if it exists
        local_cfg = pmp_tests / "local_config.cfg"
        if local_cfg.exists():
            shutil.copy2(local_cfg, tmpdir / "local_config.cfg")

        # Symlink compatible test files
        for fname in COMPAT_FILES:
            src = pmp_tests / fname
            if src.exists():
                os.symlink(src, tmpdir / fname)
            else:
                print(f"  Warning: {fname} not found, skipping")

        # Run pytest
        extra_args = sys.argv[1:]
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(tmpdir),
            "--tb=short",
            "-v",
            *extra_args,
        ]
        print(f"Running: {' '.join(cmd[:6])} ...")
        print(f"  Skipping {len(SKIP_TESTS)} incompatible tests (with reasons)")
        result = subprocess.run(cmd)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
