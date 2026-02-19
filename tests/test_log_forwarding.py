"""Tests for server-side log and warning forwarding over WebSocket."""

import logging
import threading
import time

from useq import MDAEvent, MDASequence


class TestLogForwarding:
    def test_mda_logs_forwarded_to_client(self, core):
        """Server-side MDA runner INFO logs should arrive on the client."""
        received = []
        event = threading.Event()

        class _Collector(logging.Handler):
            def emit(self, record):
                received.append(record)
                event.set()

        pmm_logger = logging.getLogger("pymmcore-plus")
        handler = _Collector(level=logging.INFO)
        pmm_logger.addHandler(handler)
        try:
            seq = MDASequence(time_plan={"interval": 0.01, "loops": 2})
            core.mda.run(seq)

            assert event.wait(timeout=5.0), "No log records received"
            # The MDA runner logs "MDA Started", each event, and "MDA Finished"
            info_msgs = [r for r in received if r.levelno >= logging.INFO]
            assert len(info_msgs) >= 2, (
                f"Expected at least 2 INFO records, got {len(info_msgs)}: "
                f"{[r.getMessage()[:60] for r in info_msgs]}"
            )
        finally:
            pmm_logger.removeHandler(handler)

    def test_log_record_has_server_origin(self, core):
        """Forwarded log records should preserve the server filename/lineno."""
        received = []
        done = threading.Event()

        class _Collector(logging.Handler):
            def emit(self, record):
                received.append(record)
                done.set()

        pmm_logger = logging.getLogger("pymmcore-plus")
        handler = _Collector(level=logging.INFO)
        pmm_logger.addHandler(handler)
        try:
            core.mda.run(MDASequence(time_plan={"interval": 0.01, "loops": 1}))
            assert done.wait(timeout=5.0)

            filenames = [r.filename for r in received]
            assert any("_runner" in f for f in filenames), (
                f"Expected a record from _runner.py, got filenames: {filenames}"
            )
        finally:
            pmm_logger.removeHandler(handler)

    def test_streaming_mda_logs_forwarded(self, core):
        """Log forwarding should also work for streaming (iterator-based) MDA."""
        received = []
        done = threading.Event()

        class _Collector(logging.Handler):
            def emit(self, record):
                received.append(record)
                # "MDA Finished" means all per-event logs have been sent
                if "MDA Finished" in record.getMessage():
                    done.set()

        pmm_logger = logging.getLogger("pymmcore-plus")
        handler = _Collector(level=logging.INFO)
        pmm_logger.addHandler(handler)
        try:
            events = [
                MDAEvent(exposure=10),
                MDAEvent(exposure=20),
                MDAEvent(exposure=30),
            ]
            core.mda.run(events)

            assert done.wait(timeout=5.0), (
                f"MDA Finished not received. Got: "
                f"{[r.getMessage()[:60] for r in received]}"
            )
            # Should have: MDA Started, 3 per-event logs, MDA Finished = 5
            info_msgs = [r for r in received if r.levelno >= logging.INFO]
            assert len(info_msgs) >= 4, (
                f"Expected at least 4 INFO records (started + events + finished), "
                f"got {len(info_msgs)}"
            )
        finally:
            pmm_logger.removeHandler(handler)

    def test_warning_dispatch_on_client(self, core):
        """_internal._warning signals should be re-emitted as Python warnings."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Simulate what the server sends when a pymmcore warning fires
            core._dispatch_signal({
                "group": "_internal",
                "signal": "_warning",
                "args": ["UserWarning", "test server warning"],
            })
            assert len(w) >= 1, f"Expected a warning, got {len(w)}"
            assert "test server warning" in str(w[0].message)
            assert w[0].category is UserWarning

    def test_warning_level_log_forwarded(self, core):
        """WARNING-level log records should also arrive via _internal._log."""
        received = []
        event = threading.Event()

        class _Collector(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    received.append(record)
                    event.set()

        pmm_logger = logging.getLogger("pymmcore-plus")
        handler = _Collector(level=logging.WARNING)
        pmm_logger.addHandler(handler)
        try:
            # Simulate what the server sends for a WARNING-level log record
            core._dispatch_signal({
                "group": "_internal",
                "signal": "_log",
                "args": [logging.WARNING, "test warning log", "test_file.py", 42],
            })
            assert event.wait(timeout=2.0), "No WARNING log records received"
            assert len(received) >= 1
            assert "test warning log" in received[0].getMessage()
            assert received[0].levelno == logging.WARNING
        finally:
            pmm_logger.removeHandler(handler)
