"""Tests for server-driven MDA execution."""

import threading
import time

import numpy as np
import pytest
from useq import MDASequence


class TestMDABasics:
    def test_run_simple_sequence(self, core):
        """Run a simple time-lapse and collect frames."""
        frames = []

        def on_frame(img, event, meta):
            frames.append((img, event, meta))

        core.mda.events.frameReady.connect(on_frame)
        try:
            seq = MDASequence(
                time_plan={"interval": 0.01, "loops": 3},
            )
            core.mda.run(seq)
            assert len(frames) == 3
            for img, event, meta in frames:
                assert isinstance(img, np.ndarray)
                assert img.ndim == 2
        finally:
            core.mda.events.frameReady.disconnect(on_frame)

    def test_sequence_started_finished_signals(self, core):
        """sequenceStarted and sequenceFinished should fire."""
        started = []
        finished = []

        core.mda.events.sequenceStarted.connect(
            lambda seq, meta: started.append(seq)
        )
        core.mda.events.sequenceFinished.connect(
            lambda seq: finished.append(seq)
        )

        seq = MDASequence(time_plan={"interval": 0.01, "loops": 2})
        core.mda.run(seq)

        assert len(started) == 1
        assert len(finished) == 1


class TestMDAWithPositions:
    def test_z_stack(self, core):
        """Run a Z-stack and verify frames are collected."""
        frames = []
        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )

        seq = MDASequence(z_plan={"range": 4.0, "step": 2.0})
        core.mda.run(seq)

        # range=4, step=2 → 3 steps (-2, 0, 2)
        assert len(frames) >= 2


class TestMDACancel:
    def test_cancel_stops_sequence(self, core):
        """Canceling mid-sequence should stop and emit sequenceCanceled."""
        frames = []
        canceled = []

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 2:
                core.mda.cancel()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequenceCanceled.connect(
            lambda seq: canceled.append(seq)
        )

        seq = MDASequence(time_plan={"interval": 0.01, "loops": 100})
        core.mda.run(seq)

        assert len(frames) < 100, "Sequence was not canceled"
        assert len(canceled) == 1

    def test_pause_and_resume(self, core):
        """Pausing should delay frames, resuming should continue."""
        frames = []
        paused_signals = []

        def on_frame(img, ev, meta):
            frames.append(img)
            if len(frames) == 1:
                core.mda.toggle_pause()
                # Resume after a short delay (from another thread)
                def resume():
                    time.sleep(0.3)
                    core.mda.toggle_pause()
                threading.Thread(target=resume, daemon=True).start()

        core.mda.events.frameReady.connect(on_frame)
        core.mda.events.sequencePauseToggled.connect(
            lambda p: paused_signals.append(p)
        )

        seq = MDASequence(time_plan={"interval": 0.01, "loops": 5})
        core.mda.run(seq)

        assert len(frames) == 5, "All frames should complete after resume"
        assert len(paused_signals) == 2  # pause + unpause


class TestMDAFromDict:
    def test_run_from_dict(self, core):
        """MDA.run() should accept a dict that gets converted to MDASequence."""
        frames = []
        core.mda.events.frameReady.connect(
            lambda img, ev, meta: frames.append(img)
        )

        core.mda.run({"time_plan": {"interval": 0.01, "loops": 2}})
        assert len(frames) == 2
