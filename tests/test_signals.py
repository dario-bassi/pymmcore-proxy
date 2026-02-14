"""Tests for signal forwarding over WebSocket."""

import threading
import time

import pytest


class TestCoreSignals:
    def test_exposure_changed_signal(self, core):
        """Changing exposure on the server should fire exposureChanged locally."""
        received = []
        event = threading.Event()

        def on_exposure(*args):
            received.append(args)
            event.set()

        core.events.exposureChanged.connect(on_exposure)
        try:
            core.setExposure(77.0)
            assert event.wait(timeout=3.0), "exposureChanged signal not received"
            assert len(received) >= 1
        finally:
            core.events.exposureChanged.disconnect(on_exposure)

    def test_property_changed_signal(self, core):
        """Setting a property should fire propertyChanged."""
        received = []
        event = threading.Event()

        def on_prop(*args):
            received.append(args)
            event.set()

        core.events.propertyChanged.connect(on_prop)
        try:
            # DemoCamera has an "Exposure" property on the "Camera" device
            core.setProperty("Camera", "Exposure", "50")
            assert event.wait(timeout=3.0), "propertyChanged signal not received"
            assert len(received) >= 1
        finally:
            core.events.propertyChanged.disconnect(on_prop)

    def test_xy_stage_signal(self, core):
        """Moving XY stage should fire XYStagePositionChanged."""
        received = []
        event = threading.Event()

        def on_xy(*args):
            received.append(args)
            event.set()

        core.events.XYStagePositionChanged.connect(on_xy)
        try:
            core.setXYPosition(10.0, 20.0)
            core.waitForDevice(core.getXYStageDevice())
            assert event.wait(timeout=3.0), "XYStagePositionChanged signal not received"
            assert len(received) >= 1
        finally:
            core.events.XYStagePositionChanged.disconnect(on_xy)


class TestSignalReconnect:
    def test_client_without_signals(self, core_no_signals):
        """A client with connect_signals=False should still work for RPC."""
        core_no_signals.snapImage()
        img = core_no_signals.getImage()
        assert img is not None
