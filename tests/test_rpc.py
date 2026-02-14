"""Integration tests for RPC calls through the proxy."""

import numpy as np
import pytest


class TestHealthAndBasics:
    def test_health(self, core_no_signals):
        result = core_no_signals.health()
        assert result["status"] == "ok"

    def test_repr(self, core_no_signals):
        assert "RemoteMMCore" in repr(core_no_signals)


class TestDeviceInfo:
    def test_get_loaded_devices(self, core_no_signals):
        devices = core_no_signals.getLoadedDevices()
        assert isinstance(devices, (list, tuple))
        assert len(devices) > 0

    def test_get_camera_device(self, core_no_signals):
        cam = core_no_signals.getCameraDevice()
        assert isinstance(cam, str)
        assert len(cam) > 0

    def test_get_xy_stage_device(self, core_no_signals):
        stage = core_no_signals.getXYStageDevice()
        assert isinstance(stage, str)


class TestProperties:
    def test_get_exposure(self, core_no_signals):
        exp = core_no_signals.getExposure()
        assert isinstance(exp, (int, float))
        assert exp > 0

    def test_set_and_get_exposure(self, core_no_signals):
        core_no_signals.setExposure(100.0)
        assert core_no_signals.getExposure() == pytest.approx(100.0)
        core_no_signals.setExposure(50.0)  # restore

    def test_get_image_width_height(self, core_no_signals):
        w = core_no_signals.getImageWidth()
        h = core_no_signals.getImageHeight()
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w > 0
        assert h > 0


class TestImaging:
    def test_snap_and_get_image(self, core_no_signals):
        core_no_signals.snapImage()
        img = core_no_signals.getImage()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_image_is_writable(self, core_no_signals):
        core_no_signals.snapImage()
        img = core_no_signals.getImage()
        img[0, 0] = 0  # should not raise


class TestStage:
    def test_set_and_get_xy(self, core_no_signals):
        core_no_signals.setXYPosition(123.0, 456.0)
        core_no_signals.waitForDevice(core_no_signals.getXYStageDevice())
        x = core_no_signals.getXPosition()
        y = core_no_signals.getYPosition()
        assert x == pytest.approx(123.0, abs=1.0)
        assert y == pytest.approx(456.0, abs=1.0)

    def test_set_and_get_z(self, core_no_signals):
        focus = core_no_signals.getFocusDevice()
        core_no_signals.setPosition(focus, 10.0)
        core_no_signals.waitForDevice(focus)
        z = core_no_signals.getPosition(focus)
        assert z == pytest.approx(10.0, abs=1.0)


class TestConfig:
    def test_get_available_config_groups(self, core_no_signals):
        groups = core_no_signals.getAvailableConfigGroups()
        assert isinstance(groups, (list, tuple))

    def test_get_available_configs(self, core_no_signals):
        groups = core_no_signals.getAvailableConfigGroups()
        if groups:
            configs = core_no_signals.getAvailableConfigs(groups[0])
            assert isinstance(configs, (list, tuple))


class TestErrorHandling:
    def test_nonexistent_method(self, core_no_signals):
        with pytest.raises(RuntimeError, match="AttributeError"):
            core_no_signals.thisMethodDoesNotExist()

    def test_bad_args(self, core_no_signals):
        with pytest.raises(RuntimeError):
            core_no_signals.setExposure("not_a_number")
