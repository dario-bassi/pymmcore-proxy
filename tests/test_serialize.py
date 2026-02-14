"""Unit tests for the wire-format serialization."""

import numpy as np
import pytest

from pymmcore_proxy._serialize import decode, encode


class TestEncodeDecode:
    """Roundtrip tests for encode → decode."""

    def test_none(self):
        assert decode(encode(None)) is None

    def test_bool(self):
        assert decode(encode(True)) is True
        assert decode(encode(False)) is False

    def test_int(self):
        assert decode(encode(42)) == 42

    def test_float(self):
        assert decode(encode(3.14)) == pytest.approx(3.14)

    def test_string(self):
        assert decode(encode("hello")) == "hello"

    def test_list(self):
        assert decode(encode([1, "two", 3.0])) == [1, "two", 3.0]

    def test_dict(self):
        d = {"a": 1, "b": [2, 3]}
        assert decode(encode(d)) == d

    def test_nested_dict(self):
        d = {"outer": {"inner": [1, 2, 3]}}
        assert decode(encode(d)) == d

    def test_tuple(self):
        t = (1, "two", 3.0)
        result = decode(encode(t))
        assert result == t
        assert isinstance(result, tuple)

    def test_bytes(self):
        b = b"\x00\x01\x02\xff"
        result = decode(encode(b))
        assert result == b
        assert isinstance(result, bytes)

    def test_ndarray_uint16(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = decode(encode(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.uint16

    def test_ndarray_float32(self):
        arr = np.random.rand(10, 10).astype(np.float32)
        result = decode(encode(arr))
        np.testing.assert_array_almost_equal(result, arr)
        assert result.dtype == np.float32

    def test_ndarray_3d(self):
        arr = np.zeros((3, 512, 512), dtype=np.uint8)
        result = decode(encode(arr))
        assert result.shape == (3, 512, 512)
        assert result.dtype == np.uint8

    def test_ndarray_is_writable(self):
        """Decoded arrays must be writable (not read-only frombuffer views)."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = decode(encode(arr))
        result[0] = 99  # should not raise
        assert result[0] == 99

    def test_numpy_scalar_int(self):
        val = np.int64(42)
        result = encode(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_scalar_float(self):
        val = np.float32(3.14)
        result = encode(val)
        assert isinstance(result, float)

    def test_numpy_bool(self):
        val = np.bool_(True)
        result = encode(val)
        assert result is True

    def test_mda_event_roundtrip(self):
        from useq import MDAEvent

        event = MDAEvent(exposure=50.0, x_pos=100.0, y_pos=200.0)
        encoded = encode(event)
        assert encoded["__type__"] == "model"
        result = decode(encoded)
        assert isinstance(result, MDAEvent)
        assert result.exposure == 50.0
        assert result.x_pos == 100.0

    def test_mda_sequence_roundtrip(self):
        from useq import MDASequence

        seq = MDASequence(
            time_plan={"interval": 1.0, "loops": 3},
            z_plan={"range": 5.0, "step": 1.0},
        )
        encoded = encode(seq)
        result = decode(encoded)
        assert isinstance(result, MDASequence)

    def test_unknown_object_fallback(self):
        """Objects without a known serializer become strings."""

        class Foo:
            def __repr__(self):
                return "Foo()"

        result = encode(Foo())
        assert result == "Foo()"
