# pymmcore-proxy

Network proxy for [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) — control microscopes remotely over HTTP and WebSocket.

The server wraps a `CMMCorePlus` instance and exposes it over the network. The client (`RemoteMMCore`) is a drop-in replacement that forwards all calls via RPC. Signals (property changes, MDA frame events, etc.) stream back to clients over WebSocket in real time.

## Installation

```bash
pip install -e ".[server,test]"
```

## Quick start

### Server (microscope machine)

```python
from pymmcore_plus import CMMCorePlus
from pymmcore_proxy import serve

core = CMMCorePlus()
core.loadSystemConfiguration("MMConfig_Demo.cfg")
serve(core, port=5600)
```

### Client (any machine on the network)

```python
from pymmcore_proxy import connect

core = connect("http://microscope-host:5600")

# Use exactly like CMMCorePlus:
core.snapImage()
img = core.getImage()           # real numpy array
core.setXYPosition(100, 200)
core.setExposure(50.0)

# Signals work:
core.events.propertyChanged.connect(my_callback)

# MDA runs on the server, signals stream back:
from useq import MDASequence
core.mda.events.frameReady.connect(on_frame)
core.mda.run(MDASequence(time_plan={"interval": 1, "loops": 10}))

core.close()
```

## Architecture

```
 Client (RemoteMMCore)                    Server (ProxyServer)
 ~~~~~~~~~~~~~~~~~~~~~~                   ~~~~~~~~~~~~~~~~~~~~
      |                                         |
      |--- POST /rpc {"method": "snapImage"} -->|--- core.snapImage()
      |<-- {"ok": true, "value": ...} ----------|
      |                                         |
      |--- WS /signals <------------------------|--- core.events.*.connect(...)
      |    (propertyChanged, frameReady, ...)   |
      |                                         |
      |--- WS /mda/stream --------------------->|--- core.mda.run(iterator)
      |    (streaming events from Queue/gen)    |
```

### RPC

All method calls go through `POST /rpc` with JSON payloads. The server resolves dotted paths (`mda.engine.setup_event`), so nested attribute access works transparently. Three actions are supported:

- **call** (default) — call a method: `{"method": "setExposure", "args": [50.0]}`
- **getattr** — read an attribute: `{"action": "getattr", "attr": "mda._running"}`
- **setattr** — write an attribute: `{"action": "setattr", "attr": "mda.engine.use_hardware_sequencing", "value": false}`

Long-running calls (like `mda.run()`) execute in `asyncio.to_thread()` so signal forwarding is never blocked.

### Signals

Clients connect to `WS /signals` to receive real-time signal notifications. The server hooks into `core.events` and `core.mda.events`, serializes signal arguments, and broadcasts them to all connected WebSocket clients. The client deserializes and emits them on local `psygnal` signal objects.

### Streaming MDA

`core.mda.run()` accepts three input types:

| Input type | Transport | Use case |
|---|---|---|
| `MDASequence` or `dict` | Single RPC call | Pre-defined acquisition plans |
| List / generator / Queue iterator | `WS /mda/stream` | Reactive feedback loops |

For streaming, events are sent one-by-one over a dedicated WebSocket. The server feeds them into a queue-backed iterator that the MDA runner consumes. Cancel is handled by injecting a sentinel into the queue directly from the `mda.cancel` RPC handler — no polling.

### Serialization

All data crosses the wire as JSON. Custom types are tagged with `"__type__"`:

- **numpy arrays** — dtype + shape + base64 data
- **pydantic models** (MDAEvent, MDASequence) — module + class + `model_dump()`
- **tuples, bytes, enums** — tagged wrappers

## Tests

```bash
# Run the proxy's own test suite (52 tests)
pytest

# Run pymmcore-plus tests against the proxy (requires pymmcore-plus repo as sibling)
python scripts/run_compat_tests.py
```

## pymmcore-plus compatibility

The compat test runner (`scripts/run_compat_tests.py`) runs pymmcore-plus's `test_events.py` and `test_mda.py` against `RemoteMMCore`. Out of 73 tests, **47 pass** and **26 are skipped** with documented reasons.

The skipped tests fall into a few categories — none represent limitations of normal proxy usage. They are all test-specific issues where the test infrastructure assumes in-process access.

### CMMCorePlus internals (3 tests)

These test signal backend selection, signaler protocol conformance, and deprecated-signature warnings — internal implementation details specific to CMMCorePlus.

| Test | Reason |
|---|---|
| `test_signal_backend_selection` | Tests `CMMCoreSignaler` / `QCoreSignaler` selection logic |
| `test_events_protocols` | Tests signaler class protocol conformance |
| `test_deprecated_event_signatures` (2 params) | Tests `FutureWarning` on deprecated callback signatures |

### Requires Qt event loop (6 tests)

These tests use `qtbot` (pytest-qt) to process events. The proxy doesn't require or use Qt.

| Test | Additional blocker |
|---|---|
| `test_mda_failures` | Also needs `patch.object` on remote engine |
| `test_autofocus` | Also needs `mock_fullfocus` fixture |
| `test_autofocus_relative_z_plan` | Also needs `mock_fullfocus` |
| `test_autofocus_retries` | Also needs `mock_fullfocus_failure` |
| `test_set_mda_fov` | |
| `test_mda_iterable_of_events` (3 params) | |

### Can't mock/monkeypatch remote objects (4 tests)

These tests use `MagicMock(wraps=engine)` or `patch.object()` to instrument the MDA engine. This can't work through a network proxy — the engine lives on the server.

| Test | Reason |
|---|---|
| `test_runner_cancel` | `MagicMock(wraps=engine)` + qtbot |
| `test_runner_pause` | `MagicMock(wraps=engine)` + qtbot |
| `test_engine_protocol` | Passes a local `MyEngine` class to remote server |
| `test_queue_mda` | `MagicMock(wraps=engine)` to remote server |

### Can't capture server-side logs (1 test)

| Test | Reason |
|---|---|
| `test_mda_no_device` (5 params) | `caplog` flooded with httpcore debug logs |

### Other (1 test)

| Test | Reason |
|---|---|
| `test_get_handlers` | Uses `weakref` on output handlers — doesn't work through proxy |

### Tests that pass (47)

These pymmcore-plus tests run correctly against `RemoteMMCore`. Event tests use the signal flush mechanism (`_AutoFlushCore` wrapper) to make async signal delivery appear synchronous. Server-side warnings are forwarded to clients via WebSocket.

| Test | What it verifies |
|---|---|
| `test_set_property_events` | `propertyChanged` fires on `setProperty` |
| `test_set_state_events` | `propertyChanged` fires on `setState`/`setStateLabel` |
| `test_set_statedevice_property_emits_events` | `propertyChanged` fires on state device property changes |
| `test_device_property_events` | `devicePropertyChanged` callable filter routes by device/property |
| `test_shutter_device_events` | `propertyChanged` fires on `setShutterOpen` |
| `test_set_focus_device` | `propertyChanged` fires on `setFocusDevice` |
| `test_sequence_acquisition_events` | Sequence acquisition start/stop signals fire |
| `test_autoshutter_device_events` | `autoShutterSet` fires on `setAutoShutter` |
| `test_groups_and_presets_events` | `configDeleted`/`configGroupDeleted`/`configDefined` fire |
| `test_set_camera_roi_event` | `roiSet` fires on `setROI` |
| `test_pixel_changed_event` | `pixelSizeChanged` fires on pixel config changes |
| `test_set_channelgroup` | `channelGroupChanged` fires on `setChannelGroup` |
| `test_event_signatures` (24 params) | All signal types satisfy `PSignalInstance` protocol |
| `test_mda_waiting` | MDA with time intervals waits correctly |
| `test_setting_position` | XY/Z positions are set during MDA |
| `test_keep_shutter_open` | Shutter stays open across channels when configured |
| `test_reset_event_timer` | Event timer resets between MDA events |
| `test_custom_action` | Custom action events are handled |
| `test_restore_initial_state` (3 params) | Hardware state restored after MDA (with `pytest.warns` for unknown focus direction) |
| `test_restore_initial_state_enabled_by_default` (3 params) | State restoration auto-enabled based on `FocusDirection` |
