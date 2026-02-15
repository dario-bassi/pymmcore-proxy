# pymmcore-proxy

> **Warning:** This project is experimental and under active development. APIs may change without notice.

Network proxy for [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus): control microscopes remotely over HTTP and WebSocket.

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

### Custom MDA engines

Custom MDA engines must be registered on the server side — they cannot be passed from the client since they are Python objects that need direct access to the hardware:

```python
# Server setup
from pymmcore_plus import CMMCorePlus
from pymmcore_proxy import serve

core = CMMCorePlus()
core.loadSystemConfiguration("MMConfig_Demo.cfg")
core.mda.set_engine(MyCustomEngine(core))  # register before serving
serve(core, port=5600)
```

Once registered, the custom engine is used transparently for all MDA runs initiated by any client. Engine properties can be read and written remotely via `core.mda.engine.<attr>`.

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

The compat test runner (`scripts/run_compat_tests.py`) runs 18 of 21 pymmcore-plus test files against `RemoteMMCore`. Out of 233 tests, **186 pass** and **46 are skipped** with documented reasons. Tests that don't use the `core` fixture run as local pymmcore-plus tests to verify no interference from the proxy test infrastructure.

The skipped tests fall into a few categories, none should represent limitations of normal proxy usage. They are all test-specific issues where the test infrastructure assumes in-process access.

### Test files not included (3 files)

| File | Reason |
|---|---|
| `test_sequencing.py` | Hardware sequencing internals + hangs without qtbot |
| `test_bench.py` | Benchmarks, no functional tests |
| `test_thread_relay.py` | Signal threading internals, no tests collected |

### Requires Qt event loop (12 tests)

| Test | File |
|---|---|
| `test_cb_exceptions` | test_core.py |
| `test_mda` | test_core.py |
| `test_mda_pause_cancel` | test_core.py |
| `test_register_mda_engine` | test_core.py |
| `test_not_concurrent_mdas` | test_core.py |
| `test_snap_signals` | test_core.py |
| `test_mda_failures` | test_mda.py |
| `test_autofocus` | test_mda.py |
| `test_autofocus_relative_z_plan` | test_mda.py |
| `test_autofocus_retries` | test_mda.py |
| `test_set_mda_fov` | test_mda.py |
| `test_mda_iterable_of_events` (3 params) | test_mda.py |

### isinstance check on core object (1 test)

| Test | Reason |
|---|---|
| `test_core` | `isinstance(core, CMMCorePlus)` — `RemoteMMCore` is not a `CMMCorePlus` subclass |

### Can't mock/monkeypatch remote objects (7 tests)

| Test | Reason |
|---|---|
| `test_guess_channel_group` | `patch.object` on core |
| `test_set_autofocus_offset` | `monkeypatch` on internal dict |
| `test_runner_cancel` | `MagicMock(wraps=engine)` |
| `test_runner_pause` | `MagicMock(wraps=engine)` |
| `test_engine_protocol` | Passes local `MyEngine` to server |
| `test_queue_mda` | `MagicMock(wraps=engine)` |
| `test_describe` | `capsys` can't capture server stdout |

### Proxy transport limitations (3 tests)

| Test | Reason |
|---|---|
| `test_search_paths` | `os.getenv` on client doesn't reflect server-side PATH |
| `test_load_system_config` | macOS `/var` symlink: server resolves to `/private/var` |
| `test_get_handlers` | `weakref` on proxy objects |

### Test environment limitations (8 tests)

| Test | Reason |
|---|---|
| `test_ome_generation` (5 params) | `local_config.cfg` not found (test creates own CMMCorePlus) |
| `test_ome_generation_from_events` | `local_config.cfg` not found (test creates own CMMCorePlus) |
| `test_stupidly_empty_metadata` | Requires `lxml` or `xmlschema` |
| `test_create_schema` | Requires `lxml` or `xmlschema` |

### CMMCorePlus internals (3 tests)

| Test | Reason |
|---|---|
| `test_signal_backend_selection` | Signal backend selection logic |
| `test_events_protocols` | Signaler protocol conformance |
| `test_deprecated_event_signatures` (2 params) | Deprecated-signature `FutureWarning` |
