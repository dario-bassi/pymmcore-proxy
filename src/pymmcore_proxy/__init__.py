"""pymmcore-proxy — network proxy for pymmcore-plus.

Start a server wrapping a real CMMCorePlus/UniMMCore instance,
then connect from anywhere with a drop-in client.

Server (microscope side):
    from pymmcore_proxy import serve
    serve(core, port=5600)

Client (agent side):
    from pymmcore_proxy import connect
    core = connect("http://127.0.0.1:5600")
    core.snapImage()
    img = core.getImage()  # real numpy array
"""

from .client import RemoteMMCore  # noqa: F401
from .server import ProxyServer, serve  # noqa: F401

connect = RemoteMMCore  # convenience alias

__all__ = ["connect", "RemoteMMCore", "serve", "ProxyServer"]
