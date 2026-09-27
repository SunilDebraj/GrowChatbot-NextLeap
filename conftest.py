"""Session-wide guard: the test suite must never open an outbound connection.

GR14 says the tests are hermetic. That was easy to honour while
``generation.provider`` was a ``<TBD: OQ2>`` placeholder, because nothing could
be reached even by accident. It is no longer free: the shipped config names a
real provider, so a test that constructs a client and forgets to stub it will
spend money, take seconds, and assert on whatever the model happened to say.

The first version of that failure was silent. ``AnswerPipeline.build(llm=None)``
resolved a live client and ``test_absent_llm_fails_closed`` began testing
Groq's opinion of HDFC's exit load. Nothing about the test looked wrong.

So the rule is enforced here rather than trusted: any attempt to connect to a
non-loopback address during a test raises immediately, naming the test that
tried. Loopback stays permitted, since the ASGI tests drive the app in-process
and some fixtures bind a local port.
"""

from __future__ import annotations

import socket

import pytest

#: Set by the autouse fixture. A plain module flag rather than a monkeypatch of
#: socket.socket itself, so the failure message can name the offending test.
_OFFENDING_TEST: list[str] = []


def _is_loopback(host: object) -> bool:
    if not isinstance(host, str):
        # AF_UNIX and similar. Not an outbound network connection.
        return True
    if host in ("localhost", "", "::1"):
        return True
    return host.startswith("127.") or host == "::1"


@pytest.fixture(autouse=True)
def _no_outbound_network(request):
    """Fail loudly if a test tries to reach the network."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    _OFFENDING_TEST.clear()
    _OFFENDING_TEST.append(request.node.nodeid)

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_loopback(host):
            raise RuntimeError(
                f"outbound network blocked: {request.node.nodeid} tried to connect to "
                f"{host!r}. Tests must be hermetic (GR14). Inject a stub LLM "
                f"(FakeLLM / a ScriptedLLM) instead of calling a provider."
            )
        return original_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_loopback(host):
            raise RuntimeError(
                f"outbound network blocked: {request.node.nodeid} tried to connect to {host!r}"
            )
        return original_connect_ex(self, address, *args, **kwargs)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        _OFFENDING_TEST.clear()
