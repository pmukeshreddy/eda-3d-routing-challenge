"""Wall-clock containment for the unchanged Block 1 Python/native adapter."""
from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Queue
from threading import Thread
from time import perf_counter

from .wire_engine import WireEngine, WireResult


def _serve(connection, instance):
    try:
        engine = WireEngine(instance)
        connection.send((True, None))
        while True:
            nid, kwargs = connection.recv()
            try:
                connection.send((True, engine.route_net(nid, **kwargs)))
            except Exception as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class BoundedWireEngine:
    """One reusable native engine per episode, killed on a call timeout.

    The parent keeps all layout state. A killed worker cannot install a late
    result. The next call starts a fresh worker; it never chooses a net itself.
    Use from a guarded main when embedding in an executable Python script.
    """

    def __init__(self, instance):
        self._instance = instance
        self._process = None
        self._connection = None

    def _receive(self, deadline, request=None):
        # poll() alone is insufficient: it can see a frame header while recv()
        # still blocks on the body. Bound the entire send/receive transaction.
        # Killing the worker on timeout closes its pipe end and releases this
        # daemon thread. Capture this generation's connection, never a restart's.
        connection = self._connection
        response = Queue(maxsize=1)

        def exchange():
            try:
                if request is not None:
                    connection.send(request)
                response.put((None, connection.recv()))
            except Exception as exc:
                response.put((exc, None))

        Thread(target=exchange, daemon=True).start()
        try:
            error, result = response.get(timeout=max(0.0, deadline - perf_counter()))
        except Empty as exc:
            raise TimeoutError("Block 1 call exceeded its wall-clock allowance") from exc
        if error is not None:
            raise error
        ok, value = result
        if not ok:
            raise RuntimeError(value)
        return value

    def route_net(self, net_id: int, *, timeout_s: float, **kwargs) -> WireResult:
        deadline = perf_counter() + timeout_s
        try:
            if self._process is None:
                context = mp.get_context("spawn")
                parent, child = context.Pipe()
                self._connection = parent
                self._process = context.Process(target=_serve, args=(child, self._instance),
                                                daemon=True)
                self._process.start()
                child.close()
                self._receive(deadline)
            if perf_counter() >= deadline:
                raise TimeoutError("Block 1 startup consumed the call allowance")
            result = self._receive(deadline, (net_id, kwargs))
            if perf_counter() > deadline:
                raise TimeoutError("Block 1 response arrived after the call deadline")
            return result
        except (TimeoutError, RuntimeError):
            self.close()
            raise
        except (EOFError, OSError) as exc:
            self.close()
            raise RuntimeError(f"Block 1 worker failed: {exc}") from exc

    def close(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.2)
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
