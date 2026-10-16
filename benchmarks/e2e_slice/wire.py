"""The one link between the slice's prover and its challenger: JSON lines over a TCP socket (loopback, or the challenger's
machine's address when it runs elsewhere), bytes as hex.

A reader thread files each message under (kind, round), so the prover's round loop and its proof worker each wait on their
own (`take`); `send` is safe from any thread."""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
from collections import defaultdict


class Wire:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.r, self.w = sock.makefile("r"), sock.makefile("w")
        self.lock = threading.Lock()
        self.boxes: dict = defaultdict(queue.Queue)
        self.closed = threading.Event()

    def send(self, kind: str, **fields) -> None:
        with self.lock:
            self.w.write(json.dumps({"kind": kind, **fields}) + "\n")
            self.w.flush()

    def recv(self) -> dict | None:
        line = self.r.readline()
        return json.loads(line) if line else None

    def start_reader(self) -> None:
        def run():
            while (msg := self.recv()) is not None:
                self.boxes[(msg["kind"], msg.get("round"))].put(msg)
            self.closed.set()
            for b in list(self.boxes.values()):
                b.put(None)
        threading.Thread(target=run, daemon=True).start()

    def take(self, kind: str, rnd: int | None = None) -> dict:
        box = self.boxes[(kind, rnd)]
        if self.closed.is_set() and box.empty():
            raise EOFError(f"the link closed before {kind} {rnd}")
        msg = box.get()
        if msg is None:
            raise EOFError(f"the link closed waiting for {kind} {rnd}")
        return msg


def connect(port: int, host: str = "127.0.0.1", tries: int = 1) -> Wire:
    """The prover's end; ``tries`` attempts a second apart (a challenger on another machine may still be staging)."""
    for i in range(tries):
        try:
            return Wire(socket.create_connection((host, port)))
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(1)
    raise ValueError("tries >= 1")


def listen(host: str = "127.0.0.1", port: int = 0) -> tuple[socket.socket, int]:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    return s, s.getsockname()[1]
