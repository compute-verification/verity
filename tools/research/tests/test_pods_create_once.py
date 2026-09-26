"""`research pods create` makes at most one pod per --name: a live pod of the name refuses another, a pod that appears despite a
failed create call is used, and only RunPod's explicit no-stock reply lets a caller try another spec."""
from __future__ import annotations

import argparse

from research.pods import part, runpod


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(name="vy-lane", allow_duplicate=False, **kw)


def _fake(monkeypatch, pods: list[dict], create):
    monkeypatch.setattr(runpod, "_request", lambda method, path, body=None: list(pods))
    monkeypatch.setattr(runpod, "create", create)


def test_a_live_pod_of_the_name_refuses_a_second(monkeypatch, capsys):
    made = []
    _fake(monkeypatch, [{"id": "p1", "name": "vy-lane-veritor-campaign", "desiredStatus": "RUNNING"}], lambda ns: made.append(1))
    assert part.create_once(_ns()) == (None, part.EXIT_EXISTS) and not made
    assert "already has live pod(s) p1" in capsys.readouterr().err


def test_a_terminated_pod_of_the_name_does_not_block(monkeypatch):
    _fake(monkeypatch, [{"id": "old", "name": "vy-lane-veritor-campaign", "desiredStatus": "TERMINATED"}], lambda ns: {"id": "new"})
    assert part.create_once(_ns()) == ({"id": "new"}, 0)


def test_a_pod_that_appears_despite_a_failed_create_is_the_pod(monkeypatch):
    pods: list[dict] = []

    def create(ns):
        pods.append({"id": "p2", "name": "vy-lane-veritor-campaign", "desiredStatus": "RUNNING"})
        raise runpod.PodError("POST /pods: TimeoutError: timed out")

    _fake(monkeypatch, pods, create)
    monkeypatch.setattr(runpod, "_request", lambda method, path, body=None: list(pods))
    pod, rc = part.create_once(_ns())
    assert (pod["id"], rc) == ("p2", 0)


def test_only_an_explicit_no_stock_reply_is_exit_5(monkeypatch):
    def no_stock(ns):
        raise runpod.PodError("POST /pods -> HTTP 500: There are no longer any instances available with the requested specifications")

    _fake(monkeypatch, [], no_stock)
    assert part.create_once(_ns()) == (None, part.EXIT_NO_STOCK)

    def other(ns):
        raise runpod.PodError("POST /pods -> HTTP 400: bad image")

    _fake(monkeypatch, [], other)
    assert part.create_once(_ns()) == (None, part.EXIT_FAILED)
