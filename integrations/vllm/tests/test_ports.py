"""`verity_vllm.ports`: a part is the one implementation its entry point group names, and none or two refuse by name."""
from __future__ import annotations

import json
import types
from importlib.metadata import EntryPoint

import pytest

from verity_vllm import ports
from verity_vllm.errors import Refused


@pytest.fixture
def installed(monkeypatch):
    """Install ``eps`` as the group's entry points, in place of the environment's."""
    def install(*eps: EntryPoint) -> None:
        monkeypatch.setattr(ports, "entry_points", lambda group, name: [e for e in eps if e.group == group and e.name == name])
        ports._part.cache_clear()
    yield install
    ports._part.cache_clear()


def _ep(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=ports.GROUP)


def test_a_part_is_its_one_entry_point(installed):
    installed(_ep("release", "types:SimpleNamespace"), _ep("io_tie", "json"))
    assert ports.release() is types.SimpleNamespace and ports.io_tie() is json


@pytest.mark.parametrize("eps", [(), (_ep("active", "json"), _ep("active", "struct"))])
def test_none_or_two_refuse(installed, eps):
    installed(*eps)
    with pytest.raises(Refused, match=rf"{len(eps)} 'active' parts .* not one") as e:
        ports.active()
    assert e.value.what == "verity_vllm.ports.active"
