"""The constant text a zk point writes into its records, pinned by name against main's bytes.

A word change alters no recorded text: each field's source, the stand-in's statement, the passing validation's detail
keep their bytes. A fault, refusal or help text is no site.
"""
from __future__ import annotations

import hashlib
import inspect
import re

import pytest

from verity_proof_throughput import zk

#: name -> the text, as this tree writes it
SITES = {
    **{f"FIELDS.{f[0]}.source": (lambda f=f: f[3]) for f in zk.FIELDS},
    "STANDIN": lambda: zk.STANDIN,
    "point.detail.zk_server": lambda: re.findall(r'\+ \(", ([^"]+)" if [a-z_]+ else ""\)', inspect.getsource(zk.point))[0],
}

#: sha256 of each site's text as main's records carry it
PINS = {
    "FIELDS.net.bytes_worker_to_server.source": "d39eae40cab4355c28e3f5cce3ba5133975a302d360da6b416f65ddc0c78a96e",
    "FIELDS.proof_bytes.released.source": "52c9319b70ab623fa89ee021c760bb78521dd54fa96c82732606fa0d19b026bf",
    "FIELDS.worker.vu_per_second.source": "37d1dfa2b61ebc3ad74fd5a502df2e0d86ac508f85c6e85525cc87d3bb80c1a3",
    "FIELDS.zk_server.cores_per_gpu.source": "50e4330846f05fd0ac7e992cdd594f1a491b1de5d2050d5ae78f0dfe78d882a1",
    "FIELDS.zk_server.cpu_s_per_session.source": "05d4ba742fb043db6ec4c9de2f220c9cf9ff7ed31522ebb5240dd7dec12f7fc2",
    "FIELDS.zk_server.cpu_s_per_vu.source": "cafe9b06906677bcd38e6b786ab6eb65bae1a1ff127ca20a99c8f783de1ff4b2",
    "STANDIN": "9c478db7c4809bc13b15c8e15ded1da378fa3e65c7b0c973d6724cb7f26f2b3f",
    "point.detail.zk_server": "10f0538e2010a093635a6c40ebfee823b4fded01141266e9d50cabfbbf818084",
}


def test_every_site_is_pinned():
    assert sorted(SITES) == sorted(PINS), "a site and its pin are added together, the pin from main's text"


@pytest.mark.parametrize("name", sorted(PINS))
def test_a_recorded_text_keeps_its_bytes(name):
    assert hashlib.sha256(SITES[name]().encode()).hexdigest() == PINS[name], (
        f"{name}: the text zk.py writes into its records changed. A word change alters no recorded text: put main's bytes "
        f"back (`git show origin/main:benchmarks/proof_throughput/verity_proof_throughput/zk.py`). Only a new record format "
        f"re-pins, with its schema id bumped.")
