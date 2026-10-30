"""The lead's words root hashed on the device: `LeadRun` over a backend that keeps each call's export on the GPU and hashes
the window's words instance there (`lead_rows.DeviceWords`, ``native_rows.cu``) records the CPU's words root of the same
calls' committed streams (`lead_rows.WindowKeying.words_root`)."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.pod
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the lead's device words need a GPU", allow_module_level=True)

from tests.protocols import test_pouw_lead as T  # noqa: E402
from verity_vllm.commit import lead_rows as L  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.protocols.pouw import lead as PL  # noqa: E402

BN = 128


@pytest.fixture(scope="module")
def ext():
    return NC.ext()


@pytest.fixture(scope="module")
def hasher(ext):
    return ext.StepHasher()


class DeviceBackend:
    """The lead's kernel, standing in on the device: the plain product as y, each call's export random s32 words of its
    ``64 ceil(m / 64)`` x ``BN ceil(n / BN)`` computed words at its S steps, kept on the GPU until the window's end, where
    `lead_rows.DeviceWords` hashes them to the instance's top. ``streams`` keeps each call's committed stream on the host
    for the CPU's recompute."""

    variant = "device-stand-in"

    def __init__(self, ext, hasher, seed: int):
        self.ext, self.hasher, self.rng = ext, hasher, np.random.default_rng(seed)
        self.exports, self.streams = [], []

    def load(self, w, layer):
        pass

    def call(self, w, layer, xq, s_a, feed):
        c = feed.call
        S = c.shape.S
        words = 64 * -(-c.m // 64) * BN * -(-c.n // BN) * S
        ex = self.rng.integers(-(2**31), 2**31, size=words, dtype=np.int64).astype("<i4")
        self.exports.append((torch.from_numpy(ex).cuda(), c.m, c.n, S))
        self.streams.append(L.stream(ex, c.m, c.n, S, BN))
        return T._plain(layer, xq, s_a)

    def words(self, dom):
        dw = L.DeviceWords(self.ext, self.hasher, dom, torch.cuda.current_stream().cuda_stream)
        for ex, m, n, S in self.exports:
            dw.add(ex, m, n, S, BN)
        self.exports = []
        return dw.top()


@pytest.mark.parametrize("shapes", [((5, 24),), ((64, 512),), ((64, 512), (3, 40)),
                                    ((80, 3072), (80, 1024), (1, 8), (80, 2048), (17, 136))])
def test_the_lead_records_the_cpus_words_root_from_the_devices_top(ext, hasher, shapes):
    """One chunk (the top is the chunk), one call of two chunks, pads after a ragged call, and a forward of ragged calls
    across lanes: the root `LeadRun` records from the device's top is the CPU's of the same streams."""
    backend = DeviceBackend(ext, hasher, len(shapes))
    run = PL.LeadRun(T.ENTRY, backend, T.Link([]), T.W, len(shapes), keying=T.Keys())
    for w, (m, n) in enumerate(shapes):
        run.call(w, T.Linear(n, w), *E.quantize_rows(T._x(m, w)))
    record = run.record
    run.end_pass()
    twin = T.Keys.twin(0, len(shapes), record.salt)
    assert record.words_root == twin.words_root(PL.FORWARD, backend.streams) == run.passes[0].transcript_root
    assert backend.exports == []
