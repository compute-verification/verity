"""The lead's words root hashed on the device call by call: `LeadRun` over a backend that adds each call's export to a
`lead_rows.DeviceWords` (``native_rows.cu``) under the window's domain as it serves the call records the CPU's words root
of the same calls' committed streams (`lead_rows.WindowKeying.words_root`)."""
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
from verity_vllm.errors import Refused  # noqa: E402
from verity_vllm.protocols.pouw import lead as PL  # noqa: E402

BN = 128


@pytest.fixture(scope="module")
def ext():
    return NC.ext()


@pytest.fixture(scope="module")
def hasher(ext):
    return ext.StepHasher()


class DeviceBackend:
    """The lead's kernel, standing in on the device: the plain product as y, and each call's export random s32 words of
    its ``64 ceil(m / 64)`` x ``BN ceil(n / BN)`` computed words at its S steps, added at once to the window's
    `lead_rows.DeviceWords`, made at its first call under the domain the call is handed. ``streams`` keeps each call's
    committed stream on the host for the CPU's recompute; the backend keeps no export."""

    variant = "device-stand-in"

    def __init__(self, ext, hasher, seed: int):
        self.ext, self.hasher, self.rng = ext, hasher, np.random.default_rng(seed)
        self.words_on, self.streams = None, []

    def load(self, w, layer):
        pass

    def call(self, w, layer, xq, s_a, feed, dom):
        c = feed.call
        S = c.shape.S
        size = 64 * -(-c.m // 64) * BN * -(-c.n // BN) * S
        ex = self.rng.integers(-(2**31), 2**31, size=size, dtype=np.int64).astype("<i4")
        if self.words_on is None:
            self.words_on = L.DeviceWords(self.ext, self.hasher, dom, torch.cuda.current_stream().cuda_stream)
        self.words_on.add(torch.from_numpy(ex).cuda(), c.m, c.n, S, BN)
        self.streams.append(L.stream(ex, c.m, c.n, S, BN))
        return T._plain(layer, xq, s_a)

    def words(self):
        dw, self.words_on = self.words_on, None
        return dw.top()


@pytest.mark.parametrize("m,ns", [(5, (24,)), (64, (512,)), (64, (512, 40)), (80, (3072, 1024, 8, 2048, 136))])
def test_the_lead_records_the_cpus_words_root_from_the_devices_top(ext, hasher, m, ns):
    """One chunk (the top is the chunk), one call of two chunks, pads after a ragged call, and a forward of ragged calls
    across lanes, each export hashed as its call is served: the root `LeadRun` records from the device's top is the
    CPU's of the same streams."""
    backend = DeviceBackend(ext, hasher, m + len(ns))
    run = PL.LeadRun(T.ENTRY, backend, T.Link([]), T.W, T._linears(*ns), keying=T.Keys())
    for w, n in enumerate(ns):
        run.call(w, T.Linear(n, w), *E.quantize_rows(T._x(m, w)))
    record = run.record
    run.end_pass()
    twin = T.Keys.twin(0, len(ns), record.salt)
    assert record.words_root == twin.words_root(PL.FORWARD, backend.streams) == run.passes[0].transcript_root
    assert backend.words_on is None


def test_a_window_short_of_its_linears_calls_is_refused_by_the_device(ext, hasher):
    run = PL.LeadRun(T.ENTRY, DeviceBackend(ext, hasher, 1), T.Link([]), T.W, T._linears(512, 40), keying=T.Keys())
    run.call(0, T.Linear(512, 0), *E.quantize_rows(T._x(64, 0)))
    with pytest.raises(Refused, match="a words instance of 3 leaves got 2"):
        run.end_pass()
    assert not run.passes
