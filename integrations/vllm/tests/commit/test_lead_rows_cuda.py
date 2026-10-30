"""``native_rows.cu``'s lead kernels (``StepHasher.lead_stream``, ``lead_leaves``) compute what ``verity_vllm.commit.lead_rows`` does,
bit for bit: the committed stream of random exports at ragged and served shapes, the inline leaves of chunks and segments at ranks
across every position width, both schemas and short last values, and whole subtrees with the rows-sha512 pads and nodes."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytestmark = pytest.mark.pod
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the lead kernels need a GPU", allow_module_level=True)

from verity_vllm.commit import lead_rows as L  # noqa: E402
from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.errors import Refused  # noqa: E402

SALT = hashlib.sha512(b"lead_rows_cuda/salt").digest()


@pytest.fixture(scope="module")
def ext():
    return NC.ext()


@pytest.fixture(scope="module")
def hasher(ext):
    return ext.StepHasher()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _dev(b: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(b), dtype=torch.uint8).cuda()


def _export(rng: np.random.Generator, m: int, n: int, S: int, bn: int) -> np.ndarray:
    words = 64 * -(-m // 64) * bn * -(-n // bn) * S
    return rng.integers(-(2**31), 2**31, size=words, dtype=np.int64).astype("<i4")


@pytest.mark.parametrize("m,n,S,bn", [(1, 128, 3, 128), (17, 136, 3, 128), (65, 72, 2, 128), (100, 300, 2, 256), (130, 128, 1, 128),
                                      (1, 3072, 74, 256), (16, 2048, 293, 128), (64, 2048, 74, 128), (200, 16384, 2, 256)])
def test_stream_is_lead_rows(hasher, m, n, S, bn):
    ex = _export(np.random.default_rng(m + n + S + bn), m, n, S, bn)
    got = hasher.lead_stream(torch.from_numpy(ex).cuda(), m, n, S, bn, _stream())
    assert got.cpu().numpy().tobytes() == L.stream(ex, m, n, S, bn)


def test_stream_refuses_an_export_of_another_shape(hasher):
    with pytest.raises(RuntimeError):
        hasher.lead_stream(torch.zeros(64 * 128 * 2 - 1, dtype=torch.int32, device="cuda"), 1, 128, 2, 128, _stream())


def _leaves(ext, hasher, dom, schema: str, values: list[bytes], first: int) -> list[bytes]:
    ls = ext.LeadStep(dom.domain_id, schema)
    size = len(values[0])
    buf = _dev(b"".join(v.ljust(size, b"\0") for v in values))
    out = hasher.lead_leaves(ls, buf, size, len(values), len(values[-1]), first, _stream()).cpu().numpy().tobytes()
    return [out[64 * i:64 * i + 64] for i in range(len(values))]


@pytest.mark.parametrize("schema", [L.WORDS_LEAF_TAG, L.A_LEAF_TAG, "s", "x" * 32])
@pytest.mark.parametrize("first", [0, 250, 65530, (1 << 24) - 3, (1 << 32) - 8])
@pytest.mark.parametrize("size,last", [(8, 8), (1024, 1024), (1024, 32), (4 * L.CHUNK_WORDS, 4 * L.CHUNK_WORDS), (4 * L.CHUNK_WORDS, 512),
                                       (1032, 1000), (136, 8)])
def test_leaves_are_core_inline_leaves(ext, hasher, schema, first, size, last):
    rng = np.random.default_rng(first + size + last + len(schema))
    values = [rng.bytes(size) for _ in range(7)] + [rng.bytes(last)]
    dom = L.instance_domain(SALT, L.words_position(3), first + len(values))
    assert _leaves(ext, hasher, dom, schema, values, first) == [dom.leaf(first + i, first + i, schema, v) for i, v in enumerate(values)]


def test_leaves_refuse_ranks_past_2_32_and_ragged_bytes(ext, hasher):
    dom = L.instance_domain(SALT, 0, 4)
    ls, buf = ext.LeadStep(dom.domain_id, L.A_LEAF_TAG), _dev(bytes(4096))
    for args in ((1024, 1, 1024, (1 << 32)), (1020, 2, 1020, 0), (1024, 4, 1028, 0), (1024, 5, 1024, 0)):
        with pytest.raises(RuntimeError):
            hasher.lead_leaves(ls, buf, *args, _stream())
    with pytest.raises(RuntimeError):
        ext.LeadStep(dom.domain_id, "x" * 33)


def _root(ext, hasher, dom, leaves: list[bytes]) -> bytes:
    rs = ext.RowStep(dom.domain_id, bytes(64), bytes(128), bytes(128), bytes(128), R.SCHEMA, 64)
    n = len(leaves)
    width = 1 << max(0, (n - 1).bit_length())
    nodes = _dev(b"".join(leaves))
    if width > n:
        pads = hasher.rows_frame(rs, torch.empty((0, 64), dtype=torch.uint8, device="cuda"), n, width - n, n, _stream())
        nodes = torch.cat([nodes.view(n, 64), pads])
    _levels, root, _ms = hasher.rows_tree(rs, nodes.view(width, 64), 0, _stream())
    return bytes(root)


@pytest.mark.parametrize("m,n,S,bn,K", [(1, 3072, 74, 256, 2048), (17, 256, 9, 128, 2048), (1, 128, 1, 128, 1024), (64, 2048, 74, 128, 8192)])
def test_subtrees_are_lead_rows(ext, hasher, m, n, S, bn, K):
    rng = np.random.default_rng(m * n + K)
    ex = _export(rng, m, n, S, bn)
    words = hasher.lead_stream(torch.from_numpy(ex).cuda(), m, n, S, bn, _stream())
    a = rng.bytes(m * K)
    for values, schema, pos in ((L.chunks(L.stream(ex, m, n, S, bn)), L.WORDS_LEAF_TAG, L.words_position(2)),
                                (L.a_segments(a), L.A_LEAF_TAG, L.a_position(2, 64, 9))):
        dom = L.instance_domain(SALT, pos, len(values))
        sub = L.subtree(dom, schema, values)
        src = words.view(torch.uint8) if schema == L.WORDS_LEAF_TAG else _dev(a)
        size = len(values[0])
        got = hasher.lead_leaves(ext.LeadStep(dom.domain_id, schema), src, size, len(values), len(values[-1]), 0, _stream())
        leaves = got.cpu().numpy().tobytes()
        assert [leaves[64 * i:64 * i + 64] for i in range(len(values))] == list(sub.levels[0][:len(values)])
        assert _root(ext, hasher, dom, list(sub.levels[0][:len(values)])) == sub.root


def _window(calls: int = 64) -> L.WindowKeying:
    w = L.WindowKeying(calls, ks_w=hashlib.sha512(b"lead_rows_cuda/ks_w").digest(), y0=bytes(L.KEY_SALT_BYTES))
    w.register(hashlib.sha512(b"lead_rows_cuda/weights").digest())
    w.open(SALT)
    return w


@pytest.mark.parametrize("shapes", [((1, 128, 1, 128),), ((1, 128, 1, 128),) * 3, ((1, 3072, 64, 256),),
                                    ((17, 256, 9, 128), (1, 3072, 74, 256), (64, 2048, 74, 128), (1, 8, 1, 128)),
                                    ((16, 2048, 256, 256), (16, 16384, 16, 256), (16, 3072, 64, 256))])
@pytest.mark.parametrize("lanes", [1, 3, 16])
def test_device_words_top_keys_the_cpus_words_root(ext, hasher, shapes, lanes):
    """`DeviceWords` over a forward's calls (one chunk; one-chunk calls with pads; whole powers of two; ragged last chunks
    across calls), on one lane, on fewer lanes than calls and on more, hands `words_top` the CPU subtree's top: check mode
    passes and d is `words_root`'s."""
    rng = np.random.default_rng(len(shapes) + sum(map(sum, shapes)))
    exports = [_export(rng, *s) for s in shapes]
    streams = [L.stream(ex, *s) for ex, s in zip(exports, shapes)]
    count = sum(L.chunk_count(m, n, S) for m, n, S, _ in shapes)
    w = _window()
    dw = L.DeviceWords(ext, hasher, w.words_domain(0, count), _stream(), lanes=lanes)
    for ex, s in zip(exports, shapes):
        dw.add(torch.from_numpy(ex).cuda(), *s)
    top = dw.top()
    assert len(top) == (len(streams[0]) if count == 1 else 128)
    assert w.words_top(0, count, top, check=streams) == _window().words_root(0, streams)


def test_device_words_lanes_wait_for_the_export_the_stream_writes(ext, hasher):
    """A lane gathers a call's export only after the work its stream holds when the call is added: each export is written on
    the stream behind a long kernel, and the top is still the CPU's."""
    shapes = ((64, 2048, 74, 128), (17, 256, 9, 128), (1, 3072, 64, 256))
    rng = np.random.default_rng(5)
    exports = [_export(rng, *s) for s in shapes]
    streams = [L.stream(ex, *s) for ex, s in zip(exports, shapes)]
    count = sum(L.chunk_count(m, n, S) for m, n, S, _ in shapes)
    staged = [torch.from_numpy(ex).cuda() for ex in exports]
    torch.cuda.synchronize()
    w = _window()
    dw = L.DeviceWords(ext, hasher, w.words_domain(0, count), _stream(), lanes=4)
    for src, s in zip(staged, shapes):
        ex = torch.zeros_like(src)
        torch.cuda._sleep(1 << 26)
        ex.copy_(src)
        dw.add(ex, *s)
    assert w.words_top(0, count, dw.top(), check=streams) == _window().words_root(0, streams)


def test_device_words_needs_a_lane(ext, hasher):
    for lanes in (0, -1, 2.0, True):
        with pytest.raises(ValueError):
            L.DeviceWords(ext, hasher, _window().words_domain(0, 4), _stream(), lanes=lanes)


def test_device_words_refuses_leaves_past_or_short_of_its_count(ext, hasher):
    shape = (17, 256, 9, 128)
    ex = torch.from_numpy(_export(np.random.default_rng(9), *shape)).cuda()
    k = L.chunk_count(*shape[:3])
    dw = L.DeviceWords(ext, hasher, _window().words_domain(0, k + 1), _stream())
    dw.add(ex, *shape)
    with pytest.raises(Refused):
        dw.top()
    with pytest.raises(Refused):
        dw.add(ex, *shape)
