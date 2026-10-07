#!/usr/bin/env python
"""R9 cmt-hidden: GPU-resident commitment path (option (c)) over the M1 stream and the H1 digests.

  verity-vllm hidden-gpu check  --out J          root of the GPU chunk tree == hidden_stream.commit_stream (hashlib) on random streams;
                                        tree_levels over synthetic H1 digests == ThreadLeafCommitment; openings served from the
                                        GPU levels verify against the host verifier
  verity-vllm hidden-gpu bench  --out J          leaf kernel / tree time (CUDA events, n reps) for the accepted shapes' streams and 1024^2 /
                                        4096^2 streams, vs the host path (D2H + hashlib leaves + fold) on the same data
  verity-vllm hidden-gpu commit --npz S --out J  commit a saved M1 stream (hidden_tap.py stream) on the GPU and on the host; roots equal?
"""
import json, os, sys, time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_of_this_file():
    """[R18 infra, F-r17-tp-06] the tree THIS FILE lives in -- never $VERITY_TREE, never a default path.  The former
    `REPO = os.environ.get(<tree variable>, <the pod's default tree>)` made a Commit under --gpu-tree without the variable import verity.ml.circuits
    from the pod's DEFAULT tree while verity_vllm ran the release (tp M-1830).  This file is shipped at
    <tree>/verity_vllm/commit/committer/hidden_gpu_src/ and materialised at <tree>/out/gen/r9/cmt-hidden/src/ (pod_hidden_gpu.sh), so the tree
    is found by walking up to the first ancestor holding verity_vllm/ -- the same answer from either copy."""
    p = os.path.realpath(HERE)
    while True:
        if os.path.isfile(os.path.join(p, 'verity_vllm', '__init__.py')):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            raise ImportError(f'hidden_gpu: no source tree above {HERE} (verity_vllm/ not found in any ancestor)')
        p = parent


REPO = _repo_of_this_file()
_env_repo = os.environ.get('VERITY_TREE')
if _env_repo and os.path.realpath(_env_repo) != REPO:
    raise ImportError(f'PRECHECK_FAIL_SOURCE_IDENTITY: hidden_gpu at {HERE} belongs to tree {REPO} but $VERITY_TREE={_env_repo} names another tree')
# verity_vllm must resolve from THIS tree: if it is already imported from elsewhere, refuse (mixed tree) rather than import silently
_vc = sys.modules.get('verity_vllm')
if _vc is not None and getattr(_vc, '__file__', None) and os.path.realpath(os.path.dirname(os.path.dirname(_vc.__file__))) != REPO:
    raise ImportError(f'PRECHECK_FAIL_SOURCE_IDENTITY: hidden_gpu at {HERE} (tree {REPO}) imported beside verity_vllm from {_vc.__file__}')
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import numpy as np  # noqa: E402  -- identity is settled before anything heavy loads
import torch  # noqa: E402
from verity_vllm.commit import hidden_stream as hs  # noqa: E402
from verity_vllm.commit import scheme  # noqa: E402
from verity_vllm.config import option  # noqa: E402

_EXT = None


def ext():
    """JIT-build the GPU tree extension once per process into `<HIDDEN_GPU_BUILD or torch's default>/<source digest12>`, serialised
    with other jobs on the host (`native_jit.jit_load`)."""
    global _EXT
    if _EXT is None:
        from verity_vllm.commit.committer import native_jit

        _EXT, _bd, _digest = native_jit.jit_load('hidden_gpu_tree', [os.path.join(HERE, 'hidden_gpu_tree.cu')],
                                                 parent=os.environ.get('HIDDEN_GPU_BUILD') or None, cuda_cflags=('-O3', '--use_fast_math'))
    return _EXT


def gpu_commit(stream_cuda_i32, layout, launch_tag, src_mask, chunk_words=64, sync=True):
    e = ext()
    t = {}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    ev[0].record()
    leaves = e.chunk_leaves(stream_cuda_i32, chunk_words, launch_tag, src_mask, layout.HB, layout.M, layout.NB)
    ev[1].record()
    levels = e.tree_levels(leaves)
    ev[2].record()
    if sync:
        torch.cuda.synchronize()
        t = {'leaf_ms': ev[0].elapsed_time(ev[1]), 'tree_ms': ev[1].elapsed_time(ev[2]), 'levels': len(levels)}
    mr = levels[-1].cpu().numpy().astype(np.uint32).tobytes()
    root = scheme.stream_root(layout.digest(), launch_tag, src_mask, chunk_words, leaves.shape[0], mr)
    return root, mr, levels, t


def gpu_open(levels, index):
    """Sibling path from GPU-resident levels (one small D2H per level)."""
    path = []
    cursor = index
    for lv in levels[:-1]:
        n = lv.shape[0]
        if cursor == n - 1 and n % 2 == 1:
            path.append(None)
        else:
            sib = cursor ^ 1
            path.append(lv[sib].cpu().numpy().astype(np.uint32).tobytes())
        cursor //= 2
    return tuple(path)


def cmd_check(args):
    res = {'_device': torch.cuda.get_device_name(), '_torch': torch.__version__, 'cases': []}
    ok_all = True
    for (HB, M, NB, cw) in [(1, 64, 1, 64), (2, 64, 2, 64), (3, 128, 3, 64), (9, 192, 2, 64), (2, 64, 2, 128), (2, 64, 2, 32), (1, 64, 3, 64)]:
        L = hs.StreamLayout(HB, M, NB)
        rng = np.random.default_rng(HB * 1000 + NB)
        s = rng.integers(0, 2**32, size=L.padded_words(cw), dtype=np.uint64).astype(np.uint32)
        s[L.total_words:] = 0
        host = hs.commit_stream(s, L, launch_tag=5, src_mask=63, chunk_words=cw)
        d = torch.from_numpy(s.view(np.int32)).cuda()
        root, mr, levels, t = gpu_commit(d, L, 5, 63, cw)
        eq_root = root == host.root
        # leaves and every level identical
        eq_levels = all(lv.cpu().numpy().astype(np.uint32).tobytes() == b''.join(host.levels[i]) for i, lv in enumerate(levels))
        # openings served from the GPU levels verify with the host verifier
        eq_open = True
        for w in [0, L.total_words - 1, L.word_index('P', HB - 1, M // 2, col=L.N - 1), L.word_index('ROW', 0, 1, block=NB - 1, lane=3)]:
            ci, off = divmod(w, cw)
            chunk = s[ci * cw:(ci + 1) * cw].tobytes()
            op = hs.ChunkOpening(ci, off, chunk, gpu_open(levels, ci))
            eq_open &= hs.verify_chunk_opening(root, L, 5, 63, cw, host.leaf_count, w, op)
            bad = bytearray(chunk); bad[4 * off] ^= 1
            eq_open &= not hs.verify_chunk_opening(root, L, 5, 63, cw, host.leaf_count, w, hs.ChunkOpening(ci, off, bytes(bad), op.path))
        res['cases'].append({'HB': HB, 'M': M, 'NB': NB, 'chunk_words': cw, 'words': L.total_words, 'leaves': host.leaf_count, 'root_equal': eq_root,
                             'levels_equal': eq_levels, 'openings_ok': eq_open, 'gpu_ms': t, 'host_s': host.timing})
        ok_all &= eq_root and eq_levels and eq_open
        print(f'HB{HB} M{M} NB{NB} cw{cw}: words {L.total_words} leaves {host.leaf_count} root_equal {eq_root} levels_equal {eq_levels} open {eq_open} gpu {t}')
    # H1-style digests: tree_levels over given leaves == ThreadLeafCommitment
    L = hs.StreamLayout(2, 64, 2)
    g = hs.ThreadGeometry(HB=2, b=2, h=1, M=64, NB=2, num_m_block=1, seqlens_q=(50, 64), seqlens_k=(150, 256), causal=True)
    rng = np.random.default_rng(3)
    dig = rng.integers(0, 2**32, size=(g.leaf_count, 8), dtype=np.uint64).astype(np.uint32)
    c = hs.ThreadLeafCommitment.from_digests(dig, g, 7, 31)
    lv = ext().tree_levels(torch.from_numpy(dig.view(np.int32)).cuda())
    mr = lv[-1].cpu().numpy().astype(np.uint32).tobytes()
    eq = mr == c.merkle_root
    res['thread_digest_tree_root_equal'] = eq
    ok_all &= eq
    print('H1 digest tree root equal:', eq)
    res['_all_ok'] = ok_all
    json.dump(res, open(args.out, 'w'), indent=1)
    print('GPU-CHECK', 'ALL-OK' if ok_all else 'MISMATCH', '->', args.out)


def bench_streams():
    # (name, HB, M, NB): accepted SmolLM2 shapes (9 heads, hdim 64) and the 1024^2 / 4096^2 single-head prefill
    return [('prefill_155', 9, 155, 2), ('decode_1x169', 3, 3, 2), ('prefill_1024', 9, 1024, 8), ('prefill_4096_h1', 1, 4096, 32), ('prefill_1024_h1', 1, 1024, 8)]


def cmd_bench(args):
    e = ext()
    out = {'_device': torch.cuda.get_device_name(), '_torch': torch.__version__, '_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'streams': {}}
    for name, HB, M, NB in bench_streams():
        L = hs.StreamLayout(HB, M, NB)
        nbytes = L.padded_words() * 4
        if nbytes > args.max_bytes:
            out['streams'][name] = {'skipped': f'{nbytes} B > max'}
            continue
        d = torch.randint(-2**31, 2**31 - 1, (L.padded_words(),), dtype=torch.int32, device='cuda')
        # warmup
        gpu_commit(d, L, 1, 63)
        leaf_ms, tree_ms, total_ms = [], [], []
        for _ in range(args.reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            root, mr, levels, t = gpu_commit(d, L, 1, 63)
            total_ms.append((time.perf_counter() - t0) * 1e3)
            leaf_ms.append(t['leaf_ms']); tree_ms.append(t['tree_ms'])
        r = {'bytes': nbytes, 'leaves': L.padded_words() // 64, 'levels': len(levels),
             'gpu_leaf_ms': sorted(leaf_ms)[len(leaf_ms) // 2], 'gpu_tree_ms': sorted(tree_ms)[len(tree_ms) // 2],
             'gpu_total_wall_ms': sorted(total_ms)[len(total_ms) // 2], 'gpu_leaf_GBps': nbytes / (sorted(leaf_ms)[len(leaf_ms) // 2] * 1e-3) / 1e9,
             'retained_gpu_bytes': sum(lv.numel() * 4 for lv in levels)}
        if nbytes <= args.host_max_bytes:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            h = d.cpu().numpy().view(np.uint32)
            t1 = time.perf_counter()
            hc = hs.commit_stream(h, L, 1, 63)
            t2 = time.perf_counter()
            r.update({'host_d2h_s': t1 - t0, 'host_leaf_s': hc.timing['leaf_hash_s'], 'host_tree_s': hc.timing['tree_fold_s'], 'host_total_s': t2 - t0,
                      'host_root_equal_gpu': hc.root == root})
        out['streams'][name] = r
        print(name, json.dumps(r))
        del d, levels
        torch.cuda.empty_cache()
    json.dump(out, open(args.out, 'w'), indent=1)
    print('GPU-BENCH-OK ->', args.out)


def cmd_commit(args):
    z = np.load(args.npz)
    L = hs.StreamLayout(int(z['HB']), int(z['M']), int(z['NB']))
    s = z['stream'].astype(np.uint32)
    if s.size != L.padded_words():
        s = np.concatenate([s, np.zeros(L.padded_words() - s.size, dtype=np.uint32)])
    tag = int(z['tag'])
    d = torch.from_numpy(s.view(np.int32)).cuda()
    root, mr, levels, t = gpu_commit(d, L, tag, args.src)
    t0 = time.perf_counter(); hc = hs.commit_stream(s, L, tag, args.src); t1 = time.perf_counter()
    r = {'npz': args.npz, 'bytes': s.size * 4, 'leaves': hc.leaf_count, 'root_hex': root.hex(), 'gpu': t, 'host_s': t1 - t0, 'host_timing': hc.timing, 'roots_equal': root == hc.root}
    print(json.dumps(r)); json.dump(r, open(args.out, 'w'), indent=1)


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    out: str = option('--out', default='gpu_check.json')
    cmd: str = field(default="check", init=False)


@dataclass(frozen=True, kw_only=True)
class BenchOptions:
    out: str = option('--out', default='gpu_bench.json')
    reps: int = option('--reps', type=int, default=20)
    max_bytes: int = option('--max-bytes', type=int, default=6 * 2**30)
    host_max_bytes: int = option('--host-max-bytes', type=int, default=600 * 2**20)
    cmd: str = field(default="bench", init=False)


@dataclass(frozen=True, kw_only=True)
class CommitOptions:
    npz: str = option('--npz')
    out: str = option('--out', default='gpu_commit.json')
    src: int = option('--src', type=int, default=63)
    cmd: str = field(default="commit", init=False)


COMMANDS = {"check": CheckOptions, "bench": BenchOptions, "commit": CommitOptions}


def main(a: CheckOptions | BenchOptions | CommitOptions):
    {'check': cmd_check, 'bench': cmd_bench, 'commit': cmd_commit}[a.cmd](a)
