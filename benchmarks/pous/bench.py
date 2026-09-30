"""The POUS benchmark harness: one command per (scheme, implementation, weight set, hardware), recorded through
`research run` as an Attempt.

    python benchmarks/pous/bench.py --scheme band-chain/d12/v1 --impl reference --weight-set qwen2.5-0.5b/linears+head \\
      --hardware l40s-48gb/bf16 --out DIR [--workers N] [--max-segments N] [--audits 20]
    python benchmarks/pous/bench.py --scheme band-chain/d12/v1 --impl vllm-gpu --impl-tree TREE --impl-commit SHA \\
      --build-dir DIR --weight-set qwen2.5-0.5b/linears+head --hardware l40s-48gb/bf16 --out DIR

Every implementation passes the same gates and reports the same metrics, written to `bench-result.json`
(`verity-pous/bench-result/v1`) and, under `research run`, to `$RESEARCH_RUN_DIR/result.json`:

* **Gates.**
  - The scheme's pinned conformance vectors (`protocols/pous/tests/fixtures/schemes/<name>.json`), through
    `verity_pous.protocol`.
  - The implementation's own gates: known answers, and decoded blocks equal to the reference's.
  - The timed audit, and a negative control where one block in ten is wrong.
  - `certificate`: the scheme's certificate covers the audit's k and deadline.
* **Metrics.**
  - The decode pass over the weight set: seconds, and EWB, which is weight bytes over the pass time.
  - Encode time per segment.
  - The timed audit: accepted out of N, the worst answer, the limit.
  - The storage profile the certificate gives an accepted audit, with its gaps.

Implementations (`IMPLS`), each adding one class:

* `reference`: `verity_pous` itself, the specification, over CPU processes. W is filled deterministically, since the
  decode's cost does not depend on its bytes.
* `vllm-gpu`: the GPU twins and kernels of `verity_vllm.engine.pous` (#172, #188), from `--impl-tree`.
  - It runs through that tree's frozen harness, `verity-vllm pous-dense-bench`, with the flags of the band's run of
    record (`r20260928-032701-b4a1`, `FROZEN_FLAGS`).
  - Nothing here changes that harness; its `dense-bench.json` is read back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = "verity-pous/bench-result/v1"
WEIGHT_SETS = HERE / "weight_sets.json"
FROZEN_FLAGS = ["--reps", "3", "--thread-rows", "500", "--word-io", "--fused", "--pair", "--pair-max-rows", "7500", "--chunked",
                "--chunked-threads", "64", "--split", "--fused-x", "--whole-chain"]
"""`pous-dense-bench`'s kernel flags in the band's run of record (`r20260928-032701-b4a1`, #188 at `4e491631`)."""
SALT = bytes(range(24))
BATCHES = ("1", "8", "32")


def cpus() -> int:
    """The CPUs this process may use: its affinity, capped by a cgroup CPU quota (v2 `cpu.max`, v1 `cpu.cfs_quota_us`)."""
    n = len(os.sched_getaffinity(0))
    for path, parse in (("/sys/fs/cgroup/cpu.max", lambda t: t.split()),
                        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", lambda t: [t.strip(), Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text()])):
        try:
            quota, period = parse(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if quota not in ("max", "-1"):
            n = min(n, max(1, int(quota) // int(period)))
    return n


def weight_set(ws_id: str) -> dict[str, Any]:
    doc = json.loads(WEIGHT_SETS.read_text())
    for ws in doc["weight_sets"]:
        if ws["id"] == ws_id:
            return ws
    raise SystemExit(f"no weight set {ws_id!r} in {WEIGHT_SETS}: {[w['id'] for w in doc['weight_sets']]}")


def conformance_gate(name: str) -> dict[str, Any]:
    """The scheme's pinned vectors, recomputed through `verity_pous.protocol` (the conformance suite's own check)."""
    sys.path.insert(0, str(ROOT / "protocols" / "pous" / "tests"))
    import pous_scheme_vectors as sv

    pinned = json.loads(sv.path(name).read_text())
    got = sv.vector(name, sv.pinned_blocks(pinned) if name in sv.SLOW_ENCODE else None)
    return {"passed": got == pinned, "vector": str(sv.path(name).relative_to(ROOT)), "root": got["small"]["root"]}


TAG_SEGMENTS = 448
"""Segment tags are gated over at least this many segments (`band_meets_14`'s 14 GB store)."""


def tags_gate(tags: list[bytes], segments: int) -> dict[str, Any]:
    """The implementation's segment tags equal the reference's `tag_s = salt ‖ u64 s` and are injective across all
    segments: with a shared domain, identical segments get identical labels and the audit accepts with probability 1
    (Lean `shared_domain_breaks`)."""
    from verity_pous.labels import salt_value, tag

    ref = [tag(salt_value(SALT), s).to_bytes(32, "little") for s in range(segments)]
    injective, equal = len(set(tags)) == segments, tags == ref
    return {"passed": injective and equal, "segments": segments, "injective": injective, "equal_to_reference": equal}


def certificate_gate(scheme, k: int, limit_ns: int) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Whether the certificate covers an audit of `k` challenges within `limit_ns`, and the profile's gaps."""
    cert = scheme.certificate
    if cert is None:
        return {"passed": False, "reason": f"{scheme.name} has no certificate at {scheme.params.manifest()}"}, ("no certificate",)
    refused = cert.refusals(k, limit_ns)
    gaps = refused
    if cert.segments is not None and scheme.params.segments != cert.segments:
        gaps += (f"W has {scheme.params.segments} segments; {cert.theorem} is about {cert.segments}",)
    return {"passed": not refused, "refusals": list(refused), "effective_rounds": cert.effective_rounds(limit_ns)}, gaps


def profile_json(scheme, k: int, limit_ns: int, gaps: tuple[str, ...]) -> dict[str, Any]:
    from verity_pous.protocol import StorageProfile

    p = scheme.params
    return StorageProfile(scheme.name, p.manifest(), p.blocks, p.m, k, limit_ns, scheme.certificate, gaps,
                          scheme.experimental).to_json()


# ---------------------------------------------------------------------------------------------------------- reference (CPU)

def _w_segment(label: str, s: int, size: int, used: int) -> bytes:
    """Segment `s` of the deterministic W: SHAKE-256 of the label, zero past the weight set's `used` bytes."""
    start = s * size
    live = max(0, min(size, used - start))
    return hashlib.shake_256(f"verity-pous/bench/{label}/{s}".encode()).digest(live) + bytes(size - live)


def _encode_job(job: tuple[str, int, str, int, int]) -> tuple[int, list[bytes], float]:
    from verity_pous import schemes

    name, segments, label, s, used = job
    scheme = schemes.get(name, segments=segments)
    w = _w_segment(label, s, scheme.params.segment_bytes, used)
    t = time.perf_counter()
    blocks = scheme.encode_segment(SALT, s, w)
    return s, blocks, time.perf_counter() - t


def _decode_job(job: tuple[str, int, str, int, int, list[bytes]]) -> tuple[int, bool, float]:
    from verity_pous import schemes

    name, segments, label, s, used, blocks = job
    scheme = schemes.get(name, segments=segments)
    t = time.perf_counter()
    w = scheme.decode_segment(SALT, s, blocks)
    dt = time.perf_counter() - t
    return s, w == _w_segment(label, s, scheme.params.segment_bytes, used), dt


class Reference:
    """`verity_pous` itself: encode and decode on CPU processes, `vk` and the timed audit through `verity_pous.protocol`."""

    name = "reference"

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path) -> dict[str, Any]:
        from verity_pous import protocol, schemes

        base = schemes.get(a.scheme)
        segments = -(-ws["bytes"] // base.params.segment_bytes)
        n = min(segments, a.max_segments) if a.max_segments else segments
        scheme = schemes.get(a.scheme, segments=n)
        used = min(ws["bytes"], n * scheme.params.segment_bytes)
        label = ws["id"]
        with ProcessPoolExecutor(a.workers) as pool:
            t = time.perf_counter()
            enc = sorted(pool.map(_encode_job, [(a.scheme, n, label, s, used) for s in range(n)]))
            encode_wall = time.perf_counter() - t
            c = [blocks for _, blocks, _ in enc]
            t = time.perf_counter()
            dec = sorted(pool.map(_decode_job, [(a.scheme, n, label, s, used, c[s]) for s in range(n)]))
            decode_wall = time.perf_counter() - t
        w_segments = [_w_segment(label, s, scheme.params.segment_bytes, used) for s in range(n)]
        commitment = protocol.commit(scheme, w_segments)
        t = time.perf_counter()
        vk = protocol.verifier_key(scheme, commitment, SALT, w=w_segments, encoded=c)
        vk_s = time.perf_counter() - t
        blocks = [b for seg in c for b in seg]
        honest = self._audits(protocol, scheme, vk, blocks.__getitem__, a.audits, a.rtt_allowance_ns)
        wrong = [bytes(len(b)) if i % 10 == 0 else b for i, b in enumerate(blocks)]
        negative = self._audits(protocol, scheme, vk, wrong.__getitem__, a.audits, a.rtt_allowance_ns)
        k, limit = honest["k"], honest["limit_ns"]
        gate, gaps = certificate_gate(scheme, k, limit)
        from verity_pous import dense as dense_codec

        nt = max(n, TAG_SEGMENTS)
        wide = scheme.params.with_segments(nt)
        tags = tags_gate([dense_codec._check(wide, SALT, s).to_bytes(32, "little") for s in range(nt)], nt)   # encode_segment's own tag
        return {
            "implementation": {"name": self.name, "commit": a.impl_commit, "workers": a.workers},
            "segments": n, "segments_of_weight_set": segments, "blocks": scheme.params.blocks, "weight_bytes": used,
            "gates": {"round_trip": {"passed": all(ok for _, ok, _ in dec)}, "certificate": gate, "segment_tags": tags,
                      "timed_audit": {"passed": honest["accepted"] == honest["audits"]},
                      "negative_control": {"passed": negative["accepted"] == 0, "accepted": negative["accepted"]}},
            "metrics": {"decode_pass_s": decode_wall, "ewb": used / decode_wall,
                        "decode_s_per_segment": statistics.median(dt for _, _, dt in dec),
                        "encode_pass_s": encode_wall, "encode_s_per_segment": statistics.median(dt for _, _, dt in enc),
                        "vk_s": vk_s, "audit": honest, "negative_audit": negative},
            "profile": profile_json(scheme, k, limit, gaps) if honest["accepted"] else None,
        }

    @staticmethod
    def _audits(protocol, scheme, vk, block, audits: int, rtt_ns: int) -> dict[str, Any]:
        results = []
        for j in range(audits):
            v = protocol.Verifier(scheme, vk, protocol.challenge_key(f"bench-audit/{j}".encode(), vk), rtt_allowance_ns=rtt_ns,
                                  experimental=True)   # an experimental scheme's profile says so
            results.append(v.run(protocol.responder(scheme, block)))
        elapsed = [x.elapsed_ns for r in results for x in r.answers]
        return {"audits": audits, "accepted": sum(r.accepted for r in results), "k": v.k, "limit_ns": v.limit_ns,
                "elapsed_max_ns": max(elapsed), "elapsed_median_ns": statistics.median(elapsed)}


# ---------------------------------------------------------------------------------------------------------- vllm-gpu (#188)

class VllmGpu:
    """The GPU twins of `verity_vllm.engine.pous`, through their tree's frozen harness `verity-vllm pous-dense-bench`."""

    name = "vllm-gpu"
    MODELS = {"qwen2.5-0.5b/linears+head": "Qwen/Qwen2.5-0.5B", "qwen2.5-7b/layer0+head": "Qwen/Qwen2.5-7B"}

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path) -> dict[str, Any]:
        from verity_pous import schemes

        tree = Path(a.impl_tree).resolve()
        model = self.MODELS[ws["id"]]
        impl_out = out / "impl"
        impl_out.mkdir(parents=True, exist_ok=True)
        paths = [tree / "integrations" / "vllm", tree / "packages" / "verity" / "src", tree / "protocols" / "pous",
                 tree / "protocols" / "sampled_proofs", tree / "tools" / "research" / "src"]
        env = {**os.environ, "PYTHONPATH": ":".join(map(str, paths)), "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
        cmd = [a.python, "-m", "verity_vllm.pipeline.cli", "pous-dense-bench", "--codec", a.scheme, "--out", str(impl_out),
               "--shapes", str(tree / "integrations/vllm/workloads/dense_decode_shapes.json"), "--build-dir", a.build_dir,
               "--models", model, *FROZEN_FLAGS, "--segment-vector", "--audit", str(a.audits)]
        t = time.perf_counter()
        with (out / "impl.log").open("w") as log:
            rc = subprocess.run(cmd, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - t
        doc = json.loads((impl_out / "dense-bench.json").read_text()) if (impl_out / "dense-bench.json").exists() else None
        if doc is None:
            raise SystemExit(f"pous-dense-bench exited {rc} without dense-bench.json (see {out / 'impl.log'})")
        m = doc["models"][model]
        audit = m.get("audit") or {}
        scheme = schemes.get(a.scheme, segments=m["segments"])
        k, limit = int(audit.get("k", scheme.certificate.k if scheme.certificate else 0)), int(audit.get("limit_ns", 0))
        gate, gaps = certificate_gate(scheme, k, limit)
        gates = {g: {"passed": bool(v.get("passed"))} for g, v in doc["gates"].items()}
        gates["certificate"] = gate
        n = max(m["segments"], TAG_SEGMENTS)
        probe = ("import sys; from verity_vllm.engine.pous import codec; "
                 "t = codec.segment_tags(bytes.fromhex(sys.argv[1]), list(range(int(sys.argv[2]))), 'cpu'); "
                 "sys.stdout.write(bytes(t.flatten().tolist()).hex())")
        raw = bytes.fromhex(subprocess.run([a.python, "-c", probe, SALT.hex(), str(n)], cwd=tree, env=env, check=True,
                                           capture_output=True, text=True).stdout)
        gates["segment_tags"] = tags_gate([raw[i * 32:(i + 1) * 32] for i in range(n)], n)
        accepted = bool(audit.get("accepted"))
        return {
            "implementation": {"name": self.name, "commit": a.impl_commit, "harness": "verity-vllm pous-dense-bench",
                               "flags": FROZEN_FLAGS, "rc": rc, "wall_s": wall},
            "device": doc.get("device"), "segments": m["segments"], "blocks": m["blocks"], "weight_bytes": m["weight_bytes"],
            "gates": gates,
            "metrics": {"decode_pass_s": m["decode_s"], "ewb": m["ewb"], "plain_ewb": m["plain_ewb"],
                        "keccak_per_s": m["keccak_per_s"], "integer_share": m.get("integer_share"), "latency": m["latency"],
                        "decode_reps_s": m["decode_reps_s"],
                        "audit": {key: audit.get(key) for key in ("audits", "audits_accepted", "k", "limit_ns", "audit_of_record",
                                                                   "elapsed_s", "device_s", "late", "wrong", "cgroup_cpu")}},
            "profile": profile_json(scheme, k, limit, gaps) if accepted else None,
        }


# ---------------------------------------------------------------------------------------------------------- p2-gpu

class P2Gpu:
    """P2's GPU decode (`benchmarks/pous/p2_gpu/`, the efficient-cryptography campaign's tuned CGBN kernel, bc-28faaf0e).

    Two builds run on the same C:
    - `-DP2_DERIVE`, the scheme's own key schedule (M1: fresh per-block `K_j, t_j`), whose rate is the result. The keys
      are expanded in the timed kernel, by ChaCha8 from one 32-byte seed per segment, so only the seeds cross from the
      host and nothing key-sized is stored (the 1.05× space bound).
    - The campaign's shared-key build (M2, not covered by `P2MeetsM1Uncond`), for comparison only.

    C is random blocks below p, with four blocks encoded by the reference planted in it. The kernel's sampled outputs
    (the first 1,024 blocks, 1,024 pseudo-random ones, the last, and the planted ones) are checked against the
    reference's decode.  The timed audit and its negative control run on the same C through `verity_pous.protocol`."""

    name = "p2-gpu"

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path) -> dict[str, Any]:
        import numpy as np

        from verity_pous import protocol, schemes
        from verity_pous.labels import salt_value, tag
        from verity_pous.schemes import p2
        from verity_pous.verifier import VerifierKey, leaf_hash

        base = schemes.get(a.scheme)
        if not isinstance(base, schemes.P2):
            raise SystemExit(f"p2-gpu runs a P2 scheme, not {a.scheme}")
        pb, nb, bps = base.params.payload_bytes, base.params.block_bytes, base.params.blocks_per_segment
        n = -(-ws["bytes"] // pb)
        if a.max_segments:
            n = min(n, a.max_segments * bps)
        scheme = schemes.get(a.scheme, segments=-(-n // bps))
        params, f, nl = scheme.params, p2._field(scheme.params), nb // 4
        work = Path(a.scratch) / "p2"
        work.mkdir(parents=True, exist_ok=True)
        bins = Path(a.p2_bin)

        rng = np.random.default_rng(20260928)
        c = rng.integers(0, 1 << 32, size=(params.blocks, nl), dtype=np.uint32)
        c[:, -1] = np.minimum(c[:, -1], 0xFFFFFFFE)                   # every block below p = 2^w − c0
        planted = {}
        t0 = time.perf_counter()
        for g in sorted({1, n // 3, (2 * n) // 3, n - 2}):
            s, j = divmod(g, bps)
            m = int.from_bytes(hashlib.shake_256(f"verity-pous/bench/p2/{g}".encode()).digest(pb), "little")
            k, tj = p2.block_keys(params, SALT, s, j)
            c[g] = np.frombuffer(f.rho(f.sigma(f.rho((m + tj) % f.p), k)).to_bytes(nb, "little"), dtype=np.uint32)
            planted[g] = m
        encode_s = (time.perf_counter() - t0) / len(planted)
        (work / "c.bin").write_bytes(c.tobytes())
        seeds = [p2.segment_seed(SALT, s) for s in range(params.segments)]
        (work / "seeds.bin").write_bytes(params.segments.to_bytes(4, "little") + bps.to_bytes(4, "little") + b"".join(seeds))
        shared = [int.from_bytes(hashlib.shake_256(f"verity-pous/bench/p2/shared/{x}".encode()).digest(nb + 8), "little")
                  for x in ("K", "t0", "step")]
        shared = [shared[0] % (1 << params.w), shared[1] % f.p, shared[2] % f.p]
        (work / "params.bin").write_bytes(params.w.to_bytes(4, "little") + params.c0.to_bytes(4, "little")
                                          + b"".join(x.to_bytes(nb, "little") for x in shared))
        (work / "specials.bin").write_bytes(len(planted).to_bytes(4, "little") + b"".join(
            g.to_bytes(8, "little") + c[g].tobytes() for g in planted))
        (work / "nospec.bin").write_bytes(bytes(4))

        def kernel(binary: str, specials: str, sample: Path, extra: list[str]) -> dict[str, Any]:
            cmd = [str(bins / binary), str(work / "params.bin"), str(work / specials), str(sample), str(n * nb), "7",
                   str(work / "c.bin"), *extra]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (out / f"{binary}.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode != 0:
                raise SystemExit(f"{binary} exited {proc.returncode}: {proc.stderr[-400:]}")
            return json.loads(proc.stdout.strip().splitlines()[-1])

        m1 = kernel("p2dec_16448_32_s2_derive", "specials.bin", out / "sample-m1.bin", [str(work / "seeds.bin")])
        m2 = kernel("p2dec_16448_32_s2", "nospec.bin", out / "sample-m2.bin", [])

        def check(sample: Path, decode, expect: dict[int, int]) -> dict[str, Any]:
            data = sample.read_bytes()
            count = int.from_bytes(data[4:8], "little")
            seen, bad, off = set(), 0, 8
            for _ in range(count):
                g = int.from_bytes(data[off:off + 8], "little")
                cv = int.from_bytes(data[off + 8:off + 8 + nb], "little")
                mv = int.from_bytes(data[off + 8 + nb:off + 8 + 2 * nb], "little")
                off += 8 + 2 * nb
                if g in seen:
                    continue
                seen.add(g)
                bad += mv != decode(g, cv) or (g in expect and mv != expect[g])
            return {"passed": bad == 0 and bool(seen) and all(g in seen for g in expect), "checked": len(seen),
                    "mismatches": bad, "planted": sum(g in seen for g in expect), "planted_of": len(expect)}

        def m1_decode(g: int, cv: int) -> int:
            k, tj = p2.expand_keys(params, seeds[g // bps], g % bps)
            return (f.sq(f.sigma(f.sq(cv), k)) - tj) % f.p

        def m2_decode(g: int, cv: int) -> int:
            return (f.sq(f.sigma(f.sq(cv), shared[0])) - (shared[1] + g * shared[2])) % f.p

        blocks = [c[g].tobytes()[::-1] for g in range(params.blocks)]    # the device holds little-endian limbs; the wire is big-endian
        vk = VerifierKey.from_leaves(params, SALT, [leaf_hash(params, SALT, g, b) for g, b in enumerate(blocks)])
        honest = Reference._audits(protocol, scheme, vk, blocks.__getitem__, a.audits, a.rtt_allowance_ns)
        wrong = [bytes(nb) if g % 10 == 0 else b for g, b in enumerate(blocks)]
        negative = Reference._audits(protocol, scheme, vk, wrong.__getitem__, a.audits, a.rtt_allowance_ns)
        gate, gaps = certificate_gate(scheme, honest["k"], honest["limit_ns"])
        nt = max(params.segments, TAG_SEGMENTS)
        pass_s = m1["ms_med"] / 1e3
        return {
            "implementation": {"name": self.name, "commit": a.impl_commit, "kernel": "benchmarks/pous/p2_gpu s2, TPI 32, 64-thread blocks",
                               "keys": "per block, expanded on the device by ChaCha8 from a 32-byte seed per segment"},
            "device": {"gpu": m1.get("gpu"), "sms": m1.get("sms"), "regs": m1.get("regs")},
            "segments": params.segments, "blocks": n, "weight_bytes": n * pb,
            "gates": {"reference_decode": check(out / "sample-m1.bin", m1_decode, planted),
                      "m2_reference_decode": check(out / "sample-m2.bin", m2_decode, {}),
                      "certificate": gate, "segment_tags": tags_gate([tag(salt_value(SALT), s).to_bytes(32, "little")
                                                                       for s in range(nt)], nt),
                      "timed_audit": {"passed": honest["accepted"] == honest["audits"]},
                      "negative_control": {"passed": negative["accepted"] == 0, "accepted": negative["accepted"]}},
            "metrics": {"decode_pass_s": pass_s, "ewb": n * pb / pass_s, "stored_GBps": m1["GBps_stored_med"],
                        "m2_shared_key": {"decode_pass_s": m2["ms_med"] / 1e3, "ewb": n * pb / (m2["ms_med"] / 1e3),
                                          "stored_GBps": m2["GBps_stored_med"]},
                        "h2d_GBps": m1.get("h2d_GBps"), "reference_encode_s_per_block": encode_s,
                        "audit": honest, "negative_audit": negative, "kernel": {"m1": m1, "m2": m2}},
            "profile": profile_json(scheme, honest["k"], honest["limit_ns"], gaps) if honest["accepted"] else None,
        }


IMPLS = {c.name: c for c in (Reference, VllmGpu, P2Gpu)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scheme", default="band-chain/d12/v1", help="a registered name (verity_pous.schemes)")
    ap.add_argument("--impl", required=True, choices=sorted(IMPLS))
    ap.add_argument("--weight-set", required=True, help="an id of benchmarks/pous/weight_sets.json")
    ap.add_argument("--hardware", required=True, help="a census hardware id (census/hardware.json), e.g. l40s-48gb/bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audits", type=int, default=20)
    ap.add_argument("--rtt-allowance-ns", type=int, default=0, help="reference: the verifier's round-trip allowance")
    ap.add_argument("--workers", type=int, default=cpus(), help="reference: CPU processes (default: the usable CPUs)")
    ap.add_argument("--max-segments", type=int, default=0, help="reference, p2-gpu: only the first N segments (0: all)")
    ap.add_argument("--impl-tree", help="vllm-gpu: the tree holding verity_vllm.engine.pous (#188)")
    ap.add_argument("--impl-commit", help="the implementation's source commit, recorded")
    ap.add_argument("--build-dir", default="/workspace/pous_build", help="vllm-gpu: the pous_native JIT build directory")
    ap.add_argument("--python", default=sys.executable, help="vllm-gpu: the interpreter with torch")
    ap.add_argument("--p2-bin", default=str(HERE / "p2_gpu" / "bin"), help="p2-gpu: the directory of the built kernels")
    ap.add_argument("--scratch", default="/tmp/pous-bench", help="large inputs (C, keys), kept out of the run directory")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ws = weight_set(a.weight_set)
    t = time.perf_counter()
    from verity_pous import schemes as _schemes

    doc: dict[str, Any] = {"schema": SCHEMA, "scheme": a.scheme, "weight_set": a.weight_set, "hardware": a.hardware,
                           "experimental": _schemes.get(a.scheme).experimental,
                           "source_commit": os.environ.get("RESEARCH_SOURCE_COMMIT"), "run_id": os.environ.get("RESEARCH_RUN_ID"),
                           "gates": {"conformance": conformance_gate(a.scheme)}}
    body = IMPLS[a.impl]().run(a, ws, out)
    doc["gates"].update(body.pop("gates"))
    doc.update(body)
    doc["wall_s"] = time.perf_counter() - t
    doc["passed"] = all(g.get("passed") for g in doc["gates"].values())
    (out / "bench-result.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    ewb = doc["metrics"]["ewb"]
    print(json.dumps({"scheme": a.scheme, "impl": a.impl, "weight_set": a.weight_set, "passed": doc["passed"],
                      "ewb_MBps": {t: round(v / 1e6, 1) for t, v in ewb.items()} if isinstance(ewb, dict) else round(ewb / 1e6, 3),
                      "gates": {g: v.get("passed") for g, v in doc["gates"].items()}}), flush=True)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        rows = ([{"name": f"ewb_T{t}", "value": v, "unit": "B/s"} for t, v in ewb.items()] if isinstance(ewb, dict)
                else [{"name": "ewb", "value": ewb, "unit": "B/s"}])
        rows += [{"name": "decode_pass_s", "value": doc["metrics"]["decode_pass_s"], "unit": "s"}]
        result = out.resolve() / "bench-result.json"
        rel = str(result.relative_to(Path(run_dir).resolve())) if result.is_relative_to(Path(run_dir).resolve()) else str(result)
        Path(run_dir, "result.json").write_text(json.dumps({
            "schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID"),
            "workload_fingerprint": {"benchmark": SCHEMA, "scheme": a.scheme, "impl": a.impl, "weight_set": a.weight_set,
                                     "hardware": a.hardware, "impl_commit": a.impl_commit},
            "validation": {"status": "passed" if doc["passed"] else "failed", "checker": "benchmarks/pous/bench.py gates",
                           "detail": json.dumps({g: v.get("passed") for g, v in doc["gates"].items()}),
                           "evidence": [rel]},
            "measurements": rows, "measurement_files": [rel], "artifacts": [rel]}, indent=1))
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
