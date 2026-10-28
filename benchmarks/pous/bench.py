"""The POUS benchmark harness: one command per (scheme, implementation, weight set, hardware).

    python benchmarks/pous/bench.py --scheme band-chain/d12/v1 --impl reference --weight-set qwen2.5-0.5b/linears+head \\
      --hardware l40s-48gb/bf16 --out DIR [--workers N] [--max-segments N] [--audits 20]
    python benchmarks/pous/bench.py --scheme band-chain/d12/v1 --impl vllm-gpu --impl-tree TREE --impl-commit SHA \\
      --build-dir DIR --weight-set qwen2.5-0.5b/linears+head --hardware l40s-48gb/bf16 --out DIR
    python benchmarks/pous/bench.py --scheme band-chain/d12/v1 --impl band-gpu --build-dir DIR \\
      --weight-set qwen2.5-0.5b/linears+head --hardware rtx-4090/e4m3 --out DIR [--nvme-dir DIR]

Every implementation passes the same gates and reports the same metrics, written to `bench-result.json`
(`verity-pous/bench-result/v1`) and, when `RESEARCH_RUN_DIR` is set, to `$RESEARCH_RUN_DIR/result.json`:

* **Gates.**
  - The scheme's pinned conformance vectors (`verity/core/protocols/pous/tests/fixtures/schemes/<name>.json`), through
    `verity.core.protocols.pous.protocol`.
  - The implementation's own gates: known answers, and decoded blocks equal to the reference's.
  - The timed audit, and a negative control where one block in ten is wrong.
  - `certificate`: the scheme's certificate covers the audit's k and deadline.
* **Metrics.**
  - The decode pass over the weight set: seconds, and EWB, which is weight bytes over the pass time.
  - Encode time per segment.
  - The timed audit: accepted out of N, the worst answer, the limit.
  - The storage profile the certificate gives an accepted audit, with its gaps.

Implementations (`IMPLS`), each adding one class:

* `reference`: `verity.core.protocols.pous` itself, the specification, over CPU processes. W is filled deterministically, since the
  decode's cost does not depend on its bytes.
* `vllm-gpu`: the GPU twins and kernels of `verity_vllm.engine.pous`, from `--impl-tree`.
  - It runs through that tree's frozen harness, `verity-vllm pous-dense-bench`, with the flags of the band's run of
    record (`FROZEN_FLAGS`).
  - Nothing here changes that harness; its `dense-bench.json` is read back.
* `p2-v1-gpu`: P2 v4, the specification (`p2_v1/`): its reference pinned by the spec's vectors, its GPU library
  (decode with per-block keys on `p2_gpu/p2dec.cu`'s tuned squaring, encode by two roots) and the same harness steps as `band-gpu`
  at k = 105, so the two compare side by side (`side_by_side.py`). It is its own scheme (`--scheme` is ignored): the spec's P2 is
  not a `verity.core.protocols.pous` registered scheme, whose P2 is the MVP's `p2-16448/v4` (the spec's
  counter-mode SHA-512 keys, its 24-byte salt and leaves) and `p2-16448/v2` (ChaCha8 keys).
* `p2-v3-gpu`: the same kernels and steps, under the registered P2's framing (`p2-16448/v4`: its commitment, salt, key prefix and leaves,
  `p2_v1/harness.py`'s `V3`), so the device serves its blocks, and its audits go through `protocol.TimedAudit` with its
  certificate (`band_gpu/memory_challenger.py --strict`), which refuses any `k` or `Δ + RTT` the certificate does not cover.
* `p2-v3-mc-gpu`: `p2-v3-gpu` audited by the memory challenger (`memory_challenger/live.py`, `p2_v1/harness.py`'s `MC`) at
  `Δ + RTT_ALLOWANCE_CAP_NS`, with a tampered-block and a bad-signature control beside the others.  Its sessions are signed
  with S1's LMS key in `--mc-lms-key DIR`, every session's id in the session log `--mc-session-log FILE` (a file of its
  own); without a key they are test-signed, so the verdict rejects them all (its `fail_closed` gate), and it issues no
  storage profile.
* `band-gpu`: the same kernels, moved into `band_gpu/` with no vLLM dependency, the whole-chain decode on lane pairs
  (`harness.KERNELS`), run by the POUS-only harness (`band_gpu/harness.py`): kernel gates that stop the run when they fail,
  encode GB/s, `C` resident, decode overhead on a synthetic loop of the weight set's GEMMs (CUDA-graphed where capturable),
* `band-gpu`: the same kernels, moved into `band_gpu/` with no vLLM dependency, the whole-chain decode on lane pairs
  (`harness.KERNELS`), run by the POUS-only harness (`band_gpu/harness.py`): kernel gates that stop the run when they fail,
  encode GB/s, `C` resident, decode overhead on a synthetic loop of the weight set's GEMMs (CUDA-graphed where capturable),
  the limit fixed before any load, timed audits under that loop with value, late and recompute controls, the `Π₂` step
  floor (an upper bound on a whole call, not on an attacker's answer: `band_gpu/harness.py`, step 6), and optionally one
  block's read latency from NVMe.
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
"""`pous-dense-bench`'s kernel flags in the band's run of record (the implementation at `4e491631`)."""
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
    """The scheme's pinned vectors, recomputed through `verity.core.protocols.pous.protocol` (the conformance suite's own check)."""
    sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "pous" / "tests"))
    import pous_scheme_vectors as sv

    pinned = json.loads(sv.path(name).read_text())
    got = sv.vector(name, sv.pinned_blocks(pinned) if name in sv.SLOW_ENCODE else None)
    return {"passed": got == pinned, "vector": str(sv.path(name).relative_to(ROOT)), "root": got["small"]["root"]}


TAG_SEGMENTS = 448
"""Segment tags are gated over at least this many segments (the 14 GB store of
`Pous.SecurityProofs.BandMultiMeets14`)."""


def tags_gate(tags: list[bytes], segments: int) -> dict[str, Any]:
    """The implementation's segment tags equal the reference's `tag_s = salt ‖ u64 s` and are injective across all
    segments: with a shared domain, identical segments get identical labels and the audit accepts with probability 1
    (Lean `shared_domain_breaks`)."""
    from verity.core.protocols.pous.labels import salt_value, tag

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
    from verity.core.protocols.pous.protocol import StorageProfile

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
    from verity.core.protocols.pous import schemes

    name, segments, label, s, used = job
    scheme = schemes.get(name, segments=segments)
    w = _w_segment(label, s, scheme.params.segment_bytes, used)
    t = time.perf_counter()
    blocks = scheme.encode_segment(SALT, s, w)
    return s, blocks, time.perf_counter() - t


def _decode_job(job: tuple[str, int, str, int, int, list[bytes]]) -> tuple[int, bool, float]:
    from verity.core.protocols.pous import schemes

    name, segments, label, s, used, blocks = job
    scheme = schemes.get(name, segments=segments)
    t = time.perf_counter()
    w = scheme.decode_segment(SALT, s, blocks)
    dt = time.perf_counter() - t
    return s, w == _w_segment(label, s, scheme.params.segment_bytes, used), dt


class Reference:
    """`verity.core.protocols.pous` itself: encode and decode on CPU processes, `vk` and the timed audit through `verity.core.protocols.pous.protocol`."""

    name = "reference"

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path) -> dict[str, Any]:
        from verity.core.protocols.pous import protocol, schemes

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
        from verity.core.protocols.pous import dense as dense_codec

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
            v = protocol.TimedAudit(scheme, vk, protocol.challenge_key(f"bench-audit/{j}".encode(), vk), rtt_allowance_ns=rtt_ns,
                                  experimental=True)   # an experimental scheme's profile says so
            results.append(v.run(protocol.responder(scheme, block)))
        elapsed = [x.elapsed_ns for r in results for x in r.answers]
        return {"audits": audits, "accepted": sum(r.accepted for r in results), "k": v.k, "limit_ns": v.limit_ns,
                "elapsed_max_ns": max(elapsed), "elapsed_median_ns": statistics.median(elapsed)}


# ---------------------------------------------------------------------------------------------------------------- vllm-gpu

class VllmGpu:
    """The GPU twins of `verity_vllm.engine.pous`, through their tree's frozen harness `verity-vllm pous-dense-bench`."""

    name = "vllm-gpu"
    MODELS = {"qwen2.5-0.5b/linears+head": "Qwen/Qwen2.5-0.5B", "qwen2.5-7b/layer0+head": "Qwen/Qwen2.5-7B"}

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path) -> dict[str, Any]:
        from verity.core.protocols.pous import schemes

        if ws["id"] not in self.MODELS:
            raise SystemExit(f"vllm-gpu: its frozen harness runs only {sorted(self.MODELS)}, not {ws['id']!r}")
        tree = Path(a.impl_tree).resolve()
        model = self.MODELS[ws["id"]]
        impl_out = out / "impl"
        impl_out.mkdir(parents=True, exist_ok=True)
        paths = [tree / "integrations" / "vllm", tree, tree / "verity" / "core" / "protocols" / "pous",
                 tree / "verity" / "core" / "protocols" / "sampled_proofs", tree / "tools" / "research" / "src"]
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


# ---------------------------------------------------------------------------------------------------------- band-gpu

class BandGpu:
    """The band (or dense) scheme's GPU kernels (`band_gpu/`) through the POUS-only harness (`band_gpu/harness.py`)."""

    name = "band-gpu"

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path, **kw: Any) -> dict[str, Any]:
        sys.path.insert(0, str(HERE))
        from band_gpu import harness

        body = harness.run(a, ws, out, w_segment=_w_segment, tag_segments=TAG_SEGMENTS, **kw)
        scheme, extra, rows = body.pop("scheme_obj"), body.pop("extra_gaps"), body.pop("tag_rows")
        body["scheme"], body["scheme_params"] = scheme.name, scheme.params.manifest()     # the band's d among them
        if body["metrics"] is None:
            body["profile"] = None
            return body
        body["gates"]["segment_tags"] = tags_gate(rows, len(rows))
        if body.get("timed") is False:      # --audits 0: no audit, so no certificate gate and no profile
            body["profile"] = None
            return body
        audit = body["metrics"]["audit"]
        k, limit = int(audit.get("k") or 0), int(audit.get("limit_ns") or 0)
        gate, gaps = certificate_gate(scheme, k, limit)
        body["gates"]["certificate"] = gate
        body["profile"] = profile_json(scheme, k, limit, gaps + extra) if body["gates"]["timed_audit"]["passed"] else None
        return body


# ---------------------------------------------------------------------------------------------------------- p2-v1-gpu

class P2V1Gpu:
    """P2's GPU kernels (`p2_v1/`) through P2's arm of the POUS-only harness (`p2_v1/harness.py`)."""

    name = "p2-v1-gpu"
    own_scheme = "p2/v4"
    experimental = "P2 remains experimental: the P2–M1-SGI instantiation assumption is not accepted (spec §10.8)"

    def conformance_gate(self) -> dict[str, Any]:
        sys.path.insert(0, str(HERE))
        from p2_v1 import reference

        return reference.conformance()

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path, **kw: Any) -> dict[str, Any]:
        sys.path.insert(0, str(HERE))
        from p2_v1 import harness

        return harness.run(a, ws, out, w_segment=_w_segment, **kw)


class P2V3Gpu(P2V1Gpu):
    """The same kernels under the registered P2's framing (`schemes.p2.P2`, `p2-16448/v4`), audited by `protocol.TimedAudit`
    (`p2_v1/harness.py`'s `V3`)."""

    name = "p2-v3-gpu"

    @property
    def own_scheme(self) -> str:
        from verity.core.protocols.pous.schemes import p2

        return p2.P2().name

    def conformance_gate(self) -> dict[str, Any]:
        """The spec's vectors (the kernels' map and its SHA-512 streams), and the registered scheme's expander given the
        spec's prefix: its `sha512_keys` gives every vector's `K` and `t`, so the two framings differ only in their prefix."""
        sys.path.insert(0, str(HERE))
        from p2_v1 import reference

        from verity.core.protocols.pous.schemes import p2

        got = reference.conformance()
        doc = json.loads(reference.VECTORS.read_bytes())
        for v in doc["vectors"]:
            w, c0 = int(v["w"]), int(v["c0"])
            ref = reference.Params(w, c0, w // 8 - 2)
            params = p2.P2Params(w, c0, 1 << 16)
            model = bytes.fromhex(v["model_commitment"])
            prefix = reference.expand_prefix(bytes.fromhex(v["setup_salt"]), ref, model)
            k, t = p2.sha512_keys(params, prefix, int(v["segment"]), int(v["block"]))
            n = params.block_bytes
            got["checks"][f"expander/{v['id']}"] = k.to_bytes(n, "big").hex() == v["K"] and t.to_bytes(n, "big").hex() == v["t"]
        got["passed"] = all(got["checks"].values())
        return got

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path, **kw: Any) -> dict[str, Any]:
        sys.path.insert(0, str(HERE))
        from p2_v1 import harness

        return harness.run(a, ws, out, w_segment=_w_segment, framing=harness.V3, **kw)


class P2V3McGpu(P2V3Gpu):
    """The same kernels and framing, audited by the memory challenger (`p2_v1/harness.py`'s `MC`)."""

    name = "p2-v3-mc-gpu"

    def run(self, a: argparse.Namespace, ws: dict[str, Any], out: Path, **kw: Any) -> dict[str, Any]:
        sys.path.insert(0, str(HERE))
        from p2_v1 import harness

        return harness.run(a, ws, out, w_segment=_w_segment, framing=harness.MC, **kw)


IMPLS = {c.name: c for c in (Reference, VllmGpu, BandGpu, P2V1Gpu, P2V3Gpu, P2V3McGpu)}


def main(argv: list[str] | None = None, **kw: Any) -> int:
    """`kw` goes to the implementation's `run` (the P2 arms: `p2_v1.harness.run`'s `load`, from a caller hosting this harness
    in its own process)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scheme", default="band-chain/d12/v1", help="a registered name (verity.core.protocols.pous.schemes)")
    ap.add_argument("--impl", required=True, choices=sorted(IMPLS))
    ap.add_argument("--weight-set", required=True, help="an id of benchmarks/pous/weight_sets.json")
    ap.add_argument("--hardware", required=True, help="a census hardware id (benchmarks/census/hardware.json), e.g. l40s-48gb/bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audits", type=int, default=20,
                    help="timed audits per round; 0 (band-gpu, p2-v1-gpu): an untimed placement, which runs the gates, encode, "
                         "the loop and the step floor but no limit, audit or control, and is labelled timed: false")
    ap.add_argument("--rtt-allowance-ns", type=int, default=0, help="reference: the memory challenger's round-trip allowance")
    ap.add_argument("--workers", type=int, default=cpus(), help="reference: CPU processes (default: the usable CPUs)")
    ap.add_argument("--max-segments", type=int, default=0, help="reference, band-gpu, p2-v1-gpu: only the first N segments (0: all)")
    ap.add_argument("--impl-tree", help="vllm-gpu: the tree holding verity_vllm.engine.pous")
    ap.add_argument("--impl-commit", help="the implementation's source commit, recorded")
    ap.add_argument("--build-dir", default="/workspace/pous_build", help="vllm-gpu, band-gpu, p2-v1-gpu: the pous_native JIT build directory")
    ap.add_argument("--python", default=sys.executable, help="vllm-gpu: the interpreter with torch")
    ap.add_argument("--kernels", default="", help="band-gpu: kernel settings over harness.KERNELS, as JSON (harness.SWITCHES)")
    ap.add_argument("--tokens", default="1,8,32,2048", help="band-gpu, p2-v1-gpu: the synthetic loop's token counts T")
    ap.add_argument("--reps", type=int, default=20, help="band-gpu, p2-v1-gpu: timed repetitions after one warm-up; every rep is recorded")
    ap.add_argument("--reference-segments", type=int, default=2, help="band-gpu: segments of C checked against verity.core.protocols.pous")
    ap.add_argument("--pings", type=int, default=5000, help="band-gpu, p2-v1-gpu: round-trip calibration pings before and after the audits")
    ap.add_argument("--k", type=int, default=0, help="band-gpu, p2-v1-gpu: challenges per audit (0: the certificate's)")
    ap.add_argument("--allowance", default="p99.9", choices=["p50", "p99.9", "max"],
                    help="band-gpu, p2-v1-gpu: the calibration quantile added to the 0.5 ms deadline")
    ap.add_argument("--deadline-ns", type=int, default=0, help="band-gpu, p2-v1-gpu: the audit's Δ (0: the protocol's 0.5 ms)")
    ap.add_argument("--clock", default="wall", choices=["wall", "logical"],
                    help="band-gpu, p2-v1-gpu: time answers on the memory challenger's clock (wall) or by the responder's reported time (logical)")
    ap.add_argument("--verifier-cpu", type=int, default=None, help="band-gpu, p2-v1-gpu: the memory challenger process's core")
    ap.add_argument("--responder-cpu", type=int, default=None, help="band-gpu, p2-v1-gpu: the responder thread's core")
    ap.add_argument("--load-tokens", type=int, default=1, help="band-gpu, p2-v1-gpu: T of the POUS forwards run during the audit (0: none)")
    ap.add_argument("--step-reps", type=int, default=200, help="band-gpu: single-row Π₂ calls timed per kernel variant")
    ap.add_argument("--nvme-dir", default="", help="band-gpu, p2-v1-gpu: also time one block's read of C from a file here (O_DIRECT)")
    ap.add_argument("--nvme-samples", type=int, default=2000, help="band-gpu, p2-v1-gpu: block reads timed from --nvme-dir")
    ap.add_argument("--p2-lib", default=str(HERE / "p2_v1" / "bin"), help="p2-v1-gpu: the directory of libp2v1.so")
    ap.add_argument("--p2-encode-blocks", type=int, default=2048,
                    help="p2-v1-gpu: blocks encoded on the device for encode GB/s (0: all; the rest of C is random canonical)")
    ap.add_argument("--p2-c-cache", default="",
                    help="the P2 arms, with all of C encoded: keep C = Enc(W) in this directory and reuse it when the setup "
                         "matches (a timed run of a large W then skips the encode an untimed run measured)")
    ap.add_argument("--mc-lms-key", default="",
                    help="p2-v3-mc-gpu: sign the memory challenger's sessions with S1's LMS key in this directory (made "
                         "there when absent); without one they are test-signed and the verdict rejects them all")
    ap.add_argument("--bandwidth-salt", default="",
                    help="p2-v3-gpu, p2-v3-mc-gpu, joint runs only: set C up under the bandwidth audit's salt for this GPU (as "
                         "nvidia-smi's -i takes it), bandwidth.audit.draw_salt over W's commitment and the id the challenger "
                         "reads (audit.read_device), so the bandwidth verifier accepts C as the device's store; without it, "
                         "PoUS's own salt (protocol.draw_salt)")
    ap.add_argument("--mc-session-log", default="",
                    help="p2-v3-mc-gpu with --mc-lms-key: the memory challenger's used-session log, a file of its own outside "
                         "the key's directory (made with a new key; for a key that exists, a missing log is refused)")
    ap.add_argument("--band-d", type=int, default=0,
                    help="band-gpu: the band's in-degree d (0: the registered scheme's, 12); d >= 5, where the certificate "
                         "holds with its cliff at d")
    ap.add_argument("--gates-only", action="store_true",
                    help="band-gpu, p2-v1-gpu: build the kernels, run the kernel gates, and stop: nothing is timed or published")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ws = weight_set(a.weight_set)
    t = time.perf_counter()
    from verity.core.protocols.pous import schemes as _schemes

    if a.audits < 0 or (a.audits == 0 and a.impl not in ("band-gpu", "p2-v1-gpu", "p2-v3-gpu", "p2-v3-mc-gpu")):
        raise SystemExit(f"--audits {a.audits}: only band-gpu and the P2 arms run untimed (--audits 0)")
    if a.band_d:
        if a.impl != "band-gpu":
            raise SystemExit(f"--band-d is band-gpu's, not {a.impl}'s")
        sys.path.insert(0, str(HERE))
        from band_gpu.harness import with_in_degree
        try:
            with_in_degree(_schemes.get(a.scheme).params, a.band_d)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    impl = IMPLS[a.impl]()
    if getattr(impl, "own_scheme", None):
        a.scheme = impl.own_scheme
        experimental, conformance = impl.experimental, impl.conformance_gate()
    else:
        experimental, conformance = _schemes.get(a.scheme).experimental, conformance_gate(a.scheme)
    doc: dict[str, Any] = {"schema": SCHEMA, "scheme": a.scheme, "weight_set": a.weight_set, "hardware": a.hardware,
                           "experimental": experimental,
                           "source_commit": os.environ.get("RESEARCH_SOURCE_COMMIT"), "run_id": os.environ.get("RESEARCH_RUN_ID"),
                           "gates": {"conformance": conformance}}
    body = impl.run(a, ws, out, **kw)
    measurements = body.pop("measurements", [])
    doc["gates"].update(body.pop("gates"))
    doc.update(body)
    doc["wall_s"] = time.perf_counter() - t
    doc["passed"] = all(g.get("passed") for g in doc["gates"].values())
    (out / "bench-result.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    ewb = (doc.get("metrics") or {}).get("ewb")
    print(json.dumps({"scheme": doc["scheme"], "impl": a.impl, "weight_set": a.weight_set, "passed": doc["passed"],
                      **({"timed": False, "audits": doc["audits"]} if doc.get("timed") is False else {}),
                      "ewb_MBps": {t: round(v / 1e6, 1) for t, v in ewb.items()} if isinstance(ewb, dict)
                      else (round(ewb / 1e6, 3) if ewb is not None else None),
                      "gates": {g: v.get("passed") for g, v in doc["gates"].items()}}), flush=True)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        rows = []
        if doc["passed"] and doc.get("metrics"):   # a failed gate publishes no measurement, nor does a --gates-only run
            rows = ([{"name": f"ewb_T{t}", "value": v, "unit": "B/s"} for t, v in ewb.items()] if isinstance(ewb, dict)
                    else [{"name": "ewb", "value": ewb, "unit": "B/s"}])
            rows += [{"name": "decode_pass_s", "value": doc["metrics"]["decode_pass_s"], "unit": "s"}, *measurements]
        result = out.resolve() / "bench-result.json"
        rel = str(result.relative_to(Path(run_dir).resolve())) if result.is_relative_to(Path(run_dir).resolve()) else str(result)
        Path(run_dir, "result.json").write_text(json.dumps({
            "schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID"),
            "workload_fingerprint": {"benchmark": SCHEMA, "scheme": doc["scheme"], "scheme_params": doc.get("scheme_params"),
                                     "impl": a.impl, "weight_set": a.weight_set, "hardware": a.hardware,
                                     "impl_commit": a.impl_commit,
                                     **({"timed": False, "audits": doc["audits"]} if doc.get("timed") is False else {})},
            "validation": {"status": "passed" if doc["passed"] else "failed", "checker": "benchmarks/pous/bench.py gates",
                           "detail": json.dumps({g: v.get("passed") for g, v in doc["gates"].items()}),
                           "evidence": [rel]},
            "measurements": rows, "measurement_files": [rel], "artifacts": [rel]}, indent=1))
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
