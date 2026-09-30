"""Named NVFP4 conformance vectors from a generated Lean vector file, captured on the card (``probe.py --instruction
nvf4_lean_vectors --vectors Fp4Vectors.lean``).

The file is a ``Pouw.PearlC.Fp4Vectors`` module: ``Vec`` entries ⟨acc, a, b, sa, sb, d⟩, one output word of
``blackwell_sm120_nvf4_m16n8k64`` (the accumulator word, A's row and B's column as 64 codes packed little-endian four bits each,
four UE4M3 bytes per side packed likewise, and the word the model predicts, ``none`` outside it), and ``ChainVec`` entries
⟨acc, n, a, b, sa, sb, d⟩ (``n`` firings, firing j on codes 64j..64j+63 and bytes 4j..4j+3, D fed back as C).
``--vector-ids`` picks entries of ``--vector-list`` and ``--chain-ids`` entries of ``nvf4Chains`` (0-based); each must carry a
predicted word.  Each selected vector also runs as its low-7-bit twin (every scale byte ``& 0x7F``).

Each vector sits at row 0 / column 0 of one tile and at row 13 / column 6 of another, the other lanes holding random codes,
scale bytes without bit 7 and finite accumulators.  Every tile runs through two kernels: firing by firing through the
dedicated tile kernel with D fed back as C, and through the generic row kernel (a chain's firings kept in registers).

Gates, before a selected word is recorded: the device UUID; both kernels agree on every word, and both placements on every
vector's word; every background lane (outside the vector's row and column) reproduces ``BLACKWELL_SM120_NVF4``; and the
controls, the other entries of the same two lists that carry a predicted word and no bit-7 byte, reproduce the file's words
(``verity.ml.tc`` generated them), so a layout error fails the run instead of reading as a finding.  The finding: each selected
word against the file's ``d``, against its twin's device word, and against the pinned model on the twin (the pinned model
refuses bit 7).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

import probe as P

INSTRUCTIONS = ("nvf4_lean_vectors",)
ROW = "nvf4"
K, NBLK = 64, 4
CHAIN_LIST = "nvf4Chains"
PLACEMENTS = ((0, 0), (13, 6))
_ENTRY = re.compile(r"⟨([^⟩]*)⟩")


def parse_list(text: str, name: str) -> list[dict]:
    """The entries of ``def NAME : List Vec`` (or ``ChainVec``) as dicts; ``d`` is None where the file says ``none``."""
    m = re.search(r"def %s : List (Vec|ChainVec) := \[(.*?)\]\n" % re.escape(name), text, re.S)
    if m is None:
        raise SystemExit(f"--vectors: no list {name!r}")
    chain = m.group(1) == "ChainVec"
    out = []
    for body in _ENTRY.findall(m.group(2)):
        f = [p.strip() for p in body.split(",")]
        d = None if f[-1] == "none" else int(f[-1].split()[1], 16)
        vals = [int(x, 0) for x in f[:-1]]
        acc, n, rest = (vals[0], vals[1], vals[2:]) if chain else (vals[0], 1, vals[1:])
        out.append({"acc": acc, "n": n, "a": rest[0], "b": rest[1], "sa": rest[2], "sb": rest[3], "d": d})
    return out


def unpack(v: dict, low7: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = v["n"]
    a = np.array([(v["a"] >> (4 * i)) & 15 for i in range(K * n)], np.uint8)
    b = np.array([(v["b"] >> (4 * i)) & 15 for i in range(K * n)], np.uint8)
    sa = np.array([(v["sa"] >> (8 * i)) & 255 for i in range(NBLK * n)], np.uint8)
    sb = np.array([(v["sb"] >> (8 * i)) & 255 for i in range(NBLK * n)], np.uint8)
    if low7:
        sa, sb = sa & 0x7F, sb & 0x7F
    return a, b, sa, sb


def bit7_bytes(v: dict) -> list[int]:
    _, _, sa, sb = unpack(v)
    return sorted({int(x) for x in np.concatenate([sa, sb]) if x & 0x80})


def build_tiles(vs: list[dict], place: tuple[int, int], rng, low7: list[bool]):
    """One tile per vector (all of one chain length n), the vector at ``place``."""
    n = vs[0]["n"]
    t = len(vs)
    A = rng.integers(0, 16, (t, 16, K * n)).astype(np.uint8)
    Bt = rng.integers(0, 16, (t, 8, K * n)).astype(np.uint8)
    SFA = P.UE4M3_VALID[rng.integers(0, len(P.UE4M3_VALID), (t, 16, NBLK * n))]
    SFB = P.UE4M3_VALID[rng.integers(0, len(P.UE4M3_VALID), (t, 8, NBLK * n))]
    C = P.f32_bits((rng.standard_normal((t, 16, 8)) * np.exp2(rng.integers(-8, 8, (t, 16, 8)))).astype(np.float32))
    r, c = place
    for i, v in enumerate(vs):
        A[i, r], Bt[i, c], SFA[i, r], SFB[i, c] = unpack(v, low7[i])
        C[i, r, c] = v["acc"]
    return A, Bt, SFA, SFB, C


def run_both(dev, A, Bt, SFA, SFB, C, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(firing by firing through the tile kernel, D fed back as C; the generic row kernel, the chain in registers)."""
    D = C
    for s in range(n):
        ks, bs = slice(s * K, (s + 1) * K), slice(s * NBLK, (s + 1) * NBLK)
        D = dev.tiles(ROW, np.ascontiguousarray(A[:, :, ks]), np.ascontiguousarray(Bt[:, :, ks]), D,
                      np.ascontiguousarray(SFA[:, :, bs]), np.ascontiguousarray(SFB[:, :, bs]))
    return D, dev.rows(ROW, A, Bt, C, SFA, SFB, steps=n)


def model_word(model, v: dict, low7: bool) -> int | None:
    a, b, sa, sb = unpack(v, low7)
    c = v["acc"]
    try:
        for s in range(v["n"]):
            ks, bs = slice(s * K, (s + 1) * K), slice(s * NBLK, (s + 1) * NBLK)
            c = model.step_scaled(c, a[ks].tolist(), b[ks].tolist(), sa[bs].tolist(), sb[bs].tolist())
    except Exception:  # noqa: BLE001 - outside the model's domain
        return None
    return c


class FakeLow7:
    """CPU stand-in (``--fake-silicon low7``): the pinned model on every scale byte's low 7 bits (the decode the assessor measured,
    ``r20260930-093120-b2f2``), NaN where it refuses; ``--fake-silicon verity`` refuses bit 7 as the pinned model does.  Refused
    under a research run."""

    def __init__(self, name: str):
        if name not in ("low7", "verity"):
            raise SystemExit("--fake-silicon: nvf4_lean_vectors takes 'low7' or 'verity'")
        self.name = name
        self.fake = P.FakeProbe(ROW, "verity")

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.name})", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "has_mxf4": True}

    def clocks(self, uuid):
        return None

    def rows(self, instr, A, Bt, C, SFA, SFB, steps: int = 1):
        mask = 0x7F if self.name == "low7" else 0xFF
        D, err = P.model_words(P.verity_models(ROW)["verity"], A, Bt, SFA & mask, SFB & mask, C, K, NBLK)
        D[err] = 0x7FFFFFFF
        return D

    def tiles(self, instr, A, Bt, C, SFA=None, SFB=None, steps: int = 1):
        return self.rows(instr, A, Bt, C, SFA, SFB, steps)


def main(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    arch = args.arch or "sm_120a"
    if args.fake_silicon and os.environ.get("RESEARCH_RUN_ID"):
        raise SystemExit("--fake-silicon is a dry run and never publishes: not under a research run")
    if not args.vectors:
        raise SystemExit("nvf4_lean_vectors needs --vectors (a generated Fp4Vectors.lean)")
    src = Path(os.path.expandvars(args.vectors))
    raw = src.read_bytes()
    text = raw.decode()
    vec_ids = [int(x) for x in (args.vector_ids or "").split(",") if x]
    chain_ids = [int(x) for x in (args.chain_ids or "").split(",") if x]
    lists = {args.vector_list: parse_list(text, args.vector_list), CHAIN_LIST: parse_list(text, CHAIN_LIST)}
    results: dict = {"schema": "pouw-rtxpro/tc-probe-fp4-lean-vectors/v1", "instruction": P.PTX[ROW],
                     "instruction_id": P.INSTRUCTION_ID[ROW], "fake_silicon": args.fake_silicon,
                     "vectors_file": {"name": src.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                     "selected": {args.vector_list: vec_ids, CHAIN_LIST: chain_ids}, "seed": args.seed}
    failures: list[str] = []

    def finish() -> int:
        results["seconds"] = round(time.time() - t_all, 1)
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        print("FAILURES" if failures else "ALL GATES PASSED", failures, f"{results['seconds']}s", flush=True)
        return 1 if failures else 0

    selected = [(args.vector_list, i) for i in vec_ids] + [(CHAIN_LIST, i) for i in chain_ids]
    for name, i in selected:
        if not 0 <= i < len(lists[name]):
            failures.append(f"{name}[{i}] does not exist ({len(lists[name])} entries)")
        elif lists[name][i]["d"] is None:
            failures.append(f"{name}[{i}] carries no predicted word")
    if not selected:
        failures.append("nothing selected: pass --vector-ids and/or --chain-ids")
    if failures:
        return finish()
    chosen = set(selected)
    controls = [(name, i) for name, vs in lists.items() for i, v in enumerate(vs)
                if (name, i) not in chosen and v["d"] is not None and not bit7_bytes(v)]

    if args.fake_silicon:
        so, nvcc, dev = None, "not built (fake silicon)", FakeLow7(args.fake_silicon)
    else:
        if args.prebuilt:
            so = Path(os.path.expandvars(args.prebuilt))
            nvcc = f"prebuilt {so.name} sha256 {hashlib.sha256(so.read_bytes()).hexdigest()}"
        else:
            so, nvcc = P.build(arch, out, P.SOURCE[ROW])
        dev = P.Probe(so)
    identity = dev.identity()
    identity.update(nvcc=nvcc, arch=arch)
    results["identity"] = identity
    print("device", json.dumps(identity), flush=True)
    if args.expect_uuid and not args.fake_silicon and not P.uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        failures.append(f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement")
        return finish()
    if so is not None and args.sass_check:
        sass = results["sass"] = P.sass_check(so)
        for kern, rec in sass.items():
            if rec["mma_count"] != 1:
                failures.append(f"sass: {kern} has {rec['mma_count']} MMA instructions")
        if P.KERNEL[ROW] not in sass:
            failures.append(f"sass: the {ROW} kernel {P.KERNEL[ROW]} is missing from the build")
        if failures:
            return finish()
    results["clocks_start"] = dev.clocks(identity.get("gpu_uuid"))

    # every run: (list, id, low7); selected vectors run as themselves and as their low-7 twins
    runs = [(name, i, False) for name, i in selected] + [(name, i, True) for name, i in selected] + \
           [(name, i, False) for name, i in controls]
    rng = np.random.default_rng(args.seed)
    model = P.verity_models(ROW)["verity"]
    words: dict[tuple, list[int]] = {}
    background_mismatch = background_checked = kernel_disagree = 0
    t0 = time.time()
    for n in sorted({lists[name][i]["n"] for name, i, _ in runs}):
        group = [r for r in runs if lists[r[0]][r[1]]["n"] == n]
        vs = [lists[name][i] for name, i, _ in group]
        for place in PLACEMENTS:
            A, Bt, SFA, SFB, C = build_tiles(vs, place, rng, [lo for _, _, lo in group])
            try:
                D_tile, D_rows = run_both(dev, A, Bt, SFA, SFB, C, n)
            except RuntimeError as e:
                failures.append(str(e))
                return finish()
            kernel_disagree += int((D_tile != D_rows).sum())
            Dm, err = P.model_words(model, A, Bt, SFA, SFB, C, K, NBLK)
            bg = np.ones((16, 8), bool)
            bg[place[0], :] = False
            bg[:, place[1]] = False
            bgm = np.broadcast_to(bg, Dm.shape)
            background_checked += int(bgm.sum())
            background_mismatch += int(((Dm != D_rows) | err)[bgm].sum())
            for (name, i, lo), w in zip(group, D_rows[:, place[0], place[1]].tolist()):
                words.setdefault((name, i, lo), []).append(int(w))
    results["gpu_seconds"] = round(time.time() - t0, 2)
    results["clocks_end"] = dev.clocks(identity.get("gpu_uuid"))
    placement_disagree = sorted(f"{name}[{i}]{' low7' if lo else ''}" for (name, i, lo), ws in words.items() if len(set(ws)) != 1)
    control_miss = [{"vector": f"{name}[{i}]", "d": f"{lists[name][i]['d']:08x}", "device": f"{words[(name, i, False)][0]:08x}"}
                    for name, i in controls if words[(name, i, False)][0] != lists[name][i]["d"]]
    results["gates"] = {"kernels_disagree_words": kernel_disagree, "placements_disagree": placement_disagree,
                        "background": {"checked": background_checked, "mismatches": background_mismatch},
                        "controls": {"vectors": len(controls), "chains": sum(1 for name, _ in controls if name == CHAIN_LIST),
                                     "mismatches": control_miss}}
    if kernel_disagree:
        failures.append(f"the tile kernel and the row kernel disagree on {kernel_disagree} words")
    if placement_disagree:
        failures.append(f"placements disagree: {placement_disagree}")
    if background_mismatch:
        failures.append(f"{background_mismatch} of {background_checked} background words differ from BLACKWELL_SM120_NVF4")
    if control_miss:
        failures.append(f"{len(control_miss)} of {len(controls)} controls differ from the file's words (first {control_miss[:3]})")
    print("gates", json.dumps({k: v for k, v in results["gates"].items() if k != "controls"}),
          f"controls {len(controls)}, mismatches {len(control_miss)}", flush=True)
    if failures:
        return finish()

    recs = []
    for name, i in selected:
        v = lists[name][i]
        dev_w, twin_w = words[(name, i, False)][0], words[(name, i, True)][0]
        m7 = model_word(model, v, True)
        recs.append({"vector": f"{name}[{i}]", "n": v["n"], "bit7_bytes": [f"0x{x:02X}" for x in bit7_bytes(v)],
                     "acc": f"{v['acc']:08x}", "d_predicted": f"{v['d']:08x}", "device": f"{dev_w:08x}",
                     "device_low7_twin": f"{twin_w:08x}", "pinned_model_low7_twin": None if m7 is None else f"{m7:08x}",
                     "pinned_model_as_is": "refused" if model_word(model, v, False) is None else "evaluated",
                     "matches_prediction": dev_w == v["d"], "matches_twin": dev_w == twin_w})
        print(" ", recs[-1]["vector"], recs[-1]["device"], "predicted", recs[-1]["d_predicted"],
              "MATCH" if recs[-1]["matches_prediction"] else "DIFFERS", flush=True)
    results["vectors"] = recs
    results["summary"] = {"selected": len(recs), "match_prediction": sum(r["matches_prediction"] for r in recs),
                          "match_low7_twin": sum(r["matches_twin"] for r in recs),
                          "pinned_model_refuses": sum(r["pinned_model_as_is"] == "refused" for r in recs)}
    print("summary", json.dumps(results["summary"]), flush=True)
    rc = finish()
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        s = results["summary"]
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": P.INSTRUCTION_ID[ROW],
            "ptx": P.PTX[ROW], "arch": arch, "sku": identity["device"], "compute_capability": identity.get("compute_capability"),
            "driver": identity.get("driver"), "gpu_uuid": identity.get("gpu_uuid"), "nvcc": nvcc,
            "cuda_visible_devices": identity.get("cuda_visible_devices"),
            "vectors_file": results["vectors_file"], "gates": results["gates"],
            "words": {r["vector"]: r["device"] for r in recs},
            "counts": s, "model_agrees": s["match_prediction"] == s["selected"],
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"),
                       "note": "recorded, not locked: a capture's words do not depend on the clock"},
            "workload_fingerprint": {"probe": "tools/tc_probe_fp4/lean_vectors.py", "instruction": P.INSTRUCTION_ID[ROW],
                                     "arch": arch, "device": identity["device"], "seed": args.seed,
                                     "vectors_sha256": results["vectors_file"]["sha256"], "selected": results["selected"]},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 lean_vectors.py: UUID; tile and row kernels agree; two placements agree; "
                                      "background lanes reproduce BLACKWELL_SM120_NVF4; controls reproduce the file's words",
                           "detail": "; ".join(failures) or f"{s['match_prediction']} of {s['selected']} selected words match "
                                                            "the file's prediction",
                           "evidence": ["probe_results.json"]},
            "measurements": [{"name": "selected_matching_prediction", "value": s["match_prediction"], "unit": "vectors"},
                             {"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": [],
        }
        (out / "result.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(out / "result.json.tmp", out / "result.json")
    return rc
