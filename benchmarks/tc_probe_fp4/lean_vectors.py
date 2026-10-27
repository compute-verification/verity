"""Named NVFP4 conformance vectors from a generated Lean vector file, captured on the card (``probe.py --instruction
nvf4_lean_vectors --vectors Fp4Vectors.lean``).

The file is a ``Definitions.Pouw.PearlC.Fp4Vectors`` module: ``Vec`` entries ⟨acc, a, b, sa, sb, d⟩, one output word of
``blackwell_sm120_nvf4_m16n8k64`` (the accumulator word, A's row and B's column as 64 codes packed little-endian four bits each,
four UE4M3 bytes per side packed likewise, and the word the model predicts, ``none`` outside it), and ``ChainVec`` entries
⟨acc, n, a, b, sa, sb, d⟩ (``n`` firings, firing j on codes 64j..64j+63 and bytes 4j..4j+3, D fed back as C).
``--vector-ids`` picks entries of ``--vector-list``, ``--chain-ids`` entries of ``nvf4Chains`` and ``--special-ids`` entries of
``nvf4Specials`` (0-based; the specials lie outside the model, a NaN or infinite accumulator, and carry the record-only word);
each must carry a predicted word.  Each selected vector also runs as its low-7-bit twin (every scale byte ``& 0x7F``).

Each vector sits at row 0 / column 0 of one tile and at row 13 / column 6 of another, the other lanes holding random codes,
scale bytes without bit 7 and finite accumulators.  Every tile runs through two kernels: firing by firing through the
dedicated tile kernel with D fed back as C, and through the generic row kernel (a chain's firings kept in registers).  Every
device D buffer is filled with 0xA5 bytes before its launch.

Gates, before a selected word is recorded: the device UUID; the operands hold each vector where it was placed; both kernels agree
on every word, and both placements on every vector's word; no word reads the poison 0xA5A5A5A5; every background lane (outside the
vector's row and column) reproduces ``BLACKWELL_SM120_NVF4``; the controls, the other entries of ``--vector-list`` and
``nvf4Chains`` that carry a predicted word and no bit-7 byte, reproduce the file's words (``verity.ml.circuits.silicon`` generated them); and a
no-write control, the first tiles run again with every launch skipped, reads back nothing but the poison and fails the background
gate on every lane.  So a layout error or a kernel that wrote nothing fails the run instead of reading as a finding.  The finding:
each selected word against the file's ``d``, against its twin's device word, against the low-7-bit decode (the pinned model on the
twin; for a NaN accumulator, which the pinned model refuses, the record-only word 0x7FFFFFFF), and whether the pinned model as-is
refuses it (it reads a scale byte's low 7 bits, as sm_120 does, and refuses NaN accumulators).

A fill chunk captures (``--out``: probe_results.json and dumps.npz, every tile's operands and both kernels' words); ``--collect
DIR`` then derives every gate and word again from DIR's dumps on the CPU in a recorded run, checks them against the chunk's
record, the library (``--prebuilt``, its SASS with ``--sass-check``) and the vectors file, and publishes them.
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
from verity_config.settings import SHARED

INSTRUCTIONS = ("nvf4_lean_vectors",)
ROW = "nvf4"
K, NBLK = 64, 4
CHAIN_LIST = "nvf4Chains"
SPECIAL_LIST = "nvf4Specials"
PLACEMENTS = ((0, 0), (13, 6))
POISON = 0xA5A5A5A5
NAN_WORD = 0x7FFFFFFF  # the record-only word for a NaN accumulator
DUMP_FIELDS = ("A", "Bt", "SFA", "SFB", "C", "D_tile", "D_rows")
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


def acc_class(word: int) -> str:
    if word & 0x7F800000 != 0x7F800000:
        return "finite"
    return "NaN" if word & 0x7FFFFF else "inf"


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


def low7_decode_word(model, v: dict) -> int | None:
    """What the low-7-bit decode predicts: the pinned model on the twin, or 0x7FFFFFFF for a NaN accumulator (record-only; the
    pinned model refuses it).  None where neither applies."""
    w = model_word(model, v, True)
    if w is None and acc_class(v["acc"]) == "NaN":
        return NAN_WORD
    return w


def plan(lists: dict, selected: list, controls: list) -> list[tuple[int, list]]:
    """Every run (list, id, low7), grouped by chain length: selected vectors as themselves and as their low-7 twins, then the
    controls."""
    runs = [(name, i, False) for name, i in selected] + [(name, i, True) for name, i in selected] + \
           [(name, i, False) for name, i in controls]
    return [(n, [r for r in runs if lists[r[0]][r[1]]["n"] == n]) for n in sorted({lists[name][i]["n"] for name, i, _ in runs})]


def dump_key(n: int, place: tuple[int, int]) -> str:
    return f"n{n}_r{place[0]}_c{place[1]}"


def background(model, d: dict, place: tuple[int, int]) -> tuple[int, int]:
    """(lanes checked, lanes where the row kernel's word differs from the pinned model's or the model refuses)."""
    Dm, err = P.model_words(model, d["A"], d["Bt"], d["SFA"], d["SFB"], d["C"], K, NBLK)
    bg = np.ones((16, 8), bool)
    bg[place[0], :] = False
    bg[:, place[1]] = False
    bgm = np.broadcast_to(bg, Dm.shape)
    return int(bgm.sum()), int(((Dm != d["D_rows"]) | err)[bgm].sum())


def evaluate(lists: dict, groups: list, dumps: dict, nowrite: dict, model) -> tuple[dict, dict, list[str]]:
    """Every gate from the dumps: (gates, {(list, id, low7): the row kernel's word per placement}, failures)."""
    words: dict[tuple, list[int]] = {}
    misplaced: list[str] = []
    kernel_disagree = poison = checked = mismatched = 0
    for n, group in groups:
        for place in PLACEMENTS:
            d = dumps[dump_key(n, place)]
            r, c = place
            for t, (name, i, lo) in enumerate(group):
                v = lists[name][i]
                a, b, sa, sb = unpack(v, lo)
                if not (np.array_equal(d["A"][t, r], a) and np.array_equal(d["Bt"][t, c], b) and np.array_equal(d["SFA"][t, r], sa)
                        and np.array_equal(d["SFB"][t, c], sb) and int(d["C"][t, r, c]) == v["acc"]):
                    misplaced.append(f"{name}[{i}]{' low7' if lo else ''} at {place}")
            kernel_disagree += int((d["D_tile"] != d["D_rows"]).sum())
            poison += int((d["D_tile"] == POISON).sum() + (d["D_rows"] == POISON).sum())
            ch, mm = background(model, d, place)
            checked, mismatched = checked + ch, mismatched + mm
            for (name, i, lo), w in zip(group, d["D_rows"][:, r, c].tolist()):
                words.setdefault((name, i, lo), []).append(int(w))
    placement_disagree = sorted(f"{name}[{i}]{' low7' if lo else ''}" for (name, i, lo), ws in words.items() if len(set(ws)) != 1)
    controls = [(name, i) for n, g in groups for name, i, lo in g if not lo and (name, i, True) not in words]
    control_miss = [{"vector": f"{name}[{i}]", "d": f"{lists[name][i]['d']:08x}", "device": f"{words[(name, i, False)][0]:08x}"}
                    for name, i in controls if words[(name, i, False)][0] != lists[name][i]["d"]]
    n0, place0 = groups[0][0], PLACEMENTS[0]
    nw_words = nowrite["D_tile"].size + nowrite["D_rows"].size
    nw_poison = int((nowrite["D_tile"] == POISON).sum() + (nowrite["D_rows"] == POISON).sum())
    nw_checked, nw_mismatched = background(model, nowrite, place0)
    nw_rejected = nw_poison == nw_words and nw_checked > 0 and nw_mismatched == nw_checked
    gates = {"misplaced": misplaced, "kernels_disagree_words": kernel_disagree, "placements_disagree": placement_disagree,
             "poison": {"word": f"0x{POISON:08X}", "words_reading_poison": poison},
             "background": {"checked": checked, "mismatches": mismatched},
             "controls": {"vectors": len(controls), "chains": sum(1 for name, _ in controls if name == CHAIN_LIST),
                          "mismatches": control_miss},
             "no_write_control": {"tiles": dump_key(n0, place0), "words": nw_words, "words_reading_poison": nw_poison,
                                  "background": {"checked": nw_checked, "mismatches": nw_mismatched}, "rejected": nw_rejected}}
    failures = []
    if misplaced:
        failures.append(f"{len(misplaced)} vectors are not where they were placed (first {misplaced[:3]})")
    if kernel_disagree:
        failures.append(f"the tile kernel and the row kernel disagree on {kernel_disagree} words")
    if placement_disagree:
        failures.append(f"placements disagree: {placement_disagree}")
    if poison:
        failures.append(f"{poison} words read the poison 0x{POISON:08X} (a kernel left them unwritten)")
    if mismatched:
        failures.append(f"{mismatched} of {checked} background words differ from BLACKWELL_SM120_NVF4")
    if control_miss:
        failures.append(f"{len(control_miss)} of {len(controls)} controls differ from the file's words (first {control_miss[:3]})")
    if not nw_rejected:
        failures.append(f"the no-write control was not rejected: {nw_poison} of {nw_words} words read the poison, "
                        f"{nw_mismatched} of {nw_checked} background lanes fail")
    return gates, words, failures


def records(lists: dict, selected: list, words: dict, model) -> list[dict]:
    recs = []
    for name, i in selected:
        v = lists[name][i]
        dev_w, twin_w = words[(name, i, False)][0], words[(name, i, True)][0]
        m7, dec = model_word(model, v, True), low7_decode_word(model, v)
        cls = acc_class(v["acc"])
        recs.append({"vector": f"{name}[{i}]", "list": name, "index": i, "n": v["n"],
                     "bit7_bytes": [f"0x{x:02X}" for x in bit7_bytes(v)], "acc": f"{v['acc']:08x}",
                     "outside_model": None if cls == "finite" else f"{cls} accumulator",
                     "d_predicted": f"{v['d']:08x}", "device": f"{dev_w:08x}", "device_low7_twin": f"{twin_w:08x}",
                     "pinned_model_low7_twin": None if m7 is None else f"{m7:08x}",
                     "low7_decode_predicts": None if dec is None else f"{dec:08x}",
                     "pinned_model_as_is": "refused" if model_word(model, v, False) is None else "evaluated",
                     "matches_prediction": dev_w == v["d"], "matches_twin": dev_w == twin_w, "matches_low7_decode": dev_w == dec})
    return recs


class FakeLow7:
    """CPU stand-in (``--fake-silicon low7``): the pinned model on every scale byte's low 7 bits (the decode measured on the device),
    NaN where it refuses; ``--fake-silicon verity`` is the pinned model as-is, which reads the same
    low 7 bits.  With the launch switched off it returns the poison, as the library does.  Refused with ``RESEARCH_RUN_ID`` set."""

    def __init__(self, name: str):
        if name not in ("low7", "verity"):
            raise SystemExit("--fake-silicon: nvf4_lean_vectors takes 'low7' or 'verity'")
        self.name = name
        self.fake = P.FakeProbe(ROW, "verity")
        self.launch = True

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.name})", "compute_capability": "fake", "driver": None, "gpu_uuid": "GPU-fake",
                "backend": "fake", "has_mxf4": True}

    def clocks(self, uuid):
        return None

    def set_launch(self, on: bool) -> int:
        self.launch = on
        return POISON & 0xFF

    def rows(self, instr, A, Bt, C, SFA, SFB, steps: int = 1):
        if not self.launch:
            return np.full((A.shape[0], 16, 8), POISON, np.uint32)
        mask = 0x7F if self.name == "low7" else 0xFF
        D, err = P.model_words(P.verity_models(ROW)["verity"], A, Bt, SFA & mask, SFB & mask, C, K, NBLK)
        D[err] = 0x7FFFFFFF
        return D

    def tiles(self, instr, A, Bt, C, SFA=None, SFB=None, steps: int = 1):
        return self.rows(instr, A, Bt, C, SFA, SFB, steps)


def capture(dev, lists: dict, groups: list, seed: int) -> tuple[dict, dict]:
    """(every group's operands and words, the no-write control on the first group's first placement)."""
    rng = np.random.default_rng(seed)
    dumps = {}
    for n, group in groups:
        vs = [lists[name][i] for name, i, _ in group]
        for place in PLACEMENTS:
            A, Bt, SFA, SFB, C = build_tiles(vs, place, rng, [lo for _, _, lo in group])
            D_tile, D_rows = run_both(dev, A, Bt, SFA, SFB, C, n)
            dumps[dump_key(n, place)] = dict(zip(DUMP_FIELDS, (A, Bt, SFA, SFB, C, D_tile, D_rows)))
    n0 = groups[0][0]
    d0 = dumps[dump_key(n0, PLACEMENTS[0])]
    byte = dev.set_launch(False)
    try:
        D_tile, D_rows = run_both(dev, d0["A"], d0["Bt"], d0["SFA"], d0["SFB"], d0["C"], n0)
    finally:
        dev.set_launch(True)
    if byte * 0x01010101 != POISON:
        raise RuntimeError(f"the library's poison byte is 0x{byte:02X}, not 0x{POISON & 0xFF:02X}")
    return dumps, {**{f: d0[f] for f in DUMP_FIELDS[:5]}, "D_tile": D_tile, "D_rows": D_rows}


def save_dumps(path: Path, dumps: dict, nowrite: dict) -> None:
    flat = {f"{k}.{f}": v for k, d in dumps.items() for f, v in d.items()}
    flat.update({f"no_write.{f}": v for f, v in nowrite.items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def load_dumps(path: Path) -> tuple[dict, dict]:
    dumps: dict = {}
    with np.load(path) as z:
        for key in z.files:
            k, f = key.split(".")
            dumps.setdefault(k, {})[f] = z[key]
    nowrite = dumps.pop("no_write")
    return dumps, nowrite


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args) -> int:
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    arch = args.arch or "sm_120a"
    collect_dir = Path(os.path.expandvars(args.collect)) if getattr(args, "collect", None) else None
    if args.fake_silicon and SHARED.RESEARCH_RUN_ID:
        raise SystemExit("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    if collect_dir is not None and args.fake_silicon:
        raise SystemExit("--collect reads a capture; it takes no --fake-silicon")
    if not args.vectors:
        raise SystemExit("nvf4_lean_vectors needs --vectors (a generated Fp4Vectors.lean)")
    src = Path(os.path.expandvars(args.vectors))
    raw = src.read_bytes()
    text = raw.decode()
    ids = {args.vector_list: args.vector_ids, CHAIN_LIST: args.chain_ids, SPECIAL_LIST: getattr(args, "special_ids", None)}
    ids = {name: [int(x) for x in (s or "").split(",") if x] for name, s in ids.items()}
    lists = {name: parse_list(text, name) for name in (args.vector_list, CHAIN_LIST)}
    if ids[SPECIAL_LIST]:
        lists[SPECIAL_LIST] = parse_list(text, SPECIAL_LIST)
    results: dict = {"schema": "pouw-rtxpro/tc-probe-fp4-lean-vectors/v2", "instruction": P.PTX[ROW],
                     "instruction_id": P.INSTRUCTION_ID[ROW], "fake_silicon": args.fake_silicon,
                     "vectors_file": {"name": src.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                     "selected": {name: v for name, v in ids.items() if v or name != SPECIAL_LIST}, "seed": args.seed,
                     "expect_uuid": os.path.expandvars(args.expect_uuid) if args.expect_uuid else None}
    failures: list[str] = []

    def finish() -> int:
        results["seconds"] = round(time.time() - t_all, 1)
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        print("FAILURES" if failures else "ALL GATES PASSED", failures, f"{results['seconds']}s", flush=True)
        return 1 if failures else 0

    selected = [(name, i) for name in (args.vector_list, CHAIN_LIST, SPECIAL_LIST) for i in ids[name]]
    for name, i in selected:
        if not 0 <= i < len(lists[name]):
            failures.append(f"{name}[{i}] does not exist ({len(lists[name])} entries)")
        elif lists[name][i]["d"] is None:
            failures.append(f"{name}[{i}] carries no predicted word")
    if not selected:
        failures.append("nothing selected: pass --vector-ids, --chain-ids and/or --special-ids")
    if failures:
        return finish()
    chosen = set(selected)
    controls = [(name, i) for name in (args.vector_list, CHAIN_LIST) for i, v in enumerate(lists[name])
                if (name, i) not in chosen and v["d"] is not None and not bit7_bytes(v)]
    groups = plan(lists, selected, controls)
    model = P.verity_models(ROW)["verity"]

    chunk = None
    if collect_dir is not None:
        so = Path(os.path.expandvars(args.prebuilt)) if args.prebuilt else None
        chunk = json.loads((collect_dir / "probe_results.json").read_text())
        identity = dict(chunk.get("identity") or {})
        results["collected_from"] = {"dir": str(collect_dir), "probe_results_sha256": sha256(collect_dir / "probe_results.json"),
                                     "dumps_sha256": sha256(collect_dir / "dumps.npz"), "kept": "chunk/"}
        (out / "chunk").mkdir(exist_ok=True)
        for f in ("probe_results.json", "dumps.npz"):
            (out / "chunk" / f).write_bytes((collect_dir / f).read_bytes())
        nvcc = (chunk.get("library") or {}).get("nvcc")
        if chunk.get("failures"):
            failures.append(f"the chunk failed: {chunk['failures']}")
        if chunk.get("fake_silicon") or identity.get("backend") == "fake":
            failures.append("the chunk ran on fake silicon (a dry run's invented words)")
        for k in ("schema", "vectors_file", "selected", "seed", "instruction_id"):
            if chunk.get(k) != results[k]:
                failures.append(f"the chunk's {k} {chunk.get(k)!r} != this collect's {results[k]!r}")
        if not chunk.get("expect_uuid") or not P.uuid_allowed(identity.get("gpu_uuid"), chunk["expect_uuid"]):
            failures.append(f"the chunk's device {identity.get('gpu_uuid')} was not gated on its lease ({chunk.get('expect_uuid')})")
        if so is None:
            failures.append("--collect needs --prebuilt, the library the chunk loaded")
        elif sha256(so) != (chunk.get("library") or {}).get("sha256"):
            failures.append(f"the chunk loaded {(chunk.get('library') or {}).get('sha256')}, not {so} ({sha256(so)})")
        results["expect_uuid"] = chunk.get("expect_uuid")
    elif args.fake_silicon:
        so, nvcc, dev = None, "not built (fake silicon)", FakeLow7(args.fake_silicon)
        identity = dev.identity()
    else:
        if args.prebuilt:
            so = Path(os.path.expandvars(args.prebuilt))
            nvcc = f"prebuilt {so.name} sha256 {sha256(so)}"
        else:
            so, nvcc = P.build(arch, out, P.SOURCE[ROW])
        dev = P.Probe(so)
        identity = dev.identity()
    identity.update(nvcc=nvcc, arch=arch)
    results["identity"] = identity
    results["library"] = None if so is None else {"path": str(so), "sha256": sha256(so), "nvcc": nvcc}
    print("device", json.dumps(identity), flush=True)
    if failures:
        return finish()
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

    if chunk is not None:
        dumps, nowrite = load_dumps(collect_dir / "dumps.npz")
        for k in ("gpu_seconds", "clocks_start", "clocks_end"):
            results[k] = chunk.get(k)
    else:
        results["clocks_start"] = dev.clocks(identity.get("gpu_uuid"))
        t0 = time.time()
        try:
            dumps, nowrite = capture(dev, lists, groups, args.seed)
        except RuntimeError as e:
            failures.append(str(e))
            return finish()
        results["gpu_seconds"] = round(time.time() - t0, 2)
        results["clocks_end"] = dev.clocks(identity.get("gpu_uuid"))
        save_dumps(out / "dumps.npz", dumps, nowrite)
    missing = sorted({dump_key(n, p) for n, _ in groups for p in PLACEMENTS} - set(dumps))
    if missing:
        failures.append(f"the dumps lack {missing}")
        return finish()
    gates, words, gate_failures = evaluate(lists, groups, dumps, nowrite, model)
    results["gates"] = gates
    failures += gate_failures
    print("gates", json.dumps({k: v for k, v in gates.items() if k != "controls"}),
          f"controls {gates['controls']['vectors']}, mismatches {len(gates['controls']['mismatches'])}", flush=True)
    if failures:
        return finish()

    recs = records(lists, selected, words, model)
    for r in recs:
        print(" ", r["vector"], r["device"], "predicted", r["d_predicted"], "MATCH" if r["matches_prediction"] else "DIFFERS", flush=True)
    results["vectors"] = recs
    results["summary"] = {"selected": len(recs), "match_prediction": sum(r["matches_prediction"] for r in recs),
                          "match_low7_twin": sum(r["matches_twin"] for r in recs),
                          "match_low7_decode": sum(r["matches_low7_decode"] for r in recs),
                          "pinned_model_refuses": sum(r["pinned_model_as_is"] == "refused" for r in recs),
                          "outside_model": sum(r["outside_model"] is not None for r in recs)}
    if chunk is not None:
        for k in ("gates", "vectors", "summary"):
            if chunk.get(k) != results[k]:
                failures.append(f"the chunk's recorded {k} differ from the ones derived again from its dumps")
    print("summary", json.dumps(results["summary"]), flush=True)
    rc = finish()
    run_id = SHARED.RESEARCH_RUN_ID
    if run_id:
        s = results["summary"]
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id, "instruction": P.INSTRUCTION_ID[ROW],
            "ptx": P.PTX[ROW], "arch": arch, "sku": identity["device"], "compute_capability": identity.get("compute_capability"),
            "driver": identity.get("driver"), "gpu_uuid": identity.get("gpu_uuid"), "nvcc": nvcc,
            "cuda_visible_devices": identity.get("cuda_visible_devices"), "library": results["library"],
            "collected_from": results.get("collected_from"),
            "vectors_file": results["vectors_file"], "gates": results["gates"],
            "words": {r["vector"]: r["device"] for r in recs},
            "counts": s, "model_agrees": s["match_prediction"] == s["selected"],
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"),
                       "note": "recorded, not locked: a capture's words do not depend on the clock"},
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/lean_vectors.py", "instruction": P.INSTRUCTION_ID[ROW],
                                     "arch": arch, "device": identity["device"], "seed": args.seed,
                                     "vectors_sha256": results["vectors_file"]["sha256"], "selected": results["selected"]},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 lean_vectors.py: UUID; vectors where placed; tile and row kernels agree; two "
                                      "placements agree; no word reads the 0xA5 poison; background lanes reproduce "
                                      "BLACKWELL_SM120_NVF4; controls reproduce the file's words; the no-write control reads only "
                                      "the poison and fails the background gate"
                                      + ("; every gate and word derived again from the chunk's dumps, and equal to its record"
                                         if chunk is not None else ""),
                           "detail": "; ".join(failures) or f"{s['match_prediction']} of {s['selected']} selected words match "
                                                            "the file's prediction",
                           "evidence": ["probe_results.json"] + (["chunk/probe_results.json", "chunk/dumps.npz"]
                                                                 if chunk is not None else [])},
            "measurements": [{"name": "selected_matching_prediction", "value": s["match_prediction"], "unit": "vectors"},
                             {"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": [],
        }
        (out / "result.json.tmp").write_text(json.dumps(result, indent=1))
        os.replace(out / "result.json.tmp", out / "result.json")
    return rc
