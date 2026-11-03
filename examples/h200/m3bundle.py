"""M3 proper's sampled proof blocks: after the live coin, each sampled proof block of the registered H200 step as the
tree ``prove_bundle.py --step`` proves: ``blocks/<g>/window/`` (a one-member window of the block, registered
``subset:1``) and ``blocks/<g>/bundle.{json,bin}`` (`replay_bundle` v2, one unit, its sample the step's live one).

    python examples/h200/m3bundle.py weights --state DIR --registration REG.json --draw DRAW.json --out OPEN.json
    python examples/h200/m3bundle.py replay --private DIR --secret FILE --registered ROOTS.json \\
        --registration REG.json --draw DRAW.json [run flags as m3draw.py's]
    python examples/h200/m3bundle.py build --private DIR --registration REG.json --draw DRAW.json \\
        --receipt RECEIPT.json --weights OPEN.json --out TREE [--workers W]

* **Which block.** Drawn g is slot ``g // units_per_slot``'s layout unit ``g % units_per_slot`` (`replay_bundle.block_of`),
  unit ``index`` of its family (`m3draw.families`). A gather's unit is no proof block (the Program has its parts'
  references and no Definition for it), and an embedding at a prompt position reads ingress: each is a visible failure,
  written to ``summary.json``, and never resampled. A norm's gain (one registered word a row) is a port of
  `RegisteredRead`\\ s, the embedding's token past the prompt the argmax its sequence committed a step before, and a PV
  block's V row past n a row of no reads (all zero).
* **weights** (where the weights registration's witness is): every registered row a drawn block reads (`window.Window.locate`'s
  `Registered` pieces), opened from the commit's state (`weights_commit.State`): its bytes, salt, string and path,
  each checked against the registered root (``REG.verify_read``).
* **replay** (the card's part): the run again under the prover's salt secret with kv_reads' `KeepingDriver` for the sampled
  proof blocks' slots, every step root and slot root held to the registered ``roots.json`` (exit 5 otherwise), and the drawn
  slots' committed rows at every step written to ``--private`` (0600, the prover's).
* **build** (CPU only): per block, its ports' bytes from the committed rows (`Window.inputs`, `Window.outputs`) and the
  opened weights, as one unit of the Program's Definition of its family (`definition`); the window commits each whole
  committed row under its step commitment's salt and each registered row under its registered salt, so every window
  string is the one its opening reaches (a port made of reads is the window's own row, its reads opened in the bundle),
  and its registration names every step root (`replay_bundle.steps_provenance`). The bundle is `replay_bundle.build`'s
  with circuits' `Window.bundle` pieces; then `replay_bundle.check` against the window's registration and `check_step`
  against the step registration at g must both be empty.

Exit 0 when every sampled proof block is bundled and checks; 3 when one isn't (a visible failure); 5 when a root differs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "interfaces")]
import kv_reads as KV  # noqa: E402
import m3draw as M3  # noqa: E402
from e2e_h200.commit import _slots, layout as commit_layout  # noqa: E402
from e2e_h200.plan import EPS_KEY  # noqa: E402
from e2e_h200.shape import PV_K, SHAPES  # noqa: E402
from e2e_h200.window import Registered, Run, Segment, Window  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG, registration as R  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops.boolean import gemm as BG  # noqa: E402
from verity.ml.circuits.ops.boolean import quant as BQ  # noqa: E402
from verity.ml.circuits.ops.boolean import rmsnorm as RN  # noqa: E402
from verity.ml.circuits.ops.boolean import rows as BR  # noqa: E402
from verity.ml.circuits.ops.boolean import rowwise as RW  # noqa: E402
from verity.ml.circuits.ops.boolean import softmax_row as SR  # noqa: E402
from verity.ml.windows import interface as IF, rows as CS  # noqa: E402
from verity_e2e_demo import replay_bundle as RB  # noqa: E402

FORMAT = "verity/e2e-h200/m3-drawn-blocks/v1"
LOOP = "e2e-h200-m3"
COINS = "vy-challenger-1's gate session, the step registration's digest its root (m3coin.py)"
#: the map's units with no bundle: a gather is references in the Program (`program.GATHER`), no Definition
UNBUNDLED = {"AllGather_v1": "a gather is its parts' references in the Program, which has no Definition for it; the "
                             "shared-row layout (window.shared_layout) has no gather units"}


class Visible(Exception):
    """A sampled proof block with no bundle: reported, never resampled."""


def definition(win: Window, fam: str) -> Any:
    """Family ``fam``'s proof block as the served Program calls it (`program.program`'s binds at the window's shape)."""
    t = win.units[fam].template
    if t in UNBUNDLED:
        raise Visible(f"{fam} ({t}): {UNBUNDLED[t]}")
    u = win.family(fam)
    f, sh = win.f, win.shape
    if t == "ScaledMmS8CoordinateRows_v1":
        K = win.tables[win.src[u.table] + "_quant"][2].nbytes
        return bind(BG.ScaledMmS8CoordinateRows, K=K, R=K // f.qchunk(K), DOT=BG.GEMM_SM120_S8["DOT"])
    if t == "QuantChunkS8_v1":
        op = u.table[:-len("_quant")]
        return bind(BQ.QuantChunkS8, N=u.nbytes, RELU=op.rsplit(".", 1)[-1] == "down")
    if t == "RowAbsMax_v1":
        op = u.table[:-len("_amax")]
        parts, w = f.parts(win.tables[win.src[op + "_quant"]][2].nbytes // 2)
        n = w if fam.endswith(".part") else (w + parts - 1 if parts > 1 else parts * w)
        return bind(BQ.RowAbsMax, N=n, RELU=op.rsplit(".", 1)[-1] == "down")
    if t == "Bf16AddRow_v2":
        return bind(BR.Bf16AddRow, N=f.chunk)
    if t == "QkSoftmaxRowPad_v1":
        return bind(SR.QkSoftmaxRowPad, T=PV_K, D=sh.head_dim, DOT=BG.GEMM_HOPPER["DOT"])
    if t == "PvRow_v1":
        return bind(BG.PvRowPad, T=PV_K, K=PV_K, D=sh.head_dim, DOT=BG.GEMM_HOPPER["DOT"])
    if t == "RmsSumSqChunks_v1":
        n = win.tables[win.src[u.table[:-len("_sums")]]][2].nbytes // 2
        return bind(RN.RmsSumSqChunks, C=f.sums_c(n), K=f.chunk)
    if t == "RmsNormOutSums_v1":
        n = sh.width
        return bind(RN.RmsNormOutSums, C=f.chunk, S=n // f.chunk, G=f.groups(n), N=n, EPS=EPS_KEY)
    if t == "EmbeddingCoordinate_v1":
        return bind(RW.EmbeddingCoordinate, V=sh.vocab)
    if t == "Argmax_v2":
        return bind(BR.Argmax, V=sh.vocab)
    raise Visible(f"{fam}: no Definition for {t}")


def binding_of(template: str) -> str:
    return "BOOLEAN_SM90_S8" if template.startswith("ScaledMm") else "BOOLEAN_HOPPER"


class _At:
    """A step of the schedule `Window` reads: batch row b is sequence b at ``positions[b]``."""

    def __init__(self, positions: Any) -> None:
        self.slots, self.positions = None, np.asarray(positions, np.int64)


def lockstep(steps: int, slots: int) -> list[_At]:
    """A lockstep run's schedule: every slot at position s at step s."""
    return [_At(np.full(slots, s)) for s in range(steps)]


def blocks_of(reg: dict, draw: dict, fams: list) -> list[dict]:
    """Each drawn g's slot, layout unit, family and index."""
    out = []
    for g in draw["units"]:
        b, u = RB.block_of(reg, g)
        fam, idx = M3.locate(fams, u)
        out.append({"g": int(g), "slot": int(b), "unit": int(u), "family": fam, "index": int(idx)})
    return out


def _families(shape: Any) -> list:
    """`m3draw.families` without a device: each family's first commit unit in the step's layout."""
    from e2e_h200.commit import Committer, fixture_secrets
    ctx, key = fixture_secrets("layout")
    return M3.families(Committer("slots", ctx, key, shape, "cpu", lambda s: bytes(32)), shape)


def registered_reads(win: Window, blk: dict, step: int) -> tuple[dict[str, tuple[str, int]], dict[str, list[RB.RegisteredRead]]]:
    """A block's registered reads: ``{port: (value, row)}`` of the ports that are one whole registered row, and
    ``{port: [RegisteredRead]}`` of those made of pieces of registered rows (a norm's gain, one word a row)."""
    whole, pieces = {}, {}
    for p in win.ports(blk["family"]):
        ps = win.locate(step, blk["slot"], blk["family"], blk["index"], p)
        if not any(isinstance(x, Registered) for x in ps):
            continue
        if not all(isinstance(x, Registered) for x in ps):
            raise Visible(f"{blk['family']} {p}: registered rows and other pieces in one port")
        if len(ps) == 1 and not ps[0].offset and ps[0].nbytes == 2 * win._weights[ps[0].value].words:
            whole[p] = (ps[0].value, ps[0].row)
            continue
        at, out = 0, []
        for x in ps:
            out.append(RB.RegisteredRead(x.value, x.row, at, x.offset, x.nbytes))
            at += x.nbytes // 2
        pieces[p] = out
    return whole, pieces


def token_pieces(win: Window, step: int, slot: int, prompt: int) -> tuple[Segment, ...]:
    """The token a block at ``step`` and ``slot`` embeds, where a commitment holds it: past the prompt (``prompt`` bytes),
    position p's token is the argmax its sequence committed at position p - 1 (`driver.Driver.run`)."""
    p = win.position(step, slot)
    if p < prompt:
        raise Visible(f"the embedding at position {p} reads the prompt's token, ingress that no commitment holds")
    t, s = win.history(step, slot)[p - 1]
    (segs,) = win.outputs(t, s, "argmax", 0)
    return segs


# -- weights (cpu-2) -------------------------------------------------------------------------------------------------------

def weights_phase(a: argparse.Namespace) -> int:
    from e2e_h200 import weights_commit as WC
    reg, draw = json.loads(Path(a.registration).read_text()), json.loads(Path(a.draw).read_text())
    st = WC.open_state(a.state, WC.seed_of(a.seed))
    if st.doc["program"] != reg["program"]:
        raise SystemExit("the weights' state registers another program than the step registration's")
    shape = SHAPES[st.doc["shape"]]
    win = Window(shape, lockstep(reg["step"] + 1, reg["slots"]), commit_layout(shape))
    want: dict[str, set[int]] = {}
    for blk in blocks_of(reg, draw, _families(shape)):
        try:
            win.family(blk["family"])
            whole, pieces = registered_reads(win, blk, reg["step"])
            for value, pos in [*whole.values(), *((q.value, q.position) for ps in pieces.values() for q in ps)]:
                want.setdefault(value, set()).add(pos)
        except (Visible, ValueError) as e:
            print(f"SKIP {blk['g']}: {e}", flush=True)
    regd = {e["value"]: e for e in reg["weights"]}
    out, ok = {"format": "verity/e2e-h200/m3-weights-open/v1", "program": st.doc["program"], "values": {}}, True
    for value, ps in sorted(want.items()):
        v, e = st.value(value), st.value(value).entry()
        if regd.get(value, {}).get("root") != e["root"]:
            raise SystemExit(f"{value}: the state's root is not the step registration's")
        dom = v.domain
        rows = {}
        for pos in sorted(ps):
            bc, path = v.bcs[pos], v.path(pos)
            good = dom.domain_id.hex() == e["domain"] and REG.verify_read(dom, v.root, pos, bc, path, v.schema)
            ok &= bool(good)
            rows[str(pos)] = {"row": v.row_bytes(pos).hex(), "salt": v.salt(pos).hex(), "string": bc.hex(),
                              "path": path.hex(), "ok": bool(good)}
        out["values"][value] = {"entry": e, "schema": v.schema, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("WEIGHTS " + json.dumps({"values": len(out["values"]), "rows": sum(len(v["rows"]) for v in out["values"].values()),
                                   "ok": ok}), flush=True)
    return 0 if ok else 3


class Opened:
    """A registered value's opened rows (``weights``'s), as `replay_bundle.build` reads a registered value."""

    def __init__(self, program: str, value: str, d: dict) -> None:
        self.program, self.value, self.schema, self._entry = program, value, d["schema"], d["entry"]
        self.rows = {int(p): {k: bytes.fromhex(v) for k, v in r.items() if k != "ok"} for p, r in d["rows"].items()}

    def entry(self) -> dict[str, Any]:
        return dict(self._entry)

    def _row(self, pos: int) -> dict[str, bytes]:
        if int(pos) not in self.rows:
            raise ValueError(f"{self.value} row {pos} was not opened")
        return self.rows[int(pos)]

    def string(self, pos: int) -> bytes:
        return self._row(pos)["string"]

    def path(self, pos: int) -> bytes:
        return self._row(pos)["path"]

    def salt(self, pos: int) -> bytes:
        return self._row(pos)["salt"]

    def row_bytes(self, pos: int) -> bytes:
        return self._row(pos)["row"]


# -- replay (the card) -----------------------------------------------------------------------------------------------------

def replay_phase(a: argparse.Namespace) -> int:
    from e2e_h200.commit import Committer, fixture_secrets
    from e2e_h200.driver import Params
    from e2e_h200.weights import Weights
    import run as RUN
    from e2e_h200 import served_weights as SW
    reg, draw = json.loads(Path(a.registration).read_text()), json.loads(Path(a.draw).read_text())
    shape = SHAPES[a.shape]
    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}.get(a.seed, a.seed.encode())
    weights = Weights(shape, seed)
    store = RUN.make_store(argparse.Namespace(store="generator", salt="", stage_lib="", weights_file="", weights_manifest="",
                                              workers=a.workers, run_id=a.run_id), weights, a.device)
    ctx, key = fixture_secrets(a.run_id)
    committer = Committer("slots", ctx, key, shape, a.device, KV.salt_from(M3.read_secret(Path(a.secret))))
    blocks = blocks_of(reg, draw, M3.families(committer, shape))
    slots = sorted({blk["slot"] for blk in blocks})
    t0 = time.perf_counter()
    driver = KV.KeepingDriver(shape, Params(shape, store, a.device, True), a.batch, committer, a.device,
                              micro_batch=a.micro_batch or None, attention=a.attention, rows=slots)
    results = driver.run(RUN.prompts_for(a.run_id, a.batch, a.prompt_bytes), a.steps)
    seconds = {"run": round(time.perf_counter() - t0, 1)}
    rc = KV.root_check(results, json.loads(Path(a.registered).read_text()), committer)
    rc["registration_step_root_equal"] = results[reg["step"]].root.hex() == reg["step_root"]
    rc["ok"] = rc["ok"] and rc["registration_step_root_equal"]
    print("ROOTCHECK " + ("EQUAL " if rc["ok"] else "UNEQUAL ") + json.dumps(rc), flush=True)
    if not rc["ok"]:
        return 5
    t0 = time.perf_counter()
    st = {"shape": a.shape, "batch": a.batch, "steps": len(results), "prompt_bytes": a.prompt_bytes, "context": committer.context,
          "layout_digest": committer.describe()["layout_digest"], "root_check": rc, "blocks": blocks,
          "salt_keys": [r.salt_key for r in results], "slot_roots": [tuple(r.slot_roots) for r in results],
          "positions": [np.asarray(r.positions) for r in results],
          "sources": {(t, b): KV.slot_rows(committer, driver.steps_of[b][t]).sources for b in slots for t in range(len(results))}}
    path = _private(a)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
    seconds["keep"] = round(time.perf_counter() - t0, 1)
    store.close()
    print("REPLAYED " + json.dumps({"slots": slots, "steps": len(results), "seconds": seconds}), flush=True)
    return 0


def _private(a: argparse.Namespace) -> Path:
    d = Path(a.private).resolve()
    if d.exists() and d.stat().st_mode & 0o077:
        raise SystemExit(f"{d}: the prover's state is 0700")
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d / "replay.pkl"


# -- build (CPU) -----------------------------------------------------------------------------------------------------------

def _words(b: bytes, words: int, what: str) -> np.ndarray:
    if len(b) != 2 * words:
        raise Visible(f"{what}: {len(b)} bytes, not the Definition's {words} words")
    return np.frombuffer(b, "<u2").astype(np.uint16).reshape(1, words)


def bundle_block(blk: dict, st: dict, com: RB.Commitment, win: Window, weights: dict[str, Opened], reg: dict,
                 receipt: dict, out: Path, workers: int) -> dict:
    """One sampled proof block's window and bundle in ``out`` (module docstring) -> its record."""
    S, _ = _slots()
    lay, step, b, fam, idx = com.layout, reg["step"], blk["slot"], blk["family"], blk["index"]
    fn = definition(win, fam)
    tl = IF.layout(fn)
    if [p for p, _ in tl.rows] != list(win.ports(fam)):
        raise Visible(f"{fam}: the Definition's ports {[p for p, _ in tl.rows]} are not the window's {list(win.ports(fam))}")
    rows_of = lambda t, s: KV.SlotRows(lay, st["sources"][t, s])  # noqa: E731
    reads, pieces = registered_reads(win, blk, step)
    wrow = lambda value, r: weights[value].row_bytes(r)  # noqa: E731

    def run(t: int, s: int, name: str) -> int:
        if name != "n":
            raise Visible(f"{fam}: it reads run input {name!r}, which no commitment holds")
        return int(st["positions"][t][s]) + 1

    c = {"block": 0, "step": step, "slot": b, "op": fam, "unit": idx}
    answers, data = {}, {"rows": {}, "outs": {}}
    for p, words in tl.rows:
        located = win.locate(step, b, fam, idx, p)
        if located == (Run("token"),):
            located = token_pieces(win, step, b, st.get("prompt_bytes", 0))
            answers[p] = [{"step": x.step, "slot": x.slot, **dict(zip(("unit", "rank"), win.unit_of(x.row))), "at": 0,
                           "start": x.offset, "nbytes": x.nbytes} for x in located]
        elif p in pieces:
            answers[p] = pieces[p]
        else:
            answers[p] = None if p in reads else win.bundle(step, b, fam, idx, p)
        data["rows"][p] = _words(win.assemble(located, rows_of, wrow, run, step, b), words, p)
    outs = win.outputs(step, b, fam, idx)
    if len(outs) != len(tl.outs):
        raise Visible(f"{fam}: {len(outs)} committed outputs, the Definition's {len(tl.outs)}")
    for (p, words), segs in zip(tl.outs, outs, strict=True):
        data["outs"][p] = _words(win.assemble(segs, rows_of, wrow, run, step, b), words, p)
        answers[p] = ans = win.bundle(step, b, fam, idx, p)
        units = {ans[0]} if isinstance(ans, tuple) else {q["unit"] for q in ans if "unit" in q}
        if units != {blk["unit"]}:
            raise Visible(f"{fam} {p}: its output is in layout units {sorted(units)}, not drawn unit {blk['unit']}")

    circuit = IF.Circuit(f"{LOOP}/{fam}", ((fn, 1),))
    run_root = hashlib.sha512(f"{LOOP}/{reg['run']}/{blk['g']}".encode()).hexdigest()
    iw = IF.Window(circuit, run_root, D.parse_law("subset:1"), 1)
    tid = circuit.ids[0]
    iw._recorded[tid] = [IF._Recorded([], [], [*data["rows"].values(), *data["outs"].values()], c, 1, 0)]
    (m,) = iw.window.members
    salts_of = {}
    for p, ans in answers.items():
        if p in reads:
            salts_of[m.port(p)] = weights[reads[p][0]].salt(reads[p][1])
        elif isinstance(ans, tuple):
            lu, rank, _ = ans
            salts_of[m.port(p)] = S.row_salts(st["salt_keys"][step], b, lay.row_base[lu] + rank, 1)
    committed = CS.commit(iw.window, [data], salts_of=salts_of)
    wdir = out / "window"
    iw.write(committed, wdir, {"loop": LOOP, "block": blk["g"], **RB.steps_provenance(com)}, registrar=R)

    def locate(cc: dict, template: str, port: str) -> Any:
        ans = answers[port]
        if ans is None or isinstance(ans, tuple):
            return ans
        return [q if not isinstance(q, dict) else RB.Position(**q) if "offset" in q else RB.Read(**q) for q in ans]

    doc, pay = RB.build(wdir, [0], commitment=com, locate=locate, binding=binding_of(win.family(fam).template),
                        coords=lambda label, offset: dict(label), reads=lambda cc, t, p: reads.get(p), weights=weights,
                        draw={"source": "live", "law": D.parse_law("subset:1"), "receipt": receipt, "coins": COINS},
                        workers=workers)
    RB.write(doc, pay, out, "bundle")
    rec = json.loads((wdir / "registration.json").read_text())
    check = RB.check(doc, pay, registration=rec)
    step_check = RB.check_step(doc, pay, reg, blk["g"], registration=rec)
    flipped = bytearray(pay)
    flipped[doc["units"][0]["rows"][0]["row"][0]] ^= 1
    refused = RB.check(doc, bytes(flipped), registration=rec)
    other = copy.deepcopy(reg)
    other["step_root"] = "00" * 64
    return {**blk, "template": IF.descriptor_id(fn) if hasattr(IF, "descriptor_id") else fn.id, "map_template":
            win.family(fam).template, "binding": binding_of(win.family(fam).template),
            "ports": {p: ("registered" if a_ is None else "row" if isinstance(a_, tuple) else f"{len(a_)} reads")
                      for p, a_ in answers.items()},
            "read_steps": sorted({q["slot"]["step"] for r in doc["units"][0]["rows"] for q in r.get("reads") or ()
                                  if q["kind"] == "step"}),
            "registered": sorted({f"{v}[{p}]" for v, p in reads.values()} |
                                 {f"{q.value}[{q.position}]" for ps in pieces.values() for q in ps}),
            "check": check, "check_step": step_check, "ok": not check and not step_check,
            "refused": {"a flipped row bit": sorted(refused),
                        "another step root": sorted(RB.check_step(doc, pay, other, blk["g"], registration=rec))},
            "payload_bytes": len(pay), "registration": R.digest(rec)}


def build_phase(a: argparse.Namespace) -> int:
    with _private(a).open("rb") as f:
        st = pickle.load(f)
    reg, draw = json.loads(Path(a.registration).read_text()), json.loads(Path(a.draw).read_text())
    receipt = json.loads(Path(a.receipt).read_text())
    if receipt.get("registration") != RB.step_digest(reg) or not D.check_draw(draw, reg, reg["law"]).ok:
        raise SystemExit("the receipt doesn't name the step registration, or the draw isn't one under its law")
    wdoc = json.loads(Path(a.weights).read_text()) if a.weights else {"program": reg["program"], "values": {}}
    weights = {v: Opened(wdoc["program"], v, d) for v, d in wdoc["values"].items()}
    shape = SHAPES[st["shape"]]
    S, _ = _slots()
    lay = commit_layout(shape)
    if S.layout_digest(lay).hex() != reg["layout_digest"] or st["layout_digest"] != reg["layout_digest"]:
        raise SystemExit("the replay's layout is not the registered step's")
    slots_of: dict[int, dict[int, Any]] = {}
    for (t, b), src in st["sources"].items():
        slots_of.setdefault(t, {})[b] = KV.SlotRows(lay, src)
    com = RB.Commitment(st["context"], lay, st["batch"],
                        {(0, t): RB.StepRecord(st["salt_keys"][t], slots_of.get(t, {}), st["slot_roots"][t])
                         for t in range(st["steps"])})
    mine = RB.step_registration(com, {e["value"]: type("W", (), {"root": bytes.fromhex(e["root"])})() for e in reg["weights"]},
                                reg["law"], run=reg["run"], program=reg["program"], block=reg["block"], step=reg["step"],
                                read_steps=RB.STEPS_KEY in reg)
    if RB.step_digest(mine) != RB.step_digest(reg):
        raise SystemExit("the replay's commitments are not the step registration's")
    win = Window(shape, [_At(p) for p in st["positions"]], lay)
    tree = Path(a.out)
    (tree / "blocks").mkdir(parents=True, exist_ok=True)
    for name, doc in (("step-registration.json", reg), ("receipt.json", receipt), ("draw.json", draw)):
        (tree / name).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    got, failures = [], {}
    for blk in st["blocks"]:
        t0 = time.perf_counter()
        try:
            r = bundle_block(blk, st, com, win, weights, reg, receipt, tree / "blocks" / str(blk["g"]), a.workers)
        except (Visible, ValueError, KeyError) as e:
            failures[str(blk["g"])] = f"{type(e).__name__}: {e}"
            print("FAILED " + json.dumps({**blk, "why": failures[str(blk["g"])]}), flush=True)
            continue
        r["seconds"] = round(time.perf_counter() - t0, 1)
        got.append(r)
        print("BLOCK " + json.dumps({k: r[k] for k in ("g", "slot", "unit", "family", "index", "map_template", "read_steps",
                                                       "registered", "ok", "check", "check_step", "refused", "seconds")}),
              flush=True)
    ok = not failures and all(r["ok"] and all(r["refused"].values()) for r in got)
    summary = {"format": FORMAT, "registration": RB.step_digest(reg), "run": reg["run"], "program": reg["program"],
               "step": reg["step"], "draw": draw["units"], "root_check": st["root_check"], "blocks": got,
               "visible_failures": failures, "ok": ok}
    (tree / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("VERDICT " + json.dumps({"ok": ok, "registration": summary["registration"], "drawn": len(draw["units"]),
                                   "bundled": len(got), "checks_clean": sum(r["ok"] for r in got),
                                   "visible_failures": failures}), flush=True)
    return 0 if ok else 3


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("weights", "replay", "build"))
    ap.add_argument("--registration", required=True)
    ap.add_argument("--draw", required=True)
    ap.add_argument("--private", default="", help="replay, build: the prover's state between them (0700)")
    ap.add_argument("--state", default="", help="weights: the weights registration's commit state")
    ap.add_argument("--secret", default="")
    ap.add_argument("--registered", default="", help="replay: m3draw commit's roots.json")
    ap.add_argument("--receipt", default="")
    ap.add_argument("--weights", default="", help="build: the weights phase's output")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt-bytes", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--attention", default=None, choices=("attn_ref", "numpy"))
    ap.add_argument("--run-id", default="e2e-h200-fixture")
    ap.add_argument("--seed", default="test")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return {"weights": weights_phase, "replay": replay_phase, "build": build_phase}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
