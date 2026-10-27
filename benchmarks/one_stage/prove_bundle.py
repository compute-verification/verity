"""A drawn-unit replay bundle (`verity/drawn-unit-bundle/v2`, `benchmarks/e2e_demo`) proved: each drawn member's units in
one C-Flock `prove --zk` session behind the Lean firewall, served the bundle's own draw, and decided by Lean's `verify --zk`,
the verifier of record; then the honest session with one bit of its proof flipped, which Lean must refuse, and optionally a
tampered window's member, whose proof must not be accepted.

    python benchmarks/one_stage/prove_bundle.py make --scratch DIR --out DIR [--k K --lean flock-verify]
    python benchmarks/one_stage/prove_bundle.py --bundle B.json --window DIR --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --out DIR [--members N] [--tampered DIR] [--live] [--flips N]
    python benchmarks/one_stage/prove_bundle.py draw --step DIR --lean flock-verify
    python benchmarks/one_stage/prove_bundle.py --step DIR --bin ... --lean ... --firewall ... --out DIR [--flips N]

* **A registered step** (``--step DIR``, PRF-172's M3): the draw is held to a step's registration, not to a window's.
  ``DIR/step-registration.json`` (`replay_bundle.STEP_FORMAT`) registers one serving step's commit-slots roots (its step
  root, every slot root, layout digest, context, and the registered values' roots) with the law and the population, every
  ``(slot, layout unit)`` of the step: proof block ``g`` is slot ``g // units_per_slot``'s layout unit
  ``g % units_per_slot``. ``receipt.json`` is the challenger's receipt naming the registration's digest
  (`replay_bundle.step_digest`), bound before the coin, and ``draw.json`` the live draw of record under the registered
  law over the registered population (``draw`` here: Lean's `flock-verify draw --registration` on its own coins). After
  the coin each drawn block has ``blocks/<g>/``: a one-member window of its own rows and salts (``window/``), its bundle
  (``bundle.{json,bin}``, its draw live and naming the step's receipt) and optionally a tampered twin (``tampered/``). Each
  block's bundle is held to its window and, by `replay_bundle.check_step`, to the registered step at ``g``: every row's
  paths reach the registered step root and slot root, and its string is the window's leaf. A drawn index with no block,
  or a block that is refused, is a visible failure; it is not proved, and the verdict is not ok. Lean decides each
  block's window (its session); the ties to the step root are Python's (`TIES`, in the summary).

* **The draw** (`make`): the torch loop's window and its tampered twin written under the fixture's test key, then the
  bundle. With ``--k K`` the windows register `subset:K` and the draw is live (`live_draw`): the challenger logs its
  receipt of the window's registration, then draws the registered law's units uniformly over the window's whole
  partition (every unit of every member) with Lean's `flock-verify draw --registration` on its own coins, which refuses
  a law or population other than the registered ones. The bundle's ``draw`` is that draw with ``source: live``, the
  receipt, and the Lean draw of record. Without it the draw is the fixture's hand-chosen `FIXTURE_DRAW`, under the
  fixture's registered `subset:4`.
* **The bundle held to the window** before anything is proved. The window is the prover's (what ``Window.write`` wrote, its
  private files included), since C-Flock's statement is a whole member. Checks: the bundle's own check
  (`replay_bundle.check` against the window's registration: every string, path and domain, and each unit's Definition
  replayed, and the draw's law the registered one); the bundle's run, registration digest and partition are the
  window's; each drawn member's public body is the one the bundle names (SHA-512); and a live draw (``--live`` requires
  one) is held to the window (`held_draw`): its receipt names the window's registration, and its record is a draw of
  the bundle's units under the registered law over the registered population (`draw.check_draw`).
* **Staging** as a5 stages a drawn member. The circuit is composed from the template's Definition, bound to the window's
  partition and program (`records_audit.verifier_window`'s composer). The public file is made under that circuit and
  held to the registration's roots and domains, and the prover's file is made under the same header.
* **The session** serves the bundle's member draw (`serve --zk --draw-file`). The prover runs behind the firewall's send
  check (``FC_FIREWALL=1``, Lean's `flock-firewall`), and Lean `verify --zk --archive --partition --program` decides it.
  The sessions run one at a time, and Lean verifies each (one core) while the next one proves.
  A unit is accepted when its session's M0 and Lean verdicts accept, the served draw is the bundle's, and the proved
  units are the bundle's units.
* **Flipped bits** (`flipped`): `zk_mutants.py`'s one-bit proof edits of the honest session of least k_log (each with the
  retained rounds a prover sending it would have made), each verified by Lean, which must refuse it with the reason the
  mutant names. ``--flips N`` verifies only the first N edits (M3's one flipped bit at m30 costs a Lean verify each).
* **Tampered** (``--tampered DIR``): the same member of a window whose outputs are not its Definition's, proved with
  the prover's self-check skipped, once behind the firewall (whose Lean recheck must stop it) and once without it (Lean
  `verify --zk` must refuse the proof that reaches it).

The weights' rows are bound here by the window's port roots. A window whose member headers name their registered reads
(``rows.write``'s ``registered``) would be verified with ``--registered`` as well. The torch loop's headers don't name
them: its program reads the weights as unit inputs, not as registered parameters (`one_stage.registered.reads`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "benchmarks" / "prove_path"), str(REPO / "benchmarks" / "e2e_demo"),
                str(REPO / "verity" / "ml" / "flock" / "verifier")]

import a0  # noqa: E402
import a5 as A5  # noqa: E402
import records_audit as RA  # noqa: E402
import zk_mutants as ZM  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.windows import flock_inputs as FI, rows as CS  # noqa: E402
from verity_e2e_demo import replay_bundle as RB  # noqa: E402

FORMAT = "verity/prove-bundle/v0"
#: the torch loop's card binding, the fixture's (`replay_bundle.fixture_bundle`)
BINDING = "GEMM_SM120_S8"
CHALLENGER = "prove_bundle.py, the challenger: Lean flock-verify draw on its own coins (IO.getRandomBytes), after its receipt"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hold(doc: dict, payload: bytes, wdir: Path, step: tuple[dict, int] | None = None) -> tuple[dict[str, bool], dict]:
    """The bundle held to the window (module docstring): ``({check: ok}, the bundle check's findings)``. With ``step``,
    ``(step registration, drawn index)``, the bundle's check is `replay_bundle.check_step`'s."""
    rec = json.loads((wdir / "registration.json").read_text())
    win = json.loads((wdir / "window.json").read_text())
    bad = RB.check_step(doc, payload, *step, registration=rec) if step else RB.check(doc, payload, registration=rec)
    ok = {"bundle_check": not bad, "run": doc["window"]["run"] == win["run"],
          "registration": doc["window"]["registration"] == win["registration_digest"],
          "partition": doc["window"]["partition"] == win["partition_digest"]}
    for mm in doc["members"]:
        pub = wdir / f"m{mm['k']}-public.bin"
        ok[f"public:m{mm['k']}"] = pub.is_file() and hashlib.sha512(pub.read_bytes()).hexdigest() == mm["public_sha512"]
    return ok, bad


def live_draw(lean: str, wdir: Path, out: Path) -> tuple[dict, dict]:
    """The challenger's draw of record over the window's whole partition: its receipt of the window's registration, logged
    before any coin is read, then the registered law's units drawn uniformly by Lean's `flock-verify draw` on its own
    coins, which `--registration` holds to the law and population the window registered: ``(receipt, draw)``."""
    rec = json.loads((wdir / "registration.json").read_text())
    win = json.loads((wdir / "window.json").read_text())
    if R.digest(rec) != win["registration_digest"]:
        raise SystemExit("the window's registration is not the one its window.json names")
    if rec["law"].get("law") != "subset":
        raise SystemExit(f"the registered law {rec['law']} is not a subset law")
    out.mkdir(parents=True, exist_ok=True)
    receipt = R.receipt(rec, CHALLENGER)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    p = a0.run([lean, "draw", "--population", str(rec["population"]), "--k", str(rec["law"]["k"]),
                "--registration", str(wdir / "registration.json")], out / "draw.log")
    line = next((x for x in p.stdout.splitlines() if x.startswith("DRAW ")), None)
    if p.returncode != 0 or line is None:
        raise SystemExit(f"flock-verify draw: rc {p.returncode}: {p.stderr[-500:]}")
    draw = json.loads(line.split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(draw, sort_keys=True))
    return receipt, draw


def held_draw(doc: dict, wdir: Path) -> dict[str, bool]:
    """A live draw held to the window (module docstring); ``{}`` for a fixture's. Its law against the registered one is
    also the bundle check's (`replay_bundle.check`)."""
    d = doc["draw"]
    if d.get("source") != "live":
        return {}
    rec = json.loads((wdir / "registration.json").read_text())
    r, record, units = d.get("receipt") or {}, d.get("record") or {}, d.get("units") or []
    return {"draw:receipt": r.get("format") == R.RECEIPT and r.get("registration") == R.digest(rec),
            "draw:law": d.get("law") == rec["law"],
            "draw:record": D.check_draw(record, rec, rec["law"]).ok,
            "draw:units": record.get("units") == sorted(units)}


def _read_step(sdir: Path) -> dict:
    reg = json.loads((sdir / "step-registration.json").read_text())
    if not RB.STEP_KEYS <= set(reg) <= RB.STEP_KEYS | RB.STEP_OPTIONAL or reg.get("format") != RB.STEP_FORMAT:
        raise SystemExit(f"{sdir}/step-registration.json: not a {RB.STEP_FORMAT} with exactly {sorted(RB.STEP_KEYS)}")
    if reg["law"].get("law") != "subset":
        raise SystemExit(f"the registered law {reg['law']} is not a subset law")
    return reg


def step_draw(lean: str, sdir: Path) -> tuple[dict, dict]:
    """The challenger's draw of record over a registered step (`--step`): its receipt naming the step registration's
    digest, logged before any coin is read, then the registered law's proof blocks drawn uniformly over the registered
    population by Lean's `flock-verify draw` on its own coins, held by `--registration` to that law and population:
    ``(receipt, draw)``, written as ``receipt.json`` and ``draw.json`` in ``sdir``."""
    reg = _read_step(sdir)
    receipt = {**R.receipt(reg, CHALLENGER), "registration": RB.step_digest(reg)}
    (sdir / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    p = a0.run([lean, "draw", "--population", str(reg["population"]), "--k", str(reg["law"]["k"]),
                "--registration", str(sdir / "step-registration.json")], sdir / "draw.log")
    line = next((x for x in p.stdout.splitlines() if x.startswith("DRAW ")), None)
    if p.returncode != 0 or line is None:
        raise SystemExit(f"flock-verify draw: rc {p.returncode}: {p.stderr[-500:]}")
    draw = json.loads(line.split(" ", 1)[1])
    (sdir / "draw.json").write_text(json.dumps(draw, sort_keys=True))
    return receipt, draw


def hold_step(sdir: Path) -> tuple[dict, dict[str, bool], dict[int, dict], list[int]]:
    """A step tree held before anything is proved (module docstring, ``--step``): ``(registration, {check: ok} of the step,
    {index: block} of the delivered drawn blocks, the drawn indices with no block)``. Each block is ``{dir, doc, payload,
    window, held, findings}``, its ``held`` the block's bundle held to its window and to the registered step at its
    index (`replay_bundle.check_step`), and its draw a live one naming the step's receipt."""
    reg = _read_step(sdir)
    receipt = json.loads((sdir / "receipt.json").read_text())
    draw = json.loads((sdir / "draw.json").read_text())
    units = draw.get("units") if isinstance(draw.get("units"), list) else []
    dirs = {int(d.name): d for d in (sdir / "blocks").iterdir() if d.is_dir() and d.name.isdigit()} \
        if (sdir / "blocks").is_dir() else {}
    held = {"step:receipt": receipt.get("format") == R.RECEIPT and receipt.get("registration") == RB.step_digest(reg),
            "step:draw": D.check_draw(draw, reg, reg["law"]).ok,
            "step:blocks": set(dirs) <= set(units)}
    blocks = {}
    for g in sorted(set(dirs) & set(units)):
        d = dirs[g]
        doc, payload = RB.read(d / "bundle.json")
        ok, bad = hold(doc, payload, d / "window", step=(reg, g))
        bd = doc["draw"]
        ok["draw:live"] = bd.get("source") == "live" and bd.get("receipt") == receipt
        blocks[g] = {"dir": d, "doc": doc, "payload": payload, "window": d / "window", "held": ok, "findings": bad}
    return reg, held, blocks, sorted(set(units) - set(dirs))


#: infra's published pin of vy-challenger-1's key (`served_commit.CHALLENGER_KEY_PIN`; its tests pin these bytes)
CHALLENGER_PIN = REPO / "benchmarks" / "pouw" / "tests" / "fixtures" / "challenger-key.json"


def _lms_verify(key, message: bytes, signature: bytes) -> bool:
    from verity.ml.crypto import lms
    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def held_coin(sdir: Path, reg: dict, draw: dict, main: Path | None) -> tuple[dict[str, bool], list[int] | None, dict]:
    """The step's live coin, `e2e_h200/m3coin.py`'s (``coin.json`` with its session's files beside it, in ``sdir`` or
    ``sdir/coin/``), re-derived here rather than taken from ``draw.json``: ``({check: ok}, the derived units, info)``.
    The session is the one vy-challenger-1's ledger, signed under infra's pinned key (`CHALLENGER_PIN`), opens on the
    step registration's digest, from a commit on main's first-parent line (``main``: ``git rev-list --first-parent``);
    its commitment opens the recorded coin after that digest; and the units are m3coin's ``derive(coin, DOMAIN,
    {registration: digest}).subset(k, population, "m3")`` under m3coin's fixed scheme, label and domain, which must equal
    ``draw.json``'s and the coin record's."""
    sys.path[:0] = [str(REPO / "benchmarks" / "e2e_h200"), str(REPO / "benchmarks" / "pouw" / "served_zk")]
    import m3coin as COIN
    import served_commit as SC
    from verity.core.protocols.sampled_proofs import challenger as Ch, remote as RM
    path = next((p for p in (sdir / "coin.json", sdir / "coin" / "coin.json") if p.is_file()), None)
    if path is None:
        return {"coin": False}, None, {"why": "no coin.json in the step tree"}
    coin, d, digest = json.loads(path.read_text()), path.parent, RB.step_digest(reg)
    k, n = reg["law"]["k"], reg["population"]
    held = {"coin:registration": coin.get("registration") == digest,
            "coin:layout": (coin.get("scheme"), coin.get("families"), coin.get("domain"), coin.get("law"),
                            coin.get("population")) == (COIN.SCHEME, [[COIN.LABEL, 1]], COIN.DOMAIN, reg["law"], n)}
    info: dict = {"session": (coin.get("reserve") or {}).get("session"), "commitment": coin.get("commitment")}
    opening, derived = None, None
    try:
        opening = Ch.LiveOpening.from_json(coin.get("opening"))
        derived = sorted(COIN.draw(opening.coin, digest, k, n))
    except (Ch.CoinError, TypeError, ValueError) as e:
        info["why"] = f"opening: {e}"
    held["coin:units"] = derived is not None and derived == draw.get("units") and sorted(coin.get("units") or []) == derived
    try:
        if opening is None or main is None:
            raise ValueError("no opening" if opening is None else "no main first-parent history (--main) to hold the pins to")
        key = SC.pinned_challenger_key(CHALLENGER_PIN)
        sid, window, after = info["session"], {"root": digest}, bytes.fromhex(digest)
        sess = d / sid / "out" if (d / sid / "out").is_dir() else d / sid
        ledger = d / f"{sid}.ledger.jsonl"
        commitment, roots = RM.trusted_session(sess, [ledger], window, key, _lms_verify)
        pins = [x for x in (d / f"{sid}.pins.jsonl").read_text().split("\n") if x]
        audited = Ch.pinned_session(Ch.ledger_head([RM.ledger_lines(ledger)], key, _lms_verify), window, key, pins,
                                    RM.main_lines(main), _lms_verify)
        held.update({"coin:session": audited.session == sid, "coin:roots": roots == {COIN.LABEL: (after,)},
                     "coin:commitment": commitment.root.hex() == coin.get("commitment"),
                     "coin:opens": Ch.live_reason(commitment, opening, COIN.LABEL, 0, after) is None})
    except (Ch.CoinError, OSError, ValueError, KeyError, TypeError) as e:
        held["coin:session"] = False
        info["why"] = f"session: {e}"
    return held, derived, info


TIES = {"lean": "each drawn block's one-member window: its C-Flock --zk session behind the Lean firewall, decided by Lean "
                "verify --zk --partition --program (the window's statement, its port roots and its member's Definition)",
        "python": "each drawn block's rows to the registered step root (replay_bundle.check: the unit path, slot_path and "
                  "step_path, and each row's string the window's leaf; replay_bundle.check_step: the commitment's step root, "
                  "slot roots, layout and context the registration's, the block the drawn index's), the receipt naming the "
                  "step registration's digest, and the draw under the registered law over the registered population "
                  "(draw.check_draw), and with --coin those units re-derived from the challenger's live coin over the "
                  "step registration's digest, its session the one the challenger's ledger opens on it under infra's "
                  "pinned key (held_coin)"}


def make_step(s: Path, out: Path, k: int, lean: str, kv: bool = False) -> int:
    """The torch loop's step as a step tree (`--step`): its registration from the fixture's commitment, its proof blocks the
    ScaledMm coordinate units (`proof_units`; the layers' quantizer units are committed, not proved), the live draw over
    it (`step_draw`), and per drawn block a one-unit bundle of the window, registered `subset:1`, with the step's receipt.
    ``kv``: the H200's form, two steps, the drawn step the last, its registration naming every step's root (`steps`) and
    each block's ``x`` read half from the step before (`replay_bundle.torch_loop_kv`)."""
    two = {"steps": 2, "restate": RB.torch_loop_kv} if kv else {}
    _, weights, com = RB.torch_loop_window(s / "window", law="subset:1", **two)
    RB.torch_loop_window(s / "tampered", tamper=True, law="subset:1", **two)
    b, st = max(com.steps)
    locate = RB.torch_loop_kv_locate(128) if kv else RB.torch_loop_locate(128)
    # the proof blocks are the ScaledMm coordinate units; each layer's quantizer unit (its first) is committed, not proved
    pu = [u for u in range(com.layout.unit_count) if u % 129 != 0]
    reg = RB.step_registration(com, weights, D.parse_law(f"subset:{k}"), run=RB.RUN_ID,
                               program=next(iter(weights.values())).program, block=b, step=st, proof_units=pu,
                               read_steps=kv)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step-registration.json").write_text(json.dumps(reg, indent=1, sort_keys=True))
    receipt, draw = step_draw(lean, out)
    w = RB._Window(s / "window")
    at = {}
    for u in range(w.win["population"]):
        c = RB.torch_loop_coords(*w.label(u))
        if (c["block"], c["step"]) != (b, st):
            continue
        at[c["slot"] * len(pu) + pu.index(c["layer"] * 129 + 1 + c["unit"])] = u
    made = {}
    for g in draw["units"]:
        if g not in at:
            continue
        d = out / "blocks" / str(g)
        doc, pay = RB.build(s / "window", [at[g]], commitment=com, locate=locate, binding=BINDING,
                            reads=RB.torch_loop_reads, weights=weights,
                            draw={"source": "live", "law": D.parse_law("subset:1"), "receipt": receipt, "coins": CHALLENGER})
        RB.write(doc, pay, d, "bundle")
        (d / "window").symlink_to((s / "window").resolve())
        if not made:
            (d / "tampered").symlink_to((s / "tampered").resolve())
        made[g] = at[g]
    print("MAKE " + json.dumps({"step": str(out), "registration": RB.step_digest(reg), "population": reg["population"],
                                "drawn": draw["units"], "blocks": made,
                                "no_block": [g for g in draw["units"] if g not in at]}, sort_keys=True), flush=True)
    return 0


def make(argv) -> int:
    ap = argparse.ArgumentParser(prog="prove_bundle.py make", description="The torch loop's windows and the bundle of a draw.")
    ap.add_argument("--scratch", required=True, help="writes window/ and tampered/ here")
    ap.add_argument("--out", required=True, help="writes bundle.{json,bin} (and a live draw's receipt and record) here")
    ap.add_argument("--k", type=int, default=0,
                    help="the windows register subset:K and the draw is live, K units over the whole partition (else FIXTURE_DRAW)")
    ap.add_argument("--lean", help="flock-verify, for a live draw")
    ap.add_argument("--step", action="store_true", help="a step tree (`--step`) of K blocks drawn over the step's registration")
    ap.add_argument("--kv", action="store_true", help="with --step: two steps, the drawn one's x read across both (H200's form)")
    a = ap.parse_args(argv)
    s, out = Path(a.scratch), Path(a.out)
    if a.step:
        if not (a.k and a.lean):
            raise SystemExit("a step tree is drawn live: --k K --lean flock-verify")
        return make_step(s, out, a.k, a.lean, kv=a.kv)
    law = f"subset:{a.k}" if a.k else None
    _, weights, com = RB.torch_loop_window(s / "window", law=law)
    RB.torch_loop_window(s / "tampered", tamper=True, law=law)
    rec = json.loads((s / "window" / "registration.json").read_text())
    if a.k:
        if not a.lean:
            raise SystemExit("a live draw needs --lean")
        receipt, record = live_draw(a.lean, s / "window", out)
        units, draw = record["units"], {"source": "live", "law": rec["law"], "receipt": receipt, "record": record,
                                        "coins": CHALLENGER}
    else:
        units, draw = list(RB.FIXTURE_DRAW), {"source": "fixture", "law": rec["law"]}
    doc, pay = RB.build(s / "window", units, commitment=com, locate=RB.torch_loop_locate(128), binding=BINDING,
                        reads=RB.torch_loop_reads, weights=weights, draw=draw)
    pj, _ = RB.write(doc, pay, out, "bundle")
    print("MAKE " + json.dumps({"bundle": str(pj), "source": doc["draw"]["source"], "law": rec["law"],
                                "units": doc["draw"]["units"], "members": [m["k"] for m in doc["members"]],
                                "digests": RB.digests(doc, pay)}, sort_keys=True), flush=True)
    return 0


class Staged:
    """The window's verifier side, read once: its Spec, registration, ports, and each template's circuit composed once."""

    def __init__(self, wdir: Path, out: Path) -> None:
        self.wdir, self.out = wdir, out
        self.rec = json.loads((wdir / "registration.json").read_text())
        self.mine, _doc, self.sets = RA.verifier_window(wdir, D.parse_law(_law(self.rec["law"])), None)
        self.files = (json.loads((wdir / "window.json").read_text()), self.rec)
        self.want = CS.expected_ports(self.mine)
        self.roots = {r["port"]: r for r in self.rec["roots"]}
        out.mkdir(parents=True, exist_ok=True)
        (out / "partition.json").write_text(json.dumps(self.mine.partition, sort_keys=True))
        (out / "program.json").write_bytes(canonical_json(encode_program(self.mine.program)))
        self.composed: dict[str, tuple[Path, dict, object]] = {}

    def member(self, k: int) -> dict:
        """Member ``k`` staged as a5 stages a drawn one: its circuit, public file and header held to the registration, and
        its prover's file under that header."""
        m = self.mine.members[k]
        pd, prog = self.mine.partition_digest, self.mine.partition["program"]
        if m.template not in self.composed:
            cdir = self.out / f"circuit-{len(self.composed)}"
            cdir.mkdir(parents=True, exist_ok=True)
            made = self.sets[m.template][0](cdir / "circuit.txt", pd, prog)
            self.composed[m.template] = (cdir / "circuit.txt", made, FI.flock_circuit((cdir / "circuit.txt").read_bytes()))
        circuit, made, circ = self.composed[m.template]
        st_dir = self.out / f"m{k}"
        st_dir.mkdir(parents=True, exist_ok=True)
        got = FI.flock_input_files(self.wdir, k, circ, st_dir / "prover", write=("pub",), files=self.files)
        A5.link(Path(got["pub"]), st_dir / f"pub-{m.n}.bin")
        header = a0.public_header(st_dir, m.n)
        fv = header["frame_v3"]
        ok = {"pin": header.get("circuit_sha512") == made["pin"],
              "partition": circ.meta.get("partition") == {"rule": P.FORMAT, "digest": pd},
              "program": circ.meta.get("program_sha512") == prog,
              "indices": header["units"]["indices"] == list(range(m.base, m.base + m.n))}
        for p in fv["roots"]:
            name, reg = m.port(p), self.roots.get(m.port(p))
            ok[f"root:{p}"] = (name in self.want and reg is not None and reg["root"] == fv["roots"][p]
                               and reg["domain"] == fv["domain_ids"][p] == self.want[name]["domain"])
        mm = {"k": k, "m": m, "header": header, "circuit": circuit, "circ": circ,
              "st": {"dir": st_dir, "meta": circ.meta, "pin": made["pin"], "partition_file": self.out / "partition.json",
                     "program_file": self.out / "program.json"}}
        A5.stage(self.wdir, mm, self.files)
        return {**mm, "checks": ok, "k_log": circ.meta.get("k_log")}


def _law(law: dict) -> str:
    return f"{law['law']}:{law['k']}"


def session(a, mm: dict, draw: dict, sdir: Path, firewall: bool = True, forged: bool = False,
            lean_pool: ThreadPoolExecutor | None = None) -> dict:
    """One `--zk` session of member ``mm`` served ``draw``, and Lean's verdict on it: with ``lean_pool``, a future of
    it, so that the next session proves while Lean verifies this one."""
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "draw.json").write_text(json.dumps(draw, sort_keys=True))
    env = {"FC_FIREWALL": "1", "FC_FIREWALL_LEAN": a.firewall, **a0.firewall_env(mm["st"])} if firewall else {"FC_FIREWALL": "0"}
    if forged:
        env["FC_ZK_SELF_CHECK"] = "skip"
    record, live = a0.session(a.bin, mm["st"], mm["m"].n, f"subset:{draw['k']}", free_port(), sdir, draw_file=sdir / "draw.json",
                              zk=True, env=env, prove_timeout=a.session_timeout)
    served = (record or {}).get("unit_draw") or {}

    def verify() -> dict:
        lean = a0.lean_verify(a.lean, mm["st"], mm["m"].n, sdir, zk=True)
        return {x: lean.get(x) for x in ("accepted", "why", "verify_s", "partition", "digest", "registered", "line")}

    return {"m0": {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True,
                   "why": ((record or {}).get("verdict") or {}).get("why")},
            "lean": lean_pool.submit(verify) if lean_pool else verify(),
            "served_draw": served.get("units"), "proved": a0.proved_units(mm["header"], live or {}),
            "prover": {"accepted": (live or {}).get("accepted"), "why": (live or {}).get("why"),
                       "stopped": (live or {}).get("prover_stopped")},
            "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "dense_m", "units",
                                                        "proof_bytes")},
            "roots": (live or {}).get("roots") == mm["header"]["frame_v3"]["roots"],
            "session": next((str(f.parent) for f in sorted((sdir / "sessions").glob("*/session.json"))), None)}


def flipped(a, mm: dict, honest: Path, out: Path) -> dict[str, dict]:
    """`zk_mutants.py`'s one-bit proof edits of the session in ``honest`` (the first ``a.flips``, else all), each verified
    by Lean: ``{name: verdict}``."""
    made = out / "mutants"
    ZM.main(honest, made)
    cases = {}
    edits = [c for c in json.loads((made / "zk_mutants.json").read_text()) if c.get("rep") is not None]
    for c in edits[:a.flips] if a.flips else edits:
        name = c["name"]
        case = out / name
        sd = case / "sessions" / honest.name
        shutil.copytree(honest, sd, copy_function=os.link, dirs_exist_ok=True)
        for f in (made / name).iterdir():
            (sd / f.name).unlink(missing_ok=True)
            shutil.copyfile(f, sd / f.name)
        v = a0.lean_verify(a.lean, mm["st"], mm["m"].n, case, zk=True)
        want = c["expect"]["why"]
        cases[name] = {"refused": a0.lean_refused(v), "why": v.get("why"), "expected": want,
                       "reason_as_named": bool(v.get("why")) and v["why"].startswith(want)}
    return cases


def prove(a, jobs: list, tampered: tuple[Path, dict] | None, out: Path) -> tuple[dict, dict, str | None]:
    """Each job ``(key, label, stage, member, session dir)`` proved in its own session, one at a time, with up to
    ``--lean-jobs`` Lean verifies beside the one proving (``stage()`` stages the bundle's member as a5 does); then the
    flipped bits of the session of least k_log, and the tampered arms of ``(tampered window, member)``:
    ``(rows, negatives, the flipped session's key)``."""
    rows, first, proved = {}, None, []
    with ThreadPoolExecutor(max_workers=max(1, a.lean_jobs)) as lean_pool:
        for key, label, stage, bm, sdir in jobs:
            mm = stage()
            proved.append((key, label, bm, mm, session(a, mm, bm["draw"], sdir, lean_pool=lean_pool)))
            print(f"PROVEBUNDLE {label} proved: e2e {proved[-1][-1]['timing']['e2e_s']} s", flush=True)
        for *_, s in proved:
            s["lean"] = s["lean"].result()
    for key, label, bm, mm, s in proved:
        units = sorted(bm["base"] + i for i in bm["local"])
        s["checks_before_draw"] = mm["checks"]
        s["k_log"] = mm["k_log"]
        s["accepted"] = (all(mm["checks"].values()) and s["m0"]["accepted"] and s["lean"]["accepted"] is True and s["roots"]
                         and sorted(s["served_draw"] or []) == sorted(bm["local"]) and sorted(s["proved"]) == units)
        rows[key] = {**s, "units": units}
        if s["session"] and (first is None or mm["k_log"] < first[0]["k_log"]):
            first = (mm, s, key)
        print(f"PROVEBUNDLE {label} units {units}: accepted {s['accepted']} (M0 {s['m0']['accepted']}, Lean "
              f"{s['lean']['accepted']} {s['lean']['why'] or ''}), k_log {mm['k_log']}, e2e {s['timing']['e2e_s']} s, "
              f"Lean verify {s['lean']['verify_s']} s", flush=True)
    negatives: dict[str, dict] = {}
    if first:
        for name, v in flipped(a, first[0], Path(first[1]["session"]), out / "flipped").items():
            negatives[f"flipped/{name}"] = v
        n_ok = sum(v["refused"] for k, v in negatives.items() if k.startswith("flipped/"))
        print(f"PROVEBUNDLE flipped bits ({first[2]}): {n_ok}/{len(negatives)} refused by Lean", flush=True)
    if tampered:
        tw, bm = tampered
        mm = Staged(tw, out / "tampered").member(bm["k"])
        for arm, fw in (("firewalled", True), ("unfirewalled", False)):
            s = session(a, mm, bm["draw"], out / "tampered" / f"session-{arm}", firewall=fw, forged=True)
            # behind the firewall no proof may leave the prover; without it Lean's own verdict must refuse the proof
            refused = (str(s["prover"]["stopped"]).startswith("FIREWALL-REFUSED") and s["lean"]["accepted"] is not True
                       and s["m0"]["accepted"] is not True) if fw else a0.lean_refused(s["lean"])
            negatives[f"tampered/{arm}"] = {"refused": refused, "lean": s["lean"], "m0": s["m0"], "prover": s["prover"],
                                            "checks_before_draw": mm["checks"]}
            print(f"PROVEBUNDLE tampered {arm}: refused {negatives[f'tampered/{arm}']['refused']} (prover "
                  f"{str(s['prover']['stopped'] or s['prover']['why'])[:200]!r}; Lean {str(s['lean']['why'])[:200]!r})",
                  flush=True)
    return rows, negatives, first[2] if first else None


def _finish(summary: dict, out: Path, rows: dict, negatives: dict, flipped_key: str | None, failures: dict) -> int:
    summary.update(members=rows, negatives=negatives, flipped_member=flipped_key, visible_failures=failures)
    ok = (bool(rows) and all(r["accepted"] for r in rows.values()) and bool(negatives)
          and all(v["refused"] for v in negatives.values()) and not failures)
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("PROVEBUNDLE " + json.dumps({"ok": ok, "accepted": {k: r["accepted"] for k, r in rows.items()},
                                       "negatives_refused": sum(v["refused"] for v in negatives.values()),
                                       "negatives": len(negatives), "visible_failures": failures}, sort_keys=True), flush=True)
    return 0 if ok else 1


def _refuse(summary: dict, out: Path, **why) -> int:
    (out / "summary.json").write_text(json.dumps({**summary, "ok": False}, indent=1, sort_keys=True, default=str))
    print("PROVEBUNDLE " + json.dumps({"ok": False, **why}, sort_keys=True, default=str), flush=True)
    return 1


def main_window(a, out: Path) -> int:
    wdir = Path(a.window)
    doc, payload = RB.read(a.bundle)
    held, bad = hold(doc, payload, wdir)
    held.update(held_draw(doc, wdir))
    if a.live:
        held["draw:live"] = doc["draw"].get("source") == "live"
    summary: dict = {"format": FORMAT, "bundle": {"payload_sha512": doc["payload"]["sha512"], "draw": doc["draw"],
                                                  "units": len(doc["units"]),
                                                  "registered_law": json.loads((wdir / "registration.json").read_text())["law"]},
                     "window": {"run": doc["window"]["run"], "registration": doc["window"]["registration"]},
                     "held": held, "bundle_findings": bad}
    if not all(held.values()):
        return _refuse(summary, out, held=held, bundle_findings=bad)
    sv = Staged(wdir, out / "honest")
    members = doc["members"][:a.members] if a.members else doc["members"]
    jobs = [(f"m{bm['k']}", f"member m{bm['k']}", lambda k=bm["k"]: sv.member(k), bm, out / "honest" / f"session-m{bm['k']}")
            for bm in members]
    rows, negatives, fk = prove(a, jobs, (Path(a.tampered), members[0]) if a.tampered else None, out)
    return _finish(summary, out, rows, negatives, fk, {})


def main_step(a, out: Path) -> int:
    """A step tree (``--step``): every drawn block held to the registered step before anything is proved, then each held
    block's one member proved in its own session. A drawn index with no block, or a block its hold refuses, is a visible
    failure: it is not proved and the verdict is not ok (no redraw)."""
    sdir = Path(a.step)
    reg, held, blocks, missing = hold_step(sdir)
    drawn = json.loads((sdir / "draw.json").read_text())
    derived, coin = None, None
    if a.coin:
        got, derived, coin = held_coin(sdir, reg, drawn, Path(a.main) if a.main else None)
        held.update(got)
    print("DRAWN " + json.dumps({"registration": RB.step_digest(reg), "receipt": json.loads((sdir / "receipt.json").read_text()),
                                 "law": reg["law"], "population": reg["population"], "units": drawn.get("units"),
                                 "derived": derived, "coin": coin and {**coin, "held": {k: v for k, v in held.items()
                                                                                        if k.startswith("coin")}},
                                 "blocks": {str(g): RB.block_of(reg, g) for g in drawn.get("units") or []}},
                                sort_keys=True), flush=True)
    for g, b in sorted(blocks.items()):
        print("CHECK " + json.dumps({"block": g, "held": all(b["held"].values()),
                                     "failed": sorted(k for k, v in b["held"].items() if not v), "findings": b["findings"]},
                                    sort_keys=True), flush=True)
    refused = {g: b for g, b in blocks.items() if not all(b["held"].values())}
    failures = {**{str(g): "no block delivered for the drawn index" for g in missing},
                **{str(g): "refused before proving: " + ", ".join(k for k, v in b["held"].items() if not v)
                   for g, b in refused.items()}}
    summary: dict = {"format": FORMAT, "ties": TIES, "held": held, "derived": derived, "coin": coin,
                     "step": {x: reg[x] for x in ("run", "program", "block", "step", "step_root", "slots", "units_per_slot",
                                                  "population", "law", "layout_digest", "context")}
                     | {"registration": RB.step_digest(reg)},
                     "draw": json.loads((sdir / "draw.json").read_text()),
                     "blocks": {str(g): {"slot_unit": RB.block_of(reg, g), "held": b["held"], "findings": b["findings"],
                                         "payload_sha512": b["doc"]["payload"]["sha512"]} for g, b in blocks.items()}}
    print("PROVEBUNDLE step held: " + json.dumps({"held": held, "blocks": {g: all(b["held"].values()) for g, b in blocks.items()},
                                                  "visible_failures": failures}, sort_keys=True), flush=True)
    if not all(held.values()):
        return _refuse(summary, out, held=held, visible_failures=failures)
    jobs, tampered = [], None
    for g in sorted(set(blocks) - set(refused)):
        b = blocks[g]
        bm = b["doc"]["members"][0]
        bdir = out / "blocks" / str(g)
        jobs.append((f"b{g}", f"block {g} (slot, unit {RB.block_of(reg, g)})",
                     lambda w=b["window"], d=bdir, k=bm["k"]: Staged(w, d).member(k), bm, bdir / f"session-m{bm['k']}"))
        tw = Path(a.tampered) if a.tampered else b["dir"] / "tampered"
        if tampered is None and tw.is_dir():
            tampered = (tw, bm)
    rows, negatives, fk = prove(a, jobs, tampered, out)
    for key, r in rows.items():
        print("VERDICT " + json.dumps({"block": int(key[1:]), "accepted": r["accepted"], "k_log": r["k_log"],
                                       "prove_s": r["timing"]["e2e_s"], "verify_s": r["lean"]["verify_s"],
                                       "m0": r["m0"]["accepted"], "lean": r["lean"].get("line")}, sort_keys=True), flush=True)
    return _finish(summary, out, rows, negatives, fk, failures)


def draw_cmd(argv) -> int:
    ap = argparse.ArgumentParser(prog="prove_bundle.py draw", description="The challenger's live draw over a registered step.")
    ap.add_argument("--step", required=True, help="holds step-registration.json; writes receipt.json and draw.json there")
    ap.add_argument("--lean", required=True, help="flock-verify")
    a = ap.parse_args(argv)
    receipt, draw = step_draw(a.lean, Path(a.step))
    print("DRAW " + json.dumps({"receipt": receipt, "draw": draw}, sort_keys=True), flush=True)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["make"]:
        return make(argv[1:])
    if argv[:1] == ["draw"]:
        return draw_cmd(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", help="the bundle's JSON; its payload is the .bin beside it")
    ap.add_argument("--window", help="the prover's window: what Window.write wrote, private files included")
    ap.add_argument("--step", help="a step tree: step-registration.json, receipt.json, draw.json and "
                                   "blocks/<index>/{window/, bundle.{json,bin}, tampered/?}, the draw the step registration's")
    ap.add_argument("--coin", action="store_true",
                    help="with --step: the draw must re-derive from the challenger's live coin (coin.json, `held_coin`)")
    ap.add_argument("--main", help="with --coin: main's first-parent history (git rev-list --first-parent origin/main)")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=None, help="prove only the bundle's first N drawn members")
    ap.add_argument("--tampered", help="a tampered window of the same shape: its first drawn member must not be accepted")
    ap.add_argument("--session-timeout", type=float, default=14400)
    ap.add_argument("--live", action="store_true", help="the bundle's draw must be live, with its receipt")
    ap.add_argument("--flips", type=int, default=0, help="Lean-verify only the first N one-bit proof edits (0: all)")
    ap.add_argument("--lean-jobs", type=int, default=1, help="Lean verifies at once beside the one session proving")
    a = ap.parse_args(argv)
    if bool(a.step) == bool(a.bundle or a.window) or (not a.step and not (a.bundle and a.window)):
        ap.error("--step DIR, or --bundle B.json and --window DIR")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    return main_step(a, out) if a.step else main_window(a, out)


if __name__ == "__main__":
    raise SystemExit(main())
