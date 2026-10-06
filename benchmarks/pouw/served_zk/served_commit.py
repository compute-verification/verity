"""One served call's commitment in the ZK statement's format, from a retained Pearl-C pass, registered and drawn as one-stage
does, and the check that a hidden tile proof's committed rows are that registration's.

  python served_commit.py commit PASS_DIR OUT [--call I]          the committer: the call's fp8-p rows, salted and committed
  python served_commit.py register COMMITMENT OUT                 the committer: the registration record (v1)
  python served_commit.py receive COMMITMENT REGISTRATION OUT     the verifier: R1-R6 against its own copies, and its receipt
  python served_commit.py check REG COMMITMENT DRAW PUB OPENING   the verifier: the proof's rows are the drawn tile's leaves
  python served_commit.py audit REG RECEIPT DRAW STAGE PUB OUT --lean F --opening F [--rust F]   the audit record

- **The call** is one of the pass's ``o_proj``-shaped calls (k = n = `K`), drawn from ``os.urandom`` unless ``--call`` names
  it.  Its A rows (the pass's retained activations, BF16 widened to FP32) and its weight's B rows are formed on the CPU as
  `pearl_c` forms them under the pass's epoch: seeds from the manifest's salt, the call's A root and the weight's root
  (`PearlC.noise`), forming v1 and the peel factors (`fast_tile`'s batch forms, the scalar reference's bytes).
- **The rows** are the statement's: each A row i < real_rows is A′_i ‖ P_A,i, an fp8-p row of role X (``hm96-sha512/row-seg/v1``),
  each B row j < n is B̃_j ‖ P_B,j, of role W (``hm96-sha512/row/v2``), `leaves.words_fp8_p`.  A row whose forming the scheme
  refuses is excluded: its leaf commits a zero row, and the draw skips it.
- **The population** is the call's tiles: `population_program` of `Pc8TileHidden{K,1,1,sm120}` over real_rows x n
  instances under `Q_template_instance`, unit u the tile (u // n, u % n).  A tile of an excluded row is a unit no honest
  proof opens.
- **The commitment** is C-Flock's: each row's b ‖ c is hm96-sha512 over its row digest under a fresh 192-byte salt
  (`leaves.commit`), its leaf `row_leaf` of `leaves.tree_leaf(b ‖ c)`, and each side one frame-v3-sha512 tree under
  one-stage's served domain (`registration.served_domain`: the program, the partition, the side's port and schema, its row
  count and the request's run).  ``commitment.json`` is public: the roots, domains and counts, with the served -h2
  commitment it sits beside and the pass it came from.  ``prover.npz`` holds the rows and salts, the prover's alone.
- **The registration** is `verity/registration/v1` with one root per side (ports ``a`` and ``b``), the law ``subset:1`` and
  ``window.kind = "served"``.  The verifier checks it against its own program, query, law, scheme and the ports it derives
  (`receive`), writes its receipt, and only then draws: Lean ``flock-verify draw --population N --k 1``, from its own
  randomness.  Roots and counts are public here, a development shortcut on test prompts.
- **The check** reads the proof's public file (``pub.bin``, `verity_flock.circuit.write`): instance 0's b ‖ c of its A row
  and B row, the first two rows, must be the leaves of the drawn tile, each by its frame-v3 path to the registered root
  (``opening.json``, the prover's).  A proof whose A or B row is any other row, or whose rows another registration holds, is
  refused here; a proof whose opened row is not its committed row is refused by the proof's own verifier.
- **The audit record** is one-stage's (`verity/one-stage/audit/v0`): the registration, receipt and draw, the proved unit,
  the statement's digests, and the verdicts: Lean's (of record), the opening check's and Rust's beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from verity.primitives.commitments.identity import identity_digest_sha512
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.primitives.commitments.rowleaf import ROLE_W, ROLE_X, row_leaf
from verity.primitives.circuits.codec import canonical_json
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.circuit import pc8
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes import pearl_kw as P
from verity.protocols.verification.sampled_proofs.one_stage import audit as A
from verity.protocols.verification.sampled_proofs.one_stage import partition as PT
from verity.protocols.verification.sampled_proofs.one_stage import registration as R

SCHEMA = "verity-pouw/served-zk-commitment/v1"
#: the registered scheme: hm96-sha512 row leaves under frame-v3-sha512 trees, one per side
SCHEME = "verity/pouw/served-zk/hm96-sha512+frame-v3-sha512/v0"
LAW = {"law": "subset", "k": 1}
VERIFIER = "served-zk/lean-flock-verify"
#: the frame's owner, as C-Flock's statement domains have it (`verity_flock.ir_frame.OWNER`)
OWNER = -1
SALT_BYTES = 192
K = 4096
TILE = (1, 1)
DEV = "sm120"
SIDES = {"a": ROLE_X, "b": ROLE_W}
#: the rows a forming batch takes at once
CHUNK = 256
#: the audit's delta (its profile's bound on wrong units holds except with this probability)
DELTA = 0.05


def population(k: int, rows: int, n: int):
    """The call's tiles as one-stage's population: (program, query, partition object)."""
    d = pc8.templates(k, TILE, DEV, hidden=True)["tile"]
    prog = PT.population_program(d, rows * n)
    q = PT.template_instance_query(PT.template_id(d))
    return prog, q, PT.partition(prog, q)


def statement_of(k: int, rows: int, n: int, run: str) -> dict:
    """What the served domains bind: the population's program and partition digests, its size and the request's run."""
    _prog, _q, part = population(k, rows, n)
    return {"program": part["program"], "partition": PT.digest(part), "population": rows * n, "tiles": [rows, n], "run": run}


def run_of(binding: dict) -> str:
    """The request's run, as its served domains bind it: the serving run, the pass's manifest and the call."""
    return f"{binding['run']}/{binding['manifest_sha256'][:16]}/matmul-{binding['matmul']}"


def tile_of(rows: int, n: int, unit: int) -> tuple[int, int]:
    if not 0 <= unit < rows * n:
        raise ValueError(f"unit {unit} is not a tile of a {rows} x {n} call")
    return unit // n, unit % n


def f32_rows(data, k: int, idx) -> list[bytes]:
    """Rows `idx` of a BF16 row file as the committed FP32 words (BF16 widened, exact)."""
    out = []
    for i in idx:
        out.append((np.frombuffer(data[2 * k * i:2 * k * (i + 1)], "<u2").astype("<u4") << 16).tobytes())
    return out


def _form(scheme: C.PearlC, noise: C.Noise, k: int, side: str, rows: list[bytes], idx: list[int]) -> list[list[int]]:
    """The fp8-p words of rows `idx` (FP32 bytes `rows`) on one side, as `PearlC.encode` forms them; raises `P.Reject`."""
    seed_e = noise.seed_a if side == "a" else noise.seed_b
    f, e = scheme._side(rows, idx, 0 if side == "a" else 1, seed_e, noise.seed_b, k)
    peel = C.peel_factors_a if side == "a" else C.peel_factors_b
    p = peel(f.codes, C.scaled_noise(f, e), noise.seed_b, k, scheme.line_norm, scheme.device)
    ok = [P.f32_of_bits(r) * P.f32_of_bits(a) >= P.SIGMA_MIN and C.live_row(x) for r, a, x in zip(f.rho, f.alpha, f.x)]
    return [L.words_fp8_p(c, q) if good else None for c, q, good in zip(f.codes, p, ok)]


def form(scheme: C.PearlC, noise: C.Noise, k: int, side: str, rows: list[bytes], idx: list[int]) -> list[list[int] | None]:
    """Each row's fp8-p words, or None for a row the scheme refuses (alone, so one refusal leaves its batch's others)."""
    out: list[list[int] | None] = []
    for at in range(0, len(idx), CHUNK):
        r, i = rows[at:at + CHUNK], idx[at:at + CHUNK]
        try:
            out += _form(scheme, noise, k, side, r, i)
        except P.Reject:
            for one, j in zip(r, i):
                try:
                    out += _form(scheme, noise, k, side, [one], [j])
                except P.Reject:
                    out.append(None)
    return out


def domain(st: dict, side: str, count: int) -> tuple[CommitmentDomain, str]:
    """Side `side`'s frame-v3-sha512 domain over `count` rows: one-stage's served domain (`registration.served_domain`) of
    the statement's program, partition and run, the side's port and schema."""
    schema = L.ROW_SCHEMA["fp8-p", SIDES[side]]
    b = identity_digest_sha512(R.SERVED_DOMAIN, {"program": st["program"], "partition": st["partition"], "port": side,
                                                 "schema": schema, "leaves": int(count), "run": st["run"]})
    return CommitmentDomain(b, OWNER, RangeIndexedDomain(0, count), hash="sha512"), schema


def leaves_of(dom: CommitmentDomain, schema: str, bcs: list[bytes]) -> list[bytes]:
    return [row_leaf(dom, i, L.tree_leaf(bc), schema) for i, bc in enumerate(bcs)]


def path(levels, index: int) -> list[str]:
    out = []
    for level in levels[:-1]:
        out.append(level[index ^ 1].hex())
        index >>= 1
    return out


def verify_path(dom: CommitmentDomain, root: bytes, index: int, leaf: bytes, sibs: list[bytes]) -> bool:
    if not 0 <= index < dom.count or len(sibs) != merkle_depth(dom.count):
        return False
    digest = leaf
    for level, s in enumerate(sibs):
        digest = dom.node(level, index >> 1, *((digest, s) if index % 2 == 0 else (s, digest)))
        index >>= 1
    return digest == root


def commit_words(st: dict, words: dict[str, list[list[int] | None]], k: int, salts: dict | None = None) -> tuple[dict, dict]:
    """The public commitment's sides and the prover's package: per side its rows' words (an excluded row zero), salts
    (fresh from the OS unless given), b ‖ c and tree levels, under the served domains of `st` (`statement_of`)."""
    zero = [0] * L.ROW_WORDS["fp8-p"](k)
    pub, priv = {}, {}
    for side, ws in words.items():
        dom, schema = domain(st, side, len(ws))
        s = salts[side] if salts else [os.urandom(SALT_BYTES) for _ in ws]
        bcs = [L.commit("fp8-p", SIDES[side], w if w is not None else zero, salt) for w, salt in zip(ws, s)]
        levels = tree_levels(dom, leaves_of(dom, schema, bcs))
        pub[side] = {"schema": schema, "rows": len(ws), "binding": dom.binding.hex(), "domain_id": dom.domain_id.hex(),
                     "root": levels[-1][0].hex(), "excluded": [i for i, w in enumerate(ws) if w is None]}
        priv[side] = {"words": np.array([w if w is not None else zero for w in ws], dtype=np.uint16), "salts": s, "bcs": bcs}
    return pub, priv


def opening(commitment: dict, side: str, bcs: list[bytes], index: int) -> dict:
    """The prover's frame-v3 path for row `index` of one side, from every row's b ‖ c."""
    dom = side_domain(commitment, side)
    return {"index": index, "path": path(tree_levels(dom, leaves_of(dom, commitment["sides"][side]["schema"], bcs)), index)}


def save(path: Path, priv: dict) -> None:
    np.savez(path, **{f"{s}_words": p["words"] for s, p in priv.items()},
             **{f"{s}_salts": np.frombuffer(b"".join(p["salts"]), np.uint8).reshape(-1, SALT_BYTES) for s, p in priv.items()},
             **{f"{s}_bcs": np.frombuffer(b"".join(p["bcs"]), np.uint8).reshape(-1, 128) for s, p in priv.items()})


def load(path: Path) -> dict:
    z = np.load(path)
    return {s: {"words": z[f"{s}_words"], "salts": [r.tobytes() for r in z[f"{s}_salts"]], "bcs": [r.tobytes() for r in z[f"{s}_bcs"]]}
            for s in SIDES}


def side_domain(c: dict, side: str) -> CommitmentDomain:
    s = c["sides"][side]
    dom, schema = domain(c["statement"], side, s["rows"])
    if dom.domain_id.hex() != s["domain_id"] or schema != s["schema"]:
        raise ValueError(f"side {side}: its domain is not the one its statement names")
    return dom


def drawn_tile(registration: dict, draw: dict) -> tuple[int, int]:
    """The one drawn unit's tile, over the registered sides' row counts."""
    leaves = {r["port"]: r["leaves"] for r in registration["roots"]}
    if len(draw.get("units", ())) != 1:
        raise ValueError(f"the draw has {len(draw.get('units', ()))} units, not one")
    return tile_of(leaves["a"], leaves["b"], draw["units"][0])


def check(registration: dict, commitment: dict, draw: dict, pub: bytes, opening: dict) -> str | None:
    """None if the proof's public file opens the drawn tile's rows under the registered roots, else why not.  Each side's
    domain is the served domain of the registration's program, partition and run; the commitment supplies only its
    schemas and excluded rows."""
    try:
        at = dict(zip(("a", "b"), drawn_tile(registration, draw)))
    except (ValueError, KeyError) as e:
        return f"the draw: {e}"
    head, body = pub.split(b"\n", 1)
    h = json.loads(head)
    if h.get("instances") != 1:
        return f"the proof has {h.get('instances')} instances, not one tile"
    schemas = {s: L.ROW_SCHEMA["fp8-p", SIDES[s]] for s in SIDES}
    ports = list(h["frame_v3"]["schemas"].items())
    if [s for _p, s in ports[:2]] != [schemas[s] for s in ("a", "b")]:
        return f"the proof's first rows are {ports[:2]}, not an A row and a B row of the registered schemas"
    bc = {"a": body[0:128], "b": body[128:256]}
    roots = {r["port"]: r for r in registration["roots"]}
    st = {"program": registration["program"], "partition": registration["partition"], "run": registration["window"]["run"]}
    for side in ("a", "b"):
        dom, schema = domain(st, side, roots[side]["leaves"])
        if dom.domain_id.hex() != roots[side]["domain"]:
            return f"side {side}: the registered domain is not the served domain of the registration's run"
        if at[side] in commitment["sides"][side]["excluded"]:
            return f"side {side}: row {at[side]} is excluded"
        o = opening[side]
        if o["index"] != at[side]:
            return f"side {side}: the opening is of row {o['index']}, not the drawn row {at[side]}"
        leaf = row_leaf(dom, at[side], L.tree_leaf(bc[side]), schema)
        if not verify_path(dom, bytes.fromhex(roots[side]["root"]), at[side], leaf, [bytes.fromhex(x) for x in o["path"]]):
            return f"side {side}: the proof's committed row is not the registered row {at[side]}"
    return None


def registration_of(c: dict) -> dict:
    """The committer's registration record of commitment `c`: one root per side, the law subset:1, a served window."""
    prog, q, part = population(c["binding"]["shape"][1], *c["statement"]["tiles"])
    roots = [{"port": s, "root": c["sides"][s]["root"], "leaves": c["sides"][s]["rows"], "domain": c["sides"][s]["domain_id"]}
             for s in SIDES]
    b = c["binding"]
    window = {"kind": "served", "run": c["statement"]["run"], "served_run": b["run"], "manifest_sha256": b["manifest_sha256"],
              "scheme": b["scheme"], "matmul": b["matmul"], "shape": b["shape"], "weight": b["weight"],
              "served": c["served"]}
    return R.record(prog, q, LAW, SCHEME, roots, window, partition=part)


def receive(c: dict, rec: dict) -> tuple[dict, dict | None]:
    """The verifier's R1-R6 on `rec` against its own copies: the program and query of the call's tiles (its shape and real
    rows, public in this prototype), its law and scheme, and the ports it derives (each side's row count and served domain
    under its own program, partition and the run of the request it audits, `run_of` the served request's binding, never
    the record's own window).  The receipt when they hold, else None."""
    prog, q, part = population(c["binding"]["shape"][1], *c["statement"]["tiles"])
    st = {"program": part["program"], "partition": PT.digest(part), "run": run_of(c["binding"])}
    rows = dict(zip(("a", "b"), c["statement"]["tiles"]))
    ports = {s: {"leaves": rows[s], "domain": domain(st, s, rows[s])[0].domain_id.hex()} for s in SIDES}
    v = R.check(rec, prog, q, LAW, SCHEME, ports)
    return v.to_json(), (R.receipt(rec, VERIFIER) if v.ok else None)


def lean_verdict(path: Path) -> dict:
    """The Lean verifier's verdict from its output (`VERDICT {…}`), of record; accepted null when it printed none."""
    try:
        lines = [json.loads(x[8:]) for x in path.read_text().splitlines() if x.startswith("VERDICT ")]
    except (OSError, ValueError):
        lines = []
    if not lines:
        return {"of_record": True, "accepted": None, "verifier": "lean flock-verify verify --zk"}
    v = lines[0]
    return {"of_record": True, "accepted": bool(v.get("accepted")), "why": v.get("why", ""), "verifier": "lean flock-verify verify --zk",
            "digest": hashlib.sha512(canonical_json(v)).hexdigest()}


def rust_verdict(path: Path | None) -> dict | None:
    """The Rust prover-side verifier's verdict of the --zk session (`LIVE\\t{…}` lines), beside the verifier of record."""
    if path is None:
        return None
    try:
        lines = [json.loads(x[5:]) for x in path.read_text().splitlines() if x.startswith("LIVE\t")]
    except (OSError, ValueError):
        lines = []
    v = lines[0].get("verdict", lines[0]) if lines else None
    return {"of_record": False, "accepted": None if v is None else bool(v.get("accepted")), "why": (v or {}).get("why", ""),
            "verifier": "rust flock-circuit --zk (live)"}


def audit(registration: dict, receipt: dict, draw: dict, stage: dict, pub: bytes, verdicts: dict) -> dict:
    """One-stage's audit record of the drawn tile's session: the staged unit is the drawn unit, in one instance."""
    statement = {"definition": stage["definition"], "circuit_sha512": stage["circuit_sha512"], "class_sha512": stage["class_sha512"],
                 "public_sha512": hashlib.sha512(pub).hexdigest(), "ands": stage["ands"], "k_log": stage["k_log"],
                 "binding": stage["binding"], "tile": list(drawn_tile(registration, draw))}
    return A.audit_record(registration, receipt, draw, registration["law"], list(draw["units"]), 1, statement,
                          {k: v for k, v in verdicts.items() if v is not None}, DELTA)


# ---- the CLI ------------------------------------------------------------------------------------------------------------

def cmd_commit(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c_vllm"))
    import fast_tile as FT
    from verity_vllm.protocol_options import pouw_pearl_c_device as PD

    pass_dir = Path(a.pass_dir)
    raw = (pass_dir / "manifest.json").read_bytes()
    man = json.loads(raw)
    if man.get("schema") != PD.MANIFEST_SCHEMA:
        raise SystemExit(f"{pass_dir}: not a retained pass ({man.get('schema')!r})")
    scheme = PD.scheme_of(man["scheme"])
    calls = [u for u in man["matmuls"] if u["k"] == K and u["n"] == K]
    if not calls:
        raise SystemExit(f"{pass_dir}: no call of k = n = {K}")
    source = os.urandom(32)
    u = next(x for x in calls if x["index"] == a.call) if a.call is not None else \
        calls[int.from_bytes(hashlib.sha256(source).digest()[:8], "little") % len(calls)]
    w = man["weights"][str(u["weight"])]
    k, n, real = u["k"], u["n"], u["real_rows"]
    ctx = Context(bytes.fromhex(man["salt"]), Matmul(u["index"], Shape(u["m"], k, n), u["weight"]), bytes.fromhex(u["root_a"]),
                  bytes.fromhex(w["root"]))
    noise = scheme.noise(ctx)
    undo = FT.install(scheme)
    try:
        a_data = (pass_dir / u["rows"]["path"]).read_bytes()
        words = {"a": form(scheme, noise, k, "a", f32_rows(a_data, k, range(real)), list(range(real)))}
        b_data = np.memmap(pass_dir / w["rows"]["path"], dtype=np.uint8, mode="r")
        words["b"] = form(scheme, noise, k, "b", f32_rows(b_data, k, range(n)), list(range(n)))
    finally:
        undo()
    sha = hashlib.sha256(raw).hexdigest()
    binding = {"run": man.get("run"), "manifest_sha256": sha, "scheme": man["scheme"],
               "matmul": u["index"], "shape": [u["m"], k, n], "weight": u["weight"], "real_rows": real}
    st = statement_of(k, real, n, run_of(binding))
    sides, priv = commit_words(st, words, k)
    c = {"schema": SCHEMA, "binding": binding, "statement": st, "sides": sides,
         "served": {"transcript_root": man.get("transcript_root"), "commitment": man.get("commitment"),
                    "records_root": man.get("records_root"), "root_a": u["root_a"], "weight_root": w["root"]},
         "call_draw": {"source": source.hex(), "among": [x["index"] for x in calls]} if a.call is None else {"given": a.call}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "commitment.json").write_text(json.dumps(c, sort_keys=True))
    save(out / "prover.npz", priv)
    print(json.dumps({"call": u["index"], "rows": {s: v["rows"] for s, v in sides.items()},
                      "excluded": {s: len(v["excluded"]) for s, v in sides.items()},
                      "roots": {s: v["root"][:16] for s, v in sides.items()}}), flush=True)
    return 0


def _json(p) -> dict:
    return json.loads(Path(p).read_text())


def cmd_register(a) -> int:
    rec = registration_of(_json(a.commitment))
    Path(a.out).write_text(json.dumps(rec, sort_keys=True))
    print(json.dumps({"registration": R.digest(rec)[:32], "population": rec["population"], "law": rec["law"],
                      "roots": {r["port"]: r["root"][:16] for r in rec["roots"]}}), flush=True)
    return 0


def cmd_receive(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v, receipt = receive(_json(a.commitment), _json(a.registration))
    (out / "check.json").write_text(json.dumps(v, sort_keys=True))
    if receipt is not None:
        (out / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    print(f"{'RECEIVED' if receipt else 'REFUSED'} {json.dumps(v)}", flush=True)
    return 0 if (receipt is not None) != a.expect_refused else 1


def cmd_check(a) -> int:
    reg, c, d = _json(a.registration), _json(a.commitment), _json(a.draw)
    why = check(reg, c, d, Path(a.pub).read_bytes(), _json(a.opening))
    if a.out:
        Path(a.out).write_text(json.dumps({"of_record": False, "accepted": why is None, "why": why or "",
                                           "verifier": "served_zk opening check"}, sort_keys=True))
    print(f"ACCEPT {a.pub} opens unit {d['units']} of {a.registration}" if why is None else f"REJECT {a.pub}: {why}", flush=True)
    return 0 if (why is None) != a.expect_reject else 1


def cmd_audit(a) -> int:
    reg, receipt, d = _json(a.registration), _json(a.receipt), _json(a.draw)
    verdicts = {"lean": lean_verdict(Path(a.lean)), "opening": _json(a.opening),
                "rust": rust_verdict(Path(a.rust) if a.rust else None)}
    rec = audit(reg, receipt, d, _json(a.stage), Path(a.pub).read_bytes(), verdicts)
    Path(a.out).write_text(json.dumps(rec, sort_keys=True))
    print(f"AUDIT {'ACCEPT' if rec['accepted'] else 'REJECT'} {json.dumps({k: v.get('accepted') for k, v in rec['verdicts'].items()})} "
          f"checks {json.dumps({k: v['ok'] for k, v in rec['checks'].items()})}", flush=True)
    return 0 if rec["accepted"] != a.expect_reject else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("commit")
    p.add_argument("pass_dir")
    p.add_argument("out")
    p.add_argument("--call", type=int)
    p = sub.add_parser("register")
    p.add_argument("commitment")
    p.add_argument("out")
    p = sub.add_parser("receive")
    p.add_argument("commitment")
    p.add_argument("registration")
    p.add_argument("out")
    p.add_argument("--expect-refused", action="store_true")
    p = sub.add_parser("check")
    for x in ("registration", "commitment", "draw", "pub", "opening"):
        p.add_argument(x)
    p.add_argument("--out", help="the opening check's verdict, for the audit record")
    p.add_argument("--expect-reject", action="store_true")
    p = sub.add_parser("audit")
    for x in ("registration", "receipt", "draw", "stage", "pub", "out"):
        p.add_argument(x)
    p.add_argument("--lean", required=True)
    p.add_argument("--opening", required=True)
    p.add_argument("--rust")
    p.add_argument("--expect-reject", action="store_true")
    a = ap.parse_args(argv)
    return {"commit": cmd_commit, "register": cmd_register, "receive": cmd_receive, "check": cmd_check,
            "audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
