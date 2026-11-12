"""A second stage's member restaged at its sampled proof blocks alone, and its session: a measurement of whether Flock's
prover and Lean's session verifier take a statement that holds only the units its session proves.

`two_stage.py second` stages each member at every slot of its part: one table row and one path a slot at each port, of
which its session proves the sampled ones. From a finished `second`'s files (its audit directory's member: the public
file, the prover's file, the circuit, the verifier's registered reads and the session's sample) `restage` makes the
member's statement at some of its units: each port's table the rows those units read, each with its path to the
registered root, the instances' refs into them, the header's `frame_v3` roots those of the smaller tables, and its
`units.indices` those units' own. `probe` restages a member twice, runs one `--zk` session on each and takes Lean's
verdicts (`a0.lean_verify`):

* `drawn`: the units its session proved and no other, the session's sample every unit of the statement;
* `drawn-and-one`: those and one unit more, the session's sample the same units, so a verdict that differs between
  the two is the sample's doing (every unit sampled) and not the smaller tables'.

Each session is verified under three sets of registered reads: the verifier's own for the member (every slot's), those
with every unproved slot's read left out, and those with one proved slot's read moved to another row, which Lean must
refuse. A member holds when its `drawn` statement's session is accepted under the first two and refused under the third.
The sessions' coins are their servers' own.

No rule changes here: round 2 holds a statement's header to every slot of its part (`second-population`), so the second
stage refuses a statement staged so, and no outcome reads this script's verdicts.

    python two_stage_drawn.py --audit SECOND/audit --members 0,19 --bin flock-circuit --lean flock-verify --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a0
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, tree_levels
from verity.commitments.rowleaf import row_leaf
from verity.commitments.windows import flock_inputs as FI, rows as CS


def table_root(binding: str, schema: str, bcs: list[bytes]) -> tuple[str, str]:
    """The frame-v3 root and domain id of a table of rows' ``b || c``, as a staged header's ``frame_v3`` names them: the
    tree of the table's rows under the port's binding, row r its leaf r."""
    dom = CommitmentDomain(bytes.fromhex(binding), CS.OWNER, RangeIndexedDomain(0, len(bcs)), hash="sha512")
    leaves = [row_leaf(dom, r, CS.INST.tree_leaf(CS.INST.default_key, bc), schema) for r, bc in enumerate(bcs)]
    return tree_levels(dom, leaves)[-1][0].hex(), dom.domain_id.hex()


def split(inst: bytes, pub: bytes) -> tuple[dict, bytes, bytes]:
    """A prover's file's header, private body and public body, the last held to the public file's."""
    head, rest = inst.split(b"\n", 1)
    pub_head, body = pub.split(b"\n", 1)
    header = json.loads(head)
    if dict(header, rows=False) != json.loads(pub_head):
        raise ValueError("the prover's file is not under the public file's header")
    if not rest.endswith(body):
        raise ValueError("the prover's file does not end with the public file's body")
    return header, rest[:len(rest) - len(body)], body


def restage(header: dict, private: bytes, body: bytes, ports: list[tuple[str, int]], units_per_block: int,
            local: list[int]) -> tuple[dict, bytes, bytes, dict]:
    """A shared-row statement (its ``header``, its prover's ``private`` body and its public ``body``, over the circuit's
    row ``ports``, each with its words) at its instances ``local`` alone, ascending: -> its header, private body and
    public body, and what was held of the statement it was made from (each port's root recomputed from its table)."""
    n, shared, fv = int(header["instances"]), header["shared_rows"], header["frame_v3"]
    if sorted(shared) != sorted(p for p, _ in ports):
        raise ValueError(f"shared_rows names {sorted(shared)}, not the circuit's rows {[p for p, _ in ports]}")
    if local != sorted(set(local)) or not local or local[0] < 0 or local[-1] >= n:
        raise ValueError(f"{local} are not ascending instances of the statement's {n}")
    starts, at = {}, 0
    for p, _ in ports:
        starts[p], at = at, at + int(shared[p]) * CS.BC_BYTES
    if len(body) != at + 4 * n * len(ports):
        raise ValueError(f"a public body of {len(body)} bytes, not its tables' and refs' {at + 4 * n * len(ports)}")
    refs = np.frombuffer(body, dtype="<u4", offset=at).reshape(n, len(ports))
    want = sum(int(shared[p]) * (2 * w + CS.SALT_BYTES) for p, w in ports)
    if len(private) != want:
        raise ValueError(f"a private body of {len(private)} bytes, not its tables' rows and salts' {want}")
    rows_at, salts_at, at = {}, {}, 0
    for p, w in ports:
        rows_at[p], at = at, at + 2 * w * int(shared[p])
    for p, _ in ports:
        salts_at[p], at = at, at + CS.SALT_BYTES * int(shared[p])

    held, tables, new_refs, new_rows, new_salts = {}, [], np.zeros((len(local), len(ports)), dtype="<u4"), [], []
    new_shared, new_roots, new_domains, new_registered = {}, {}, {}, {}
    for j, (p, w) in enumerate(ports):
        bcs = [body[starts[p] + CS.BC_BYTES * r:starts[p] + CS.BC_BYTES * (r + 1)] for r in range(int(shared[p]))]
        root, domain = table_root(fv["bindings"][p], fv["schemas"][p], bcs)
        held[p] = {"root": root == fv["roots"][p], "domain": domain == fv["domain_ids"][p], "rows": int(shared[p])}
        need = sorted({int(r) for r in refs[local, j]})
        rank = {r: i for i, r in enumerate(need)}
        new_refs[:, j] = [rank[int(r)] for r in refs[local, j]]
        tables.append(b"".join(bcs[r] for r in need))
        new_rows.append(b"".join(private[rows_at[p] + 2 * w * r:rows_at[p] + 2 * w * (r + 1)] for r in need))
        new_salts.append(b"".join(private[salts_at[p] + CS.SALT_BYTES * r:salts_at[p] + CS.SALT_BYTES * (r + 1)] for r in need))
        new_shared[p] = len(need)
        new_roots[p], new_domains[p] = table_root(fv["bindings"][p], fv["schemas"][p], [bcs[r] for r in need])
        e = (header.get("registered") or {}).get(p)
        if e is not None:
            new_registered[p] = {**e, "positions": [e["positions"][r] for r in need], "paths": [e["paths"][r] for r in need]}
    idx = [header["units"]["indices"][u] for u in local]
    units = {**header["units"], "indices": idx, "blocks": [idx[b:b + units_per_block] for b in range(0, len(idx), units_per_block)],
             "classes": [header["units"]["classes"][u] for u in local]}
    new = {**header, "instances": len(local), "range": [0, len(local)], "units": units, "shared_rows": new_shared,
           "frame_v3": {**fv, "roots": {**fv["roots"], **new_roots}, "domain_ids": {**fv["domain_ids"], **new_domains}}}
    if "registered" in header:
        new["registered"] = new_registered
    return new, b"".join(new_rows) + b"".join(new_salts), b"".join(tables) + new_refs.tobytes(), held


def write(d: Path, header: dict, private: bytes, body: bytes) -> None:
    """A statement's public file and its prover's file under ``d``, as `flock_inputs.flock_input_files` writes them."""
    n = int(header["instances"])
    line = json.dumps(dict(header, rows=False), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    (d / f"pub-{n}.bin").write_bytes(line.encode() + b"\n" + body)
    fd = os.open(d / f"inst-{n}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(json.dumps(dict(header, rows=True), sort_keys=True).encode() + b"\n" + private + body)


def sparse(reads: dict, units: list[int]) -> dict:
    """Registered reads ``reads`` with every slot's read but ``units``' left out (null)."""
    def port(e: dict) -> dict:
        first = int(e.get("reads_from") or 0)
        return {**e, "reads": [r if first + k in units else None for k, r in enumerate(e["reads"])]}
    return {**reads, "ports": {p: port(e) for p, e in reads["ports"].items()}}


def moved(reads: dict, unit: int) -> dict | None:
    """Registered reads ``reads`` with ``unit``'s read at the first port of more than one leaf moved to the next row."""
    for p, e in reads["ports"].items():
        k = unit - int(e.get("reads_from") or 0)
        if int(e["leaves"]) > 1 and 0 <= k < len(e["reads"]) and e["reads"][k] is not None:
            rs = list(e["reads"])
            rs[k] = (rs[k] + 1) % int(e["leaves"])
            return {**reads, "ports": {**reads["ports"], p: {**e, "reads": rs}}}
    return None


def free_port() -> int:
    """A loopback port free now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def probe(a: argparse.Namespace, k: int, name: str, spare: bool) -> dict:
    """Member ``k`` of the audit restaged at its session's proved units (``spare``: and its first unit that is none of
    them) under ``a.out/m<k>/<name>``, one session that samples the proved units, and Lean's verdicts."""
    src, out = a.audit / f"m{k}", a.out / f"m{k}" / name
    n = next(int(f.name[4:-4]) for f in src.glob("pub-*.bin"))
    proved = sorted(json.loads((a.audit / f"session-m{k}" / "draw.json").read_text())["units"])
    more = [u for u in range(n) if u not in proved][:1] if spare else []
    local = sorted(proved + more)
    circ = FI.flock_circuit((src / "circuit.txt").read_bytes())
    ports = [(str(p[0]), int(p[1])) for p in circ.meta["ports"]]
    header, private, body = split((src / f"inst-{n}.bin").read_bytes(), (src / f"pub-{n}.bin").read_bytes())
    new, new_private, new_body, held = restage(header, private, body, ports, int(circ.meta["vus_per_block"]), local)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "circuit.txt", out / "circuit.txt")
    if (src / "tables").is_dir():
        shutil.copytree(src / "tables", out / "tables")
    write(out, new, new_private, new_body)
    kk, reads = len(local), json.loads((src / "registered.json").read_text())
    sampled = [local.index(u) for u in proved]
    units = [new["units"]["indices"][i] for i in sampled]
    got = {"member": k, "form": name, "template": header["subcircuit"], "instances": {"staged": n, "restaged": kk},
           "units": new["units"]["indices"], "sampled": sampled, "held": held,
           "rows": {"staged": header["shared_rows"], "restaged": new["shared_rows"]},
           "bytes": {"public": {"staged": (src / f"pub-{n}.bin").stat().st_size, "restaged": (out / f"pub-{kk}.bin").stat().st_size},
                     "prover": {"staged": (src / f"inst-{n}.bin").stat().st_size, "restaged": (out / f"inst-{kk}.bin").stat().st_size}}}
    st = {"dir": out, "pin": header["circuit_sha512"], "partition_file": a.audit / "partition.json",
          "program_file": a.audit / "program.json"}
    sdir = out / "session"
    sdir.mkdir()
    (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": kk, "k": len(sampled), "units": sampled}, sort_keys=True))
    record, live = a0.session(a.bin, st, kk, f"subset:{len(sampled)}", free_port(), sdir, draw_file=sdir / "draw.json", zk=True,
                              env=a0.prover_env(None, st), prove_timeout=a.session_timeout)
    got["session"] = {"recorded": record is not None, "live": {x: (live or {}).get(x) for x in ("accepted", "why")},
                      "unit_draw": (record or {}).get("unit_draw")}
    cases = {"every-slot": reads, "proved-slots-only": sparse(reads, units), "a-proved-slot-at-another-row": moved(reads, units[0])}
    got["lean"] = {}
    for case, r in cases.items():
        if r is None:
            continue
        alt = {**st, "registered_file": out / f"registered-{case}.json"}
        alt["registered_file"].write_text(json.dumps(r, sort_keys=True))
        v = a0.lean_verify(a.lean, alt, kk, sdir, zk=True, log=f"lean-{case}")
        part = v.get("partition")
        got["lean"][case] = {"accepted": v["accepted"], "why": v.get("why"), "unit_draw": v.get("unit_draw"),
                             "partition": part.get("note") if isinstance(part, dict) else None}
    lean = got["lean"]
    got["ok"] = bool(all(h["root"] and h["domain"] for h in held.values()) and lean["every-slot"]["accepted"]
                     and lean["proved-slots-only"]["accepted"]
                     and not lean.get("a-proved-slot-at-another-row", {"accepted": False})["accepted"])
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--audit", type=Path, required=True, help="a finished second stage's audit directory")
    ap.add_argument("--members", default="0", help="the members to restage, by their place in the window (comma-separated)")
    ap.add_argument("--bin", required=True, help="flock-circuit")
    ap.add_argument("--lean", required=True, help="flock-verify")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session-timeout", type=float, default=1800)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for k in (int(x) for x in a.members.split(",")):
        for name, spare in (("drawn", False), ("drawn-and-one", True)):
            try:
                got = probe(a, k, name, spare)
            except Exception as e:  # noqa: BLE001  (one statement's failure is its result; the next is still measured)
                got = {"member": k, "form": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-2000:]}
            results.append(got)
            print("DRAWN " + json.dumps(got, sort_keys=True, default=str), flush=True)
    (a.out / "drawn.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    held = all(r["ok"] for r in results if r["form"] == "drawn")
    print("DRAWN-HELD " + json.dumps(held), flush=True)
    return 0 if held else 1


if __name__ == "__main__":
    sys.exit(main())
