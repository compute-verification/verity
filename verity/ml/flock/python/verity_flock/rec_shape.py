"""Outer proofs of one fixed shape (lane outer-shape): every inner statement of a deployment padded to a public cap, so that V*'s
statements over any of its sessions are the cap's, whatever the inner statement's real instance count.

V*'s statements over one inner session (`rec_vstage`) are a function of the inner table's shape alone: its m, k_log and claims
(`rec_vstage.shape`) fix the fast100 schedule (`rec_algebra.fast100`), so each level's RecOpen template (its LANES and H) and
its openings (each rep's queries of the level, positions from the coins, never deduplicated), and the algebra's parts
(`rec_residuals.parts`). A deployment's `Cap` names the inner circuit's k_log and vus_per_block (its META), the instances every
inner statement is padded to (`class_statement --real`: the real tiles, then fillers) and the claims (2 + the circuit's extra
claims), all public; `outer` is the statements every outer proof of the deployment has, and `check` refuses a session of
another shape before V* is staged over it (``rec_vstage stage --cap``, as the ZK server would).

A filler only adds instances: a tile whose rows are zero and whose output is the class's own on them, so the padded statement
holds exactly when the unpadded one does and its fillers do, and the fillers hold of any table. `forge_filler` is the negative
control: one filler's output row one bit off in the inner prover's file, a session V*'s algebra must refuse.

    python -m verity_flock.rec_shape outer CAP                         # the cap's statements (name, template digest, instances)
    python -m verity_flock.rec_shape forge-filler DIR N REAL [--slot S]  # DIR/inst-N.bin with filler S's out row forged

``CAP`` is ``instances=N,claims=C,k_log=K,vus_per_block=V`` (`Cap.parse`).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from . import rec_algebra as A
from . import rec_open as R
from . import rec_residuals as RR
from . import rec_vstage as S

REPS = 2


class Refused(ValueError):
    """An inner session whose V* would not have the cap's shape."""


@dataclass(frozen=True)
class Cap:
    """A deployment's public bound on its inner statements: the inner circuit's k_log and vus_per_block, the instances every
    inner statement is padded to, and the claims (2 + the circuit's extra claims)."""

    instances: int
    claims: int
    k_log: int
    vus_per_block: int

    @classmethod
    def parse(cls, spec: str) -> Cap:
        kv = dict(x.split("=", 1) for x in spec.split(","))
        if set(kv) != {"instances", "claims", "k_log", "vus_per_block"}:
            raise ValueError(f"cap {spec!r}: instances=N,claims=C,k_log=K,vus_per_block=V")
        return cls(**{k: int(v) for k, v in kv.items()})

    @classmethod
    def of(cls, circuit: Path, instances: int, claims: int) -> Cap:
        m = S.meta(Path(circuit))
        return cls(instances, claims, m["k_log"], m["vus_per_block"])

    def spec(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())

    @property
    def shape(self) -> A.Shape:
        return S.shape({"k_log": self.k_log, "vus_per_block": self.vus_per_block}, self.instances, self.claims)


def levels(cap: Cap) -> dict[str, tuple[int, int, int]]:
    """Each level's RecOpen statement under the cap: (LANES, H, instances)."""
    return {f"L{i}": (1 << lv.lanes, lv.d - lv.cap, REPS * lv.q) for i, lv in enumerate(cap.shape.sched.levels)}


def outer(cap: Cap) -> dict[str, dict]:
    """Every statement of an outer proof under the cap: each level's and each of the algebra's parts, its template's descriptor
    id and its instances (`rec_outer.stage`'s and `rec_residuals.write`'s ``template`` and ``instances``)."""
    out = {name: {"template": R.definition(lanes, h).id, "instances": n} for name, (lanes, h, n) in levels(cap).items()}
    for j, p in enumerate(RR.parts(RR.spec(cap.shape))):
        out[f"alg-p{j}"] = {"template": RR.definition(p.S).id, "instances": REPS}
    return out


def check(sh: A.Shape, lvls: list[dict], cap: Cap) -> None:
    """Refuse (`Refused`) an inner session whose table shape ``sh`` or opened levels ``lvls`` (`rec_outer.levels`) are not the
    cap's: its V* would be another shape. The algebra's parts are a function of ``sh``."""
    want = cap.shape
    if (sh.m, sh.k_log, sh.claims) != (want.m, want.k_log, want.claims):
        raise Refused(f"SHAPE-REFUSED: the inner table's (m, k_log, claims) {(sh.m, sh.k_log, sh.claims)}, not the cap's "
                      f"{(want.m, want.k_log, want.claims)} ({cap.spec()})")
    got = {f"L{lv['level']}": (lv["LANES"], lv["H"], len(lv["queries"])) for lv in lvls}
    if got != levels(cap):
        raise Refused(f"SHAPE-REFUSED: the session's levels (LANES, H, openings) {got}, not the cap's {levels(cap)}")


def forge_filler(d: Path, n: int, real: int, slot: int | None = None) -> dict:
    """The inner prover's ``d/inst-n.bin`` with filler ``slot``'s (default the first, ``real``) output row one bit off (bit 0
    of its first word), every other row, salt and the public file as staged; written to a new file, so a stage cache the
    staging links to is left as it was. The slot's row must be the last filler's (fillers write one row)."""
    d = Path(d)
    slot = real if slot is None else slot
    if not real <= slot < n:
        raise ValueError(f"slot {slot}: not a filler of {n} instances with {real} real")
    inst, text = d / f"inst-{n}.bin", (d / "circuit.txt").read_text()
    data = bytearray(inst.read_bytes())
    words = next(w for name, w, *_ in json.loads(text.split("\n", 2)[1][5:])["ports"] if name == "out")
    at, last = S._offsets(bytes(data), text, "out", slot)[0], S._offsets(bytes(data), text, "out", n - 1)[0]
    if data[at:at + 2 * words] != data[last:last + 2 * words]:
        raise ValueError(f"instance {slot}'s output row is not the last filler's: --real {real} is not the staging's")
    was = int.from_bytes(data[at:at + 2], "little")
    data[at:at + 2] = (was ^ 1).to_bytes(2, "little")
    tmp = inst.with_name(inst.name + ".forging")
    tmp.write_bytes(bytes(data))
    os.replace(tmp, inst)
    return {"slot": slot, "real": real, "instances": n, "port": "out", "word": 0, "was": was, "now": was ^ 1}


def _digest(template: str) -> str:
    return hashlib.sha512(template.encode()).hexdigest()[:32]


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m verity_flock.rec_shape")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("outer", help="the cap's statements")
    o.add_argument("cap")
    f = sub.add_parser("forge-filler", help="a filler's output row one bit off in the inner prover's file")
    f.add_argument("dir")
    f.add_argument("n", type=int)
    f.add_argument("real", type=int)
    f.add_argument("--slot", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "outer":
        cap = Cap.parse(a.cap)
        sh = cap.shape
        st = {k: {"template_sha512": _digest(v["template"]), "instances": v["instances"]} for k, v in outer(cap).items()}
        print(json.dumps({"cap": cap.spec(), "shape": [sh.m, sh.k_log, sh.claims], "statements": st}))
        return 0
    print(json.dumps(forge_filler(Path(a.dir), a.n, a.real, a.slot)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
