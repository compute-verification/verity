"""A drawn-unit replay bundle (`verity/drawn-unit-bundle/v2`, `examples/h200/interfaces`) proved: each drawn member's units in
one C-Flock `prove --zk` session behind the Lean firewall, served the bundle's own draw, and decided by Lean's `verify --zk`,
the verifier of record; then the honest session with one bit of its proof flipped, which Lean must refuse, and optionally a
tampered window's member, whose proof must not be accepted.

    python benchmarks/one_stage/prove_bundle.py --bundle B.json --window DIR --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --out DIR [--members N] [--tampered DIR] [--live] [--flips N]

* **The bundle** is what `replay_bundle.build` wrote of the window's drawn units. `prove_bundle.sh` proves the
  store's `BUNDLE_WINDOW` (a W8A8 MLP's window, its tampered twin and the H200 interfaces' bundle of four drawn units under its
  registered `subset:4`, a fixture's draw) unless it's given others.
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
(``rows.write``'s ``registered``) would be verified with ``--registered`` as well. The fixture's headers don't name
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
sys.path[:0] = [str(HERE), str(REPO / "benchmarks" / "prove_path"), str(REPO / "examples" / "h200" / "interfaces"),
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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def hold(doc: dict, payload: bytes, wdir: Path) -> tuple[dict[str, bool], dict]:
    """The bundle held to the window (module docstring): ``({check: ok}, the bundle check's findings)``."""
    rec = json.loads((wdir / "registration.json").read_text())
    win = json.loads((wdir / "window.json").read_text())
    bad = RB.check(doc, payload, registration=rec)
    ok = {"bundle_check": not bad, "run": doc["window"]["run"] == win["run"],
          "registration": doc["window"]["registration"] == win["registration_digest"],
          "partition": doc["window"]["partition"] == win["partition_digest"]}
    for mm in doc["members"]:
        pub = wdir / f"m{mm['k']}-public.bin"
        ok[f"public:m{mm['k']}"] = pub.is_file() and hashlib.sha512(pub.read_bytes()).hexdigest() == mm["public_sha512"]
    return ok, bad


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
        return {x: lean.get(x) for x in ("accepted", "why", "verify_s", "partition", "digest", "registered")}

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True, help="the bundle's JSON; its payload is the .bin beside it")
    ap.add_argument("--window", required=True, help="the prover's window: what Window.write wrote, private files included")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=None, help="prove only the bundle's first N drawn members")
    ap.add_argument("--tampered", help="a tampered window of the same shape: its first drawn member must not be accepted")
    ap.add_argument("--session-timeout", type=float, default=14400)
    ap.add_argument("--live", action="store_true", help="the bundle's draw must be live, with its receipt")
    ap.add_argument("--flips", type=int, default=0, help="Lean-verify only the first N one-bit proof edits (0: all)")
    a = ap.parse_args(argv)
    out, wdir = Path(a.out), Path(a.window)
    out.mkdir(parents=True, exist_ok=True)
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
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("PROVEBUNDLE " + json.dumps({"ok": False, "held": held, "bundle_findings": bad}, sort_keys=True))
        return 1
    sv = Staged(wdir, out / "honest")
    members = doc["members"][:a.members] if a.members else doc["members"]
    rows, first, proved = {}, None, []
    with ThreadPoolExecutor(max_workers=1) as lean_pool:
        for bm in members:
            mm = sv.member(bm["k"])
            proved.append((bm, mm, session(a, mm, bm["draw"], out / "honest" / f"session-m{bm['k']}", lean_pool=lean_pool)))
            print(f"PROVEBUNDLE member m{bm['k']} proved: e2e {proved[-1][2]['timing']['e2e_s']} s", flush=True)
        for _, _, s in proved:
            s["lean"] = s["lean"].result()
    for bm, mm, s in proved:
        units = sorted(bm["base"] + i for i in bm["local"])
        s["checks_before_draw"] = mm["checks"]
        s["k_log"] = mm["k_log"]
        s["accepted"] = (all(mm["checks"].values()) and s["m0"]["accepted"] and s["lean"]["accepted"] is True and s["roots"]
                         and sorted(s["served_draw"] or []) == sorted(bm["local"]) and sorted(s["proved"]) == units)
        rows[f"m{bm['k']}"] = {**s, "units": units}
        if s["session"] and (first is None or mm["k_log"] < first[0]["k_log"]):
            first = (mm, s)
        print(f"PROVEBUNDLE member m{bm['k']} units {units}: accepted {s['accepted']} (M0 {s['m0']['accepted']}, Lean "
              f"{s['lean']['accepted']} {s['lean']['why'] or ''}), k_log {mm['k_log']}, e2e {s['timing']['e2e_s']} s, "
              f"Lean verify {s['lean']['verify_s']} s", flush=True)
    summary["members"] = rows
    negatives: dict[str, dict] = {}
    if first:
        summary["flipped_member"] = first[0]["k"]
        for name, v in flipped(a, first[0], Path(first[1]["session"]), out / "flipped").items():
            negatives[f"flipped/{name}"] = v
        n_ok = sum(v["refused"] for k, v in negatives.items() if k.startswith("flipped/"))
        print(f"PROVEBUNDLE flipped bits: {n_ok}/{len(negatives)} refused by Lean", flush=True)
    if a.tampered:
        tw = Path(a.tampered)
        bm = members[0]
        tv = Staged(tw, out / "tampered")
        mm = tv.member(bm["k"])
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
    summary["negatives"] = negatives
    ok = bool(rows) and all(r["accepted"] for r in rows.values()) and bool(negatives) and all(v["refused"] for v in negatives.values())
    summary["ok"] = ok
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("PROVEBUNDLE " + json.dumps({"ok": ok, "accepted": {k: r["accepted"] for k, r in rows.items()},
                                       "negatives_refused": sum(v["refused"] for v in negatives.values()),
                                       "negatives": len(negatives)}, sort_keys=True), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
