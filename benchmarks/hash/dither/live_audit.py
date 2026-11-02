"""A live dither-a-so window's audit, run from the machine that holds the gate key:
the session and the auditor's ledger fetched from the challenger's gate, the GPU run's manifest and stop record and the
challenger's key record fetched from the store, then

* ``manifest-check.json``: `dither_manifest.check` at this tree (every round root and the manifest root recomputed from
  the manifest's leaves, the coins held to the challenger's own run, its session run from a commit the host's pins log
  records on ``--main``) under the key record, the period and the session's stand-in root (`coins.stand_in_root`);
* ``core-check.json``: core's own check, run from a checkout of main (``--core-from``, its ``verity`` and nothing of
  this tree): the session trusted on the stand-in root (`remote.trusted_session`, every head of the ledger), the round
  coins (`core_check.rounds` over the recomputed round roots, its session run from a commit the host's pins log
  ``--pins`` records on the auditor's first-parent list of main ``--main``), the salt coin after the stand-in root and
  the sampling coin
  after the recomputed manifest root (`challenger.live_reason`, each the root the challenger's signed log shows), the
  commitment the manifest's, a live epoch's;
* ``check-end.json``: the GPU run's own `dither_so.check_end` of the window (its stop record's ``end``), which needs the
  serving process's device state and so runs there, copied with its verdict.

The pins log is ``--pins``, else the newest one infra published: of the challenger-pins/v1 artifacts its publisher
labelled `challenger-pins head` (each head keeps its label when the next is put), the one labelled last (`newest_pins`).

``audit.json`` holds the three verdicts, the commits and the inputs' SHA-512s; the exit status is 0 only when all hold.

    OUT=DIR python3 benchmarks/hash/dither/live_audit.py audit --session SID --run RUN --arm ARM --gate USER@HOST \\
        --ssh-key KEY --key-record ART --period P --core-from MAIN --main FIRST_PARENT [--pins PINS]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESEARCH = os.environ.get("RESEARCH", "research")
PINS_KIND, PINS_LABEL, PINS_BY = "challenger-pins/v1", "challenger-pins", "infra"


def _run(argv: list[str], **kw) -> bytes:
    return subprocess.run(argv, check=True, stdout=subprocess.PIPE, **kw).stdout


def _sha(p: Path) -> str:
    return hashlib.sha512(p.read_bytes()).hexdigest()


def _dump(p: Path, doc) -> None:
    p.write_text(json.dumps(doc, indent=1) + "\n")


def _git(d: Path) -> str:
    return _run(["git", "-C", str(d), "rev-parse", "HEAD"]).decode().strip()


def fetch(a, out: Path) -> dict:
    """The session (``fetch SID``) and every ledger head (``ledger``) from the gate, the run's record and the key record
    from the store; their paths."""
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-i", a.ssh_key, a.gate]
    tgz = out / f"session-{a.session}.tgz"
    tgz.write_bytes(_run([*ssh, f"fetch {a.session}"]))
    sdir = out / "session"
    sdir.mkdir()
    with tarfile.open(tgz) as t:
        t.extractall(sdir, filter="data")
    ledger = out / "ledger.jsonl"
    ledger.write_bytes(_run([*ssh, "ledger"]))
    art = a.run_record or json.loads(_run([RESEARCH, "data", "show", a.run, "--json"]))["outputs"]["run_record"]
    rdir = out / "run"
    _run([RESEARCH, "data", "fetch", art, "--to", str(rdir), "--path", "out/manifest-*.json", "--path", "out/stop.json",
          "--path", "out/commit.txt"])
    kdir = out / "key"
    _run([RESEARCH, "data", "fetch", a.key_record, "--to", str(kdir)])
    session = next(p for p in sdir.rglob("out") if p.is_dir())
    manifest = rdir / "out" / f"manifest-{a.session}.json"
    return {"session": session, "ledger": ledger, "manifest": manifest, "stop": rdir / "out" / "stop.json",
            "commit": rdir / "out" / "commit.txt", "key": next(kdir.rglob("*.json")), "tgz": tgz, "run_record": art}


def newest_pins(out: Path) -> tuple[Path, str]:
    """The host's pins log infra published last, fetched into `out`/pins: of the store's challenger-pins/v1 artifacts
    labelled `challenger-pins head` by infra, the one whose label is newest; its path and art id."""
    ids = _run([RESEARCH, "data", "select", "--kind", PINS_KIND, "--label", f"{PINS_LABEL}=head", "--ids"]).decode().split()

    def labelled(art: str) -> str:
        got = json.loads(_run([RESEARCH, "data", "labels", art, "--key", PINS_LABEL, "--json"]))
        return max((x["ts"] for x in got if x.get("by") == PINS_BY and x.get("value") == "head"), default="")

    when = {art: labelled(art) for art in ids}
    if not any(when.values()):
        raise SystemExit(f"the store holds no {PINS_KIND} labelled `{PINS_LABEL} head` by {PINS_BY}: pass --pins")
    art = max(when, key=when.__getitem__)
    pdir = out / "pins"
    _run([RESEARCH, "data", "fetch", art, "--to", str(pdir)])
    return next(pdir.rglob("*.jsonl")), art


def manifest_check(a, p: dict, out: Path) -> dict:
    """`dither_manifest.check` at this tree, and what the verifier recomputes from the manifest for core's check."""
    for d in (ROOT / "benchmarks" / "pouw" / "served_h1", HERE):
        sys.path.insert(0, str(d))
    import coins as CO
    import dither_manifest as MF
    root = CO.stand_in_root(a.session)
    pins = [x for x in p["pins"].read_text().split("\n") if x]
    got = MF.check(p["manifest"], p["session"], [str(p["ledger"])], pins, CO.R.main_lines(p["main"]),
                   CO.key_record(p["key"]), a.period, root)
    _dump(out / "manifest-check.json", got)
    m = json.loads(p["manifest"].read_text())
    e = MF.expect(m)
    _dump(out / "recomputed.json", {"root": root.hex(), "rounds": [x.hex() for x in e["rounds"]],
                                     "manifest_root": e["draw_after"].hex()})
    return got


def core_check(a) -> int:
    """Core's check from main's `verity` (this process's PYTHONPATH): see the module docstring."""
    import verity
    from verity.core.protocols.pouw import core_check as CC
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.sampled_proofs import challenger as Ch
    from verity.core.protocols.sampled_proofs import remote as R
    src = Path(verity.__file__).resolve().parent
    if not src.is_relative_to(Path(a.core_from).resolve()):
        raise SystemExit(f"core's check imported verity from {src}, not --core-from {a.core_from}")
    re_ = json.loads(Path(a.recomputed).read_text())
    m = json.loads(Path(a.manifest).read_text())["coins"]
    key = Ch.key_of_record(json.loads(Path(a.key).read_text()))
    root, mroot = bytes.fromhex(re_["root"]), bytes.fromhex(re_["manifest_root"])
    window, heads = {"root": root.hex()}, [a.ledger]
    doc: dict = {"verity_from": str(src), "session": a.session, "window_root": root.hex(), "heads": heads}
    try:
        commitment, received = R.trusted_session(a.session_dir, heads, window, key, CC.lms_verify)
        ledger = Ch.ledger_head([R.ledger_lines(h) for h in heads], key, CC.lms_verify)
        audited = Ch.audited_session(ledger, window, key, CC.lms_verify)
    except Ch.CoinError as e:
        doc.update(trusted_session=f"refused: {e}", verdict="refused")
        print(json.dumps(doc, indent=1))
        return 1
    entries = {e["kind"]: e for e in map(json.loads, ledger) if e["session"] == audited.session}
    pins = [x for x in Path(a.pins).read_text().split("\n") if x]
    rounds = CC.rounds(key.to_json(), ledger, pins, R.main_lines(a.main), window, R.session_files(a.session_dir),
                       re_["rounds"], [r["opening"] for r in m["rounds"]], CC.lms_verify)
    salt = Ch.LiveOpening.from_json(m["salt_coin"])
    draw = Ch.LiveOpening.from_json(m["draw_coin"])
    checks = {
        "trusted_session": True,
        "commitment_is_the_manifests": Ch.LiveCommitment.from_json(m["commitment"]) == commitment,
        "live_epoch": Wd.live_epoch_reason(commitment) is None,
        "rounds": rounds is None,
        "salt_after_window_root": received.get(Wd.SALT) == (root,)
        and Ch.live_reason(commitment, salt, Wd.SALT, 0, root) is None,
        "draw_after_manifest_root": received.get(Wd.DRAW) == (mroot,)
        and Ch.live_reason(commitment, draw, Wd.DRAW, 0, mroot) is None,
        "period": audited.period == a.period,
        "end_complete": bool(entries.get("end", {}).get("body", {}).get("complete")),
    }
    doc.update(checks=checks, rounds_refusal=rounds, families=dict(commitment.families),
               ledger={"session": audited.session, "prover": audited.prover, "period": audited.period,
                       "cap": audited.cap, "circuit": audited.circuit, "refused_hellos": audited.refused_hellos,
                       **{k: entries[k]["seq"] if k in entries else None for k in Ch.SESSION_KINDS}},
               round_roots=len(re_["rounds"]), verdict="accepted" if all(checks.values()) else "refused")
    print(json.dumps(doc, indent=1))
    return 0 if doc["verdict"] == "accepted" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    au = sub.add_parser("audit", help="the whole audit, into $OUT")
    au.add_argument("--session", required=True)
    au.add_argument("--run", required=True, help="the GPU run whose window the session served")
    au.add_argument("--run-record", help="its run record's art: (default: the store's attempt for --run)")
    au.add_argument("--arm", required=True, help="the arm whose window the session served")
    au.add_argument("--gate", required=True, help="the challenger's gate, USER@HOST")
    au.add_argument("--ssh-key", required=True, help="the SSH key the gate accepts")
    au.add_argument("--key-record", required=True, help="the challenger's key record's art: (challenger-key current)")
    au.add_argument("--period", required=True)
    au.add_argument("--core-from", required=True, help="a checkout of main whose verity runs core's check")
    au.add_argument("--pins", help="the challenger host's published pins log, a file (default: the newest challenger-pins "
                    "head infra published, `newest_pins`)")
    au.add_argument("--main", required=True, help="the auditor's own `git rev-list --first-parent origin/main`, a file")
    co = sub.add_parser("core", help="core's check alone, under main's verity on PYTHONPATH")
    for k in ("session", "session-dir", "ledger", "manifest", "key", "period", "recomputed", "core-from", "pins",
              "main"):
        co.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "core":
        return core_check(a)
    out = Path(os.environ.get("OUT") or ".").resolve()
    p = fetch(a, out)
    p["pins"], pins_record = (Path(a.pins).resolve(), None) if a.pins else newest_pins(out)
    p["main"] = Path(a.main).resolve()
    mc = manifest_check(a, p, out)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {"PYTHONPATH": str(Path(a.core_from).resolve())}
    cp = subprocess.run([sys.executable, __file__, "core", "--session", a.session, "--session-dir", str(p["session"]),
                         "--ledger", str(p["ledger"]), "--manifest", str(p["manifest"]), "--key", str(p["key"]),
                         "--period", a.period, "--recomputed", str(out / "recomputed.json"), "--core-from", a.core_from,
                         "--pins", str(p["pins"]), "--main", str(p["main"])],
                        check=False, env=env, cwd="/", stdout=subprocess.PIPE, text=True)
    (out / "core-check.json").write_text(cp.stdout)
    core = json.loads(cp.stdout)
    stop = json.loads(p["stop"].read_text())
    end = stop[a.arm]["end"]
    _dump(out / "check-end.json", {"arm": a.arm, "run": a.run, "fill_steps": stop[a.arm].get("fill_steps"), **end})
    doc = {"session": a.session, "run": a.run, "arm": a.arm, "run_record": p["run_record"],
           "run_commit": p["commit"].read_text().strip(), "audit_commit": _git(ROOT), "core_commit": _git(Path(a.core_from)),
           "period": a.period, "key_record": a.key_record, "pins_record": pins_record,
           "inputs_sha512": {k: _sha(p[k]) for k in ("tgz", "ledger", "manifest", "stop", "key", "pins", "main")},
           "manifest_check": {"ok": mc["ok"], "coins": mc.get("coins"), "recomputed": mc.get("recomputed")},
           "core_check": {"verdict": core.get("verdict"), "checks": core.get("checks"), "ledger": core.get("ledger"),
                          "verity_from": core.get("verity_from")},
           "check_end": {"verdict": end.get("verdict"), "live_ok": end.get("live", {}).get("ok"),
                         "spill_ok": (end.get("live", {}).get("spill") or {}).get("ok")}}
    doc["ok"] = bool(mc["ok"] and core.get("verdict") == "accepted" and end.get("verdict", {}).get("ok"))
    _dump(out / "audit.json", doc)
    print(json.dumps({"ok": doc["ok"], "manifest_check": mc["ok"], "core_check": core.get("verdict"),
                      "check_end": end.get("verdict", {}).get("ok")}))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
