"""The first served H1 window end to end against a challenger gate: rehearsed against infra's gate on the serving node
(`local_gate.py`), and the same script run live against vy-challenger-1's. Run by hand or as a research run; never collected.

**What it does.** Opens the lane's period (``period-open --name P --cap 1 --circuit C``), reserves the window's session
(``reserve --peer PEER --scheme S --families F``), and serves `pouw_h1.FILL` (Llama-3.1-8B int8, 64 in flight, T = 64,
S = 18) with ``pouw_h1_gpu.py --workload fill --served --challenger AT --session RESERVE --gate GATE``: the window opens its
session through ``GATE open --session S --root R`` once round 0 is admitted, R = `salt.window_open_root(weights' root,
round 0's request table root, S)`, its salt coin opens after R, and its A codes and B rows are committed as the window
Program's registered values (`pouw_h1_served.Served`), which ``calls_root`` binds. Once the session has its ``end`` entry
it closes the period, takes the ledger (``GATE ledger``), the session's tree (``GATE fetch S``) and the key's pin
(``GATE pubkey``), and runs the verifier's side on them (`pouw_h1_gpu.check_challenger`): `pouw_h1_challenger.session_reason`
against the ledger's newest head, `trusted_rounds_reason` on the session's ``out/`` with the window's ``{root}``,
`pouw_h1.manifest_reason` and `period_reason`. ``pouw_h1_served.register``'s outcome is in the serve record (its
``register_served`` reads row/v1 alone until proofs' REGV2 follow-up, so it refuses the v2 trees until then). Then the
red team (`red_team`), CPU only, reruns the verifier's side on edited copies of the record and the ledger, each of which
must be refused: R replaced by the bare weights' root, R replaced by another window's root, the ledger cut before the
session's end entry, a call's table root changed; and with the local gate, sessions forged through the gate itself (a
period of their own), opened on the bare weights' root and on another window's root with the window's other coins
after its own roots. The verdict is ``$OUT/rehearsal.json``: ``ok`` when the honest window is accepted by
``session_reason``, ``manifest_reason`` and ``period_reason`` and every red-team case is refused.

**The rehearsal** (a research run on a GPU node; infra's tree from ``cursor/infra-challenger-host-98f5``):

    git archive --format=tar.gz -o infra-tree.tgz 5902f9e85d36b0de284dd4de819cad464a801739 verity benchmarks/live_challenger
    research run --on vy-nebius-2 --project verity --source . --cwd source --send infra-tree.tgz --gpus 1 --mem-gb 400 \\
      --declared-output 'out/*' -- bash -lc 'cd integrations/vllm && PYTHONPATH=$PWD/../..:$PWD \\
      HF_HOME=/workspace/hf /workspace/jobs/venv312/bin/python -m tests.protocol_options.pouw_h1_rehearsal --out $OUT \\
      --tree $RESEARCH_RUN_DIR/inputs/infra-tree.tgz --commit 5902f9e85d36b0de284dd4de819cad464a801739'

**The live run** is the same command with the gate vy-challenger-1's, over the lane's own gate key (INF-170), and no
``--tree``:

    research run --on vy-nebius-2 --project verity --source . --cwd source --gpus 1 --mem-gb 400 --declared-output 'out/*' \\
      -- bash -lc 'cd integrations/vllm && PYTHONPATH=$PWD/../..:$PWD HF_HOME=/workspace/hf \\
      /workspace/jobs/venv312/bin/python -m tests.protocol_options.pouw_h1_rehearsal --out $OUT \\
      --gate "ssh -i $HOME/.ssh/vllm-gate -o BatchMode=yes research@vy-challenger-1" --peer vy-nebius-2 \\
      --at 127.0.0.1:7811 --key-sha512 PIN --period vllm-h1-live-1'

What changes from the rehearsal: the gate is vy-challenger-1's forced command over ssh (``by`` the lane the key is
registered for, ``vllm``), not ``python -I local_gate.py DIR``; its key is ``lms.DEPLOYED``, pinned by ``--key-sha512``
(the auditor's ``public_sha512`` from config, which the run checks against ``GATE pubkey``), not a test key made in a
temporary directory; serve.py runs on vy-challenger-1 and reaches the serving node through ``ssh -R`` to the peer named
in ``/etc/vy/peers`` (``--peer``, the serving node), whose 127.0.0.1:PORT the window connects to, PORT the one the open
answers (7811..7899, `pouw_h1_window.at_port`), not 27811.. on the node itself; its coins are the Lean verifier's on
vy-challenger-1; the session's end is read from ``GATE status`` (the live gate has no ``wait``); and no session is
forged (each would be a live window of the lane's period). The period's circuit (``--circuit``) is `period_circuit`
until compute accounting names the declared circuit's SHA-512. Never run anything on vy-challenger-1 itself.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LOCAL_GATE = HERE / "local_gate.py"
SCHEMA = "verity-vllm/pouw-h1-rehearsal/v1"
RECORD = "rehearsal.json"


def period_circuit() -> str:
    """The period's ``--circuit``: SHA-512 over H1's scheme and `pouw_h1.FILL`'s public declaration."""
    from verity_vllm.protocol_options import pouw_h1 as PH
    d = PH.FILL
    doc = {"scheme": PH.SCHEME, "slots": d.slots, "prompt_lens": list(d.prompt_lens), "max_tokens": d.max_tokens,
           "rows": d.rows, "steps": d.steps}
    return hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()


class Gate:
    """The gate's verbs as ``argv + VERB``, each call and its answer kept in ``log``."""

    def __init__(self, argv: list[str]) -> None:
        self.argv, self.log = argv, []

    def raw(self, *args: str) -> tuple[int, bytes]:
        p = subprocess.run([*self.argv, *args], capture_output=True)
        return p.returncode, p.stdout + (p.stderr if p.returncode else b"")

    def __call__(self, *args: str) -> dict[str, Any]:
        rc, out = self.raw(*args)
        try:
            doc = json.loads(out)
        except ValueError:
            doc = {"refused": f"exit {rc}: {out[-2000:].decode(errors='replace')}"}
        self.log.append({"verb": list(args), "rc": rc, "answer": {k: v for k, v in doc.items() if k != "secret"}})
        return doc


def _poll_end(gate: Gate, session: str) -> dict[str, Any]:
    """The live gate has no ``wait``: read ``status`` until the session has its end entry (serve.py's exit after the
    window's close), at most an hour."""
    import time
    for _ in range(720):
        s = gate("status", session)
        if s.get("end") is not None or "refused" in s or not s.get("active"):
            return s
        time.sleep(5)
    return gate("status", session)


def serve(a: argparse.Namespace, gate: Gate, reserved: dict[str, Any], out: Path) -> int:
    argv = [sys.executable, "-m", "tests.protocol_options.pouw_h1_gpu", "--workload", "fill", *(["--served"] if a.served else []), "--out", str(out),
            "--challenger", a.at_addr, "--session", json.dumps(reserved), "--gate", shlex.join(gate.argv),
            "--at", a.pin_at, "--by", a.pin_by]
    with open(out.parent / "serve.log", "wb") as log:
        return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parents[1]).returncode


def evidence(gate: Gate, session: str, d: Path) -> dict[str, Any]:
    """The ledger, the key's pin and the session's tree as the gate gives them, in `d`."""
    d.mkdir(parents=True, exist_ok=True)
    rc, ledger = gate.raw("ledger")
    (d / "windows.jsonl").write_bytes(ledger if rc == 0 else b"")
    pub = gate("pubkey")
    (d / "pubkey.json").write_text(json.dumps(pub, indent=1) + "\n")
    rc, tgz = gate.raw("fetch", session)
    if rc == 0:
        with tarfile.open(fileobj=io.BytesIO(tgz)) as t:
            t.extractall(d / "fetch", filter="data")
    return {"ledger_lines": ledger.count(b"\n") if rc == 0 else 0, "pubkey": pub, "fetched": rc == 0}


def check(calls: Path, tree: Path, pin: str, ledger: list[Path], out: Path, window: dict | None = None,
          period: str | None = None) -> dict[str, Any]:
    """`pouw_h1_gpu.check_challenger`, the verifier's side: with `window` on the session's ``out/``
    (`trusted_rounds_reason`), else on its tree (`session_reason`)."""
    from tests.protocol_options import pouw_h1_gpu as GPU
    wfile = None
    if window is not None:
        out.mkdir(parents=True, exist_ok=True)
        wfile = out / "window.json"
        wfile.write_text(json.dumps(window) + "\n")
    doc = GPU.check_challenger(str(calls), str(tree / "out" if window is not None else tree), pin, str(out),
                               [str(p) for p in ledger], None if wfile is None else str(wfile), period)
    return {k: doc.get(k) for k in ("ok", "reason", "manifest", "period")}


def _edited(record: dict, edit: Any, path: Path) -> Path:
    doc = json.loads(json.dumps(record))
    edit(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) + "\n")
    return path


def forge(gate: Gate, honest: dict[str, Any], opened_on: bytes, peer: str, at: str) -> tuple[dict[str, Any], str]:
    """`honest` (the window's commitment) with its coins swapped for a session the gate reserves and opens on
    `opened_on`, its salt opened after `opened_on` and its round and draw coins after the window's own roots, as
    `test_pouw_h1._salted` forges one in process: (the forged commitment, its session)."""
    from verity.core.protocols.pouw import window as Wd
    from verity_vllm.protocol_options import pouw_h1_challenger as PHC
    from verity_vllm.protocol_options import pouw_h1_requests as RQ
    from verity_vllm.protocol_options import pouw_h1_window as PHW
    reserved = gate("reserve", "--peer", peer, "--scheme", PHC.SCHEME, "--families", PHC.families(len(honest["rounds"])),
                    "--minutes", "30")
    if "refused" in reserved:
        raise ValueError(f"reserve: {reserved['refused']}")
    answer = gate("open", "--session", reserved["session"], "--root", opened_on.hex())
    if "refused" in answer:
        raise ValueError(f"open: {answer['refused']}")
    if answer.get("commitment") is None:
        answer["commitment"] = gate("status", reserved["session"]).get("commitment")
    client = PHW.connect(PHW.at_port(at, answer.get("port")), answer)
    salt = client.open(Wd.SALT, 0, opened_on)
    log = [{**r, "coin": client.open(Wd.ROUND, s, bytes.fromhex(r["root"])).to_json()} for s, r in enumerate(honest["rounds"])]
    root = RQ.draw_root(bytes.fromhex(honest["calls_root"]), bytes.fromhex(honest["draw"]["receipts"]), log)
    draw = client.open(Wd.DRAW, 0, root)
    client.close()
    coins = client.commitment.to_json()
    return ({**honest, "coins": coins, "round_coins": coins, "salt_opening": salt.to_json(), "rounds": log,
             "session": PHC.carried(answer), "draw": {**honest["draw"], "root": root.hex(), "coin": draw.to_json()}},
            reserved["session"])


def red_team(out: Path, pin: str, period: str, gate: Gate | None, peer: str, at: str) -> dict[str, Any]:
    """Each case the verifier's side must refuse, on edited copies of the run's record and ledger (CPU only), and with
    `gate` (the local gate's), sessions forged through it."""
    from verity.core.protocols.pouw import salt as PS
    record_path = out / "serve" / "pouw-h1-calls.json"
    record = json.loads(record_path.read_text())
    c = record["commitment"]
    sid, R = c["session"]["session"], c["session"]["root"]
    weights, table = c["weights_root"], c["requests_root"]
    other = PS.window_open_root(bytes.fromhex(weights), bytes.fromhex(table), "s20261021T000000Z-0000").hex()
    g = out / "gate"
    tree, ledger = g / "fetch" / sid, [g / "windows.jsonl"]
    rt = out / "red_team"
    cases: dict[str, Any] = {}

    def case(name: str, rec: Path, *, window: dict | None = None, led: list[Path] | None = None, tr: Path | None = None) -> None:
        got = check(rec, tr or tree, pin, led or ledger, rt / name, window, period)
        cases[name] = {**got, "refused": not got["ok"]}

    case("r_bare_weights_root", _edited(record, lambda d: d["commitment"]["session"].update(root=weights), rt / "bare.json"))
    case("r_bare_weights_root_window", record_path, window={"root": weights})
    case("r_other_window_root", _edited(record, lambda d: d["commitment"]["session"].update(root=other), rt / "other.json"))
    case("r_other_window_root_window", record_path, window={"root": other})
    lines = (g / "windows.jsonl").read_bytes().split(b"\n")
    end = next(i for i, ln in enumerate(lines) if ln and json.loads(ln)["kind"] == "end" and json.loads(ln)["session"] == sid)
    cut = rt / "ledger-cut.jsonl"
    rt.mkdir(parents=True, exist_ok=True)
    cut.write_bytes(b"".join(ln + b"\n" for ln in lines[:end]))
    case("ledger_cut_before_end", record_path, led=[cut])
    first = next(i for i, x in enumerate(record["calls"]) if x.get("tables"))
    case("table_root_changed", _edited(record, lambda d: d["calls"][first]["tables"].update(a="00" * 64), rt / "table.json"))
    if gate is not None:
        forged_period = f"{period}-forged"
        cases["forged_period_open"] = gate("period-open", "--name", forged_period, "--cap", "2", "--circuit", period_circuit())
        forged = {}
        for name, root in (("forged_bare_weights_root", bytes.fromhex(weights)), ("forged_other_window_root", bytes.fromhex(other))):
            try:
                forged[name] = forge(gate, c, root, peer, at)
            except Exception as e:  # noqa: BLE001
                cases[name] = {"refused": True, "at": "forge", "reason": f"{type(e).__name__}: {e}"}
        for name, (_c, s) in forged.items():
            gate("wait", s)
        cases["forged_period_close"] = gate("period-close", "--name", forged_period)
        fe = evidence(gate, sid, rt / "forged_gate")
        cases["forged_gate"] = fe
        for name, (commits, s) in forged.items():
            rc, tgz = gate.raw("fetch", s)
            with tarfile.open(fileobj=io.BytesIO(tgz)) as t:
                t.extractall(rt / "forged_gate" / "fetch", filter="data")
            rec = _edited(record, lambda d, commits=commits: d.update(commitment=commits), rt / f"{name}.json")
            case(name, rec, led=[rt / "forged_gate" / "windows.jsonl"], tr=rt / "forged_gate" / "fetch" / s)
        case("honest_under_later_head", record_path, led=[rt / "forged_gate" / "windows.jsonl"])
        cases["honest_under_later_head"]["refused"] = not cases["honest_under_later_head"]["ok"]
    must = [k for k in cases if k.startswith(("r_", "ledger_", "table_", "forged_b", "forged_o"))]
    return {"cases": cases, "all_refused": all(cases[k].get("refused") for k in must), "must_refuse": must}


def rehearse(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm.protocol_options import pouw_h1 as PH
    from verity_vllm.protocol_options import pouw_h1_challenger as PHC
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "period": a.period, "live": a.gate is not None, "served": a.served}
    if a.gate is None:
        gdir = Path(tempfile.mkdtemp(prefix="h1-gate-")) / "gate"
        gdir.mkdir()
        gate = Gate([sys.executable, "-I", str(LOCAL_GATE), str(gdir)])
        rc, init = gate.raw("init", "--tree", a.tree, "--commit", a.commit, "--peer", a.peer, "--coins", a.coins)
        doc["gate"] = json.loads(init) if rc == 0 else {"init_failed": init[-3000:].decode(errors="replace")}
        if rc:
            return doc
        doc["gate"]["lean_build"] = json.loads((gdir / "gate.json").read_text()).get("lean_build")
        pin = doc["gate"]["pubkey"]["public_sha512"]
        lo = doc["gate"]["ports"][0]
        a.at_addr = a.at or f"127.0.0.1:{lo}"
    else:
        gate = Gate(shlex.split(a.gate))
        pin = a.key_sha512
        a.at_addr = a.at
        if gate("pubkey").get("public_sha512") != pin:
            doc["refused"] = "the gate's pubkey is not the pinned key (--key-sha512)"
            return doc
    doc["key_sha512"] = pin
    doc["period_open"] = gate("period-open", "--name", a.period, "--cap", "1", "--circuit", period_circuit())
    reserved = gate("reserve", "--peer", a.peer, "--scheme", PHC.SCHEME, "--families", PHC.families(PH.FILL.steps),
                    "--minutes", str(a.minutes))
    doc["reserved"] = reserved
    if "refused" in reserved:
        doc["gate_log"] = gate.log
        return doc
    sid = reserved["session"]
    doc["serve_rc"] = serve(a, gate, reserved, out / "serve")
    status = gate("status", sid)
    if status.get("open") is not None:
        doc["unit"] = gate("wait", sid) if a.gate is None else _poll_end(gate, sid)
    doc["period_close"] = gate("period-close", "--name", a.period)
    doc["evidence"] = evidence(gate, sid, out / "gate")
    try:
        serve_doc = json.loads((out / "serve" / "pouw_h1_gpu.json").read_text())
        h = serve_doc["h1"]
        doc["serve"] = {"ok": serve_doc["ok"], "u": h.get("u"), "same_tokens": h.get("same_tokens"),
                        "decode_step_s_median": h.get("decode_step_s_median"), "close_s": h.get("close_s"),
                        "challenger": h.get("challenger"), "served": {k: v for k, v in (h.get("served") or {}).items()
                                                                      if k != "entries"},
                        "served_values": [(e["value"], e["leaves"], e["words"]) for e in (h.get("served") or {}).get("entries", [])]}
    except (OSError, ValueError, KeyError) as e:
        doc["serve"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    calls = out / "serve" / "pouw-h1-calls.json"
    if calls.is_file() and doc["evidence"]["fetched"]:
        tree, ledger = out / "gate" / "fetch" / sid, [out / "gate" / "windows.jsonl"]
        record = json.loads(calls.read_text())
        doc["verify"] = {
            "session": check(calls, tree, pin, ledger, out / "verify" / "session", None, a.period),
            "trusted": check(calls, tree, pin, ledger, out / "verify" / "trusted", {"root": record["commitment"]["session"]["root"]}),
        }
        doc["red_team"] = red_team(out, pin, a.period, gate if a.gate is None else None, a.peer, a.at_addr)
    doc["gate_log"] = gate.log
    v = doc.get("verify") or {}
    doc["ok"] = bool(doc.get("serve", {}).get("ok") and (v.get("session") or {}).get("ok")
                     and (doc.get("red_team") or {}).get("all_refused"))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", help="the live gate's command (ssh ... research@vy-challenger-1); default: a local gate of --tree")
    ap.add_argument("--tree", help="infra's tree (git archive COMMIT verity benchmarks/live_challenger, gzipped)")
    ap.add_argument("--commit", help="the commit --tree is (the local gate's pinned commit)")
    ap.add_argument("--coins", choices=("lean", "stub"), default="lean")
    ap.add_argument("--peer", default="local", help="the serving node's name in the gate's peers")
    ap.add_argument("--at", help="the challenger's HOST:PORT before the open answers its port (live: 127.0.0.1:7811)")
    ap.add_argument("--key-sha512", help="live: the auditor's pinned public_sha512")
    ap.add_argument("--period", default="vllm-h1-rehearsal")
    ap.add_argument("--no-served", dest="served", action="store_false",
                    help="serve without pouw_h1_served.Served (its window Program over the 8B window's 2304 calls takes hours)")
    ap.add_argument("--minutes", type=int, default=300, help="the reserve's minutes, which also bound the session")
    ap.add_argument("--pin-at", default="2026-10-21T00:00:00Z", help="pouw_h1_gpu.py's --at (resolved_at, if a pin is made)")
    ap.add_argument("--pin-by", default="vllm", help="pouw_h1_gpu.py's --by")
    a = ap.parse_args(argv)
    if a.gate is None and not (a.tree and a.commit):
        ap.error("the local gate needs --tree and --commit")
    if a.gate is not None and not (a.key_sha512 and a.at):
        ap.error("the live gate needs --key-sha512 (the auditor's pin) and --at (127.0.0.1:7811 on the peer)")
    doc = rehearse(a)
    Path(a.out, RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    rt = doc.get("red_team") or {}
    print(f"POUW-H1-REHEARSAL {'OK' if doc.get('ok') else 'FAIL'} serve={doc.get('serve', {}).get('ok')} "
          f"session={(doc.get('verify') or {}).get('session')} trusted={(doc.get('verify') or {}).get('trusted')} "
          f"red_team_all_refused={rt.get('all_refused')}", flush=True)
    return 0 if doc.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
