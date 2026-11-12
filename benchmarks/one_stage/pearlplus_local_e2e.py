"""Pearl+'s live draw end to end on one CPU machine, against infra's own gate code (`verity/composite/device/live_challenger/auditor.py`)
on a scratch auditor host: its LMS-signed ledger, and each session's challenger a real ``serve.py`` process whose coins
are the Lean verifier of record's (``sampled-proofs-verifier``, built first).

    python benchmarks/one_stage/pearlplus_local_e2e.py run --out OUT [--lean build|EXE]

The gate is this file's ``gate`` verb, a command in place of ``ssh research@vy-challenger-1`` (PEARLC_CHALLENGER_GATE):
it runs `auditor.gate` as lane `LANE`'s gate key on the scratch host, and starts each opened session's challenger in a
process of its own (this file's ``serve`` verb, `auditor.run` with no coin link, its client reaching serve.py's port on
127.0.0.1). ``run`` then takes the steps `served_pearlplus.sh` takes, through the same commands:

1. ``served_verdicts.py pearlplus-period open``: the lane's Pearl-C period, cap 1.
2. A salt-only session reserved in that period and opened on salt epoch 0's root over the passes' weights' root
   (`pouw_h1_window.reserve_salt` and ``open_session``, as `bench_arm.live_coins`' install does), its salt coin salting
   a toy pass (the pouw suite's `proved_pass`, k = 128 and 16 x 16 tiles, in place of a GPU serve), then closed.
3. ``served_verdicts.py pearlplus-period close``, once that session has ended.
4. ``served_passes.py verify --tie W --test-instance`` under PEARLC_CHALLENGER_GATE/PEER/LANE: the verifier's draw a draw
   session reserved withheld after the period closed and opened on the run's commitment (`bench_arm.LiveDraw`), recorded
   in coin.json. The toy isn't the served instance (k_min 1,024 at 64 x 64 tiles), so its passes are read as a
   test instance's, crediting nothing.
5. ``served_verdicts.py coin`` and ``served_verdicts.py pearlplus``: the coin published and the piece written, each
   session fetched through the gate and held to the signed ledger under the scratch key's pin.
6. `served_verdicts.piece_row`: PoUW's row, composed of record when `pearlplus_of_record` holds. A test instance
   credits nothing, so on the toy every condition holds up to its last, the Lean's credit, which the row names
   (`TOY_ROW`).

Stand-ins, named in OUT/e2e.json: the serve (one toy pass, `served.json` written here), sampled proofs' window
registration and records audit (two files tied to W), and, unless ``--lean`` names a ``pouw-pearlplus``, the Lean
``PouwCheck.PearlPlus`` verdict (``pearlplus-lean.json``, copied from Pearl+'s answer); with it, `served_pearlplus.sh`'s
Lean step writes that file (``served_passes.py lean``, both verdicts in OUT/e2e.json's ``lean``). Then, as a control, a
second draw on the same commitment: the gate refuses its open (one session a root, ever), so no second coin exists, and
the piece written after it composes the same row."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
LANE, PEER, HOST = "vllm", "local", "local-challenger"
#: the scratch challenger key's LMS (W, H): 2^6 signatures, over the ledger's and both sessions' few dozen
PARAMS = (4, 6)
SCHEME = "pearl-c-h100-v1-h3"
PERIOD = "pearlplus-e2e"
#: the toy row's one refusal, after every other condition of record held: a test instance's passes credit nothing
TOY_ROW = "the Lean verdict credits 0 MACs over {tiles} tiles: nothing"
#: how long a session's end, or a session's challenger, is waited for
WAIT_S = 300


def _auditor():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e2e_auditor", TREE / "verity" / "composite" / "device" / "live_challenger" / "auditor.py")
    module = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


def _host(root: Path):
    from verity.commitments.crypto import lms
    A = _auditor()
    host = A.Host(root=root / "var", peers=root / "peers", name=HOST, pins=root / "pins.jsonl")
    return A, host, A.key_signer(host, lms.Params(64, *PARAMS), workers=1), (root / "pin").read_text().strip()


def _alive(root: Path, session: str) -> bool:
    pid = root / f"{session}.pid"
    return pid.exists() and _alive_pid(int(pid.read_text()))


def gate(root: Path, words: str) -> int:
    """`auditor.gate` on `words` as lane `LANE`'s key, its answer on stdout and its exit code returned."""
    A, host, signer, pin = _host(root)
    now = datetime.now(UTC)

    def active(s: str) -> bool:
        return _alive(root, s)

    def start(s: str) -> None:
        with open(root / f"serve-{s}.log", "wb") as log:
            p = subprocess.Popen([sys.executable, __file__, "serve", "--root", str(root), s], stdout=log, stderr=log,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        (root / f"{s}.pid").write_text(str(p.pid))

    hooks = dict(now=now, own=lambda p: None, start=start, active=active,
                 reserve=lambda s: A.reserve(host, s, signer=signer, pinned=pin, now=now),
                 admit=lambda s: A.admit(host, s, signer=signer, pinned=pin, now=now, active=active),
                 ready=lambda s: A.await_commitment(host, s, active),
                 period=lambda args: A.period(host, args, signer=signer, pinned=pin, now=now, active=active))
    rc, out = A.gate(["--ssh", words, "--key", LANE], host, **hooks)
    sys.stdout.buffer.write(out)
    return rc


def serve(root: Path, session: str) -> int:
    """The session's challenger, as the host's unit runs it, on its open's port with no coin link."""
    A, host, signer, pin = _host(root)
    body = A.sessions_of(A.read_ledger(host.ledger))[session]["open"]["body"]
    argv = [*A.serve_argv(host, session, body), "--params", ",".join(map(str, PARAMS)), "--keygen-workers", "1"]
    A.run(host, session, signer=signer, pinned=pin, now=lambda: datetime.now(UTC), argv=argv, link=None)
    return 0


def _ledger(root: Path) -> list[dict]:
    path = root / "var" / "windows.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _ended(root: Path, session: str) -> dict:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        end = next((e for e in _ledger(root) if e["kind"] == "end" and e.get("session") == session), None)
        if end is not None:
            return end
        time.sleep(0.5)
    raise RuntimeError(f"session {session} didn't end within {WAIT_S} s")


def _cli(log: list, name: str, argv: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=TREE, env={**os.environ, **(env or {})})
    log.append({"step": name, "argv": argv, "rc": p.returncode, "stdout": p.stdout[-3000:], "stderr": p.stderr[-3000:]})
    print(f"[{name}] exit {p.returncode}: {(p.stdout.strip().splitlines() or [''])[-1][:400]}", flush=True)
    if p.returncode and name not in ("verify",):
        print(p.stderr[-2000:], file=sys.stderr, flush=True)
    return p


def _logs(root: Path, out: Path) -> None:
    """The scratch host's logs and its sessions' public files into `out`, its key's private half left behind."""
    import shutil
    dest = out / "auditor"
    dest.mkdir(parents=True, exist_ok=True)
    for f in root.glob("serve-*.log"):
        shutil.copyfile(f, dest / f.name)
    for name in ("windows.jsonl",):
        if (root / "var" / name).exists():
            shutil.copyfile(root / "var" / name, dest / name)
    for d in (root / "var" / "sessions").glob("*/out"):
        shutil.copytree(d, dest / "sessions" / d.parent.name, dirs_exist_ok=True)


def _free_ports(n: int = 8) -> range:
    """`n` consecutive ports of `auditor.PORTS` that nothing on this machine listens on: a session's challenger listens
    on its open's port, and another run's may still hold the first."""
    import socket
    A = _auditor()
    def listened(port: int) -> bool:
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    start = next((s for s in range(A.PORTS.start, A.PORTS.stop - n) if not any(map(listened, range(s, s + n)))), None)
    if start is None:
        raise RuntimeError(f"no {n} free ports in {A.PORTS}")
    return range(start, start + n)


def _stop(root: Path) -> None:
    """SIGTERM to each challenger this run started (its ``serve`` process and serve.py, its child) still running."""
    import signal
    for f in root.glob("*.pid"):
        pid = int(f.read_text())
        kids = Path(f"/proc/{pid}/task/{pid}/children")
        for p in (*(map(int, kids.read_text().split()) if kids.exists() else ()), pid):
            if _alive_pid(p):
                os.kill(p, signal.SIGTERM)


def _alive_pid(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def run(out: Path, lean: str | None = None) -> int:
    root = Path(tempfile.mkdtemp(prefix="pp-auditor-"))
    try:
        return _run(out, root, lean)
    finally:
        _stop(root)
        _logs(root, out)


def _run(out: Path, root: Path, lean: str | None = None) -> int:
    from verity.protocols.pouw.verifier import salt as PS
    from verity_vllm.pipeline.commit_coins import ChallengerGate
    from verity_vllm.protocols.pouw.h1 import window as PHW

    sys.path[:0] = [str(HERE), str(TREE / "benchmarks" / "pouw" / "tests")]
    import served_verdicts as S
    import test_pearl_c_vllm as T

    from verity.protocols.spot.verifier.one_stage import verifier as V

    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="pp-e2e-"))
    t = time.monotonic()
    print(f"[coins] {V.executable()} built in {time.monotonic() - t:.0f} s: each challenger's coins are its", flush=True)
    A = _auditor()
    host = A.Host(root=root / "var", peers=root / "peers", name=HOST, pins=root / "pins.jsonl")
    (root / "var" / "sessions").mkdir(parents=True)
    (root / "peers").write_text(f"{HOST} research@127.0.0.1\n{PEER} research@127.0.0.1\n")
    ports = _free_ports()
    (root / "coin-ports").write_text(f"{PEER} {ports.start}-{ports.stop - 1}\n")
    head = subprocess.run(["git", "-C", str(TREE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pin = head if len(head) == 40 else "e2" * 20
    (root / "pin").write_text(pin + "\n")
    host.pins.write_text(A.pin_line(pin, pin, A.utc(datetime.now(UTC))) + "\n")
    (main := out / "main.txt").write_text(pin + "\n")
    from verity.commitments.crypto import lms
    key_sha512 = A.key_signer(host, lms.Params(64, *PARAMS), workers=1)[2]
    gate_cmd = shlex.join([sys.executable, str(Path(__file__).resolve()), "gate", "--root", str(root)])
    piece, log = out / "pieces" / "pouw", []
    piece.mkdir(parents=True)
    report: dict = {"gate": "verity/composite/device/live_challenger/auditor.py on a scratch host, as lane " + LANE, "pin": pin,
                    "key_sha512": key_sha512, "period": PERIOD, "steps": log,
                    "stand_ins": {"serve": "one toy pass (proved_pass, k 128, 16 x 16 tiles) salted by the live session",
                                  "sampled_proofs": "registration.json chained after W, a records-audit summary tied to W "
                                                    "and a toy circuit record beside R, the period's circuit",
                                  "lean": "pearlplus-lean.json copied from Pearl+'s answer (--lean runs the node's step)"}}

    # 1. the lane's Pearl-C period, its circuit the window's circuit record
    (circuit_record := scratch / S.CIRCUIT_RECORD).write_bytes(b'{"name":"toy window circuit (pearlplus_local_e2e)"}\n')
    if _cli(log, "period open", ["benchmarks/one_stage/served_verdicts.py", "pearlplus-period", "open", str(piece),
                                 "--gate", gate_cmd, "--name", PERIOD, "--scheme", SCHEME,
                                 "--circuit-record", str(circuit_record)]).returncode:
        return 1

    # 2. a salt-only session in it salts the toy pass
    probe = T.proved_pass(scratch / "probe", SCHEME)
    w = T.weights_root(probe[3].weight_roots)
    g = ChallengerGate(gate_cmd)
    reserved, period = PHW.reserve_salt(g.call, g.ledger, LANE, PEER, 0)
    client, commitment, opening, _salt, carried = PHW.open_session("pearl-plus e2e", g.call, reserved["session"], w, 0)
    assert opening.after == PS.epoch_open_root(w, 0)
    T.live_coin = lambda after, entropy=7: (commitment, opening)
    path = T.proved_pass(scratch / "pass", SCHEME, live=0)[0]
    man = json.loads(path.read_text())
    path.write_text(json.dumps({**man, "session": carried}))
    client.close()
    report["salt_session"] = {"reserved": reserved, "period": (period or {}).get("period"),
                              "end": _ended(root, reserved["session"])["body"]}
    print(f"[salt session] {reserved['session']} ended: {report['salt_session']['end']}", flush=True)

    # 3. the period closes once its session has ended
    if _cli(log, "period close", ["benchmarks/one_stage/served_verdicts.py", "pearlplus-period", "close", str(piece),
                                  "--gate", gate_cmd]).returncode:
        return 1

    # 4. Pearl+'s verdict, its draw a draw session's coin on the run's commitment
    served = scratch / "served"
    served.mkdir()
    (served / "served.json").write_text(json.dumps({"run": T.RUN, "passes": [
        {"pass": 0, "timed": True, "units": 2, "tiles": 4, "manifest": str(path)}]}))
    (served / "coins.json").write_text((path.parent / "coins.json").read_text())
    (wfile := scratch / "window-record.json").write_text(json.dumps(T.window_record(path)))
    env = {"PEARLC_CHALLENGER_GATE": gate_cmd, "PEARLC_CHALLENGER_PEER": PEER, "PEARLC_CHALLENGER_LANE": LANE}
    _cli(log, "verify", ["benchmarks/pouw/pearl_c_vllm/served_passes.py", "verify", "--passes", str(served), "--out",
                         str(piece / "verify.json"), "--answer", str(piece / S.PEARLPLUS_ANSWER), "--jobs", "2",
                         "--tie", str(wfile), "--test-instance"], env)
    record = json.loads((served / "coin.json").read_text())
    dsid = (record["draw"].get("reserved") or {}).get("session")
    report["draw"] = {"from": record["draw"]["from"], "session": dsid, "end": _ended(root, dsid)["body"] if dsid else None,
                      "source": record["source"], "commitment": record["commitment"]}
    print(f"[draw] {report['draw']}", flush=True)

    # 5. the coin published, the piece written
    from verity_config.settings import SHARED

    run_id, run_dir = SHARED.RESEARCH_RUN_ID or "", SHARED.RESEARCH_RUN_DIR
    _cli(log, "coin", ["benchmarks/one_stage/served_verdicts.py", "coin", str(piece), "--record", str(served / "coin.json"),
                       "--run", run_id, *(["--run-dir", run_dir] if run_dir else [])])
    verified = json.loads((piece / "verify.json").read_text())
    answer = json.loads((piece / S.PEARLPLUS_ANSWER).read_text())
    work = (answer.get("public") or {}).get("work") or {}
    (doc := scratch / "served-doc.json").write_text(json.dumps({
        "arm": f"pouw:{SCHEME}", "spec": "toy", "run": T.RUN, "served_tokens": 1, "generate_s": 1.0, "prompt_tokens": 1,
        "output_tokens": 1, "ok": True, "over_caps": [],
        "coins": {"source": S.LIVE_SOURCE, "session": carried, "closed": {"closed": True, "complete": True}},
        "passes": [{"pass": 0, "timed": True, "units": 2, "macs": work.get("macs"), "manifest": str(path)}]}))
    _cli(log, "pearlplus", ["benchmarks/one_stage/served_verdicts.py", "pearlplus", str(piece), "--served", str(doc),
                            "--verify", str(piece / "verify.json"), "--gate", gate_cmd, "--key-sha512", key_sha512,
                            "--main", str(main), "--coin", str(piece / S.COIN_PUT)])

    # 6. the compose
    tie = ((answer.get("public") or {}).get("tie") or {}).get("record")
    if lean is None:
        (piece / S.PEARLPLUS_LEAN).write_text(json.dumps({
            "verdict": answer.get("verdict"), "reason": None, "commitment": answer.get("commitment"),
            "source": (answer.get("profile") or {}).get("source"), "drawn": verified.get("draws"), "work": work,
            "tie": {"record": tie}, "stand_in": True}))
    else:
        exe = lean
        if lean == "build":
            pkg = TREE / "verity" / "protocols" / "pouw" / "verifier" / "lean" / "pearlplus"
            b = subprocess.run(["lake", "build", "pouw-pearlplus"], cwd=pkg, capture_output=True, text=True)
            log.append({"step": "lean build", "rc": b.returncode, "stdout": b.stdout[-3000:], "stderr": b.stderr[-3000:]})
            exe = str(pkg / ".lake" / "build" / "bin" / "pouw-pearlplus") if b.returncode == 0 else "none"
        _cli(log, "lean", ["benchmarks/pouw/pearl_c_vllm/served_passes.py", "lean", "--passes", str(served), "--verify",
                           str(piece / "verify.json"), "--exe", exe, "--out", str(piece / S.PEARLPLUS_LEAN)])
        del report["stand_ins"]["lean"]
        held = json.loads((piece / "pearlplus-lean-check.json").read_text())
        report["lean"] = {k: held.get(k) for k in ("exe", "exe_sha256", "argv", "rc", "stderr", "differ", "reason")} | {
            "lean": {k: (held.get("lean") or {}).get(k) for k in ("verdict", "reason", "refused", "source", "work")}}
    audit = out / "sampled-proofs"
    (audit / S.WINDOW_REGISTRATION).parent.mkdir(parents=True, exist_ok=True)
    (audit / S.WINDOW_REGISTRATION).write_text(json.dumps({"previous": tie, "registered": [], "stand_in": True}))
    (audit / S.AUDIT_SUMMARY).write_text(json.dumps({"tie": {"record": tie}, "stand_in": True}))
    (audit / S.CIRCUIT_RECORD).write_bytes(circuit_record.read_bytes())

    def row() -> dict:
        r = S.piece_row("work", piece.parent, sampled_proofs=audit, key_sha512=key_sha512, lane=LANE)
        return {k: r.get(k) for k in ("protocol", "verdict", "why", "commitment", "quantity")} | {
            "profile": {k: ((r.get("answer") or {}).get("profile") or {}).get(k)
                        for k in ("lane", "period", "session", "draw_session", "source", "source_from")}}

    dev = json.loads((piece / S.DEVELOPMENT).read_text())
    report["piece"] = {k: dev.get(k) for k in ("verdict", "reasons", "coin_source")} | {
        "coins": {k: (dev.get("coins") or {}).get(k) for k in ("check", "draw")}, "coin": dev.get("coin")}
    report["row"] = row()
    print(f"[row] {json.dumps(report['row'])[:1500]}", flush=True)

    # control: a second draw session opened on the same commitment leaves a piece the compose refuses
    import importlib.util
    spec = importlib.util.spec_from_file_location("e2e_bench_arm", TREE / "benchmarks" / "pouw" / "pearl_c_vllm" / "bench_arm.py")
    BA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BA)
    from verity.protocols.spot.device.challenger import CoinError
    marks: list = []
    try:
        BA.LiveDraw(gate_cmd, PEER, LANE)(bytes.fromhex(record["commitment"]), marks.append)
        refused = None
    except CoinError as e:
        refused = str(e)
    _cli(log, "pearlplus (after a second draw)", ["benchmarks/one_stage/served_verdicts.py", "pearlplus", str(piece),
                                                  "--served", str(doc), "--verify", str(piece / "verify.json"), "--gate",
                                                  gate_cmd, "--key-sha512", key_sha512, "--main", str(main), "--coin",
                                                  str(piece / S.COIN_PUT)])
    report["control"] = {"second_draw_reserved": [m.get("reserved") for m in marks], "gate_refused": refused,
                         "row": row()}
    print(f"[control] {json.dumps(report['control'])[:1500]}", flush=True)

    auditor = out / "auditor"
    auditor.mkdir(exist_ok=True)
    (auditor / "pins.jsonl").write_bytes(host.pins.read_bytes())
    (auditor / "public").write_bytes((host.key / "public").read_bytes())
    report["ledger"] = [{"kind": e["kind"], "session": e.get("session"),
                         **{k: (e.get("body") or {}).get(k) for k in ("period", "by", "withheld", "families")}}
                        for e in _ledger(root)]
    (out / "e2e.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    control = report["control"]
    tiles = (((answer.get("public") or {}).get("work") or {}).get("credited_tiles"))
    ok = (report["row"]["verdict"] == "missing" and report["row"]["why"] == f"{S.DEV_VERDICT}: {TOY_ROW.format(tiles=tiles)}"
          and report["draw"]["from"] == "challenger" and dsid is not None
          and "one session a root" in str(control["gate_refused"]) and control["row"] == report["row"])
    print(f"E2E {'ok' if ok else 'FAILED'}: row {report['row']['verdict']} ({report['row'].get('protocol')}: "
          f"{report['row'].get('why')}), its draw "
          f"session {dsid}; a second draw on the run's commitment: {control['gate_refused']}; the row then "
          f"{control['row']['verdict']}", flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lean", help="the Lean verdict by served_passes.py lean with this pouw-pearlplus ('build': built in "
                   "PouwCheck.PearlPlus's package) in place of the stand-in")
    gt = sub.add_parser("gate")
    gt.add_argument("--root", type=Path, required=True)
    gt.add_argument("words")
    sv = sub.add_parser("serve")
    sv.add_argument("--root", type=Path, required=True)
    sv.add_argument("session")
    a = p.parse_args(argv)
    if a.verb == "gate":
        return gate(a.root, a.words)
    if a.verb == "serve":
        return serve(a.root, a.session)
    return run(a.out, a.lean)


if __name__ == "__main__":
    sys.exit(main())
