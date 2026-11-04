"""One-stage v2's audit (`verity/sampled-proofs/one-stage/v2`, `SampledProofs.OneStageZk`) on a real session: the case
the composition difftest's sampled proofs aims take for an acceptance (`verity/core/protocols/composition/tests`).

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/benchmarks/numerical/python \\
      python benchmarks/one_stage/v2_case.py --stage DIR --bin flock-circuit --zk-check sampled-proofs-one-stage-zk \\
      --main FILE --out DIR

`v2_case.sh` runs it on a CPU machine. The statement is `--stage`'s, `verity/ml/flock/tests/plain_hidden.py`'s (RoPE d64 at 32
instances, hidden outputs, the verifier's own program and partition), its registration the record its public file's roots
and domains give (`registration.record`, a ``stand-in`` window, the law ``subset`` of all 32), fixed before any coin.

* **The coins.** One withheld development session of the auditor host's own service on this machine (`a5_coins.Local`,
  under a key made here: its draws show the path, not an auditor's coins), on a fresh window root, its one family
  ``draw``: one coin after the window's root, then the draw's after the registration's digest (`ChallengerCoins`), which
  `coins.draw_first` reads as the verifier's sample (all 32 proof blocks, whatever the bytes).
* **The session.** `flock-circuit serve --zk --draw subset:32 --tree-from` and `prove --zk`, the session's coin tree sampled
  and opened by that same challenger session: the session has one client, so `serve` reaches it through a relay on this
  process (`relay`), which answers its hello and close itself and passes each ``tree`` and ``tree-open`` frame to the
  session's client (`RemoteChallenger.tree`, `tree_open`). The draw's coins and the coin tree are then in one signed
  release log, the one `OneStageZk.treesHeld` reads.
* **The case.** Once the session has closed: the parameters (the verifier's program, query, partition, ports, the law,
  the challenger's key and family, the window root, and ``proofs``: the circuit and public files' SHA-512s, no registered
  file), the record, the evidence (the draw, its coins as `Challenger.drawCoins` reads them, and ``proofs``: the session's
  record and rep files by SHA-512) and the blobs (the circuit, public, record, rep and table files), as the difftests'
  `serve` reads a case. `--zk-check serve` decides it; ``case/case.json`` under `--out` is the case and
  ``case/verdict.json`` the answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TREE / "verity" / "ml" / "flock" / "tests"))
sys.path.insert(0, str(TREE / "verity" / "ml" / "flock" / "verifier"))

import a5_coins as AC  # noqa: E402
import plain_hidden as PH  # noqa: E402
from record_text import session_text  # noqa: E402

from verity.core.protocols.sampled_proofs import challenger as Ch, coins as CC, remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as REG  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402

FAMILY = "draw"
COINS = 16
LAW = {"law": "subset", "k": PH.N}
DELTA = "1/20"
SESSION = "v2-case"
RECEIVED = "2026-11-04T00:00:00Z"


def sha512(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def canon(v) -> bytes:
    return canonical_json(v)


def header(pub: Path) -> dict:
    with open(pub, "rb") as f:
        return json.loads(f.readline())


def relay(listener: socket.socket, client: R.RemoteChallenger, secret: bytes, done: dict) -> None:
    """Serve `serve --tree-from`'s one connection on `listener` from the session's `client`: its hello (with `secret`)
    answered with the session's layout, each ``tree`` and ``tree-open`` frame asked of the challenger through `client`,
    its close answered here, the session staying open for the client to close. `done` gets how it ended."""
    conn, _ = listener.accept()
    wire = R._Wire(conn)
    try:
        if wire.recv() != {"type": "hello", "secret": secret.hex()}:
            raise Ch.CoinError("a client's first frame is hello with the session's secret")
        scheme, epoch, families = client._layout
        wire.send({"type": "layout", "layout": {"epoch": epoch, "families": [list(f) for f in families], "scheme": scheme}})
        while True:
            head = wire.recv()
            kind = head.get("type")
            if kind == "tree" and sorted(head) == ["nonce", "spec", "type"]:
                nonce, root, key = client.tree(head["spec"], None if head["nonce"] is None else bytes.fromhex(head["nonce"]))
                wire.send({"type": "tree", "tree": {"key": key.hex(), "nonce": nonce.hex(), "root": root.hex()}})
                done["trees"] = done.get("trees", 0) + 1
            elif kind == "tree-open" and sorted(head) == ["after", "n", "round", "stream", "tree", "type"]:
                _, words = client.tree_open(bytes.fromhex(head["tree"]), head["stream"], head["round"], head["n"],
                                            bytes.fromhex(head["after"]))
                wire.send({"type": "tree-coins", "tree-coins": [f"{w:032x}" for w in words]})
                done["rounds"] = done.get("rounds", 0) + 1
            elif head == {"type": "close"}:
                wire.send({"type": "closed"})
                done["ended"] = None
                return
            else:
                raise Ch.CoinError(R.SEQUENCE)
    except (Ch.CoinError, OSError, ValueError, KeyError) as e:
        done["ended"] = str(e)
        try:
            wire.send({"type": "refused", "reason": str(e)})
        except OSError:
            pass
    finally:
        conn.close()


def prove(a: argparse.Namespace, stage: Path, pin: str, tree_from: str, secret_file: Path, out: Path) -> Path:
    """One --zk session of all 32 instances under `out/sessions`, its coin tree from `tree_from`: its directory."""
    st = ["--circuit", str(stage / "circuit.txt"), "--tables", str(stage / "tables")]
    with open(out / "serve.log", "w") as log:
        srv = subprocess.Popen([a.bin, "serve", "--zk", "--listen", "127.0.0.1:0", "--out", str(out / "sessions"),
                                "--instances", str(stage / f"pub-{PH.N}.bin"), *st, "--pin", pin, "--operator", "loopback",
                                "--draw", f"subset:{PH.N}", "--sessions", "1", "--tree-from", tree_from,
                                "--tree-secret-file", str(secret_file)], stdout=log, stderr=subprocess.STDOUT)
    port = None
    while port is None:
        text = (out / "serve.log").read_text()
        port = next((ln.split(":")[-1] for ln in text.splitlines() if ln.startswith("SERVING 127.0.0.1:")), None)
        if port is None and srv.poll() is not None:
            raise SystemExit(f"serve exited {srv.returncode} before SERVING: {text[-500:]}")
        if port is None:
            threading.Event().wait(0.5)
    with open(out / "prove.out", "w") as o, open(out / "prove.err", "w") as e:
        rc = subprocess.run([a.bin, "prove", "--zk", "--verifier", f"127.0.0.1:{port}", "--instances",
                             str(stage / f"inst-{PH.N}.bin"), *st], stdout=o, stderr=e).returncode
    if rc or srv.wait():
        raise SystemExit(f"prove exit {rc}, serve exit {srv.returncode}: {(out / 'prove.err').read_text()[-500:]}")
    (session,) = [p.parent for p in (out / "sessions").glob("*/session.json")]
    return session


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, help="plain_hidden.py's --out")
    ap.add_argument("--bin", required=True, help="flock-circuit (the proving build)")
    ap.add_argument("--zk-check", required=True, help="sampled-proofs-one-stage-zk")
    ap.add_argument("--main", required=True, help="the verifier's copy of main's first-parent history, a commit a line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stage, out = Path(a.stage), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="v2-case-"))

    program, query = PH.program()
    part = P.partition(program, query)
    pub = stage / f"pub-{PH.N}.bin"
    h = header(pub)
    fv = h["frame_v3"]
    ports = {p: {"leaves": PH.N, "domain": fv["domain_ids"][p]} for p in sorted(fv["roots"])}
    leaf_layer = sha512(pub.read_bytes())
    window_root = secrets.token_bytes(64).hex()
    rec = REG.record(program, query, LAW, h["scheme"],
                     [{"port": p, "root": fv["roots"][p], **ports[p]} for p in ports],
                     {"kind": "stand-in", "of": "plain_hidden.py's RoPE d64 statement (v2_case.py)"}, leaf_layer=leaf_layer)
    after = bytes.fromhex(REG.digest(rec))
    matches = REG.matches_public(rec, h["scheme"], fv["roots"], fv["domain_ids"], h["instances"])
    if not matches.ok:
        raise SystemExit(f"the public file is not the registration's: {matches.to_json()}")

    main_lines = R.main_lines(a.main)
    local = AC.Local(scratch / "challenger", main_lines)
    address, secret, _ = local.open(window_root, [[FAMILY, 1 + COINS]], scheme=AC.SCHEME, withhold=True)
    client = AC.connect_when_listening(address, bytes.fromhex(secret))
    coins = CC.ChallengerCoins(Ch.LiveRecord(client, FAMILY))
    coins.commit(bytes.fromhex(window_root))
    coins(Ch.COIN_BYTES)
    segment = coins.commit(after)
    mine = D.derive(program, query, LAW, None)
    draw = CC.draw_first(CC.OwnRandomness(coins), mine, P.units(program, query))

    relay_secret = secrets.token_bytes(32)
    (scratch / "relay-secret").write_text(relay_secret.hex())
    listener = socket.create_server(("127.0.0.1", 0))
    done: dict = {}
    t = threading.Thread(target=relay, args=(listener, client, relay_secret, done), daemon=True)
    t.start()
    pin = sha512((stage / "circuit.txt").read_bytes())
    session = prove(a, stage, pin, f"127.0.0.1:{listener.getsockname()[1]}", scratch / "relay-secret", out)
    t.join()
    client.close()
    coins.settle()
    if done.get("ended") is not None or not done.get("trees"):
        raise SystemExit(f"the relay ended: {done}")
    directory, ledger, pins_log, key = local.finish(scratch / "finished")
    R.trusted_session(directory, [ledger], {"root": window_root}, key, AC.lms_verify)
    led, pins = R.ledger_lines(ledger), AC.lines(pins_log)
    Ch.pinned_session(led, {"root": window_root}, key, pins, main_lines, AC.lms_verify)
    pin_commit = AC.opened_pin(led, local.session)
    kept_main = main_lines[:main_lines.index(pin_commit) + 1]

    record = session_text(json.loads((session / "session.json").read_text())).encode()
    reps = {f.name: f.read_bytes() for f in sorted(session.glob("*.bin"))}
    tables = [f.read_bytes() for f in sorted((stage / "tables").glob("*")) if f.is_file()]
    circuit = (stage / "circuit.txt").read_bytes()
    blobs = [circuit, pub.read_bytes(), record, *reps.values(), *tables]
    proved = D.proved(draw)
    name = session.name
    evidence = {"coins": {"ledger": led, "main": kept_main, "pins": pins, "segment": segment.to_json(),
                          "session": R.session_files(directory)},
                "draw": draw, "instances": len(proved), "proved_units": proved,
                "proofs": {"sessions": [{"name": name, "record": sha512(record),
                                         "reps": {f: sha512(b) for f, b in reps.items()}}]},
                "receipt": REG.receipt(rec, "v2_case.py", received=RECEIVED), "served": None, "session_roots": None,
                "sessions": [name], "statement": {}}
    params = {"anchors": [], "challenger": {"key": key.to_json(), "label": FAMILY}, "delta": DELTA, "law": LAW,
              "leaf_layer": leaf_layer, "log": [], "partition": part, "ports": ports,
              "program": json.loads(canon(encode_program(program))), "query": query, "registered": None,
              "scheme": h["scheme"], "window_root": window_root, "work_table": None,
              "proofs": {"circuit": sha512(circuit), "public": sha512(pub.read_bytes()), "registered": None}}
    case = {"blobs": [b.hex() for b in blobs], "commitment": canon(rec).hex(), "evidence": canon(evidence).hex(),
            "params": params}
    line = canon(case).decode()
    (out / "case").mkdir(exist_ok=True)
    (out / "case" / "case.json").write_text(line + "\n")
    got = subprocess.run([a.zk_check, "serve"], input=line + "\n", capture_output=True, text=True)
    answer = json.loads(got.stdout) if got.stdout.strip() else {"stderr": got.stderr[-2000:], "rc": got.returncode}
    summary = {"answer": answer, "relay": done, "session": name, "proved_units": proved, "window_root": window_root,
               "registration": REG.digest(rec), "pin": pin_commit, "blobs": len(blobs),
               "case_bytes": len(line)}
    (out / "case" / "verdict.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if answer.get("verdict") == "accept" else 1


if __name__ == "__main__":
    raise SystemExit(main())
