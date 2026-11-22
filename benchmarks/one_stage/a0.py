"""The one-stage audit's shared steps, which `a5`, `records_audit` and the other runs import: the verifier's sample, M0's
session, the Lean verifier's verify (`lean_verify`, `VerifyMembers`), the archive, and the audit record with its integrity
profile.

* **Stand-in commitment.** M0 commits the captured values afresh (`frame-v3-sha512`, `hm96-sha512/row/v1` leaves, fresh
  OS salts). The values were read under the served run's `vllm-v1` root and replayed bit-exact there by `vu_export`; that
  link is provenance, not proof, and the docket says so.
* **Partition.** `Q_template_instance` v0 over the population program: one unit per instance, no width rule.
* **The verifier.** Lean's `verity-verify-proof verify` replays the session offline with the circuit and the public
  file held by content; Lean also checks U1-U3 on the sample, and M0's own verifier (`flock-circuit serve`) checks it live.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import select
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from verity.commitments.identity import identity_digest, identity_digest_sha512
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.circuits.codec import program_digest
from verity.circuits.program import Program
from verity.commitments.domains import FLOCK_CIRCUIT_BINDING, HM96_SHA512_ROW_V1
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch, coins as CC
from verity.protocols.proof.spot.verifier.one_stage import audit, partition as P, verifier as V
from verity.protocols.proof.spot.verifier.one_stage.proofs import file_sha512 as sha512_file, verifier_env
from verity.protocols.proof.spot.verifier.one_stage.verdict import Verdict
from verity.protocols.proof.spot.verifier.one_stage.lean_child import (LEAN_PANIC, VERIFY_EXITS, exit_why, panic_line,
                                                                        refuse_exit, refuse_panic)
from verity.protocols.proof.spot.verifier.one_stage.settings import ABORT_ON_PANIC, lean_env


SCHEME = "frame-v3-sha512/hm96-sha512"
#: M0's set-bound frame domains (verity_flock.circuit.BINDING_TAG), each tag with its identity digest; v1 is M0's before
#: 64-byte bindings, for a pinned tree that old (--binding)
STAND_IN_BINDINGS = {FLOCK_CIRCUIT_BINDING: identity_digest_sha512,
                     "verity/flock-ir-frame/binding/v1": lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))}
STAND_IN_BINDING = FLOCK_CIRCUIT_BINDING
STATEMENT_ID = "verity/flock-circuit"
#: a statement's key under which its circuit file is linked into an archive rather than put there (`archive_put`'s ``link``)
ARCHIVE_LINK = "archive_link"


def environ(env: dict | None) -> dict:
    """This process's environment with ``env`` over it, a None value removing its variable (flock-circuit takes the host's
    witness when FC_HOST_WITNESS is set at all), under `lean_env`."""
    return lean_env({k: v for k, v in {**os.environ, **(env or {})}.items() if v is not None})


def run(cmd: list[str], log: Path, env: dict | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=environ(env), timeout=timeout)
    log.write_text(f"$ {' '.join(cmd)}\n# rc {p.returncode} in {time.time() - t:.1f} s\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    refuse_panic(p.stderr)
    return p


#: the development challenger's layout: one family, the coins one sample reads
DEV_SCHEME, DEV_FAMILY = "verity/one-stage/dev-draw/v1", "draw"


def challenger_sample(lean: str, args: list[str], sampling_design: dict, population: int, after: str | bytes,
                    log: Path) -> subprocess.CompletedProcess:
    """Lean's sample, ``verity-verify-proof draw ARGS --stream-file F``, on a development challenger's coins (the verifier has
    none of its own). The challenger is an in-process `LiveChallenger` on its own entropy (`secrets.token_bytes`),
    committed before any coin opens. Its one family holds the coins the sample reads, opened after `after` (the
    docket's digest) by `ChallengerCoins`. F holds the bytes `sampling_design` reads over `population`, each stream once, in
    Lean's `Draw.sampleOn` layout (`coins.draw_first`'s). ``<log stem>.coins.json`` keeps the challenger's commitment and
    the segment, from which `coins.challenger_stream` replays F. Its samples show the path, not an auditor's coins: those
    are `a5_coins`' (`a5.py --challenger`)."""
    reads: list[int] = []
    CC.draw_first(CC.OwnRandomness(lambda n: reads.append(n) or bytes(n)), sampling_design, population)
    need = sum(reads)
    ch = Ch.LiveChallenger(DEV_SCHEME, 0, [(DEV_FAMILY, max(1, -(-need // Ch.COIN_BYTES)))], secrets.token_bytes)
    coins = CC.ChallengerCoins(Ch.LiveRecord(ch, DEV_FAMILY))
    coins.commit(bytes.fromhex(after) if isinstance(after, str) else bytes(after))
    stream = log.with_name(log.stem + ".coins.bin")
    stream.write_bytes(coins(need))
    log.with_name(log.stem + ".coins.json").write_text(json.dumps(
        {"source": "development challenger (in-process, secrets.token_bytes)", "commitment": ch.commitment.to_json(),
         "segment": coins.segments[-1].to_json(), "need": need}, sort_keys=True))
    return run([lean, "draw", *args, "--stream-file", str(stream)], log)


def template_program_ok(meta: dict, defn) -> bool:
    """META's one `program_digests` entry is this template's: keyed by its descriptor id (or its display id, as M0 keys it
    through 68ae79f2), valued core's `program_digest` of its Program."""
    pd = meta.get("program_digests", {})
    return len(pd) == 1 and next(iter(pd)) in (P.template_id(defn), defn.id) and next(iter(pd.values())) == program_digest(Program(defn))


def bindings(meta: dict, header: dict, rec: dict, cls: str) -> dict:
    """What the staged statement binds, against the docket: the partition, the program, the global unit indices of
    `Q_template_instance` (instance i is unit i), and one class per instance."""
    n = rec["population"]
    got = {"partition": meta.get("partition"), "program": meta.get("program_sha512"),
           "indices": header["units"]["indices"] == list(range(n)), "classes": header["units"].get("classes") == [cls] * n}
    ok = {"partition": got["partition"] == {"rule": P.FORMAT, "digest": rec["partition"]},
          "program": got["program"] == rec["program"], "indices": got["indices"], "classes": got["classes"]}
    return {"ok": all(ok.values()), "each": ok, "class_sha512": cls}


def public_header(stage_dir: Path, n: int) -> dict:
    with open(stage_dir / f"pub-{n}.bin", "rb") as f:
        return json.loads(f.readline())


def hidden_outputs(meta: dict) -> bool:
    """M0 from hidden outputs (META `input_ports`): every output port is a row after the input rows, and no domain binds the
    set's unsalted content digest."""
    return "input_ports" in meta


def ports(meta: dict, n: int) -> dict[str, tuple[str, int]]:
    """Each port the layout derives: its schema and leaf count (before hidden outputs, each output port a tree of u16 words)."""
    out = {name: (HM96_SHA512_ROW_V1, n) for name, *_ in meta["ports"]}
    if not hidden_outputs(meta):
        out.update({name: ("u16", n * words) for name, words in meta["out_ports"]})
    return out


def stand_in_ports(name: str, content_digest: str, meta: dict, n: int) -> dict[str, dict]:
    """The verifier's own derivation of the stand-in's domains: M0's frame domains bound to the set (before hidden outputs,
    its content digest too), the range, the port and its schema."""
    out = {}
    content = {} if hidden_outputs(meta) else {"content_digest": content_digest}
    for p, (schema, count) in ports(meta, n).items():
        b = STAND_IN_BINDINGS[STAND_IN_BINDING](STAND_IN_BINDING, {"set": name, **content, "lo": 0, "hi": n, "port": p, "schema": schema})
        out[p] = {"leaves": count, "domain": CommitmentDomain(b, -1, RangeIndexedDomain(0, count), hash="sha512").domain_id.hex()}
    return out


def registered_roots(header: dict, want: dict[str, dict]) -> list[dict]:
    fv = header["frame_v3"]
    return [{"port": p, "root": fv["roots"][p], "leaves": want[p]["leaves"], "domain": fv["domain_ids"][p]} for p in want]


def stand_in(set_dir: str) -> dict:
    prov = json.loads((Path(set_dir) / "manifest.json").read_text()).get("provenance", {})
    return {"kind": "stand-in", "of": "vllm-v1 commitment of a served run", "input_set": Path(set_dir).name, "row": prov.get("row"), "run": prov.get("research_run") or prov.get("run"),
            "run_root": prov.get("run_root"), "model": prov.get("model"), "program_digests": prov.get("program_digests"),
            "link": "provenance, not proof: vu_export read these values under run_root and replayed each bit-exact"}


def own_challenger(env: dict | None = None, zk: bool = False) -> bool:
    """Whether `session`, given ``env`` and ``zk``, takes its coin tree from a challenger of its own (FLOCK_CHALLENGER=1 on the
    keyed coin tree)."""
    senv = {**os.environ, **(env or {})}
    return _flock_challenger(senv) == "1" and (zk or senv.get("FC_COINS") == "tree")


def challenger_off(env: dict | None = None) -> bool:
    """Whether ``env`` opts its sessions out of every challenger (FLOCK_CHALLENGER=0, a development run's): each samples its
    coin tree itself, even where the audit's challenger session could give it one (`a5.window_trees`)."""
    return _flock_challenger({**os.environ, **(env or {})}) == "0"


def _flock_challenger(senv: dict) -> str | None:
    """FLOCK_CHALLENGER in a session's environment, read here alone (`own_challenger`, `challenger_off`)."""
    return senv.get("FLOCK_CHALLENGER")


#: `session` reads its verifier's log for ``SERVING`` this often, for this long at most (seconds): the prover starts
#: within a read of the verifier's listening, and the wait is outside the prover's clock
SERVING_POLL, SERVING_WAIT = 0.02, 900.0


def session(bin_: str, st: dict, n: int, sampling_design: str, port: int, out: Path, sample_file: Path | None = None,
            zk: bool = False, env: dict | None = None, prove_timeout: float = 3600,
            tree_from: Sequence[str] | None = None, prover_prefix: Sequence[str] = (),
            prover_args: Sequence[str] = ()) -> tuple[dict | None, dict | None]:
    """One live session: `serve --draw` (the verifier's side: docket, the sample, coins, the verdict) and `prove`.
    Under FLOCK_CHALLENGER=1 a session on the keyed coin tree (`zk`, or `FC_COINS=tree`) takes its tree from a challenger of
    its own under ``out/challenger`` (`verity_flock.challengers`, `serve --tree-from`): ``live["challenger"]`` is its exit
    and its audit of the session's record, and a session whose challenger failed is refused. With ``tree_from`` (the
    caller's ``--tree-from`` flags) it takes its tree from the challenger those name instead, whose audit is the caller's.
    `sample_file`: the server serves the verifier's own sample from that file (`--draw-file`) instead of sampling one. `zk`: both
    sides run the masked protocol (`--zk`); without it the Lean verifier accepts a subset sample at hidden outputs and one table
    (`Flock.ProvedScope`). `env`: added to
    both sides' environment (a zk_served prover's, `zk_server_env`). `prove_timeout`: the prover's limit in seconds.
    ``st["tmpl_unit"]``: a templated statement's unit rows at their positions, which both sides read (`--tmpl-unit`).
    `prover_prefix`: the command the prover runs under (a card's lease, ``gpu-lease 1 --wait --``), and `prover_args` more of
    `prove`'s (``--gpu``)."""
    out.mkdir(parents=True, exist_ok=True)
    d = st["dir"]
    tables = d / "tables"
    zf = ["--zk"] if zk else []
    tu = ["--tmpl-unit", str(st["tmpl_unit"])] if st.get("tmpl_unit") else []
    ch = None
    if tree_from is None and own_challenger(env, zk):
        from verity_flock import challengers as CH
        ch = CH.start(out / "challenger")
    serve_log = open(out / "serve.log", "w")
    serve = subprocess.Popen([bin_, "serve", *zf, "--listen", f"127.0.0.1:{port}", "--out", str(out / "sessions"),
                              "--instances", str(d / f"pub-{n}.bin"), "--circuit", str(d / "circuit.txt"), *tu, "--tables", str(tables),
                              "--pin", st["pin"], "--operator", "one-stage-a0",
                              *(["--draw-file", str(sample_file)] if sample_file else ["--draw", sampling_design]), "--sessions", "1",
                              *(ch[1] if ch else tree_from or [])],
                             stdout=serve_log, stderr=subprocess.STDOUT, text=True, env=environ(env))
    got = None
    try:
        late = time.monotonic() + SERVING_WAIT
        while "SERVING" not in (out / "serve.log").read_text() and serve.poll() is None and time.monotonic() < late:
            time.sleep(SERVING_POLL)
        p = run([*prover_prefix, bin_, "prove", *zf, *prover_args, "--verifier", f"127.0.0.1:{port}", "--instances", str(d / f"inst-{n}.bin"),
                 "--circuit", str(d / "circuit.txt"), *tu, "--tables", str(tables), "--runs", "1", "--warm", "0"], out / "prove.log", env=env,
                timeout=prove_timeout)
        live = next((json.loads(line[5:]) for line in p.stdout.splitlines() if line.startswith("LIVE\t")), None)
        if live is None:
            refused = next((line for line in p.stdout.splitlines() if line.startswith("REFUSED\t")), "")
            live = {"accepted": False, "why": f"prover exited {p.returncode}: {refused[:600]} {p.stderr[-600:]}".strip()}
        serve.wait(timeout=600)
    finally:
        if serve.poll() is None:
            serve.kill()
        serve_log.close()
        if ch:
            got = CH.end(out / "challenger", ch[0], CH.records(out / "sessions"))
    if got is not None:
        live = {**live, "challenger": got}
        if not got["ok"]:
            live.update(accepted=False, why=f"challenger: {json.dumps(got)[:600]}")
    recs = sorted((out / "sessions").glob("*/session.json"))
    return (json.loads(recs[-1].read_text()) if recs else None), live


def templated(st, d: Path, tmpl_unit_bin: str | None) -> dict:
    """A templated statement's reads beyond a flat one's (none for a flat one): `verify`'s statement, and the prover's unit
    rows at their positions (``d/unit.placed``, `--tmpl-unit`), which `flock-tmpl-unit` places from the circuit's template
    section and META's layout mode (``d/unit.tmpl``, ``d/unit.layout``)."""
    from verity_flock import circuit as FC

    if st.meta.get("statement") != FC.TMPL_STATEMENT:
        return {}
    if not tmpl_unit_bin:
        raise SystemExit("a templated statement's prover reads its unit placed by flock-tmpl-unit: give --tmpl-unit-bin")
    placed = d / "unit.placed"
    if not placed.exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / "unit.tmpl").write_text(FC.template_of(st.text))
        (d / "unit.layout").write_text(st.meta["layout"])
        p = run([tmpl_unit_bin, str(d / "unit.tmpl"), "--layout", st.meta["layout"], "--out", str(placed)], d / "tmpl-unit.log")
        if p.returncode:
            raise SystemExit(f"flock-tmpl-unit: exit {p.returncode} ({d / 'tmpl-unit.log'})")
    return {"statement": FC.TMPL_STATEMENT, "tmpl_unit": placed}


def lean_sample_test(lean: str, cases: list[dict], out: Path, name: str) -> list[dict]:
    f = out / f"{name}.cases.json"
    f.write_text(json.dumps(cases))
    p = run([lean, "draw-test", str(f)], out / f"{name}.lean.log")
    return [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("DRAWCHECK")]


def zk_server_env(st: dict) -> dict:
    """What a zk_served prover (`FC_ZK_SERVER=1`) sets its Lean `Hello` up on (live PROTOCOL.md §10.4): the statement
    `lean_verify` verifies under, and the statement's partition and program; and `ABORT_ON_PANIC` for the Lean server."""
    args = [x for k in ("partition", "program") if st.get(f"{k}_file") for x in (f"--{k}", str(st[f"{k}_file"]))]
    return {"FC_ZK_SERVER_STATEMENT": st.get("statement", STATEMENT_ID), "FC_ZK_SERVER_HELLO_ARGS": " ".join(args),
            ABORT_ON_PANIC: "1"}


#: a `--zk` session's prover with no ZK server, as SPOT[Flock]'s proof server runs it: `flock-circuit prove --zk` itself,
#: its witness built where the prover builds it (`zk_cost.py`'s `dev` arm, which Figure 1 prices)
DIRECT = {"FC_ZK_SERVER": "0", "FC_HOST_WITNESS": None}


def prover_env(zk_server: str | None, st: dict) -> dict:
    """A `--zk` session's prover environment: behind the ZK server's send check when ``zk_server`` names Lean's
    `flock-zk-server` (the core protocols' path), else `DIRECT`."""
    if zk_server:
        return {"FC_ZK_SERVER": "1", "FC_ZK_SERVER_LEAN": zk_server, **zk_server_env(st)}
    return dict(DIRECT)


def identity(f: Path) -> tuple | None:
    """File ``f``'s identity, the same for every link of one file; None for a file that is not there."""
    if not f.exists():
        return None
    s = f.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def bound_files(st: dict) -> list[str]:
    """What statement ``st`` is verified against besides its circuit, public file and tables, each by `verify`'s flag
    for it (the file is ``st["<flag>_file"]``): its partition, program and registered reads where it has them, and with
    registered reads the window's registered circuit record (``interface``) where it has one."""
    return [x for x in ("partition", "program", "registered", "interface")
            if st.get(f"{x}_file") and (x != "interface" or st.get("registered_file"))]


def archive_files(st: dict, n: int) -> list[Path]:
    """What `lean_verify` archives for statement ``st`` on ``n`` units: the circuit, the public file, the tables, and
    what else the statement is verified against (`bound_files`)."""
    d = st["dir"]
    return [d / "circuit.txt", d / f"pub-{n}.bin", *sorted(f for f in (d / "tables").glob("*") if f.is_file()),
            *(Path(st[f"{x}_file"]) for x in bound_files(st))]


def archive_put(lean: str, archive: Path, files: Sequence[Path], log: Path, held: dict | None = None,
                link: Collection[Path] = ()) -> dict[Path, str]:
    """Each of ``files`` in ``archive`` under its SHA-512 (Lean's ``archive-put``, its log ``log``), and each one's
    key there; a file Lean could not put has none. ``held``, a dict a run's archives share: a file archived once, by its
    identity (`identity`), is linked into a later archive rather than read again, and a call whose every file is held
    starts no process. Lean holds an object to its key when it reads it (`archiveGet`).

    A file of ``link`` that no earlier call archived is linked into the archive under its SHA-512, hashed here, and not
    put, so the archive holds no second copy of it. A wrong hash here is Lean's to refuse when it reads the object, and
    where the link can't be made (another filesystem) the file is put as without it. A link shares the file's bytes: a
    write in place through either name changes both."""
    ids = {f: identity(f) if held is not None else None for f in files}
    linked: dict[Path, tuple[str, Path]] = {}

    def known(f: Path) -> tuple[str, Path] | None:
        pre = held.get(ids[f]) if ids[f] is not None else None
        return pre if pre is not None and pre[1].exists() else linked.get(f)

    for f in dict.fromkeys(f for f in files if f in link):
        if known(f) is not None or not f.is_file():
            continue
        key = sha512_file(f)
        try:
            (archive / "sha512").mkdir(parents=True, exist_ok=True)
            if not (archive / "sha512" / key).exists():
                os.link(f, archive / "sha512" / key)
        except OSError as e:
            print(f"A0 archive-link: {f} not linked into {archive} ({type(e).__name__}: {e}), so it is put, a copy",
                  file=sys.stderr)
            continue
        linked[f] = (key, archive / "sha512" / key)
        if ids[f] is not None:
            held[ids[f]] = linked[f]
    new, seen = [], set()
    for f in files:
        if f in linked:
            continue
        if ids[f] is None or (known(f) is None and ids[f] not in seen):
            new.append(f)
            seen.add(ids[f])
    put = {}
    if new:
        p = run([lean, "archive-put", str(archive), *map(str, new)], log)
        put = {x[1]: x[0] for x in (line.split() for line in p.stdout.splitlines()) if len(x) == 2}
        for f in new:
            if ids[f] is not None and str(f) in put:
                held[ids[f]] = (put[str(f)], archive / "sha512" / put[str(f)])
    keys = {}
    for f in files:
        pre = known(f)
        if pre is None:
            if str(f) in put:
                keys[f] = put[str(f)]
            continue
        keys[f] = pre[0]
        if not (archive / "sha512" / pre[0]).exists():
            (archive / "sha512").mkdir(parents=True, exist_ok=True)
            os.link(pre[1], archive / "sha512" / pre[0])
    return keys


#: `VerifyMembers` waits this long for a member's answer, as `run` waits for a `verify` process, and this long for the
#: process to exit once its standard input is closed or its output has ended (seconds), by this clock
MEMBER_WAIT, EXIT_WAIT = 3600.0, 60.0
monotonic = time.monotonic


class VerifyMembers:
    """Lean's ``verify-members``, one process for a run's members (`Flock.VerifyMembers`): started on the run's
    registered program (``program``, its key in ``archive``), which it reads once, it takes a member as one line, the JSON
    array of `verify`'s arguments for it, and answers with what `verify` prints for that member and then
    ``MEMBER {"exit":E,"index":I}``, E `verify`'s exit status and I the line's place.

    `verify` hands it the next member and writes that member's log as `run` writes a `verify` process's, with one more
    line saying how it was run, so the log's readers read either. A member is handed in once, in the members' order. A
    member the process does not answer (it ended, it wrote nothing for `MEMBER_WAIT` seconds, or it answered another place)
    has no verdict, and neither has any member after it: the process is ended, by its PID, and never started again.
    `close` ends it and writes its own log (``log``); until then its standard error is the ``.stderr`` file beside it.

    It runs in this process's environment as it is, not in `verifier_env`. A `verify` process exits after its member;
    this one stays for the run, beside the run's provers, and under the verifier's allocator setting it would hold
    every page it freed once it had read the program. So it starts under the allocator's default unless this process's
    environment sets a value."""

    def __init__(self, lean: str, archive: Path, program: str, log: Path):
        self.cmd = [lean, "verify-members", "--archive", str(archive), "--program", program]
        self.log, self.stderr, self.started = log, log.with_suffix(".stderr"), time.time()
        # its standard error is a file of this run's alone, which `_stderr` reads through an opening of its own; the pipes
        # are unbuffered, read and written through their descriptors
        with self.stderr.open("wb") as err:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=0,
                                         env=environ(None))
        self.members, self.ended, self.statuses = 0, None, []
        self._buf, self._err_read, self._ends = b"", 0, []

    def _stop(self, why: str, kill: bool = False) -> None:
        """The process answers no member from here: ``why``, with its exit status, is every later member's reason."""
        if kill and self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=EXIT_WAIT)
        except subprocess.TimeoutExpired as e:
            self.proc.kill()
            self.proc.wait()
            why = f"{why}, and did not exit in {e.timeout:.0f} s"
        self.ended = f"{why} (it exited {self.proc.returncode})"

    def _line(self, wait: float) -> bytes | str:
        """The process's next line of standard output; or, a str, why there is none: its output ended, or it wrote
        nothing for ``wait`` seconds."""
        late = monotonic() + wait
        while b"\n" not in self._buf:
            left = late - monotonic()
            if left <= 0 or not select.select([self.proc.stdout], [], [], left)[0]:
                return f"wrote nothing for {wait:.0f} s"
            got = os.read(self.proc.stdout.fileno(), 1 << 16)
            if not got:
                return "ended"
            self._buf += got
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _stderr(self) -> str:
        """What the process wrote on standard error since the last call."""
        with self.stderr.open("rb") as f:
            f.seek(self._err_read)
            got = f.read()
        self._err_read += len(got)
        return got.decode(errors="replace")

    def verify(self, args: Sequence[str], log: Path) -> tuple[str, str]:
        """The next member, `verify`'s arguments ``args``: what `verify` printed for it, nothing where the process did not
        answer it; and why it has no verdict if it has none (`verify`'s exit status, or what the process did instead).
        ``log``: the member's log."""
        i, t = self.members, time.time()
        self.members += 1
        printed, status = [], None
        if self.ended is None:
            line = json.dumps(list(args)).encode() + b"\n"
            try:
                while line:
                    line = line[os.write(self.proc.stdin.fileno(), line):]
            except OSError as e:
                self._stop(f"the kept verifier could not be handed member {i} ({e})")
        while self.ended is None and status is None:
            line = self._line(MEMBER_WAIT)
            if isinstance(line, str):
                self._stop(f"the kept verifier {line} before member {i}'s MEMBER line", kill=line != "ended")
            elif not line.startswith(b"MEMBER "):
                printed.append(line.decode(errors="replace"))
            else:
                self._ends.append(line.decode(errors="replace"))
                try:
                    end = json.loads(line[7:])
                except ValueError as e:
                    end = f"not JSON ({e})"
                if isinstance(end, dict) and end.get("index") == i and isinstance(end.get("exit"), int):
                    status = end["exit"]
                    self.statuses.append(status)
                else:
                    self._stop(f"the kept verifier answered member {i} with {self._ends[-1][:200]}", kill=True)
        stderr = self._stderr()
        panic = panic_line(stderr)
        refused = (f"{LEAN_PANIC}: {panic}" if panic is not None
                   else None if status is None else exit_why(status, VERIFY_EXITS))
        if refused is not None:
            status = None
        # what the process printed for a member it did not answer is kept, and is no verdict to any reader of the log
        stdout = "".join(("" if status is not None else "not taken: ") + x + "\n" for x in printed)
        why = (refused if refused is not None else f"no verdict (rc {status}): {stderr[-400:]}" if status is not None
               else f"no verdict: {self.ended}: {stderr[-400:]}")
        log.write_text(f"$ {self.cmd[0]} verify {' '.join(args)}\n# rc {status} in {time.time() - t:.3f} s\n"
                       f"# run as member {i} of: {' '.join(self.cmd)}" + ("" if status is not None else f"\n# {self.ended if refused is None else why}")
                       + f"\n--- stdout\n{stdout}\n--- stderr\n{stderr}")
        return (stdout if status is not None else ""), why

    def close(self) -> int:
        """Ends the process, its standard input closed so that it exits once it has answered what it was handed: its exit
        status, and its log as `run` writes one (its members' ``MEMBER`` lines; their verdicts are in their own logs).
        Where it answered every member and then did not end as `verify-members` documents (0 when every member's exit
        was 0, else 1), wrote a panic line after its last answer or was killed, raises `LeanPanic`: the verdicts it
        gave were read before its end, and stand on nothing."""
        answered = self.ended is None
        if self.ended is None:
            self.proc.stdin.close()
            while isinstance(line := self._line(EXIT_WAIT), bytes):
                self._ends.append(line.decode(errors="replace"))
            self._stop("closed after its last member" if line == "ended" else f"the kept verifier {line} once its input closed",
                       kill=line != "ended")
        self.log.write_text(f"$ {' '.join(self.cmd)}\n# rc {self.proc.returncode} in {time.time() - self.started:.1f} s\n"
                            f"# {self.members} members handed in; {self.ended}\n--- stdout\n"
                            + "".join(x + "\n" for x in self._ends) + self._buf.decode(errors="replace")
                            + f"\n--- stderr\n{self.stderr.read_text(errors='replace')}")
        tail = self._stderr()
        self.stderr.unlink()
        for f in (self.proc.stdin, self.proc.stdout):
            f.close()
        if answered:
            refuse_panic(tail)
            refuse_exit(self.proc.returncode, {0 if all(s == 0 for s in self.statuses) else 1})
        return self.proc.returncode


_CLAIMS = threading.Lock()


def archiving(held: dict | None, ident: tuple | None) -> AbstractContextManager:
    """The lock a run's checks (`lean_verify`, sharing ``held``) on the circuit file of identity ``ident`` archive their
    files under (`archive_put`): the first puts them, and one running beside it waits, then links what the first put."""
    if held is None:
        return nullcontext()
    with _CLAIMS:
        return held.setdefault(("archiving", ident), threading.Lock())


#: The keys of a statement's registered file: its path, the object the audit built and that object's SHA-512 as written.
REGISTERED_KEYS = ("registered_file", "registered", "registered_sha512")


def registered_written(path: Path, doc: dict) -> dict:
    """The registered file ``doc`` written at ``path``, as a statement's keys (`REGISTERED_KEYS`): the file read after
    proving is held to the SHA-512 of these bytes (`registered_held`), and the audit's own readers take ``doc``."""
    body = json.dumps(doc, sort_keys=True).encode()
    path.write_bytes(body)
    return {"registered_file": path, "registered": doc, "registered_sha512": hashlib.sha512(body).hexdigest()}


def registered_changed_why(path: Path) -> str:
    return (f"registered file {path}: not the bytes the audit built: a registered file read after proving is the one "
            "the audit wrote, by its SHA-512, never a file another process changed")


def registered_held(st: dict, keys: Mapping[Path, str]) -> str | None:
    """Why the archive's copy of the statement's registered file (its key in ``keys``, the SHA-512 Lean holds the object
    it reads to) is not the file the audit built (``st["registered_sha512"]``); None where it is, or ``st`` has none."""
    want, f = st.get("registered_sha512"), st.get("registered_file")
    return None if want is None or f is None or keys.get(Path(f)) == want else registered_changed_why(Path(f))


def lean_verify(lean: str, st: dict, n: int, out: Path, zk: bool = False, log: str = "lean-verify",
                flags: Sequence[str] = (), members: VerifyMembers | None = None) -> dict:
    """The verifier: Lean `verity-verify-proof verify` on the session's record, with the circuit and the public file
    held by content (`--archive`, SHA-512 keys), and the verifier's own registered reads (`st["registered_file"]`: a5's,
    bound to the docket's roots and domains after the checks before the sample, `REG.bind_ports`; prove_subprotocol's
    stand-in's) when the statement reads registered values, with the window's registered circuit record
    (`st["interface_file"]`, `--interface`) whose declared registered ports the file never binds (`bound_files`); `zk`,
    a `--zk` session's. `log` names its logs under `out` (a negative's own, beside the honest verdict's). `flags`: more
    of `verify`'s (a plain session's ``--coins tree --stratified K``). ``members``: the run's kept verifier
    (`VerifyMembers`), which is handed `verify`'s arguments for this session as its next member in place of a `verify`
    process of its own, and gives the same verdict. ``st["archived"]``, a dict a run's members share (`archive_put`'s
    ``held``): a file archived once, the members' circuit or a file the run put before its sessions, is linked into
    every later archive rather than put again, a check running beside it (`a5.LeanPool`) on the same circuit file
    waiting for the first to put its files (`archiving`). With ``st["archive_link"]`` (`ARCHIVE_LINK`) true, a circuit
    file no earlier member archived is linked into the archive and not put (`archive_put`'s ``link``). ``st["statement"]``:
    the statement `verify` sets up
    (a templated one's ``verity/flock-circuit/tmpl``), else `STATEMENT_ID`. The verdict keeps what `verify` names of the
    public file's header, the registered reads, the sample and the tables it checked the session against
    (``public_header_sha512``, ``registered_sha512``, ``unit_draw``, ``session_tables``), which the second stage's audit
    holds each statement to. `verify` runs in `verifier_env`: under the verifier's allocator setting unless this process's
    environment sets its own. The kept verifier does not (`VerifyMembers`)."""
    sessions = sorted((out / "sessions").glob("*/session.json"))
    if not sessions:
        return {"accepted": False, "of_record": True, "why": "no session recorded"}
    archive, files = out / "archive", archive_files(st, n)
    with archiving(st.get("archived"), identity(files[0])):
        keys = archive_put(lean, archive, files, out / ("archive-put.log" if log == "lean-verify" else f"{log}.archive-put.log"),
                           st.get("archived"), files[:1] if st.get(ARCHIVE_LINK) else ())
    if (why := registered_held(st, keys)) is not None:
        return {"accepted": False, "of_record": True, "why": why}
    bound = [y for x in bound_files(st) for y in (f"--{x}", keys.get(Path(st[f"{x}_file"]), ""))]
    args = ["--statement", st.get("statement", STATEMENT_ID), "--archive", str(archive),
            "--circuit", keys.get(files[0], ""), "--public", keys.get(files[1], ""), *bound,
            *(["--zk"] if zk else []), *flags, "--session", str(sessions[-1].parent)]
    t = time.time()
    if members is None:
        p = run([lean, "verify", *args], out / f"{log}.log", env=verifier_env())
        exited = exit_why(p.returncode, VERIFY_EXITS)
        stdout, none = ("", exited) if exited else (p.stdout, f"no verdict (rc {p.returncode}): {p.stderr[-400:]}")
    else:
        stdout, none = members.verify(args, out / f"{log}.log")
    line = next((x for x in stdout.splitlines() if x.startswith("VERDICT ")), None)
    v = json.loads(line.split(" ", 1)[1]) if line else {"accepted": False, "why": none}
    return {"accepted": v.get("accepted") is True, "why": v.get("why"), "of_record": True, "verify_s": round(time.time() - t, 1),
            "partition": v.get("partition"), **({"registered": v["registered"]} if "registered" in v else {}),
            **{k: v[k] for k in ("public_header_sha512", "registered_sha512", "unit_draw", "session_tables") if k in v},
            "digest": hashlib.sha512(line.encode()).hexdigest() if line else None}


#: how Lean's reason opens where a file of the session, its record or a rep's proof, was not there to read (`verify` in
#: FlockVerify.lean, after the statement is set up and before any check of a proof)
NO_SESSION_FILES = "session files:"


def lean_refused(v: dict, names: str | None = None) -> bool:
    """Lean's own verdict (a VERDICT line, not a0's stand-in for none) refused the proof, and not where it read none: at
    setup, where a missing or unreadable input is refused, or for want of a file of the session (`NO_SESSION_FILES`);
    unless the refusal names `names`, a check Lean makes there."""
    why = v.get("why") or ""
    return v.get("accepted") is False and v.get("digest") is not None and bool(why) and (
        names in why if names else not why.startswith(("setup:", NO_SESSION_FILES)))


def lean_read_no_proof(v: dict) -> bool:
    """Lean's own verdict refused the session for want of a file of it (`NO_SESSION_FILES`), so it read no proof: what a
    prover that sent none, or a ZK server that held one back, leaves. It says no proof left, never that Lean refused one."""
    return lean_refused(v, NO_SESSION_FILES) and v["why"].startswith(NO_SESSION_FILES)


def lean_answered(verifiers: dict | None) -> bool:
    """Whether a verdict's or an audit record's `verifiers` holds an answer of the Lean's. In its place the verifier
    writes `{"refused": reason}` for a Lean that refused the input and for one that could not run (`verifier.lean`), and
    fails its check either way. History tables the Lean itself refuses are its answer
    (`verifier.check`'s `R.HISTORY`)."""
    return "refused" not in ((verifiers or {}).get("lean") or {"refused": None})


def negative(v: Verdict) -> dict:
    """A negative's row from the verifier's verdict `v` on its altered input: refused only where the check failed and
    the Lean answered (`lean_answered`). Under a Lean that could not run the verifier fails every check, the honest
    input's too, so that failure refuses no negative: the row then carries what the verifier wrote in the Lean's place
    (``lean``), and the audit fails."""
    answered = lean_answered(v.verifiers)
    return {"refused": not v.ok and answered, "codes": list(v.codes),
            **({} if answered else {"lean": (v.verifiers or {}).get("lean")})}


def proved_units(header: dict, live: dict) -> list[int]:
    """The sampled statement's units as global unit indices: the population header's index of each sampled position."""
    pop = header["units"]["indices"]
    return [pop[u] for u in (live.get("unit_draw") or {}).get("units", [])]


def audit_session(rec: dict, receipt: dict, st: dict, header: dict, sampling_design: dict, record, live, lean: str, lean118: str, out: Path,
                  delta: float) -> dict:
    """Every check on one session, then the audit record."""
    draw = (record or {}).get("unit_draw")
    verifier = (record or {}).get("verdict") or {}
    lean_u = lean_sample_test(lean, [{"record": draw, "statement": (live or {}).get("unit_draw"),
                                    "instances": (live or {}).get("instances", 0)}], out, "u1-u3") if draw else []
    statement = {"circuit_sha512": (live or {}).get("circuit_sha512"), "statement_digest": (live or {}).get("statement_digest"),
                 "public_sha512": (live or {}).get("public_sha512"), "sigma": (live or {}).get("sigma"),
                 "roots": (live or {}).get("roots"), "coin_commitment": ((record or {}).get("coin_seed") or {}).get("commitment"),
                 "roots_are_registered": (live or {}).get("roots") == {r["port"]: r["root"] for r in rec["roots"]}}
    verdicts = {
        "m0-verifier": {"accepted": verifier.get("accepted") is True, "why": verifier.get("why"),
                        "tool": "flock-circuit serve (M0)"},
        "lean-draw-u1-u3": {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u,
                            "tool": "verity-verify-proof draw-test (9d0d53df)"},
        "lean-verify": {**lean_verify(lean118, st, header["instances"], out), "tool": "verity-verify-proof verify --archive"},
    }
    if draw is None:
        return {"accepted": False, "why": "no draw recorded", "verdicts": verdicts, "statement": statement}
    sessions = [sha512_file(f) for f in sorted((out / "sessions").glob("*/session.json"))]
    return V.audit_record(rec, receipt, draw, sampling_design, proved_units(header, live or {}), (live or {}).get("instances", -1),
                          statement, verdicts, delta, sessions=sessions, session_roots=(live or {}).get("roots") or {})


def audit_negatives(result: dict, sampling_design: dict, delta: float, session_roots: dict,
                    served: list | None = None) -> dict[str, dict]:
    """The audit record's own refusals, each re-deciding the real record with one input altered: the verifier
    not run or absent, a session root that isn't the registered one, the verifier reporting a registered value's
    reads opened against another root than its registered one, and, when the sessions were handed the verifier's sample
    (`served`), a served share that isn't it. Each is decided by the verifier (`verifier.audit_record`) and refused only
    where the Lean answered (`lean_answered`);
    `agree` is whether the Lean and the Python decided it alike."""
    base = dict(docket=result["registration"], receipt=result["receipt"], draw=result["draw"], sampling_design=sampling_design,
                proved_units=result["proved_units"], instances=len(result["proved_units"]), statement=result["statement"],
                verdicts=result["verdicts"], delta=delta, served=served, session_roots=session_roots)

    def decide(**change) -> dict:
        r = V.audit_record(**{**base, **change})
        return {"refused": not r["accepted"] and r["profile"] is None and lean_answered(r["verifiers"]),
                "failed": [k for k, c in r["checks"].items() if not c["ok"]], "agree": r["verifiers"]["agree"]}

    vs = base["verdicts"]
    port = next(iter(session_roots), None)
    neg = {"verifier-of-record-did-not-run (C2)": decide(verdicts={k: {**v, "accepted": None} if v.get("of_record") else v
                                                                   for k, v in vs.items()}),
           "no-verifier-of-record (C2)": decide(verdicts={k: v for k, v in vs.items() if not v.get("of_record")}),
           "session-root-not-registered": decide(session_roots={**session_roots, port: "00" * 64})}
    opened = audit.session_registered(vs, base["docket"])
    if opened:
        value = next(iter(opened))
        neg["registered-read-opened-against-another-root"] = decide(verdicts={
            k: {**v, "registered": {p: {**e, "root": "00" * 64} if e["value"] == value else e for p, e in v["registered"].items()}}
            if v.get("of_record") and isinstance(v.get("registered"), dict) else v for k, v in vs.items()})
    if served is not None:
        drawn = set(served)
        other = [u for u in range(result["registration"]["population"]) if u not in drawn][:1]
        neg["served-share-not-the-verifier's-draw (C1)"] = decide(served=sorted(served)[1:] + other)
    return neg


def altered_public(lean: str, st: dict, n: int, session_dir: Path, out: Path, rec: dict, check, zk: bool = False) -> dict:
    """A public file whose body is altered (its last output word's high byte flipped) and whose leaf-layer digest is updated
    to match, so the docket passes R1-R6 and only the verifier, recomputing the roots from the file, can
    refuse it. `check(record, leaf_layer)` is the verifier's docket check."""
    stage = out / "stage"
    shutil.rmtree(out, ignore_errors=True)
    (stage / "tables").mkdir(parents=True)
    os.link(st["dir"] / "circuit.txt", stage / "circuit.txt")
    for f in (st["dir"] / "tables").glob("*"):
        if f.is_file():
            os.link(f, stage / "tables" / f.name)
    shutil.copytree(session_dir / "sessions", out / "sessions", copy_function=os.link)
    body = bytearray((st["dir"] / f"pub-{n}.bin").read_bytes())
    body[-1] ^= 1
    (stage / f"pub-{n}.bin").write_bytes(bytes(body))
    digest = hashlib.sha512(bytes(body)).hexdigest()
    registration = check(dict(rec, leaf_layer=digest), digest)
    lean_v = lean_verify(lean, dict(st, dir=stage), n, out, zk=zk)
    return {"refused": registration.ok and lean_refused(lean_v, names="do not hash to the header's"),
            "registration_passes": registration.ok,
            "lean": lean_v.get("why")}
