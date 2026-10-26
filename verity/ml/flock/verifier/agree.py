"""The agreement check (PROTOCOL.md §17.1): the Lean verifier and upstream's Rust verifier on the same sessions.

    python agree.py --lean BIN --upstream BIN [--statement ID] --circuit F [--comp F] [--partition F] [--program F] \
        --public F [--registered F] [--tables DIR] [--stratified K] [--zk] [--session-tables J] [--expect FILE]... [--lean-refuses WHY] \
        [--lean-refuses-drawn WHY] [--lean-refuses-drawless WHY] [--jobs J] --out DIR --sessions DIR...

`--upstream` is PR #83's binary built with the vectors patch (`netlist-vectors.patch` at 9294e161, `flock-netlist`;
`circuit-vectors.patch` at 19c7269a, `flock-circuit`): its `replay` subcommand is upstream's offline verdict. Each
`--expect` file is a list of `{"name", "expect": {"accepted", "why"}}` (`redigest.py`'s `mutants.json`); a session's name
is its directory's basename, and `why` is a prefix of the Lean verifier's reason. An expectation may add
`upstream_accepted` for an intended divergence (PROTOCOL.md §17.1's D list): upstream must then return that verdict instead
of the Lean verifier's. `--lean-refuses-drawn WHY` is D12 at a draw law no soundness claim draws by, which the Lean
verifier refuses at setup: it must reject every session whose record carries a unit draw, one expected accepted with a
reason starting `WHY` and one expected rejected with `WHY` or its expected reason (a statement that does not set up is
refused for that first), and upstream must return each session's expected verdict; a session with none (a fuzz case) only
needs the refusal. `--lean-refuses WHY` is D12 at a statement whose form the soundness proof does not cover, and
`--lean-refuses-drawless WHY` at a session without a draw, both of which the Lean verifier refuses last, once every check
has passed (the form's first): a session they cover (every session, and each without a draw; neither a drawn session of
`--lean-refuses-drawn`) refused with their `WHY` passed every check, and is compared as accepted, any other verdict as it
is, and the Lean verifier must accept none of them. So is every session of a set without `--program`, which the Lean
verifier refuses last (`NO_PROGRAM`). A session directory may carry its own public file (`pub.bin`, a forgery's false
statement), the verifier's own registered reads (`registered.json`, in place of `--registered`; null for none; upstream
reads none) and its expected verdict (`case.json`, from the vectors patch's `forgeries` subcommand). The Lean verifier loads the circuit, public, compression and table files from DIR/archive by
their digests (PROTOCOL.md §16.8); upstream reads the paths. Writes DIR/agreement.tsv, the two
verdict logs and DIR/result.json; exits 1 if the verifiers disagree on any session or either misses an expectation.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def reads(a, s):
    """The verifier's own registered reads for session `s`: its directory's `registered.json` when it has one (null: none),
    else the set's `--registered`."""
    own = Path(s) / "registered.json"
    if own.exists():
        return None if json.loads(own.read_text()) is None else str(own)
    return a.registered


def groups(a):
    """Sessions by the public file and the registered reads their verifier reads."""
    g = {}
    for s in a.sessions:
        own = Path(s) / "pub.bin"
        g.setdefault((str(own) if own.exists() else a.public, reads(a, s)), []).append(s)
    return g


def drawn(s):
    """The session's record carries a unit draw (PR #83 from eb90718f)."""
    try:
        return bool(json.loads((Path(s) / "session.json").read_text()).get("unit_draw"))
    except (OSError, ValueError):
        return False


def archived(a, path):
    """Hold `path` in the run's archive, `OUT/archive/sha512/<hex>`, and return its SHA-512: the Lean verifier loads the
    statement's objects by content (PROTOCOL.md §16.8)."""
    h = hashlib.sha512(Path(path).read_bytes()).hexdigest()
    d = a.out / "archive" / "sha512"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / h).exists():
        try:
            os.link(path, d / h)
        except OSError:
            shutil.copyfile(path, d / h)
    return h


#: the Lean verifier's refusal of a session verified without `--program` (`FlockVerify.NO_PROGRAM`), last like D12's:
#: every session of a set with no program (upstream reads none) is refused with it once every check has passed
NO_PROGRAM = "refused: a session without the verifier's program (--program)"


#: `--upstream live`: upstream's verdicts are its own live ones, recorded in each selftest session's `case.json`, for a
#: prover build with no `replay` subcommand
LIVE = "live"


#: sessions per verifier process at least, where there are enough: each process sets its statement up once (about 20 s)
MIN_CHUNK = 4


def chunks(xs, k):
    """`xs` dealt round-robin into at most `k` non-empty chunks of about `MIN_CHUNK` or more (costly and cheap sessions spread
    evenly)."""
    k = max(1, min(k, len(xs) // MIN_CHUNK or 1))
    return [xs[i::k] for i in range(k)]


def lean_cmd(a, public, registered=None):
    """The Lean verifier's command for a public file and the verifier's registered reads: every statement object from the
    run's archive, by SHA-512. An upstream binary's plain sessions are upstream's form (`--upstream-plain`: no zk patch,
    so no `zk_level0` tag after the pair, PROTOCOL.md §8.1); `--zk` sessions are the zk patch's."""
    if a.tables:
        for t in sorted(Path(a.tables).glob("*.u32")):
            archived(a, t)
    cmd = [a.lean, "verify", "--statement", a.statement, "--archive", str(a.out / "archive"), "--circuit", archived(a, a.circuit),
           "--public", archived(a, public)]
    if a.comp:
        cmd += ["--comp", archived(a, a.comp)]
    if a.partition:
        cmd += ["--partition", archived(a, a.partition)]
    if a.program:
        cmd += ["--program", archived(a, a.program)]
    if registered:
        cmd += ["--registered", archived(a, registered)]
    if a.coins:
        cmd += ["--coins", a.coins]
    if getattr(a, "stratified", None):
        cmd += ["--stratified", a.stratified]
    if a.upstream != LIVE and not getattr(a, "zk", False):
        cmd += ["--upstream-plain"]
    return cmd + modes(a) + ["--session"]


def modes(a):
    """`--zk` and `--session-tables J`, which both verifiers read (absent from check.py's preflight namespace)."""
    tables = getattr(a, "session_tables", None)
    return (["--zk"] if getattr(a, "zk", False) else []) + (["--session-tables", tables] if tables else [])


def upstream_cmd(a, public, extra=()):
    flag = "--netlist" if a.statement == "verity/flock-netlist/v1" else "--circuit"
    tables = ["--tables", a.tables] if a.tables else []
    return [a.upstream, "replay", *extra, *modes(a), "--instances", public, flag, a.circuit, *tables, "--sessions"]


def verdicts(a, slots=None):
    """Both verifiers on every session, `--jobs` processes at once: per public file, the sessions dealt into chunks, one Lean and
    one upstream process per chunk, and each drawn session in its own upstream process (`replay --drawn-only` builds only its
    drawn units' statement; upstream leaks a statement's matrices, so one process holds one statement). A session's verdict does
    not depend on its chunk: the logs are written in the sequential run's order (by public file, then session)."""
    tasks = []                                                        # (kind, order, command, sessions)
    order = {}
    for gi, ((public, registered), sessions) in enumerate(groups(a).items()):
        for si, s in enumerate(sessions):
            order[s] = (gi, si)
        base = lean_cmd(a, public, registered)
        # a keyed coin tree's key is fresh once per verifier run, and a tampered session keeps its original's key
        for c in [[s] for s in sessions] if a.zk else chunks(sessions, a.jobs):
            tasks.append(("lean", base, c))
        if a.upstream == LIVE:
            continue
        plain = [s for s in sessions if not drawn(s)]
        for c in chunks(plain, a.jobs) if plain else []:
            tasks.append(("upstream", upstream_cmd(a, public), c))
        for s in sessions:
            if drawn(s):
                tasks.append(("upstream", upstream_cmd(a, public, ["--drawn-only"]), [s]))

    def run(t):
        # `slots` (ci.py): one budget of verifier processes shared by every set running at once
        if slots is not None:
            slots.acquire()
        try:
            p = subprocess.run(t[1] + t[2], capture_output=True, text=True)
        finally:
            if slots is not None:
                slots.release()
        return p.stdout, p.stderr

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        done = list(ex.map(run, tasks))
    lean, up, logs = {}, {}, {"lean": ([], []), "upstream": ([], [])}
    for (kind, _cmd, _c), (out, err) in zip(tasks, done):
        lines, errs = logs[kind]
        errs.append(err)
        for line in out.splitlines():
            if kind == "lean" and line.startswith("VERDICT "):
                v = json.loads(line[8:])
                lean[v["session"]] = {"accepted": v["accepted"], "why": v["why"]}
                lines.append((order.get(v["session"], (1 << 30, 0)), line))
            elif kind == "upstream" and line.startswith("UPSTREAM\t"):
                v = json.loads(line.split("\t", 1)[1])
                up[v["session"]] = v["verdict"]
                lines.append((order.get(v["session"], (1 << 30, 0)), line))
            else:
                lines.append(((1 << 30, len(lines)), line))
    if a.upstream == LIVE:
        for s in a.sessions:
            live = json.loads((Path(s) / "case.json").read_text()).get("live")
            if live is not None:
                up[s] = {"accepted": live["accepted"], "why": live.get("why", "")}
    for kind in ("lean", "upstream"):
        lines, errs = logs[kind]
        (a.out / f"{kind}.txt").write_text("".join(l + "\n" for _k, l in sorted(lines, key=lambda x: x[0])))
        (a.out / f"{kind}.err").write_text("".join(errs))
    return lean, up


def main(argv=None, slots=None, quiet=False):
    ap = argparse.ArgumentParser()
    for k in ("--lean", "--upstream", "--circuit", "--public"):
        ap.add_argument(k, required=True)
    ap.add_argument("--comp", default=None, help="the BLAKE3 row leaf's compression circuit (statements before eb90718f)")
    ap.add_argument("--partition", default=None, help="the verity/partition/v1 object the circuit binds (PR #83 from e51e2b86)")
    ap.add_argument("--program", default=None, help="the program's descriptor bytes, to derive a template query's units")
    ap.add_argument("--registered", default=None, help="the verifier's own registered reads (verity/registered-reads/v0)")
    ap.add_argument("--statement", default="verity/flock-circuit")
    ap.add_argument("--coins", default=None)
    ap.add_argument("--stratified", default=None, metavar="K",
                    help="the verifier's own K, for sessions drawn by the stratified law (Lean only; upstream reads the draw)")
    ap.add_argument("--zk", action="store_true", help="flock-circuit --zk sessions (upstream: the live build's replay)")
    ap.add_argument("--session-tables", default=None, help="the tables of every session")
    ap.add_argument("--tables", default=None, help="the pinned MUFU tables, for circuits with lookup slots")
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--lean-refuses", default=None, metavar="WHY",
                    help="D12 at a statement's form: every session refused with WHY once every check has passed, compared as accepted")
    ap.add_argument("--lean-refuses-drawn", default=None, metavar="WHY",
                    help="D12 at a draw law: every session whose record carries a unit draw refused at setup with WHY; upstream "
                         "gives each its expected verdict")
    ap.add_argument("--lean-refuses-drawless", default=None, metavar="WHY",
                    help="D12 at a session without a draw: refused with WHY once every check has passed, compared as accepted")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="verifier processes at once (each builds its own statement)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("lean.txt", "lean.err", "upstream.txt", "upstream.err"):
        (a.out / f).write_text("")
    expect = {e["name"]: e["expect"] for f in a.expect for e in json.loads(Path(f).read_text())}
    for s in a.sessions:
        case = Path(s) / "case.json"
        if case.exists():
            expect.setdefault(Path(s).name, {"accepted": json.loads(case.read_text())["expect"] == "accept", "why": ""})
    lean, up = verdicts(a, slots)
    rows, bad, passed = ["session\texpect\tlean\tupstream\tok\tlean_why\tupstream_why"], [], 0
    for s in a.sessions:
        name = Path(s).name
        L, U, e = lean.get(s), up.get(s), expect.get(name)
        shown = L
        refuses = [a.lean_refuses_drawn] if a.lean_refuses_drawn is not None and drawn(s) else []
        last = [] if refuses else [w for w, on in ((a.lean_refuses, True), (a.lean_refuses_drawless, not drawn(s)),
                                                   (NO_PROGRAM, a.program is None)) if w is not None and on]
        if last and L is not None:
            if L["accepted"]:
                bad.append(name)
            elif L["why"].startswith(tuple(last)):
                # refused last, once every check has passed: for the comparison, the Lean verifier's checks accept it
                L, passed = {"accepted": True, "why": ""}, passed + 1
        if refuses and e is not None:
            # a refusal replaces every verdict of a statement that sets up; one that doesn't keeps its setup's reason
            e = {"accepted": False, "whys": refuses + ([] if e["accepted"] else [e["why"]]),
                 "upstream_accepted": e.get("upstream_accepted", e["accepted"])}
        up_want = e.get("upstream_accepted", L["accepted"] if L else None) if e else (L["accepted"] if L else None)
        ok = L is not None and U is not None and U["accepted"] == up_want
        if e is None and refuses:
            # no expected verdict (a fuzz case that may not change it): the Lean verifier verifies nothing to compare
            ok = L is not None and U is not None and not L["accepted"] and L["why"].startswith(tuple(refuses))
        if e is not None:
            ok = ok and L["accepted"] == e["accepted"] and any(L["why"].startswith(w) for w in e.get("whys") or [e["why"]])
        if not ok and name not in bad:
            bad.append(name)
        av = lambda v: "-" if v is None else "A" if v["accepted"] else "R"
        rows.append("\t".join([name, av(e), av(shown), av(U), "yes" if name not in bad else "NO", (shown or {}).get("why", ""),
                               (U or {}).get("why", "")]))
    (a.out / "agreement.tsv").write_text("\n".join(rows) + "\n")
    result = {"statement": a.statement, "sessions": len(a.sessions), "agree": len(a.sessions) - len(bad), "disagree": bad,
              "accepted": sum(1 for s in a.sessions if (lean.get(s) or {}).get("accepted")), "passed_every_check": passed}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    if not quiet:
        print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
