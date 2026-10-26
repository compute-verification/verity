"""One full-size window's opening checks, timed through each verifier: `samples` openings (33,273, the stated window's
default) of a session of one window over a store of 2^`blocks_log2` registered blocks, `passes` whole-store passes,
under the default codec and receipt kind, checked by the Python (`checks.check` in one process, and `Checks` over
`workers` processes as `verifier.py` batches them) and by the Lean verifier of record (`remote-audit-verifier`, one
fresh process at each `--chunks` size: 1024 is `verifier.py`'s, the window's whole draw one thread).

The window is synthetic, and each verifier checks it as it would a prover's: the payload is zero blocks under one
salt, its tree built whole; each pass's seed is the live coin the session releases for it, its receipts real at the drawn positions
and random elsewhere, committed in a tree whose subtrees off the drawn leaves are random digests (no verifier can tell
them from a pass's). Every event of the session goes to both verifiers and their answers are compared, the draws
among them, and the accepted session's record. A negative control flips a byte of the block at draw `samples // 2`,
which both must reject there. Writes OUT/window_bench.json after each phase; exits 1 unless both verifiers agree
throughout and accept the honest window.

    python window_bench.py --out OUT [--blocks-log2 25 --passes 918 --samples 33273 --workers N --chunks 1024,S]

    research run --on auto --cwd clone --cpus 16 --mem-gb 48 --max-min 90 -- bash -c 'uv sync -q --all-packages \
      --extra torch-cpu && PATH=$HOME/.elan/bin:$PATH uv run --with gmpy2 python \
      benchmarks/remote_session/window_bench.py --out "$OUT"'

The encode of the drawn blocks (k square roots a word) uses gmpy2 where it is importable (`uv run --with gmpy2`),
checked against `codec.decode`.
"""
from __future__ import annotations

import argparse
import bisect
import functools
import itertools
import json
import multiprocessing
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, codec, coins, receipts  # noqa: E402
from verity.core.protocols.remote_bandwidth import wire as rwire  # noqa: E402
from verity.ml.commitments import merkle_depth  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402

import lean_verifier as lv  # noqa: E402
import wire  # noqa: E402
from checks import Checks, check  # noqa: E402
from prover import OPENINGS_PART  # noqa: E402
from verifier import CHECK_BATCH  # noqa: E402

NODE = b"window-bench"
DEADLINE_NS = 10**9


@functools.cache
def _payload_domain(node: bytes, blocks: int):
    return audit.payload_domain(node, blocks)


def _subtree(node: bytes, blocks: int, value: bytes, first: int, size: int, wanted: list[int]):
    """The payload subtree over blocks `first … first + size − 1` (`first` a multiple of `size`), every leaf `value`:
    its root, and each wanted block's path inside it."""
    d = _payload_domain(node, blocks)
    level = [d.leaf(j, j, audit.PAYLOAD_SCHEMA, value) for j in range(first, first + size)]
    paths: dict[int, list[bytes]] = {j: [] for j in wanted}
    h = 0
    while len(level) > 1:
        for j in wanted:
            paths[j].append(level[((j - first) >> h) ^ 1])
        base = first >> (h + 1)
        level = [d.node(h, base + k, level[2 * k], level[2 * k + 1]) for k in range(len(level) // 2)]
        h += 1
    return level[0], paths


def _encode(payload: bytes, j: int, k: codec.Key) -> bytes:
    """`codec.encode(payload, j, k)`, its square roots by gmpy2 where it is importable."""
    try:
        import gmpy2
    except ImportError:
        return codec.encode(payload, j, k)
    f = k.codec.field
    p, s_ = f.p, f.word_bytes
    gp, ge = gmpy2.mpz(p), gmpy2.mpz((p + 1) // 4)
    i0, out = f.block_words * j, []
    for w in range(f.block_words):
        x, t = int.from_bytes(payload[s_ * w:s_ * (w + 1)], "little"), (k.tau * (i0 + w)) % p
        for r, c in enumerate(k.c):
            if r and k.mask is not None:
                x = k.sigma(x)
            a = (x + c + t) % p
            s = int(gmpy2.powmod(a, ge, gp))
            square = s * s % p == a
            x = s if (s & 1) == (0 if square else 1) else (p - s if s else 0)
        out.append(x.to_bytes(s_, "little"))
    return b"".join(out)


def _encode_blocks(key: codec.Key, payload: bytes, js: list[int]) -> dict[int, bytes]:
    out = {}
    for j in js:
        out[j] = block = _encode(payload, j, key)
        if codec.decode(block, j, key) != payload:
            raise AssertionError(f"block {j}'s encode does not decode to its payload")
    return out


def _sparse(d, depth: int, leaves: dict[int, bytes], rnd) -> tuple[bytes, dict[int, list[bytes]]]:
    """A tree of 2^depth leaves in which `leaves` are real and every node off their paths is random: its root and
    each real leaf's path."""
    level, under = dict(leaves), {c: [c] for c in leaves}
    paths: dict[int, list[bytes]] = {c: [] for c in leaves}
    for h in range(depth):
        up, up_under = {}, {}
        for q in sorted({k >> 1 for k in level}):
            left, right = level.get(2 * q) or rnd(64), level.get(2 * q + 1) or rnd(64)
            for c in under.get(2 * q, ()):
                paths[c].append(right)
            for c in under.get(2 * q + 1, ()):
                paths[c].append(left)
            up[q] = d.node(h, q, left, right)
            up_under[q] = under.get(2 * q, []) + under.get(2 * q + 1, [])
        level, under = up, up_under
    return level[next(iter(level))], paths


def _draws(coin: bytes, digest: bytes, t: int, samples: int, lengths: list[int]) -> list[tuple[int, int]]:
    """Window t's draws as `audit.Session.close` takes them from its openings coin, known before its passes: they
    depend on the coin, the registration and the passes' lengths, not on their roots."""
    key = derive(coin, audit.OPENINGS_DOMAIN, {"registration": digest, "window": t})
    ends = list(itertools.accumulate(lengths))
    out = []
    for u in key.subset(min(samples, ends[-1]), ends[-1], "openings"):
        i = bisect.bisect_right(ends, u)
        out.append((i, u - (ends[i - 1] if i else 0)))
    return out


def _lean_run(exe: Path, chunk: int, source: bytes, empty: dict, register: dict, session: dict,
              events: list[tuple[dict, dict]], bodies: list[bytes], record: dict | None) -> dict:
    """One fresh Lean process through the session: each answer held to the Python's, then the window's frames, its
    check (timed from `verify` to its verdict's line) and its settled result."""
    done, verdicts = threading.Event(), []

    def on_done(msg: dict) -> None:
        verdicts.append(msg)
        done.set()

    lean = lv.Lean(exe, on_done)
    out: dict = {"chunk": chunk, "disagreements": []}

    def agree(what: str, got, want) -> None:
        if got != want:
            out["disagreements"].append({"at": what, "lean": got, "python": want})

    try:
        got = lean.call({"op": "init", "source": source.hex(), "registry": empty, "chunk": chunk})
        agree("init", got, {"registry": empty})
        agree("register", lean.call(register["request"]), register["answer"])
        agree("session", lean.call(session), {"session": 0})
        t0 = time.perf_counter_ns()
        for ev, want in events:
            got = lean.call({"op": "event", "node": NODE.hex(), "session": 0, "event": ev, "coin": None})
            agree(ev["kind"], got, want)
        out["events_ns"] = time.perf_counter_ns() - t0
        t0 = time.perf_counter_ns()
        for body in bodies:
            lean.call({"op": "openings", "node": NODE.hex(), "session": 0, "t": 0, "bytes": len(body)}, body)
        out["openings_ns"] = time.perf_counter_ns() - t0
        t0 = time.perf_counter_ns()
        agree("verify", lean.call({"op": "verify", "node": NODE.hex(), "session": 0, "t": 0, "id": 0}),
              {"spawned": True})
        done.wait()
        out["check_ns"] = time.perf_counter_ns() - t0
        settled = lean.call({"op": "settle", "id": 0})
        out["settle_ns"] = settled.pop("ns", None)
        out["openings"] = settled.pop("openings", None)
        out["result"], out["done"] = settled, verdicts
        out["outcome"] = lean.call({"op": "outcome", "node": NODE.hex(), "session": 0})
        if record is not None:
            agree("record", out["outcome"], {"outcome": "accepted", "record": record})
    except lv.LeanError as e:
        out["error"] = str(e)
    finally:
        lean.close()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks-log2", type=int, default=25)
    ap.add_argument("--passes", type=int, default=918)
    ap.add_argument("--samples", type=int, default=audit.SAMPLES)
    ap.add_argument("--codec", default=codec.DEFAULT.name)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--chunks", default=f"{CHECK_BATCH},S", help="the Lean's check chunk per process (S: the window's draws)")
    ap.add_argument("--subtree-log2", type=int, default=17, help="the payload tree's subtree a worker task builds")
    ap.add_argument("--lean-exe", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    N, P, S, W = 1 << a.blocks_log2, a.passes, a.samples, 1
    T = audit.WINDOW_NS
    cdc = wire.codec_named(a.codec)
    rng = random.Random(a.seed)
    source, nonce, payload_salt = rng.randbytes(32), rng.randbytes(audit.NONCE_BYTES), rng.randbytes(audit.SALT_BYTES)
    payload = bytes(codec.BLOCK_BYTES)
    model = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                  if ln.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else None
    out: dict = {"blocks": N, "passes": P, "samples": S, "codec": cdc.name, "receipt": receipts.DEFAULT,
                 "workers": a.workers, "cpus": len(os.sched_getaffinity(0)), "cpu_model": model,
                 "gmpy2": _has_gmpy2(), "setup_ns": {}}

    def save(phase: str) -> None:
        (a.out / "window_bench.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({phase: out[phase][-1] if phase == "lean" else out[phase]}), flush=True)

    pool = ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context("spawn"))
    value = receipts.hidden(payload, payload_salt)
    size = min(N, 1 << a.subtree_log2)
    t0 = time.perf_counter_ns()
    roots = [r for r, _ in pool.map(_subtree, *zip(*[(NODE, N, value, f, size, []) for f in range(0, N, size)]))]
    tops = [roots]
    while len(tops[-1]) > 1:
        h, lvl = size.bit_length() - 1 + len(tops) - 1, tops[-1]
        d = _payload_domain(NODE, N)
        tops.append([d.node(h, k, lvl[2 * k], lvl[2 * k + 1]) for k in range(len(lvl) // 2)])
    payload_root = tops[-1][0]
    out["setup_ns"]["payload_tree"] = time.perf_counter_ns() - t0

    registry = audit.Registry()
    empty = registry.to_json()
    reg = registry.register(source, NODE, N, payload_root, cdc, receipts.DEFAULT, nonce=nonce)
    register = {"request": lv.register_request(NODE, N, payload_root, cdc, receipts.DEFAULT, None, nonce),
                "answer": {"codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()}}
    params = audit.Params(N, DEADLINE_NS, W, S, window_ns=T, codec=cdc, receipt=reg.receipt, rows=reg.rows)
    clock = [0]
    s = registry.session(params, reg, None, not_before_ns=0, clock=lambda: clock[0], entropy=rng.randbytes, index=0)
    committed = [c[:coins.COIN_BYTES] for c, _ in s.coins.challenger._coins[:2 * W]]
    session = {"op": "session", "node": NODE.hex(), "params": lv.params_json(params),
               "coins": [c.hex() for c in committed], "append_coins": [], "not_before_ns": 0, "index": 0}
    draws = _draws(committed[W], reg.digest, 0, S, [N] * P)
    blocks_drawn = sorted({p for _, p in draws})

    t0 = time.perf_counter_ns()
    key = reg.key
    per = max(1, len(blocks_drawn) // (4 * a.workers))
    encodes = [pool.submit(_encode_blocks, key, payload, blocks_drawn[k:k + per])
               for k in range(0, len(blocks_drawn), per)]
    by_subtree = defaultdict(list)
    for j in blocks_drawn:
        by_subtree[j - j % size].append(j)
    paths_f = [pool.submit(_subtree, NODE, N, value, f, size, js) for f, js in sorted(by_subtree.items())]
    blocks: dict[int, bytes] = {}
    for f in encodes:
        blocks.update(f.result())
    out["setup_ns"]["encode"] = time.perf_counter_ns() - t0

    events: list[tuple[dict, dict]] = []

    def step(ev: dict, got: dict) -> None:
        events.append((ev | {"now": clock[0]}, got))

    t0 = time.perf_counter_ns()
    step({"kind": "start"}, lv.released_json(s.start()))
    rdomain_depth = merkle_depth(receipts.leaves(N))
    by_pass = defaultdict(list)
    for i, p in draws:
        by_pass[i].append(p)
    leaves: dict[tuple[int, int], tuple[bytes, bytes, tuple[bytes, ...]]] = {}
    plan = receipts.Plan.whole(N)
    for i in range(P):
        opening = s.seed()
        events.append(({"kind": "seed"}, lv.seeded_json(opening)))
        seed = coins.seed(opening)
        cells: dict[int, list[bytes]] = {}
        if i in by_pass:
            receipter = receipts.receipter(reg.receipt, seed, reg.rows)
            for p in by_pass[i]:
                c = p // receipts.PER_LEAF
                cell = cells.setdefault(c, [rng.randbytes(receipts.DIGEST_BYTES) for _ in range(receipts.PER_LEAF)])
                cell[p % receipts.PER_LEAF] = receipter(p, blocks[p])
        if cells:
            rd = receipts.domain(seed, plan)
            salts = {c: rng.randbytes(audit.SALT_BYTES) for c in cells}
            real = {c: rd.leaf(c, c, receipts.SCHEMA, receipts.hidden(b"".join(cell), salts[c]))
                    for c, cell in cells.items()}
            root, rpaths = _sparse(rd, rdomain_depth, real, rng.randbytes)
            for c in cells:
                leaves[i, c] = (b"".join(cells[c]), salts[c], tuple(rpaths[c]))
        else:
            root = rng.randbytes(receipts.DIGEST_BYTES)
        clock[0] = i + 1
        step(lv.receive_event(0, i, root, None, i + 1), lv.done_json(s.receive(0, i, root, step=i + 1)))
    session["pass_coins"] = [coins.seed(o).hex() for o in s.coins.opened(coins.PASS)]
    clock[0] = T
    step({"kind": "boundary"}, lv.released_json(s.boundary()))
    clock[0] = T + DEADLINE_NS + 1
    drew = s.close(0)
    step({"kind": "close", "t": 0}, lv.drew_json(drew))
    out["setup_ns"]["passes"] = time.perf_counter_ns() - t0
    if list(drew) != draws:
        raise AssertionError("audit.Session.close drew other than the bench's draws")

    t0 = time.perf_counter_ns()
    ppaths: dict[int, tuple[bytes, ...]] = {}
    for f in paths_f:
        _, got = f.result()
        for j, inner in got.items():
            top = [tops[h][((j // size) >> h) ^ 1] for h in range(len(tops) - 1)]
            ppaths[j] = tuple(inner + top)
    pool.shutdown()
    openings = []
    for i, p in draws:
        leaf, lsalt, rpath = leaves[i, p // receipts.PER_LEAF]
        openings.append(audit.Opening(i, p, blocks[p], leaf, lsalt, rpath, payload_salt, ppaths[p], b""))
    frames = [openings[k:k + OPENINGS_PART] for k in range(0, len(openings), OPENINGS_PART)]
    bodies = [rwire.encode_openings(fr) for fr in frames]
    bad = S // 2
    o = openings[bad]
    flipped = audit.Opening(o.pass_index, o.position, bytes([o.block[0] ^ 1]) + o.block[1:], o.leaf, o.leaf_salt,
                            o.receipt_path, o.payload_salt, o.payload_path, b"")
    bad_frame = bad // OPENINGS_PART
    bad_bodies = list(bodies)
    bad_bodies[bad_frame] = rwire.encode_openings([flipped if x is o else x for x in frames[bad_frame]])
    out["setup_ns"]["openings"] = time.perf_counter_ns() - t0
    out["frames"], out["frame_bytes"] = len(bodies), sum(map(len, bodies))
    save("setup_ns")

    snap = Checks.snapshot(s, 0)
    t0, c0 = time.perf_counter_ns(), time.process_time_ns()
    single = check(snap, 0, draws, bodies)
    out["python_single"] = {"ns": time.perf_counter_ns() - t0, "cpu_ns": time.process_time_ns() - c0,
                            "failure": single["failure"], "n": single["n"]}
    save("python_single")
    t0 = time.perf_counter_ns()
    negative = check(snap, 0, draws, bad_bodies)
    want_bad = {"failure": negative["failure"], "window": 0, "at": None if negative["at"] is None else list(negative["at"])}
    out["python_negative"] = {"ns": time.perf_counter_ns() - t0, **want_bad}
    save("python_negative")

    checks = Checks(a.workers)
    batches, cur, n, first = [], [], 0, 0
    for fr, body in zip(frames, bodies):
        cur.append(body)
        n += len(fr)
        if n >= CHECK_BATCH:
            batches.append((first, n, cur))
            first, cur, n = first + n, [], 0
    if cur:
        batches.append((first, n, cur))
    t0 = time.perf_counter_ns()
    futures = [checks.submit(snap, 0, draws[f:f + k], bs) for f, k, bs in batches]
    results = [f.result() for f in futures]
    verdict = Checks.settle(s, 0, results, len(openings))
    out["python_pool"] = {"ns": time.perf_counter_ns() - t0, "batches": len(batches),
                          "cpu_ns": sum(r["cpu_ns"] for r in results), "verdict": lv.verdict_json(verdict)}
    checks.close()
    record = lv.summary_json(s.record()) if s.outcome == "accepted" else None
    save("python_pool")

    exe = a.lean_exe or lv.build()
    out["lean"] = []
    chunks = [S if c == "S" else int(c) for c in a.chunks.split(",")]
    for chunk, bs, rec, name in [(chunks[0], bad_bodies, None, "negative"), *[(c, bodies, record, "honest") for c in chunks]]:
        row = _lean_run(exe, chunk, source, empty, register, session, events, bs, rec)
        row["case"] = name
        out["lean"].append(row)
        save("lean")

    honest = [r for r in out["lean"] if r["case"] == "honest"]
    neg = next(r for r in out["lean"] if r["case"] == "negative")
    out["agree"] = (all(not r["disagreements"] and "error" not in r and r["result"] == {"accepted": 0} for r in honest)
                    and not neg["disagreements"] and "error" not in neg and neg["result"] == want_bad
                    and single["failure"] is None and verdict.accepted
                    and want_bad["failure"] is not None and want_bad["at"] == list(draws[bad]))
    save("agree")
    return 0 if out["agree"] else 1


def _has_gmpy2() -> bool:
    try:
        import gmpy2  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
