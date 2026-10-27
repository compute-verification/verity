"""One-stage audit, run A7: the requirement's two proof blocks (I4), sampled in one audit and wired to the remote bandwidth
audit's ledger as its requirement link. `PlanMatch_v1` (each pass's plan is exactly the blocks its step reads, which gives
`PlanMatches`) and
`MemoryMapStep_v3` (the blocks a step writes carry its values, written at their step and freed at their last structural
reader, each registered block freed at its values' last structural reader, and each re-key's old block written below
the cap the committed pass row and the public lifetime cap give), one instance of each per step
(`verity.ml.circuits.requirement_units`), the layout row at two epochs a value.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm:<tree>/tools/verity/circuit_check/src \\
      python benchmarks/one_stage/a7.py --bin flock-circuit --lean flock-verify --out DIR [--k K] [--steps 32] [--delta 2^-20]

* **Stand-in.** A decoder's requirement built directly (`requirement.Requirement`): 8 weight blocks, every step reading
  them but the last, an expert's, which only the first half of the steps reads, a
  48-row embedding table (2 rows a block, blocks 8 .. 31) each active request gathers one row of (its token), 10 requests
  (one through every step, nine with seeded arrivals and lengths), at most 4 active, some paused (their pages stay
  resident and no plan holds them), each
  active step producing a 2,048-byte KV value its request reads at its later steps. The values are seeded bytes, every
  64-byte piece's top byte zero (p512's words); the layout packs each step's values in fresh blocks of 2 slots in a seeded
  order (private), the store frees each block, registered or written, at its values' last structural reader. vllm's Store backend
  replaces these.
* **Commitments.** The values and the shared rows are committed once, outside the units, as frame-v3-sha512 trees of
  hm96-sha512/row/v2 rows under keyed salts (the canonical commitments, registered beside the statements' ports): the
  layout row, the map row, the pass row, each step's plan and gathered tokens, each written value (the one-stage value
  commitments' stand-in) and each written block's payload. Each unit's witness rows are those rows under those salts, so their
  ``b || c`` in the statement's public file are the canonical ones; the verifier holds each at the position the public
  requirement names (`bindings`).
* **Statements and draw.** One circuit per template, staged by C-Flock's `UnitStager` bound to the partition
  (`Q_template_instances` over `TemplatePopulations`, one unit per step each), drawn by the verifier's Lean
  (`flock-verify draw --stratified K`, K every unit by default, so no unit is uncertified), each member's draw served
  `--zk` and held by Lean `verify --zk --partition --program`. For every drawn unit the prover opens its public row (the
  verifier's own copy of the step's public data) and its ok row (1): the units the proofs cover are the steps whose
  clauses hold.
* **The link.** The remote session (v0: appends, releases and plans on the wire) runs a pass a step, its plan the
  certified one, the memory map's appends and releases after each step's reads (memory-map/v1, `Replay`), each append's
  coin `c_a` encoding its blocks (`Store.encode`) and each pass's seed binding the appends before it. The appends are
  the map unit's own: a payload row's ``b || c`` is its append leaf's commit string (both hm96-sha512 over the block's
  sha512/row/v2 digest under one salt), so the prover appends each step's blocks under its payload rows' salts and the
  verifier recomputes the session's chain of appends (`audit.append_head`) from the map member's public file and each
  step's public first and count (`units_heads`): `UnitsLink.mapped` holds of the record when its chain is that one,
  each pass's head is the chain through the appends of the steps below its own, each map unit's public lifetime cap is
  the record's, and every map unit is drawn and proved. The heads fix the window each append joined. The passes' steps
  are a value: each map unit's pass row (each window's last pass step, `requirement_units.pass_words`) is bound to one
  canonical commitment, from which the unit computes each re-key's cap in gates (the stale bound in step form, which no
  public input carries). The bridge (`bridge`) binds the rest on public data: each pass's plan length is its step's
  plan unit's public count, each pass's seed binds the units' chain of the steps before it (`units_heads`), and the
  registered blocks released after each step are its map unit's public frees. The one canonical row it opens is the
  pass row, whose words are the passes' steps v0's wire carries, until the record commits to them. The ledger then
  credits each pass the bits its step reads from complete blocks, less a block for each planned block it credits no
  receipt of (`audit.linked_bits` on `RequirementLink`, `UnitsLink`), the session 1 − ε of their sum, ε the share of
  it the wrong units the draw allows can carry (`uncertified`), its δ added to the ledger's error.
* **Check 3.** The session draws ``--samples`` receipts a window, and nothing is opened: each draw goes through the
  remote audit's opening unit (`remote_bandwidth.opening`, `verity.ml.commitments.gates.block_check`, §8's checks 1-3
  and 5 as one unit): the receipt leaf on its path to the pass's receipt root and the step's plan row holding the block
  at the drawn position, the block decoding to its payload under b_j(c_a), the payload on its path to its append's root (the
  registration's payload root for a registered block) and to the canonical payload commitment at the row the memory-map
  unit binds the block's payload row to, and the block's sketch receipt the leaf's entry. The units are proved as a
  second statement (`check3_audit`: every unit drawn, --zk, Lean). The verifier sees the units' roots, its own public
  rows (the codec seed, the block, the coin, the pass's prefix and signs, the roots with their domains, ranks and depths)
  and each unit's verdict, and holds the one other row, the plan row, to the canonical plan row by ``b || c``. The
  negatives' sessions decide check 3 by the unit's reference (`opening.reference`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import r2_population as R2  # noqa: E402
from verity.core.protocols.remote_bandwidth import audit, codec, coins, memory_map as MM, opening as C3, receipts  # noqa: E402
from verity.core.protocols.remote_bandwidth import schedule as SC  # noqa: E402
from verity.core.protocols.remote_bandwidth.receipts import Plan  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V  # noqa: E402
from verity.ml.circuits import requirement as RQ, requirement_units as RU  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.commitments.gates import block_check as BC  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_flock import circuit as FC, class_statement as CS  # noqa: E402

M_MAX = 35
WEIGHTS, TABLE_ROWS, RPB, TB, SLOTS = 8, 48, 2, 8, 2
EXPERT = WEIGHTS - 1
NR = TB + TABLE_ROWS // RPB
VW = RU.value_bits(SLOTS)
REQUESTS, ACTIVE, PAUSE = 10, 4, 0.15
WINDOWS = 4
#: the remote session's lifetime cap in windows (the memory-map unit's `life`, a U16): more than it has, so no block
#: re-keys
LIFE = 60
#: the cap on its schedule's bits the remote session commits before its coins (`audit.Params.schedule_cap`): its
#: schedule check refuses a schedule over it, and a ledger crediting appended blocks credits it as the steering
SCHEDULE_CAP = 1 << 16
#: each written value's epochs in the layout row: its production block's and one re-key's
EPOCHS = 2
#: the remote session's draws a window, each proved by check 3's unit (`opening`); at 2 the seeded run draws registered
#: blocks as well as appended ones
SAMPLES = 2
NODE, SOURCE = b"a7-node", b"a7 verifier's secret"
P512 = codec.Codec(codec.P512, 2, False)
WINDOW_OF = ("a stand-in decoder's requirement (8 weight blocks, a 48-row gathered table, 10 requests of 2,048-byte KV "
             "values, seeded), its layout and store; vllm's Store backend replaces these")


def _bytes(label: str, n: int) -> bytes:
    """n seeded bytes, each 64-byte piece's top byte zero (a p512 word)."""
    raw = bytearray(hashlib.shake_256(f"verity/one-stage/a7/v1\0{label}".encode()).digest(n))
    raw[63::64] = bytes(len(raw[63::64]))
    return bytes(raw)


def _int(b: bytes) -> int:
    return int.from_bytes(b, "little")


# --- the stand-in -----------------------------------------------------------------------------------------------------------

def standin(steps: int, seed: int):
    """The requirement, its written values in order, each step's gathered table rows, and the paused (request, step)s."""
    rng = random.Random(seed)
    width, born, last, uses, values = {}, {}, {}, {s: set() for s in range(steps)}, {}
    for v in range(WEIGHTS):
        width[v], born[v], values[v] = RQ.BLOCK_BITS, -1, _int(_bytes(f"weights {v}", RQ.BLOCK_BITS // 8))
        last[v] = steps // 2 - 1 if v == EXPERT else steps - 1
    table = list(range(WEIGHTS, WEIGHTS + TABLE_ROWS))
    for r, v in enumerate(table):
        width[v], born[v], last[v], values[v] = VW, -1, steps - 1, _int(_bytes(f"table row {r}", VW // 8))
    reqs = [(0, steps)] + sorted(((rng.randrange(0, steps * 2 // 3), rng.randrange(4, 12)) for _ in range(REQUESTS - 1)))
    left = {r: n for r, (_a, n) in enumerate(reqs)}
    active = {s: [] for s in range(steps)}
    paused = []
    for s in range(steps):
        for r, (a, _n) in enumerate(reqs):
            if a > s or left[r] == 0 or len(active[s]) == ACTIVE:
                continue
            if r and left[r] < reqs[r][1] and rng.random() < PAUSE:
                paused.append((r, s))
                continue
            active[s].append(r)
            left[r] -= 1
    steps_of = {r: [s for s in range(steps) if r in active[s]] for r in range(REQUESTS)}
    kv, gathers, nxt = {}, {s: [] for s in range(steps)}, 1000
    for r in range(REQUESTS):
        mine = []
        for i, s in enumerate(steps_of[r]):
            tok = rng.randrange(TABLE_ROWS)
            gathers[s].append(tok)
            uses[s] |= {w for w in range(WEIGHTS) if s <= last[w]} | {table[tok]} | set(mine)
            for v in mine:
                last[v] = s
            if i + 1 < len(steps_of[r]):
                kv[nxt] = (r, s)
                width[nxt], born[nxt], values[nxt] = VW, s, _int(_bytes(f"kv {r} {s}", VW // 8))
                mine.append(nxt)
                nxt += 1
    req = RQ.Requirement(width, frozenset(range(WEIGHTS + TABLE_ROWS)), born, last,
                         {s: frozenset(u) for s, u in uses.items()}, steps, values)
    positions = sorted(kv, key=lambda v: (born[v], v))
    return SimpleNamespace(req=req, positions=positions, gathers={s: sorted(g) for s, g in gathers.items()}, kv=kv,
                           table=table, paused=paused, requests=reqs, active=active)


def lay_out(req: RQ.Requirement, positions, seed: int):
    """The registered values at their public blocks; each step's values in fresh blocks of SLOTS slots, in a seeded order;
    each block, registered or written, freed at its values' last structural reader. Returns the layout and the store."""
    rng = random.Random(seed ^ 0x5A7)
    block, offset = {}, {}
    for v in range(WEIGHTS):
        block[v], offset[v] = v, 0
    for r in range(TABLE_ROWS):
        block[WEIGHTS + r], offset[WEIGHTS + r] = TB + r // RPB, (r % RPB) * VW
    written, freed, payload, nxt = {}, {}, {}, NR
    for s in range(req.steps):
        group = [v for v in positions if req.born[v] == s]
        rng.shuffle(group)
        for u, v in enumerate(group):
            block[v], offset[v] = nxt + u // SLOTS, (u % SLOTS) * VW
        js = range(nxt, nxt + -(-len(group) // SLOTS))
        if js:
            written[s] = frozenset(js)
        nxt = js.stop
    for v, j in block.items():
        payload[j] = payload.get(j, 0) | req.values[v] << offset[v]
    for j in set(block.values()):
        freed.setdefault(max(req.last[v] for v, k in block.items() if k == j), set()).add(j)
    return RQ.Layout(block, offset), RQ.Store(NR, written, {s: frozenset(b) for s, b in freed.items()}, payload)


# --- the canonical commitments ----------------------------------------------------------------------------------------------

def _salts(key: bytes, kind: str, n: int) -> list[bytes]:
    return [hashlib.shake_256(b"verity/one-stage/a7/canon-salt\0" + key + kind.encode() + i.to_bytes(8, "big")).digest(FC.SALT_BYTES)
            for i in range(n)]


def canon_dom(kind: str, n: int) -> CommitmentDomain:
    """The verifier's own derivation of a canonical commitment's frame domain: C-Flock's set-bound domain of port ``kind``
    over rows [0, n) of set ``a7-canon-<kind>``."""
    b = identity_digest_sha512(FC.BINDING_TAG, {"set": f"a7-canon-{kind}", "lo": 0, "hi": n, "port": kind, "schema": V2})
    return CommitmentDomain(b, FC.OWNER, RangeIndexedDomain(0, n), hash="sha512")


def canon_domain(kind: str, bits: int, n: int) -> str:
    """`canon_dom`'s domain id, hex."""
    return canon_dom(kind, n).domain_id.hex()


def canon(kind: str, bits: int, words: list[list[int]], key: bytes) -> dict:
    """One canonical commitment: rows ``words`` (each ``bits`` long) under keyed salts, its frame-v3 root and every row's
    ``b || c``."""
    port = FC._port(kind, -(-bits // 16), bits, False)
    rows = {kind: np.asarray(words, dtype=np.uint16).reshape(len(words), port.words)}
    salts = {kind: _salts(key, kind, len(words))}
    fv, bcs = FC.statement([port], SimpleNamespace(name=f"a7-canon-{kind}"), 0, len(words), rows, salts)
    return {"kind": kind, "bits": bits, "n": len(words), "root": fv["roots"][kind], "domain": fv["domain_ids"][kind],
            "bc": bcs[kind], "salts": salts[kind], "words": words, "port": port}


def canon_root(c: dict, bcs: list[bytes]) -> str:
    """The root the verifier recomputes from a canonical commitment's published ``b || c`` list."""
    from verity.ml.commitments import hm96
    from verity.ml.commitments.frame_v3 import FrameV3, Port as FPort
    from verity.ml.commitments.rowleaf import row_leaf

    dom = canon_dom(c["kind"], len(bcs))
    fv = FrameV3({c["kind"]: FPort(dom, c["port"].leaf())})
    inst = hm96.SHA512
    return fv.root(c["kind"], [row_leaf(dom, i, inst.tree_leaf(inst.default_key, bc), V2) for i, bc in enumerate(bcs)]).hex()


def opens(port, words: list[int], salt: bytes, bc: bytes) -> bool:
    """Whether ``b || c`` opens to ``words`` under ``salt``: the verifier recomputes the commit string."""
    got = FC._commit_strings([port], {port.name: np.asarray([words], dtype=np.uint16)}, {port.name: [salt]})[port.name][0]
    return got == bc


def public_bcs(path: Path, ports: list) -> list[dict[str, bytes]]:
    """Each instance's ``b || c`` by port, from a public file (its header line, then 128 bytes a row, instance-major)."""
    raw = path.read_bytes()
    head, body = raw.split(b"\n", 1)
    n = json.loads(head)["instances"]
    w = FC.BC_BYTES
    body = body[:n * len(ports) * w]
    return [{p.name: body[(i * len(ports) + q) * w:(i * len(ports) + q + 1) * w] for q, p in enumerate(ports)} for i in range(n)]


# --- the remote session (v0) ------------------------------------------------------------------------------------------------

class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


def _runs(blocks) -> tuple[tuple[int, int], ...]:
    return MM._runs(blocks)


def check3_inputs(canons: dict, sp, sm, samples: int = SAMPLES) -> SimpleNamespace:
    """What check 3 reads of the canonical commitments: each step's certified plan row (its words, salt and ``b || c``) and
    the canonical payload commitment (`opening.Values`, written block j at row j - NR, the row `bound_row` binds its
    ``pay`` row to); the unit's codec, P and depths (a receipt tree of a plan of P + 1 blocks, the registration's payload
    tree or an append of up to 2 C blocks, the payload commitment); and the draws a window it proves (``samples``), which
    every session of the run draws."""
    c, pay = canons["plan"], canons["payload"]
    values = C3.Values(bytes.fromhex(pay["root"]), canon_dom("payload", pay["n"]), NR, tuple(pay["bc"]))
    return SimpleNamespace(plan_rows=c["words"], plan_salts=c["salts"], plan_bcs=c["bc"], values=values, codec=P512, P=sp.P,
                           DR=C3.depth(receipts.leaves(sp.P + 1)), DP=C3.depth(max(NR, 2 * sm.C)), DV=C3.depth(pay["n"]),
                           samples=samples)


def remote_session(req, layout, store, plans: dict[int, list[int]], salts: dict[int, bytes],
                   extra: dict[int, list[int]] | None = None, late: dict[int, int] | None = None, alter=None,
                   early: int | None = None, dummy: int | None = None, rekey: tuple[int, int] | None = None,
                   drop: int | None = None, *, c3: SimpleNamespace, defer: bool = False):
    """The v0 session over the registered store: a pass a step, its plan sent as a change from the last (``plans[s]``, and
    ``extra[s]``'s blocks too: a prover that receipts more than the certified plan), then step s's memory-map record
    (`memory_map.Replay`): its releases (``late``: a block released at another step) and its append, written block j
    under ``salts[j]`` (its payload row's salt), encoded under the coin the session replies with. A prover that departs
    from the map: ``alter`` ({step: f}, f(payloads, salts) the append it sends in place of the step's), ``early`` (step
    t's append sent before t's pass, after step t − 1's), ``dummy`` (a step whose append adds a block holding no value),
    ``rekey`` ((step, j): block j appended again at that step, in its own slot, of the same payload at a fresh index,
    the later plans and its free naming that index), ``drop`` (the last step that writes: its append never sent, and its
    blocks in no later plan and no release). Each window draws ``c3.samples`` receipts, and check 3 decides them
    (``c3``, `check3_inputs`) in place of their openings: by its unit's reference on the prover's witness
    (`opening.reference`), or, ``defer``, not yet: the caller proves them (`check3_audit`) and decides each window.
    Returns the session, the prover's store and passes, its record and the verifier's v0 view: each pass's plan and step,
    its (window, step) in order, each appended index's write step and each block's release step."""
    steps = req.steps
    gone = frozenset(store.written.get(drop, ())) if drop is not None else frozenset()
    payloads = [store.payload.get(j, 0).to_bytes(RQ.BLOCK_BITS // 8, "little") for j in range(NR)]
    ys = [Entropy(b"a7 payload salts %d" % j)(receipts.SALT_BYTES) for j in range(NR)]
    root = audit.payload_tree(NODE, payloads, ys).commitment.root
    rs = C3.Store(audit.register(SOURCE, NODE, NR, root, P512, nonce=bytes(32)), payloads, ys)
    p = audit.Params(blocks=NR, deadline_ns=10**6, windows=WINDOWS, samples=c3.samples, delta=1e-6, window_ns=10, codec=P512,
                     lifetime=LIFE, schedule_cap=SCHEDULE_CAP)
    clock = Clock()
    s = C3.Session(p, rs.registration, None, coins.local(rs.registration.digest, 0, p.windows, Entropy(b"a7 coins")),
                   not_before_ns=0, clock=clock)
    prover = audit.Prover(rs, None, entropy=Entropy(b"a7 receipt salts"))
    head = MM.Header(rs.registration.digest, NR)
    replay = MM.Replay(head, rs.registration)
    slot_of, free_slots, nxt_slot, record, index = {}, [], NR, [], {}
    fresh = Entropy(b"a7 fresh salts")
    freed_at = {j: (late or {}).get(j, at) for at, js in store.freed.items() for j in js}
    challenges, previous, plan, t, i = [s.start()], rs.registration.digest, Plan.whole(NR), 0, 0
    passes, view = {}, {"plans": {}, "passes": [], "written": {}, "released": {}}
    per_window = -(-steps // WINDOWS)
    out, sent_early = None, None

    def honest(st):
        js = sorted(store.written.get(st, ()))
        return js, [payloads_of(store, j) for j in js], [salts[j] for j in js]

    def send(st, pay, ys_):
        a = rs.append(pay, ys_)
        got = s.append(a.root, a.count)
        if not isinstance(got, tuple):
            return got
        first, coin = got
        rs.encode(first, coin)
        for u in range(a.count):
            view["written"][first + u] = st
        return first

    for st in range(steps):
        if st and st % per_window == 0:
            clock.now += p.window_ns
            coin = s.boundary()
            if coin is not None:
                challenges.append(coin)
            t, i = t + 1, 0
        want = sorted(({index.get(b, b) for b in plans[st]} | set((extra or {}).get(st, ()))) - gone)
        new = Plan(_runs(want), rs.extent)
        change = (_runs(set(want) - set(plan)), _runs(set(plan) - set(want)))
        ps = prover.run(challenges[t], i, previous, new, head=rs.head)
        v = s.receive(t, i, ps.root, change=change, step=st)
        if v is not None:
            out = v
            break
        passes[t, i], previous, plan = ps, ps.root, ps.plan
        view["plans"][st] = want
        view["passes"].append((t, st))
        i += 1
        freed = sorted(j for j, f in freed_at.items() if f == st)
        written, pay, ys_ = honest(st)
        fs = [slot_of.pop(j, j) for j in freed]
        free_slots = sorted(free_slots + fs)
        for j in written:
            slot = free_slots.pop(0) if free_slots else nxt_slot
            nxt_slot += slot == nxt_slot
            slot_of[j] = slot
        slots, ext = [slot_of[j] for j in written], replay.extent
        if alter and st in alter:
            pay, ys_ = alter[st](pay, ys_)
        if dummy == st:
            slots, nxt_slot = slots + [nxt_slot], nxt_slot + 1
            pay, ys_ = pay + [bytes(RQ.BLOCK_BITS // 8)], ys_ + [fresh(receipts.SALT_BYTES)]
        if rekey is not None and rekey[0] == st:
            slots = slots + [slot_of[rekey[1]]]
            pay, ys_ = pay + [payloads_of(store, rekey[1])], ys_ + [fresh(receipts.SALT_BYTES)]
            index[rekey[1]] = ext + len(pay) - 1
        ws = tuple(MM.Written(slot, ext + u, codec.digest(m)) for u, (slot, m) in enumerate(zip(slots, pay, strict=True)))
        step_rec = MM.Step(st, tuple(fs), ws)
        record.append(step_rec)
        ch = replay.apply(step_rec)
        if ch.count and ch.first != sent_early and st != drop:
            got = send(st, pay, ys_)
            if not isinstance(got, int):
                out = got
                break
            if got != ch.first:
                raise AssertionError(f"step {st}: the session appended at {got}, the memory map at {ch.first}")
        if early == st + 1 and store.written.get(st + 1):
            _js, pay1, ys1 = honest(st + 1)
            got = send(st, pay1, ys1)
            if not isinstance(got, int):
                out = got
                break
            sent_early = got
        released = _runs(j for f, c in ch.released for j in range(f, f + c) if j not in gone)
        if released:
            v = s.release(released)
            if v is not None:
                out = v
                break
            for f, c in released:
                for j in range(f, f + c):
                    view["released"][j] = st
    verdicts = []
    if out is None:
        clock.now += p.window_ns
        s.boundary()
        clock.now = s.deadline(len(s.windows) - 1) + 1
        for w in range(len(s.windows)):
            got = s.close(w)
            if isinstance(got, audit.Verdict):
                out = got
                break
    by_check3 = None
    if out is None and not defer:
        verdicts = C3.reference(s, rs, passes, c3)
        out = by_check3 = next((v for v in verdicts if not v.accepted), None)
    accepted = out is None and len(verdicts) == WINDOWS and all(v.accepted for v in verdicts)
    return SimpleNamespace(session=s, store=rs, passes=passes, verdicts=verdicts, refused=out, by_check3=by_check3,
                           accepted=accepted, view=view, record=record, header=head, codec=P512,
                           map_text=MM.dumps(head, record), registration=rs.registration.digest)


def payloads_of(store, j: int) -> bytes:
    return store.payload.get(j, 0).to_bytes(RQ.BLOCK_BITS // 8, "little")


@dataclass(frozen=True)
class UnitsLink:
    """`audit.RequirementLink` on A7's units: it holds when the units' audit is accepted, every drawn unit's public and ok
    rows open, every witness row is bound to its canonical commitment, and the bridge (`bridge`) holds the session to the
    units' public data and the canonical pass row. `mapped` holds of a record, once every map unit is drawn and proved (`proved`), whose chain is
    the one `appends` the map units' writes make, each pass's head (`audit.WindowRecord.heads`) the chain through those
    of the steps below its own (`heads`, `units_heads`), and the map units' public lifetime cap (`life`) the record's.
    The stale bound, which no public input carries, is then what each re-key's old block is held below in the unit, which
    computes it from its committed pass row and that cap. `read_bits` is the step's reads from complete blocks
    (`RQ.credited_reads`, `creditedReadBits`). `epsilon` and `delta` are what the units' draw established
    (`uncertified`): the most share of the linked credit the wrong units it allows can carry, at the audit's delta; until
    they are set, the link credits nothing."""
    holds: bool
    req: RQ.Requirement
    layout: RQ.Layout
    appends: bytes | None = None
    proved: bool = False
    epsilon: float = 1.0
    delta: float = 0.0
    heads: tuple[bytes | None, ...] = ()
    life: int | None = None

    def read_bits(self, step: int) -> int:
        return self.req.bits(RQ.credited_reads(self.req, self.layout, step))

    def mapped(self, record: audit.Record) -> bool:
        passes = [(s, h) for w in record.windows for s, h in zip(w.steps, w.heads, strict=True)]
        return (self.holds and self.proved and record.appends == self.appends and self.life == record.lifetime
                and all(s < len(self.heads) and h == self.heads[s] for s, h in passes))


def units_heads(m: dict, pub_args: list[dict], registration: bytes) -> list[bytes | None]:
    """The chains over the appends member m's writes make (`audit.append_head`), from its public file and each step's
    public first and count: step t's append is blocks first .. first + count − 1, its leaves its payload rows' ``b || c``
    as hm96-sha512 tree leaves (a payload row's commit string is its append leaf's), in `audit.append_domain`; the
    steps in order, a step that writes nothing making none. Entry s is the chain through the appends of the steps below
    s, the head a pass at step s carries; the last, past every step, the whole chain."""
    bcs = public_bcs(m["dir"] / f"pub-{m['n']}.bin", m["ports"])
    head, out = None, []
    for i, a in enumerate(pub_args):
        out.append(head)
        if not a["count"]:
            continue
        leaves = {a["first"] + c: receipts.HM96.tree_leaf(receipts.KEY, bcs[i][f"pay{c}"]) for c in range(a["count"])}
        tree = MerkleTree(audit.append_domain(registration, a["first"], a["count"]), leaves, lambda _j: audit.PAYLOAD_SCHEMA)
        head = audit.append_head(head, a["first"], a["count"], tree.commitment.root)
    return out + [head]


def session_passes(steps: int) -> list[tuple[int, int]]:
    """The v0 session's passes, (window, step): a pass a step, `WINDOWS` windows of equal runs of steps."""
    per_window = -(-steps // WINDOWS)
    return [(s // per_window, s) for s in range(steps)]


@dataclass(frozen=True)
class StandInPostSeed:
    """A stand-in for I2's property verdict on the circuits that wrote the session's appends (not A7's to establish):
    the post-seed receipts credited, no carrier capacity."""
    holds: bool = True

    def carrier_bytes(self, read_bytes: int) -> float:
        return 0.0


# --- the units' instances --------------------------------------------------------------------------------------------------

def statics(steps: int, sd, layout, store) -> tuple[dict, dict]:
    reads = [RQ.read_blocks(sd.req, layout, s) for s in range(steps)]
    P_ = 16 * -(-max(len(r) for r in reads) // 16)
    V_ = 16 * -(-max(len(sd.positions), 1) // 16)
    M_ = max(16, 16 * -(-sum(len(b) for b in store.written.values()) // 16))
    K_ = max(sum(1 for v in sd.positions if sd.req.born[v] == s) for s in range(steps))
    C_ = max((len(b) for b in store.written.values()), default=1)
    return (dict(P=P_, NR=NR, V=V_ * EPOCHS, G=ACTIVE, TB=TB, RPB=RPB),
            dict(K=K_, C=C_, V=V_, E=EPOCHS, M=M_, N=NR, SLOTS=SLOTS, T=WINDOWS))


def instances(sd, layout, store, SP, SM, plans, pi, life):
    """Each step's PlanMatch and MemoryMapStep arguments, the latter with the pass row ``pi`` and lifetime cap ``life``."""
    lam = RU.layout_words(SM, sd.req, layout, sd.positions)
    phi = RU.map_words(SM, store)
    reg_block = {v: layout.block[v] for v in sd.req.registered}
    pa, ma = [], []
    for s in range(sd.req.steps):
        gathered = frozenset(sd.table[r] for r in sd.gathers[s])
        pa.append(RU.plan_inputs(SP, sd.req, s, plans[s], reg_block, sd.gathers[s], gathered, sd.positions, lam))
        ma.append(RU.map_inputs(SM, sd.req, store, s, sd.positions, lam, phi, reg_block, pi, life))
    return pa, ma


def public_words(args: dict, rows) -> list[int]:
    return [w for p in rows[0][2] for w in (args[p] if isinstance(args[p], list) else [args[p]])]


def stage(defn, S, rows, args: list[dict], name: str, out: Path, base: int, bound: dict, canons: dict | None, key: bytes | None,
          salt_of=None) -> dict:
    """One template's member: all its instances in one statement (checked against flock-circuit's shape), the bound rows
    under their canonical salts, or, ``salt_of``, row r of instance i under ``salt_of(r, i)`` (None: a fresh salt)."""
    t0 = time.time()
    in_rows = [(r, V2, b, ps) for r, b, ps in rows]
    st = CS.UnitStager(defn, f"unit/a7-{name}", in_rows, [(V2, (0,))], circuit=out / "circuit.txt", bind=bound)
    n = len(args)
    k_log = st.meta["k_log"]
    nbl = max(3, (-(-n // st.meta["vus_per_block"]) - 1).bit_length())
    if k_log + nbl > M_MAX or (k_log + nbl) < 25:
        raise SystemExit(f"{name}: m = {k_log} + {nbl} is outside 25 .. {M_MAX} (--zk and flock-circuit's shape)")
    words = [RU.row_words(S, rows, a) for a in args]
    data = {r: np.asarray([w[r] for w in words], dtype=np.uint16) for r, _b, _ps in rows}
    set_name = f"{st.iset.name}-c0"
    iset = SimpleNamespace(**{**vars(st.iset), "name": set_name})
    salt_key = hashlib.sha256(f"verity/one-stage/a7/salts\0{name}".encode()).digest()
    salts = FC.stage_salts(iset, 0, n, st.ports, salt_key)
    for i, a in enumerate(args):
        for r, b, ps in rows:
            if salt_of is not None:
                if (x := salt_of(r, i)) is not None:
                    salts[r][i] = x
                continue
            c = bound_row(name, r, i, a, S)
            if c is not None:
                kind, idx = c
                salts[r][i] = canons[kind]["salts"][idx]
    got = st.stage(data, out / "c0", salt_key, salts=salts, indices=range(base, base + n), set_name=set_name)
    d = out / "c0"
    (d / "inst.bin").rename(d / f"inst-{n}.bin")
    (d / "pub.bin").rename(d / f"pub-{n}.bin")
    (d / "tables").mkdir(exist_ok=True)
    os.link(out / "circuit.txt", d / "circuit.txt")
    oks = [int(x) for x in np.asarray(got["outs"]["o0"])[:, 0]]
    return {"tag": f"{name}-c0", "dir": d, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name, "n": n,
            "lo": 0, "cap": n, "ands": got["ands"], "unit_nnz": got["unit_nnz"], "salts": got["salts"], "ports": st.ports,
            "oks": oks, "words": words, "stager": st, "seconds": {"stage": round(time.time() - t0, 1)}}


def bound_row(template: str, r: str, i: int, a: dict, S) -> tuple[str, int] | None:
    """The canonical commitment and row instance i's row ``r`` is bound to, or None (its public row, its ok row, a padding
    row): the positions the public requirement names."""
    if r == "lam":
        return "layout", 0
    if template == "plan":
        if r == "plan":
            return "plan", i
        if r.startswith("tok"):
            g = int(r[3:])
            return ("token", i * S.G + g) if (a["gmask"] >> g) & 1 else None
        return None
    if r == "phi":
        return "map", 0
    if r == "pi":
        return "passes", 0
    kind, at = RU.map_value_leaves(S, a).get(r, (None, 0))
    return ("value", at) if kind == "value" else ("payload", at - S.N) if kind == "block" else None


def bindings(m: dict, canons: dict, pub_args: list[dict], S) -> dict:
    """Every bound row's ``b || c`` in member m's public file against its canonical commitment's, at the positions the
    verifier's own copy of the public requirement names (``pub_args``: each instance's public data)."""
    bcs = public_bcs(m["dir"] / f"pub-{m['n']}.bin", m["ports"])
    bad = []
    for i, a in enumerate(pub_args):
        for p in m["ports"]:
            c = bound_row(m["template"], p.name, i, a, S)
            if c is not None and bcs[i][p.name] != canons[c[0]]["bc"][c[1]]:
                bad.append({"instance": i, "row": p.name, "canonical": list(c)})
    return {"ok": not bad, "codes": ["binding"] if bad else [], "detail": bad[:8]}


def openings(m: dict, drawn: list[int], pub_args: list[dict], rows) -> dict:
    """Each drawn instance's public row opened to the verifier's own copy of its public data and its ok row to 1, under the
    salts the prover reveals for those two rows alone."""
    bcs = public_bcs(m["dir"] / f"pub-{m['n']}.bin", m["ports"])
    port = {p.name: p for p in m["ports"]}
    bad = []
    for i in drawn:
        if not opens(port["pub"], public_words(pub_args[i], rows), m["salts"]["pub"][i], bcs[i]["pub"]):
            bad.append({"instance": i, "row": "pub"})
        if not opens(port["o0"], [1], m["salts"]["o0"][i], bcs[i]["o0"]):
            bad.append({"instance": i, "row": "o0"})
    return {"ok": not bad, "codes": ["opening"] if bad else [], "detail": bad[:8]}


def frame_ports(set_name: str, meta: dict, n: int) -> dict[str, dict]:
    out = {}
    for (port, _w, _c), (schema, _bits) in zip(meta["ports"], meta["row_schemas"], strict=True):
        b = identity_digest_sha512(a0.STAND_IN_BINDING, {"set": set_name, "lo": 0, "hi": n, "port": port, "schema": schema})
        out[port] = {"leaves": n, "domain": CommitmentDomain(b, -1, RangeIndexedDomain(0, n), hash="sha512").domain_id.hex()}
    return out


def statement_checks(m: dict, rec: dict, want: dict) -> dict:
    meta, header, fv = m["meta"], m["header"], m["header"]["frame_v3"]
    ok = {"scheme": header.get("scheme") == a0.SCHEME, "pin": header.get("circuit_sha512") == m["pin"],
          "partition": meta.get("partition") == {"rule": P.FORMAT, "digest": rec["partition"]},
          "program": meta.get("program_sha512") == rec["program"], "template_program": a0.template_program_ok(meta, m["defn"]),
          "instances": header["instances"] == m["n"],
          "indices": header["units"]["indices"] == list(range(m["base"], m["base"] + m["n"])),
          "classes": header["units"].get("classes") == [m["class"]] * m["n"]}
    registered = {r["port"]: r for r in rec["roots"]}
    for port, root in fv["roots"].items():
        name = f"{m['name']}/{port}"
        r = registered.get(name)
        ok[f"root:{port}"] = r is not None and r["root"] == root and r["domain"] == fv["domain_ids"][port] == want[name]["domain"]
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {}}


# --- check 3 ----------------------------------------------------------------------------------------------------------------

def check3_audit(a, out: Path, rs, c3, port: int) -> dict:
    """The remote session's check 3 by proofs: each draw's opening unit (`opening.witness`), one member a receipt-leaf
    size, its plan row under the canonical plan's salt of the draw's step and every other row under a fresh salt,
    registered, every unit drawn by the verifier's Lean and each member proved --zk and held by Lean `verify --zk
    --partition --program`. The verifier opens each drawn unit's public rows to its own copy (`opening.public`) and its
    verdict row, holds its plan row to the certified one by ``b || c`` (`opening.plan_link`), and decides each window of
    the session on its draws. No block, payload, receipt leaf or plan row is opened."""
    out.mkdir(parents=True, exist_ok=True)
    sess, st_ = rs.session, rs.store
    draws = [d for t in range(len(sess.windows)) for d in sess.drawn(t)]
    seed = st_.registration.codec_seed
    Ss = [C3.statics(c3.codec, d.leaf_size, c3.P, c3.DR, c3.DP, c3.DV) for d in draws]
    wits = [C3.witness(d, rs.passes[d.window, d.index], st_, c3.plan_rows[d.step], S, c3.values) for d, S in zip(draws, Ss)]
    prover = [C3.check(d, w, seed, S, c3.values, c3.plan_salts[d.step], c3.plan_bcs[d.step]) for d, w, S in zip(draws, wits, Ss)]
    if any(prover):
        raise SystemExit(f"the honest prover's check-3 units are refused by their reference: {prover}")
    specs = []
    for leaf in sorted({d.leaf_size for d in draws}):
        at = [u for u, d in enumerate(draws) if d.leaf_size == leaf]
        S = Ss[at[0]]
        specs.append({"template": f"opening{leaf}", "defn": bind(BC.BlockOpeningCheck, **vars(S)), "S": S,
                      "rows": BC.opening_rows(S), "args": [wits[u] for u in at], "draw": at})
    program = P.mixed_population_program([(s["defn"], len(s["args"])) for s in specs])
    query = P.template_instances_query([s["defn"] for s in specs])
    part = P.partition(program, query)
    n_all, ranges = P.units(program, query), P.ranges(program, query)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    law = D.parse_law(f"stratified:{n_all}")
    bound = {"partition": P.digest(part), "program": part["program"]}
    members = []
    for s, (tid, base, cnt) in zip(specs, ranges, strict=True):
        assert tid == P.template_id(s["defn"]) and cnt == len(s["args"])

        def salt_of(r, i, s=s):
            return c3.plan_salts[draws[s["draw"][i]].step] if r == "plan" else None

        m = stage(s["defn"], s["S"], s["rows"], s["args"], f"c3-{s['template']}", out / f"stage-{s['template']}", base, bound,
                  None, None, salt_of=salt_of)
        m.update(partition_file=out / "partition.json", program_file=out / "program.json", template=s["template"],
                 name=m["tag"], base=base, defn=s["defn"], S=s["S"], rows=s["rows"], args=s["args"], draw=s["draw"],
                 header=a0.public_header(m["dir"], m["n"]))
        members.append(m)
        print(f"A7 check 3 staged {m['tag']}: {m['n']} units from {base}, {m['ands']} ANDs, k_log {m['meta']['k_log']}, "
              f"verdicts {m['oks']}, {m['seconds']}", flush=True)

    want, roots, publics = {}, [], {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for p_, w in frame_ports(m["set"], m["meta"], m["n"]).items():
            want[f"{m['name']}/{p_}"] = w
            roots.append({"port": f"{m['name']}/{p_}", "root": fv["roots"][p_], "leaves": w["leaves"], "domain": fv["domain_ids"][p_]})
        publics[m["name"]] = a0.sha512_file(m["dir"] / f"pub-{m['n']}.bin")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    window = {"kind": "stand-in", "of": "check 3 of the remote session's drawn receipts (a7.py, check3_audit), on the stand-in's store",
              "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]]} for m in members},
              "link": "opening.plan_link: the plan row by its b || c; every other witness row inside the unit"}
    rec = R.record(program, query, law, a0.SCHEME, roots, window, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        checks[f"statement/{m['name']}"] = statement_checks(m, rec, want)
    receipt = R.receipt(rec, "one-stage-e2e verifier (a7.py, check 3)")
    summary = {"draws": [{"window": d.window, "pass": d.index, "position": d.position, "block": d.block, "step": d.step,
                          "appended": d.coin is not None, "leaf": d.leaf_size} for d in draws],
               "members": [{"name": m["name"], "n": m["n"], "ands": m["ands"], "unit_nnz": m["unit_nnz"], "k_log": m["meta"]["k_log"],
                            "pin": m["pin"], "verdicts": m["oks"], "stage_s": m["seconds"],
                            "statics": vars(m["S"])} for m in members],
               "population": n_all, "checks_before_draw": {c: v["ok"] for c, v in checks.items()}}
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks

    verdicts, proved, n_inst, statement, sessions, served, session_roots = {}, [], 0, {}, [], [], {}
    lean_draw = {"units": []}
    if all(v["ok"] for v in checks.values()):
        p = a0.run([a.lean, "draw", "--program", str(out / "program.json"), "--partition", str(out / "partition.json"),
                    "--stratified", str(n_all)], out / "draw.log")
        lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
        for j, m in enumerate(members):
            local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
            m["drawn"] = local
            if not local:
                continue
            sdir = out / f"session-{m['name']}"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": m["n"], "k": len(local), "units": local}, sort_keys=True))
            t = time.time()
            record, live = a0.session(a.bin, m, m["n"], f"subset:{len(local)}", port + j, sdir, draw_file=sdir / "draw.json", zk=True)
            draw_t = (record or {}).get("unit_draw") or {}
            v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
            served += [m["base"] + u for u in draw_t.get("units", [])]
            session_roots.update({f"{m['name']}/{p_}": r for p_, r in ((live or {}).get("roots") or {}).items()})
            proved += a0.proved_units(m["header"], live or {})
            n_inst += (live or {}).get("instances", 0)
            verdicts[f"m0-verifier/{m['name']}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                                    and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
                                                    "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
            verdicts[f"lean-verify/{m['name']}"] = {**a0.lean_verify(a.lean, m, m["n"], sdir, zk=True),
                                                    "tool": "flock-verify verify --zk --archive --partition --program"}
            statement[m["name"]] = {"session_s": round(time.time() - t, 1),
                                    **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                                    "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")},
                                    "lean_verify_s": verdicts[f"lean-verify/{m['name']}"].get("verify_s")}
            sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
            lv = verdicts[f"lean-verify/{m['name']}"]
            print(f"A7 check 3 session {m['name']}: {len(local)} units, Lean {lv['accepted']} {lv.get('why') or ''}, "
                  f"{statement[m['name']]['session_s']} s", flush=True)
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), n_inst, statement, verdicts, a.delta,
                            sessions=sessions, served=sorted(served), session_roots=session_roots)
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    # the verifier: each draw's unit proved, its public rows and verdict opened, its plan row linked; each window decided
    held = set(proved) if result.get("accepted") else set()
    failures = []
    for u, d in enumerate(draws):
        m = next(m for m in members if u in m["draw"])
        i = m["draw"].index(u)
        S = Ss[u]
        mine = C3.public(d, seed, S, c3.values)
        bcs = public_bcs(m["dir"] / f"pub-{m['n']}.bin", m["ports"])[i]
        port_ = {p_.name: p_ for p_ in m["ports"]}
        word = m["oks"][i]
        opened = ((m["base"] + i) in held and opens(port_["pub"], public_words(mine, m["rows"]), m["salts"]["pub"][i], bcs["pub"])
                  and opens(port_["signs"], mine["signs"], m["salts"]["signs"][i], bcs["signs"])
                  and opens(port_["o0"], [word], m["salts"]["o0"][i], bcs["o0"]))
        failures.append("malformed" if not opened else BC.verdict(word) or C3.plan_link(bcs["plan"], c3.plan_bcs[d.step]))
    decided = []
    for t in range(len(sess.windows)):
        decided.append(sess.decide(t, [f for d, f in zip(draws, failures) if d.window == t]))
        if not decided[-1].accepted:
            break
    rs.verdicts = decided
    rs.refused = next((v for v in decided if not v.accepted), None)
    rs.accepted = rs.refused is None and len(decided) == WINDOWS
    summary.update(failures=failures, accepted=rs.accepted, audit={x: result.get(x) for x in ("accepted", "complete", "wrong_units")},
                   opened_rows=["o0", "pub", "signs"], linked_rows=["plan"],
                   verdicts={x: y.get("accepted") for x, y in verdicts.items()}, timing=statement, session_verdicts=[
                       {"window": v.window, "accepted": v.accepted, "failure": v.failure} for v in decided])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return {**summary, "members_": members, "draws_": draws, "witnesses_": wits, "statics_": Ss}


def check3_negatives(c3a: dict, rs, c3) -> dict[str, dict]:
    """Check 3's negatives, each on one drawn unit of the run (an appended block's, if one was drawn) with one witness or
    public input altered: the unit's verdict on it (the lowering's own evaluation, `_forged`) and the check it names, or,
    for a prover whose unit accepts a plan row other than the certified one, the verifier's plan link on that row's
    ``b || c`` (`opening.plan_link`)."""
    neg: dict[str, dict] = {}
    members, draws, wits = c3a["members_"], c3a["draws_"], c3a["witnesses_"]
    u0 = next((u for u, d in enumerate(draws) if d.coin is not None), 0)
    d0, a0_ = draws[u0], wits[u0]
    m = next(m for m in members if u0 in m["draw"])
    i = m["draw"].index(u0)
    seed = rs.store.registration.codec_seed

    def unit(name: str, want: str, **alter) -> dict:
        f = _forged(m, i, dict(a0_, **alter))
        got = BC.verdict(f["ok_output"])
        neg[name] = {"refused": got == want and not f["opens_to_1"], "verdict": got, "block": d0.block,
                     "appended": d0.coin is not None, "unit": f}
        return f

    def flip(words: list[int], at: int, bit: int = 1) -> list[int]:
        return words[:at] + [words[at] ^ bit] + words[at + 1:]

    def u16s(b: bytes) -> list[int]:
        return list(struct.unpack(f"<{len(b) // 2}H", b))

    unit("tampered-block (codec)", "codec", block=flip(a0_["block"], 0))
    w0 = c3.codec.field.n // 16
    other = bytearray(rs.store.payload(d0.block))
    other[2 * (w0 - 1)] ^= 1 << 3
    k = codec.key(seed, c3.codec)
    reenc = codec.encode(bytes(other), d0.block, k, bound=d0.coin is not None, coin=d0.coin)
    unit("payload-not-the-committed-one (payload-path: a block that decodes to another payload)", "payload-path",
         block=u16s(reenc), payload=u16s(bytes(other)))
    unit("payload's-salt-not-its-commitment's (payload-path)", "payload-path", salts=flip(a0_["salts"], 96 + 5))
    if d0.coin is not None:
        rows = len(c3.values.bcs)
        r2 = next(r for r in range(rows) if r != d0.block - c3.values.base)
        unit("payload-not-the-committed-KV-row (payload-path)", "payload-path",
             vpath=BC._u16s(b"".join(c3.values.path(r2)) + bytes(64 * (m["S"].DV - len(c3.values.path(r2))))))
        oc = next((c for (_f, _d, _c, c) in rs.session._appends if c != d0.coin), None)
        if oc is None:
            neg["wrong-append-coin (codec)"] = {"refused": False, "why": "one append only"}
        else:
            coin = u16s(oc)
            unit("wrong-append-coin (codec)", "codec", coin=coin)
            port = {p_.name: p_ for p_ in m["ports"]}["pub"]
            bcs = public_bcs(m["dir"] / f"pub-{m['n']}.bin", m["ports"])[i]
            pub = opens(port, public_words(dict(a0_, coin=coin), m["rows"]), m["salts"]["pub"][i], bcs["pub"])
            neg["wrong-append-coin (codec)"].update(public_row_opens=pub, refused=neg["wrong-append-coin (codec)"]["refused"] and not pub)
    else:
        neg["payload-not-the-committed-KV-row (payload-path)"] = {"refused": False, "why": "no draw of an appended block"}
        neg["wrong-append-coin (codec)"] = {"refused": False, "why": "no draw of an appended block"}
    kk = d0.position % receipts.PER_LEAF
    unit("wrong-receipt-leaf (receipt-path)", "receipt-path", leaf=flip(a0_["leaf"], 32 * kk))
    unit("receipt-leaf's-salt-not-the-pass's (receipt-path)", "receipt-path", salts=flip(a0_["salts"], 5))
    if a0_["rdepth"][0]:
        unit("receipt-path-not-the-pass's (receipt-path)", "receipt-path", rpath=flip(a0_["rpath"], 7))
    unit("other-pass's-signs (receipt)", "receipt", signs=[x ^ 0x5A5A for x in a0_["signs"]])
    pos = d0.position
    unit("plan-entry-not-the-block (receipt-path)", "receipt-path", plan=flip(a0_["plan"], pos))
    plan = flip(a0_["plan"], next(x for x in range(len(a0_["plan"])) if x != pos))
    f = _forged(m, i, dict(a0_, plan=plan))
    why = C3.plan_link(C3.commit(16 * m["S"].P, plan, c3.plan_salts[d0.step]), c3.plan_bcs[d0.step])
    neg["plan-row-not-the-step's (plan link)"] = {"refused": BC.verdict(f["ok_output"]) is None and why == "plan", "unit": f,
                                                   "link": why}
    return neg


# --- the run ----------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=None, help="the stratified rule's k over both templates' units (default: all)")
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--samples", type=int, default=SAMPLES, help="the remote session's draws a window, each proved by check 3")
    ap.add_argument("--port", type=int, default=7571)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"steps": a.steps, "k": a.k, "delta": a.delta, "seed": a.seed}

    # the stand-in requirement, its private layout and store, each step's certified plan
    sd = standin(a.steps, a.seed)
    req = sd.req
    layout, store = lay_out(req, sd.positions, a.seed)
    honest = all(RQ.store_matches(req, layout, store, RQ.BLOCK_BITS, s) for s in range(req.steps)) and all(
        RQ.plan_matches(req, layout, RQ.read_blocks(req, layout, s), RQ.BLOCK_BITS, s) for s in range(req.steps))
    plans = {s: sorted(RQ.read_blocks(req, layout, s)) for s in range(req.steps)}
    SP, SM = statics(a.steps, sd, layout, store)
    summary["standin"] = {"honest": honest, "values": len(sd.positions), "written_blocks": sum(len(b) for b in store.written.values()),
                          "paused": len(sd.paused), "requests": sd.requests, "plan_statics": SP, "map_statics": SM,
                          "read_bits": [req.read_bits(s) for s in range(req.steps)],
                          "live_bits": [req.live_bits(s) for s in range(req.steps)]}
    print("A7 stand-in " + json.dumps(summary["standin"], default=str)[:1500], flush=True)
    if not honest:
        raise SystemExit("the stand-in's layout is not honest: fix lay_out")
    plan_def, map_def = bind(RU.PlanMatch, **SP), bind(RU.MemoryMapStep, **SM)
    sp, sm = SimpleNamespace(**SP), SimpleNamespace(**SM)
    plan_rows, map_rows = RU.plan_rows(sp), RU.memory_map_rows(sm)
    pi = RU.pass_words(sm, session_passes(req.steps))
    pa, ma = instances(sd, layout, store, sp, sm, plans, pi, LIFE)
    if [RU.plan_match(sp, **{k: x[k] for k in ("count", "regmask", "kvmask", "gmask", "lam", "plan", "tok")}) for x in pa] != [1] * a.steps:
        raise SystemExit("a step's plan instance is refused by the reference")
    if [RU.memory_map(sm, **x) for x in ma] != [1] * a.steps:
        raise SystemExit("a step's memory-map instance is refused by the reference")

    # the canonical commitments
    key = hashlib.sha256(b"verity/one-stage/a7/canon\0%d" % a.seed).digest()
    vww = VW // 16
    value_words = [RU._words(req.values[v], vww) for v in sd.positions]
    written = sorted(j for b in store.written.values() for j in b)
    canons = {"layout": canon("layout", 32 * sm.V * sm.E, [pa[0]["lam"]], key),
              "map": canon("map", 16 * (sm.N + 2 * sm.M), [ma[0]["phi"]], key),
              "passes": canon("passes", 16 * sm.T, [pi], key),
              "plan": canon("plan", 16 * sp.P, [x["plan"] for x in pa], key),
              "token": canon("token", 16, [[x["tok"][g]] for x in pa for g in range(sp.G)], key),
              "value": canon("value", VW, value_words, key),
              "payload": canon("payload", RQ.BLOCK_BITS, [RU._words(store.payload[j], RU.BLOCK_WORDS) for j in written], key)}
    for c in canons.values():
        (out / f"canon-{c['kind']}.bc").write_bytes(b"".join(c["bc"]))
    salts = {j: canons["payload"]["salts"][u] for u, j in enumerate(written)}

    # the program, its partition and the law
    program = P.mixed_population_program([(plan_def, a.steps), (map_def, a.steps)])
    query = P.template_instances_query([plan_def, map_def])
    part = P.partition(program, query)
    pd = P.digest(part)
    n_all = P.units(program, query)
    ranges = P.ranges(program, query)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    a.k = n_all if a.k is None else a.k
    summary["k"] = a.k
    law = D.parse_law(f"stratified:{a.k}")
    bound = {"partition": pd, "program": part["program"]}

    # serving commits: each template's one member, bound to the partition
    members = []
    for (defn, S, rows, args, tmpl), (tid, base, cnt) in zip(((plan_def, sp, plan_rows, pa, "plan"), (map_def, sm, map_rows, ma, "map")), ranges):
        assert tid == P.template_id(defn) and cnt == a.steps
        st = stage(defn, S, rows, args, tmpl, out / f"stage-{tmpl}", base, bound, canons, key)
        st.update(partition_file=out / "partition.json", program_file=out / "program.json", template=tmpl, name=st["tag"],
                  base=base, defn=defn, S=S, rows=rows, args=args, header=a0.public_header(st["dir"], st["n"]))
        members.append(st)
        print(f"A7 staged {st['tag']}: {st['n']} units from {base}, {st['ands']} ANDs, {st['unit_nnz']} nonzeros, k_log "
              f"{st['meta']['k_log']}, ok {sum(st['oks'])}/{st['n']}, {st['seconds']}", flush=True)
    summary["members"] = [{"name": m["name"], "template": P.template_id(m["defn"]), "n": m["n"], "base": m["base"], "ands": m["ands"],
                           "unit_nnz": m["unit_nnz"], "k_log": m["meta"]["k_log"], "pin": m["pin"], "class": m["class"],
                           "oks": sum(m["oks"]), "stage_s": m["seconds"]} for m in members]

    # one registration, before the draw: the statements' ports and the canonical commitments
    want, roots, publics = {}, [], {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for port, w in frame_ports(m["set"], m["meta"], m["n"]).items():
            want[f"{m['name']}/{port}"] = w
            roots.append({"port": f"{m['name']}/{port}", "root": fv["roots"][port], "leaves": w["leaves"], "domain": fv["domain_ids"][port]})
        publics[m["name"]] = a0.sha512_file(m["dir"] / f"pub-{m['n']}.bin")
    for c in canons.values():
        name = f"canon/{c['kind']}"
        want[name] = {"leaves": c["n"], "domain": canon_domain(c["kind"], c["bits"], c["n"])}
        roots.append({"port": name, "root": c["root"], "leaves": c["n"], "domain": c["domain"]})
        publics[name] = a0.sha512_file(out / f"canon-{c['kind']}.bc")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    window = {"kind": "stand-in", "of": WINDOW_OF,
              "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]]} for m in members},
              "link": "none: no served run; vllm's Store backend and its served values replace these"}
    rec = R.record(program, query, law, a0.SCHEME, roots, window, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for c in canons.values():
        bcs = [bytes(x) for x in np.frombuffer((out / f"canon-{c['kind']}.bc").read_bytes(), dtype=np.uint8).reshape(-1, 128)]
        checks[f"canon/{c['kind']}"] = {"ok": canon_root(c, bcs) == c["root"], "codes": [], "detail": {}}
    for m in members:
        pub_args = m["args"]
        checks[f"statement/{m['name']}"] = statement_checks(m, rec, want)
        checks[f"bindings/{m['name']}"] = bindings(m, canons, pub_args, m["S"])
    receipt = R.receipt(rec, "one-stage-e2e verifier (a7.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(population=n_all, partition=pd, registration=R.digest(rec), law=rec["law"],
                   checks_before_draw={c: v["ok"] for c, v in checks.items()})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("A7 " + json.dumps({"refused_before_draw": summary["checks_before_draw"], "checks": checks}, default=str)[:4000])
        return 1

    # the verifier's stratified draw, by its own Lean; each member serves its stratum, --zk
    p = a0.run([a.lean, "draw", "--program", str(out / "program.json"), "--partition", str(out / "partition.json"),
                "--stratified", str(a.k)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(lean_draw, sort_keys=True))
    served, proved, n_inst, verdicts, statement, sessions, counts, session_roots, opened = [], [], 0, {}, {}, [], {}, {}, {}
    for j, m in enumerate(members):
        local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
        counts[m["name"]] = len(local)
        m["drawn"] = local
        if not local:
            continue
        sdir = out / f"session-{m['name']}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": m["n"], "k": len(local), "units": local}, sort_keys=True))
        t = time.time()
        record, live = a0.session(a.bin, m, m["n"], f"subset:{len(local)}", a.port + j, sdir, draw_file=sdir / "draw.json", zk=True)
        session_s = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
        served += [m["base"] + u for u in draw_t.get("units", [])]
        session_roots.update({f"{m['name']}/{port}": r for port, r in ((live or {}).get("roots") or {}).items()})
        proved += a0.proved_units(m["header"], live or {})
        n_inst += (live or {}).get("instances", 0)
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        verdicts[f"m0-verifier/{m['name']}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                                and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
                                                "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        verdicts[f"lean-draw/{m['name']}"] = {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}
        verdicts[f"lean-verify/{m['name']}"] = {**a0.lean_verify(a.lean, m, m["n"], sdir, zk=True),
                                                "tool": "flock-verify verify --zk --archive --partition --program"}
        opened[m["name"]] = openings(m, local, m["args"], m["rows"])
        verdicts[f"openings/{m['name']}"] = {"accepted": opened[m["name"]]["ok"], "detail": opened[m["name"]]["detail"]}
        statement[m["name"]] = {"session_s": session_s, **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                                "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")},
                                "verify_s": ((record or {}).get("verdict") or {}).get("verify_s"),
                                "lean_verify_s": verdicts[f"lean-verify/{m['name']}"].get("verify_s"),
                                "lean_partition": verdicts[f"lean-verify/{m['name']}"].get("partition")}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        lv = verdicts[f"lean-verify/{m['name']}"]
        print(f"A7 session {m['name']}: {len(local)} units, Lean {lv['accepted']} {lv.get('why') or ''}, openings "
              f"{opened[m['name']]['ok']}, {session_s} s", flush=True)
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), n_inst, statement,
                            {k: v for k, v in verdicts.items() if not k.startswith("openings/")}, a.delta,
                            sessions=sessions, served=sorted(served), session_roots=session_roots)
    result["draw_counts"] = counts
    result["checks_before_draw"] = checks
    result["openings"] = opened
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    units_ok = bool(result.get("accepted")) and all(o["ok"] for o in opened.values())

    # the link: the v0 session, its check 3 proved, the bridge, the ledger
    c3 = check3_inputs(canons, sp, sm, a.samples)
    rs = remote_session(req, layout, store, plans, salts, c3=c3, defer=True)
    (out / "memory-map.jsonl").write_text(rs.map_text)
    if rs.refused is not None:
        raise SystemExit(f"the honest remote session was refused before check 3: {rs.refused}")
    check3 = check3_audit(a, out / "check3", rs, c3, a.port + len(members))
    summary["check3"] = {x: check3[x] for x in ("draws", "members", "failures", "accepted", "audit", "verdicts")}
    print("A7 check 3 " + json.dumps(summary["check3"], default=str)[:3000], flush=True)
    if not check3["accepted"]:
        summary["check3"]["checks"] = check3.get("checks")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("A7 " + json.dumps({"refused_by_check3": check3["session_verdicts"], "checks": check3.get("checks")}, default=str)[:4000])
        return 1
    br = bridge(rs, members[0], members[1], req.steps, canons["passes"], sm)
    session_ok = rs.accepted
    map_m = members[1]
    proved = units_ok and map_m.get("drawn") == list(range(map_m["n"]))
    heads = units_heads(map_m, map_m["args"], rs.registration)
    lives = {x["life"] for x in map_m["args"]}
    link = UnitsLink(holds=units_ok and session_ok and br["ok"] and all(v["ok"] for v in checks.values()),
                     req=req, layout=layout, appends=heads[-1], proved=proved, heads=tuple(heads),
                     life=min(lives) if len(lives) == 1 else None)
    rrec = rs.session.record()
    mapped = link.mapped(rrec)
    common = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))
    uncert = uncertified(result, rrec, rs.view, store, req, link, common, members)
    if uncert.get("credit_bytes"):
        link = replace(link, epsilon=min(1.0, uncert["uncertified_bytes"] / uncert["credit_bytes"]), delta=a.delta)
    scheduled, schedule_why, schedule_bits = None, None, None
    try:
        sched = MM.schedule(rs.header, rs.record, rs.session.appended, [s for w in rrec.windows for s in w.steps])
        schedule_bits = sched.bits
        scheduled = rs.session.scheduled(sched)
    except (MM.MemoryMapError, SC.Refused) as e:
        schedule_why = str(e)
    ledgers = {"store": audit.ledger(rrec, **common).to_json()}
    if scheduled is not None:
        ledgers["store+post-seed"] = audit.ledger(rrec, **common, credit_post_seed=StandInPostSeed(), schedule=scheduled).to_json()
    if link.holds and mapped:
        ledgers["linked"] = audit.ledger(rrec, **common, requirement=link).to_json()
        if scheduled is not None:
            ledgers["linked+post-seed"] = audit.ledger(rrec, **common, requirement=link, credit_post_seed=StandInPostSeed(),
                                                       schedule=scheduled).to_json()
    whole = common["capacity_bytes_per_s"] * rrec.span_ns / 1e9
    summary["link"] = {"holds": link.holds, "mapped": mapped, "every_map_unit_proved": proved,
                       "appends": None if rrec.appends is None else rrec.appends.hex(),
                       "units_appends": None if link.appends is None else link.appends.hex(),
                       "heads_are_the_units'": all(h == link.heads[s] for w in rrec.windows for s, h in zip(w.steps, w.heads)),
                       "life": link.life, "lifetime": rrec.lifetime, "pass_row": pi,
                       "units": units_ok, "session": session_ok, "check3": check3["accepted"], "bridge": br,
                       "epsilon": link.epsilon, "delta": link.delta,
                       "verdicts": [v.accepted for v in rs.verdicts], "uncertified": uncert,
                       "schedule": {"bits": schedule_bits, "cap": None if scheduled is None else scheduled.cap,
                                    "refused": schedule_why},
                       "circuit_read_bytes": sum(req.read_bits(s) for s in range(req.steps)) // 8,
                       "ledgers": {k: {"gamma": v["gamma"], "timing": v["terms"]["timing"], "linked": v.get("linked"),
                                       "credited_bytes": round((1 - v["terms"]["timing"]) * whole)} for k, v in ledgers.items()}}
    (out / "ledger.json").write_text(json.dumps(ledgers, indent=1, sort_keys=True, default=str))

    neg = negatives(a, out, rec, receipt, result, members, canons, sd, layout, store, plans, sp, sm, req, lean_draw, law,
                    program, query, want, leaf_layer, publics, session_roots, served, n_all, common, salts, link, rrec, c3, rs)
    neg.update(check3_negatives(check3, rs, c3))
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True, default=str))
    summary["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units", "profile")}
    summary["audit"]["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    summary["audit"]["why"] = {x: y.get("why") for x, y in verdicts.items() if y.get("accepted") is not True}
    summary.update(draw=lean_draw["units"], draw_counts=counts, served_is_lean_draw=sorted(served) == sorted(lean_draw["units"]),
                   timing=statement, negatives={x: y["refused"] for x, y in neg.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = units_ok and check3["accepted"] and link.holds and mapped and all(summary["negatives"].values())
    print("A7 " + json.dumps({x: summary[x] for x in ("population", "draw_counts", "audit", "link", "negatives")}, sort_keys=True,
                             default=str)[:6000])
    return 0 if ok else 1


def bridge(rs, plan_m: dict, map_m: dict, steps: int, passes: dict, sm) -> dict:
    """The bridge from the session to the units, on public data and the pass row: each pass's plan is as long as its
    step's plan unit's public count, and check 3's unit holds each drawn entry to the step's certified plan row; each
    pass's seed binds the chain over the map units' appends of the steps before it (`units_heads`), so each append
    follows its step's pass and precedes the next; the registered blocks the session releases after each step are those
    that step's map unit frees (its public ``rfree``); and the prover opens the canonical pass row (``passes``, its salt)
    to each window's last pass step on v0's wire (`RU.pass_words`), the one canonical row opened, until the record
    commits to the passes' steps. A written block's release step is the map row's, private, and isn't bound: a release
    only narrows the plans the session admits, which are the certified ones, so it moves no credit."""
    bad = []
    heads = units_heads(map_m, map_m["args"], rs.registration)
    for w in rs.session.windows:
        for plan, s, head in zip(w.plans, w.steps, w.heads, strict=True):
            if plan.length != plan_m["args"][s]["count"]:
                bad.append({"plan": s, "length": plan.length, "count": plan_m["args"][s]["count"]})
            if head != heads[s]:
                bad.append({"head": s})
    for s in range(steps):
        got = sorted(j for j, at in rs.view["released"].items() if at == s and j < NR)
        want = [j for j, b in enumerate(RU._bits(map_m["args"][s]["rfree"], NR)) if b]
        if got != want:
            bad.append({"released": s, "session": got, "map": want})
    pi = RU.pass_words(sm, rs.view["passes"])
    if not opens(passes["port"], pi, passes["salts"][0], passes["bc"][0]):
        bad.append({"passes": 0})
    return {"ok": not bad, "detail": bad[:8]}


def uncertified(result: dict, rrec, view: dict, store, req, link, common, members) -> dict:
    """What the sample leaves uncertified, for memory accounting's ledger: the profile's bound on wrong units of each
    template at delta, and the most credit those could carry (the passes whose plan unit is wrong, and those holding a block
    a wrong map unit wrote), as a fraction of the session's capacity; over the credit, it is the link's ε."""
    wrong = result.get("wrong_units")
    if not isinstance(wrong, dict):
        return {"wrong_units": wrong, "term": None}
    per_t = wrong["per_template"]
    sizes = {s["name"]: s["size"] for s in (result.get("profile") or {}).get("strata", [])}
    w_plan, w_map = (per_t[P.template_id(m["defn"])] for m in members)
    credit = {}
    for w in rrec.windows:
        for c, s in zip(audit.linked_bits(w, link, rrec.block_bytes, True), w.steps, strict=True):
            credit[s] = c / 8
    whole = common["capacity_bytes_per_s"] * rrec.span_ns / 1e9
    top_plan = sum(sorted(credit.values(), reverse=True)[:w_plan])
    reach = {t: sum(c for s, c in credit.items() if s > t and set(view["plans"].get(s, ())) & set(js)) for t, js in store.written.items()}
    top_map = sum(sorted(reach.values(), reverse=True)[:w_map])
    at_most = min(top_plan + top_map, sum(credit.values()))
    return {"wrong_units": wrong, "epsilon": {t: per_t[t] / sizes[t] for t in per_t if sizes.get(t)},
            "plan_credit_bytes": top_plan, "map_credit_bytes": top_map, "credit_bytes": sum(credit.values()),
            "uncertified_bytes": at_most, "term": at_most / whole}


def _forged(m: dict, i: int, args: dict) -> dict:
    """Instance i of member m with ``args`` in place of its own: the circuit's ok output on them (the lowering's own
    evaluation), and whether that ok row, under the instance's ok salt, opens to 1."""
    st = m["stager"]
    words = RU.row_words(m["S"], m["rows"], args)
    flat = np.concatenate([np.asarray(words[r], dtype=np.uint16) for r in st.row_ports])[None, :]
    got, _cut, okbits = st.low.evaluate(flat.astype(np.uint64))
    o0 = int(np.asarray(got)[0, 0])
    port = {p.name: p for p in m["ports"]}["o0"]
    bc = FC._commit_strings([port], {"o0": np.asarray([[o0]], dtype=np.uint16)}, {"o0": [m["salts"]["o0"][i]]})["o0"][0]
    return {"ok_output": o0, "satisfied": int(okbits) & 1 == 1, "opens_to_1": opens(port, [1], m["salts"]["o0"][i], bc)}


def _rebound(m: dict, canons: dict, i: int, row: str, words: list[int]) -> dict:
    """Instance i of member m with its row ``row`` committed to ``words`` under its own salt: refused when that ``b || c``
    isn't the one of the canonical commitment the row is bound to."""
    port = {p.name: p for p in m["ports"]}[row]
    bc = FC._commit_strings([port], {row: np.asarray([words], dtype=np.uint16)}, {row: [m["salts"][row][i]]})[row][0]
    c = bound_row(m["template"], row, i, m["args"][i], m["S"])
    return {"refused": bc != canons[c[0]]["bc"][c[1]], "row": row, "instance": i}


def rekey_negatives(members: list[dict], sd, layout, req, sm, canons: dict, rs) -> dict[str, dict]:
    """A re-key held by the map unit to the cap its committed pass row and lifetime cap L give: value v of block o, whose
    append joined window w ≥ 1, re-keyed at a step r whose first pass after is window w + age, by a forged instance of r
    with one block more (v at its slot, its epoch-1 layout entry, the new block written at r and freed at v's last structural reader).
    At L = age the unit accepts it; at L = age + 1 the re-key is one window early, and the unit refuses it. Two forged
    stale bounds at L = age + 1: o's write step forged below the cap in the map row (r's unit accepts, and o's writer,
    whose instance pins the step, refuses), and a pass row that dates r's window one later, window w + age's last pass
    at r (r's unit accepts; the row isn't the canonical one, and a canonical row committed to it isn't the session's
    passes, `bridge`)."""
    plan_m, map_m = members
    steps = req.steps
    passes = session_passes(steps)
    joined = [next((w for w, s in passes if s > x), None) for x in range(steps)]

    def room(r):
        a = map_m["args"][r]
        return a["count"] < sm.C and bin(a["present"]).count("1") < sm.K and a["first"] - sm.N + a["count"] < sm.M

    pick = next(((r, v) for r in range(steps) if joined[r] is not None and room(r) for v in sd.positions
                 if layout.block[v] >= sm.N and req.born[v] < r < req.last.get(v, -1) and joined[req.born[v]]
                 and joined[r] > joined[req.born[v]]), None)
    if pick is None:
        return {"re-key-one-window-early (at the cap)": {"refused": False, "why": "no step has room for a re-key"}}
    r, v = pick
    s, a = req.born[v], map_m["args"][r]
    age = joined[r] - joined[s]
    i, sl, k = sd.positions.index(v), layout.offset[v] // VW, bin(a["present"]).count("1")
    o, j, x = layout.block[v], a["first"] + a["count"], req.values[v] % (1 << VW)
    assert a["phi"][sm.N + 2 * (o - sm.N)] == s and a["lam"][2 * i:2 * i + 2] == [o, sl]

    def put(xs, y):
        return xs[:k] + [y] + xs[k + 1:]

    def rekey(life, phi, pi):
        lam, ph, pay = list(a["lam"]), list(phi), list(a["pay"])
        lam[2 * (sm.V + i)], lam[2 * (sm.V + i) + 1] = j, sl
        ph[sm.N + 2 * (j - sm.N)], ph[sm.N + 2 * (j - sm.N) + 1] = r, req.last[v]
        pay[a["count"]] = x << (sl * VW)
        return dict(a, count=a["count"] + 1, present=a["present"] | 1 << k, pos=put(a["pos"], i), ep=put(a["ep"], 1),
                    last=put(a["last"], req.last[v]), life=life, lam=lam, phi=ph, pi=list(pi), val=put(a["val"], x), pay=pay)

    pi = a["pi"]
    at_age, early = rekey(age, a["phi"], pi), rekey(age + 1, a["phi"], pi)
    forged = list(a["phi"])
    forged[sm.N + 2 * (o - sm.N)] = 0
    f_age, f_early = _forged(map_m, r, at_age), _forged(map_m, r, early)
    f_re, f_writer = _forged(map_m, r, rekey(age + 1, forged, pi)), _forged(map_m, s, dict(map_m["args"][s], phi=forged))
    where = {"step": r, "value": i, "old_block": o, "written_at": s, "new_block": j, "age_windows": age, "pass_row": pi,
             "caps": {f"L={L}": RU.rekey_cap(sm, pi, L, r) for L in (age, age + 1)}}
    out = {"re-key-one-window-early (at the cap)": {
               "refused": f_age["ok_output"] == 1 and f_early["ok_output"] == 0 and not f_early["opens_to_1"], **where,
               "at_L=age": f_age, "at_L=age+1": f_early,
               "reference": [RU.memory_map(sm, **at_age), RU.memory_map(sm, **early)]},
           "forged-stale: the old block's write step below the cap (refused at its writer)": {
               "refused": f_re["ok_output"] == 1 and f_writer["ok_output"] == 0 and not f_writer["opens_to_1"], **where,
               "re-key": f_re, "writer": f_writer}}
    name = "forged-stale: a pass row dating the re-key's window one later (binding, bridge)"
    if joined[r] + 1 >= sm.T:
        out[name] = {"refused": False, "why": f"no window after step {r}'s, {joined[r]}, below T = {sm.T}"}
        return out
    late = list(pi)
    late[joined[r]] = r
    f_late = _forged(map_m, r, rekey(age + 1, a["phi"], late))
    c = canons["passes"]
    bc = FC._commit_strings([c["port"]], {"passes": np.asarray([late], dtype=np.uint16)},
                            {"passes": [c["salts"][0]]})["passes"][0]
    br = bridge(rs, plan_m, map_m, steps, dict(c, bc=[bc]), sm)
    rb = _rebound(map_m, canons, r, "pi", late)
    out[name] = {"refused": f_late["ok_output"] == 1 and rb["refused"] and not br["ok"], **where, "forged_row": late,
                 "forged_cap": RU.rekey_cap(sm, late, age + 1, r), "unit": f_late, "binding": rb, "bridge": br["detail"]}
    return out


def negatives(a, out, rec, receipt, result, members, canons, sd, layout, store, plans, sp, sm, req, lean_draw, law, program, query,
              want, leaf_layer, publics, session_roots, served, n_all, common, salts, link, rrec, c3, rs0) -> dict[str, dict]:
    """Each re-decides one input altered. The clause negatives are a cheating prover consistent everywhere (its canonical
    commitments and its session agree with the altered witness), so only the unit can refuse it: its ok row commits 0.
    The appends negatives are a session the remote audit accepts whose appends aren't the map units' writes: the
    record's chain or a pass's head isn't the units' (`mapped`), and the ledger refuses the link. Their sessions' check 3
    is decided by its unit's reference (`opening.reference`), the run's own (``rs0``) by proofs. The stale negatives
    are map units whose lifetime cap isn't the record's (`mapped`, and the opening of their public rows) and a re-key the
    unit holds to the cap its committed pass row gives (`rekey_negatives`)."""
    neg: dict[str, dict] = {}
    plan_m, map_m = members
    req_steps = req.steps

    # the plan unit's clause: a wrong block in the plan. The unit asks for exactly the read blocks; the one-sided
    # `PlanMatches` allows the extra block as filler
    def plan_case(name: str, pick) -> None:
        for s in range(req_steps):
            extra = pick(s)
            if extra is None:
                continue
            bad = sorted(set(plans[s]) | {extra})
            assert RQ.plan_matches(req, layout, bad, RQ.BLOCK_BITS, s)
            args = dict(plan_m["args"][s], count=len(bad), plan=bad + [0] * (sp.P - len(bad)))
            f = _forged(plan_m, s, args)
            rs = remote_session(req, layout, store, plans, salts, extra={s: [extra]}, c3=c3)
            br = bridge(rs, plan_m, map_m, req_steps, canons["passes"], sm)
            neg[name] = {"refused": f["ok_output"] == 0 and f["satisfied"] and not f["opens_to_1"], "step": s, "block": extra,
                         "unit": f, "plan_matches": True, "honest_canon_bridge_refuses": not br["ok"]}
            return
        neg[name] = {"refused": False, "why": "no step has such a block"}

    def paused_block(s):
        resident = store.resident(s)
        for r, ps in sd.paused:
            if ps != s:
                continue
            for v, (rr, born) in sd.kv.items():
                if rr == r and born < s and req.last[v] > s and layout.block[v] in resident and layout.block[v] not in plans[s]:
                    return layout.block[v]
        return None

    def unselected_table_block(s):
        return next((TB + r // RPB for r in range(TABLE_ROWS) if TB + r // RPB not in plans[s]), None)

    plan_case("plan-holds-a-paused-request's-page (PlanMatches)", paused_block)
    plan_case("plan-holds-an-unselected-table-block (PlanMatches)", unselected_table_block)
    s0 = next(s for s in range(req_steps) if len(plans[s]) > 1)
    short = plans[s0][:-1]
    f = _forged(plan_m, s0, dict(plan_m["args"][s0], count=len(short), plan=short + [0] * (sp.P - len(short))))
    neg["plan-misses-a-block-the-step-reads (PlanMatches)"] = {"refused": f["ok_output"] == 0 and not f["opens_to_1"], "unit": f}

    # the memory-map unit's clauses: a payload mismatch, a late free, an early free, on a block whose values' last structural reader is
    # at least two steps after its write and before the last step, so both frees fall inside the run after the write
    def block_last(a_s, j):
        return max(a_s["last"][i] for i in range(sm.K) if (a_s["present"] >> i) & 1 and a_s["lam"][2 * a_s["pos"][i]] == j)

    t, j, lastj = next((s, j_, block_last(map_m["args"][s], j_)) for s in range(req_steps) for j_ in sorted(store.written.get(s, ()))
                       if s + 2 <= block_last(map_m["args"][s], j_) <= req_steps - 2)
    a_t = map_m["args"][t]
    pay = list(a_t["pay"])
    pay[j - a_t["first"]] ^= 1 << 1234
    f = _forged(map_m, t, dict(a_t, pay=pay))
    neg["payload-does-not-carry-its-value (Carries)"] = {"refused": f["ok_output"] == 0 and not f["opens_to_1"], "step": t, "block": j, "unit": f}
    for name, at in (("late-free (a block kept past its values' last structural reader)", lastj + 1),
                     ("early-free (a block freed before its values' last structural reader)", lastj - 1)):
        phi = list(a_t["phi"])
        phi[sm.N + 2 * (j - sm.N) + 1] = at
        f = _forged(map_m, t, dict(a_t, phi=phi))
        late_store = replace(store, freed={**{s: frozenset(b - {j}) for s, b in store.freed.items()},
                                           at: store.freed.get(at, frozenset()) - {j} | {j}})
        breaks = not all(RQ.store_matches(req, layout, late_store, RQ.BLOCK_BITS, s) for s in range(req_steps))
        neg[name] = {"refused": f["ok_output"] == 0 and not f["opens_to_1"], "step": t, "block": j, "free_at": at,
                     "honest_free": lastj, "store_matches_breaks": breaks, "unit": f}
    phi = list(a_t["phi"])
    phi[sm.N + 2 * (j - sm.N)] = t + 1
    f = _forged(map_m, t, dict(a_t, phi=phi))
    neg["write-at-another-step (L4)"] = {"refused": f["ok_output"] == 0 and not f["opens_to_1"], "unit": f}

    # the registered blocks' frees: the expert's weight block, whose values' last structural reader is mid-run, freed a step late or
    # a step early in the map row (and the store), at the step whose instance names it in rfree
    jx = layout.block[EXPERT]
    lx = req.last[EXPERT]
    a_x = map_m["args"][lx]
    assert RU._bits(a_x["rfree"], sm.N)[jx] and a_x["phi"][jx] == lx
    for name, at in (("registered-late-free (a registered block kept past its values' last structural reader)", lx + 1),
                     ("registered-early-free (a registered block freed before its values' last structural reader)", lx - 1)):
        phi = list(a_x["phi"])
        phi[jx] = at
        f = _forged(map_m, lx, dict(a_x, phi=phi))
        moved = replace(store, freed={**{s: frozenset(b - {jx}) for s, b in store.freed.items()},
                                      at: store.freed.get(at, frozenset()) - {jx} | {jx}})
        breaks = not all(RQ.store_matches(req, layout, moved, RQ.BLOCK_BITS, s) for s in range(req_steps))
        neg[name] = {"refused": f["ok_output"] == 0 and not f["opens_to_1"], "step": lx, "block": jx, "free_at": at,
                     "store_matches_breaks": breaks, "unit": f}

    # a written block holding no value the step produces (a dummy append, or a re-key of an earlier block's payload): the
    # instance of a step after block NR's write with one block more, its payload block NR's
    steps_w = sorted(s for s, js in store.written.items() if js)
    tz = next(s for s in range(steps_w[0] + 1, req_steps) if map_m["args"][s]["count"] < sm.C
              and map_m["args"][s]["first"] - sm.N + map_m["args"][s]["count"] < sm.M)
    a_z = map_m["args"][tz]
    pay = list(a_z["pay"])
    pay[a_z["count"]] = store.payload[NR]
    phi = list(a_z["phi"])
    b = sm.N + 2 * (a_z["first"] - sm.N + a_z["count"])
    phi[b], phi[b + 1] = tz, req_steps - 1
    f = _forged(map_m, tz, dict(a_z, count=a_z["count"] + 1, pay=pay, phi=phi))
    neg["written-block-without-a-value (dummy append or re-key)"] = {
        "refused": f["ok_output"] == 0 and not f["opens_to_1"], "step": tz, "unit": f}

    # the appends: sessions the remote audit accepts whose appends aren't the map units' writes
    def unmapped(name: str, **kw) -> None:
        rs = remote_session(req, layout, store, plans, salts, c3=c3, **kw)
        if not rs.accepted:
            neg[name] = {"refused": rs.by_check3 is not None, "by": "check 3" if rs.by_check3 else None,
                         "why": f"the session refused it: {rs.refused}"}
            return
        r = rs.session.record()
        try:
            audit.ledger(r, **common, requirement=link)
            why = None
        except ValueError as e:
            why = str(e)
        heads = all(h == link.heads[s] for w in r.windows for s, h in zip(w.steps, w.heads))
        neg[name] = {"refused": not link.mapped(r) and "memory-map unit" in (why or ""), "session_accepted": True,
                     "chain_is_the_units'": r.appends == link.appends, "heads_are_the_units'": heads, "ledger": why}

    t_last = steps_w[-1]
    t_two = next(s for s in steps_w if len(store.written[s]) > 1)
    j_live = next(j for s in steps_w if s < t_last for j in sorted(store.written[s])
                  if all(f > t_last for f, js in store.freed.items() if j in js))
    unmapped("append-of-another-payload (mapped)",
             alter={steps_w[0]: lambda pay, ys: ([bytes([pay[0][0] ^ 1]) + pay[0][1:]] + pay[1:], ys)})
    unmapped("append-out-of-the-map's-order (mapped)", alter={t_two: lambda pay, ys: (pay[1::-1] + pay[2:], ys[1::-1] + ys[2:])})
    unmapped("dummy-append (mapped)", dummy=t_last)
    unmapped("dropped-append (mapped)", drop=t_last)
    unmapped("re-key-not-the-map's (mapped)", rekey=(t_last, j_live))
    t_early = next(s for s in steps_w if s >= 1)
    unmapped("append-before-its-production-step (mapped)", early=t_early)
    neg["append-before-its-production-step (mapped)"]["step"] = t_early
    rs = remote_session(req, layout, store, plans, salts, early=t_early, c3=c3)
    br = bridge(rs, plan_m, map_m, req_steps, canons["passes"], sm)
    neg["append-before-its-production-step (bridge: heads)"] = {
        "refused": rs.accepted and not br["ok"], "step": t_early, "detail": br["detail"],
        "chain_is_the_units'": rs.accepted and rs.session.record().appends == link.appends}

    # the stale bound, which no public input carries: the map units' lifetime cap against the record's, and a re-key
    # held in the unit to the cap its committed pass row gives, one window early or with a forged stale bound
    loose = 1
    try:
        audit.ledger(rrec, **common, requirement=replace(link, life=loose))
        why = None
    except ValueError as e:
        why = str(e)
    neg["forged-stale: map units at another lifetime cap (mapped)"] = {
        "refused": not replace(link, life=loose).mapped(rrec) and "memory-map unit" in (why or ""), "life": loose,
        "record_lifetime": rrec.lifetime, "ledger": why}
    i_c = map_m["drawn"][-1] if map_m.get("drawn") else 0
    other_life = dict(map_m["args"][i_c], life=loose)
    port = {p.name: p for p in map_m["ports"]}["pub"]
    bcs = public_bcs(map_m["dir"] / f"pub-{map_m['n']}.bin", map_m["ports"])
    neg["forged-stale: a map unit's public row with another lifetime cap (opening)"] = {
        "refused": not opens(port, public_words(other_life, map_m["rows"]), map_m["salts"]["pub"][i_c], bcs[i_c]["pub"]),
        "instance": i_c, "life": loose}
    neg.update(rekey_negatives(members, sd, layout, req, sm, canons, rs0))
    try:
        audit.ledger(rrec, **common, requirement=replace(link, appends=hashlib.sha512(b"another chain").digest()))
        neg["ledger-on-a-link-whose-units-make-other-appends"] = {"refused": False}
    except ValueError as e:
        neg["ledger-on-a-link-whose-units-make-other-appends"] = {"refused": "memory-map unit" in str(e), "why": str(e)}
    try:
        audit.ledger(rrec, **common, requirement=replace(link, proved=False))
        neg["ledger-on-a-link-without-every-map-unit-proved"] = {"refused": False}
    except ValueError as e:
        neg["ledger-on-a-link-without-every-map-unit-proved"] = {"refused": "memory-map unit" in str(e), "why": str(e)}

    # bindings: a witness row other than its canonical commitment's
    other = list(plan_m["args"][1]["lam"])
    other[0], other[2] = other[2], other[0]
    neg["layout-row-not-the-shared-one (binding)"] = _rebound(plan_m, canons, 1, "lam", other)
    neg["map-unit's-layout-row-not-the-plan-unit's (binding)"] = _rebound(map_m, canons, 1, "lam", other)
    t2 = next(s for s in range(req_steps) if map_m["args"][s]["count"] > 1 and bin(map_m["args"][s]["present"]).count("1") > 1)
    a_2 = map_m["args"][t2]
    neg["payload-row-of-another-block (binding)"] = _rebound(map_m, canons, t2, "pay0",
                                                             RU._words(a_2["pay"][1], RU.BLOCK_WORDS))
    neg["value-row-of-another-value (binding)"] = _rebound(map_m, canons, t2, "val0", RU._words(a_2["val"][1], VW // 16))

    # the bridge and the link: a session plan larger than the certified one, a registered block released at another step
    rs = remote_session(req, layout, store, plans, salts, extra={s0: [unselected_table_block(s0)]}, c3=c3)
    br = bridge(rs, plan_m, map_m, req_steps, canons["passes"], sm)
    neg["session-plan-not-the-certified-one (bridge: count)"] = {"refused": not br["ok"], "detail": br["detail"]}
    jx, lx = layout.block[EXPERT], req.last[EXPERT]
    rs = remote_session(req, layout, store, plans, salts, late={jx: lx + 1}, c3=c3)
    br = bridge(rs, plan_m, map_m, req_steps, canons["passes"], sm)
    neg["session-registered-release-not-the-map's (bridge: rfree)"] = {"refused": not br["ok"], "block": jx, "detail": br["detail"]}
    try:
        audit.ledger(rrec, **common, requirement=UnitsLink(False, req, layout))
        neg["ledger-on-a-link-that-does-not-hold"] = {"refused": False}
    except ValueError as e:
        neg["ledger-on-a-link-that-does-not-hold"] = {"refused": "don't hold" in str(e), "why": str(e)}

    # the openings: a public row other than the verifier's copy, an ok row the proof covers altered (Lean)
    m0 = map_m
    i0 = m0["drawn"][0] if m0.get("drawn") else 0
    other_pub = dict(m0["args"][i0], last=[x + 1 for x in m0["args"][i0]["last"]])
    port = {p.name: p for p in m0["ports"]}["pub"]
    bcs = public_bcs(m0["dir"] / f"pub-{m0['n']}.bin", m0["ports"])
    neg["public-row-not-the-verifier's-copy (opening)"] = {
        "refused": not opens(port, public_words(other_pub, m0["rows"]), m0["salts"]["pub"][i0], bcs[i0]["pub"])}
    last = next((m for m in members if any((out / f"session-{m['name']}" / "sessions").glob("*/session.json"))), None)
    if last is not None:
        sdir = out / f"session-{last['name']}"

        def check(r, ll):
            layer = hashlib.sha512(canonical_json({**publics, last["name"]: ll})).hexdigest()
            return V.check(dict(r, leaf_layer=layer), program, query, law, a0.SCHEME, want, leaf_layer=layer)

        neg["public-file-altered-after-the-proof (Lean roots)"] = {**a0.altered_public(
            a.lean, last, last["n"], sdir, out / "neg-altered-public", dict(rec, leaf_layer=leaf_layer), check, zk=True),
            "member": last["name"], "row": f"instance {last['n'] - 1}, port {last['ports'][-1].name}"}
    # the sampled-proofs registration and draw, as A6 checks them
    fewer = R2.program([(plan_m["defn"], plan_m["n"]), (map_m["defn"], map_m["n"])], mixed=True)
    v = V.check(rec, fewer, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, program, query, D.parse_law(f"stratified:{a.k // 2 if a.k > 1 else a.k + 1}"), a0.SCHEME, want,
                leaf_layer=leaf_layer)
    neg["registration-law-another-k (R1)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, program, query, law, a0.SCHEME, {k_: w for k_, w in want.items() if not k_.startswith("canon/")}, leaf_layer=leaf_layer)
    neg["registration-without-the-canonical-commitments (R3)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check_draw({**lean_draw, "units": lean_draw["units"][:-1] + [n_all]}, rec, rec["law"])
    neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    neg.update(a0.audit_negatives(result, rec["law"], a.delta, session_roots, served=sorted(served)))
    return neg


if __name__ == "__main__":
    raise SystemExit(main())
