"""REDTEAM phase 19 target 1: #3995 (IO-TIE @9cae989457), the Commit's port rows as S registered reads.

Can a prover feed IoTieBucket / IoTieRequestIds a port row that isn't the committed one (a wrong step, a wrong bucket,
a row from another window or run, a row whose registration names a different Commit)? Is what each registered read
binds (b || c to its leaf) enough without the in-circuit hash?

Every read is judged twice: by the PR's host emulation (`warden_rows.refusal`, which climbs under the `dom` it is
handed) and by `lean_refusal`, which reads only the verifier's port entry (its root, domain id, leaves, reads and
reads_from) as `Flock.Registered.checkPort` does (`opensLeaf`: pos < leaves, |path| = 64 log2ceil(leaves), climb under
`reg.domain` to `reg.root`; then each instance's position is the port's read for its unit)."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace

sys.path.insert(0, "verity/core/protocols/certifier/tests")

import test_network_certifier_io_ids as TI  # noqa: E402
import test_network_certifier_io_tie as TT  # noqa: E402

from verity.ml.commitments import advice as ADV, hm96  # noqa: E402
from verity.ml.commitments.gates.frame_v3 import leaf_ref, node_ref  # noqa: E402
from verity.core.protocols.certifier import io_ids as I, io_tie as IT, warden_rows as WR  # noqa: E402

INST = hm96.SHA512


def log2ceil(n: int) -> int:
    return (n - 1).bit_length()


def lean_opens(port: dict, pos: int, bc: bytes, path: bytes) -> bool:
    """`Flock.Registered.opens reg bc pos path`: the port's own domain, leaves and root, nothing else."""
    dom, root, leaves = bytes.fromhex(port["domain"]), bytes.fromhex(port["root"]), port["leaves"]
    depth = log2ceil(leaves)
    if not (pos < leaves and len(path) == 64 * depth):
        return False
    cur, idx = leaf_ref(dom, port["row_schema"], INST.tree_leaf(INST.default_key, bc), pos), pos
    for d in range(depth):
        sib = path[64 * d:64 * d + 64]
        cur = node_ref(dom, d, idx // 2, *((cur, sib) if idx % 2 == 0 else (sib, cur)))
        idx //= 2
    return cur == root


def lean_refusal(port: dict, entry: dict, bcs: list[bytes]) -> str | None:
    """`checkPort`'s order on a statement whose instance i is unit reads_from + i and reads table row i."""
    if entry["value"] != port["value"]:
        return "value"
    pos, paths = entry["positions"], [bytes.fromhex(p) for p in entry["paths"]]
    for r, (p, b, pa) in enumerate(zip(pos, bcs, paths, strict=True)):
        if not lean_opens(port, p, b, pa):
            return f"row {r} doesn't open at {p}"
    reads = port["reads"]
    for i, p in enumerate(pos):
        if i >= len(reads) or reads[i] != p:
            return f"unit {port.get('reads_from', 0) + i} reads {p}, the verifier {reads[i] if i < len(reads) else None}"
    return None


def key(S: int, s: int) -> str:
    return IT.PORT if S == 1 else f"{IT.PORT}[{s}]"


# ------------------------------------------------------------------------------------------------------------------
# Egress (IoTieBucket): BIG's busiest window, the Commit registered under the binding H("port" + "w")
# ------------------------------------------------------------------------------------------------------------------
print("== egress: IoTieBucket's S port reads ==")
SH = TT.BIG
LINK = TT.busy(1)
W0 = TT.busiest(LINK)
HON = TT.commit(SH, LINK, W0)
COUNT = len(LINK.port)
BINDING = TT._h(b"portw")
SALTS = [IT.fake_salt(b"wp", s) for s in range(COUNT)]
REG = ADV.register(LINK.port, 32 * SH.P, BINDING, SALTS)
REG_DOM = IT.domain(BINDING, COUNT)
LEVELS = IT.rows_levels(REG_DOM.domain_id, LINK.port, SALTS)
assert REG.root == HON.port_root == LEVELS[-1][0] and REG_DOM.domain_id == HON.port_domain
print(f"window {W0}: T={SH.T} buckets, S={SH.S} reads a bucket, Commit of {COUNT} steps; registration root "
      f"{REG.root.hex()[:16]}… = the instance's: True")

#: the verifier's own steps, from the advice (window_map) and nothing the prover sends
V_STEPS = IT.window_map(SH, LINK.outputs, LINK.release, W0, LINK.admitted, LINK.params.delay, LINK.params.slots,
                        LINK.order)[1]
assert V_STEPS == HON.steps


def verifier_ports(steps=V_STEPS, root=REG.root, dom=REG_DOM.domain_id, count=COUNT):
    return IT.port_reads(steps, root, dom, count, SH.P)


def judge(presented: IT.Instance, ports=None) -> tuple[list, list]:
    """Each port row s's verdicts on the presented window: (host emulation, Lean-faithful)."""
    ports = verifier_ports() if ports is None else ports
    host, lean = [], []
    for s in range(SH.S):
        entry, bcs = TT.port_statement(presented, s)
        host.append(WR.refusal(ports[key(SH.S, s)], REG_DOM, entry, bcs))
        lean.append(lean_refusal(ports[key(SH.S, s)], entry, bcs))
    return host, lean


def opened_at(inst: IT.Instance, t: int, s: int, step: int, claim: int, levels=LEVELS, rows=LINK.port,
              salts=SALTS) -> IT.Instance:
    """Bucket t's s-th read replaced by the Commit's honest opening of `step` (row, salt, path), claimed at `claim`."""
    def put(xs, v):
        return TT._set(xs, t, tuple(TT._set(xs[t], s, v)))
    return replace(inst, port=put(inst.port, rows[step]), port_salts=put(inst.port_salts, salts[step]),
                   paths=put(inst.paths, IT.rows_path(levels, step)), steps=put(inst.steps, claim))


h, l = judge(HON)
print(f"(0) honest window: host {h}, lean {l}")
assert h == l == [None] * SH.S

# (1) a wrong step: every other step's honest opening, claimed at its own step or at the verifier's
tries = acc = 0
for t in range(SH.T):
    for s in range(SH.S):
        for a in range(COUNT):
            if a == V_STEPS[t][s]:
                continue
            for claim in (a, V_STEPS[t][s]):
                bad = opened_at(HON, t, s, a, claim)
                h, l = judge(bad)
                tries += 1
                acc += h[s] is None or l[s] is None
print(f"(1) a wrong step (each of the {COUNT - 1} other steps' honest openings, at its own position or the "
      f"verifier's, every bucket and read): {tries} tries, {acc} accepted")
assert acc == 0

# (2) a wrong bucket: unit t presents bucket t2's S reads (rows, salts, paths and positions)
tries = acc = same = 0
for t in range(SH.T):
    for t2 in range(SH.T):
        if t2 == t:
            continue
        bad = replace(HON, port=TT._set(HON.port, t, HON.port[t2]), port_salts=TT._set(HON.port_salts, t, HON.port_salts[t2]),
                      paths=TT._set(HON.paths, t, HON.paths[t2]), steps=TT._set(HON.steps, t, HON.steps[t2]))
        h, l = judge(bad)
        for s in range(SH.S):
            tries += 1
            if V_STEPS[t][s] == V_STEPS[t2][s]:
                same += 1
                assert h[s] is None and l[s] is None
            else:
                acc += h[s] is None or l[s] is None
print(f"(2) a wrong bucket (unit t reads bucket t2's rows): {tries} reads, {same} are the same step (the same row), "
      f"{acc} of the rest accepted")
assert acc == 0

# (3) another window of the same run: window w2's openings presented for window W0's units
tries = acc = same = 0
for w2 in range(len(LINK.rows) // SH.T):
    if w2 == W0:
        continue
    other = TT.commit(SH, LINK, w2)
    h, l = judge(replace(HON, port=other.port, port_salts=other.port_salts, paths=other.paths, steps=other.steps))
    for s in range(SH.S):
        tries += 1
        if [st[s] for st in other.steps] == [st[s] for st in V_STEPS]:
            same += 1
        else:
            acc += h[s] is None or l[s] is None
print(f"(3) another window's reads (5 windows x S port rows): {tries}, {same} identical, {acc} of the rest accepted")
assert acc == 0

# (4) another run: the same rows and steps, under another run's Commit (its binding and salts)
run2 = TT.commit(SH, LINK, W0, seed=b"run2")
assert run2.port == HON.port and run2.steps == HON.steps and run2.port_root != REG.root
h, l = judge(run2)
print(f"(4) another run's Commit of the same rows at the same steps, under this run's registration: host "
      f"{[x is not None for x in h]}, lean {[x is not None for x in l]} (refused)")
assert all(h) and all(l)
h, l = judge(run2, verifier_ports(root=run2.port_root, dom=run2.port_domain))
print(f"    the same, with the verifier's port naming run 2's root and domain: host {h} (the emulation climbs "
      f"under the dom it's handed), lean {l}")
assert all(h) and l == [None] * SH.S

# (5) b || c: one bit of a row, the salt, a sibling, another value's row (the warden's cells) at the position
tries = acc = 0
for t in range(SH.T):
    for s in range(SH.S):
        row = bytearray(HON.port[t][s])
        for bit in (0, 7, 8 * len(row) - 1, 8 * 4 * (SH.P - 1)):
            r = bytearray(row)
            r[bit // 8] ^= 1 << (bit % 8)
            for bad in (replace(HON, port=TT._set(HON.port, t, tuple(TT._set(HON.port[t], s, bytes(r))))),):
                h_, l_ = judge(bad)
                tries += 1
                acc += h_[s] is None or l_[s] is None
        bad = replace(HON, port_salts=TT._set(HON.port_salts, t, tuple(TT._set(HON.port_salts[t], s, IT.fake_salt(b"x", s)))))
        h_, l_ = judge(bad)
        tries += 1
        acc += h_[s] is None or l_[s] is None
print(f"(5) a row bit, a salt: {tries} tries, {acc} accepted")
assert acc == 0
# hm96 binds the row v2 digest (row bits and their count) and the salt into b || c, and the leaf binds b || c to
# (domain id, position, schema): the leaf the PR's tree builds (`leaf_ref(dom, SCHEMA, hiding_value(r, s), i)`) is the
# one the verifier climbs from b || c, for every row of the Commit
for i in range(COUNT):
    assert IT.HC.hiding_value(LINK.port[i], SALTS[i]) == INST.tree_leaf(INST.default_key, WR.bc(LINK.port[i], SALTS[i]))
print(f"    hiding_value(row, salt) == tree_leaf(b || c) on all {COUNT} Commit rows: True")

# ------------------------------------------------------------------------------------------------------------------
# Ingress (IoTieRequestIds): the Commit's c00.input/ids rows, read through io_ids.port_reads
# ------------------------------------------------------------------------------------------------------------------
print("\n== ingress: IoTieRequestIds' S reads of c00.input/ids ==")
steps = TI._steps(TI.ACROSS)
user = TI._tokens(TI._prompts(TI.ACROSS))
computed = [list(x) for x in user]
computed[2][5] ^= 1                      # REDTEAM phase 9 #1: the forward computes request 2's position 5 on another id
REAL = I.commit_rows(steps, computed, TI.P)
r_salts, r_levels, r_dom = I.commit(REAL)
REAL_ROOT = r_levels[-1][0]
assert ADV.register(REAL, 32 * TI.P, hashlib.sha512(b"commit").digest(), r_salts).root == REAL_ROOT
COUNT_I = len(REAL)
honest_real = TI._link(TI.ACROSS, tokens=computed)
print(f"the registered Commit (the forward's, request 2 position 5 substituted): root {REAL_ROOT.hex()[:16]}…; "
      f"its own buckets' verdicts {TI._verdicts(honest_real)}")

# the prover's second tree: the user's ids, under the same (public) domain; the instances it hands the verifier
forged = TI._link(TI.ACROSS)
F_ROOT = forged[0].port_root
assert F_ROOT != REAL_ROOT and forged[0].port_domain == r_dom
print(f"the prover's forged tree of the user's ids under the same domain: root {F_ROOT.hex()[:16]}…; every unit "
      f"{[I.unit_reference(TI.SHAPE, i) for i in forged]}")


def ingress_verdicts(ports, insts) -> list:
    dom = IT.domain(hashlib.sha512(b"commit").digest(), COUNT_I)
    out = []
    for s in range(TI.SHAPE.S):
        entry = {"value": I.INPUT_IDS, "positions": [i.map.steps[s] for i in insts],
                 "paths": [b"".join(i.paths[s]).hex() for i in insts]}
        bcs = [WR.bc(i.port[s], i.port_salts[s]) for i in insts]
        out.append((WR.refusal(ports[key(TI.SHAPE.S, s)], dom, entry, bcs), lean_refusal(ports[key(TI.SHAPE.S, s)], entry, bcs)))
    return out


via_helper = I.port_reads(forged, COUNT_I, TI.P)
print(f"(6) io_ids.port_reads(instances, count, P), the verifier's own ports, names root "
      f"{bytes.fromhex(via_helper[key(TI.SHAPE.S, 0)]['root']).hex()[:16]}… (the forged tree's: "
      f"{via_helper[key(TI.SHAPE.S, 0)]['root'] == F_ROOT.hex()})")
v = ingress_verdicts(via_helper, forged)
print(f"    the forged instances' reads under it: host {[a for a, _ in v]}, lean {[b for _, b in v]}")
whole = all(a is None and b is None for a, b in v) and all(I.unit_reference(TI.SHAPE, i) == 1 for i in forged)
print(f"    the tie of every bucket accepted (reads and units) while the forward computed another prompt: {whole}")
assert whole
held = IT.port_reads([i.map.steps for i in forged], REAL_ROOT, r_dom, COUNT_I, TI.P, I.INPUT_IDS)
v = ingress_verdicts(held, forged)
print(f"    the same instances under the registration's root (io_tie.port_reads' form): "
      f"{sum(a is not None for a, _ in v)} of {TI.SHAPE.S} port rows refused (host), "
      f"{sum(b is not None for _, b in v)} (lean)")
assert all(a is not None and b is not None for a, b in v)

# (7) the host emulation reads neither the port's domain nor its leaves; Lean reads both
p0 = dict(verifier_ports()[key(SH.S, 0)])
entry, bcs = TT.port_statement(HON, 0)
wrong = p0 | {"domain": TT._h(b"another domain").hex()}
print(f"\n(7) a verifier port naming another domain: host {WR.refusal(wrong, REG_DOM, entry, bcs)!r}, "
      f"lean {lean_refusal(wrong, entry, bcs)!r}")
wrong = p0 | {"leaves": 2 * COUNT}
print(f"    naming twice the leaves: host {WR.refusal(wrong, REG_DOM, entry, bcs)!r}, "
      f"lean {lean_refusal(wrong, entry, bcs)!r}")
print("\nall assertions hold")
