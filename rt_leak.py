"""REDTEAM probe -- Item 7 (#3984, leak_scan): the shared received-pieces leak scan.

Two deliverables the lead asked for:

  1. Review the module for a leak that is MISSED (a find that should be `leaked` but is excused).
  2. Assess the residual risk the docstring names -- a leak written at a byte shift ACROSS fields -- and
     whether it is realistic for C-Flock's writer.

FINDING (this probe): `reads` builds its nonce set from layout keys ending in "nonce", but Ligerito's
grinding nonces are keyed `lig.qn`, `lig.cbn`, `lig.consn` (consumed by transcript_check's `t.pow`, and
redigest's `lig-query-nonce`/`lig-claim-nonce` mutants flip them and the verifier still ACCEPTs -- so the
prover controls them). They are therefore classified as reader-fixed integers, and `public_bytes` puts their
bytes in a "public" span. A prover can write a 16-byte witness value across two adjacent such nonces to
manufacture a public alibi that excuses the SAME bytes leaked in one element field elsewhere -- turning the
rule's headline `leaked` verdict (a switch row in one field, REDTEAM phase 16) into `structured: public`.

ASSESSMENT (across-fields residual): the reader tiles the stream with no gaps, so "across" means spanning a
field boundary. C-Flock's writer emits each field as one whole value, so a value that genuinely leaks lands
inside one field (caught); an "across" find is a coincidental alignment of two independent fields and carries
no information. Deliberately splitting a secret across a salt/field boundary needs a MODIFIED prover, outside
this scan's threat model. The residual is correctly documented and not realistic for the honest writer --
EXCEPT the grinding nonces above, which the scan actively mislabels as public (the finding).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path("benchmarks/network_traces").resolve()))
import leak_scan as LS  # noqa: E402

RD = LS.RD  # redigest, with its layout reader

U = lambda v: struct.pack("<Q", v)          # noqa: E731  a u64 field (8 bytes)
U32 = lambda v: struct.pack("<I", v)        # noqa: E731
F = b"\xa0" * 16                             # an f128 element (16 bytes)
H = b"\xb0" * 32                             # a 32-byte hash (Sha256 kind)
SW = b"".join(x.to_bytes(2, "little") for x in (0, 1, 2, 3, 2, 1, 0, 3))  # the routed tie's switch row (16 B)


def build_zk_stream(qn=2, cbn=2, consn=2, fa=F):
    """A minimal valid `--zk` proof stream (Sha256 Merkle kind, unsalted) in redigest.layout's read order, with
    `qn`/`cbn`/`consn` grinding-nonce vectors populated and `fa` the zc.fa element. Returns the bytes."""
    p = []
    p += [U(0)]                       # cap: vec(h32), count 0
    p += [U(0), U(0), U(0), U32(0)]   # m, k0, ..., u32
    p += [b"\x00"]                    # t flag u8 = 0 (no extra u64)
    p += [U32(0)]                     # k = 0 -> Sha256, hlen 32, salt_bytes 0
    p += [U(0), U(0), U(0)]           # zc.ab, zc.c, zc rounds -- all empty vecs
    p += [fa, F, F]                   # zc.fa, zc.fb, zc.fc
    p += [U(0)]                       # zc.nonce vec, count 0
    p += [U(0), U(0), U(0)]           # lc rounds, lc.zp, lc.me -- empty
    p += [U(0)]                       # lc.nonce vec 0
    p += [U(0)]                       # rs: for k in range(u64()) -> count 0
    p += [U(0)]                       # batching.nonce vec 0
    p += [U(0)]                       # lig.icap vec 0
    p += [U(0), U(0)]                 # level l0: vec(vec f) 0 ; vec h 0
    p += [U(0)]                       # rcaps: for i in range(u64()) -> 0
    p += [U(0)]                       # l{i+1}: for i in range(u64()) -> 0
    p += [U(0)]                       # lig.yr vec 0
    p += [U(0), U(0)]                 # level final: vec(vec f) 0 ; vec h 0
    p += [U(0)]                       # lig.sc vec 0
    p += [U(0)]                       # lig.msg vec 0
    p += [U(qn)] + [U(0)] * qn        # lig.qn: vec(u), count qn  <-- grinding nonces
    p += [U(0)]                       # lig.ood vec(f) 0
    p += [U(0)]                       # lig.fold vec(u) 0
    p += [U(cbn)] + [U(0)] * cbn      # lig.cbn: vec(u), count cbn <-- grinding nonces
    p += [U(consn)] + [U(0)] * consn  # lig.consn: vec(u), count consn <-- grinding nonces
    # zk tail
    p += [b"\x01"]                    # zk.tag (1 byte)
    p += [b"\xe0" * 32] * 3           # zk.e0, zk.e1, zk.tp
    p += [U(0)]                       # zk.ybar vec 0
    p += [U(0)]                       # in.root: at(len=u64()) with len 0  (reads the u64, then 0 bytes)
    p += [F, F]                       # in.rho, in.sigma
    p += [U(0), U(0)]                 # in.tauc len 0, in.taus len 0
    p += [F]                          # in.tau
    p += [U(0)]                       # in.y vec 0
    p += [U(0)]                       # in.open vec 0
    p += [U(0)]                       # in.salt vec 0
    p += [U(0)]                       # in.path vec 0
    return b"".join(p)


def banner(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


def main():
    stream = build_zk_stream()
    # 1) The stream parses, and reads tile it with no gaps.
    got = LS.reads(stream)
    assert got is not None, "layout did not read the stream whole"
    fields, nonces = got
    covered = []
    at = 0
    for o, n in fields:
        assert o == at, f"gap/overlap at {o} (expected {at})"
        at = o + n
    assert at == len(stream), (at, len(stream))
    print(f"stream {len(stream)} bytes, {len(fields)} fields, tiled with no gaps; nonce offsets {sorted(nonces)}")

    off = RD.layout(stream, zk=True)
    banner("FINDING: the Ligerito grinding nonces are classified as reader-fixed integers")
    for k in ("lig.qn", "lig.cbn", "lig.consn", "zc.nonce", "lc.nonce"):
        os = off.get(k, [])
        inn = all(o in nonces for o in os) if os else None
        tag = "nonce-set" if inn else ("NOT in nonce-set (treated as a public integer)" if os else "(empty here)")
        print(f"  layout key {k:<12} offsets {os}  ->  {tag}")
    missed = [k for k in ("lig.qn", "lig.cbn", "lig.consn") if off.get(k) and not all(o in nonces for o in off[k])]
    print(f"  grinding nonces the scan misses: {missed}")

    # 2) public_bytes consequently exposes the grinding-nonce run as 'statement-fixed'.
    banner("the grinding-nonce bytes become a 'public' span (public_bytes)")
    # write a recognisable 16-byte value across the two qn nonces, then read the public spans.
    qn0 = off["lig.qn"][0]
    probe = bytearray(stream)
    probe[qn0:qn0 + 16] = SW                      # SW spans qn[0] and qn[1] (two 8-byte nonce fields)
    probe = bytes(probe)
    assert LS.reads(probe) is not None, "edited stream still parses"
    vdir = Path("/tmp/rt/_leakprobe"); vdir.mkdir(exist_ok=True)
    (vdir / "s-circuit.rep0.bin").write_bytes(probe)
    (vdir / "s-session.json").write_text("{}")
    pub = LS.public_bytes({"s-circuit.rep0.bin": 1, "s-session.json": 1}, vdir)
    in_pub = any(SW in p for p in pub)
    print(f"  SW written across qn[0]/qn[1]; is SW inside a 'public' span? {in_pub}  (True = mislabelled public)")

    # 3) End-to-end EVASION: SW leaked in one prover-controlled hiding field (zk.e0, a ZK mask) + a qn
    #    'public alibi' -> excused. The leak field is immaterial to the bug (the rule must catch SW in ANY
    #    field); zk.e0 is used so the leak sits in a field the prover genuinely controls.
    banner("EVASION: a phase-16 switch-row leak in one field, excused by a grinding-nonce alibi")
    leaked_stream = bytearray(build_zk_stream())
    off2 = RD.layout(bytes(leaked_stream), zk=True)
    e0 = off2["zk.e0"][0]
    leaked_stream[e0:e0 + 16] = SW                         # SW in zk.e0, a genuine in-one-field hiding-mask leak
    qa = off2["lig.qn"][0]
    leaked_stream[qa:qa + 16] = SW                         # SW also across the two qn nonces (the alibi)
    leaked_stream = bytes(leaked_stream)
    (vdir / "s-circuit.rep0.bin").write_bytes(leaked_stream)
    pieces = {"sw_route 0 @0": SW}
    received = _received(vdir, pieces)
    placed = LS.placements(received, pieces, vdir)
    leaked, structured = LS.classify(received, pieces, vdir)
    print(f"  SW placements: {placed['sw_route 0 @0']}")
    print(f"  classify -> leaked={leaked}  structured={structured}")
    bug = (leaked == [] and structured.get("sw_route 0 @0") == "public")
    print(f"  the in-field switch-row leak is excused as 'public'? {bug}  <<< FINDING" if bug
          else f"  (not reproduced: leaked={leaked})")

    banner("the fix: count the grinding nonces as nonces")
    leaked_fix, structured_fix = LS.classify(received, pieces, vdir, reads=_reads_fixed)
    print(f"  with lig.qn/cbn/consn in the nonce set: leaked={leaked_fix} structured={structured_fix}")
    print(f"  -> the leak is caught: {leaked_fix == ['sw_route 0 @0']}")

    banner("ASSESSMENT: across-fields residual for the honest writer")
    print("  Reads tile the stream with no gaps, so 'across' = spanning a field boundary. C-Flock writes each")
    print("  field as one whole value, so a value that genuinely leaks lands inside one field (caught); an")
    print("  'across' find is a coincidental two-field alignment and carries no information. Splitting a secret")
    print("  across adjacent prover-free fields (salts) needs a MODIFIED prover, outside this scan's threat")
    print("  model. The residual is correctly documented -- EXCEPT the grinding nonces, which the scan actively")
    print("  mislabels as public (the finding above); that is a real false-negative, not a documented residual.")


def _received(vdir: Path, pieces: dict) -> dict:
    """`a8.received`-shaped dict built from what is actually in vdir (raw or either hex case)."""
    def has(data, b):
        return any(x in data for x in (b, b.hex().encode(), b.hex().upper().encode()))
    files = {f.name: {"bytes": f.stat().st_size, "found": {k: 1 for k, b in pieces.items() if has(f.read_bytes(), b)}}
             for f in sorted(vdir.iterdir())}
    return {"files": files, "verifier_found": sorted({k for x in files.values() for k in x["found"]})}


def _reads_fixed(stream, _cache={}):
    """`reads` with the one-line fix: the grinding nonces lig.qn/cbn/consn join the nonce set."""
    if stream in _cache:
        return _cache[stream]
    base = LS.reads(stream)
    if base is None:
        _cache[stream] = None
        return None
    fields, nonces = base
    off = RD.layout(stream, zk=True)
    extra = {o for k in ("lig.qn", "lig.cbn", "lig.consn") for o in off.get(k, [])}
    _cache[stream] = (fields, nonces | extra)
    return _cache[stream]


if __name__ == "__main__":
    main()
