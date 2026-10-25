"""REDTEAM probe -- Target 6 (#3881, #3912, #3915): the certifier's signed clock certificates.

Lead's questions:
  (a) Can a clock certificate be replayed into another window or direction (or link, session, key)?
  (b) Can a record certificate be passed as a clock one (and the reverse)?
  (c) CERT-VARLEN's difftest: a clock certificate re-signed for another window verifies as that window's. A problem?
  (d) Can #3915's driver (`n2_advised_zk.served_clocks`) be made to take the unsigned `CertifierClocks` stand-in when
      the run has a signed ``clocks.jsonl``?

Everything here runs the PRs' own code: `certificate.Issuer` / `ClockCertificate` / `CertifiedRecord`,
`advised.SignedClocks`, `service.LinkWindows.certified`, and the driver's `served_clocks` / `clocks_label`.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import secrets
import sys
import tempfile
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "verity/core/protocols/certifier/tests"), str(ROOT / "benchmarks/network_traces")]

import signers  # noqa: E402
from advised_traffic import LINK, P, PIN, SYNC_SET, SYNCS, serve  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity.core.protocols.certifier import advised as AD  # noqa: E402
from verity.core.protocols.certifier.certificate import (CertifiedRecord, ClockCertificate, Direction, Issuer,  # noqa: E402
                                                         certificate_digest, clock_digest, parameters_digest)
from verity.core.protocols.certifier.schedule import ClockSync  # noqa: E402
from verity.core.protocols.certifier.service import LinkWindows  # noqa: E402

S0, S1 = signers.SESSION, signers.OTHER_SESSION
fails: list[str] = []


def banner(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def expect(name: str, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'XX'}] {name}: {got!r}" + ("" if ok else f"   (expected {want!r})"))
    if not ok:
        fails.append(name)


def q_of(cert) -> int:
    return lms.Signature.decode(signers.SMALL, cert.signature).q


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="rt-clock-"))
    for d in ("own", "stranger"):
        (tmp / d).mkdir()
    issuer = Issuer(signers.signer(tmp / "own"), S0)
    key = issuer.signer.public_key()
    W = len(SYNCS)
    sess = serve(1)
    pdig = parameters_digest(P, SYNC_SET)

    # The certifier's own order per egress window: the record, then its clock (CertifyingPublisher.keep).
    records, clocks = [], []
    for w in range(W):
        records.append(issuer.certify(sess.egress[w], Direction.EGRESS, P, pdig))
        clocks.append(issuer.clock(LINK, w, SYNCS[w]))
    ingress = [issuer.certify(sess.ingress[w], Direction.INGRESS, PIN, parameters_digest(PIN)) for w in range(W)]

    def syncs(certs, session=S0, link=LINK):
        return AD.SignedClocks(key, session, certs).syncs(link)

    banner("(0) honest")
    expect("every clock certificate verifies", [c.verify(key, S0) for c in clocks], [True] * W)
    expect("SignedClocks gives the signed syncs", syncs(clocks), tuple(SYNCS))

    banner("(a) replay into another window / direction / link / session / key, without the key")
    relabel = [(w, v) for w in range(W) for v in range(W + 2) if v != w]
    moved = [replace(clocks[w], certificate=replace(clocks[w].certificate, window=v)) for w, v in relabel]
    expect(f"window relabelled ({len(relabel)} (w, w') pairs) verifies", sum(c.verify(key, S0) for c in moved), 0)
    for w in range(W):
        others = [c for c in clocks if c.window != w]
        stand = [m for (a, v), m in zip(relabel, moved) if v == w]
        expect(f"window {w} filled by another window's relabelled certificate: syncs",
               syncs(others + stand), tuple(SYNCS[:w]))
    flipped = [replace(c, certificate=replace(c.certificate, direction=Direction.INGRESS)) for c in clocks]
    expect("direction relabelled to ingress verifies", sum(c.verify(key, S0) for c in flipped), 0)
    resigned_in = [ClockCertificate(SYNCS[w], issuer.sign(LINK, Direction.INGRESS, w, clock_digest(SYNCS[w])))
                   for w in range(W)]
    expect("signed by the key holder for ingress (same digest) verifies as a clock", sum(
        c.verify(key, S0) for c in resigned_in), 0)
    other_link = [replace(c, certificate=replace(c.certificate, link="requests")) for c in clocks]
    expect("link relabelled verifies", sum(c.verify(key, S0) for c in other_link), 0)
    expect("honest certificates checked in another session", sum(c.verify(key, S1) for c in clocks), 0)
    stranger = Issuer(signers.signer(tmp / "stranger", seed=b"stranger"), S0)
    expect("a stranger key's (the prover's own) certificates: syncs",
           syncs([stranger.clock(LINK, w, s) for w, s in enumerate(SYNCS)]), ())
    swapped = [ClockCertificate(SYNCS[(w + 1) % W], clocks[w].certificate) for w in range(W)]
    expect("sync swapped under a signed certificate verifies", sum(c.verify(key, S0) for c in swapped), 0)

    banner("(b) a record certificate as a clock one, and a clock certificate as a record one")
    cand = sorted({ClockSync(d, r) for d, r in product(SYNC_SET.deltas, SYNC_SET.rhos)} | set(SYNCS),
                  key=lambda s: (s.delta, s.rho))
    as_clock = [ClockCertificate(s, r.certificate) for r in records + ingress for s in cand]
    expect(f"record certificate under any of {len(cand)} syncs, as a clock ({len(as_clock)} tries)",
           sum(c.verify(key, S0) for c in as_clock), 0)
    expect("record certificates in SignedClocks: syncs", syncs(as_clock), ())
    as_record = [CertifiedRecord(r.head, r.salt, c.certificate) for r, c in zip(records, clocks)]
    expect("clock certificate under its window's record head and salt, as a record", sum(
        x.verify(key, S0) for x in as_record), 0)
    salts = [secrets.token_bytes(64) for _ in range(2000)]
    hit = sum(certificate_digest(s, records[0].head.digest()) == clocks[0].certificate.digest for s in salts)
    expect("2000 prover-chosen salts making the record digest the clock digest", hit, 0)
    lw = LinkWindows(LINK, P, SYNC_SET, key, "rt-clock", S0, ingress=PIN)
    host = [lw.certified(w, [clocks[w].certificate, ingress[w].certificate]) for w in range(W)]
    print("  boundary: the host's certificates check (`LinkWindows.certified`) reads signatures and fields alone, so a")
    print("  clock certificate's `Certificate` passes it as the egress record certificate of its window:")
    expect("LinkWindows.certified(w, [clock cert, ingress record cert])", host, [True] * W)
    print("  -> what refuses it is the proved head check: the egress head's salted digest must equal the clock digest,")
    print("     a SHA-512 second preimage (146-byte record input vs a 93-byte tagged input under another prefix).")

    banner("(c) CERT-VARLEN's case: a clock certificate signed again for another window")
    resigned = [ClockCertificate(SYNCS[w], issuer.sign(LINK, Direction.EGRESS, w ^ 1, clock_digest(SYNCS[w])))
                for w in range(W)]
    expect("re-signed for w^1 verifies (as w^1's)", [c.verify(key, S0) and c.window == (w ^ 1)
                                                      for w, c in enumerate(resigned)], [True] * W)
    expect("beside window 1's honest certificate it gives window 1 two syncs: refused from window 1 on", syncs(
        clocks + resigned[:1]), tuple(SYNCS[:1]))
    same = [clocks[1], ClockCertificate(SYNCS[1], issuer.sign(LINK, Direction.EGRESS, 0, clock_digest(SYNCS[1])))]
    expect("signed for window 0 with window 1's sync, no window-0 certificate otherwise: window 0's sync",
           syncs(same + clocks[2:])[:1], (SYNCS[1],))
    env = Issuer(signers.signer(tmp / "env", seed=b"env") if (tmp / "env").mkdir() is None else None, S0)
    env_clocks = [env.clock(LINK, w, SYNCS[0]) for w in range(W)]
    expect("an envelope certifier (one sync every window): its windows' clock digests are one digest",
           len({c.certificate.digest for c in env_clocks}), 1)
    print("  -> under the envelope policy the honest certifier signs exactly the same digest for every window; the")
    print("     window is only in the signed message. A certificate 'signed again for another window' is what the")
    print("     certifier issues for that window. Producing one takes the private key (the stranger key's are refused,")
    print("     the relabelled ones are refused), and a certifier that signs two syncs for a window has that window")
    print("     refused, not resolved.")

    banner("(c') the difftest's re-sign reuses the LMS leaf: does any verifier notice two messages at one leaf?")
    q0 = q_of(clocks[0].certificate)
    reuse = ClockCertificate(SYNCS[1], replace(clocks[1].certificate, signature=lms.sign_at(
        signers.key(), q0, clocks[1].certificate.message(S0)).encode()))
    print(f"  window 0's clock certificate is at leaf {q0}; window 1's is signed again at that leaf")
    expect("window 1's leaf-reused certificate verifies", reuse.verify(key, S0), True)
    expect("SignedClocks with both (one leaf, two messages): syncs", syncs([reuse, *clocks[:1], *clocks[2:]]),
           tuple(SYNCS))
    used = [q_of(c.certificate) for c in records + clocks + ingress]
    expect("an honest issuer's leaves are distinct across records and clocks", len(used) == len(set(used)), True)

    banner("(d) #3915's driver: when does `served_clocks` take the unsigned stand-in?")
    spec = importlib.util.spec_from_file_location("rt_n2_advised_zk", ROOT / "benchmarks/network_traces/n2_advised_zk.py")
    n2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = n2
    spec.loader.exec_module(n2)
    from certifier_serve import clock_json

    def run_dir(name: str, log_syncs) -> Path:
        d = tmp / "runs" / name
        d.mkdir(parents=True)
        with gzip.open(d / "proxy.json.gz", "wt") as fh:
            json.dump({"syncs": [[s.delta, str(s.rho)] for s in log_syncs]}, fh)
        return d

    def taken(d: Path) -> str:
        try:
            got = n2.served_clocks(d, LINK)
        except (OSError, SystemExit) as e:
            return f"refused ({type(e).__name__})"
        lwx = replace(lw, **got)
        lab = n2.clocks_label(lwx, loose)
        return f"{lab['source']} syncs={len(lab['syncs'])} as_declared={lab['as_declared']}"

    signed_lines = "".join(json.dumps(clock_json(c)) + "\n" for c in clocks)
    # phase 11's attack: the prover declares δ = 0 every window (the certifier signed δ ≥ 1) and logs the same in
    # proxy.json.gz, which is in the run's directory beside clocks.jsonl. `as_declared` is whether the clock source
    # gives exactly the declared syncs, which `advised.view` requires of every window.
    loose = [ClockSync(0, s.rho) for s in SYNCS]
    d = run_dir("regular", loose)
    (d / "clocks.jsonl").write_text(signed_lines)
    expect("clocks.jsonl a regular file", taken(d), f"signed syncs={W} as_declared=False")
    d = run_dir("empty", loose)
    (d / "clocks.jsonl").write_text("")
    expect("clocks.jsonl empty", taken(d), "signed syncs=0 as_declared=False")
    d = run_dir("dangling", loose)
    os.symlink(tmp / "runs" / "gone" / "clocks.jsonl", d / "clocks.jsonl")
    got_dangling = taken(d)
    d = run_dir("directory", loose)
    (d / "clocks.jsonl").mkdir()
    (d / "clocks.jsonl" / "part-0").write_text(signed_lines)
    got_dir = taken(d)
    d = run_dir("removed", loose)
    got_removed = taken(d)
    print(f"  clocks.jsonl a dangling symlink (a fetched tree whose blob is missing): {got_dangling}")
    print(f"  clocks.jsonl a directory: {got_dir}")
    print(f"  clocks.jsonl removed: {got_removed}")
    standin = [g for g in (got_dangling, got_dir) if g.startswith("stand-in")]
    print(f"  => a clocks.jsonl entry that isn't a regular file makes the driver take the stand-in: {bool(standin)}")
    print("     (labelled 'stand-in' in the summary), and then the declared δ = 0 is as_declared, the loosened floor the")
    print("     signed syncs refuse. A removed file looks like a run that predates clocks.jsonl: nothing else in the run")
    print("     (deployment.json, certified.jsonl) records that its certifier signs syncs.")

    banner("SUMMARY")
    print(f"  failed expectations: {fails or 'none'}")
    print(f"  driver falls back to the stand-in beside a non-regular clocks.jsonl: {bool(standin)}")


if __name__ == "__main__":
    main()
