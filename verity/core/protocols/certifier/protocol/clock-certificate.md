Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The clock certificate (`certificate.CertifiedClock`)

N2's advised floor reads each egress window's sync (δ, ρ), and a δ below the one the window's traffic was certified
under loosens it (the phase-11 attack). So `advised.view` refuses a window whose declared sync isn't the one the certifier
ran it under, which the verifier takes only from its own clock source (`advised.ClockSource`). The clock certificate
makes that source the certifier's signed record.

- **What the certifier signs.** At each egress window's close, beside the window's record certificate, the certifier
  signs the sync its own shaper ran the window under (`active.Shaper.syncs`, the sync it computed each frame's
  eligibility with), salted as a record digest is, as `Issuer.salted_clock(link, window, sync)`:

      clock digest  = tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)
      signed digest = SHA-512("verity/net-cert/v1" ‖ salt_c ‖ clock digest)        (clock_commitment)

  with ρ = p/q in lowest terms, and `salt_c` 64 fresh bytes of the certifier's own coins.

  It signs that digest under the record certificate's message (`certificate_message`: the key, the certifier session,
  the link, `egress` and the window). `CertifiedClock` is the sync, the salt and that certificate. A rehearsal's
  certifier (`certifier_serve.ServedProxy`, which hands its `CertifierProcess` each egress record with its shaper's
  sync) writes one line of its `clocks.jsonl` a window, with the opening (`certifier_serve.read_clocks`).
- **What is public.** What the certifier signs is witness (AGENTS.md, "Auditor-trusted devices"): the sync and its
  salt never reach the auditor in the clear. Only the certificate is public, a signature over
  a hiding commitment: its key id, link, direction and window, the salted digest and the LMS signature. The reveal
  endpoint serves that alone (`CertifyingPublisher.public`), and the client keeps it in its `clocks.jsonl`
  (`certifier_serve.keep_certificates`), checked under the deployment's key and session; it holds no opening.
- **A fresh salt each window.** The clock digest is a function of the sync alone, so under one salt two windows of one
  sync would sign one digest, which shows anyone holding the certificates that their syncs are equal. The issuer never
  signs a clock under a salt it drew for an earlier one (`Issuer.salted_clock` raises), the verifier refuses two
  openings sharing a salt, and the client, holding no salt, refuses a clock certificate whose signed digest an earlier
  window's signs.
- **A certificate of its own, not a head field.** Signing the sync inside the egress head (`RecordHead.encode`) would
  change the head encoding that the head check's circuit, the Lean records and the CertifierDevice pin. The record
  certificate, the head and `active.py` are unchanged.
- **Domain separation.** The record certificate of the same sampling block signs SHA-512("verity/net-cert/v1" ‖ salt_c ‖
  R), R a record digest, and the clock certificate the same over a clock digest, each digest under its own tag. So one
  opens as the other only on a SHA-512 collision, as with the network certifier's cells certificate (`warden_rows`). Nor can the
  head check take a clock certificate as a record's: that needs a head whose record digest is the clock digest.
- **The verifier of record** is `LinkWindows(certified_clocks=…)`, whose `clock_source` is `advised.SignedClocks` under
  the link's own certifier key and session. It reads each window's sync through its opening, against the salted digest
  its certificate signs (`CertifiedClock.refusal`): the openings are the verifier's own read of the certifier's
  output, never a public answer. Window w's sync is the sync of the openings that name the link and window w and
  hold, and the windows run from 0 to the first one without exactly one such sync. `SignedClocks.refused` names each
  opening it took no sync from and why: not a salted clock's opening, not egress, another key's, a sync and salt that
  don't open the signed digest (an altered digest or sync), a signature that doesn't verify (forged, another
  session's, or relabelled to another link or window), a salt another sampling block's opening has too, or two signed
  syncs for one window. Once a frame is released in a window without a sync no floor is under a signed sync, and
  every window of the link is refused. The link's verdict is the AND over its windows, so the link is refused either
  way. Whoever delivers the openings, the prover included, can't change what they open to.
  `LinkWindows(clocks=…)` takes a stand-in source in their place, never beside them, each labelled as one: the tests'
  (`StandInClocks` in `tests/advised_traffic.py`), or, only when asked for, a run's proxy log for a run from before
  the certifier signed its syncs (`n2_advised_zk.py --clock-stand-in`). Without that flag the driver refuses a run
  without `clocks.jsonl`, and any `clocks.jsonl` entry, a regular file or not, is read as the run's certificates, so a
  dangling link, a directory, an unsalted line or the client's public copy there is refused, not taken for a run
  without them. A deleted `clocks.jsonl` still looks exactly like a run from before it, so only the flag proves such a
  run, on a source labelled a stand-in. The driver's summary names the source and whether its syncs are the declared
  ones, never the syncs.
- **The opening is on the host, not yet in the unit** (PROTOCOL.md's named gap, "The clock's opening"): the verifier
  of record opens each window's clock on the host, as the in-the-clear check opens a record over
  `CertifiedRecord.verify`. On an accepted window the sync it opens is the declared one, a public output already; on a
  refused one it is the sync the certifier ran, which the unit's opening would keep hidden. The design: the unit takes
  each window's clock opening as witness, checks it against the certificate's signed digest (the clock digest and its
  salted digest, two SHA-512s of a few blocks, as the head check computes a head's signed digest), and computes the
  floor from the opened syncs inside the unit, so the floor leaves the public row and the verifier reads no sync. It
  is a circuit change to N2's advised unit (its Definition, pinned digest and public row, with its `circuit-check`
  report), not a Lean one: the network certifier's Lean issuer already signs the salted form
  (`CertifierDevice.Issuer.clock`).
- **What it rests on** is what the record certificate rests on: LMS's unforgeability (`euf-cma/lms-sha512-n64`), and a
  certifier that is the auditor's device, on a host the prover doesn't control. A certifier run on the prover's own
  machine (a rehearsal's against an endpoint on its own loopback) signs whatever that machine runs, so there its signed
  syncs prove no more than its records do.
- **Tests:** `tests/test_network_certifier_advised_signed_clocks.py` covers an honest link accepted; a declared sync
  off the signed one refused, phase 11's δ = 0 against a signed δ = 1 among them; each refused opening named with why
  (a forged signature, an altered digest, a sync swapped, a wrong window, a reused salt, another key, session or link,
  an ingress or record certificate, a record's opening, two syncs for a window); and the issuer refusing a salt an
  earlier signed clock took, but not one a refused clock had.
  `benchmarks/network_traces/tests/test_certifier_serve.py` checks that the served proxy hands its certifier its
  shaper's syncs, and that the certifier process signs them as the inline publisher does, each under its own salt;
  `test_client_certificates.py` that the client keeps every clock certificate's public part and nothing of a sync or
  salt, and refuses a forged signature, an altered digest, a wrong window and a reused salt by name.
- **The Lean device** signs the same salted form: `CertifierDevice.Issuer.clock`, and `clockWith` against the world,
  which draws `salt_c` from the issuer's coins as `certifyWith` does, beside `certify` in
  `verity/core/devices/certifier/lean/CertifierDevice/Issuer.lean`. It signs `clockCommitment salt sync`
  (`certificateDigest salt (clockDigest sync)`), refuses a salt one of its signed clock certificates took
  (`clockSalts`, carried beside the `Issuer`, whose structure `Properties.CertifierDevice.issuerVerifies` reads), and
  its `ClockCertificate` holds the sync and the salt, witness, with the certificate; `ClockCertificate.verify` checks
  egress, the key, that the sync and salt open the signed digest, and the signature. A salt is taken once signed
  under, in both: a refused reservation signed nothing, so nothing public shares its salt. The issuer difftest holds
  it to `Issuer.salted_clock` and `CertifiedClock` (`verity/core/devices/certifier/lean/scripts/issuer_vectors.py`):
  clock calls under their own salts, some taking an earlier call's, the salt among the tampers and a salt not 64
  bytes among the refusals.
- **Not yet:** a run made before the salted form has no salted clock certificates, so the source of record refuses its
  advised links.
