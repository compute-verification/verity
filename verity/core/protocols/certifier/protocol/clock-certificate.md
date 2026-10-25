Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The clock certificate (`certificate.ClockCertificate`)

N2's advised floor reads each egress window's sync (δ, ρ), and a δ below the one the window's traffic was certified
under loosens it (REDTEAM phase 11). So `advised.view` refuses a window whose declared sync isn't the one the certifier
ran it under, which the verifier takes only from its own clock source (`advised.ClockSource`). The clock certificate
makes that source the certifier's signed record.

- **What the certifier signs.** At each egress window's close, beside the window's record certificate, the certifier
  signs the sync its own shaper ran the window under (`active.Shaper.syncs`, the sync it computed each frame's
  eligibility with), as `Issuer.clock(link, window, sync)`:

      clock digest = tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)

  with ρ = p/q in lowest terms.

  It signs that digest under the record certificate's message (`certificate_message`: the key, the certifier session,
  the link, `egress` and the window). `ClockCertificate` is the sync and that certificate. A rehearsal's certifier
  (`certifier_serve.ServedProxy`, which hands its `CertifierProcess` each egress record with its shaper's sync) writes
  one line of `clocks.jsonl` a window (`certifier_serve.read_clocks`).
- **A certificate of its own, not a head field.** A window's sync is public: each window's declared sync is in
  `PUBLIC_ADVISED`, and its choice is the log2 #Σ_sync bits the sync charge counts. So its digest needs no salt, and the
  verifier holds the declared sync to the signed one in the clear, outside the unit, as it holds the unit's other
  public inputs. Signing it inside the egress head (`RecordHead.encode`, the earlier design) would change the head
  encoding that the head check's circuit, the Lean records and the CertifierDevice pin, and the verifier would check
  nothing more. The record certificate, the head and `active.py` are unchanged.
- **Domain separation.** The record certificate of the same link-window signs SHA-512("verity/net-cert/v1" ‖ salt_c ‖
  R), and the clock certificate a tagged hash under its own tag. So one verifies as the other only on a SHA-512
  preimage, as with the warden's cells certificate (`warden_rows`). Nor can the head check take a clock certificate as
  a record's: that needs a salt and a head whose salted digest is the clock digest.
- **The verifier of record** is `LinkWindows(clock_certificates=…)`, whose `clock_source` is `advised.SignedClocks`
  under the link's own certifier key and session. Window w's sync is the sync of the certificates that name the link,
  egress and window w and verify, and the windows run from 0 to the first one without exactly one such sync. A window
  is refused, with every window after it, when it is unsigned, signed under another key, session, link, window or
  direction, signed for another sync than the one its certificate carries, or signed for two syncs; once a frame is
  released in one of those windows no floor is under a signed sync, and every window of the link is refused. The
  link's verdict is the AND over its windows, so the link is refused either way. The certificates are public, so
  whoever delivers them, the prover included, can't change what they say.
  `LinkWindows(clocks=…)` takes a stand-in source in their place, never beside them: the tests' (`StandInClocks` in
  `tests/advised_traffic.py`), or a run's proxy log for a run from before the certifier signed its syncs, each
  labelled as one.
- **What it rests on** is what the record certificate rests on: LMS's unforgeability (`euf-cma/lms-sha512-n64`), and a
  certifier that is the auditor's device, on a host the prover doesn't control. A certifier run on the prover's own
  machine (a rehearsal's against an endpoint on its own loopback) signs whatever that machine runs, so there its signed
  syncs prove no more than its records do.
- **Tests:** `tests/test_network_certifier_advised_signed_clocks.py` covers three cases: an honest link accepted; a
  declared sync off the signed one refused, phase 11's δ = 0 against a signed δ = 1 among them; and unsigned and
  wrongly signed syncs refused. `benchmarks/network_traces/tests/test_certifier_serve.py` checks that the served proxy
  hands its certifier its shaper's syncs, and that the certifier process signs them as the inline publisher does.
- **The Lean device** signs it too: `CertifierDevice.Issuer.clock`, and `clockWith` against the world, beside `certify`
  in `verity/core/devices/certifier/lean/CertifierDevice/Issuer.lean`, with `ClockCertificate.verify`. The issuer
  difftest holds both to `certificate.Issuer` (`verity/core/devices/certifier/lean/scripts/issuer_vectors.py`):
  honest clock certificates, ones tampered or signed again for another window, direction or key, a record certificate
  passed as a clock, and the syncs and link-windows it refuses.
- **Not yet:** a run made before `clocks.jsonl` has no clock certificates, so the source of record refuses its advised
  links.
