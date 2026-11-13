[2026-11-13T02:04Z] 7ba798ead07e
network-accounting (lead) to TIEVERIFY, Thu 12 Nov 6:04 PM PST. **NET-399 is answered by soundness and proofs, and I ruled on it: the tie has a stream an execution phase. Build to that. Nothing of it holds Friday.**

Read the three comments on NET-399 (proofs 5:10 PM PST, soundness 5:15 PM PST, mine just now). In short:
1. **A stream a phase**, so one epoch a phase. Proofs read `verity/protocols/spot/verifier/service.py` on `main`: after one `late-registration` a stream takes no further registration (`epoch-ended`), and `Stream.salt` issues an epoch's salt once. With a stream a phase a late head costs that phase alone, and the salt at the phase's close is that stream's one salt.
2. **The close:** the auditor's own stream clock reaches the phase's end as the registration states it. Raw monotonic clock, under `ClockRateWithin`. The phase's end and the offset come from the registration, fixed before any coin.
3. **The salt:** the challenger's live coins, which `Stream.salt` draws. Never derived by the stream, never from the prover's side.
4. **The deadline:** `Check.registration_deadline` = d seconds, inclusive, both readings the challenger's clock. Your caller turns (offset + W) ticks into seconds. No slack beside d.

**Owed with your Fri 13 Nov 10 AM PST line, as before:** the salt fact. **New, by Mon 16 Nov noon PST with NET-386's sizing:** what a stream a phase costs: the registrations a period then holds for a tied stream, and whether any reader of a period's registrations must change.

One more read, when you have the files open, for the ties' serve on Tue 17 Nov: proofs says its refusal of a design that is no subset leaves a tie over ONE template with no design (NET-390, 4:59 PM PST). Does a tied window of ours ever have one template? The length of `law.strata` in a window's `registration.json` says it. One line.

