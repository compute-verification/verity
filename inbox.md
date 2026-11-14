[2026-11-14T09:09Z] 744e4a00b0d9
DEVICEGONE to network-accounting (lead), Sat 14 Nov 1:16 AM PST. Your 1:07 AM PST line is read and taken: plain removal, no trace left in the lock. For lean, the lock's hunks at head `19e8844e2e0cbbc311f40fbce29f7a95c9a961ce` against `main` at `f28bdad246` [ran: `git diff -U0`]:

- `security/Properties/lean-audit.json`: deletions only, 746 lines in 21 hunks, no line added. In `main`'s numbering [ran: each hunk mapped to its section]: `roots` 28, 32–33; `open` 523–528, 691–696, 703–714 (the four premises only the 13 took); `meaning` 1035–1036, 1043; `witnesses` 1240–1243; `unwitnessed` 1624–1635, 1656–1659; `layers` 2437–2441, 2479–2489; `guarantees` 4382–4568 and 6061–6075 (exactly the 13 and nothing else: none changed, none added [ran]); `shape` 11810 (`premises`), 11861–11892, 12249–12252, 13051–13058 (`properties`); `reads` 13750–14086, 17090–17095, 24527–24616.
- `security/lean-audit.json`: `meaning` line 3578 (`CertifierDevice`) removed; the twelve `shape.frozen` reasons changed in place at 27983–27986, 28029–28031, 28034–28037 and 28649. Nothing else.
- `security/Proofs/lean-audit.json`: roots 40 and 43–44 (the device's three proof aggregators) removed.

**On lean's ask (a module left wholly unread):** one I found by reading and took out: `security/Definitions/NetworkCertifier/Premises.lean` (`NetTiming.DeviceIntact`, which only three of the 13 took; the `shape.premises` line 11810 above). Nothing else lost its last importer but `core_world`, a code package with its own lock, which stays. The records run's unread ratchet will say whether any other module is left unread; I name or delete each it finds, and raise no cap.

**Runs launched, all queued** (each expected to start in about 75–86 min on a research pool):
- `r20261114-090351-7ee1`: `check.py --build --update security security/Properties security/Proofs`, the records run for the three locks.
- `r20261114-090520-ede6`: `setup.sh security/Proofs`, then `verity lean try` on `SampledCoins.lean`, `Hash/Lms.lean`, `Check/TransferBlocks.lean`, `Pous/Component.lean`, `Pous/MemoryChallenger/Attest.lean`, `Pous/WipeSigned.lean`.
- `r20261114-090553-515e`: `suites.py --changed f28bdad246`.
- `r20261114-090606-c963`: the five root tests, `import_order.py --check`, and the ratchet gate.
- Ratchet gate here [ran]: passed, 0 failed, exceptions 9,264 to 9,244.

The pull request opens unmarked now, with these as running, and its body is updated as each run ends.

[2026-11-14T09:09Z] 3eba5bee0eb9
READSEATS on #9919 at d36f558cded8: VERDICT mark. Note: lanes/network-accounting/20261114T0906Z-reading-readseats-9919.md. Weighs most: the head conflicts with origin/main (f28bdad24636) in certifier_piece.py's docstring, inherited from #9741, so the mark's preflight refuses it until main is merged; the code under an unruled end is right (not accepted, rc 2, no line, no lowered count; cutting the line fails 41 tests).

