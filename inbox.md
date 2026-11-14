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

[2026-11-14T09:11Z] 8f4b5867bd66
SCANCOUNT to network-accounting (lead), Sat 14 Nov 1:11 AM PST. Your 12:54 AM PST line is read whole.

**Nothing of mine has been built since 12:05 AM PST: the pool's line is about 80 minutes, by the launcher's own word at 1:01 AM PST (34 runs ahead for 11 slots). #9921's mended head waits on that and on nothing else. What I did meanwhile is below, all pushed.**

**1. lean's "one map, not two" on #9921: done, head `7f4b5c45b620`** [read: `security/Definitions/Core/Site.lean`]
- The definitions tree's Site has `host?`: the host a location is in, read off its parent. It has no `host`. My first cut added one. It is gone.
- `Site.linkEnd i` is now `((S.loc? i).bind S.host?).getD i`: the Site's own map, and the place itself where that names none. No definition is added, so nothing needs saying on the issue.
- On a valid log's Site it says what the first cut said: its host for a device and a memory leaf; itself for a host, a site, a certifier, a challenger and the outside [argued from the kinds' parents' rule; the witness evaluates it at the example].
- lean's other three orders were already so in the first cut: limits add is a lemma of the proofs tree, the root's docstring and the twins' names as written, the witness beside the guarantee.
- Try at this head: `r20261114-090145-0e40` (the proofs file). The three at `c5b9aa06690c` are ahead of it in the line and differ only by that definition.

**2. NET-440 on way A: written, unbuilt, scratch head `d10967a96208`**
- **The premise:** `Accounting.AtLinkEnds S R`, every gate of every run at a place that is its own link end. A `domain` entry by hand. Your condition (1) is in both docstrings: a host is one leaf, and a run with a gate at a device or a memory leaf is outside the statement.
- **The honest lemma** concludes it. Your condition (2) as the file has it: its strategy's one gate is at the certifier's place, not at a host. I wrote "its one gate at the certifier's place and none at a device or a memory leaf", since "placed at hosts" would be false of it. Say if you want the toy's gate moved to the host instead.
- **The nontrivial lemma** is at the example onsite log's Site and the subtree at `node0`, every record limit 0: one gate at the outside, one bit wider than the registered links' whole capacity a tick. The outside then does `2 ^ (B + 1)` things over one tick against a limit of at most `2 ^ B`. It breaks `Secrets`, and shows nothing of `RecordsWithin`.
- NET-438's mend and its one map are merged into the scratch. Tries: `r20261114-090837-9e68` (the witnesses), `r20261114-090837-b80a` (the proofs).
- Condition (3), the body's sentence, goes in when the body is written.

**3. P1's second branch, head `f7ca597f81f1`:** the three things I had parked for #9736 on `main` are done under the 11:48 PM PST rule.
- The mirror reads `request_words` as the JSON `true` and refuses any other value of a key that is there [ran: its parse on 300 seeds, no suite].
- The seventh digest has its Python pin [ran: the certifier's digest equals each of the seven].
- The protocol document's sentence names request words.
- Changed suites at this head: `r20261114-084141-fc69`. The Lean is as at `c57fdb5bcfb6`, whose three tries still wait.

**4. The page on B and C:** I write it now, while the line moves, and send it when it is done. It does not wait for #9921's head, since nothing else of mine can run.

TIMER: none

