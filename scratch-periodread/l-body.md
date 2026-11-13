Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. One more statement about the ledger SPOT's period check reads, and its proof. It is on the proofs' head (#9625), which is on #9581's, #9595's and #9537's; all four land first.

**Every presented line but the last is the device's own line whole, its signature included; or a break.** The two ledger statements before it hold each line to the device's "its signature aside". The check reads an LMS leaf from each line's signature, so that left the leaf free. This holds every line's signature but the last's.

## The statement

`Properties.Spot.Period.LedgerOfDeviceSigned`, in `security/Properties/Spot/PeriodLedger.lean`: for every device that numbers and chains its ledger's lines under the verifier's key, and every ledger the period's check reads, the presented lines but the last are the first lines of the device's own ledger, each whole, in order; or the finder at the lines' whole texts outputs a break.

- **Why it holds.** The reader chains the presented lines: each names as `prev` the SHA-512 of the whole text of the line before it, signature included, and `prev` is in what the device signs. So two lines before one signed line have one digest, and are one line or a collision.
- **Where it stops.** No line names the digest of the last presented line, so its signature, and the leaf read from it, stay free. The statement's docstring says so.
- **No new cryptographic assumption.** It does not take strong unforgeability of LMS: a second signature on a signed message is excluded by the chain, not by the scheme.

## Two hypotheses, both facts of the device

- **`Numbered`**, as `LedgerOfDevice` takes it: a line of the device's ledger that is an entry numbered `i` is its line `i`.
- **`Chained`, new**: where a line of the device's ledger and its next line are both entries, the next names as `prev` the SHA-512 of the earlier line's whole text.
- **Neither is a named assumption, and no listed guarantee reads either.** Each is the device's behaviour, which the model of the device (`History`) does not hold. That is soundness's decision for the device's ledger facts (PRF-490): no law goes into `History`; each fact is taken as a hypothesis of the one statement that uses it, and becomes a lemma of the modelled device once soundness's device pull request builds that device's whole ledger. The docstrings say so without the issue.

## Its own finder

- **`findLineBreak`** calls the fixed finder `Properties.Challenger.findBreak` on the ledger and, if that outputs nothing, outputs the first SHA-512 collision among the whole texts of the presented lines and of the device's own.
- **`findBreak` and what it hashes are untouched**, so no record of a guarantee that reads them moves. `findBreak` hashes each line without its signature, which is why it cannot see two lines that differ only there.
- **A break of `findBreak` on the ledger is a break of this finder** (`lineBreak_of_break`).

## The proof

`security/Proofs/Properties/Spot/PeriodSigned.lean`:

- **`verifyLedger_prev`**: the reader's fold holds each entry's `prev` to the digest of the entry before it.
- **`prev_of_text`**, **`ledgerEntry_text`**: two entries whose texts without `sig` agree name one `prev`, and a line is its entry's text.
- **`lineBreak_of_collision`**: two different whole texts with one digest are a break.
- **`ledgerOfDeviceSigned`**: the statement.

In `Proofs/Properties/Spot/PeriodLedger.lean`, the per-place step of `ledgerOfDevice` is lifted out as `own_line_at`, which both proofs use. `ledgerOfDevice`'s statement is unchanged.

## The lock

- **`security/Properties/lean-audit.json`**: one more `unstated` line, for `LedgerOfDeviceSigned`, naming both hypotheses and the theorem that proves it. The period has nine.
- **`security/Proofs/lean-audit.json`** and the lakefile's roots gain the module.
- **No guarantee entry**, as for the other eight.

## What it does not say

- **The last presented line's signature**, and so the leaf of a window's end or open when it is the last line presented.
- **That the device chains its lines**: taken, not proved.
- **Anything of the check's verdict.** It changes no check and no existing statement.

## Checks

CHECKS

Refs PRF-600
