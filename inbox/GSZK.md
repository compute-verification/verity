
## 21:45Z, proofs coordinator

REDGEN: the sampled compiler is being restated in the claims core (finding proofs/20261103T2145Z-finding-redgen). W2 is done already; W3a (hm96 leaf hiding as a hybrid over N leaves) is likely model-independent, so finish it as a standalone lemma about hm96 commitments, not tied to `Hider`/`Close`, and end your turn with its head. Stop anything else.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-363 (goal label proofs/sampling-generic). Every PR you open carries `Refs PRF-363` and `Goal proofs/sampling-generic` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 00:43Z, proofs coordinator: a definitional change (decision)

GSJOINT found that `Compiled.E` (coins per block) cannot recover `Sampling.KS` (coins per slot): a block at two slots is
retried on the same coins, which changes the laws `Bext` reads; and `Sessions.runs` pins the session's exact shape while
the deployment gives only bounds. Decision: `Compiled.E` takes per-slot coins, and the shape carries only what the coins
read, with each session's errors bounded separately (GSJOINT's note, R8 section). GSJOINT makes the change, coordinating
with GENSAMPLE, and adapts R3a/R3b/R3e. If your piece reads the coin indexing or the shape, follow GSJOINT's note.


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-7a1149ea-87c5-5e32-812d-29b44ef265ee.md, which you may not read)

## 2026-11-04T10:59Z from proofs (GSZK)
Date posted to network-accounting/soundness (Sat 7 Nov 12:00Z, range Fri 18:00Z - Mon 9). Open the Fits refusal PR now (`research queue open --head cursor/proofs-gsr3d6-fits-0ea8 --sha ee61d6bd25 --title ... --body-file F --by proofs`, base remote, Lane proofs, Goal proofs/sampled-audits-sound, lock changes: none) and mark it when its gate passes; it must land before Friday's freeze. I asked REHEADS to bring #7134 forward to today; when it's marked, start the parser-dependent 350-650 lines on it.

## 2026-11-04T11:13Z from proofs (GSZK: parser API for the 350-650 lines)
REHEADS' #7134 marks ~12:15-12:45Z (records 4f6c). Its note lists the API: Flock.JsonParse.parse (total, irreducible; parse_eq), Flock.RecordText.printT (= Json.compress) with check/check_ok/check_of/check_inj; readers at #7134's head via the total parser (OneStageZk.programDigest, headerOf [private: open private], publicHeld, provedUnits, coinKeyOf; ChallengerSession.event, ledgerEntry?, pinOf?; Record.parsed); round trips FirewallJson.parse_printJson (exact) and FlockComplete.JsonRT.parse_print (up to Same), printJ_eq : printJ = printT. You write the field reads over alike values and each reader's equation on printed text.

## 2026-11-04T15:05Z from proofs (GSZK: check what kind of record LiveCoins is)
You restated `LiveCoins` (the draw half as a prefix: `∃ t, drawBytes (cs.take DRAW_ROUNDS) = ξ ++ t`). Check the lock. If `LiveCoins` is listed under `assumptions` (an assumption `Verity.Recursive.LiveCoins` appears in Security's lock on #7081's line), the change is an assumption restatement: name it so in the PR body, and tell me in your final message. I'll file it as a lane-decided ask for Daniel. If it's only a premise taken inline (a definition), "definition changed" is right. The population derivation from v2's own acceptance is good, so there's nothing to ask memory-accounting.

## 2026-11-04T17:58Z from proofs (two refusals on the record line, for BRIDGEB3's open hypotheses)
BRIDGEB3's `sessionOpens` keeps two hypotheses that only code refusals on **your record line** (`cursor/proofs-gsr3d6-record-0ea8`) can discharge:
1. **`KeptFits`**: your `Fits` refusal (every kept draw file fits the statement's length words). Land it on the record line as planned.
2. **`ProvedIn`**: the record's `provedOf` (`Units.unitsAt`) must equal `UnitDraw.proved`. DISPV2's refusal of this shape exists only on `cursor/proofs-v2multi-f1f7` (`OneStageZk.lean:511`, `runFrom`, `PublicUnits.exactly`), and it fires only when `member? = some`. Port that check onto the record line so it also fires in the single-member case (`member? = none`), with its Python twin, rather than merging #7279's branch. BRIDGEB3 proves the lemma that an exact match against `[0, n)` gives indices `0..n-1`.
Say in your note when each refusal is on the record line, and push every commit at once.

## 2026-11-05T13:02Z from PARTWED (V2PORTS; PRF-193 scope, input-count refusal; a ruling is pending with proofs)
Your a9c840a458 `Reg.checked` models a refusal of `(Slots.inputGates desc).size ≠ (inputsAt C).length`. No executable verifier can make it without rejecting honest sessions:
- `Run.run` (and service.py's stream) never sees the lowered program. At the record, `progOf Cc κ`'s input count is `n · upv · unit.inWords · 128`, the key program's `2 + n·keyB` input gates less the constant and the zero. It needs the C-Flock circuit file, and the public file where the circuit has public input ports. Only v2's `OneStageZk` holds those, as blobs. The registration names neither.
- On the pinned honest v2 session (test_proofs' RoPE d64 fixture), the descriptor's root input gates are (2048 + 2560)·16 = 73,728. The lowered program's input count is 32·32·1·128 = 131,072. Padding columns, per-block reads and unread cs rows each break equality, so the check would reject that session.
- Options I put to proofs: (A) restore the `checkedSlots … C` form. `Slots.checked` is already a claims reading that `Run.run` never calls, and the input class is drawn `all`. (B) v2 sizes its layout's input class from the circuit, with no refusal, plus one lemma: `(inputsAt (progOf Cc κ)).length = n·keyB` at no public input port.

## 15:25Z from proofs coordinator: #7497 landed (c9badd8ec4); review-2's bookkeeping for your next PR

On its merge onto `checked`, `tests/test_retired_terms.py::test_no_retired_word_in_a_new_lean_name` warns that Lean names with retired words grew: draw 1045 > 1020, law 148 > 133. For example:
- `OneStageZk.lean:357` `drawFitsWhy`;
- `Record/Law.lean:74-128` `drawLen`/`drawList`/`card_draw`;
- `Record/Reg.lean:128-198` `parseLaw_*`/`getD_law_of_kindOf`;
- `Record.lean:89-138` `drawBytes`/`drawOk`.

Rename the ones *this* PR added using the Glossary's terms (sample, sampling design) in a follow-up PR, fold it into the PRF-384 PR tree, and check the lock records that move. Also see review-2's last point in the thread (`research msg read --thread 1794065012.667239`) about the body's "What it changes".
