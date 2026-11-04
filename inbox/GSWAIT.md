
## 00:43Z, proofs coordinator: a definitional change (decision)

GSJOINT found that `Compiled.E` (coins per block) cannot recover `Sampling.KS` (coins per slot): a block at two slots is
retried on the same coins, which changes the laws `Bext` reads; and `Sessions.runs` pins the session's exact shape while
the deployment gives only bounds. Decision: `Compiled.E` takes per-slot coins, and the shape carries only what the coins
read, with each session's errors bounded separately (GSJOINT's note, R8 section). GSJOINT makes the change, coordinating
with GENSAMPLE, and adapts R3a/R3b/R3e. If your piece reads the coin indexing or the shape, follow GSJOINT's note.


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-bd19c2b5-e87a-5f08-b9ae-47fd8c964428.md, which you may not read)

## 2026-11-04T11:44Z from proofs (GSWAIT: witnesses target)
PEARLPROPS opened #7267 into remote (stacked on #7221 and #7248) at 0d8f5ec473: SEVEN Props proved and listed (planMeans, acceptMeans, refusedBeforeTheCoin, yIsTheUnscale, loadedGivesUp, tileAcceptMeans, tileKeepsTheCap; owner-only guarantee entries, shape.properties entries), with NO witnesses: each fails witnesses.missing until yours land. Build your Witnesses file on #7267's head (merge it), cover all seven plus OpeningBinds (#7248), and push so PEARLPROPS' records run (d8b2, ~13:45-14:00Z) can be followed by one with your witnesses. OpeningBinds and WorkIsTheCredit stay unstated except OpeningBinds proved on #7248.

## 2026-11-04T13:52Z from proofs (GSWAIT: your Load.lean change touches PEARLPROPS's proof)
PEARLPROPS's `LoadedGivesUp` proof in #7267 (head 524a494acf) unfolds `PouwPearlPlus/Load.lean`, which your Json `==` fix changes. When you merge #7267 into the witnesses branch, or #7267 merges your fix, build `LoadedGivesUp`. If it breaks, fix the proof there, not the statement, and tell PEARLPROPS in your note. Both lock edits add the same `PouwPearlPlus` entry to Security/Properties' `meaning`, so they merge as one.

## 2026-11-04T15:10Z from proofs (Pearl+ words: "debit" is now "unnecessary work")
Daniel (6:49 AM PST, via compute-research): Pearl's "debit" is now **"unnecessary work"**: the multiply-adds a tile can leave out with its output bit-for-bit unchanged. The cap check is **"the unnecessary-work bound"**, and rho is **"the unnecessary-work share"**. Incorrect work, which sampling bounds, is separate. Use the new words in prose, docstrings, PR bodies and notes now. **Don't rename identifiers** (`debit_units`, `debitOf`, `capOK`, `Device.cap`, `DebitIsTheReplay`, `Debit.lean`, `TilePassGivesTheCap`) until after the Pearl+ run of record. The glossary rows are on CMP-310's PR 2.

## 2026-11-04T15:22Z from proofs (correction: "inert work", not "unnecessary work")
Daniel corrected it at 6:53 AM PST: the term is **"inert work"**, meaning operations that don't affect a tile's output. With it come **"the inert-work bound"** (the cap check) and **"the inert-work share rho"**. Use these in prose now, in place of the 15:10Z "unnecessary work". Identifiers still stay until after the Pearl+ run of record.

## 2026-11-04T16:10Z from proofs (GSWAIT: add Evaluate.TC.Fp8Atom to Properties' meaning)
PEARLPROPS found a second layer: four of #7267's Props (`acceptMeans`, `refusedBeforeTheCoin`, `tileAcceptMeans`, `tileKeepsTheCap`) read definitions in `Evaluate.TC.Fp8Atom`, which isn't under Security/Properties' `meaning`, so the records run fails "reads outside meaning". PEARLPROPS added it right after `Evaluate.TC.Dots` (eb0210d9a5 on #7267). Your witnesses branch has `PouwPearlPlus` under `meaning` but not this entry, so add the same line, or merge #7267's head, before your next records run.
