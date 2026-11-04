
## 21:54Z, proofs coordinator: every PR names its goal (Daniel, today)

From now on each PR body carries `Goal <id>` naming the Goals-page goal it serves (e.g. `Goal proofs/zero-knowledge`,
`Goal proofs/sampled-audits-sound`, `Goal proofs/recursion-knowledge-sound`, `Goal proofs/sampling-generic`,
`Goal proofs/recursion-generic`, `Goal proofs/terminology-clean`, `Goal proofs/zero-knowledge-any-worker`,
`Goal proofs/zk-leaks-closed`, `Goal proofs/firewall-checks-provable`). Add it to any open PR of yours the next time you
edit its body (a body edit needs no new mark). ci may soon refuse a mark on a PR that names none.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-360 (goal label proofs/finders-charged). Every PR you open carries `Refs PRF-360` and `Goal proofs/finders-charged` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 23:17Z, proofs coordinator: pins.py --merged runs Lean builds locally

`pins.py --merged` runs pinned fixtures' generators, which `lake build` on your VM, and nothing refuses it yet (ci CI-383,
hook refusal Wed). Until then, run it only through `research run --on auto`, or recompute pins by hashing.

## 00:53Z, proofs coordinator: check shape.inline before every mark (until LEAN-172)

lean found that something run AFTER a merge (pins.py --merged, check.py --update, or committing a records run's lock)
can drop `shape.inline` entries from a lean-audit.json: #6676 lost 22 that way, and #6075's committed record may have too.
Until lean's fix (LEAN-172) lands: before marking, diff each lock's `shape.inline` against the merge driver's output for
your merge (`python3 tools/verity/lean/merge.py`) and against next's. Restore any entry whose premise the merged tree still
takes inline, as a normal commit, and say so in the PR body.

## 01:24Z, proofs coordinator: the repository suite fails on next itself

`next` alone fails seven `repository` tests (tools/research's environment reads, the writing-goals skill's retired terms,
lint baselines; ci holds #7035, #7049, #6899 for the lane-text lint). If your suites run fails only in `repository`, compare
the failing test ids against next's own (run the repository suite on next's tip, or read ZKD3's #7175 body, which lists
them). Identical set: not yours; say so in the PR body and proceed. Anything extra: fix it.


## 2026-11-04T16:05Z from proofs (copied: these went to inbox/bc-5b73404f-1ae2-595d-ab16-5855746d41ea.md, which you may not read)

## 2026-11-04T10:02Z from proofs (LINKFIX, NET-205 10:00Z)
1. "MODSYS's map" is ODLEAN's SampledProofs.Bits: rowBit, instance-level inBit/outBit and wiresOf are all in Bits, on its own branch off origin/main, due Thu 06:00Z; your four conditions were passed on. So item 4 (row-map agreement) builds on ODLEAN's branch.
2. The 262,144-bit refusal: network-accounting is checking that no V3c circuit-file unit (counting upv instances and source bits) exceeds it, answer by Wed 18:00Z; if one does, the refusal would refuse Monday's sampled audit. Open the PR, but don't mark it until that check is clear. If you can check V3c's circuit file yourself faster (the evidence store has it), do, and put the max unit's bits in your note.

## 2026-11-04T10:04Z from proofs (LINKFIX, NET-205 10:03Z)
ODLEAN's final SampledProofs.Bits is on `cursor/sampled-bits-061a` off origin/main: rowBit (RowPos: row, leaf, bit; word x bit t = row bit 16(x-s)+t), instance-level inBit/outBit and wiresOf (check's full body by Thu 12:00Z). Build item 3 (redefinitions) and item 4 (row-map agreement) on it. I answered their Q1 (wiresOf across members: NetworkCheck checks it, agreed). Q2 is yours, answer on NET-205 by 14:00Z via your note (I post): a unit input copied from another net's output inside a template instance (a tail stage) is, in keyProg, a free source gate with no rowBit, so it stays unwired and counts as a free input of the claims program (the bridge's current treatment). Is that right against Correct/Boundary/Complete (an unopened free input makes the honest unit non-Correct?), or should Bits wire it to its net's output? Give the answer with file:line and what it changes.

## 2026-11-04T10:31Z from proofs (LINKFIX)
GSJOINT takes your item 2 (the deployed Code proofs) from now; you keep items 1, 3 and 4. Coordinate file ownership in your note (one owner per Lean file).

## 2026-11-04T12:55Z from proofs (LINKFIX: item 2 is yours after all)
Item 2 ended up proved twice: yours (cursor/proofs-bits-deploy-c9c9 at 7b406525ad) and GSJOINT's (cursor/proofs-bits-code-c62f at 410bafc917; 643 lines, std axioms, no class bound). We keep yours: it is stacked on item 1, so Partitions has the class bound. GSJOINT has stopped item 2 and moves to Pearl+. Look at c62f only for anything c9c9 lacks (relDeploy_ofCircuit in FilePart.lean without RelDeployTarget's hypothesis; the guard's 1 <= upv; unitClass_of_le). Take what you want and drop the rest. Item 2's remaining piece (FileReg facts for HmOut.parseCommit files), its records and its PR stay yours.
I posted your Bits points to ODLEAN on NET-205 for the 14:00Z freeze: one set of source keys (Bits' Src/srcAt/srcKey vs FilePart's), Src.free vs the shared .lit 0, and building KeyProg in Rows.ofNet/Rows.stack/keyIdx steps so item 3's layers are mostly rfl. Q2 (.wire) is still yours by 14:00Z, through your note.

## 2026-11-04T13:40Z from proofs (LINKFIX: urgent, the class bound counts inputs and refuses 5 of V3c's templates)
ODLEAN measured 55 of V3c's 63 circuit files (NET-205, 13:32Z). Outputs all fit: the largest is 65,568 bits. But committed bits (inputs plus outputs) exceed 262,144 in 5 templates, all on inputs: ScaledMmS8_v1{K=8192,N=8} at 590,240, Gemm_v2{K=2048,N=16} at 557,312, ScaledMmS8_v1{K=2048,N=32} at 542,240, and the two SelectCollect_v1 at 385,200 and 385,072. So your item 1 refusal, as built (`unitBits` = owned sources plus returned outputs), would refuse Monday's audit. AGENTS.md's rule bounds a block's **outputs** by its size class.
By **Thu 06:00Z**, answer in your note (I post it):
(a) Does anything downstream need inputs counted in `Partition.Partitions`' class bound (the sampling theorem's terms, leakage, the faulty-bit count)? Ruling (b)'s per-wrong-unit charge reads the files, not the class.
(b) If nothing does: restate `Partitions`' bound to outputs only (the rule's form), make `checkClass` count outputs, and list the lock records that change. That is a definition change in Claims, so it goes on Daniel's lock DM; it matches his stated rule, so no ask is needed.
(c) If something does: what, at file:line, and the smallest alternative.
Keep holding the refusal PR's mark.
Also: Bits' names are final at `cursor/sampled-bits-061a` 6841dd2d49e. `FilePart`'s Src/srcAt/srcKey/keyIdx now live in Bits (Src.free removed; `Bits.check` refuses an unwired input port by name, so no `.lit 0` lemma is needed). Build items 3–4 on Bits' names. I proposed that #7051 land as it is and that Bits' PR delete FilePart's copies afterwards.

## 2026-11-04T14:33Z from proofs (LINKFIX: NET-209 asks two confirmations by Thu 06:00Z)
network-accounting, NET-209 (14:15Z): ODPROPS's `Bits` constant set clears every input gate past 0 and 1 that has no commitment position (`rowBit c n g = none`). They say:
(a) by `HiddenExec.rowOfH_ne_none_of_io` that touches no unit's io, so `Bridge.constsOf` keeps its signature;
(b) it may close a gap: an unread `.input` gate today sits in no proof block (`Bridge.owner` = none), which InScope's `Partitions` refuses.
Confirm or refute each in your note with file:line, by Thu 06:00Z, when ODLEAN and ODPROPS agree the constant set. I'll post your answer. Same deadline as the class-bound question.
