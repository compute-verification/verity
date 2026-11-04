## 2026-11-04T13:38Z from proofs (PEARLDEBIT: heads to merge into #7251)
- PEARLNOISE's `statsAreTheRow`: `cursor/proofs-pearlstats-0d74` at a5256bfe6e (Stats.lean; it also merges PEARLWORDS 7a6bca7fbc and PEARLNUM's latest). Green. Merge it into #7251 when you next merge.
- PEARLNUM's Keys (c1) and Rho: `cursor/proofs-pearlnum-1283`, head SHA in its note once its try ebbc is green.
- Coming: GSJOINT's Draw.lean, PEARLWORDS's Words, PEARLNOISE's Peel.lean (the peel and quant8 forming, which Words consumes).


## 2026-11-04T15:10Z from proofs (Pearl+ words: "debit" is now "unnecessary work")
Daniel (6:49 AM PST, via compute-research): Pearl's "debit" is now **"unnecessary work"**: the multiply-adds a tile can leave out with its output bit-for-bit unchanged. The cap check is **"the unnecessary-work bound"**, and rho is **"the unnecessary-work share"**. Incorrect work, which sampling bounds, is separate. Use the new words in prose, docstrings, PR bodies and notes now. **Don't rename identifiers** (`debit_units`, `debitOf`, `capOK`, `Device.cap`, `DebitIsTheReplay`, `Debit.lean`, `TilePassGivesTheCap`) until after the Pearl+ run of record. The glossary rows are on CMP-310's PR 2.
