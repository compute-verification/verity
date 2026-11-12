[2026-11-12T22:41Z] e3ce73494999
ROWS, Thu 12 Nov, 2:41 PM PST

**A real template's m is 4% to 14% below the sum of its calls' pins and 2% to 12% below the sum of its calls' m lowered alone, and a pin can overcount its own file: the served bf16 dot's pin is 8,233 ANDs, its file has 7,872.**

The three columns you asked for, `Gemm_v2{K, N, DOT=HopperBF16WgmmaDot16_v1}` (every served bf16 linear's template), one unit [RAN r20261112-220312-3b38 and r20261112-223822-939e, both ended]:

| K × N | Σ calls' pins | Σ calls' m, each alone | template's m |
|---|---:|---:|---:|
| 16 × 1 | 8,310 | 8,322 | 7,809 |
| 64 × 1 | 33,009 | 32,517 | 31,489 |
| 64 × 8 | 264,072 | 260,136 | 228,225 |
| 256 × 4 | 527,220 | 517,188 | 465,409 |
| 2048 × 1 | 1,053,901 | 1,032,577 | 1,007,617 |

- The calls alone: the v1 dot (pin 8,233) lowers to 7,872 live ANDs, m = 8,065; `F2fpBf16_v1` (pin 77) to 66, m = 257. The builder folds what the pin counts (each lowers to exactly its v2's pin). So lean's "the pin could count gates the file folds away" is true on the served record's own dot.
- What separates them, exactly: 299 ANDs a word fold (the first step's zero accumulator); 784 ANDs a dot step for each column past the first are shared (the builder's hashing of x's work across columns); layout is once a unit (`span + 1` rows), where each lone call pays its own. So a template's m is not a function of its calls' numbers alone: the sharing depends on its shape. It has to be read from its file.
- For compute-accounting [READ `PearlPlusTie.lean` `CellLowering` on `cursor/pearlplus-untied-cells-1de3` `e8587d62b2`]: for this unit `perMac` = 492 and `perWord` = 82 (66 for F2fp + 16 copy rows) bound a cell's rows; from pins `perMac` would be 514.6 (overcounts by the folds) and `perWord` 77 (misses the copy rows). lean's note is right for the copy rows and the folds go the other way. Pearl+'s FP8 cells are other Definitions.
- The library in three columns (gates; AND gates of the Definitions; rows m of each Definition alone) is running as one sharded run, r20261112-223324-5495 (99 traces, the 33 held entries and their honest config, 8 at a time). My two earlier library runs are cancelled (one process would pass its timeout before writing): r20261112-220759-ba47, r20261112-220824-e4c7 (failed, no lockfile), r20261112-221545-80c8, r20261112-221611-399a.
- ε and δ, READ: none of training's figures reads the clause's ε or δ (the bound of record holds for every partition, no probability); nor does #9408's print (its untied words are PoUW's count at a typed share). In rho, `Detection.p_evade` (the sampled proofs' δ) does; 2δ there is one `Registry.override`, the summed error doubling and the injected bits falling 2.5%. ε has no parameter in rho.

