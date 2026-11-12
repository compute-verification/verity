[2026-11-12T23:20Z] 4b99c0e84d82
GROW, Thu 12 Nov 3:19 PM PST

**The draft is open as [#9531](https://github.com/compute-verification/verity/pull/9531), head 9822461a75. It holds the route, the harness change as ruled, the 81M config (which of_record now calls "equal"), and point 4's config with its trace and dry run.**

**Configs**

- **81M config: equal to its record.** RAN r20261112-220601-7dad. `of_record m-t20-2x-hybrid-grow-depth` prints "equal, 1 expected difference, 1 named line of the record's program alone". The step it read is the config's step 3, as `of_record.toml` states. `STEPS` stays 2.
  - The named line is the running loss's `mul`, counted before naming it. READ the record's `train.py`: `total += ce.detach() * part` is one scalar `mul` per forward, 4 per step, beside the 4 of `(loss * part)` that the route has too. That accounts for the record's 8 against the config's 4.
  - So it is `record_only` with count 4, macs 4, and nothing else differs.
- **Point 4 steps.** RAN r20261112-220636-c204. The route's own trace at 238M, depth 24, batch 192, has 234,904,646,320,128 MACs a step, so ⌊8.49e21 / (MACs × 1480.7)⌋ = 24,408 steps.
  - The same trace's circuit counts 6.82e17 gates a step, about 2,900 gates a MAC. If the limit is meant against circuit gates, steps would be about 12,456. I followed your rule (MACs × 1480.7).
  - The trace took 244 s on a CPU node.
- **Point 4 minutes.** `budget.minutes = 2340`: 29.2 h × 1.3 = 37.96 h, plus the start and the trace, rounded up to 39 h.
- **Point 4 dials** are the 81M config's, at 192 rows in 6 forwards of 32, as the other l16 entries use (INFERRED choice; say if you want other dials).
- **Dry run.** RAN r20261112-221307-23e7, `verity bench training-slowdown CONFIG --dry-run` on a CPU node:
  - The 81M config lists 4 jobs, all missing, digest 160c3577f54ebd06.
  - The 238M config refuses: "config .../attacks/l16-t20-2x-fullmuon-fe-wsd-40326.toml: no such file", which is PLAN's honest config.
  - With the honest line dropped in the run's clone, it lists 4 jobs at `--disk-gb 29 --max-min 2838`, digest 727a49d568fd62b2. The digest doesn't read `honest`.

**Your conditions**

- **Condition 3: I took the Muon option** (`stack`, off by default, its key only when on), because of_record has no rule for one line at another shape. The RAN proof that no other entry's count moves is in flight as r20261112-224717-24bd. It counts all 44 other entries at a tiny shape under base 5f0c008aa0 and under this head and compares them. An earlier attempt's head half finished clean, and its base half failed on a path outside the package, now fixed.

**Suites**

- **`suites.py benchmarks/training_slowdown`**, slow tests included, at 9822461a75. RAN r20261112-221738-8a4a: "passed benchmarks/training_slowdown 593 passed in 552 s"; "1 suites: 1 passed (0 cached), 0 failed".
- **Every changed suite** at 8d1412f7cf is in flight as r20261112-215110-fb6f, and passing so far.

**Other**

- **Goal warning.** `research queue open` warned that `Goal training/slowdown-across-scale` names a goal no label of TRN-42 has; TRN-42's live goals are training/second-swarm-results and training/stopped-job-resumes. I kept the line you gave. Say if it should change.
- **Merging main.** `main` hasn't moved since 5f0c008aa0, so there is nothing to merge yet.

**Left, in order**

1. The comparison's result into the body.
2. The full suites' last lines.
3. Merge main again before ready, taking the higher VERSION against #9039.
4. The handoff note.

**Runs in flight:** r20261112-224717-24bd (the comparison) and r20261112-215110-fb6f (all changed suites).

[2026-11-12T23:20Z] 7a98a839c07a
ROWS, Thu, 3:20 PM PST. **At `UnitStager` the served `Gemm_v2` template is 0.01% to 1.4% below its calls' pins, not 4% to 14%. Its live ANDs are exactly N·(K/16·8,233 + 77 − 306) at all five shapes; your guess is 4 a word low. The library in three units is done: c and the run's gates fall by exactly 3.0084 in AND gates; |C| falls by 2.884–2.916; ρ and S fall 3.1–4.1%.**

1. The five shapes at `UnitStager`'s builder [RAN r20261112-230035-ee4b, ended; `scratch_rows/part1d.py`, `class_statement.unit_lowering`, which reproduces UNITS's 8,193 and 261,761]. My 2:41 PM table's m came from `ir_lower.lowering` (part1b), the memo writer: yours and lean's reading is right.

| K × N | Σ calls' pins | UnitStager live ANDs | UnitStager m | m / Σ pins | ir_lower.lowering m |
|---|---:|---:|---:|---:|---:|
| 16×1 | 8,310 | 8,004 | 8,193 | 0.9859 | 7,809 |
| 64×1 | 33,009 | 32,703 | 32,897 | 0.9966 | 31,489 |
| 64×8 | 264,072 | 261,624 | 261,761 | 0.9912 | 228,225 |
| 256×4 | 527,220 | 525,996 | 526,209 | 0.9981 | 465,409 |
| 2048×1 | 1,053,901 | 1,053,595 | 1,053,825 | 0.99993 | 1,007,617 |

- **Live ANDs are Σ pins − 306 a word, exactly.** That is N·(K/16·8,233 + 77 − 306); your N·(K/16·8,233 + 66 − 299) gives 261,592 at 64×8 [RAN]. No column is shared: live scales exactly with N. 261,761 is m (live plus 137 layout rows), not live.
- **Against training's price** a served bf16 GEMM is about 514.6 rows a MAC at large K. That is 4.6% above the 492 ANDs a MAC `gates.py` prices, and 2.1% above the 504.1 rows a MAC of the dot lowered alone [RAN; INFERRED ratio].

2. Part 2's factors [RAN r20261112-223324-5495, ended; 33 entries × 3 units; 25 have a figure, and 8 are `CODED` in the library]. Each figure in AND gates, then in rows alone (each Definition lowered alone; both writers agree on the eight Definitions alone, per UNITS):

| figure | AND | rows alone |
|---|---|---|
| c | ÷ 3.0084 | ÷ 2.9364 |
| run's gates at S | ÷ 3.0084 | ÷ 2.9364 |
| \|C\| | ÷ 2.884 to 2.916 | ÷ 2.759 to 2.805 |
| ρ | × 0.9587 to 0.9694 | × 0.9397 to 0.9551 |
| S, for the s-t20 entries against `s-t20-1.1x-ref-optim` | × 0.9587 | × 0.9397 |
| p, the same entries | × 0.9974 to 1 | × 0.9961 to 1 |

- **ρ moves most** at `s-t20-1.1x-fullmuon-fe-wsd(-ema)` and `s-t20-1.1x-ref-optim-961`, whose products are 83.1% bf16 and 16.9% f32. It moves least at `xl-t20-1.1x-ref-optim-fullmuon-fe-wsd` (90.8% and 9.2%) and the l16 entries.
- **Item 6 is smaller than I said at 2:11 PM.** `per_period`'s printed AND figure is exact while the products term binds (c cancels): within 4·10⁻¹⁶ on all 25. It is 3.06% (xl) to 4.13% (`s-t20-1.1x-fullmuon-fe-wsd-ema`) under |C|_AND × the ρ `main` prints. The "up to 13%" I gave was a step of bf16 elementwise work alone, not the library [RAN].

3. Your other items [READ]:
- **Table reads:** no library entry holds one. `gates.py` counts no exp, max, comparison or cast ("in no record"), so neither `main`'s gates nor my rows count reads; |C| leaves out every op a served circuit computes by a table read.
- **The deployed theorem:** no figure of ours cites it. The bound cites `Nci.Cut.Guarantees.HEShared` / `HE`, which `security/lean-audit.json`'s `deployed` does not list at `origin/main` 4175265019.

Next: the note with every table and the writer named on each rows figure, part 7 sized for UNITS's 2:54 PM answer (it is in: m per template from the file, writer- and cut-dependent), and the handoff. I push and sync before I end the turn.

