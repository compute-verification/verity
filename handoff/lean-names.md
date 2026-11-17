# TRN-177 check 2, Lean names: the map for lean's pass

From PAGE. This scratch branch is never opened as a pull request. It is the map for the Lean-names window (lean's one pass for all lanes, with the lock's moved records): the names of `Slowdown.Cut` in the page's words, old name to new, for the kept modules only.

Sources:
- the page's names are its statement, `security/Definitions/Slowdown/README.md` on #10553 (`165ee26c49e`);
- the Lean is `main` at `c6be0556609`, after today's deletions: #10564, #10687 and #10703, read at #10703's head `3438c2ec93a`;
- #10597, #10611 and #10613 touch none of the files below except `premises.toml`, whose rows move up and stay the same.

Marks: [read] I read the uses; [ran] counted; [inferred] from a name or one use.

## The kept modules

- `security/Definitions/Slowdown/Cut.lean`, `Cut/Toy.lean`, `Guarantees/Cut.lean`, and `Definitions/Assumptions.lean`'s `Slowdown.Cut.Assumptions`.
- `security/Proofs/Slowdown/Cut.lean`, `CutWitness.lean` and `CutCharged.lean`.
- `security/Proofs/Properties/InferenceOnly/Count.lean`, which reads `Work`.
- `security/Definitions/Slowdown/Covering.lean` for `FeedsWithin` and `InferenceBlock`, which `Cut` reads.

`Proofs/Slowdown/Untied.lean` and `CutWitness`'s untied toys are a draft deletion (#10713) that waits on Daniel's page. If it doesn't land, they are a kept module too, and their uses move with the same renames (counted below).

## The map

One row per definition of the page, with every Lean declaration of it.

| The page's definition | The page's name | Lean on `main` | New | Mark |
|---|---|---|---|---|
| Word | word, its width | `Finset (Fin C.N)`, no declaration | none | no Lean declaration today. The difference table has no row for it and needs none: a word is a set of wires in both |
| Cut | a set *cuts* A from B | `Slowdown.Cut.Separates C Z X Y` (Z cuts X from Y) | `Slowdown.Cut.Cuts` | rename |
| Cut | κ(A → B) | `Slowdown.Cut.kappa` | none | keep (the page's own name, κ) |
| Cut | κ_θ of a set of words | `Slowdown.Cut.kappaWords` | none | keep (the page's own name, κ_θ, over words) |
| Cut | the unregistered inputs, the inputs outside θ | `Slowdown.Cut.unregisteredInputs` | none | keep (the page's own name). The retired-terms lint counts the identifier as "unregistered input" [ran], and the allowlist keeps it where it stands. Top's Glossary pull request decides the word, and a rename to the Glossary's word would contradict the page |
| Matmul structure | 𝓜, a matmul structure, a set of matmuls | `Slowdown.Cut.Work C ι` | `Slowdown.Cut.MatmulStructure` | rename. Its namespace moves with it: `Work.N`, `.O`, `.pi`, `.mac`, `.Mac`, `.Out`, `.out`, `.y`, `.wWords`, `.xWords`, `.allW`, `.allX`, `.inside`, `.fed`, `.mac_disjoint` keep their last part |
| Matmul structure | a matmul | `Slowdown.Cut.Matmul` and its fields `w`, `x`, `y`, `mac`, `reads_w`, `reads_x`, `feeds_y` | none | keep (the page's own name: weights w, activations x, outputs y) |
| Matmul structure | W_m, X_m, the weight and activation words | `Work.wWords`, `Work.xWords` | `MatmulStructure.wWords`, `.xWords` | keep (moves with the structure) |
| Matmul structure | N, O | `Work.N`, `Work.O` | `MatmulStructure.N`, `.O` | keep (moves with the structure) |
| Matmul structure | E(A), the exports of A | `Work.clear` | `MatmulStructure.exports` | rename: T2's "exports are wide" names them |
| Matmul structure | b, the export width | `Work.bY` | `MatmulStructure.exportWidth` | rename |
| Matmul structure | π(S) and Π_θ(k), the pairs within reach | `Work.pi`, `Slowdown.Cut.piMax` | `MatmulStructure.pi` | keep (the page's own names, π and Π) |
| MAC | a MAC M^m_{i,j,l}, a connected set of gates | `Matmul.mac`, `Work.mac`, `Work.Mac`, `Slowdown.Cut.Connected`, `Work.mac_disjoint` | the `Work.` ones under `MatmulStructure.` | keep (the page's own name) |
| Inference block | a (c, k)-inference block | none in `Slowdown.Cut`: a part of Core's `Verity.Partition` meeting `Verity.Partition.IsCK`; also `Slowdown.Covering.InferenceBlock`, on subcircuits | none | no Lean declaration today (difference table: "an inference block is also `Slowdown.Covering.InferenceBlock`") |
| Inference block | each block reads at most k bits outside θ | `Slowdown.Cut.ReadsAtMost d P k`; with a set of blocks charged whole, `Slowdown.Cut.ReadsOn` | none | keep (no page name: the k half of an inference block, over every part) |
| Inference circuit | a (c, k, n)-inference circuit | `InferenceOnly.CompliesAtMost c k n` | none | keep: the inference-only policy's verdict, not `Slowdown.Cut`'s, and the page names it as it is |
| Property T1 | T1 (operands are hard) | `Slowdown.Cut.Assumptions.Hardness` | `Slowdown.Cut.Assumptions.OperandsHard` | rename; moved record, statement unchanged |
| Property T2 | T2 (exports are wide) | `Slowdown.Cut.Assumptions.Exports` | `Slowdown.Cut.Assumptions.ExportsWide` | rename; moved record, statement unchanged |
| Training circuit | a (k, q, ω, β_y)-training circuit | none: five separate premises of `HE` and `HEShared`. Conditions 3 to 5 are `Slowdown.Cut.NoSharedPair` (with `ValuesInjective`), `Slowdown.Cut.Arrays` and `Slowdown.Cut.CostsAtLeast` | none | no Lean declaration today (difference table: "the training circuit is five separate premises"). The three conditions' names are kept: the page doesn't name them |
| The theorem (training slowdown) | ρ ≥ cν/μ, F = 1 and F = 2/√3 | `Slowdown.Cut.Guarantees.HE`, `HEShared`; its forms `CountHE` (on `CompliesAtMost`) and `HECharged`; `alphaHE` (α), `shareF` (F), `workHE`, `workHEOn` | none | keep (the page's own name: its statement names `HE`, `HEShared` and `CountHE`) |
| Lemma (general bound) | the general bound, ρ ≥ cν/μ_* | `Slowdown.Cut.Guarantees.Slowdown`, `Slowdown_nontrivial`; the proofs `Slowdown.Cut.SecurityProofs.Slowdown`, `Slowdown_nontrivial`; α_* is `Slowdown.Cut.alpha` | `Guarantees.GeneralBound`, `GeneralBound_nontrivial`, and the same under `SecurityProofs`; `alpha` keeps its name | rename. The difference table's row on the weaker closed form stays: the name is the page's lemma, and the Lean still states a weaker form of it |
| Lemma (MAC count) | the MAC count, N ≤ n(k + α_*) + √(Π n O) | `Slowdown.Cut.Guarantees.Bound`, `Bound_nontrivial`, and the same under `SecurityProofs` | `Guarantees.MacCount`, `MacCount_nontrivial`, and the same under `SecurityProofs` | rename |
| Lemma (T1 bounds the pairs within reach) | Π_θ(k) ≤ F²k²/(4q) | none: proved inside `HE` and `HEShared` (`Proofs/Slowdown/Cut.lean`) | none | no Lean declaration today (difference table: "the lemma on T1 …; not planned") |
| The toys' joint witness | none | `Slowdown.Cut.Guarantees.Cut_honest`, the honest witness of `Bound`, `Slowdown`, `HE` and `HEShared` | none | keep (no page name; it is the namespace's) |

The difference table's first row, "T1 and T2 are named (H) and (E)", is closed by this pass and is deleted with it.

## How many uses move

These are word matches in the kept Lean files at `3438c2ec93a`, so tactics and prose inflate them [ran]:

| Name | Uses | Files | Of them in `Untied.lean` |
|---|---|---|---|
| `Hardness` | 40 | 7 | 6 |
| `Exports` | 27 | 6 | 2 |
| `Separates` | 11 | 4 | 2 |
| `Work` | 89 | 9 | 8 |
| `.clear` | 43 | 11 | 0 |
| `bY` | 41 | 4 | 4 |
| `Bound` | 13 | 3 | 0 |
| `Bound_nontrivial` | 3 | 2 | 0 |
| `Slowdown_nontrivial` | 7 | 4 | 0 |

- `.clear`'s 43 include other areas' own `clear` fields (Flock, PoUS, Compose). `Work.clear` is used in `Definitions/Slowdown/Cut.lean`, `Definitions/Assumptions.lean`, `Proofs/Slowdown/Cut.lean` and `CutWitness.lean` [read].
- `Separates`' four files are `Slowdown.Cut`'s. Three other `Separates` are other areas' and are not renamed: Flock's audit `Separates`, a downstream cut (`Proofs/Flock/Soundness/Audit/Influence.lean`); Core's device boundary in `Definitions/Core/Blocks.lean`; and `Proofs/Flock/Claims/RootFindAlg.lean`'s.
- `Slowdown.Cut.Cuts` has two neighbours, `Slowdown.Covering.MulBlocks.Cuts` and WeightSecurity's `Cuts`. Both are reached only by dot notation, and no file opens `MulBlocks` [ran], so there is no clash [inferred].

## The spellings that move in the same commit

Each line is a spelling outside Lean of a renamed name. A Lean rename whose lock record moves and whose spelling here doesn't leaves a dangling name.

- **T1, `Hardness` → `OperandsHard`:**
  - `benchmarks/slowdown/data/premises.toml`, `[premise.training-cut-hardness]`'s `lean` (line 116 on `main`, line 8 after #10597);
  - `verity/commitments/claim_ids.json`, `training-cut-hardness`'s Lean column (line 67); the id stays;
  - `benchmarks/slowdown/slowdown/min_cut.py` line 70, `AREA = HARDNESS.removesuffix(".Assumptions.Hardness")`, whose suffix becomes `".Assumptions.OperandsHard"`. With the old suffix, `AREA` would be the whole name, and `HE`, `HE_SHARED` and `HE_CHARGED` (lines 71 and 73) wrong with it.
- **T2, `Exports` → `ExportsWide`:**
  - `premises.toml`, `[premise.training-cut-exports]`'s `lean` (line 123 on `main`, line 15 after #10597);
  - `claim_ids.json`, `training-cut-exports`'s Lean column (line 66).
  - `min_cut.py` reads `EXPORTS` from the row and spells nothing.
- **`tests/benchmarks/slowdown/test_lean_names.py`** spells no Lean name (its `ALLOWED` is empty). It is the test that fails a missed pair: every name the calculator cites, `premises.toml`'s rows and `min_cut`'s built names among them, must be one the locks list. So the rows and the lock's moved records go in one commit.
- **The docs, same commit:**
  - `security/Definitions/Slowdown/README.md`: T1 at line 68; T2 at 84; `Separates` at 25; `Work` and its fields at 35, 38 and 39; `clear` at 40; `bY` at 121 and 150; `Guarantees.Slowdown` at 130; `Guarantees.Bound` at 133; and the difference table's first row.
  - `security/README.md`'s guarantee table, which after #10553 names T1 and T2 (line 104).
  - `verity/protocols/compose/verifier/remote/README.md` line 508 after #10553, line 511 on `main`: T1 and T2.
  - `benchmarks/slowdown/slowdown/min_cut.py` line 8 and `benchmarks/slowdown/slowdown/profile.py` line 54, docstrings that cite `Work.N` and `Work.O`.
- **`Bound`, `Slowdown`, `Separates`, `clear` and `bY`** have no spelling in Python, TOML or `claim_ids.json` [ran]. The calculator's `Bound` is `min_cut.py`'s own class (line 154), and every other `Bound` in Python is prose about it or about other areas [read].
- **Left as they are:** `tools/move/slowdown_namespace_moves.json` and `slowdown_area_map.toml`, the records of earlier renames, and `tools/research/src/research/store/moves.json`, the store's.

## The noes

- **`HE` and `HEShared` stay.** The page's statement names them; `CountHE` and `HECharged` are their forms and stay with them.
- **The namespace `Slowdown.Cut` stays**, not the page's `Nci.Cut`: "NCI" is retired, and the page's citation is what to fix.
- **The claim ids `training-cut-hardness` and `training-cut-exports` stay.** Locks and credit records name them, and a new id would be a second record of one assumption. Only their Lean column moves.
- **Core's names stay:** `Verity.Circuit`, its gates, `Verity.Partition`, `Partition.IsCK`. Every protocol reads them.
- **`InferenceOnly.CompliesAtMost` stays.** It is the inference-only policy's verdict, read by its verifier package's properties (`DescCountsCK`), not `Slowdown.Cut`'s.
- **No new declaration:** word, inference block, training circuit and the third lemma get none in this pass. Each is in the difference table or needs no row, and a structure for the training circuit is planned work, not a rename.
- **`(H)` and `(E)` in Lean docstrings** become T1 and T2 with the names, in the same files. That is prose in Lean, so it goes with this pass, not today's window.
