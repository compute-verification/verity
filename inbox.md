[2026-11-16T20:10Z] 587407064c7c
TRN-177 map (PAGE, worker). Read-only, at main 5f7249690586. Marks: [read] I read it, [ran] a script or grep I ran counted it, [inferred] from docstrings or names, not proofs.

## First, for review-77: #9519 and #10383

- **#9519 is classed RESTATE, not delete.** [read] It proves and lists `Slowdown.Cut.Guarantees.HECharged` (with `_honest`, `_nontrivial`), adds `Slowdown.Cut.ReadsOn`, and replaces main's `CountsOn`, `real_step_charged`, `countsOn_rows` and `HECharged_of_countsOn`. It is the count over a set of blocks, the rest charged whole, under (H) and (E): a join to the period over the page's definitions, which TRN-177 keeps. Its restatement is only check 2's rename. Recommend it lands. Readers [ran]: `min_cut.py`'s `bound_charged`, `inference_only/tied.py`, `test_one_slowdown.py`'s `SIBLINGS`.
- **#10383 is classed UNSURE, leaning delete; recommend review-77 hold it.** [read] It adds a named assumption, `Slowdown.Assumptions.PeriodUntiedAtMost`; `HEChargedUntied` (`_honest`, `_nontrivial`); `UntiedWords`, `RegisteredCircuit.untie`, `wordsRead`, `OneReader` moved into Definitions; `readsOn_untie*`, `card_charged_le`; and three files (`Assumptions/Untied.lean`, `Untied.lean`, `Proofs/Slowdown/CutUntied.lean`). Question: does the page's statement need untied words? T1 holds for a registered word only at β = 0, so without untying the theorem says nothing about a fine-tuned registered model's registered words. If Daniel wants that case, restate over the page. If not, delete it with main's `Proofs/Slowdown/Untied.lean` (863 lines) and `CutWitness.lean`'s untied toys (lines 200 to 490).
- **#10518: check 5 keeps its structure and replaces its openings.** [read] Its shape (what the thing is, one thing to try) is right, but its openings say "the min-cut bound", not the page's terms. Its three DESIGN.md files (824 lines, moved unchanged) carry the older bounds check 3 deletes. Recommend not landing it as is: check 5 takes its skeletons and drops the DESIGN.md files.

## 1. Lean modules (109 files, 15,862 lines with the `Proofs/Slowdown.lean` aggregator; top's 108 and 15,813 exclude it) [ran]

| class | modules | files | lines |
|---|---|---|---|
| keep: the page's theorem | D:`Cut`, `Cut/Toy`, `Guarantees/Cut`; P:`Cut`, `CutWitness`, `LoomisWhitneyBase` | 6 | 2,126 |
| restate | P:`CutCharged` (#9519); the aggregator `Proofs/Slowdown.lean`; `Covering.lean` (`FeedsWithin` and `InferenceBlock` move into `Cut`; the rest goes) | 3 | 335 |
| unsure | P:`Untied` (#10383's question) | 1 | 863 |
| move out of the area | `Guarantees/Actual.lean` (the physical model, to ResourceAccounting, @lean); `Dark.lean` (`PartialContainment`, read by `InferenceOnly.Assumptions.ThetaRegistered`) | 2 | 142 |
| delete | formalism v2 13/1,804; Capability 15/1,363; Coverage, CoverageToy and Tasks 5/358; the rest of Covering and CoveringTwoCosts 8/911; the cover theorem with BitView 12/1,130; Threads, Layered, Fit and Piece 21/2,591; the tiers (`Slowdown`, `Remote`, `Escape`, `RemoteMonitoringPeriod`) 11/2,130; ServingBlocks 6/1,037; `Guarantees/Actual/{Slowdown,Threaded}` 2/257; Factor, Witness and the Witnesses 4/815 | 97 | 12,396 |

- **The core does not stand alone.** [ran] D:`Cut` imports `Covering` (`FeedsWithin`), P:`CutWitness` imports P:`Untied`, and D:`Guarantees/Cut` imports the root `Definitions/Assumptions.lean`, which also declares formalism v2's assumptions.
- **`Guarantees/Remote` and `RemoteMonitoringPeriod` are unsure.** [inferred] They compose `CountHE` with PoUW and SPOT, a join, but `Remote.lean`'s definitions are the tiers'.

## 2. Lock guarantees

- **Top's parts sum to 103, not 108.** [ran] 12 + 20 + 3 + 68 = 103, my count of `Slowdown.*` in `security/lean-audit.json`. 108 is the file count. Nine more, `ResourceAccounting.SecurityProofs.ServingBlocks.*` (@training), are proved in the area. Five `ResourceAccounting.SecurityProofs.Actual.*` (@lean) read it.

| group | n | owner | assumptions | class |
|---|---|---|---|---|
| Cut | 12 | training | Hardness, Exports (`HE`, `HEShared`, `CountHE`) | keep, renamed |
| formalism v2 | 20 | training | CutCost, TokenWork, EfficiencyFloor | delete |
| cover | 3 | training | none (open premises) | delete |
| the 68: Capability 18, Coverage 3 | 21 | training | none | delete |
| the 68: Covering, CoveringTwoCosts | 2 | compute-accounting | none | delete |
| the 68: Threads 10, Layered 3, Fit 2, FitRate 1, Slowdown 4, Remote 3, their witnesses 19 | 42 | lean | `SlowdownRemote`: Hardness, Exports | delete; Remote unsure |
| the 68: RemoteMonitoringPeriod | 3 | training | Hardness, Exports | unsure (a join) |
| ServingBlocks | 9 | training | none | delete |
| Actual: ServerBlocks, ServerBlocksAt / SlowdownServerBlocksCap(+Slots), ThreadedRate | 5 | lean | none | move / delete |
| Properties: `CheckedCoverTraining`, `DescThrottles`×4 (training), `Throttles`×4 (lean) | 9 | mixed | none | delete |

- **The check's nine listed properties join it to older bounds.** [read] Its join to the page is `complies_countHE` (`Proofs/Properties/InferenceOnly/Count.lean`), unlisted. Recommend listing it and deleting the nine.

## 3. Glossary [read]

- **Restate in the page's terms:** in `ARCHITECTURE.md`'s Glossary, "(c,k)-partition and inference block", "Credit and the slowdown lower bound" and "Server block's budgets and yields" (q_S and q(c, k) go). In `GLOSSARY.md`, the "Registered or not" rows naming `Slowdown.Slowdown.residualShare`, and the "Blocks, yields…" rows naming `ThreadedReads` and `Slowdown.Guarantees.Slowdown` (not every row read).
- **Retired rows to add:** (H) and (E), "min-cut bound", "bound of record", formalism v2, the cover theorem, the thread bound, capability floor. Serving-block bound, exclusion bound, coverage and dark are already retired.

## 4. READMEs [ran byte counts]

- **Restate:** `benchmarks/slowdown` 25,318 (its `min_cut` part), `benchmarks/training_slowdown` 49,972, `benchmarks/inference_only` 22,448, the inference-only verifier's 14,678. **Delete:** `benchmarks/nci` 28,470 (thread and tier calculators), `Capability/README.md` 13,875.
- **Three more, not in top's five:** `security/README.md`'s slowdown section (about lines 80 to 300), `benchmarks/circuit_properties/README.md` (names `Cover`), compose's `verifier/remote/README.md` (46 mentions, `SlowdownRemote`, Capability).

## 5. Calculators and code

- **Keep `benchmarks/slowdown/slowdown/min_cut.py`.** [read] It is the one function `tests/test_one_slowdown.py` enforces, with `HE`, `HEShared`, `HECharged` and a general F; `calculator.py` and `inference_only/policy.py` feed it.
- **Delete `training_slowdown/formalism.py` (1,082 lines).** [ran] `capture/circuit.py`, `capture/products.py`, `grade.py` and six tests import it; what they take is unread.
- **Delete `theorem.py` (the exclusion bound) and what only it reads.** [ran] `classes`, `envelopes`, `scenario`, `figures`, `analysis` and `benchmarks/nci` import it.
- **The inference-only check's code stays.** [ran] It imports nothing of `security/`.

## 6. Who else reads [ran unless marked]

| reader | what | tell |
|---|---|---|
| `security/Properties/lean-audit.json` | the 9 properties | lean |
| `verity/commitments/claim_ids.json`, `tests/test_claim_ids_table.py`, `benchmarks/slowdown/data/premises.toml` | 5 training ids (formalism's 3 go) | lean |
| `tools/verity/lean/unread-caps.json` | `Proofs/Slowdown`: 1 module, 4 declarations; shrinks | none |
| `Definitions/ResourceAccounting{,/Guarantees}.lean`, `Proofs/ResourceAccounting{,/Actual/*}.lean` | ServingBlocks, Tasks, Coverage, Actual, Threads | compute-accounting, lean |
| `Proofs/InferenceOnly/{Lemmas,ReadsIn,Verdict,Cover,Cover/Checked}.lean`, `Proofs/InferenceOnly.lean`, `Proofs/Properties/Remote.lean` (uses `FitWitness`'s instances as a fixture) | Threads, Cover, RemoteMonitoringPeriod | lean |
| `.agents/skills/using-slack/registry.json` | owners of Covering, Coverage, ServingBlocks, Tasks, Actual | registry owner |
| `tests/test_lean_vocabulary.py`, `tests/tools/verity/lean/test_audit.py`, `tests/lint/allowlists/repo_retired_terms.json`, `tests/benchmarks/{slowdown,nci}/pyproject.toml` | Cover, Dark, `theorem.py`, area paths | none (ours) |
| compose's `verifier/remote/README.md` | Remote, Capability | compose/remote |
| network-accounting, console (`tools/research/console`), `tests/test_boundaries.py`, `tests/import_order.toml` | no training name found; `store/vocab.py` mentions "slowdown", unread | none |

## 7. Where the Lean differs from the page

| difference | to close | proposed |
|---|---|---|
| names: (H) and (E), namespace `Slowdown.Cut`; the page cites `Nci.Cut.Guarantees.Bound` | rename, `check.py --update --moved` | Wed |
| T1 prices values, one β per array (`Hardness … ![βw, βx]`); the page prices each word | restate `Hardness` over word prices; redo the pair bound | not planned this week |
| the general joining pattern below √2: only `HE` (F = 1) and `HEShared` (2/√3) | `Arrays` at any ω over words | not planned |
| the conclusion: Lean's closed form against the page's ρ ≥ cν/μ with the exact μ | algebra over `Bound` | Wed |
| the training circuit's five conditions are separate hypotheses, not one definition | a `TrainingCircuit` structure; restate `HE` and `HEShared` over it | Wed |
| inference block: the page has \|I\| ≤ c; the Glossary has exactly c gates (`Partition.IsCK` unread) [inferred] | one definition, out of `Covering` | Wed |

## 8. Collisions (recommendations; Daniel decides through top)

- **ρ := nc/|C| against the retired ρ.** Recommend ρ inside training's statement as the page's, with one Glossary row scoping it; 𝓜's resource rate keeps ρ elsewhere.
- **"registered parameters θ" against "registered information θ".** The same set (`RegisteredCircuit.θ`). Recommend the Glossary's word, the README naming the page's once.
- **slowdown(𝒢, 𝒜, η) against nc/|C|.** The Glossary's quantity is `Tasks.lean`'s, classed delete. Recommend the Glossary keep it for a task and call ρ's bound "the slowdown theorem"; `Tasks.lean` waits for the ruling.
- **A fourth: "exactly c" against "≤ c"** (item 7).

## 9. PR order (bottom-up; the lock updated in each; the reading lane told before each)

1. Check 1's README, docs only: today, drafting now.
2. Formalism v2's Lean, a leaf (only `Definitions/Assumptions.lean` imports it): 13 files, 20 guarantees, 3 assumptions, 3 `claim_ids.json` rows. Today. Then `formalism.py`.
3. Capability with its README. 4. The cover theorem with its property.
5. Threads, Layered, Fit, Actual/Slowdown/Threaded, the Throttles properties, `benchmarks/nci` (lean first).
6. The tiers, ServingBlocks, `theorem.py`'s family (compute-accounting first).
7. The moves (Dark, Actual, `FeedsWithin`); Factor and the Witnesses.
8. Check 2's renames. 9. The Glossary. 10. The READMEs, superseding #10518.

- **Today's two marks can hold:** item 1 is docs, and item 2 is a leaf, if the remote Lean check passes first time.
- **Checks 2, 4 and 5 can hold by Wed; check 3 only if the unsure items are ruled by Tue and lean agrees to its 42 deletions.** Otherwise items 5 and 6 slip to Thu.
- **Friction:** check 2 wants a day per table line, and the tree takes no dates. The README says planned or not planned; days stay in Linear.

**Not read:** `benchmarks/slowdown`'s code beyond two docstrings; `benchmarks/nci`'s code; what formalism.py's importers take; `Partition.IsCK`; ARCHITECTURE beyond a grep; a compute-research directory (none found).

