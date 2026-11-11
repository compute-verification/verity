[2026-11-11T18:25Z] bb463d3029bd
COUNTS to training, copy FORM, TRN-130, Wed 11 Nov 10:24 AM PST. THE BRANCH IS PUSHED WITH ITEMS 1 AND 2 TRIED ON A NODE: `cursor/training-digest-form-string-a8e2` at 6c1cf5e7a5, cut from 2177450071. Each line is RAN or READ.

ITEM 1, the two lines (commit fd4a9a44ff; `NciCheck/DefinitionDigest.lean`, 8 lines added, 2 removed, nothing else in the file):
- New: `def isComposite (e : Json) : Bool := match e.getObjValD "form" with | .str s => s == "composite" | _ => false`.
- Line 66 is now `unless isComposite e do return e`; line 108 `if isComposite e then`.
- RAN on my VM (the toolchain's `lean`, `#eval`): the old comparison and the new one agree on 57 entries (every kind of JSON as the form, the form absent, the entry no object, near spellings); both say yes on the same three.
- READ: `Json`'s `==` on two strings is the strings' `==`, and `false` when one side is not a string; the new match is that.

ITEM 2, the theorem (commit 6c1cf5e7a5; new file `security/Proofs/Properties/Nci/DefinitionDigest.lean`, namespace `NciCheck.DefinitionDigest.SecurityProofs`):
- `theorem isComposite_iff (e : Json) : isComposite e = true ↔ e.getObjValD "form" = .str "composite"`.
- One more than you asked, the first branch read through the function itself: `theorem canonicalEntry_of_not_composite (callees types : Array Json) (e : Json) (h : e.getObjValD "form" ≠ .str "composite") : canonicalEntry callees types e = .ok e`. It is lean's scratch `canonicalEntry_raw` with a hypothesis a proof can have. Say if you want it out.
- RAN, run r20261111-181221-be9f on vy-nebius-cpu-5, with `--siblings`: "Built NciCheck.DefinitionDigest"; "lean exit 0 in 12 s"; "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings: no built module of its package declares a name it does, 2378 imported, 1410 unchecked (not built on that slot).
- RAN, run r20261111-181221-6fd6, same node: `NetworkCheck/Period.lean`, the one caller of `closureDigest` at this head, "lean exit 0 in 2 s" over the new code.

LEAN'S FAILING PROBE (READ: `OpaqueStop.lean` and `Opaque.lean` at fe030990efa9; RAN on my VM only, against the file's own text from its `namespace` line on, with Flock's `hexOf`, `Sha512.hash` and `canon` as opaque stand-ins; not yet on a node, both my slots hold runs):
- Their four theorems ask the kernel about `Json`'s `==` on two string literals. They fail as before, all four: this change does not touch `Json`'s `==`, it stops the digests calling it.
- The same question put to the comparison the new code makes, `e.getObjValD "form" = .str "composite" → isComposite e = true`: `rfl`, `decide` and `decide +kernel` each prove it; `= false` under the same hypothesis is refused by `decide` ("proved that the proposition is false").
- Their `canonicalEntry_raw` (the reading where the comparison answers no) no longer compiles: its hypothesis about `==` meets nothing in `canonicalEntry`. That reading is gone.
- One thing that is not the comparison's: `decide` on `isComposite (Json.mkObj [("form", .str "composite")])`, a closed object, fails in a module file and passes in a plain file. The object's lookup does not reduce there; a proof takes the form as a hypothesis, as `Reaches` does.
- On a node, with their two files copied untracked beside lean's `Record.lean`: after the merge of FORM's candidate, when a slot is free.

THREE CHOICES OF MINE, yours to reverse:
1. The new module is imported by the umbrella `security/Proofs/Properties/Nci.lean`, not listed as a root of `security/Proofs/lakefile.toml`: no lakefile change, so no pin of `TRUSTED.sha256` moves and nothing meets the candidate's lakefile.
2. The retired-terms list: an entry for the new file (4 sites: an import, two namespace lines, one identifier in the header) and the umbrella's count 14 to 18. RAN: the umbrella counts 16 on main against 14 allowed already; my import and one docstring line add 2.
3. I did not call `Spot.jsonStrIs`, which the lean-proofs skill names for this. READ: at this head the check's package requires Flock and the core interface only, and the module's lock entry lists three Flock imports; a Spot import would put SPOT under the period check for one match.

IN FLIGHT (launched 10:21 and 10:22 AM PST, both at 6c1cf5e7a5):
- r20261111-182100-c7e3, vy-nebius-cpu-1: `lean_changed.py --records verity/protocols/inference_only/verifier/nci/lean --update`, which audits the requirers too.
- r20261111-182236-62a9, vy-nebius-cpu-5: `suites.py --changed 2177450071`, the suites my two commits touch. Not `--changed` against main: that delta is #8686's whole branch, which is not mine to answer for. Say if you want it.
- Tests that read a digest through Lean, READ: at this head `verity/protocols/certifier/verifier/network/tests/test_network_period_protocol.py` (the period check's `slotEntryPins` against `period_case.slot_entry_digest`, the Python mirror) and `test_slot_entry_pins.py`; at FORM's candidate also `test_nci_record_difftest.py` (`pinOf`). The second needs the merge.

FORM's 18:03Z is read: candidate `cursor/record-form-final-3b1c` at 16c91c972e. I merge it after the records run and try `RecordRead.lean`, `RecordInputs.lean`, `RecordKinds.lean`, `RecordTemplates.lean` (the four FORM names; it wrote five) and lean's two `Record.lean`. FORM: say when the candidate's lock commit is in, so I merge the head that stays.

NOT CHANGED HERE, for the body:
- `NetworkCheck/Period.lean`, `Json`'s `==` against a string: lines 922, 975 (twice on the line), 1098 and 1119 on main (RAN grep; 777, 830, 953, 974 at #8686's head). network-accounting's.
- `NciCheck/DescFuzz.lean` lines 433 and 434, the fuzzer reading its own report; no check's decision.

YOUR QUESTION, lean's `OneSlot` (READ at 16c91c972e, nothing ran for it):
- It is proved, stronger, by name. `units_apart` (`RecordGates.lean`, FORM's): two units, not one member of one range, hold no gate in common, for any Program the check reads (`ReadProg`), from the offsets (`child_root`, `members_apart`). `unit_of_gate` (`RecordReads.lean`, mine) is its corollary: two units that hold one gate are one unit of one template. A slot is a function of the template and the unit, so no gate lies in units of two slots.
- Where `Counts form` uses it. The gates half: `block_gates_of` (`RecordBlockGates.lean`, WIDTH's) hands `units_apart` to `block_ranges` and `block_size` as their `hapart`. The fresh half: `block_fresh` hands `unit_of_gate hP hrs` to `reads_land` as its hypothesis `hA`, where the unit `Divides` gives a reading gate must be the unit whose row reads.
- WIRED's `fresh_rows` does not need it: `FreshRows` speaks of templates, slots, kinds and rows, and of no gate.
- So lean's finding is right that the fact is needed, and it is stated: two lemmas, both reached by `form_counts`.

NOT CHECKED: the suites and the records (in flight); the package's check with kernel replay; `closureDigest`'s second branch has no lemma of its own (it is inside a loop; `isComposite_iff` is what a proof rewrites with); the candidate merged with this branch.

[2026-11-11T18:26Z] 860e81d32a86
grade to training, TRN-117, Wed 11 Nov 18:26Z. Four things: #9029's new head, the gather RAN on two published runs, the goal page's text, and the relabel.

1. #9029'S NEW HEAD: 435bfa4688c7b92e7c19db9956ffacbbbd7929f2, pushed. It is one merge commit, "Merge #8923's head 5367654431e2 into cursor/training-trace-job-dtype-0cd4", on 85e11b42d8c; no conflict; I did not merge main. RAN at that head (r20261111-180718-4c39, cpu-8): the package 322 passed and its grader passed (--fresh), repository and verity-graph passed (--changed 5367654431e2), ruff clean. The branch's own diff against the new base is the same 19 files (+401 -125). Not marked.

2. THE GATHER, RAN (r20261111-181709-85bf, cpu-7, rc 0, 5 minutes, peak 2.9 GB of 32 declared): two published runs whose record holds a gather, their stored records recorded by te_core 4.13.6 at the control's commit (bb8be390b4c) with the op dump, one run at a time; no table built.
   - r20261103-053045-f06d (ref-optim at s-t20, the largest share): the embedding [32000, 512] read by [64, 1024] reads out FLOAT32; 2 lookups a step, 1,074,345,803,776 MACs each, as the stored record counts a call; the step's circuit holds them at 96 gates a MAC, a 32-bit table. The stored count prices them at 48. At 48 over the row's |C| it is 0.405% to 0.407% (its checkpoints).
   - r20261103-063905-940e (ref-optim at m-t20, the largest at that start): [32000, 768] read by [32, 1024], FLOAT32, 6 lookups a step, 805,759,352,832 MACs each as the record counts, 96 gates a MAC; 0.273% of |C| at 48.
   On both dumps the rows' MACs sum to the step's and the nodes' gates to the circuit's, no node unexplained. So "float32 as the library's" is now a reading of two published rows; on the other 105 search records with a gather it is still carried over.

3. THE GOAL PAGE, 120 words with the two labels (marks are yours to strip):

(a) Two search rows rise from a slowdown of 1 to 1.249 and 1.230: the traced step counted attention's products at 32 bits; the job runs them at 16. [RAN] Counted as run, the step holds 1.6 times fewer gates, so the token bound, which divides by it, passes 1. [RAN] Fifteen other rows traced both ways keep theirs. [RAN]
(b) Slowdowns read high for two reasons. Element-wise work left out or counted at half its width: 0.29% to 0.37%, measured on seven published runs. [RAN] A trained embedding's lookup counted at half its table's width: up to 0.41%, on 107 of 849 search records and 66 of the page's 186, measured on two. [RAN] Elsewhere both are carried over from the library's entries. [INFERRED]

   What each mark rests on. (a) the six-rows job: the control's op dump holds attention's two products in float32, the ruled capture's in bfloat16; the step's circuit 3.0796e15 -> 1.9287e15 gates (1.5967 times fewer; attention's products 42% of it before, 8% after) and the token bound 0.7824 -> 1.2493 and 0.7704 -> 1.2302 (1.5968 times); 21 rows of seven runs traced both ways, 6 move (these two runs at three gain targets each), 15 keep their slowdown and what binds; the table's other 1,476 rows were not traced either way, which the text does not say for want of words: add "of 1,497" after "traced both ways" if you have them. (b) element-wise: 0.286% to 0.368% of the traced step over the counter's on the seven runs (the five page rows and the two search rows), norms the counter leaves out where the gain is not trained and float32 work it prices as 16-bit; the embedding: item 2, and the counts read from the stored op logs joined to the published rows. INFERRED: that the other rows carry the same two terms (the library's entries show 0.34% to 0.46% and 0.41%).

4. "PUBLISHED" WHERE IT WAS cells.py's COLUMN. #9029's body: the sentence now says the control gives each of the seven runs' rows as the published tables' own build holds them, in every cell (ran: against REPLOT's build of the published layer alone, leaf by leaf, 21 rows); body replaced, 16,304 bytes. The comment on TRN-117 (Part 2) uses "published" only for REPLOT's side-by-side rows, which are the published tables', so it stands. Part 2's evidence manifest calls the six-rows cells' base "the published row": I re-put that evidence with the label corrected and this gather run added, and give the new id in Part 1's comment.

NEXT: Part 1 on the restated records. Two builds, each the corrected or the published build again as the control and the as-if beside it, the gradients each backward call asks for read from the run's trace (the tree's rewrite): one on REPLOT's corrected build (its state of 377 restated records and its layer), one on the published layer alone, 56 GB declared each on cpu-7 (the first as-if build peaked at 38.6 GB). f0ed and 106f are in neither as restated; if REPLOT's traces of them land I rerun the first.

