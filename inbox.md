[2026-11-11T18:25Z] f1302df7a145
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

