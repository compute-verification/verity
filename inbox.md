[2026-11-14T20:52Z] b1b14837f7b0
model: claude-opus-5-5-high; subscriptions removed: none held.

PREMISES, Sat 14 Nov 12:52 PM PST.

**The calculator's retired-term sites are done: `benchmarks/slowdown` goes from 324 sites in 38 files to 76 in 19 by main's lint (166 to 72 by this branch's), and every site left is either another lane's word or listed below with its reason. The last push is `5d9c749de268` on `architecture`, and no printed number changed.**

Sites, before (base f7d87a365505) and after (5d9c749de268), by main's lint, which is the reader's measure (RAN):
- cap 119, Cap 21 and caps 18 go to Cap 3 and cap 1: `CapBelowDepth` and its premise key `cap-below-depth`, both listed per your ruling. The broader cap grep goes from 179 to 18; the 18 are `min_cut.capped` (`benchmarks/inference_only` calls it, and neither lint counts it) and the Lean name.
- legs 27 goes to 0: CALLERS's fanin map took 26, and I renamed by hand the one it missed, `test_fanin_and_fanout_are_derived_from_the_matrices`.
- unmonitored 20 and Unmonitored 3 go to 1 and 1, both listed below.
- warden 14 goes to 0. Warden 7 stays (`WardenParams`).
- These go to 0: uncertified gates 9, faulty_unit 8, reference circuit 3, unregistered input bits 3, unregistered input 2, per-block cap 2, declared circuit 2, the physical model's class 1, δ_link 1, dampener 1.
- These are unchanged, as other lanes' words: link-window(s) 16, drawn/draws 12, SPOT's older name 3.

Names renamed (RAN), all in `tools/move/slowdown_yield_map.toml`, scoped to `benchmarks/slowdown/**`:
- `Cap` becomes `Yield`; a value `cap` becomes `block_yield` (`yield` is a keyword); `caps` becomes `yields`.
- `unmonitored` and `Unmonitored` become `unregistered` and `Unregistered`. This covers the class `envelopes.Unregistered`, `[regime.unregistered]` and the scenario `unregistered-h200-kimi`.
- `warden_rate` becomes `line_rate`; `faulty_unit_bits` becomes `incorrect_block_bits` (a key in `run.py`'s own record, read by nothing else).
- `nci.state_bits_cap`, `nci.rate_cap` and `nci.boundary_bits_cap` become `*_ceiling`: registry ids read only by `families.py` and `test_families.py`, and no figure or metric reads them.
- `classes.output_cap` becomes `output_threshold`.
- By hand, inside one file each: `families._cap` becomes `_clip`, `analysis.py`'s local `cap` becomes `ceiling`, and the test names in `test_theorem`, `test_classes`, `test_scenario`, `test_operating`, `test_min_cut`, `test_attacks`, `test_analysis`, `test_families` and `test_instances`.
- `metrics.toml` keeps its id `training/slowdown-onsite-h200` and gains a title, a unit and an axis. `periods.py`'s unit matches the new unit.

Listed and left, with the reason (READ):
- `families.Nci` and the kind "nci" (9): they follow `benchmarks/nci`'s name, which belongs to that worker.
- `UNIT_CLASS_MAX` and `UNIT_CLASSES` (13): names in verity's profile.
- `WardenParams` (7): network accounting's class.
- κ/2 (6): the FLOP model's products per FLOP. Replacing it would change the model, not just a word.
- `H1J10RCheckedCreditPeriod` (2), `NetworkCheck.Period` (1) and `InferenceOnly.Assumptions.UnmonitoredAtMost` (1): Lean names belonging to other owners.
- `CapBelowDepth` (3) and `cap-below-depth` (1): your ruling.
- A test's guard that the "unmonitored" key is refused (1): kept in the map's [keep].
- `draft:nci-search` (1): a claim id.
- `min_cut.capped`: `benchmarks/inference_only` calls it, and neither lint counts it.

Checks (RAN):
- Every push was diffed with `run.py` and `periods.py` under PYTHONHASHSEED=0, `periods.py` reading a lock patched in a scratch copy. Changed lines per push: 0, 0, 12, 2, 0, 0, 0, 0. In every push, 0 lines had a changed number. The 12 were the scenario's name and the record key; the 2 were the certified-ingress line's words.
- `test_families`, `test_attacks` and `test_classes`: 173 passed and 1 failed, `test_every_class_cites_its_statement_first`, which reads the stale lock and failed before too.
- My first-parent commits touch only `benchmarks/slowdown/**` and my map. I touched no other owner's file and nothing in the certifier quiet window.

Step 5 answers (as sent earlier, READ):
1. 5.01e+12 is the DGX H200 card's `link.rate`: cluster 3.2e12, storage 1.6e12 and management 2.11e11, totalling 5.011e12. The old 4.8e12 left out the management ports on purpose. Counting every port is the safe direction for a ceiling. The low reading (cluster 3.2e12) is unchanged. The old source said BlueField-3, and the card says ConnectX-7.
2. Yes. The check is in `benchmarks/slowdown/tests/test_hardware.py` (`test_the_h200_servers_memory_and_links_are_their_cards` and `test_a_data_file_restating_a_cards_figure_is_refused`), and both pass on my VM.

Not started: TRN-173 and TRN-170.

