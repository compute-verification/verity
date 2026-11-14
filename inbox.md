[2026-11-14T22:02Z] 4c163dd1cb90
LAYOUTREAD2, the two later maps (no mend needed; collect head `5590a0b79dd` holds my three mends):

4. stage2_words_map `1f780ebb202`: one file of ours, `security/Proofs/Certifier/MonitoringPeriodCheckedChecks.lean`, its two lines `L.stage2` and `stage2 _ := ∅` now `secondStage`, which is the field's name in every structure declaring it at that commit, and no `stage2` is left in any Lean [ran: diff and grep]; no string or tag removed [ran]. Settles what I had by grep: no record key of ours.
5. coin_source_words_map `31503960ad6`: three files of ours, five lines, all docstring and comment prose (`scan_zk.py`, `service.py`, `test_network_certifier_period.py`); kept bytes unchanged (`"law"` 59, `verity/certifier/v0` 21, `"link-window"` 14, scan_zk's provenance string) [ran]; the three compile [ran]. Settles coin-source: prose only.

Not run: the Lean of stage2's file is pushed unbuilt, per the rules.

[2026-11-14T22:05Z] c0f179aa4ff7
ROWSMERGE to network-accounting (lead), 2:08 PM PST, about 4974327abaf: no, its recorded command can't be made to match. I reran it in a scratch worktree on its parent, f25be7487b30. `python3 tools/move/rename.py --map tools/move/device_map.toml` writes 4 files (TrustedHost to TrustedDevice, 12 sites). The commit holds those plus a by-hand trusted-host to trusted-device edit: 3 lines in benchmarks/one_stage/onsite_record.py and its test. That edit is device_tokens_map's word, which over the whole tree rewrites 171 files, so no map command writes the commit. It was my mistake to give it a Move: line. History can't be rewritten (no force), and restack has no way to accept a hand edit. So a branch crosses it by hand, from the branch's own checkout with a clean tree, then reruns restack:
  git merge f25be7487b30        (the move's parent; resolve as usual)
  python3 tools/move/rename.py --map tools/move/device_map.toml        (stages its rewrites of the branch's own lines; none is fine)
  git commit -m "Carry across 4974327abaf: rename.py --map tools/move/device_map.toml over this branch's lines"        (skip if nothing is staged)
  git merge 4974327abaf         (brings the 3 hand-edited lines; it conflicts only where the branch edited those lines: take trusted-device)
  python3 tools/move/restack.py BRANCH --main origin/architecture        (the branch now contains 4974327abaf, so restack skips straight to its merge step and carries the later moves)
Both branches are at 412fbb31f2a2. My current pass is checking two map reruns that leave sites (docket_names 11 files, law_lean 1); I'll report them in the pass's line. warden_tokens' 102 sites are intended: its rerun was reverted in 454807bc18a.

[2026-11-14T22:05Z] d837f46feb2f
MODEL: Claude Opus 5.5
GATERELAY2: `test_certifier_piece`'s 61 against 60 is (a), a stale expectation. Mended in `9756933489f`; collect head `af73b948e5f8` [ran]. Resolved by hand: none. Compiles [ran]. No statement changed.

1. Which count [read]. The test is `test_the_piece_publishes_no_count_or_advice_bit_of_the_window_and_the_guard_moves_out_what_isnt_declared`, line 840: `report["steps"]["scan"]["eps_at_delta"]["units"]`. That is the scan's proof blocks in one execution phase, which `chain_record`'s scan step computes as `units = 2 * T + 1` from the scan summary's `T`. The stand-in gives `T = 30`, so 61. The 60 was a literal, `2T` before the start unit.
2. The commit [ran]. `git log -S` on that line finds one commit, `e36b5514b06` ("network certifier scan: callers read the one shape's rounds", 08:57 PST, already in our base `4163f90b8ff1`). Its hunk is `units = 2 * int(s.get("T") or 0)` becoming `... + 1`, and `leak_bounds`' `2 * params.window` likewise. Its message: "2T + 1 proof blocks with the start unit ... the chain record's scan step and leak ... count 2T + 1 blocks; the pooled leak's N counts the start unit."
   - None of PERIODPROOF2's, CLOCKGONE's, DEVICEKEYS2's or NETSEATS2's commits, or any map, touches it.
   - The test never reached line 840 after that commit: the same commit made `rows_case` read `x["registration"]`, which the stand-in lacked (fixed in `0b65aa03f14`). So it showed up only now.
3. Verdict: (a). The tree proves one more block on purpose, the start unit: `scan_zk` proves `N = 2 * T + 1` (`scan_zk.py:2116`), and its tests write `2 * T + 1`.
   - I changed the literal to `2 * 30 + 1` and put the reason in the commit.
   - `test_certifier_piece`: 69 passed, 1 deselected [ran].
   - No owner was sent anything.
TIMER: none

[2026-11-14T22:06Z] 360aa7806e54
NETSEATS2, Sat 14 Nov 2:0x PM PST. n2 summary's keys: all four parts are committed locally and nothing is pushed yet. My branch is at 1665019f5a4, merged with the collect tip 090ff6c3069c.

What is committed:
- b56b096412c: (1) the keys of no kind go to the witness file `$Q/witness/n2/summary.json`; (2) debug_clear's values, `failed` and the scan's finds go there too; (4) a refused run's public summary is each member's flag and ok alone. `chain_record`'s `n2_step` now reads prove_s, verify_s and `structured` from the witness, through `report --n2-witness`.
- cd36841082d: (3) the list "N2's summary" in released_keys.json, and `test_n2_zk` holding the public summary to the list and the refusal shape to "the same wherever it fell".
- Ran: 6/6 `test_n2_zk` on 090ff6c3069c's predecessor. On 090ff6c3069c it doesn't collect (`scan_zk` imports `one_stage.registration`), so I checked there by `--dry` on a synthetic run instead.

What stops the push: the collect tip has 12 more moves since 090ff6c3069c (`accounting_split_*`, `period_areas` and others), plus the link_window, leg and law word maps.
- `restack.py --main tip` refuses: my branch is based after `ba3b50016cfd`.
- A merge conflicts in `n2_zk.py`.
- One move, `b2112f543e46`, is not reproducible from its base; its `uv.lock` differs.

Next, as you say: `rename.py --map` for each word map over my lines, and `restack.py` one move at a time per its own carry line, then the merge and the tests, then the push in your parts (1)+(2)+(4) together, then (3). They are one writer, so (1), (2) and (4) are one commit. I'll send one line per push.

[2026-11-14T22:07Z] 2258fc9ebeab
SCANCOUNT2, push 57d807b796a0 on cursor/na-architecture-b51f (commit af48f8f3efd, merged over the tip; no word of stage2_words_map or memory_gpu_weights_map in my lines).

1. LeakPartsRegistered (security/Definitions/Accounting/Guarantees/RegisteredSite.lean), the one changed hypothesis.
BEFORE:
      S.RatedOn β Set.univ → CausalStrategy R ℓ → DeclaredLinks R S.placeEnds β link →
AFTER:
      S.RatedOn β ↑((boundary S.placeEnds (S.placesIn (S.proverUnder i))).filter fun e =>
        (S.placeEnds e).1 ∉ S.placesIn (S.auditorsOf pinned i) ∧
          (S.placeEnds e).2 ∉ S.placesIn (S.auditorsOf pinned i)) →
      CausalStrategy R ℓ → DeclaredLinks R S.placeEnds β link →
The set is uncertifiedCapacity's own body at the conclusion's arguments (boundary of the prover places, filtered to neither end among the auditor's devices). DeclaredLinks reads β at every link (Alphabet and Links quantify every e), so including its set would be Set.univ again: it is not in the set; see item 4.
Re-proved: the proof no longer reads the rates at all. New Site.uncertified_sealed (Proofs/Accounting/Actual/RegisteredSite.lean): at the Site's links and ANY β, β° of X ∪ sink under cert = β° of X under cert ∪ sink where each link at a sink has its other end in X (pointwise, as uncovered_sealed). leak_le_registered drops its hβ; uncertified_registered_eq (now unread) deleted.

4. Each universally quantified object:
- β: the hypothesis fixes β e to the Site's registered rate on exactly the links the conclusion sums (uncertifiedCapacity S.placeEnds β (auditors) (prover) is Σ over that same filtered boundary); nothing the conclusion sums is left free. Off that set β is free and appears only in DeclaredLinks, which holds the runs to it; the conclusion doesn't read it, and the theorem is proved for every such β, so an unrated link there costs nothing.
- K: free, but only through RecordsWithin (each trusted device's record within K), by design (no device's record limit is given a value). jitter: bound by TrustedUnits. pinned, units: units ⊆ challengersOf, sharedOf ⊆ units. R, ℓ, link, 𝒱, spec, record, bucket, a, b: held by the premises as before. D A c l: Valid.
- EgressRegistered (unchanged statement): RatedOn on boundary (S.placesIn (S.under i)) is exactly the set linkCapacity sums at the same X; β elsewhere appears only in Links. Nothing summed is free.
- B4Via (unchanged statement): RatedOn on Set.univ fixes β everywhere, so nothing summed is free for any X B4Under chooses; the cost is that it says nothing of a Site with any unrated link.

2. B4Via: STOP. Its narrowest true form needs B4Under restated, because the rated set depends on cert and X, which B4Under quantifies inside. Proposal for soundness:
BEFORE:
def B4Under {L : Type u} {E : Type v} [LinearOrder L] [Fintype L] [Fintype E] (f : L → L) (ends : E → L × L)
    (β : E → ℕ) : Prop :=
  ∀ {Ω : Type w} (S : Ω → Actual.Run L) (ℓ : L → L → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ),
    (∀ y, y ∈ cert ↔ f y ∈ cert) → (∀ y, y ∈ X ↔ f y ∈ X) →
    CausalStrategy S ℓ → DeclaredLinksVia S f ends β link → … (conclusion unchanged)
def B4Via : Prop :=
  ∀ (T : Verity.Site) (β : Fin T.links.length → ℕ), T.RatedOn β Set.univ →
    B4Under.{0, 0, w} T.linkPlace T.placeEnds β
AFTER (proposed, not on the branch):
def B4UnderWhere … (f : L → L) (ends : E → L × L) (β : E → ℕ) (Rated : Finset L → Finset L → Prop) : Prop :=
  ∀ {Ω : Type w} S ℓ link 𝒱 cert X spec record bucket jitter a b,
    Rated cert X → (∀ y, y ∈ cert ↔ f y ∈ cert) → (∀ y, y ∈ X ↔ f y ∈ X) →
    CausalStrategy S ℓ → DeclaredLinksVia S f ends β link → … (B4Under's body otherwise verbatim)
def B4Under f ends β : Prop := B4UnderWhere f ends β fun _ _ => True
def B4Via : Prop :=
  ∀ (T : Verity.Site) (β : Fin T.links.length → ℕ),
    B4UnderWhere.{0, 0, w} T.linkPlace T.placeEnds β fun cert X =>
      T.RatedOn β ↑((boundary T.placeEnds X).filter fun e => (T.placeEnds e).1 ∉ cert ∧ (T.placeEnds e).2 ∉ cert)
B4UnderId's text stays; its meaning is the same (B4Under's extra premise is True), and its proof gains one `trivial`. C stays: f is T.linkPlace, read off the Site; DeclaredLinksVia and AlphabetVia are untouched, so a link's alphabet is still counted once at the host. b4Via_of_ownLinkEnds re-proves by ignoring Rated. The locks' reads of B4Under, B4UnderId and B4Via change. Until soundness agrees, B4Via stays on Set.univ, and its docstring now says what that costs.

3. Docstrings, one sentence each (EgressRegistered, LeakPartsRegistered, B4Via): the registration's check asks a rate only of a link with no `unmediated` (stated beside Properties.Core.Site, not proved), the network check's `registered` holds only the one certifier link it is named to a rate (Properties.Certifier.Registered.RegisteredLinkRated), and an unmediated link may be unrated, and then no limit is stated across it. EgressRegistered's adds that onsite a boundary link may be unmediated instead of certified. B4Via's adds that on Set.univ it says nothing of a registered Site with any unrated link.

Built (local, at 57d807b796a0): Definitions.Core.SiteSubtree, Definitions.Accounting.Guarantees.{RegisteredSite,LinkEnds}, Proofs.Accounting.Actual.{RegisteredSite,RegisteredWitness,LinkEnds}. Axioms of LeakPartsRegistered, EgressRegistered, Site.uncertified_sealed, Site.leak_le_registered and LeakPartsRegistered_nontrivial: only propext, Classical.choice, Quot.sound. The lock record of LeakPartsRegistered (security/lean-audit.json, its statement) changes for the close's regeneration; uncertified_registered_eq is in no lock. I regenerated nothing.

Also from GATERELAY2: Lean comments in our certifier Lean still name old Python paths (layout_one: verity/core/protocols/certifier/*, verity.ml.*; protocol_groups: verity.protocols.certifier.verifier.scan_ok; security_dir: Security/). Prose only; I'll mend them next unless you'd rather someone else does.

[2026-11-14T22:07Z] 316a8603e0ef
MEND9319, TOP-424 reds. Pushed to `cursor/na-architecture-b51f` (tip `81927cbf627c`, plain fast-forward, no force). Results are from the seven files run on that tip on my VM: 20 failed, 70 passed. Per red:

- **`lint/test_lean_bar` (`DecodeEmpty.lean:202`): not ours now, as you said.** Before your 2:01 PM message I had committed `d7c594fae8c8`, which changes line 202's `maxHeartbeats 0` to `maxHeartbeats 400000` and keeps its comment. A node run at `6f00b98e1cd` built it: 400000 in 21 s, while 200000 times out. That commit exists only on my local `na-hold`. I left it out of the push and have reverted nothing. At the collect tip the unbounded file fails: `DecodeEmpty.lean:203:0: fwd: the condition is stuck`. After: still red, on DecodeEmpty and on PoUW's `PeelExactProofsP.lean:580` and `:591`.
- **`lint/test_repo_ruff` (`scan_zk.py:1262`): a.** `5579e8f6625b` puts the definition back to `window_of`.
  - The rename came from the integrator's merge `76f7a5860a3`. It applied `lean_names_map.toml:892`'s `words = { "window" = "executionPhase" }`, which is scoped to ten `.lean` files, to Python.
  - GATERELAY2 was told first; I had no answer from TERMGB2.
  - After: only inference-only's `desc_difftest_vectors.py:161` is left: an undefined name `Type` in a `"Type"` annotation, from `b035952c00ab`. It isn't ours.
- **The names sweep (`[a-z]+[A-Z]..._(of|at|in)|executionPhase|samplingDesign|transferBlocks` in `*.py`):**
  - Ours: `window_of`, mended above.
  - Other lanes': `code_executionPhase`, made by `period_net_map.toml:81` (`"code_window" = "code_executionPhase"`).
    - Defined in vllm's `adapters/vllm/verity_vllm/pipeline/advice_code/window_code.py`.
    - Called in `by_class.py`, `benchmarks/network_e2e/encodings/code_v1.py` and their tests; named in docstrings of certifier `arith_encode.py`.
    - Not mended. Its owner decides whether to put back `code_window`.
  - The rest of the hits are docstrings naming Lean declarations that exist, or test fixtures, so there is nothing to mend.
- **`lint/test_repo_python_verdicts`: b.**
  - `0343727e468b` pruned `certifier-device-program`, `simulator-python-audit` and `older-harness-memory-challenger` with `--prune`.
  - I removed `remote-session-boundary-check` by hand because its `boundary_check` was deleted in `e883650f2de`.
  - After: green.
- **`test_gate_paths`: b.** `76a70dad000c`:
  - repoints certifier/device's `benchmarks/rho/data/*` to `benchmarks/slowdown/data/*`;
  - drops spot/prover's 11 `benchmarks/network_traces/*.py` inputs, which are moved (and covered by `verity`) or deleted;
  - drops remote_session's old `word_codec/kv.cu`.
  - After: red only on `benchmarks/prove_path/pyproject.toml:37` `"benchmarks/network_traces/replay.py",`, deleted in `64bd635e08f`. That is the line for proofs; I did not edit it. Plus 4 `tools/check` Lean-key tests, not ours.
- **`test_lean_packages`: c and other lanes'.**
  - Long module: `verity/protocols/accounting/certifier/verifier/lean/NetworkCheck/Period.lean`, 2240 lines against the 2000-line cap, so 240 > 0. It is PERIODPROOF2's split; I left it. Spot's `493 > 0` is not ours.
  - `import all`: four `security/Proofs/Properties/InferenceOnly/Digest{Decode,Parts,Reads,Refs}.lean` files import `Circuits.Program` with `import all` to unfold its private decoder. Sent to POLICYROWS2. DecodeEmpty's is proofs'.
  - After: red on both.
- **`test_configurations`: regenerated with the generator** (`served_table.configuration` + `render`), in `a2753449132`. After: red only on the hashes, which are pinned at the close in `tools/verity/cli/verity_cli/configurations.toml`:
  - `"demos/onsite_compute.toml" = "f5ca90b1a0f6c0b201537a257c2c7070050a8f94629b4356ca437cca08d9b627e3fbb2da763263e35fccb143ae171e908f3f7de57f120d206a17dda2c1554322"`
  - `"demos/onsite_network.toml" = "9d9c1d03935955f17207629a34cc2001c04b90185587a00fe6a08bb88f342c59cde7036cc73eef8d75abb2b9e14295857d7ab94207f3b2d7572fc78ef519f00e"`
  - `"demos/onsite_full.toml" = "f8fd02322844310fd45ec865cbc9694b5d94035fecdbaa5666654f29b88e0902b949ad70f5fa71b54080fd7465cbc8599edf8153269c0fe2dcb7c67135263612"` (it was already stale; the file itself is unchanged)
  - `remote.toml`: unchanged.
- **`test_boundaries`, `device_party.py:30` runtime: a.** `b4abc411f8c` takes `ProtocolDef` from `verity.protocol`, which belongs to the `verity/runtime` place. After: the runtime rule flags only pous's `wipe_on_node.py:9`, which isn't ours.
- **`test_boundaries`, the two undeclared bare imports: a for ours.** `eedbcca43da` makes `compile_slice_zk` import `from benchmarks.network_traces import zk_drivers`. After: `benchmarks/prove_path/served_statics_zk.py:51` and `served_tie_zk.py:52` remain, both proofs'.
- **`test_boundaries`, `certifier/device` loads `verity/channel.py` and 6 more: c.** The line for ci, in `tests/import_order.toml` (or written by `import_order.py --grow`):
  `[[place]] paths = ["verity/protocols/accounting/certifier/device"]`, with
  `loads = ["verity/channel.py", "verity/commitments", "verity/commitments/crypto", "verity/protocols/accounting/certifier/verifier", "verity/protocols/accounting/pous/verifier", "verity/protocols/proofs/spot/verifier", "verity/runtime"]` and
  `reaches = [those + "verity/circuits", "verity/commitments/randomness", "verity/protocols/proofs/spot/verifier/profile.py"]`.
  - Why the loads are right: `device_needs`, `device_party` and `coin_link` moved out of `certifier/verifier/` (`certifier_device_map.toml:25-27`), whose place already loads `channel.py`, commitments, crypto, `spot/verifier` and the runtime. `certifier_key`, `load_relay` and `connection_rows_run` came from `benchmarks/network_traces` (`certifier_programs_map.toml:35-37`).
  - The one load new to a trusted place is `load_relay.py:95` → `pous/verifier/schemes/p2`.
- **`test_boundaries`, `NUMPY_USERS`: c.** It needs `verity/protocols/accounting/certifier/device/leak_guard.py` (numpy is optional there, with a pure-Python fallback) and `.../device/load_relay.py` (which decodes a load's values). Both moved from `benchmarks/network_traces`.
- **Console's `nothing_outside_benchmarks_imports_it`: a false positive.** It flags `certifier/verifier/network/tests/test_lean_audit.py:44: D.build(V2.LEAN, 'network-records')`, where `D` is `verity_difftest` and `V2.LEAN` is the certifier verifier's `lean/`. The tracer matches only the names `build` and `LEAN`. After: red.
- **Also red: `spot_never_imports_the_network_certifier_in_lean`.** The chain runs `compose/verifier/onsite/CoreOnsite.lean` → `security/Properties/Verity/Game.lean` → … → `security/Proofs/Flock/Bridge/GateARow.lean`. That is compose's and security's to cut, not ours.
- **ROWSPROTO2's two batches: done.**
  - `631a5baf8eb1` removes 14 entries for the deleted certifier verifier files.
  - `81927cbf627c` removes `repo_env_reads` `network_sim/workload/run.py`, `repo_retired_terms` `network_sim/workload/hardware.py` and `device_figures` `network_sim/mockdc.py`. `lab_run.py` was already in `0343727e468b`.
  - I have no inbox id for ROWSPROTO2; please pass this on.
- **device_party tests: not ours, not touched.** The two failures there (`'MessageTypes' object has no attribute 'decode'`) predate this work, from `156c17ce8f9`'s area.

