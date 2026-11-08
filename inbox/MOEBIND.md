- 00:25Z (P1AB): correction to 00:13Z: class G passes at checked's tip. My reproduction there skipped the fixture fetch. Through `suites.py`, all five vllm failures fail at your 59245cbe16 (r20261108-001440-2c7d: `WA.follows(A_OUT, A_IN)` returns `[(0, 0)]`) and pass at checked's tip f27b190dea (r20261108-001440-bc6b).
- 00:13Z (P1AB): [#7885](https://github.com/compute-verification/verity/pull/7885) is marked at c6dcef422f. Records r20261107-232208-c2ea and r20261107-234952-4ecc pass. Quick suites r20261107-232208-e657 pass 63 of 65; the two failures also occur at your 59245cbe16, from checked's landings: verity-vllm's class-G test (a missing workload JSON) and ruff F401, and circuit-check's catalog test (`MoeExpertFp8*` labels). The last two pass at checked's tip; class G fails there too.
- 23:33Z (P1AB): [#7885](https://github.com/compute-verification/verity/pull/7885) is open (parts A and B, stacked on #7838, at c6dcef422f), unmarked until records r20261107-232208-c2ea and quick suites r20261107-232208-e657 pass. On the coordinator's fused-cut question: A and B are statements about the read's own gates, whatever block holds them, so they carry over unchanged if CIR-187's fused block emits the same `Flock.ReadNet` nets; only a gate that fusing changes (say a select fused with the dequant) needs restating.
- 23:23Z (P1AB): **A and B build**, so D4's second piece can go ahead. On `cursor/proofs-p1ab-f3f6` @ c6dcef422f (stacked on 59245cbe16), try r20261107-231818-6d08 built Sha, RowSeg, Leaf, Select and Tie with no warnings. `rowSegNet_spec`, `leaf_spec` and `select_spec` use only `propext`, `Classical.choice` and `Quot.sound`. So does a scratch `holdsRead_sound pinnedKey hD hVD hD32 hH (rowSegNet_spec (16 * H)) leaf_spec hC (select_spec hV) hrows z hsat h1`, with `hC` a hypothesis. `select_spec` still needs `hV : V < 2 ^ 32`; details in `inbox/proofs-coordinator.md` at 23:08Z. Records r20261107-232208-c2ea and quick suites r20261107-232208-e657 are running; I open the PR stacked on #7838 when they pass.
- 19:25Z (proofs): SPOTIFACE's layer interfaces are on `cursor/proofs-spotiface-874a` (note proofs/20261107T1835Z-report-spotiface): spec `Security/Definitions/Claims/SpotFlock.lean` (`Exports`, `Refines`/`RefinesTwo`, `Composes`/`ComposesTwo`, proved from L2 with no sorry); L1 `Flock.Spot.arg`; L3 `SpotFlock.deployed`/`deployedTwo`, reusing `SamplingDeploy.compiled`. DECIDED (proofs): KS composes at `Arg.ofFlock` over the same record, and ZK's scope (statements the loop serves) is handled on the ZK side, not by one shared relation. Your P2 binding belongs in L3's `Refines` (the binding field) at the per-expert cut.
- 19:35Z (proofs): you're now part of PRF-456 (every GLM-5.3 block at h <= 1e7), led by HILLCLIMB (bc-17ebf8d3-193d-53ec-9eba-07af0075377c). Keep your task. Put your numbers in the PRF-456 table's terms (h marginal and full, the split, the gap to 1e7), and report them in inbox/proofs-coordinator.md as before.
- 19:36Z (proofs): one more. State the binding generically for table reads at a router- or indexer-chosen index (MoE expert rows, sparse-attention KV rows, embedding rows), so attention's KV reads reuse it.
- 19:47Z (proofs) **ACCELERATE (Daniel, 11:40 AM PST: 'Wednesday is way too late').** Compress your plan: whatever can land or be shown on a branch by tonight (Sat 6 PM PST = Sun 02:00Z) does, and the rest by Sun 10 AM PST (18:00Z). Don't wait for landing: show results on branches with run ids. Mark PRs as soon as each piece passes. If a piece can split into independent parts, name them in inbox/proofs-coordinator.md and I'll add a helper (from finished workers). Report a blocker the moment you hit it, not at your checkpoint.
- 20:35Z (proofs): your Linear issue is PRF-458 (under PRF-456); use `--issue PRF-458`. Helpers start now: P1AB (bc-2fa1a0fa-e037-5c99-bab2-e7bcc6a2653c) takes parts A and B, and P1C (bc-93ccc272-f09a-5b29-9491-240740e198ac) takes C. You keep P2's PR and part D, the tie. I'm asking @soundness where the registration's descriptor lives in Lean for D. Fix the statements of A, B and C on your branch fast so the helpers build against them.
- 20:40Z (proofs): soundness answered part D's question on PRF-458. The registration's descriptor is `Core.Registration` (`verity/core/protocols/interface/CoreProtocol/Registration.lean` on main): `Declared` (`program` = the circuit file's SHA-512, `blocks` = its block Definitions by descriptor key, declared once a day before the first coin and chained in the registration log), plus the `Builder` item (`digest`, `advice`, `cap`). Part D reads the parsed circuit file's Definitions, held by SHA-512 to `Declared.program`, once per key in `Declared.blocks`. Do NOT add a second descriptor.
- 20:50Z (proofs): BREAKEVEN found the per-expert cut's separate table-read blocks commit the weight rows as boundary values, 99.2% of a request's committed bits. That's why (A') is worse than (A). The fix is the **fused cut**: the read fused into the expert GEMM, so rows are read inside the block from the committed table (CIR-187, circuits). Build so the read binding (P2, P1) and the serving rows carry over to the fused block: the table, opening and read are the same; only the block boundary moves.
- 20:50Z (proofs): D split into D1 to D4 acknowledged, with the generators as verifier code in `Flock.ReadNet`; tell P1AB and P1C directly through their inbox entries, which I've added. For helpers: PCINTERVAL takes D2 (template-unroll compositionality from `TmplFold`) when it frees up, likely tonight. Start D1 yourself if you have capacity after P2's PR; else I'll find another. Keep D3 and D4 behind A to C.
- 21:05Z (proofs) your D1/D2 helper ask is granted. PCINTERVAL (bc-ae672245) takes **D1** on `cursor/proofs-moebind-d1-*`, and PCRANKIN (bc-2230725f) takes **D2** on `cursor/proofs-moebind-d2-*`, both branched from your `cursor/proofs-moebind-bdb9`. Each posts its theorem's statement here by about 22:15Z, so if you'd rather fix the statements yourself, append them here before then and they'll follow yours. They don't open PRs: you land D1-D4 together. You keep D4 and the A-C coordination with P1AB and P1C. Hard deadline Sun 18:00Z.
- 21:25Z (proofs) **D3 now has a helper too: ZKSTACK** (bc-ee22c80b-afde-5e72-9dc7-926056a8f10d), on `cursor/proofs-moebind-d3-*` from your branch, against P1AB's and P1C's generator interfaces. It posts its plan here by 22:30Z. So D1 is PCINTERVAL's, D2 PCRANKIN's, D3 ZKSTACK's, and D4 and landing are yours.
- 21:10Z (PCINTERVAL): **D1 is stated and proved** on `cursor/proofs-moebind-d1-bf89` at `02916c7468` (from your `e6046484c3`). It is in `Security/Proofs/Flock/Bridge/UnitRows.lean` (namespace `Flock.Bridge`), added to the `Proofs.Flock.Bridge` umbrella. Variables: `(D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)`. The theorem: `theorem rows_of_correct {u : Fin n} {tr : (circuitOf D hU n).Transcript} (hX : (partsOf D hU n).Correct tr u) {j : ℕ} (hj : j < D.unroll.upv) : ∃ z : ℕ → Bool, (Rows.ofNet hU).Sat (fun c => z c.val) ∧ z (Rows.ofNet hU).one.val = (tr (constGate D hU n u)).getLsbD 0 ∧ ∀ c : Fin (Rows.ofNet hU).w, (c.val < (Rows.ofNet hU).nIn ∨ copyGate D hU n u hj c ∈ (partsOf D hU n).committed) → z c.val = (tr (copyGate D hU n u hj c)).getLsbD 0`. So each copy `j` of a correct unit holds the unit net's rows at values that are `tr`'s bits at its input columns and at every column whose gate is committed, and its constant is `tr`'s bit at the stack's constant gate. `(tr g).getLsbD 0` is `ReadUnit.bit tr g`. The gates: `copyGate u hj c := (unitInst D hU n u).wire ((Rows.ofNet hU).stackCol D.unroll.upv hj c)` and `constGate u := (unitInst D hU n u).wire (Placed.circuitRows D.unroll hU).one`. `unitInst u : (keyPartition D hU n).IsRowsUnit u (Placed.circuitRows D.unroll hU)` is the program's own wiring: `unitInst_wire` says its wires are `Const.Prog.rowsWireC (keyProg D.unroll hU n) u`'s, the wiring `Layout.outsOf` reads. Also exported: `correct_bits`, which says `partsOf`'s `Correct tr u` is `keyPartition`'s `Correct (fun g => (tr g).getLsbD 0) u`. The `z` is the unit's local evaluation (`FlockVBridge.copyVal (unitInst D hU n u) _ hj`), so most of the proof was already on main: `FlockVBridge.sat_copy`, `copyVal_input` and `copyVal_committed`. Remote try `r20261107-210605-01b0` on cpu-2 exits 0, and the sibling check found no clashes. Axiom probe: all 5 theorems use only propext, Classical.choice and Quot.sound. One thing to watch: `X` is a token in this file, because ArkLib's global `notation3 "X"` comes in through `CircuitOf`'s imports. **For D2 (please relay to PCRANKIN):** `FlockSoundness.TmplValue.unit_eval` and `unit_computes` (`Security/Proofs/Flock/Soundness/TmplValueUnit.lean`, on main) already give every called Definition's value, and through `DefComputes` its spec, at the local evaluation of a unit that is an `IsRowsUnit` of stacked `Rows.ofNet (unroll_built h)`. Their hypotheses are `t.wellFormed`, `apart`, `ordered` and `Tmpl.unroll t = .ok net`. Since `HmOut.unrolled` swaps the unit net for `Tmpl.unroll`'s, they apply to `unitInst D hU n u` when `D = .tmpl c`. D2 may be mostly a matter of composing these.
- 21:55Z (PCRANKIN): **D2 is stated and proved** on `cursor/proofs-moebind-d2-bb8b` at `2f3b9d6449` (from your `e6046484c3`), in `Security/Proofs/Flock/Soundness/TmplSpec.lean` (namespace `FlockSoundness.TmplSpec`), added to the `Proofs.Flock.Soundness` umbrella; no D1 file touched. Remote try `r20261107-214437-5ffd` on cpu-2: lean exit 0; axiom probe: none of its 11 theorems uses an axiom beyond propext, Classical.choice and Quot.sound. **The induction principle**, one Definition at a time (`df := defs.getD d default`): `def Steps (defs : Array Def) (Spec : ℕ → (ℕ → Bool) → List Bool → Prop) : Prop := ∀ d < defs.size, ∀ v inp, inp.length = df.ins → (∀ r ∈ df.rows.toList, v r.1 = rowVal (df.refV v inp true) r) → (∀ c ∈ df.calls.toList, ∀ j < c.count, Spec c.def_ (fun y => v (c.base + j * c.stride + y)) (c.boundV (df.refV v inp true) j)) → (∀ y < df.cols, (∀ r ∈ df.rows.toList, r.1 ≠ y) → df.copyAt defs y = none → v y = false) → Spec d v inp`; `holds_spec` gives `Spec` at every copy satisfying `Flock.Tmpl.Holds` (induction on fuel). **The unit theorem**, at D1's `D = .tmpl ct` (`Desc.unroll (.tmpl ct)` is `HmOut.unrolled ct`), with `root := tu.t.defs.getD tu.t.root default`: `theorem unrolled_spec {ct : Circuit} {tu : TmplUnit} (ht : ct.tmpl = some tu) (hsz : ct.unitNet < ct.nets.size) (hB : Built (HmOut.unrolled ct).unit) (hu : (HmOut.unrolled ct).unit.unitLog ≤ 32) {Spec} (hS : Steps tu.t.defs Spec) {z : ℕ → Bool} (hs : (Rows.ofNet hB).Sat fun c => z c.val) (hone : z (Rows.ofNet hB).one.val = true) : (∀ r ∈ root.rows.toList, tu.t.inCols ≤ r.1 → z r.1 = rowVal (root.refV z [] true) r) ∧ ∀ c ∈ root.calls.toList, tu.t.inCols ≤ c.base → ∀ j < c.count, Spec c.def_ (fun y => z (c.base + j * c.stride + y)) (c.boundV (root.refV z [] true) j)`. `unroll_spec` is the same for any `t` with `Tmpl.unroll t = .ok n`, `t.wellFormed` and `t.apart`. So D1's `rows_of_correct` feeds it directly: its `z` and `Sat` are `hs` with `hB := hU`, and its constant clause gives `hone` once `tr`'s bit at `constGate` is 1. **The leaf lemma** (`laid_rows`/`laid_outs` for a Definition): `Realizes df φ rows` says each builder row's output sits at an own column (`φ r.out < df.cols`) carried by a row of `df` whose two reference lists equal, up to `canon`, `refsOf df φ r.a` and `refsOf df φ r.b` (`refsOf` maps variables by `φ` and the constant to `df.cols + df.ins`). Then `stHolds_of_realizes : Realizes df φ st.rows → (∀ r ∈ df.rows.toList, v r.1 = rowVal (df.refV v inp true) r) → stHolds (fun x => df.refV v inp true (φ x)) st`, with `rowHolds_of_realizes` per row and `outs_of_realizes` for outputs (output `j` at own column `oc j`, carried by the row `refsOf df φ outs[j]` times the constant, gives `v (oc j) = ev (fun x => df.refV v inp true (φ x)) outs[j]`). **For D3:** take `Spec d v inp := stHolds (fun x => (defs.getD d default).refV v inp true (φ_d x)) st_d ∧` (its outputs), prove `Steps` per Definition (own rows by `stHolds_of_realizes`, calls by their callee's `Spec`), and `unrolled_spec` gives it at every root call copy. PCINTERVAL's `unit_computes` gives `DefComputes`, a Definition's evaluation; `unrolled_spec` gives any spec closed under `Steps`, such as a builder program's `stHolds`. What remains for D3 or D4 is assembling the root's `stHolds` from its own rows and its calls' specs, which depends on the generators' call shape; I'll write that lemma once P1AB's and P1C's interfaces fix it, if you or ZKSTACK want it.
- 22:20Z (proofs) D3 helper change: ZKSTACK didn't start (its turn ended, and the cloud-agent cap blocks resuming it). D3 goes to PCRANKIN when its D2 turn completes, since it knows TmplSpec now. If you'd rather start D3 yourself in the meantime, say so here and I won't assign it.
- 23:04Z (PCRANKIN) **D3's check and statement.** Branch `cursor/proofs-moebind-d3-54c9`, from your 59245cbe16 with D2 (2f3b9d6449), P1AB (b528f4ee47) and P1C (9aacf5c235) merged, no conflicts. Its elaboration try is r20261107-225503-0b12 on cpu-13, still building; I'll confirm here. The proof is in progress.
  - **The check** (verifier code, `verity/core/service/flock/Flock/ReadNet/Template.lean`): `Flock.ReadNet.check (V H : Nat) (t : Flock.Tmpl.Tmpl) : Bool`. It is true iff `2 ≤ V < 2^32`, `1 ≤ H`, each part's run writes fresh variables (`runOk`, a pass over that part's rows), and `t`'s Definitions, root and port groups are `readTmpl V H`'s (`decide (t.defs = r.defs)`; `Flock/Tmpl.lean` now derives `DecidableEq` for `Seg`, `Call` and `Def`). It reads the description once and runs each part's generator once; it never unrolls.
  - **The template is the verifier's** (`readTmpl V H`). Definitions 0–10 are the parts: a full segment, the last segment, the outer digest, the hm96 leaf, the frame leaf at `B = indexBytes D`, the frame node at `B = 1, …, 4`, `eqDigest` and the select. Each is `defOf ins (gen on input variables 0, …, ins − 1)`: the run's rows, variable `x` at own column `x − ins`, then each returned form committed to an own column (`form · 1`). Definition 11 is the root:
    - the input self rows, ports `[32, 512, 512, 16H, 1536, 512D]` in `readProg`'s order;
    - a zero column (`[const] · []`);
    - the calls in `readNet`'s order (the `s` segments, outer, leaf, frame leaf, the `D` nodes, eq, select), each input bound by a one-reference segment to a port bit, an earlier call's output, the zero column or the constant;
    - the output group (the row's `16H` bits, then `ok`) as `form · 1` copies, then the constant.
    - `flock-read-tmpl V H` (new exe) prints the template after parsing it back and running `check` on it.
  - **Why not stub the composites.** A stub has no rows, so `Sound` (for every state) can't hold of it, and the parametric `rowSeg_spec` and `readNet_sound` can't be instantiated at stubs. A part's Definition is its generator on input *variables*, while the flat run calls it on folded forms (constants, repeated variables), so the unrolled rows aren't a renaming of `readProg`'s. The template's own composition is therefore proved at values, and `HoldsRead` is for a run with no rows.
  - **The theorem** (`Security/Proofs/Accounting/ReadUnit/Check.lean`):
    ```lean
    def ReadSound (key : Hash.Msg) (V H : ℕ) {R} {run : (List Lin × Lin) × St}
        (hr : HoldsRead R (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])) : Prop :=
      -- exactly gatesRefine_of_vals's hsound
    theorem read_of_check {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
        (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
        (hB : Built (Flock.HmOut.unrolled ct).unit) (hu : (Flock.HmOut.unrolled ct).unit.unitLog ≤ 32) :
        ∃ (run : (List Lin × Lin) × St)
          (hr : HoldsRead (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])),
          ReadSound pinnedKey V H hr ∧
            ∀ j < 16 * H + 1, (hr.out j).val = Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (j / 16) (j % 16)
    ```
  - **D4 with it:** `obtain ⟨run, hr, hsound, hout⟩ := read_of_check ht hsz hcheck hU hu`, then `gatesRefine_of_vals pinnedKey P u wire hr hsound consts hvals`. D4 needs neither `holdsRead_sound` nor A–C's composite theorems. D3 uses the parts' per-part lemmas (`sound_segment_mid`, `sound_outer`, `leaf_spec`, `sound_leafHash`, `sound_nodeHash`, `sound_eqDigest`, `select_spec`), which already take arbitrary input forms, so I need nothing new from P1AB or P1C. `run` is: input variables `0, …, N − 1` at input columns `0, …, N − 1`, then `16H + 1` output variables at the output group's columns, with no rows.
  - **V and H** are `check`'s parameters. The call site reads them from the registration (`Declared.blocks`, key `EmbeddingRead_v1{V, H}`); that glue is D4's, or mine on request.
  - **The prover and the difftest.** The prover stages the read unit from `flock-read-tmpl`'s text, not from `tmpl.emit` or `struct_lower`, whose Definitions can't match a per-Definition check. The difftest evaluates that text with `verity_flock.tmpl` at small `V` and `H`, on random inputs and honest openings, against `embedding.reference`.
  - **Scope** is larger than 300–500 lines: about 250 of verifier code (written), 500–700 of proof (the root re-derives `rowSeg_spec`'s, `frameReadSpec`'s and `readNet_sound`'s composition at the root's columns), and about 100 of Python. Target unchanged: Sun 14:00Z.
- 23:23Z (P1C): taking D1's three layout facts on `cursor/proofs-p1c-d1-e824` (off 02916c7468). Draft 3d23a2dbd2, first try running. One placement change: `constsOfP_constGate` goes beside `constsOfP_empty` in `Proofs/Properties/Verity/Sampled/Unbound.lean`, because importing `V5.Statement` into `UnitRows` would pull 13 modules into its closure. `UnitRows` gets `constWiresOf_constGate` and `constsOf_constGate`. SHA and try to follow.
- 23:25Z (proofs) check, please: the headline lever is the FUSED per-expert cut (CIR-187: the expert table read fused into the GEMM), not the per-expert read as a separate block. Does P1/P2's binding (A-D) apply unchanged to the fused block's read, or does fusing change what must be bound? Answer in inbox/proofs-coordinator.md; I've asked circuits for CIR-187's design.
- 23:55Z (PCRANKIN): **Answers to your 23:10Z, 23:25Z (both) and 23:35Z.** The new statement, on `cursor/proofs-moebind-d3-54c9` once it elaborates (try r20261107-235020-7a9a running; the edits below go in the next try):
    ```lean
    theorem read_of_check {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
        (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
        (hB : Built (Flock.HmOut.unrolled ct).unit) :
        ∃ (run : (List Lin × Lin) × St)
          (hr : HoldsRead (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])),
          ReadSound pinnedKey V H hr ∧ (∀ x < readIn H (Hash.Tree.depth V), (hr.φ x).val = x) ∧
            ∀ j < 16 * H + 1, (hr.out j).val = Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (j / 16) (j % 16)
    ```
  - **Where the outputs sit (23:10Z):** given by the last conjunct. Per your 23:35Z, no range conjunct.
  - **Pinned inputs:** the middle conjunct, `(hr.φ x).val = x` for every input variable.
  - **`hu` is gone.** The unrolled unit's `useful` is its root Definition's columns (`Tmpl.unroll`), so `check` now also requires `log2ceil (t.defs.getD t.root default).cols ≤ 32`, and `unrolled_unitLog` derives `hu` from it.
  - **Also in `check`:** `16 * H < 2^64`, the row-length bound `EmbeddingRead.ok` reads.
  - **One composition: (ii).** My proof reads none of the flat composites. It reads only these per-part lemmas:
    - P1AB's `segment_spec`, `outer_spec` and `leaf_spec`;
    - `Gates.select_spec`;
    - P1C's `FrameRead.sound_nodeHash`, `sound_leafHash` and `sound_eqDigest`.

    So these needn't land: `rowSegNet_spec`, `frameReadSpec` and C's climb, and your `readNet_sound`, `holdsRead_sound` and `gatesRefine_of_run`. Please tell P1AB and P1C.
  - **`verifier.stop`:** I added `Flock.ReadNet.cvWords` and `Flock.ReadNet.midstate` to Flock's `lean-audit.json`. The lock check doesn't test stop entries for staleness, so they cause no failure before D4 puts `check` on the path from `verifier.entries`.
  - **The read's top as Definition 11 (23:35Z): not cheap now.** The composition runs at the root's valuation and reads the read's inputs as the unit's input columns. At a call copy, they are the Definition's inputs instead (`refV`'s `inp`). The move needs three changes:
    - reparameterize `Check/Compose` and `Check/Parts` (about 750 lines) over a copy's valuation and inputs;
    - add a `Steps` case for Definition 11 that composes its calls;
    - change the template (a root wrapper).

    I'll get D3 green as it stands. If there's time before Sun 14:00Z, I'll do it as a follow-up commit on the same branch. Tell me if CIR-187 needs it in the D PR itself.
  - **Branch state:**
    - Merged P1AB c6dcef422f and P1C fa935e37f3. Try 5 built every upstream module, theirs included; the only errors were in my `Check/Copies`, now fixed.
    - Importing both lanes' modules together made four names ambiguous, so I qualified them in P1C's `Frame/` files: `Frame.constBits`, `Frame.padTail`, `Frame.select`, `Discharge.Hm.hash_eq`. I've told P1C.
- 23:56Z (P1C): **D1's three layout facts are proved.** They're on `cursor/proofs-p1c-d1-e824` at 62e48d426e, off 02916c7468, with no PR. Try r20261107-234536-b3f4 (`Bridge/UnitRows.lean`) and try r20261107-234904-0665 (`Sampled/Unbound.lean`) both exited 0 with `--siblings`, and every theorem uses only `propext`, `Classical.choice` and `Quot.sound`. (a) `constGate_mem_io` and (b) `copyGate_mem_committed` are exactly your statements. For (c), `constGate_eq_oneG` is as stated (the `.val` equality). I added the Fin equality `constGate_eq : @Eq (Fin (Layout.keyN D.unroll hU n)) (constGate D hU n u) Layout.oneG`, plus `constGate_val` (`= 0`), `constGate_eq_oneAt`, `constWiresOf_constGate` and `constsOf_constGate`, all in `UnitRows`. The one shape change: `constsOfP_constGate D hU n pi pub u` is in `Sampled/Unbound.lean`, beside `constsOfP_empty`, and that file now imports `UnitRows`. Your `κ` is `constsOfP … {} pub`, so it applies directly. One tip: if you `rw` a membership at `constGate`/`copyGate`, state it at a gate `g` of the set's own type with `g = constGate …`, because `Fin (circuitOf …).N` and `Fin keyN` agree only by unfolding. Note: lanes/proofs/20261107T2355Z-result-p1c-d1.
- 00:25Z (P1C): two things.
  - **SHA-512:** P1AB's copy (`Flock.ReadNet.padTail` and `shaW`, with `sound_shaW` and `hash_chainW`) stays, because it is the one proved against `Hash.RowSeg.digest`. I'm switching part C to it on `cursor/proofs-p1c-sha-c153`. That branch merges P1AB's c6dcef422f and makes PCRANKIN's four name-clash edits. A try on vy-nebius-cpu-13 to follow. `sound_nodeHash`, `sound_leafHash` and `sound_eqDigest` keep their statements.
  - **D4: do you want help?** Name the piece, for example the port map, or wiring D1–D3 and A–C into `GatesRefine`, and what to branch from (your D4 tree is unpushed). I'll take it on a branch of my own with no PR. Also, if `frameReadSpec` and the climb shouldn't land under (ii), say so and I'll delete them in the same branch.
- 00:40Z (P1C): **One SHA-512 copy: done.** It's on `cursor/proofs-p1c-sha-c153` at fef6cfd740, which is part C's fa935e37f3 with P1AB's c6dcef422f merged in. Merge it in place of fa935e37f3, which it contains.
  - **What changed.** Part C's message pads with `Flock.ReadNet.padTail`, and its hash is `Flock.ReadNet.shaW ((ws.length + 15) / 16) ivW ws`. `Frame.padTail` and `Frame.sha512W` are deleted. `FrameRead.sound_sha512W` keeps its name and conclusion; it is now over `shaW`, derived from P1AB's `sound_shaW` and `hash_chainW`.
  - **The rows don't change.** It makes the same `compress` calls on the same 16-word blocks, and the two paddings agree byte for byte. `sound_nodeHash`, `sound_leafHash` and `sound_eqDigest` keep their statements.
  - **The clash edits are in.** These are `Frame.constBits` (also in the generator's `climbWith` and `frameWith`, where `constBits` now resolves to P1AB's) and `Frame.select`. `Frame.padTail` and the old `hash_eq` no longer occur.
  - **Try r20261108-002543-6ef4 on vy-nebius-cpu-13.** It built `Frame.lean` and its whole chain: the generator, P1AB's `Flock.ReadNet.Sha` and `ReadUnit/Sha`, and my `Frame/*`. Lean exited 0, and both probed theorems use only `propext`, `Classical.choice` and `Quot.sound`. The try exited 1 only at the sibling stub, which couldn't load `HiddenExec/Accepts.olean` in a cache dir another run had left half-built. The changed modules' only other importers are the `Flock` and `Proofs.Accounting` aggregators, which hold no proofs.
  - **D4:** my 00:25Z offer stands; name a piece and a base. I read your 00:30Z note that the D PR deletes the flat composites nothing reads, so I'll leave `frameReadSpec` and the climb to you.
- 00:50Z (MOESERVE): **`units_per_vu` is 1 for every served read and fused block: one read a unit, one unit a VU, one VU a
  block.** r20261108-001445-73d5 (cpu-1) staged each served template on `UnitStager` at GLM-5.3's statics, as
  `chunk_rows.chunk_cap` does. The results:
  - `EmbeddingRead_v1{V=256,H=96}` (w13's scale rows) and `{V=256,H=4}` (w2's): `units_per_vu` 1, `vus_per_block` 1,
    k_log 23, unit_log 22, chunk cap 4,096.
  - `MoeExpertFp8ReadGemmW_v1` (w2: H=4096, K=256, N=32) and `MoeExpertFp8ReadGemm_v1` (w13: H=3072, K=6144, N=1, at the
    card's tile): `units_per_vu` 1, `vus_per_block` 1, k_log 25, unit_log 24, chunk cap 1,024.
  - So P2 needs no index by copy inside a unit. Each fused unit's key `in0`, root and domain `in1`/`in2`, opening ports
    `in3`..`in5` and verdict `o1` are its one read.
  - Two caveats:
    - The staged template is `UnitStager`'s lowering of the Definition, not `flock-read-tmpl`'s text. Matching the two is
      PCRANKIN's read check.
    - Step 374's w13 launch runs SWAP=true with the QGMMA dot. Its cap run is r20261108-004331-0b14 (ETA 01:00Z); I'll
      post here only if it differs.
  - The serving side is PR #7905, marked at `624862cd11`. GLM-5.3 (L5, step 374, rank 0) checks against B's
    registration in r20261108-003025-a5d4: every read opens, and every fused unit is its twin over its opened row.
- 01:05Z (PCRANKIN): **`Check.lean` elaborates**, pushed on `cursor/proofs-moebind-d3-54c9` at af656cc66a (try r20261108-004830-f557: lean exit 0; `read_of_check` uses only `propext`, `Classical.choice` and `Quot.sound`). The statement is unchanged from 23:55Z apart from `hu`, which the check now discharges:
  - `read_of_check (ht : ct.tmpl = some tu) (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true) (hB : Built (Flock.HmOut.unrolled ct).unit) : ∃ run (hr : HoldsRead (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])), ReadSound pinnedKey V H hr ∧ (∀ x < readIn H (Hash.Tree.depth V), (hr.φ x).val = x) ∧ ∀ j < 16 * H + 1, (hr.out j).val = Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (j / 16) (j % 16)`. The verdict is `j = 16 H`, at `outCol … H 0`.
  - **`hV : V < 2 ^ 32`** is already in `check` (`Checked.hV32`), as are `2 ≤ V`, `1 ≤ H`, `16 H < 2^64` and the root's `log2ceil cols ≤ 32` (which gives `unitLog ≤ 32`, `unrolled_unitLog`).
  - **`H < (D.unroll.leavesOut.getD j #[]).size`**: the read's outputs are now `H + 1` ports of 16 bits (8b133165ef), not `#[16 H, 16]`. `unrolled_outPorts ht hsz hcheck : ((Flock.HmOut.unrolled ct).unit.outGroups.getD 0 default).bits.size = H + 1`. With `ExecCheck`'s `leavesOut` (where the unit returns the outputs), that is each copy's `leavesOut` size. 6ed03eed32 makes the inputs u16 ports as well. Both are `unit_ports`' shapes (`SL.Ports((16,) * n_words, …, (16,) * W, …)`), and the bit positions are unchanged. Try r20261108-010309-07eb (with `--siblings`, and `Proofs.Accounting` now imports `ReadUnit.Check`) and difftest r20261108-010309-8ae0 run on 6ed03eed32; ETA about 01:20Z. I'll post here when they pass.
  - Merged P1C's SHA switch (fef6cfd740), taking its side. The difftest passes after the merge (r20261108-004829-f2fd: 6 of 6, at (V, H) = (2,1), (5,2), (3,513), plus the three refusals).
  - **For MOESERVE's 00:50Z caveat: `check` accepts exactly `readTmpl V H`, the text `flock-read-tmpl V H` prints.** It does not accept `UnitStager`'s `struct_lower` lowering of the Python Definition, whose Definitions and columns differ. So a served read verifies only if serving stages that template as the read unit's: `T.parse` of `flock-read-tmpl V H`'s stdout, in place of `SL.lower`'s `t`. Its ports are now the served unit's for the read's signature order: rows `tok, root, domain, row, salt, path` as in0..in5, then `h` and `ok` as `H + 1` u16 outputs. The difftest holds that template to `gates.embedding.reference` through `tmpl.evaluate`. The fused cut stays out of D3, per your 00:20Z.
- 01:15Z (PCRANKIN): **D3 is green at 6ed03eed32**, pushed on `cursor/proofs-moebind-d3-54c9`.
  - Try r20261108-010309-07eb (clean tree, `--siblings`): lean exit 0. All 10 theorems of `Check.lean` (`read_of_check`, `unrolled_outPorts`, `checked_outPorts`, `unrolled_outGroups`, …) use only `propext`, `Classical.choice` and `Quot.sound`. The sibling check found no name in common with a built module of `Security/Proofs`; modules that slot hadn't built since their sources changed weren't checked.
  - Difftest r20261108-010309-8ae0 on 6ed03eed32: 6 of 6.
  - Base for D4's corollary: 6ed03eed32. Next: the 02:00Z checkpoint and a cleanup pass (the deprecation warnings and the unused simp arguments). Nothing more in D3's statement unless you need it.
- 01:55Z (P1C): **D4 piece 1 is pushed:** `cursor/proofs-p1c-len-281e` at eb9771423b, one commit on PCRANKIN's 6ed03eed32. It's your diff as written:
  - `read_of_check` gets `run.1.1.length = 16 * H` as the second conjunct, after `ReadSound pinnedKey V H hr`.
  - The final `refine` gets one more `?_`, and a `· simp` goes after `ReadSound`'s bullet.
  - I rewrapped the statement's next line as two so it stays under 120 characters.

  **Try r20261108-014758-8c8f on vy-nebius-cpu-13:** Lean exited 0, and the axiom probe says none of the file's 10 theorems uses an axiom beyond `propext`, `Classical.choice` and `Quot.sound`. The try still exited 1, but only at the `--siblings` stub. The shared cache dir build-4 has a stale `Properties.Identity.olean` from before that file moved to `Properties/Hash/Identity.lean`, so the two clash on `TaggedIsFlock`. My diff doesn't touch either. Your own try of this diff, r20261108-013508-e69f, had clean siblings.

  PCRANKIN has been told which lines this touches. **Piece 2 (`unrolled_leaves`) is next, on the same branch; I'll post here when it's tried.**

## 02:00Z from MOESERVE: the served read needs `flock-verify` to accept a templated statement, which it refuses today on every branch

- **The gap.** Once serving stages `flock-read-tmpl V H`'s template, the read unit is a templated statement
  (`verity/flock-circuit/tmpl`, `circuit.compose`'s `layout`). D4 needs exactly that form (`ht : ct.tmpl = some tu`, which
  only `parseTmpl` gives). But `flock-verify verify` refuses every templated session as its last check:
  - `ProvedScope.servedForm` calls `versionWhy`. `served` holds only `verity/flock-circuit` and
    `verity/flock-circuit+plain-leaves` (`ProvedScope.lean:519`, called at `FlockVerify.lean:1070`), so the session is
    refused as "verity/flock-circuit/tmpl, a superseded statement version".
  - That holds on `checked` and on D3 (`6ed03eed32`), D4, P1AB, P1C-sha and PCRANKIN's branches. None of them changes
    `served`.
  - Under `--zk`, `KsScope` and `SampledKSLiveV4`/`V5` are stated at `HmOut.parse`, the flat parse. `h1unit.layout`'s
    docstring says the same of H1's templated statement: "the Lean verifier refuses it after every other check".
  - A flat file can't stand in, because it has no unit template for `ReadNet.check` to accept.
- **So "flock-verify accepts" at 12:00Z needs a Lean change first.** `verify` must accept `verity/flock-circuit/tmpl`, at
  least where the unit template passes `ReadNet.check V H` (the test PCRANKIN's command makes), and the sampled knowledge
  soundness must cover that form, since no form goes ahead of its proof. That's Flock's Lean, not mine. **Who takes it,
  and can it land by 12:00Z?**
- **What I do meanwhile, unchanged.**
  - By 08:00Z, the staging code on top of #7905, which is on `checked`: each read unit staged as the templated statement
    of `T.parse(flock-read-tmpl V H)`.
  - Then the GLM-5.3 read at L5, step 374, rank 0 (w13's scale read is V = 256, H = 96; w2's is H = 4):
    - proved with `prove --zk`;
    - its circuit file put through PCRANKIN's `flock-read-tmpl check`;
    - put through `flock-verify verify --zk`. That refuses it at `servedForm` alone unless the Lean change lands first, and
      the run will show where it stops.
- **For MOEBIND:** does D4 assume a layout mode? Unless you say otherwise, I'll stage at `iv/hybrid/4`, the structural
  path's mode (`h1unit.LAYOUT`).
- 02:07Z (P1C): **Both D4 pieces are on `cursor/proofs-p1c-len-281e` at 225efedad7**, two commits on PCRANKIN's 6ed03eed32. Merge 225efedad7.
  - **Try r20261108-015512-8031 on vy-nebius-cpu-13 exited 0**, on a tree with both pieces:
    - Lean exited 0.
    - The sibling check is clean (build-2).
    - None of `Check.lean`'s 13 theorems uses an axiom beyond `propext`, `Classical.choice` and `Quot.sound`.
  - **Piece 1** (eb9771423b) is `read_of_check`'s `run.1.1.length = 16 * H`, as written.
  - **Piece 2** (225efedad7) is 39 lines after `unrolled_outPorts`. It needs one hypothesis beyond your shape: `hu : ct.unit = tu.net`.
    - **Why `hu`:** `CheckFacts`' `leavesOut` compares the leaves with `ct.unit`'s output group. Nothing in `ht`, `hsz` or `CheckFacts ct` ties `ct.unit` to the template, so without `hu` the statement is false for some `ct`.
    - **The three theorems:**
      - `tmplUnit_outBits tu`: the bits of `tu.net`'s output group 0 are `tu.t`'s, because `Net.ofRowsAt` moves only each group's column.
      - `unrolled_leaves (ht) (hsz) (hf : CheckFacts ct) (ho : ct.outNet = ct.unitNet) (hu : ct.unit = tu.net)`: its conclusion is exactly your `hupv ∧ hleaves`.
      - `unrolled_leaves_of_parse (h : Flock.HmOut.parseTmpl tags bytes tables = .ok ct) (ho)`: the same conclusion, from the parse alone. `parseTmpl_facts` gives `ht`, `hsz` and `hu`, and `check_facts` gives `CheckFacts`.
    - If D4 holds the parse, use the last one. Otherwise `hu` comes from `parseTmpl_facts`' `c₀.unitNet = 0` and `c₀.nets.getD 0 default = ("unit", tu.net)`.
  - **`flock-read-tmpl check`:** your 01:38Z note to PCRANKIN asks for that executable, if the coordinator agrees. Say if it should be mine instead.

## 02:25Z from MOESERVE: correction to my 02:00Z message: under `--zk` the templated read is refused first by `KsScope`

- **Correction.** `verify --zk` refuses a templated session first at `KsScope.checkOf`, not "at `servedForm` alone" as I wrote.
  - The verdict reads: "refused: a circuit that does not parse (verity/flock-circuit/tmpl: a templated statement's circuit is
    read from its template): a statement form the soundness proof does not cover".
  - H1's lanes record exactly this refusal (`benchmarks/pouw/h1_lanes/run.py`'s `TMPL_WHY`, labelled `TEMPLATED`, never ACCEPT).
  - `servedForm` would refuse the session next.
  - So the Lean change has two parts. First, the sampled knowledge soundness, and `KsScope`'s parse, must cover the templated
    form. Then `ProvedScope.served` must take its name.
- **History, for whoever takes it.**
  - `4eac209d8f` (10-26, PRF-133 G-C) admitted a templated statement in the form `ProvedScope.scopeOkT` covers.
  - `b29e4ce07c` (10-28) was built in parallel; neither commit is an ancestor of the other. It added `served` with the two flat
    names. Its message names only seed coins and superseded versions, and `Flock.Test.Served` pins only `Tags.all`, which
    leaves out `Tags.templated`.
  - Pod fixture `verity/ml/flock/pod/90-tmpl-unit.sh` still expects a templated `--zk` session to be accepted.
  - So the `served` refusal looks accidental. `KsScope`'s refusal is the real gap.
- **My side.** The staging code is committed: four commits on `cursor/proofs-moeserve-read-bac5`, not yet pushed. The GLM-5.3
  rerun will print Lean's verdict either way.
- 02:45Z (MOESERVE): **the served read's circuit files will be in r20261108-024228-ba42 (cpu-1, ETA about 03:45Z), and `ho`
  holds by construction.**
  - **Where the files will be.** Once the run finishes, each projection's first scale-read member is at:
    - `/workspace/research/runs/r20261108-024228-ba42/out/prove-w13/circuit.txt` (`EmbeddingRead_v1{V=256,H=96}`);
    - `.../out/prove-w2/circuit.txt` (`{V=256,H=4}`).
  - Beside each:
    - its `pub.bin`, `partition.json` and `program.json`;
    - `unit.tmpl` (the template's text);
    - `tmpl.json` (`flock-tmpl-unit`'s placement at `iv/hybrid/4`);
    - the --zk session, and Lean's verdict in `lean.log` and `verdicts.json`.
  - I'll confirm here when it lands, with the verdict.
  - **Why `ho : ct.outNet = ct.unitNet` holds.**
    - `parseTmpl` calls `HmRow.parse` without a typed template. So `outNet` is META `out_net`, which defaults to the
      unit net.
    - The stager's templated lowering has no tail returns, so it never writes `out_net`.
    - The run's harness also refuses a statement whose META has `out_net`, or whose template isn't the executable's
      text byte for byte.
- 02:51Z (MOESERVE): **correction to my 02:45Z: the run is r20261108-025029-ea20 (cpu-1, ETA about 04:15Z), and the circuit files
  go to a keep directory, not its out/.**
  - r20261108-024228-ba42 had a harness bug that fails before staging. ea20 replaces it.
  - Each templated circuit.txt carries the whole `flock-read-tmpl V H` text in META. That's about 280 MB at either shape,
    even at V=5, H=2.
  - So the proved statements are kept on cpu-1 at `/workspace/research/keep/moeserve/glm53-read-L5-s374-r0/{w13,w2}/`:
    `circuit.txt`, `pub.bin`, `partition.json` and `program.json`.
  - The run's out/ keeps their sha512 and Lean's verdict.

- 03:30Z (MOESERVE): **ea20's verdict on the served reads: w13's read proved under `--zk`, and Lean refused it only at `servedForm`. Its circuit files are kept on cpu-1 for `flock-read-tmpl check`.**
  - The files are from r20261108-025029-ea20, on cpu-1 at `/workspace/research/keep/moeserve/glm53-read-L5-s374-r0/`:
    - `w13/circuit.txt`, `EmbeddingRead_v1{V=256,H=96}`, sha512 `ff4366b3eff1…`;
    - `w2/circuit.txt`, `{V=256,H=4}`, sha512 `fef00533441c…`.
    - Beside each are its `pub.bin`, `partition.json` and `program.json`. Their digests are in the run's `out/prove-{w13,w2}/files.sha512`.
  - Lean's verdict at w13: "refused: verity/flock-circuit/tmpl, a superseded statement version: verify takes only verity/flock-circuit and verity/flock-circuit+plain-leaves".
    - That's `servedForm`, the last check. So on your #7927 Lean, `Zk.verify`, `PortRoots` and `KsScope` passed.
    - **Correction to my 02:25Z: `KsScope` doesn't refuse it first.** `circuitTmpl` inherits `hiddenOutputs` from `circuit`, so `ksParsed` is the templated parse, not `HmOut.parse`.
  - **One more item for #7927's "Not yet"** (from MOESERVE-C's 03:15Z, in my inbox):
    - With `--registered`, setup refuses every served read at `Registered.checkComplete`. The read's `tok` is a wired port (the router's output), and `row`, `salt` and `path` are the opening's witness. Neither kind is a registered port.
    - ea20 verified without `--registered`, so it didn't hold the read's root and domain to B's registration.
    - Can you list it, or say who owns it?
  - w2's proof didn't start, because of my harness's challenger name.
    - r20261108-032704-8b33 (cpu-1, about 04:00Z) reruns both projections on my head merged with your `0ac93f72fe`.
    - It rewrites the same files, with their sha512s in its `out/`.

- 03:37Z (MOESERVE): **correction to my 03:30Z: the read rerun is r20261108-033511-5c04 (cpu-1, ETA about 04:05Z), not -8b33.** 8b33 failed in 7 min, on my harness: its tree lacked MOESERVE-C's `benchmarks/spot_proved/moe_window.py`, which the GLM harness loads, and its stager-test step ran `uv run --extra torch-cpu` inside verity-flock, which has no such extra. 5c04's tree adds C's `5f8e21ecc2` (a scratch merge, `7fd5e8a687`, not pushed), and the tests run on the tree's venv.

- 04:14Z (HASHCUT, proofs, PRF-456): **[#7979](https://github.com/compute-verification/verity/pull/7979) changes nothing your binding proofs read.**
  - It moves no Definition, row-digest or leaf format, and no Lean. It only serves `ResidualAdd_v1` in column blocks, the way MoeSum and the all-reduces already serve, and changes the census and pricing that pick their widths.
  - The next cut would touch what you read, so I'll bring it here before building it. That cut is the FP8 group quant at K = 6144 in column blocks of K = 384.
    - The ScaledMm coordinates would then open the quant's q and s rows either through a row-segment digest over the column blocks, or as 16 rows each.
    - Either way, `ReadUnit`'s row and row-segment bindings would read the new layout.
  - The measurement and the ranked cuts are in `lanes/proofs/20261108T0410Z-report-hashcut.md`.

## 04:16Z from MOESERVE: 5c04's two served read statements, w13 and w2, are preserved with art ids for your `flock-read-tmpl check`

- **The files.** Each art id is a `node-data/v1` tree of four files: `circuit.txt`, `pub.bin`, `partition.json` and
  `program.json`. `research data fetch ART --path circuit.txt` puts the circuit on any machine, and cpu-1 holds the same
  files.
  - **w13**, `EmbeddingRead_v1{V=256,H=96}`:
    - art id `art:0af6dacba76ca3a3940c603776d377121f7a8aadc9ddf9f292bfbcc83a50d68a`;
    - on cpu-1 at `/workspace/research/keep/moeserve/glm53-read-L5-s374-r0/w13/`;
    - `circuit.txt` is 349,543,296 bytes, sha512
      `ff4366b3eff1c199f00b451454b15ce86ddeb04d30c643d5e60d38fa10a8ec28b8045a7b26ca7a29e9e0da2cfcc9dade3185d09ca098736a3b88e0520c4175e7`.
  - **w2**, `EmbeddingRead_v1{V=256,H=4}`:
    - art id `art:71e9db9a04596503f9e52eb9fbdda2627e397026a5cedba6b65b1d2009c5a5b7`;
    - on cpu-1 at `/workspace/research/keep/moeserve/glm53-read-L5-s374-r0/w2/`;
    - `circuit.txt` is 338,939,524 bytes, sha512
      `fef00533441c3f71fc6660bd1da25b5dcd22b5ef67faed04059b44cb2f1c5051c0707291ef54a9797aa7996a003dd9aa925d8652d7273c6e8cc91973181ed479`.
- **The checks:** `flock-read-tmpl check 256 96 <w13>/circuit.txt` and `flock-read-tmpl check 256 4 <w2>/circuit.txt`.
- **These are the bytes 5c04 proved and Lean read.**
  - All four files of each match 5c04's own `files.sha512`.
  - Each `circuit.txt` is byte-identical to ea20's.
  - cpu-1's copies have a retention record: owner @proofs, priority high, until 22 Nov.
- **5c04's verdicts on them:**
  - Both were proved under `--zk`, the live Rust verifier accepted both reps of each, and both challengers' audits
    passed.
  - Lean `--zk` refused both only at `ProvedScope.servedForm`, with "verity/flock-circuit/tmpl, a superseded statement
    version". Every earlier check passed.
  - Neither ran with `--registered`.
- **Owners of the two refusals, from the coordinator:**
  - `ProvedScope.served` is VERIFYFAST-B's (PRF-464).
  - `Registered.checkComplete` with wired and witness ports is MOESERVE-C's.
- I've stopped and run nothing more. Post here if a check fails.

## 04:39Z from VERIFYFAST-B: `verify --zk` accepts a templated session at piece 1's head; rerun the served-read check on `cursor/proofs-verifyfast-b1-986b` at 073b3a110c

- **What ran.** r20261108-043443-310a on vy-nebius-cpu-12, `flock_served.sh MoeExpertFp8GemmW_K256_swap_HopperE4m3MmaV2
  8 --layout iv/hybrid/4 --structural` at 073b3a110c. It runs `flock-verify verify --statement
  verity/flock-circuit/tmpl --zk --partition --program --session` on the MoE expert GEMM's structural templated session
  (8 units), the one `servedForm` refused in r20261107-175923-a370.
- **Verdict: `"accepted":true`.**
  - `ProvedScope.checkOf`, `PortRoots.checkOf` and `KsScope.checkOf` pass on the template, in 0 to 1 ms each.
  - `servedForm` takes `verity/flock-circuit/tmpl`.
  - `verify` took 55.9 s wall and 1.9 GB peak.
- **What it doesn't show.**
  - It is the GEMM's template, not your `EmbeddingRead_v1` reads (w13, w2).
  - It ran without `--registered`, so `Registered.checkComplete`'s refusal, which is MOESERVE-C's, is untouched.
- **To rerun the served-read check:**
  - Use branch `cursor/proofs-verifyfast-b1-986b` at 073b3a110c, and build `flock-verify` from
    `verity/core/service/flock` there.
  - Piece 1 is stacked on piece 2 (`cursor/proofs-verifyfast-b-592a`) and on #7896.
  - Its PR goes up when its records run passes: r20261108-043143-2a4c on cpu-12, about 05:05Z.
  - Post here if a read is refused, with the refusal's text.
