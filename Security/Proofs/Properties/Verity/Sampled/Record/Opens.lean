import Proofs.Properties.Verity.Sampled.Record.Runs
import Proofs.Properties.Verity.Sampled.Record.Accepts
import Proofs.Flock.Bridge.V5.Session
import Proofs.Flock.Claims.ZK.MaskedTables
import Proofs.Flock.Claims.Recursion.VStarParts
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk

/-!
# A checked unit's Flock witness opens its proof block correct (`SessionOpens`)

At the record's statement every checked unit's session is the one flock-verify `--zk` session at the statement the
draw file sets up (`sessionOf`). A Flock witness in that session's relation satisfies its one table (`tableLowers`),
which is V5's drawn table for the unit (`tabOfZHJP_one`), so the message packing it (`packZ`) reveals openings that
make the unit's proof block correct wherever the values carry those that open (`correct_of_revealV5`).

* **What a witness reveals** (`revealRec`, the record's `Ext.reveal`): at unit `u` of the run's draw file, V5's
  openings (`revealV5`) of the message packing the witness's bits at the session's table, set up at that draw file and
  the unit (`TableSetup`); none where it doesn't set up. The openings are computed from the witness; only the table's
  setup is chosen, and it is unique (`DrawSetupZHJP.table_congr`). At most one per committed wire (`revealsLe`).
* **What the record takes beside acceptance**:
  * `SessionIns`: flock-verify's reading of the session's statement at every draw file is `--zk` at `A`'s inputs, with
    no public input words, and sets up as `A`'s session at that draw. v2's names give it (`sessionIns_of`);
  * `KeptFits`: every draw file the draw file round keeps fits the statement's length words (`Fits`);
  * `ProvedIn`: every unit the coins draw is one the kept draw file proves, below the population. v2 credits the
    public file's `units.indices` of the instances the draw names (`provedOf`), so this holds where those indices are
    the instances themselves.
* `sessionOpens`: `SessionOpens` at SHA-512, at one table and a circuit whose rows are `sha512/row/v2`'s, at any
  interior roots.
-/

namespace Verity.Sampled.Record

open Claims Claims.Sampling
open Lean (Json)
open FlockDeploy (Ins)
open FlockSoundness.Discharge.Exec (mPtsOf)
open FlockSoundness.Discharge.Composed (inputsJ nTab)
open FlockSoundness.Discharge.ZkHidden (DrawSetupZHJP planZHJP tabZHJP)
open FlockSoundness.Discharge.ZkSession (TabZK tabOfZ)

variable (A : Live)

/-! ## What a session's witness reveals -/

/-- **The session's table set up at unit `u` and the draw `J`**, at some proved-unit count `kd`. -/
abbrev TableSetup (u : Fin A.n) (J : Json) : Type :=
  Σ kd : ℕ, DrawSetupZHJP (inputsJ A.I 0) none (mPtsOf A.D.unroll.g (kd / nTab A.I)) {u} J

/-- Unit `u`'s table at the setup `y`, as V5 reads it. -/
noncomputable abbrev unitTab (u : Fin A.n) (J : Json) {mPts : ℕ} {S₀ : Finset (Fin A.n)} {d₀ : Json}
    (y : DrawSetupZHJP (inputsJ A.I 0) none mPts S₀ d₀) : TabZK FlockLevel3.GF128 mPts :=
  tabOfZ (planZHJP (Reg := Unit) (fun _ => J) y) (tabZHJP (fun _ => J) y) {u} () u

/-- **The openings a witness reveals at unit `u`'s table**: V5's openings of the message packing the witness's bits at
the table's statement. -/
noncomputable def revealAt (u : Fin A.n) (J : Json) {mPts : ℕ} {S₀ : Finset (Fin A.n)} {d₀ : Json}
    (y : DrawSetupZHJP (inputsJ A.I 0) none mPts S₀ d₀) (wF : Claims.Flock.Wit) : List (List Msg × Opening Bool) :=
  Flock.Bridge.revealV5 A.ht A.hD A.hU A.hpi A.hpub A.hscope (fun _ => J) y {u} () u
    (Flock.Bridge.packZ (unitTab A u J y) (Flock.Readable.zOf 1 wF 0 (unitTab A u J y).S.m))

open Classical in
/-- `revealAt` at the draw file's reading, where the table sets up. -/
noncomputable def revealJ (u : Fin A.n) (wF : Claims.Flock.Wit) :
    Except String Json → List (List Msg × Opening Bool)
  | .ok J =>
    if h : Nonempty (TableSetup A u (Flock.norm J)) then revealAt A u (Flock.norm J) (Classical.choice h).2 wF else []
  | .error _ => []

/-- **What a session's witness reveals at unit `u`**: `revealAt` at the run's draw file, at a unit of the population. -/
noncomputable def revealRec (_s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    List (List Msg × Opening Bool) :=
  if hu : u < A.n then revealJ A ⟨u, hu⟩ wF (FlockDeploy.jsonOf "draw" ⟨(drawOf v).toArray⟩) else []

theorem revealAt_length_le (u : Fin A.n) (J : Json) {mPts : ℕ} {S₀ : Finset (Fin A.n)} {d₀ : Json}
    (y : DrawSetupZHJP (inputsJ A.I 0) none mPts S₀ d₀) (wF : Claims.Flock.Wit) :
    (revealAt A u J y wF).length ≤ A.Cc.N :=
  (Flock.Bridge.revealV5_length_le A.ht A.hD A.hU A.hpi A.hpub A.hscope (fun _ => J) y {u} () u
    (Flock.Bridge.packZ (unitTab A u J y) (Flock.Readable.zOf 1 wF 0 (unitTab A u J y).S.m))).trans
    ((Finset.card_le_univ _).trans_eq (Fintype.card_fin _))

theorem revealJ_length_le (u : Fin A.n) (wF : Claims.Flock.Wit) (j : Except String Json) :
    (revealJ A u wF j).length ≤ A.Cc.N := by
  cases j with
  | error _ => exact Nat.zero_le _
  | ok J =>
    simp only [revealJ]
    split
    · exact revealAt_length_le A u _ _ wF
    · exact Nat.zero_le _

theorem revealRec_length_le (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    (revealRec A s u v wF).length ≤ A.Cc.N := by
  unfold revealRec
  split
  · exact revealJ_length_le A _ wF _
  · exact Nat.zero_le _

/-- **At most one opening per committed wire**, where the record's reveal is `revealRec` and its count at least the
circuit's wires. -/
theorem revealsLe (o : Own) (e : Ext) (he : e.reveal = revealRec A) (hN : ∀ x, A.Cc.N ≤ e.reveals x) :
    SamplingDeploy.RevealsLe (code A o e) := by
  intro x s _ u v wF
  show (e.reveal s u v wF).length ≤ e.reveals x
  rw [he]
  exact (revealRec_length_le A s u v wF).trans (hN x)

/-! ## What the record takes -/

/-- **flock-verify reads the session's statement as `A`'s session**: at every draw file, `--zk` at `A`'s tags, with
no public input words, set up at `A`'s inputs and the draw. -/
def SessionIns (o : Own) : Prop :=
  ∀ (d : Msg) (J : Json), FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J →
    ∃ si : Ins, ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok si ∧
      si.read.zk = true ∧ si.read.tags = FlockSoundness.Discharge.ZkExec.zkTags A.I ∧ si.read.publicInputs = none ∧
      si.read.setup = Flock.Stmt.setupTables (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile
        A.I.publicFile (FlockSoundness.Discharge.ZkExec.zkCoins A.I) A.I.count A.I.tables (some (Flock.norm J))
        A.I.partition A.I.program

/-- **Every kept draw file fits the statement's length words.** -/
def KeptFits (o : Own) : Prop := ∀ d cs, drawOk A o d cs → Fits A o d

/-- **Every unit the coins draw is one the kept draw file proves**, below the population. -/
def ProvedIn (o : Own) : Prop :=
  ∀ d cs (J : Json) (D : Flock.Draw.UnitDraw), drawOk A o d cs → FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J →
    Flock.Draw.UnitDraw.ofJson (Flock.norm J) = .ok D → ∀ u ∈ drawn A (drawBytes cs), u < A.n ∧ u ∈ D.proved

section Names

variable {A} {p : _root_.SampledProofs.OneStageZk.Params} {bs : Core.Blobs}

theorem zkTags_inputsAt (hc1 : A.I.count = 1) :
    Flock.Witness.zkTags (inputsAt A p bs) = FlockSoundness.Discharge.ZkExec.zkTags A.I := by
  unfold FlockSoundness.Discharge.ZkExec.zkTags FlockSoundness.Discharge.ZkExec.zkCoins Flock.Witness.zkTags
    Flock.Witness.zkCoins
  (rw [hc1]) <;> rfl

theorem zkCoins_inputsAt (hc1 : A.I.count = 1) :
    Flock.Witness.zkCoins (inputsAt A p bs) = FlockSoundness.Discharge.ZkExec.zkCoins A.I := by
  unfold FlockSoundness.Discharge.ZkExec.zkCoins Flock.Witness.zkCoins
  (rw [hc1]) <;> rfl

/-- **v2's names give `SessionIns`**, at one table and v2's partition object (`ins_sess`). -/
theorem sessionIns_of {o : Own} (hn : Names A o p bs) (hpl : A.I.tags.plainLeaves = false)
    {rd : _root_.SampledProofs.OneStageZk.Reads}
    (hrd : _root_.SampledProofs.OneStageZk.readsOf p.own.registered = .ok rd) (hc1 : A.I.count = 1)
    (hpart : A.I.partition = some p.own.partition) : SessionIns A o := by
  intro d J hJ
  obtain ⟨si, hi, hm, hdr⟩ := ins_sess hn hpl hrd hJ
  refine ⟨si, hi, hm.zk, hm.read_tags.trans (zkTags_inputsAt hc1), hm.publicInputs, ?_⟩
  (rw [hm.read_setup hdr, zkTags_inputsAt hc1, zkCoins_inputsAt hc1, hc1, hpart, hn.program, ← hn.tables]) <;> rfl

end Names

/-! ## The run's session -/

theorem jsonOf_of_provedOf {d : Msg} {a : Array ℕ} (h : provedOf A d = some a) :
    ∃ J, FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J := by
  unfold provedOf at h
  cases hj : FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ with
  | error _ => simp [hj, Except.toOption] at h
  | ok J => exact ⟨J, rfl⟩

theorem take_coinsAt {k : ℕ} (r : Fin k → FlockDeploy.Coin) :
    ((List.range (DRAW_ROUNDS + 1)).map (coinSeq r)).take DRAW_ROUNDS = coinsAt r := by
  (rw [← List.map_take, List.take_range, Nat.min_eq_left (by omega : DRAW_ROUNDS ≤ DRAW_ROUNDS + 1)]) <;> rfl

theorem readOf_raw {w : FlockDeploy.Raw} (hfits : Verity.Recursive.Part.RawFits w) (hzk : w.zk = true) {si : Ins}
    (hins : w.ins = .ok si) : FlockDeploy.readOf true w.encode = some si.read := by
  unfold FlockDeploy.readOf FlockDeploy.insOf
  rw [Verity.Recursive.Part.decode_encode_fits hfits]
  simp [hzk, hins, Except.toOption]

/-! ## The openings make the proof block correct -/

/-- **A witness satisfying unit `u`'s table reveals openings that make `u`'s proof block correct**, at one table
(`tabOfZHJP_one`, `wit_packZ`, `correct_of_revealV5`). -/
theorem correct_revealAt (hc1 : A.I.count ≤ 1) {u : Fin A.n} {J : Json} {kd : ℕ}
    (y : DrawSetupZHJP (inputsJ A.I 0) none (mPtsOf A.D.unroll.g (kd / nTab A.I)) {u} J) {wF : Claims.Flock.Wit}
    (hsat : y.table.S.Satisfies (Flock.Readable.zOf 1 wF 0 y.table.S.m)) (val : ℕ → Option Bool) (ι : ℕ → Bool)
    (hval : ∀ q ∈ revealAt A u J y wF, A.Cd.Opens Claims.Sampling.sha512Msg q.1 q.2 = true → q.1 = A.roots →
      val q.2.pos = some q.2.val) :
    (Flock.Bridge.progOf A.Cc A.κ).Correct
      ((Flock.Bridge.progOf A.Cc A.κ).refIn (Flock.Bridge.blockOf A.P A.κ u) val ι)
      (Flock.Bridge.blockOf A.P A.κ u) := by
  have hT : unitTab A u J y = y.table := Flock.Bridge.tabOfZHJP_one hc1 y y () u
  have hM : (unitTab A u J y).S.Satisfies (Flock.Readable.zOf 1 wF 0 (unitTab A u J y).S.m) := by
    rw [hT]
    exact hsat
  have h2 := Flock.Bridge.wit_packZ (unitTab A u J y) (Flock.Readable.zOf 1 wF 0 (unitTab A u J y).S.m)
  exact Flock.Bridge.correct_of_revealV5 A.n Unit A.I none A.ht A.D A.hD A.hU _ A.hpi A.pub A.hpub A.hscope A.hdup
    A.hroots kd (fun _ => J) {u} J y {u} () u (Finset.mem_singleton_self u)
    (Flock.Bridge.packZ (unitTab A u J y) (Flock.Readable.zOf 1 wF 0 (unitTab A u J y).S.m))
    (Eq.mpr (congrArg (fun z => (unitTab A u J y).S.Satisfies z) h2) hM) val ι hval

/-- **`SessionOpens` at the record**, at SHA-512: on an accepted run, a checked unit's session's Flock witness reveals
(`revealRec`) openings such that values carrying those that open under the statement's roots make the unit's proof block
correct, whatever its interior values. -/
theorem sessionOpens (o : Own) (e : Ext) (hv2 : Flock.Bridge.RowsV2 A.D.unroll) (hc1 : A.I.count = 1)
    (hins : SessionIns A o) (hfits : KeptFits A o) (hpr : ProvedIn A o) (he : e.reveal = revealRec A)
    (iroots : Msg → Seen → ℕ → List Msg) : SessionOpens FlockDeploy.sha512 (record A o e) iroots := by
  intro x s Pt Ω P ω r u xS vS wF hs hPt _ hacc hu hsess hchk
  obtain ⟨⟨i, hi⟩, hq⟩ := hu
  obtain rfl := eq_xOf_of_nPU A o e (Nat.lt_of_le_of_lt (Nat.zero_le i) hi)
  change Fin ((V A o).k (xOf A o)) → FlockDeploy.Coin at r
  have hs' : (record A o e).parse (xOf A o) = some (stmtA A) := (record_parse A o e (xOf A o)).trans (ite_eq_left rfl)
  obtain rfl := Option.some.inj (hs.symm.trans hs')
  have hPt' : (record A o e).partition (stmtA A) = some (partA A) := rfl
  obtain rfl := Option.some.inj (hPt.symm.trans hPt')
  obtain ⟨-, hlast, hrej, -⟩ := SamplingDeploy.run_pre (prefixAt A o) P ω r
  have hok : okAt A o (((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.take (DRAW_ROUNDS + 1)).map Prod.fst)
      ((List.range (DRAW_ROUNDS + 1)).map (coinSeq r)) = true := by
    by_contra h
    have h2 := (hrej (Bool.eq_false_iff.2 h)).symm.trans hacc
    cases h2
  have hdOk := drawOk_of_okAt A o hok
  rw [take_coinsAt] at hdOk
  simp only [hok, ↓reduceIte, Nat.add_sub_cancel] at hlast
  generalize (((run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.take (DRAW_ROUNDS + 1)).map Prod.fst).getD
    DRAW_ROUNDS [] = d at hdOk hlast
  obtain ⟨J, hJ⟩ := jsonOf_of_provedOf A hdOk.1
  have hck : (record A o e).checked (xOf A o) r = drawn A (drawBytes (coinsAt r)) :=
    (record_checked A o e (xOf A o) r).trans (ite_eq_left rfl)
  rw [hck] at hq
  have hud : u ∈ drawn A (drawBytes (coinsAt r)) := List.mem_of_getElem? hq
  have hl : (run FlockDeploy.sha512 P (record A o e).V (xOf A o) ω r).seen.msgs[DRAW_ROUNDS]? = some (d, .msg []) :=
    hlast
  rw [record_session_xOf A o e u] at hsess
  unfold sessionOf at hsess
  simp only [hl, ite_eq_left hdOk.2, Option.some.injEq, Prod.mk.injEq] at hsess
  obtain ⟨rfl, rfl⟩ := hsess
  -- the session's statement as flock-verify reads it
  obtain ⟨si, hraw, hzk, htags, hpin, hsetup⟩ := hins d J hJ
  have hread : FlockDeploy.readOf true (sessStmt A o d) = some si.read := readOf_raw (hfits d _ hdOk) rfl hraw
  obtain ⟨pp, hpb, hpP, hR⟩ := Flock.Readable.relS_of_check (decode := FlockDeploy.readOf true) hchk
  unfold Flock.Readable.parseBytes at hpb
  rw [hread] at hpb
  change si.read.parse = some pp at hpb
  unfold Flock.Readable.Read.parse at hpb
  obtain ⟨sts, hts, rfl⟩ := Option.map_eq_some_iff.1 hpb
  have hsh := Flock.Readable.tablesOf_shapeOk hts hzk
  obtain ⟨spec, hsu⟩ := Flock.Readable.setup_of_tablesOf hts
  rw [hsetup] at hsu
  have hh : (FlockSoundness.Discharge.ZkExec.zkTags A.I).hiddenOutputs = true :=
    (FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hsu with ⟨-, st, hst, -, rfl⟩ | ⟨hc, -⟩
  swap
  · exact absurd hc (by omega)
  -- the session's one table, as V5 sets it up at the unit
  have hshape : Flock.Zk.shapeOk st = .ok () := FlockSoundness.Discharge.ZkExec.array_forM_ok hsh 0 (by simp)
  obtain ⟨⟨sch, tPad⟩, hz⟩ := FlockSoundness.Discharge.ZkHidden.schedule_ok_of_shapeOk hshape
  obtain ⟨sch₀, hf⟩ := FlockSoundness.Discharge.ZkHidden.fast100_of_schedule hz
  have hn1 : nTab A.I = 1 := by simp [nTab, hc1]
  have htJ : (inputsJ A.I 0).tags = FlockSoundness.Discharge.ZkExec.zkTags A.I := by
    show FlockSoundness.Discharge.Exec.tagsAt (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.count 0 = _
    unfold FlockSoundness.Discharge.Exec.tagsAt
    exact ite_eq_left (by omega)
  have hstJ : Flock.Stmt.setupHidden (inputsJ A.I 0).tags (inputsJ A.I 0).circuitFile (inputsJ A.I 0).publicFile
      (inputsJ A.I 0).coins (inputsJ A.I 0).tables (some (Flock.norm J)) (inputsJ A.I 0).partition
      (inputsJ A.I 0).program none = .ok st := by
    rw [htJ]
    exact hst
  have hcJ : Flock.HmOut.parse (inputsJ A.I 0).tags (inputsJ A.I 0).circuitFile (inputsJ A.I 0).tables =
      .ok A.D.unroll := by
    rw [htJ]
    exact Flock.Desc.parse_flat A.hD
  obtain ⟨pi', pop, -, -, hdr⟩ := FlockSoundness.Discharge.Hidden.setupHidden_drawnP hst
  obtain ⟨D, hD, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec
    (c := { st.c with outPorts := if pi'.ports.isEmpty then #[] else #[("public inputs", pi'.words)] }) hdr
  obtain ⟨hun, huD⟩ := hpr d (coinsAt r) J D hdOk hJ hD u hud
  have hmPts : st.mPts = mPtsOf A.D.unroll.g (D.proved.size / nTab A.I) := by
    rw [hn1, Nat.div_one]
    exact FlockSoundness.Discharge.ZkHidden.setupHidden_mPts_of_proved (Flock.Desc.parse_flat A.hD)
      (fun D' hD' => by rw [Except.ok.inj (hD'.symm.trans hD)]) hst
  obtain ⟨xx, hxx⟩ := FlockSoundness.Discharge.HiddenExec.drawSetupHP_of_setupHiddenI (I := inputsJ A.I 0)
    (own := none) (S := {(⟨u, hun⟩ : Fin A.n)}) (FlockSoundness.Discharge.Composed.typedJ A.ht 0) hcJ hstJ hf hD
    (fun v hv => by rw [Finset.mem_singleton.1 hv]; exact huD) hmPts
  obtain ⟨y, hyx, -, -⟩ := FlockSoundness.Discharge.ZkHidden.drawSetupZHJP_of xx (by rw [hxx]; exact hshape)
    (sch := sch) (tPad := tPad) (by rw [hxx]; exact hz)
  have hne : Nonempty (TableSetup A ⟨u, hun⟩ (Flock.norm J)) := ⟨⟨D.proved.size, y⟩⟩
  refine ⟨Flock.Bridge.blockOf A.P A.κ ⟨u, hun⟩,
    Flock.Bridge.blocks_partOf_getElem? A.P (Flock.Bridge.nsOf_sum A.hpop) ⟨u, hun⟩, fun val ι hval => ?_⟩
  -- what the witness reveals is `revealAt` at the chosen setup, under the deployed commitment
  have hcm : (record A o e).commit = A.Cd := (Flock.Bridge.commitDepL_v2 A.Cc A.rows hv2 A.Lm A.roots).symm
  have hH : FlockDeploy.sha512 = Claims.Sampling.sha512Msg := rfl
  have hdv : drawOf (run FlockDeploy.sha512 P (record A o e).V (xOf A o) ω r).seen = d := by
    unfold drawOf
    (rw [List.getD_eq_getElem?_getD, List.getElem?_map, hl]) <;> rfl
  have hrev : (record A o e).reveal (stmtA A) u (run FlockDeploy.sha512 P (record A o e).V (xOf A o) ω r).seen wF =
      revealAt A ⟨u, hun⟩ (Flock.norm J) (Classical.choice hne).2 wF := by
    show e.reveal _ _ _ _ = _
    rw [he]
    unfold revealRec
    rw [dite_eq_left hun, hdv, hJ]
    exact dite_eq_left hne
  have hy'st : (Classical.choice hne).2.x.st = st := Except.ok.inj ((Classical.choice hne).2.x.h.symm.trans hstJ)
  refine correct_revealAt A (by omega) (Classical.choice hne).2 (wF := wF) ?_ val ι fun q hq ho hr => ?_
  · exact Flock.Readable.tableLowers hpP
      (by
        show si.read.tags.typed = false
        rw [htags, FlockSoundness.Discharge.ZkExec.zkTags_typed]
        exact A.ht)
      (by
        show si.read.publicInputs = none
        exact hpin)
      hR (j := 0) (st := (Classical.choice hne).2.x.st) (by (rw [hy'st]) <;> rfl) (Classical.choice hne).2.x.wf.1
  · exact (hval q (by rw [hrev]; exact hq) (by rw [hcm, hH]; exact ho)).1 hr

end Verity.Sampled.Record
