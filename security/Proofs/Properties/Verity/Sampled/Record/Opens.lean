module

public import Proofs.Properties.Verity.Sampled.Record.Runs
public import Proofs.Properties.Verity.Sampled.Record.Accepts
public import Proofs.Properties.Verity.Sampled.Record.Nums
public import Proofs.Flock.Bridge.V5.Session
public import Proofs.Flock.Claims.ZK.MaskedTables
public import Proofs.Flock.Claims.Recursion.VStarParts
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk
import all Init.Data.Array.QSort.Basic

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
  * `Names`, v2's reads of the registered file and `A`'s partition object as v2's: through them, wherever flock-verify
    reads the session's statement at a draw file that fits its length words, the reading is `--zk` at `A`'s inputs,
    with no public input words, and sets up as `A`'s session at the draw file's JSON (`SessionIns`, `sessionIns_of`
    through `read_sess`); the draw file round keeps only a draw file that fits (`drawOk`, `fits_of_drawOk`);
  * `ProvedIn`: every unit the coins draw is one the kept draw file proves, below the population. v2 credits the
    public file's `units.indices` of the instances the draw names (`provedOf`), and refuses a public file whose
    `units.indices` are not the population's units in order, so the indices are the instances themselves
    (`provedIn_of_range`);
  * one table: `SetsUp`'s, as `Spot.OneStageZk.inputsOf` hands v2's inputs;
  * the record's commitment the deployed one (`A.Cd`), which `record_commit` gives at a circuit of `sha512/row/v2`
    rows: the record's commitment hashes each row as a `sha512/row/v2` row.
* `sessionOpens`: `SessionOpens` at SHA-512, at any interior roots: an accepted run kept its draw file
  (`accepted_draw`), whose session's witness satisfies the session's one table (`session_lowers`), the table V5 sets
  up at each drawn unit (`session_table`). The statement stays a variable equal to `xOf A o` throughout (`parse_at`,
  `checked_at`, `session_at`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling
open Lean (Json)
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

/-- **flock-verify reads the session's statement as `A`'s session**: wherever it reads the statement at a draw file
that fits its length words, the reading is `--zk` at `A`'s tags, with no public input words, set up at `A`'s inputs and
the draw file's JSON. -/
def SessionIns (o : Own) : Prop :=
  ∀ (d : Msg) (r : Flock.Readable.Read), Fits A o d → FlockDeploy.readOf true (sessStmt A o d) = some r →
    ∃ J, FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J ∧
      r.zk = true ∧ r.tags = FlockSoundness.Discharge.ZkExec.zkTags A.I ∧ r.publicInputs = none ∧
      r.setup = Flock.Stmt.setupTables (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile
        A.I.publicFile (FlockSoundness.Discharge.ZkExec.zkCoins A.I) A.I.count A.I.tables (some (Flock.norm J))
        A.I.partition A.I.program

/-- **Every unit the coins draw is one the kept draw file proves**, below the population. -/
def ProvedIn (o : Own) : Prop :=
  ∀ d cs (J : Json) (D : Flock.Draw.UnitDraw), drawOk A o d cs → FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J →
    Flock.Draw.UnitDraw.ofJson (Flock.norm J) = .ok D → ∀ u ∈ drawn A (drawBytes cs), u < A.n ∧ u ∈ D.proved

/-! ## v2's units check gives `ProvedIn`

v2 refuses a public file whose `units.indices` are not the population's units `0, …, n − 1` in order
(`Spot.OneStageZk.publicHeld`, `indicesHeld`), and its acceptance gives that match at the loaded header
(`Properties.Spot.Credited.units_of_checkBlobsAt`): `FilePart.indicesOf` reads `Array.range n`. There the
entry at each slot is the slot, so the unit `unitsAt` reads at each instance a draw names is the instance itself. -/

theorem nat?_num (m : ℕ) : _root_.Spot.nat? (.num ⟨Int.ofNat m, 0⟩) = some m := by
  simp [_root_.Spot.nat?, _root_.Spot.int?]

/-- **At such a header every unit a draw proves is a drawn unit, below the population**: `unitsAt` reads the draw's
instance `i` as the header's entry at slot `i`, which is `i`. -/
theorem units_of_range {h D : Json} {n : ℕ} {us : Array ℕ} {Dd : Flock.Draw.UnitDraw}
    (hidx : _root_.Spot.FilePart.indicesOf h = .ok (Array.range n))
    (hu : _root_.Properties.Spot.Units.unitsAt h D = some us)
    (hD : Flock.Draw.UnitDraw.ofJson D = .ok Dd) : ∀ u ∈ us, u < n ∧ u ∈ Dd.units := by
  obtain ⟨⟨a, ha, hm⟩, -, -⟩ := FlockSoundness.Discharge.Composed.ofJson_shape hD
  unfold _root_.Properties.Spot.Units.unitsAt at hu
  obtain ⟨is, his, hu⟩ := Option.bind_eq_some_iff.1 hu
  rw [hidx] at his
  obtain rfl := Option.some.inj his
  try dsimp only at hu
  rw [getD_of_arr ha] at hu
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hu
  obtain ⟨xs, hxs, rfl⟩ := hu
  intro u hu
  rw [FlockSoundness.Audit.DrawExec.mem_qsort] at hu
  obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hu
  rw [Array.mapM_eq_mapM_toList] at hxs
  obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.1 hxs
  obtain ⟨hlen, hall⟩ := _root_.Claims.Flock.mapM_option_some hl
  have hk' : k < a.size := by simpa [hlen] using hk
  have hf := hall k (by simpa using hk) (by simpa using hk')
  simp only [Option.bind_eq_some_iff, Array.getElem_toList] at hf
  obtain ⟨i, hi, hx⟩ := hf
  rw [Array.getElem?_range] at hx
  split at hx
  · rename_i hin
    rw [Option.some.injEq] at hx
    obtain ⟨hDs, hDall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hm
    have hak := FlockComplete.JsonRT.asNat_ok (hDall k hk' (by rw [hDs]; exact hk'))
    rw [hak, nat?_num, Option.some.injEq] at hi
    simp only [List.getElem_toArray]
    rw [← hx]
    refine ⟨hin, ?_⟩
    rw [← hi]
    exact Array.getElem_mem _
  · cases hx

theorem mem_proved {D : Flock.Draw.UnitDraw} {u : ℕ} (h : u ∈ D.units) : u ∈ D.proved := by
  unfold Flock.Draw.UnitDraw.proved
  split
  · exact h
  · exact (FlockSoundness.Audit.DrawExec.mem_qsort _ _ _).2 (Array.mem_append_left _ h)

/-- **v2's units check gives `ProvedIn`**: at a public file whose header's `units.indices` are the population's units
`0, …, A.n − 1`. -/
theorem provedIn_of_range (o : Own)
    (hidx : _root_.Spot.FilePart.indicesOf A.pub.header = .ok (Array.range A.n)) : ProvedIn A o := by
  intro d cs J D hdOk hJ hD u hu
  have hp : _root_.Properties.Spot.Units.unitsAt A.pub.header (Flock.norm J) =
      some ((drawn A (drawBytes cs)).toArray.qsort (· < ·)) := by
    have h := hdOk.1
    unfold provedOf at h
    rw [hJ] at h
    exact h
  obtain ⟨hn, hm⟩ := units_of_range hidx hp hD u
    ((FlockSoundness.Audit.DrawExec.mem_qsort _ _ _).2 (List.mem_toArray.2 hu))
  exact ⟨hn, mem_proved hm⟩

section Names

variable {A} {p : _root_.Spot.OneStageZk.Params} {bs : Core.Blobs}

theorem zkTags_inputsAt (hc1 : A.I.count = 1) :
    Flock.Witness.zkTags (inputsAt A p bs) = FlockSoundness.Discharge.ZkExec.zkTags A.I := by
  unfold FlockSoundness.Discharge.ZkExec.zkTags Flock.Witness.zkTags
  rw [hc1]
  rfl

theorem zkCoins_inputsAt (hc1 : A.I.count = 1) :
    Flock.Witness.zkCoins (inputsAt A p bs) = FlockSoundness.Discharge.ZkExec.zkCoins A.I := by
  unfold FlockSoundness.Discharge.ZkExec.zkCoins Flock.Witness.zkCoins
  rw [hc1]
  rfl

/-- **v2's names give `SessionIns`**, at one table and v2's partition object (`read_sess`). -/
theorem sessionIns_of {o : Own} (hn : Names A o p bs) (hpl : A.I.tags.plainLeaves = false)
    {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd) (hc1 : A.I.count = 1)
    (hpart : A.I.partition = some p.own.partition) : SessionIns A o := by
  intro d r hf hr
  obtain ⟨si, J, rfl, hm, hJ, hdr⟩ := read_sess hn hpl hrd hf hr
  refine ⟨J, hJ, hm.zk, hm.read_tags.trans (zkTags_inputsAt hc1), hm.publicInputs, ?_⟩
  rw [hm.read_setup hdr, zkTags_inputsAt hc1, zkCoins_inputsAt hc1, hc1, hpart, hn.program, ← hn.tables]
  rfl

end Names

/-! ## The run's session -/

theorem take_coinsAt {k : ℕ} (r : Fin k → FlockDeploy.Coin) :
    ((List.range (DRAW_ROUNDS + 1)).map (coinSeq r)).take DRAW_ROUNDS = coinsAt r := by
  rw [← List.map_take, List.take_range, Nat.min_eq_left (by omega : DRAW_ROUNDS ≤ DRAW_ROUNDS + 1)]
  rfl

/-! The record's fields, at a statement `x` held equal to `xOf A o` (`hx`) rather than substituted: a step that unfolds
a field at `xOf A o` itself evaluates the statement's bytes. -/

section Fields

variable (o : Own) (e : Ext)

theorem parse_at {x : Msg} (hx : x = xOf A o) : (record A o e).parse x = some (stmtA A) :=
  (record_parse_eq A o e x).trans (ite_eq_left hx)

theorem checked_at {x : Msg} (hx : x = xOf A o) (r : Fin ((record A o e).V.k x) → FlockDeploy.Coin) :
    (record A o e).checked x r = drawn A (drawBytes (coinsAt r)) := by
  have h : (record A o e).checked x r = if x = xOf A o then drawn A (drawBytes (coinsAt r)) else [] := rfl
  exact h.trans (ite_eq_left hx)

theorem session_at {x : Msg} (hx : x = xOf A o) (u : ℕ) (v : Seen) :
    (record A o e).session x u v = some (sessionOf A o v) :=
  (record_session A o e x u v).trans (ite_eq_left hx)

theorem record_partition (s : AuditStatement) : (record A o e).partition s = some (partA A) := rfl

theorem record_reveal : (record A o e).reveal = e.reveal := rfl

theorem record_flockParse : (record A o e).flock.parse = FlockDeploy.parse true _root_.Flock.Readable.relation := rfl

theorem record_commit (hv2 : Flock.Bridge.RowsV2 A.D.unroll) : (record A o e).commit = A.Cd :=
  (Flock.Bridge.commitDepL_v2 A.Cc A.rows hv2 A.Lm A.roots).symm

end Fields

theorem sha512_msg : FlockDeploy.sha512 = Claims.Sampling.sha512Msg := rfl

theorem seen_msgs (t : Claims.Transcript) : t.seen.msgs = t.msgs := rfl

theorem drawOf_of {v : Seen} {d : Msg} {a : Claims.Out} (hv : v.msgs[DRAW_ROUNDS]? = some (d, a)) : drawOf v = d := by
  simp only [drawOf, List.getD_eq_getElem?_getD, List.getElem?_map, hv, Option.map_some, Option.getD_some]

/-- A view whose draw file round kept `d` at the statement's numbers, `d` fitting its length words, shows the session
at `d`. -/
theorem sessionOf_kept (o : Own) {v : Seen} {d : Msg} (hv : v.msgs[DRAW_ROUNDS]? = some (d, .msg []))
    (hn : numsOk A o d ∧ Fits A o d) :
    sessionOf A o v = (sessStmt A o d, ⟨v.msgs.drop (DRAW_ROUNDS + 1), v.out⟩) := by
  unfold sessionOf
  simp only [hv, hn, and_self, ↓reduceIte]

/-- **An accepted run kept its draw file**: at the record's statement the draw file round's check passed (`run_pre` at
`prefixAt`), on the draw file the run shows and the draw coins. -/
theorem accepted_draw (o : Own) (e : Ext) {x : Msg} (hx : x = xOf A o) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin ((record A o e).V.k x) → FlockDeploy.Coin)
    (hacc : (run FlockDeploy.sha512 P (record A o e).V x ω r).out = .acc) :
    ∃ d, drawOk A o d (coinsAt r) ∧
      (run FlockDeploy.sha512 P (record A o e).V x ω r).seen.msgs[DRAW_ROUNDS]? = some (d, .msg []) := by
  have hpre : SamplingDeploy.PrefixAt FlockDeploy.sha512 (record A o e).V (FlockDeploy.V true) x (DRAW_ROUNDS + 1)
      (kS A o) (okAt A o) (sessStmt A o) := by
    rw [record_V, hx]
    exact prefixAt A o
  obtain ⟨-, hlast, hrej, -⟩ := SamplingDeploy.run_pre hpre P ω r
  have hok : okAt A o (((run FlockDeploy.sha512 P (record A o e).V x ω r).msgs.take (DRAW_ROUNDS + 1)).map Prod.fst)
      ((List.range (DRAW_ROUNDS + 1)).map (coinSeq r)) = true := by
    by_contra h
    have h2 := (hrej (Bool.eq_false_iff.2 h)).symm.trans hacc
    cases h2
  have hdOk := drawOk_of_okAt A o hok
  rw [take_coinsAt] at hdOk
  rw [hok, ite_eq_left rfl, Nat.add_sub_cancel] at hlast
  refine ⟨_, hdOk, ?_⟩
  rw [seen_msgs]
  exact hlast

theorem tags_inputsJ (hc1 : A.I.count = 1) : (inputsJ A.I 0).tags = FlockSoundness.Discharge.ZkExec.zkTags A.I := by
  show FlockSoundness.Discharge.Exec.tagsAt (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.count 0 = _
  unfold FlockSoundness.Discharge.Exec.tagsAt
  exact ite_eq_left (by omega)

theorem nTab_one (hc1 : A.I.count = 1) : nTab A.I = 1 := by
  unfold FlockSoundness.Discharge.Composed.nTab
  exact ite_eq_left (by omega)

/-- **A Flock witness the session's statement checks satisfies its one table**: flock-verify reads the statement at the
kept draw file, which fits its length words (`fits_of_drawOk`), as `A`'s session at the draw file's JSON (`SessionIns`),
which sets up one table (`setupTables_okHP`) that the witness satisfies (`tableLowers`). -/
theorem session_lowers (o : Own) (hc1 : A.I.count = 1) (hins : SessionIns A o) {d : Msg}
    {cs : List FlockDeploy.Coin} (hdOk : drawOk A o d cs) {wF : Claims.Flock.Wit}
    (hchk : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (sessStmt A o d) wF = true) :
    ∃ J, FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J ∧
      ∃ st, Flock.Stmt.setupHidden (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile A.I.publicFile
        (FlockSoundness.Discharge.ZkExec.zkCoins A.I) A.I.tables (some (Flock.norm J)) A.I.partition A.I.program =
        .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      ∀ hwf : FlockSoundness.Refine.StmtWF st,
        (FlockSoundness.Refine.stmtOf st hwf (FlockSoundness.Refine.regionsOf st)).Satisfies
          (Flock.Readable.zOf 1 wF 0 st.m) := by
  obtain ⟨pp, hpb, hpP, hR⟩ := Flock.Readable.relS_of_check (decode := FlockDeploy.readOf true) hchk
  unfold Flock.Readable.parseBytes at hpb
  obtain ⟨rr, hrr, hpb⟩ := Option.bind_eq_some_iff.1 hpb
  obtain ⟨J, hJ, hzk, htags, hpin, hsetup⟩ := hins d rr (fits_of_drawOk hdOk) hrr
  refine ⟨J, hJ, ?_⟩
  unfold Flock.Readable.Read.parse at hpb
  obtain ⟨sts, hts, rfl⟩ := Option.map_eq_some_iff.1 hpb
  have hsh := Flock.Readable.tablesOf_shapeOk hts hzk
  obtain ⟨spec, hsu⟩ := Flock.Readable.setup_of_tablesOf hts
  rw [hsetup] at hsu
  have hh : (FlockSoundness.Discharge.ZkExec.zkTags A.I).hiddenOutputs = true :=
    (FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs A.I).trans A.hTags.1
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hsu with ⟨-, st, hst, -, rfl⟩ | ⟨hc, -⟩
  · refine ⟨st, hst, FlockSoundness.Discharge.ZkExec.array_forM_ok hsh 0 (by simp), fun hwf => ?_⟩
    have ht : rr.tags.typed = false := by
      rw [htags, FlockSoundness.Discharge.ZkExec.zkTags_typed]
      exact A.ht
    exact Flock.Readable.tableLowers (j := 0) (st := st) hpP ht hpin hR rfl hwf
  · exact absurd hc (by omega)

/-- **The session's table at a drawn unit is V5's**: where the kept draw file proves the unit, the session's one table
sets up at it (`TableSetup`, `drawSetupHP_of_setupHiddenI`), and every such setup is that table's. -/
theorem session_table (o : Own) (hc1 : A.I.count = 1) (hpr : ProvedIn A o) {d : Msg} {cs : List FlockDeploy.Coin}
    (hdOk : drawOk A o d cs) {J : Json} (hJ : FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J) {st : Flock.Stmt}
    (hst : Flock.Stmt.setupHidden (FlockSoundness.Discharge.ZkExec.zkTags A.I) A.I.circuitFile A.I.publicFile
      (FlockSoundness.Discharge.ZkExec.zkCoins A.I) A.I.tables (some (Flock.norm J)) A.I.partition A.I.program = .ok st)
    (hshape : Flock.Zk.shapeOk st = .ok ()) {u : ℕ} (hud : u ∈ drawn A (drawBytes cs)) :
    ∃ hun : u < A.n, Nonempty (TableSetup A ⟨u, hun⟩ (Flock.norm J)) ∧
      ∀ y : TableSetup A ⟨u, hun⟩ (Flock.norm J), y.2.x.st = st := by
  have htJ := tags_inputsJ A hc1
  have hstJ : Flock.Stmt.setupHidden (inputsJ A.I 0).tags (inputsJ A.I 0).circuitFile (inputsJ A.I 0).publicFile
      (inputsJ A.I 0).coins (inputsJ A.I 0).tables (some (Flock.norm J)) (inputsJ A.I 0).partition
      (inputsJ A.I 0).program none = .ok st := by
    rw [htJ]
    exact hst
  obtain ⟨⟨sch, tPad⟩, hz⟩ := FlockSoundness.Discharge.ZkHidden.schedule_ok_of_shapeOk hshape
  obtain ⟨sch₀, hf⟩ := FlockSoundness.Discharge.ZkHidden.fast100_of_schedule hz
  obtain ⟨pi', pop, -, -, hdr⟩ := FlockSoundness.Discharge.Hidden.setupHidden_drawnP hst
  obtain ⟨D, hD, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec
    (c := { st.c with outPorts := if pi'.ports.isEmpty then #[] else #[("public inputs", pi'.words)] }) hdr
  obtain ⟨hun, huD⟩ := hpr d cs J D hdOk hJ hD u hud
  have hmPts : st.mPts = mPtsOf A.D.unroll.g (D.proved.size / nTab A.I) := by
    rw [nTab_one A hc1, Nat.div_one]
    exact FlockSoundness.Discharge.ZkHidden.setupHidden_mPts_of_proved (Flock.Desc.parse_flat A.hD)
      (fun D' hD' => by rw [Except.ok.inj (hD'.symm.trans hD)]) hst
  obtain ⟨xx, hxx⟩ := FlockSoundness.Discharge.HiddenExec.drawSetupHP_of_setupHiddenI (I := inputsJ A.I 0)
    (own := none) (S := {(⟨u, hun⟩ : Fin A.n)}) (FlockSoundness.Discharge.Composed.typedJ A.ht 0) hstJ hf hD
    (fun v hv => by rw [Finset.mem_singleton.1 hv]; exact huD) hmPts
  obtain ⟨y, -, -, -⟩ := FlockSoundness.Discharge.ZkHidden.drawSetupZHJP_of xx (by rw [hxx]; exact hshape)
    (sch := sch) (tPad := tPad) (by rw [hxx]; exact hz)
  exact ⟨hun, ⟨⟨D.proved.size, y⟩⟩, fun y' => Except.ok.inj (y'.2.x.h.symm.trans hstJ)⟩

/-- A witness satisfying a statement's table satisfies the `--zk` table of every setup of it. -/
theorem table_satisfies {mPts : ℕ} {S₀ : Finset (Fin A.n)} {d₀ : Json}
    (y : DrawSetupZHJP (inputsJ A.I 0) none mPts S₀ d₀) {st : Flock.Stmt} (hy : y.x.st = st) {wF : Claims.Flock.Wit}
    (hsat : ∀ hwf : FlockSoundness.Refine.StmtWF st,
      (FlockSoundness.Refine.stmtOf st hwf (FlockSoundness.Refine.regionsOf st)).Satisfies
        (Flock.Readable.zOf 1 wF 0 st.m)) :
    y.table.S.Satisfies (Flock.Readable.zOf 1 wF 0 y.table.S.m) := by
  subst hy
  exact hsat y.x.wf.1

/-- `revealRec` at a unit whose table sets up at the run's draw file is `revealAt` at the chosen setup. -/
theorem revealRec_eq {u : ℕ} (hun : u < A.n) {v : Seen} {d : Msg} (hdv : drawOf v = d) {J : Json}
    (hJ : FlockDeploy.jsonOf "draw" ⟨d.toArray⟩ = .ok J) (hne : Nonempty (TableSetup A ⟨u, hun⟩ (Flock.norm J)))
    (s : AuditStatement) (wF : Claims.Flock.Wit) :
    revealRec A s u v wF = revealAt A ⟨u, hun⟩ (Flock.norm J) (Classical.choice hne).2 wF := by
  unfold revealRec
  rw [dite_eq_left hun, hdv, hJ]
  exact dite_eq_left hne

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
correct, whatever its interior values. At a session v2 sets up (`SetsUp`, one table among its inputs): where v2 reads
`A`'s and `o`'s inputs (`Names`, with v2's reads and partition object), flock-verify reads the session as `A`'s
(`sessionIns_of`); where the public file's `units.indices` are the population's units in order, as v2 holds them
(`units_of_checkBlobsAt`), the kept draw file proves every drawn unit (`provedIn_of_range`); and where the record's
commitment is the deployed one (`hcd`; `record_commit` at a circuit of `sha512/row/v2` rows), an opening under it is
one under `A`'s. -/
theorem sessionOpens (o : Own) (e : Ext) {ι : Type} {dv : Accounting.Device ι} {α : Type} [Fintype α]
    {V : Vocab dv α} {θ : ByteArray} {F : Array ByteArray} (hsu : SetsUp V θ F A)
    {p : _root_.Spot.OneStageZk.Params} {bs : Core.Blobs}
    (hn : Names A o p bs) (hpl : A.I.tags.plainLeaves = false) {rd : _root_.Spot.OneStageZk.Reads}
    (hrd : _root_.Spot.OneStageZk.readsOf p.own.registered = .ok rd)
    (hpart : A.I.partition = some p.own.partition)
    (hidx : _root_.Spot.FilePart.indicesOf A.pub.header = .ok (Array.range A.n))
    (hcd : (record A o e).commit = A.Cd)
    (he : e.reveal = revealRec A)
    (iroots : Msg → Seen → ℕ → List Msg) : SessionOpens FlockDeploy.sha512 (record A o e) iroots := by
  have hc1 := hsu.count
  have hpr := provedIn_of_range A o hidx
  intro x s Pt Ω P ω r u xS vS wF hs hPt _ hacc hu hsess hchk
  obtain ⟨⟨i, hi⟩, hq⟩ := hu
  have hx := eq_xOf_of_nPU A o e (Nat.lt_of_le_of_lt (Nat.zero_le i) hi)
  obtain rfl : s = stmtA A := Option.some.inj (hs.symm.trans (parse_at A o e hx))
  obtain rfl : Pt = partA A := Option.some.inj (hPt.symm.trans (record_partition A o e _))
  obtain ⟨d, hdOk, hl⟩ := accepted_draw A o e hx P ω r hacc
  rw [checked_at A o e hx] at hq
  have hud : u ∈ drawn A (drawBytes (coinsAt r)) := List.mem_of_getElem? hq
  rw [session_at A o e hx, sessionOf_kept A o hl hdOk.2] at hsess
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hsess)
  rw [record_flockParse] at hchk
  obtain ⟨J, hJ, st, hst, hshape, hsat⟩ :=
    session_lowers A o hc1 (sessionIns_of hn hpl hrd hc1 hpart) hdOk hchk
  obtain ⟨hun, hne, hy⟩ := session_table A o hc1 hpr hdOk hJ hst hshape hud
  refine ⟨Flock.Bridge.blockOf A.P A.κ ⟨u, hun⟩,
    Flock.Bridge.blocks_partOf_getElem? A.P (Flock.Bridge.nsOf_sum A.hpop) ⟨u, hun⟩, fun val ι hval => ?_⟩
  have hrev : (record A o e).reveal (stmtA A) u (run FlockDeploy.sha512 P (record A o e).V x ω r).seen wF =
      revealAt A ⟨u, hun⟩ (Flock.norm J) (Classical.choice hne).2 wF := by
    rw [record_reveal, he]
    exact revealRec_eq A hun (drawOf_of hl) hJ hne _ wF
  refine correct_revealAt A (by omega) (Classical.choice hne).2
    (table_satisfies A (Classical.choice hne).2 (hy (Classical.choice hne)) hsat) val ι fun q hq ho hr => ?_
  exact (hval q (by rw [hrev]; exact hq) (by rw [hcd, sha512_msg]; exact ho)).1 hr

end Verity.Sampled.Record
