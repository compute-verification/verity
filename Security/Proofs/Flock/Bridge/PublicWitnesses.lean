import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Soundness.Game.Prob

/-!
# `SampledKSLivePub`'s honest and nontrivial lemmas (LEAN-66)

* **Honest** (`sampledKSLivePub_honest`): the hypotheses of `SampledKSLivePub` on the analysis' parameters hold at the
  deployed ones (`Bridge.kDep`, `ρ = 1/2`, `M = R_w = 2^192`, `t' ≥ 0`), and there `deltaPub < 1`
  (`Flock.SecurityProofs.DeltaPubDeployed`).
* **The failing instance, constructed** (`Bridge.toy_exceeds`): a two-gate circuit (an input and its negation, the
  negation returned) in one unit, drawn by a draw of size 1 (`n = k_d = K₀ = 1`), under a commitment with no rows, so
  that nothing opens and every value the extractor names fails the claim. There, for every game and every strategy,
  `Pr[accept ∧ ¬ RelS] = 1 > deltaPub` when the verifier accepts everything. It is built by hand at the model level
  (`FlockSoundness.Audit.Circuit`, `Bridge.commitDep`), with no premise.
* **The transfer** (`Bridge.not_relS_unopened`, `Bridge.exceeds_of_not`): at any circuit whose commitment opens nothing,
  a computed output fails the claim for every witness when every unit is drawn; and an event that never holds has
  probability 1 under every strategy. V2's, V3's and V4's `_nontrivial` read the instance through these.
* **Nontrivial** (`sampledKSLivePub_nontrivial`): `SampledKSLivePubAll`, `SampledKSLivePub` at the verifier that
  accepts everything, is false at any setting of its file-level fields whose circuit has no port and computes a
  returned output, with every unit drawn. `¬ RelS` is proved there, for every run. What stays a binder is what only the
  files or a named assumption give: the parses `hD` and `hpub` (`Flock.Desc.parse` and `HmOut.loadPublic` read the files
  through `Lean.Json.parse`, a `partial def` the kernel can't unfold, so no proof exhibits a file that parses), the draw
  setup `y₀` and the records' custody `hCust` (set up from those files; custody is the named assumption
  `live-verifier`), A3, and the finders' `TableCRZAt`/`LinkCRZCAt` (the named assumption `ecr/sha-512`).
-/

namespace Flock.Bridge

open Claims.Sampling

/-- A commitment that opens nothing gives every position no value. -/
theorem wit_val_none {H : Claims.Msg → Claims.Msg} {Cm : Commitment Bool} {s : Stmt}
    (h : ∀ o, Cm.Opens H s.roots o = false) (w : Wit) (g : ℕ) : w.val H Cm s g = none := by
  unfold Wit.val
  rw [List.find?_eq_none.2 fun o _ => by simp [h o]]
  rfl

/-- **`commitDep` opens nothing at a layout with no rows.** -/
theorem commitDep_opens_none (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt)
    (hr : ∀ g, rows g = none) (c : Flock.Circuit) (nl : ℕ) (H : Claims.Msg → Claims.Msg) (rs : List Claims.Msg)
    (o : Opening Bool) : (commitDep C rows c nl rs).Opens H rs o = false := by
  have ha : atOf C rows rs o.pos = none := by
    unfold atOf rowOfPos
    cases wireOf C o.pos <;> simp [hr]
  simp only [Commitment.Opens, commitDep, pathCommit, ha]
  rfl

/-- A computed gate is in some unit. -/
theorem unit_some {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    {g : Fin C.N} (hgi : (C.op g).isInput = false) : ∃ u, P.unit g = some u := by
  cases h : P.unit g with
  | none => simp [(P.input_iff g).1 h] at hgi
  | some u => exact ⟨u, rfl⟩

/-- Slot `u` of `partOf P` is unit `u`'s replay unit. -/
theorem partOf_rus {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (u : Fin n) :
    (partOf P).rus[u.val]? = some [unitGates P u] := by
  have h : (partOf P).rus = ((List.finRange n).map fun u => [unitGates P u]) ++ (inputClassOf C).flatten := by
    simp [Partition.rus, partOf]
  rw [h, List.getElem?_append_left (by simp)]
  simp

/-- **A unit holding a returned gate is faulty when nothing opens**: the returned gate's value is committed, and no
opening gives it one. -/
theorem faulty_mem {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) (kd : ℕ) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (hCm : ∀ o, Cm.Opens H roots o = false) (w : Wit)
    {g : Fin C.N} {u : Fin n} (hg : g ∈ C.outputs) (hu : P.unit g = some u) :
    u.val ∈ faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w := by
  have hr0 := partOf_rus P u
  have hlen : u.val < (partOf P).rus.length := (List.getElem?_eq_some_iff.1 hr0).1
  have hpos : 0 < gateSize (C.op g) := by cases C.op g <;> simp [gateSize]
  have hp : posOf C g ∈ unitGates P u := by
    unfold unitGates
    refine List.mem_flatMap.2 ⟨g, List.mem_finRange g, ?_⟩
    rw [if_pos hu]
    refine List.mem_map.2 ⟨gateSize (C.op g) - 1, List.mem_range.2 (by omega), ?_⟩
    unfold posOf
    omega
  have hout : posOf C g ∈ (progOf C κ).outs :=
    List.mem_map.2 ⟨g, List.mem_filter.2 ⟨List.mem_finRange g, decide_eq_true hg⟩, rfl⟩
  have hb : ∀ v, (progOf C κ).Boundary v (posOf C g) = true := by
    intro v
    unfold Prog.Boundary
    simp only [Bool.or_eq_true]
    exact Or.inl (Or.inl (List.elem_eq_true_of_mem hout))
  unfold faulty
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 hlen, ?_⟩
  rintro ⟨r, hr, ι, hc⟩
  rw [hr0] at hr
  cases hr
  obtain ⟨b, hb', -⟩ := (hc (unitGates P u) (List.mem_singleton_self _) (posOf C g) hp).2 (hb _)
  have hnone : RU.ref (stmtOf P κ kd desc q roots).prog [unitGates P u]
      (w.val H Cm (stmtOf P κ kd desc q roots)) ι (posOf C g) = none := by
    unfold RU.ref
    rw [if_neg (fun h => by have := (hb _).symm.trans h.2; cases this),
      wit_val_none (s := stmtOf P κ kd desc q roots) hCm]
  rw [hnone] at hb'
  cases hb'

/-- **A draw of every unit catches a faulty one**: under `designOf n`, a set holding a unit's slot is accepted with
probability 0. -/
theorem pa_zero {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) {B : Finset ℕ}
    {u : Fin n} (hu : u.val ∈ B) : (designOf n).Pa (partOf P) B = 0 := by
  have hn : 0 < n := u.pos
  have hc : 1 ≤ (window 0 n B).card :=
    Finset.card_pos.2 ⟨u.val, Finset.mem_image.2 ⟨u.val, Finset.mem_filter.2 ⟨hu, Nat.zero_le _, by
      simpa using u.isLt⟩, by simp⟩⟩
  simp only [Design.Pa, designOf, partOf, classesPa, Rule.Pa, OneStage.Pa, strataSizes, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, List.flatten_cons, List.flatten_nil, List.append_nil, List.length_map,
    List.length_finRange, add_zero]
  rw [Nat.choose_eq_zero_of_lt (n := n - (window 0 n B).card) (k := n) (by omega)]
  simp

/-- **The transfer: at a commitment that opens nothing, a returned computed gate fails the claim for every witness**,
when every unit is drawn (`k_d = n`), at any `δ ≥ 0`. -/
theorem not_relS_unopened {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) (kd : ℕ) (hkd : kd = n) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (Cm : Commitment Bool) (hCm : ∀ o, Cm.Opens H roots o = false) (w : Wit) {δ : ℝ}
    (hδ : 0 ≤ δ) {g : Fin C.N} (hg : g ∈ C.outputs) (hgi : (C.op g).isInput = false) :
    ¬ RelS H Cm δ (stmtOf P κ kd desc q roots) (partOf P) w := by
  subst hkd
  obtain ⟨u, hu⟩ := unit_some P hgi
  intro h
  have h' : δ < (designOf kd).Pa (partOf P) (faulty H Cm (stmtOf P κ kd desc q roots) (partOf P) w) := h
  rw [pa_zero P (faulty_mem P κ kd desc q roots H Cm hCm w hg hu)] at h'
  linarith

/-- `deltaPub < 1` at the deployed parameters (`Flock.SecurityProofs.DeltaPubDeployed`). -/
theorem deltaPub_lt_one {n kd K₀ W : ℕ} {t' v : ℝ} (h0 : 0 < n) (hK : K₀ ≤ n) (hn : n ≤ K₀ * kd) (hW : W ≤ 2 ^ 40)
    (ht0 : 0 ≤ t') (ht : t' ≤ 2 ^ 80) (hv0 : 0 ≤ v) (hv : v ≤ 2 ^ 80) :
    deltaPub n kd K₀ kDep (2 * (t' + v)) (t' + v) t' (2 ^ 192) (2 ^ 192) W (1 / 2) < 1 := by
  have h := Flock.SecurityProofs.DeltaPubDeployed
  unfold Flock.Guarantees.DeltaPubDeployed at h
  exact h.2.2 n kd K₀ W t' v h0 hK hn hW ht0 ht hv0 hv

open FlockSoundness.Game in
/-- **An event that never holds fails with probability 1**, under every strategy of every game: so it exceeds any
`x < 1`. -/
theorem exceeds_of_not {α : Type} {G : Game α} (s : Strategy G) {R : Prop} (hR : ¬ R) {x : ℝ} (hx : x < 1) :
    ENNReal.ofReal x < prob (fun _ => True ∧ ¬ R) G s := by
  have h1 : prob (fun _ => True ∧ ¬ R) G s = 1 :=
    le_antisymm (prob_le_one _ _ _) ((prob_true _ _).symm.le.trans (prob_mono (fun _ _ => ⟨trivial, hR⟩) _ _))
  rw [h1]
  exact ENNReal.ofReal_lt_one.2 hx

/-! ## The constructed instance -/

/-- **The toy circuit**: gate 0 an input, gate 1 its negation, returned. -/
def toyCircuit : FlockSoundness.Audit.Circuit where
  N := 2
  op g := if g.val = 0 then .input else .not 0
  topo := by decide
  outputs := {⟨1, by decide⟩}

/-- **The toy partition**: one unit, holding the negation. -/
def toyPartition : FlockSoundness.Audit.Partition toyCircuit 1 where
  unit g := if g.val = 0 then none else some 0
  input_iff := by decide

theorem toy_ioSum_le : ioSum toyPartition ≤ 2 ^ 40 := by
  unfold ioSum
  calc ∑ u, (toyPartition.io u).card ≤ ∑ _u : Fin 1, toyCircuit.N :=
        Finset.sum_le_sum fun u _ => (Finset.card_le_univ _).trans_eq (Fintype.card_fin _)
    _ ≤ 2 ^ 40 := by simp [toyCircuit]

/-- **The toy fails the claim for every witness**: its commitment has no rows, so nothing opens, and its one unit
returns the negation, a committed value. Every value the extractor names (`xv`) is wrong. -/
theorem toy_not_relS (κ : Fin toyCircuit.N → Option Bool) (desc : Claims.Msg) (q : Query) (roots : List Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (c : Flock.Circuit) (nl : ℕ) (xv : Fin toyCircuit.N → Bool) {δ : ℝ} (hδ : 0 ≤ δ) :
    ¬ RelS H (commitDep toyCircuit (fun _ => none) c nl roots) δ (stmtOf toyPartition κ 1 desc q roots)
      (partOf toyPartition) (witDep toyCircuit (commitDep toyCircuit (fun _ => none) c nl roots) H roots xv) :=
  not_relS_unopened toyPartition κ 1 rfl desc q roots H _
    (commitDep_opens_none _ _ (fun _ => rfl) c nl H roots) _ hδ (g := ⟨1, by decide⟩) (by decide) rfl

open FlockSoundness.Game in
/-- **The constructed failing instance**: at the toy, a draw of size 1 (`n = k_d = K₀ = 1`, so `δ = 0`) and the deployed
parameters, for every game and every strategy, the verifier that accepts everything accepts a failing claim with
probability 1, above `deltaPub`, at any `δ ≥ 0` (the guarantee's is `0` there), any per-run time `t'` and opening cost `v` up to `2^80`. -/
theorem toy_exceeds {α : Type} {G : Game α} (s : Strategy G) (κ : Fin toyCircuit.N → Option Bool) (desc : Claims.Msg)
    (q : Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (c : Flock.Circuit) (nl : ℕ)
    (xv : Fin toyCircuit.N → Bool) {δ : ℝ} (hδ : 0 ≤ δ) {t' v : ℝ} (ht0 : 0 ≤ t') (ht : t' ≤ 2 ^ 80) (hv0 : 0 ≤ v)
    (hv : v ≤ 2 ^ 80) :
    ENNReal.ofReal (deltaPub 1 1 1 kDep (2 * (t' + v)) (t' + v) t' (2 ^ 192) (2 ^ 192) (ioSum toyPartition) (1 / 2)) <
      prob (fun _ => True ∧ ¬ RelS H (commitDep toyCircuit (fun _ => none) c nl roots) δ
        (stmtOf toyPartition κ 1 desc q roots) (partOf toyPartition)
        (witDep toyCircuit (commitDep toyCircuit (fun _ => none) c nl roots) H roots xv)) G s :=
  exceeds_of_not s (toy_not_relS κ desc q roots H c nl xv hδ)
    (deltaPub_lt_one one_pos le_rfl (by norm_num) toy_ioSum_le ht0 ht hv0 hv)

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **`SampledKSLivePub`'s parameter hypotheses hold as deployed**: `1 ≤ k`, `1 ≤ R_w`, `1 ≤ M`, `ρ < 1`,
`rateM1 k ≤ ρ` and `0 ≤ t'`, at `k = ⌈2^24/e⌉`, `R_w = M = 2^192`, `ρ = 1/2` and `t' = 2^80`; there `deltaPub < 1`
(`Flock.SecurityProofs.DeltaPubDeployed`). -/
theorem sampledKSLivePub_honest :
    1 ≤ Bridge.kDep ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ (1 / 2 : ℝ) < 1 ∧
      rateM1 Bridge.kDep ≤ 1 / 2 ∧ (0 : ℝ) ≤ 2 ^ 80 :=
  ⟨Bridge.one_le_kDep, Nat.one_le_two_pow, Nat.one_le_two_pow, by norm_num, Bridge.rateM1_kDep, by positivity⟩

open Classical in
/-- **`SampledKSLivePub` at the verifier that accepts everything**: its statement with `LiveAcceptsZK I o` replaced by
`True`. -/
def SampledKSLivePubAll : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (v : ℝ)
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      k Rw M t' →
    prob (fun _ => True ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k (2 * (t' + v)) (t' + v) t' M Rw (Bridge.ioSum P) ρ)

open Classical in
/-- **`SampledKSLivePub`'s nontrivial lemma** (LEAN-66): `SampledKSLivePub` at the verifier that accepts everything is
false at any setting of its file-level fields whose circuit has no port (so its commitment opens nothing) and returns a
computed gate, with every unit drawn (`k_d = n`, `K₀ ≥ 1`), at most `2^40` committed wires, and the deployed parameters
(`k = ⌈2^24/e⌉`, `R_w = M = 2^192`, `ρ = 1/2`, `t', v ≤ 2^80`). There the extractor's values fail the claim at every run
(`Bridge.not_relS_unopened`), so `Pr[accept ∧ ¬ RelS] = 1`, while `deltaPub < 1`. The toy (`Bridge.toy_exceeds`) is
such a circuit, built by hand; the binders left are the files' parses, the draw setup, the records' custody, A3 and
the finders' `ecr/sha-512`. -/
theorem sampledKSLivePub_nontrivial :
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (t' v : ℝ)
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    kd = n → 1 ≤ K₀ → D.unroll.ports.size = 0 → (∃ g ∈ Cc.outputs, (Cc.op g).isInput = false) →
    Bridge.ioSum P ≤ 2 ^ 40 → 0 ≤ t' → t' ≤ 2 ^ 80 → 0 ≤ v → v ≤ 2 ^ 80 →
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      Bridge.kDep (2 ^ 192) (2 ^ 192) t' →
    ¬ SampledKSLivePubAll := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb
    src hA3 hTags hCust K₀ hK₀ τ t' v q L P vb σ Cc hkdn hK1 hports hout hW ht0 ht hv0 hv hCR hLink hAll
  obtain ⟨g, hg, hgi⟩ := hout
  obtain ⟨hk, hRw, hM, hρ, hρk, -⟩ := sampledKSLivePub_honest
  have h := hAll n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N
    Ω Lb src hA3 hTags hCust Bridge.kDep (2 ^ 192) (2 ^ 192) hk hRw hM (1 / 2) hρ hρk t' ht0 K₀ hK₀ τ v q hCR hLink
  have hr : ∀ g : Fin Cc.N,
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map
        (fun r => (⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩ : Bridge.RowAt)) =
        none := by
    intro g
    cases hrow : FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g with
    | none => rfl
    | some r => exact absurd r.2.1.isLt (by omega)
  obtain ⟨u, -⟩ := Bridge.unit_some P hgi
  have hn : n ≤ K₀ * kd := by
    rw [hkdn]
    exact Nat.le_mul_of_pos_left n hK1
  have hlt := Bridge.deltaPub_lt_one (W := Bridge.ioSum P) u.pos hK₀ hn hW ht0 ht hv0 hv
  exact not_le.2 (Bridge.exceeds_of_not τ (Bridge.not_relS_unopened P _ kd hkdn desc q _ _ _
    (Bridge.commitDep_opens_none Cc _ hr D.unroll n _ _) _ (by positivity) hg hgi) hlt) h

end Flock.Guarantees
