import Proofs.Flock.Bridge.PublicDeployed
import Proofs.Flock.Soundness.Game.Prob

/-!
# `SampledKSLivePub`'s honest and nontrivial lemmas

* **Honest** (`sampledKSLivePub_honest`): the hypotheses of `SampledKSLivePub` on the analysis' parameters hold at the
  deployed ones (`Bridge.kDep`, `ρ = 1/2`, `M = R_w = 2^192`), and there `δ + f(p) < 1` for `p ≤ 2^-200`
  (`Flock.SecurityProofs.DeltaPubDeployed`).
* **The failing instance, constructed** (`Bridge.toy_exceeds`): a two-gate circuit (an input and its negation, the
  negation returned) in one unit, drawn by a draw of size 1 (`n = k_d = K₀ = 1`), under a commitment with no rows, so
  that nothing opens and every value the extractor names fails the claim. There, for every game and every strategy,
  `Pr[accept ∧ ¬ RelS] = 1 > δ + f(p)` when the verifier accepts everything. It is built by hand at the model level
  (`FlockSoundness.Audit.Circuit`, `Bridge.commitDep`), with no premise.
* **The transfer** (`Bridge.not_relS_unopened`, `Bridge.exceeds_of_not`): at any circuit whose commitment opens nothing,
  a computed output fails the claim for every witness when every unit is drawn; and an event that never holds has
  probability 1 under every strategy. V2's `_nontrivial` reads the instance through these.
* **V2 and V3's parameters** (`sampledKSLiveV2_honest`): their parameter hypotheses hold as deployed. Their nontrivial
  lemmas are `RowsRegWitnesses`'.
* **`SampledKSLiveV2Z` unopened** (`sampledKSLiveV2Z_unopened`): at the verifier that accepts everything its additive
  bound is below the event's probability, 1, at every prover whose finder `B^τ` has `J²·K·Pr[B^τ] ≤ 2^-200`
  (`Bridge.boundPr_le_pub`), at a circuit with no port that returns a computed gate.
* **Nontrivial**: `SampledKSLivePubAll`, `SampledKSLivePub` at the verifier that accepts everything, states what
  `PublicNontrivial`'s `sampledKSLivePub_nontrivial` refutes at the toy session's files.
* **Vacuous as stated** (`Bridge.outputs_noPort`, `HiddenReturns`): with the parse `hD` of an untyped statement in
  scope, a circuit with no port returns no gate, so the structural premises of `sampledKSLiveV2Z_unopened` are never
  met together, and it holds for want of an instance.
-/

namespace Flock.Bridge

open Claims.Sampling

/-- A commitment that opens nothing gives every position no value. -/
theorem wit_val_none {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.CommitmentScheme Bool} {s : Claims.Sampling.AuditStatement}
    (h : ∀ o, Cm.Opens H s.roots o = false) (w : Claims.Sampling.Wit) (g : ℕ) : w.val H Cm s g = none := by
  unfold Claims.Sampling.Wit.val
  rw [List.find?_eq_none.2 fun o _ => by simp [h o]]
  all_goals rfl

/-- **`commitDep` opens nothing at a layout with no rows.** -/
theorem commitDep_opens_none (C : FlockSoundness.Audit.Circuit) (rows : Fin C.N → Option RowAt)
    (hr : ∀ g, rows g = none) (c : Flock.Circuit) (nl : ℕ) (H : Claims.Msg → Claims.Msg) (rs : List Claims.Msg)
    (o : Claims.Sampling.Opening Bool) : (commitDep C rows c nl rs).Opens H rs o = false := by
  have ha : atOf C rows rs o.pos = none := by
    unfold atOf rowOfPos
    cases wireOf C o.pos <;> simp [hr]
  simp only [Claims.Sampling.CommitmentScheme.Opens, commitDep, pathCommit, ha] <;> rfl

/-- A computed gate is in some unit. -/
theorem unit_some {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    {g : Fin C.N} (hgi : (C.op g).isInput = false) : ∃ u, P.unit g = some u := by
  cases h : P.unit g with
  | none => simp [(P.input_iff g).1 h] at hgi
  | some u => exact ⟨u, rfl⟩

/-- **A unit holding a returned gate is faulty when nothing opens**: the returned gate's value is committed, and no
opening gives it one. -/
theorem faulty_mem {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) (kd : ℕ) (desc : Claims.Msg) (q : Claims.Sampling.Query) (roots : List Claims.Msg)
    (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (hCm : ∀ o, Cm.Opens H roots o = false)
    (w : Claims.Sampling.Wit) {g : Fin C.N} {u : Fin n} (hg : g ∈ C.outputs) (hu : P.unit g = some u) :
    u.val ∈ Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w := by
  classical
  have hr0 := rus_partOf_getElem? (κ := κ) P u
  have hlen : u.val < (partOf P κ).rus.length := (List.getElem?_eq_some_iff.1 hr0).1
  have hpos : 0 < gateSize (C.op g) := by cases C.op g <;> simp [gateSize]
  have hp : posOf C g ∈ unitGates P u := by
    unfold unitGates
    refine List.mem_flatMap.2 ⟨g, List.mem_finRange g, ?_⟩
    show posOf C g ∈ (if P.unit g = some u then (List.range (gateSize (C.op g))).map (off C g.val + ·) else [])
    rw [if_pos hu]
    refine List.mem_map.2 ⟨gateSize (C.op g) - 1, List.mem_range.2 (by omega), ?_⟩
    unfold posOf
    omega
  have hout : posOf C g ∈ (progOf C κ).outs :=
    List.mem_map.2 ⟨g, List.mem_filter.2 ⟨List.mem_finRange g, decide_eq_true hg⟩, rfl⟩
  have hb : ∀ v, (progOf C κ).Boundary v (posOf C g) = true := by
    intro v
    unfold Claims.Sampling.Prog.Boundary
    simp only [Bool.or_eq_true]
    exact Or.inl (Or.inl (List.elem_eq_true_of_mem hout))
  unfold Claims.Sampling.faulty
  refine Finset.mem_filter.2 ⟨Finset.mem_range.2 hlen, ?_⟩
  rintro ⟨r, hr, ι, hc⟩
  rw [hr0] at hr
  cases hr
  have hc' := hc (unitGates P u) (List.mem_append_left _ (List.mem_singleton_self _))
  change (progOf C κ).Correct (Claims.Sampling.RU.ref (progOf C κ) (slotOf P κ u)
    (w.val H Cm (stmtOf P κ kd desc q roots)) ι) (unitGates P u) at hc'
  rw [ref_slotOf] at hc'
  obtain ⟨b, hb', -⟩ := (hc' (posOf C g) hp).2 (hb _)
  have hcond : ¬ (posOf C g ∈ Claims.Sampling.RU.gates [unitGates P u] ∧
      (progOf C κ).Boundary (Claims.Sampling.RU.gates [unitGates P u]) (posOf C g) = false) :=
    fun h => by have := (hb _).symm.trans h.2; cases this
  have hnone : Claims.Sampling.RU.ref (progOf C κ) [unitGates P u]
      (w.val H Cm (stmtOf P κ kd desc q roots)) ι (posOf C g) = none := by
    unfold Claims.Sampling.RU.ref
    rw [if_neg hcond, wit_val_none (s := stmtOf P κ kd desc q roots) hCm]
  rw [hnone] at hb'
  cases hb'

/-- **A draw of every unit catches a faulty one**: under `designOf n`, a set holding a unit's slot is accepted with
probability 0. -/
theorem pa_zero {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    {κ : Fin C.N → Option Bool} {B : Finset ℕ} {u : Fin n} (hu : u.val ∈ B) : (designOf n).Pa (partOf P κ) B = 0 := by
  have hn : 0 < n := u.pos
  have hc : 1 ≤ (Claims.Sampling.window 0 n B).card :=
    Finset.card_pos.2 ⟨u.val, Finset.mem_image.2 ⟨u.val, Finset.mem_filter.2 ⟨hu, Nat.zero_le _, by
      simpa using u.isLt⟩, by simp⟩⟩
  rw [pa_partOf_window, Nat.choose_eq_zero_of_lt (n := n - (Claims.Sampling.window 0 n B).card) (k := n) (by omega)]
  simp

/-- **The transfer: at a commitment that opens nothing, a returned computed gate fails the claim for every witness**,
when every unit is drawn (`k_d = n`), at any `δ ≥ 0`. -/
theorem not_relS_unopened {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    (κ : Fin C.N → Option Bool) (kd : ℕ) (hkd : kd = n) (desc : Claims.Msg) (q : Claims.Sampling.Query)
    (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (hCm : ∀ o, Cm.Opens H roots o = false) (w : Claims.Sampling.Wit) {δ : ℝ} (hδ : 0 ≤ δ) {g : Fin C.N}
    (hg : g ∈ C.outputs) (hgi : (C.op g).isInput = false) :
    ¬ Claims.Sampling.RelS H Cm δ (stmtOf P κ kd desc q roots) (partOf P κ) w := by
  obtain ⟨u, hu⟩ := unit_some P hgi
  intro h
  have h' : δ < (designOf kd).Pa (partOf P κ)
      (Claims.Sampling.faulty H Cm (stmtOf P κ kd desc q roots) (partOf P κ) w) := h
  rw [hkd] at h'
  rw [pa_zero P (κ := κ) (faulty_mem P κ n desc q roots H Cm hCm w hg hu)] at h'
  linarith

/-- `δ + f(p) < 1` at the deployed parameters (`Flock.SecurityProofs.DeltaPubDeployed`). -/
theorem deltaPub_lt_one {n kd K₀ W : ℕ} {p : ℝ} (h0 : 0 < n) (hK : K₀ ≤ n) (hn : n ≤ K₀ * kd) (hW : W ≤ 2 ^ 40)
    (hp0 : 0 ≤ p) (hp : p ≤ 1 / 2 ^ 200) :
    deltaPub n kd K₀ kDep (2 ^ 192) (2 ^ 192) W (1 / 2) + lossPub kDep W (1 / 2) p < 1 := by
  have h := Flock.SecurityProofs.DeltaPubDeployed
  unfold Flock.Guarantees.DeltaPubDeployed at h
  exact h.2.2 n kd K₀ W p h0 hK hn hW hp0 hp

open FlockSoundness FlockSoundness.Game in
/-- **An event that never holds fails with probability 1**, under every strategy of every game: so it exceeds any
`x < 1`. -/
theorem exceeds_of_not {α : Type} {G : Game α} (s : Strategy G) {R : Prop} (hR : ¬ R) {x : ℝ} (hx : x < 1) :
    ENNReal.ofReal x < prob (fun _ => True ∧ ¬ R) G s := by
  have h1 : prob (fun _ => True ∧ ¬ R) G s = 1 :=
    le_antisymm (prob_le_one _ _ _) ((prob_true _ _).symm.le.trans (prob_mono (fun _ _ => ⟨trivial, hR⟩) _ _))
  rw [h1]
  exact ENNReal.ofReal_lt_one.2 hx

open FlockSoundness FlockSoundness.Game in
/-- **An event whose condition fails at every outcome holds with probability 1**, under every strategy of every game:
so it exceeds any `x < 1`. -/
theorem exceeds_of_forall_not {α : Type} {G : Game α} (s : Strategy G) {R : α → Prop} (hR : ∀ a, ¬ R a) {x : ℝ}
    (hx : x < 1) : ENNReal.ofReal x < prob (fun a => True ∧ ¬ R a) G s := by
  have h1 : prob (fun a => True ∧ ¬ R a) G s = 1 :=
    le_antisymm (prob_le_one _ _ _) ((prob_true _ _).symm.le.trans (prob_mono (fun a _ => ⟨trivial, hR a⟩) _ _))
  rw [h1]
  exact ENNReal.ofReal_lt_one.2 hx

/-- **`δ + f(p) < 1` at the registrant's rows**, at the deployed parameters: `deltaPubR` and `lossPubR` exceed
`deltaPub` and `lossPub` by at most `2·2^40·2^23/2^192 + 2·2^40·p + p`, far inside the slack
`Flock.SecurityProofs.DeltaPubDeployed`'s parts leave. -/
theorem deltaPubR_lt_one {n kd K₀ W : ℕ} {p : ℝ} (h0 : 0 < n) (hK : K₀ ≤ n) (hn : n ≤ K₀ * kd) (hW : W ≤ 2 ^ 40)
    (hp0 : 0 ≤ p) (hp : p ≤ 1 / 2 ^ 200) :
    deltaPubR n kd K₀ kDep (2 ^ 192) (2 ^ 192) W (1 / 2) + lossPubR kDep W (1 / 2) p < 1 := by
  have hd := deltaPub_le_deployed n kd K₀ W hW
  have hl := lossPub_le_deployed kDep_le hW hp0 hp
  have h := miss_le_half h0 hK hn
  have hW' : (W : ℝ) ≤ 2 ^ 40 := by exact_mod_cast hW
  have he : (1 : ℝ) ≤ Real.exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have hk : (kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) ≤ 2 ^ 23 / 2 ^ 192 := by
    rw [show ((2 ^ 192 : ℕ) : ℝ) = (2 : ℝ) ^ 192 by norm_num]
    exact div_le_div₀ (by positivity) kDep_le (by positivity) (le_mul_of_one_le_left (by positivity) he)
  have hWk : (W : ℝ) * ((kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ))) ≤ 2 ^ 40 * (2 ^ 23 / 2 ^ 192) :=
    mul_le_mul hW' hk (by positivity) (by positivity)
  have hWp : (W : ℝ) * p ≤ 2 ^ 40 * (1 / 2 ^ 200) := mul_le_mul hW' hp hp0 (by positivity)
  have h100 : (2 : ℝ) ^ (-(100 : ℝ)) < 1 / 4 := by
    rw [Real.rpow_neg (by norm_num), show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have h60 : (1 : ℝ) / 2 ^ 60 < 1 / 4 := by norm_num
  have hs : (2 : ℝ) * (2 ^ 40 * (2 ^ 23 / 2 ^ 192)) + 2 * (2 ^ 40 * (1 / 2 ^ 200)) + 1 / 2 ^ 200 < 1 / 4 := by
    norm_num
  have e2 : ∀ x : ℝ, x / (1 - 1 / 2) = 2 * x := fun x => by ring
  unfold deltaPubR lossPubR
  rw [e2, e2]
  linarith

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
theorem toy_not_relS (κ : Fin toyCircuit.N → Option Bool) (desc : Claims.Msg) (q : Claims.Sampling.Query)
    (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (c : Flock.Circuit) (nl : ℕ) (xv : Fin toyCircuit.N → Bool)
    {δ : ℝ} (hδ : 0 ≤ δ) :
    ¬ Claims.Sampling.RelS H (commitDep toyCircuit (fun _ => none) c nl roots) δ
      (stmtOf toyPartition κ 1 desc q roots) (partOf toyPartition κ) (witDep toyCircuit (commitDep toyCircuit (fun _ => none) c nl roots) H roots xv) :=
  not_relS_unopened toyPartition κ 1 rfl desc q roots H (commitDep toyCircuit (fun _ => none) c nl roots)
    (commitDep_opens_none toyCircuit (fun _ => none) (fun _ => rfl) c nl H roots)
    (witDep toyCircuit (commitDep toyCircuit (fun _ => none) c nl roots) H roots xv) hδ (g := ⟨1, by decide⟩)
    (by decide) rfl

open FlockSoundness FlockSoundness.Game in
/-- **The constructed failing instance**: at the toy, a draw of size 1 (`n = k_d = K₀ = 1`) and the deployed
parameters, for every game and every strategy, the verifier that accepts everything accepts a failing claim with
probability 1, above `δ + f(p)`: at any `δ ≥ 0` (the guarantee's is `0` there) and `p ≤ 2^-200`. -/
theorem toy_exceeds {α : Type} {G : Game α} (s : Strategy G) (κ : Fin toyCircuit.N → Option Bool) (desc : Claims.Msg)
    (q : Claims.Sampling.Query) (roots : List Claims.Msg) (H : Claims.Msg → Claims.Msg) (c : Flock.Circuit) (nl : ℕ)
    (xv : Fin toyCircuit.N → Bool) {δ : ℝ} (hδ : 0 ≤ δ) {p : ℝ} (hp0 : 0 ≤ p) (hp : p ≤ 1 / 2 ^ 200) :
    ENNReal.ofReal (deltaPub 1 1 1 kDep (2 ^ 192) (2 ^ 192) (ioSum toyPartition) (1 / 2) +
      lossPub kDep (ioSum toyPartition) (1 / 2) p) <
      prob (fun _ => True ∧ ¬ Claims.Sampling.RelS H (commitDep toyCircuit (fun _ => none) c nl roots) δ
        (stmtOf toyPartition κ 1 desc q roots) (partOf toyPartition κ)
        (witDep toyCircuit (commitDep toyCircuit (fun _ => none) c nl roots) H roots xv)) G s :=
  exceeds_of_not s (toy_not_relS κ desc q roots H c nl xv hδ)
    (deltaPub_lt_one (n := 1) (kd := 1) (K₀ := 1) one_pos le_rfl (by norm_num) toy_ioSum_le hp0 hp)

end Flock.Bridge

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **`SampledKSLivePub`'s parameter hypotheses hold as deployed**: `1 ≤ k`, `1 ≤ R_w`, `1 ≤ M`, `ρ < 1` and
`rateM1 k ≤ ρ`, at `k = ⌈2^24/e⌉`, `R_w = M = 2^192` and `ρ = 1/2`; there `δ + f(p) < 1` for `p ≤ 2^-200`
(`Flock.SecurityProofs.DeltaPubDeployed`). -/
theorem sampledKSLivePub_honest :
    1 ≤ Bridge.kDep ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ (1 / 2 : ℝ) < 1 ∧ rateM1 Bridge.kDep ≤ 1 / 2 :=
  ⟨Bridge.one_le_kDep, Nat.one_le_two_pow, Nat.one_le_two_pow, by norm_num, Bridge.rateM1_kDep⟩

open Classical in
/-- **`SampledKSLivePub` at the verifier that accepts everything**: its statement with
`Discharge.ZkReg.LiveAcceptsZKR I own o` replaced by `True`. -/
def SampledKSLivePubAll : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (own : Array Flock.Registered.Port)
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
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (v qR : ℝ)
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
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
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if (¬ rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    let B := Bridge.pubFinderR dj y₀ P vb wo k Rw M (2 * (t' + v)) (t' + v) t' (reg σ) (cont σ)
      (Bridge.strictComp (FlockSoundness.StrictCR.oneRun G τ fun o =>
        rd.pick o.1 (Discharge.ZkReg.regPathsHJ I D.unroll pub o.2.2 rd.port)) qR)
    prob (fun o => True ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ENNReal.ofReal (Bridge.deltaPubR n kd K₀ k M Rw (Bridge.ioSum P) ρ +
        Bridge.lossPubR k (Bridge.ioSum P) ρ
          ((nTab I : ℝ) ^ 2 * Fintype.card (Bridge.PubIxR dj y₀ P vb L) * B.Pr Claims.Sampling.sha512Msg))

/-- **`SampledKSLiveV2`'s parameter hypotheses hold as deployed**: `1 ≤ k`, `1 ≤ R_w`, `1 ≤ M`, `ρ < 1`, and every
table of the hidden-output plan at rate at most `ρ` (`planZHJ_M1W`, `rateZ_le_m1w`), at `k = ⌈2^24/e⌉`,
`R_w = M = 2^192` and `ρ = 1/2`. `EndToEndHidden`, `SampledKSLive` and `SampledKSLiveV3` take the same. -/
theorem sampledKSLiveV2_honest {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) :
    1 ≤ Bridge.kDep ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ (1 / 2 : ℝ) < 1 ∧
      ∀ S (R : Reg) j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j Bridge.kDep ≤ 1 / 2 :=
  ⟨Bridge.one_le_kDep, Nat.one_le_two_pow, Nat.one_le_two_pow, by norm_num,
    fun S R j => (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) _).trans Bridge.rateM1_kDep⟩

open Classical in
/-- **`SampledKSLiveV2Z` at an unopened circuit**: at the verifier that accepts everything, its conclusion fails.
At any setting of its file-level fields whose circuit has no port and returns a computed gate, with a registrant who
opens no registered read (so the rows term is `0`, `rowsPrZC_noReads`, and the claims' values are the evaluation on
the extracted plurality's inputs), with every unit drawn
(`k_d = n`, `K₀ ≥ 1`), at most `2^40` committed wires and the deployed parameters (`sampledKSLiveV2_honest`), V2's
additive bound is below `Pr[True ∧ ¬ RelS] = 1` (`Bridge.not_relS_unopened`, `Bridge.exceeds_of_not`) at every prover
whose finder `B^τ` (at a per-run time `t'` and a verification `v`) has `J²·K·Pr[B^τ] ≤ 2^-200`: there the bound is at
most `δ + f(J²·K·Pr[B^τ]) < 1` (`Bridge.boundPr_le_pub`). That premise is this lemma's, not the statement's: it holds at every
prover, and the premise only places the prover where its bound is below 1. The toy
(`Bridge.toy_exceeds`) is such a circuit, built by hand. -/
theorem sampledKSLiveV2Z_unopened :
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (desc : Claims.Msg) (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
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
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) Bridge.kDep (2 ^ 192)
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    let vreg : Fin Cc.N → Bool := fun g =>
      if rd.isRead (vb.pos g) then vb.bit (vb.val (wo (vb.pos g))) g else Cc.eval Xr g
    kd = n → 1 ≤ K₀ → D.unroll.ports.size = 0 → (∃ g ∈ Cc.outputs, (Cc.op g).isInput = false) →
    (∀ p, ¬ rd.isRead p) →
    Bridge.ioSum P ≤ 2 ^ 40 →
    (nTab I : ℝ) ^ 2 * Fintype.card (Bridge.PubIx dj y₀ P vb L) *
        (Bridge.pubFinder dj y₀ P vb Bridge.kDep (2 ^ 192) (2 ^ 192) (2 * (t' + v)) (t' + v) t' (reg σ)
          (cont σ)).Pr Claims.Sampling.sha512Msg ≤ 1 / 2 ^ 200 →
    ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) Bridge.kDep (reg σ) (cont σ) +
        linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb Bridge.kDep (2 ^ 192) (2 ^ 192) (1 / 2)
          (reg σ) (cont σ) +
        rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo Bridge.kDep (1 / 2) (reg σ)
          (cont σ) <
      prob (fun _ => True ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
            (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots vreg))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ := by
  intro n Reg I ht D hD hU pub hpub hpt hscope desc kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω _ _ Lb src hTags hCust K₀ hK₀
    τ t' v q L P vb wo σ Xr δ Cc rows roots Cd κ vreg hkdn hK1 hports hout hnr hW hB
  have hnr' : rd.isRead = fun _ => False := funext fun p => propext ⟨hnr p, False.elim⟩
  have hrows0 : rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb rd.isRead wo Bridge.kDep (1 / 2) (reg σ)
      (cont σ) = 0 := by
    rw [hnr']
    exact rowsPrZC_noReads execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb wo Bridge.kDep (1 / 2) (reg σ) (cont σ)
  have hv : vreg = Cc.eval Xr := funext fun g => if_neg (hnr _)
  rw [hrows0, add_zero, hv, ← Bridge.witDep_eq_witRows]
  obtain ⟨g, hg, hgi⟩ := hout
  have hr : ∀ g : Fin Cc.N, rows g = none := by
    intro g
    show (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map
      (fun r => (⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩ : Bridge.RowAt)) = none
    cases hrow : FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g with
    | none => rfl
    | some r =>
      have := r.2.1.isLt
      omega
  obtain ⟨u, -⟩ := Bridge.unit_some P hgi
  have hn : n ≤ K₀ * kd := by
    rw [hkdn]
    exact Nat.le_mul_of_pos_left n hK1
  have hlt := Bridge.deltaPub_lt_one (W := Bridge.ioSum P) u.pos hK₀ hn hW
    (mul_nonneg (mul_nonneg (sq_nonneg _) (Nat.cast_nonneg _)) (Claims.Pr_nonneg _)) hB
  have hδ : (0 : ℝ) ≤ δ := by
    show (0 : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    positivity
  refine lt_of_le_of_lt (Bridge.boundPr_le_pub dj y₀ P vb hkd (by norm_num) Bridge.rateM1_kDep (2 * (t' + v))
    (t' + v) t' (reg σ) (cont σ)) ?_
  exact Bridge.exceeds_of_not τ (Bridge.not_relS_unopened P κ kd hkdn desc q roots Claims.Sampling.sha512Msg Cd
    (Bridge.commitDep_opens_none Cc rows hr D.unroll n _ _) _ hδ hg hgi) hlt

end Flock.Guarantees
