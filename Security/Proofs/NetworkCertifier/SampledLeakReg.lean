import Definitions.NetworkCertifier.SampledReg
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.SampledAuditAccepted
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Finder

/-!
# The leak bound at the registration's values

`RAcc` pins the circuit's inputs to the trace's committed values, and holds every faulty unit's committed gate as
advice. At a registered input that is a free input:

* in a faulty unit, the gate is held (`FaultyGate`: an input is on its unit's boundary), so its input value is never
  read;
* in any other unit, the unit is correct on its committed values, so the trace opens the gate (`val_some_of_input`) and
  its value there is the opened one (`traceVal_of_val`).

So when the trace agrees with the registration at the registered inputs it opens, the accepted advice can be moved to
the inputs pinned at the registration's values (`rLeak_reg_of_rAcc`), a set the statement and the registration fix. An
extracted trace that disagrees opens a registered position to a value the registration doesn't. The extractor checked
that opening with its own calls (`Passes.found`, `ownLog_E`), so under `CommitmentScheme.Binds` its calls and those that check
the registration's opening collide, and `B_reg` finds the collision (`regColl_of_disagree`, `Pr_regColl_le_Breg`).
`sampledLeak_registered` is the bound: `Pr[acc ∧ ¬RLeak] ≤ δ + ε_ks + ε_bind + Pr[B_reg]`, with the leak relation at the
registration's values, and the count `2^#(freeInputs \ reg) · #faultySets · 2^β · Dj^k` there.
-/

namespace NetTiming

open Claims Claims.Sampling NetTiming.SampledAuditProof

/-! ### A registered input of a correct unit is opened at the trace's value -/

theorem val_some_of_input {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition} {w : Wit}
    (hP : P.Partitions s.prog) {g : ℕ} (hg : s.prog.gates[g]? = some .input)
    (hnf : ¬ FaultyGate s.prog P (faulty H Cm s P w) g) : ∃ b, w.val H Cm s g = some b := by
  obtain ⟨ι, hι⟩ := correct_of_not_faulty H Cm s P w
  have hlt : g < s.prog.gates.length := (List.getElem?_eq_some_iff.1 hg).1
  have hmem : g ∈ P.gates := hP.2.2.1 g hlt fun b h => by rw [hg] at h; cases h
  obtain ⟨v, r, u, hr, hu, hgu⟩ := exists_of_mem_gates hmem
  have hv : v ∉ faulty H Cm s P w := fun hv => hnf ⟨v, hv, r, hr, mem_ru_gates hu hgu, boundary_of_input hg⟩
  obtain ⟨b, hb, -⟩ := (hι v r hr hv u hu g hgu).2 (boundary_of_input hg)
  have hc : ¬ (g ∈ RU.gates r ∧ s.prog.Boundary (RU.gates r) g = false) := fun h => by
    rw [boundary_of_input hg] at h
    exact Bool.noConfusion h.2
  unfold RU.ref at hb
  rw [ite_eq_right hc] at hb
  exact ⟨b, hb⟩

theorem traceVal_of_val {C : Prog} {ref : ℕ → Option Bool} {g : ℕ} {b : Bool} (hg : C.gates[g]? = some .input)
    (h : ref g = some b) : traceVal C ref g = b := by
  simp [traceVal, Prog.read, hg, h]

/-! ### Moving the advice to the registration's values -/

theorem inputOf_congr_at {L S : ℕ} (lay : BoundaryGates L S) (k : ℕ) (ins : List (List Bool)) {a a' : ℕ → Bool} {g : ℕ}
    (h : a g = a' g) : lay.inputOf k ins a g = lay.inputOf k ins a' g := by
  unfold BoundaryGates.inputOf
  split
  · rfl
  · rw [h]

theorem valOver_congr_held (C : Prog) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool)
    (h : ∀ i, C.gates[i]? = some .input → free i = none → inp i = inp' i) (g : ℕ) :
    valOver C inp free g = valOver C inp' free g := by
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    rcases hf : free g with _ | b
    · rcases hx : C.gates[g]? with _ | x
      · rw [valOver_of_none_none C inp hf hx, valOver_of_none_none C inp' hf hx]
      · rw [valOver_of_none_some C inp hf hx, valOver_of_none_some C inp' hf hx]
        have hrd : ∀ i ∈ x.reads, (if _h : i < g then some (valOver C inp free i) else some false) =
            if _h : i < g then some (valOver C inp' free i) else some false := by
          intro i _
          split_ifs with hig
          · rw [ih i hig]
          · rfl
        rw [bgate_val_congr x _ hrd]
        cases x <;> first | rfl | rw [h g hx hf]
    · rw [valOver_of_some C inp hf, valOver_of_some C inp' hf]

/-- **At the registration's values**: on a partitioned program, `RAcc δ` at a trace that opens each registered input,
where it opens one, to the registration's value `ρ` there gives the leak relation at the inputs pinned at `ρ`. -/
theorem rLeak_reg_of_rAcc {H : Msg → Msg} {Cm : CommitmentScheme Bool} {parse : Msg → Option AuditStatement}
    {partition : AuditStatement → Option Partition} {L K S : ℕ} {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ} {Dj : ℕ}
    (reg : AuditStatement → Finset ℕ) (ρ : AuditStatement → ℕ → Bool) {x : Msg} {w : Wit}
    (hpart : ∀ s P, parse x = some s → partition s = some P → P.Partitions s.prog)
    (hagree : ∀ s, parse x = some s → ∀ g ∈ reg s, ∀ b, w.val H Cm s g = some b → b = ρ s g)
    (h : RAcc H Cm parse partition L K S lay k δ x w) :
    RLeak H Cm parse partition L K S lay k δ (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (ρ s)) Dj x w := by
  obtain ⟨s, P, hs, hP, hRel, hdisj, free, hfree, hheld, hacc⟩ := h
  have hF := faulty_mem_faultySets hRel hdisj
  have hPart := hpart s P hs hP
  set tv := traceVal s.prog (w.val H Cm s) with htv
  set G := freeInputs s.prog (lay s) (k s) with hG
  let a' : ℕ → Bool := fun g => if g ∈ G then (if g ∈ reg s then ρ s g else tv g) else false
  have hpt : ∀ i, s.prog.gates[i]? = some .input → free i = none →
      (fun g => if g ∈ G then tv g else false) i = a' i := by
    intro i hi hfi
    show (if i ∈ G then tv i else false) = if i ∈ G then (if i ∈ reg s then ρ s i else tv i) else false
    by_cases hiG : i ∈ G
    · rw [ite_eq_left hiG, ite_eq_left hiG]
      by_cases hir : i ∈ reg s
      · rw [ite_eq_left hir]
        have hnf : ¬ FaultyGate s.prog P (faulty H Cm s P w) i := fun hfg => hheld i hfg hfi
        obtain ⟨b, hb⟩ := val_some_of_input hPart hi hnf
        rw [htv, traceVal_of_val hi hb]
        exact hagree s hs i hir b hb
      · rw [ite_eq_right hir]
    · rw [ite_eq_right hiG, ite_eq_right hiG]
  refine ⟨s, P, hs, hP, fun op o hC hCov hl hD => ⟨hl, hD, _, hF, a',
    ⟨fun g hg hgf => ?_, fun g hg => ite_eq_right hg⟩, free, hfree, ?_⟩⟩
  · show (if g ∈ G then (if g ∈ reg s then ρ s g else tv g) else false) = ρ s g
    rw [ite_eq_left hgf, ite_eq_left hg]
  · intro t ht
    obtain ⟨h, hr, hst⟩ := hacc op o hC hCov t ht
    refine ⟨h, ?_, hst⟩
    have hv : valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) tv) free =
        valOver s.prog ((lay s).inputOf (k s) (ingressUpTo op o t) a') free :=
      funext (valOver_congr_held s.prog free fun i hi hfi =>
        (inputOf_restrict s.prog (lay s) (k s) _ tv hi).trans
          (inputOf_congr_at (lay s) (k s) _ (hpt i hi hfi)))
    rw [hr]
    simp only [circuitProgram, hv]

/-! ### A disagreement with the registration is a collision -/

theorem mem_assemble_of_val {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {found : List (List Msg × Opening Bool)}
    {g : ℕ} {b : Bool} (h : (assemble s found).val H Cm s g = some b) :
    ∃ q ∈ found, q.1 = s.roots ∧ q.2.pos = g ∧ q.2.val = b := by
  unfold Wit.val at h
  obtain ⟨o, ho, rfl⟩ := Option.map_eq_some_iff.1 h
  have hpo := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpo
  obtain ⟨q, hq, hqo⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases hr : q.1 = s.roots
  · simp only [hr, ite_true, Option.some.injEq] at hqo
    subst hqo
    exact ⟨q, hq, hr, hpo.1, rfl⟩
  · simp [hr] at hqo

theorem regVal_mem {ops : List (Opening Bool)} {g : ℕ} (h : ∃ o ∈ ops, o.pos = g) :
    ∃ o ∈ ops, o.pos = g ∧ o.val = regVal ops g := by
  obtain ⟨o, ho⟩ : ∃ o, ops.find? (fun o => o.pos == g) = some o :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 (by obtain ⟨o, hm, hp⟩ := h; exact ⟨o, hm, by simp [hp]⟩))
  refine ⟨o, List.mem_of_find?_eq_some ho, by simpa using List.find?_some ho, ?_⟩
  simp [regVal, ho]

theorem regLog_valid (H : Msg → Msg) (Cm : CommitmentScheme Bool) (roots : List Msg) (ops : List (Opening Bool)) :
    ∀ p ∈ regLog H Cm roots ops, H p.1 = p.2 := by
  intro p hp
  obtain ⟨o, -, hpo⟩ := List.mem_flatMap.1 hp
  exact OC.eval_log H _ p hpo

/-- **Binding is the commitment's alone**: a record whose trace commitment is `D`'s binds where `D` does, whatever its
Flock verifier, the plain one included. So `sampledLeak_registered` holds at such a record with `D`'s binding and the
record's own `KS`. -/
theorem binds_of_commit {Coin Col T Coin' Col' T' : Type} {H : Msg → Msg} {D : Deploy Coin Col T}
    {D' : Deploy Coin' Col' T'} (h : D'.commit = D.commit) (hb : D.commit.Binds H) : D'.commit.Binds H := h ▸ hb

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- The registration's collision: the extraction's own calls and the registration's checks hold two different inputs
with one output. -/
def RegColl (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (ops : AuditStatement → List (Opening Bool))
    {Ω : Type} (P : Prover Ω) (x : Msg) (s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x) : Prop :=
  ∃ p ∈ ownLog H (run H P D.V x s.1) ((E D decode).ext x s.2.1 s.2.2) ++
      ((D.parse x).map fun st => regLog H D.commit st.roots (ops st)).getD [],
    ∃ p' ∈ ownLog H (run H P D.V x s.1) ((E D decode).ext x s.2.1 s.2.2) ++
      ((D.parse x).map fun st => regLog H D.commit st.roots (ops st)).getD [],
    p.1 ≠ p'.1 ∧ p.2 = p'.2

omit [DecidableEq Col] [Zero Col] in
/-- A kept opening of an accepted run passes, and its check's calls are the extraction's own. -/
theorem passes_of_mem_found (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : ECoins D x) {st : AuditStatement}
    (hst : D.parse x = some st) (hacc : (run H P D.V x ω r₀).out = .acc) {q : List Msg × Opening Bool}
    (hq : q ∈ found H D decode x st (run H P D.V x ω) c (outerList H D x (run H P D.V x ω) c)) :
    D.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((D.commit.opens q.1 q.2).eval H).2, t ∈ ownLog H (run H P D.V x ω) ((E D decode).ext x r₀ c) := by
  rw [found_eq] at hq
  obtain ⟨h1, h2⟩ := Passes.found H D decode x (run H P D.V x ω) st c _ q hq
  exact ⟨h1, fun t ht => ownLog_E H D decode P x ω r₀ c hst hacc t (h2 t ht)⟩

omit [DecidableEq Col] [Zero Col] in
/-- **A disagreement with the registration is a collision**: at an accepted run, an extracted trace that opens a
registered position to a value other than the registration's makes, with the registration's opening there, calls that
collide among the extraction's own and the registration's checks. -/
theorem regColl_of_disagree (H : Msg → Msg) (D : Deploy Coin Col T) (hb : D.commit.Binds H)
    (decode : List (ℕ × Col) → List T) (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x) {st : AuditStatement}
    (hst : D.parse x = some st) (hreg : RegOpens H D.commit st (reg st) (ops st))
    (hacc : (run H P D.V x s.1 s.2.1).out = .acc) {w : Wit} (hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 = some w)
    {g : ℕ} (hg : g ∈ reg st) {b : Bool} (hwb : w.val H D.commit st g = some b) (hne : b ≠ regVal (ops st) g) :
    RegColl H D decode ops P x s := by
  rw [E_eval H D decode P x s.1 s.2.1 s.2.2 hst hacc, Option.some.injEq] at hw
  subst hw
  obtain ⟨q, hq, hqr, hqp, hqv⟩ := mem_assemble_of_val hwb
  obtain ⟨hqo, hql⟩ := passes_of_mem_found H D decode P x s.1 s.2.1 s.2.2 hst hacc hq
  obtain ⟨o', ho', ho'p, ho'v⟩ := regVal_mem (hreg.2 g hg)
  have hqo' : D.commit.Opens H st.roots q.2 = true := by rw [← hqr]; exact hqo
  have hql' : ∀ t ∈ ((D.commit.opens st.roots q.2).eval H).2,
      t ∈ ownLog H (run H P D.V x s.1) ((E D decode).ext x s.2.1 s.2.2) := by
    intro t ht
    rw [← hqr] at ht
    exact hql t ht
  have hrl : ((D.parse x).map fun st => regLog H D.commit st.roots (ops st)).getD [] =
      regLog H D.commit st.roots (ops st) := by
    rw [hst]
    rfl
  unfold RegColl
  rw [hrl]
  exact disagree_collision hb (hqp.trans ho'p.symm) (by rw [hqv, ho'v]; exact hne) hqo' (hreg.1 o' ho')
    (fun t ht => List.mem_append_left _ (hql' t ht))
    (fun t ht => List.mem_append_right _ (List.mem_flatMap.2 ⟨o', ho', ht⟩))

omit [DecidableEq Col] [Zero Col] in
/-- **The reduction's probability step**: the registration's collisions have probability at most `B_reg`'s. -/
theorem Pr_regColl_le_Breg (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (ops : AuditStatement → List (Opening Bool)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x => RegColl H D decode ops P x s) ≤
      (Breg H D.V (E D decode) D.commit D.parse ops Ω P x).Pr H := by
  unfold Breg Finder.Pr
  refine Pr_mono fun s h => ?_
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome _ h)
  refine ⟨u, v, huv, findColl_spec H _ (fun p hp => ?_) u v huv⟩
  rcases List.mem_append.1 hp with hp | hp
  · exact ownLog_valid H _ _ p hp
  · cases hx : D.parse x with
    | none => rw [hx] at hp; exact absurd hp (by simp)
    | some st => rw [hx] at hp; exact regLog_valid H D.commit st.roots (ops st) p hp

/-- **The leak bound at the registration's values**, at any deployment `D` of the sampled verifier where `KS` holds and
the trace's commitment binds, given `SampledAuditAccepted`. Take a public layout `lay` and horizon `k`, `δ ≥ 0`, a
decoder that lists every trace within Flock's radius, each statement's registered inputs `reg` and the registration's
openings `ops` of them. On the statements in scope whose registration opens their registered inputs:

`Pr[acc ∧ ¬RLeak] ≤ δ + ε_ks + ε_bind + Pr[B_reg]`, with the leak relation at the inputs pinned at the registration's
values (`AsReg` at `regVal`), a set the statement and the registration fix. -/
theorem sampledLeak_registered (hSA : Guarantees.SampledAuditAccepted) (H : Msg → Msg) (D : Deploy Coin Col T)
    (hks : KS H D) (hb : D.commit.Binds H) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ)
    (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ) (x : Msg) (hx : ScopeAcc D lay k x)
    (hreg : ∀ st, D.parse x = some st → RegOpens H D.commit st (reg st) (ops st)) (Ω : Type) [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) :
    PrBad H D.V
        (RLeak H D.commit D.parse D.partition L K S lay k δ
          (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj) (E D decode) x P ≤
      κ D δ x + Real.sqrt (γ D x * PrB H D.V P x) + (Bext H D.V (E D decode) Ω P x).Pr H +
        (Breg H D.V (E D decode) D.commit D.parse ops Ω P x).Pr H := by
  have hacc := hSA H D L K S lay k hks δ decode hδ hdec x hx Ω P
  rw [one_mul] at hacc
  have hpart : ∀ st P', D.parse x = some st → D.partition st = some P' → P'.Partitions st.prog := by
    intro st P' hs hP'
    obtain ⟨s₁, P₁, hs₁, hP₁, hin⟩ := hx.1
    rw [hs] at hs₁
    obtain rfl := Option.some.inj hs₁
    rw [hP'] at hP₁
    obtain rfl := Option.some.inj hP₁
    exact hin.2.1
  unfold PrBad at hacc ⊢
  calc Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
        (run H P D.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLeak H D.commit D.parse D.partition L K S lay k δ
            (fun st _ => AsReg st.prog (lay st) (k st) (reg st) (regVal (ops st))) Dj).ext x
            ((E D decode).eval H P x s.1 s.2.1 s.2.2).1)
      ≤ Pr (fun s : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
          ((run H P D.V x s.1 s.2.1).out = .acc ∧
            ¬ (RAcc H D.commit D.parse D.partition L K S lay k δ).ext x
              ((E D decode).eval H P x s.1 s.2.1 s.2.2).1) ∨ RegColl H D decode ops P x s) := by
        refine Pr_mono fun s ⟨hac, hR⟩ => ?_
        rcases hw : ((E D decode).eval H P x s.1 s.2.1 s.2.2).1 with _ | w
        · exact Or.inl ⟨hac, by simp [Rel.ext]⟩
        · rw [hw] at hR
          by_cases hA : RAcc H D.commit D.parse D.partition L K S lay k δ x w
          · refine Or.inr ?_
            by_contra hnc
            refine hR (rLeak_reg_of_rAcc reg (fun st => regVal (ops st)) hpart (fun st hst g hg b hwb => ?_) hA)
            by_contra hne
            exact hnc (regColl_of_disagree H D hb decode reg ops P x s hst (hreg st hst) hac hw hg hwb hne)
          · exact Or.inl ⟨hac, hA⟩
    _ ≤ _ := Pr_union_le _ _
    _ ≤ _ := add_le_add hacc (Pr_regColl_le_Breg H D decode ops P x)

end Deploy

end NetTiming
