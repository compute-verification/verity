import Definitions.NetworkCertifier.Guarantees.SampledAudit
import Proofs.NetworkCertifier.SampledCharge
import Proofs.Flock.Claims.Prob

/-!
# The leak bound under a sampled audit: `SampledAuditAccepted` composed with the count

On `RAcc δ` the faulty set is one of `faultySets d P δ` (`faulty_mem_faultySets`), and every observation sequence the
trace's transfer gates carry with every status complete is `Accepted` with advice held only at that set's committed
gates and the trace's committed values as the inputs besides the ingress. With those inputs moved into `As`
(`AdviceIn`), it is then among the sequences `AccSeqsSampled` counts, so
`RAcc δ` implies the leak relation `RLeak` (`rLeak_of_rAcc`), and `KSLinOn`'s bound moves from one to the other
(`ksLinOn_rLeak`). At any deployment where `KS` and `SampledAuditAccepted` hold, the run's observations are then among
at most `2^c · #faultySets · 2^β · D^n` public sequences, `c + log2 #faultySets + β + n·log2 D` bits, except with
`KSAccepted`'s probability (`sampledLeak`).

`AdviceIn` is a hypothesis on the circuit and `As`. It holds, for every circuit, at the inputs pinned to the trace at a
set `R` and `false` off the free inputs (`adviceIn_asReg`), whose count is `2^#(freeInputs \ R)` at any values at `R`
(`encard_asReg_le`): `sampledLeak_reg` is the bound with `c = #(freeInputs \ R)` and no hypothesis on the inputs. With
`R` the registered inputs, the registration's values are left out of `c`.
-/

namespace NetTiming

open Claims Claims.Sampling

/-! ### `RAcc δ` implies the leak relation -/

theorem faulty_mem_faultySets {H : Msg → Msg} {Cm : Commitment Bool} {δ : ℝ} {s : Stmt} {P : Partition} {w : Wit}
    (hRel : RelS H Cm δ s P w) (hdisj : Disjoint (faulty H Cm s P w) (allSlots s.design P)) :
    faulty H Cm s P w ∈ faultySets s.design P δ := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset]
  exact ⟨fun _ hv => (Finset.mem_filter.mp hv).1, hRel, hdisj⟩

theorem rLeak_of_rAcc {H : Msg → Msg} {Cm : Commitment Bool} {parse : Msg → Option Stmt}
    {partition : Stmt → Option Partition} {L K S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
    {As : Stmt → (ℕ → Bool) → Set (ℕ → Bool)} {Dj : ℕ}
    (hin : ∀ s P, partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ) (x : Msg)
    (w : Wit) (h : RAcc H Cm parse partition L K S lay k δ x w) : RLeak H Cm parse partition L K S lay k δ As Dj x w := by
  obtain ⟨s, P, hs, hP, hRel, hdisj, free, hfree, -, hacc⟩ := h
  have hF := faulty_mem_faultySets hRel hdisj
  exact ⟨s, P, hs, hP, fun op o hC hCov hl hD =>
    ⟨hl, hD, _, hF, hin s P hP (traceVal s.prog (w.val H Cm s)) _ hF op o
      ⟨_, Set.mem_singleton _, free, hfree, hacc op o hC hCov⟩⟩⟩

theorem ext_mono {W : Type} {R R' : Rel W} {x : Msg} (h : ∀ w, R x w → R' x w) :
    ∀ ow, R.ext x ow → R'.ext x ow
  | some w, hw => h w hw
  | none, hw => hw

/-- **`KSLinOn` moves from `RAcc δ` to the leak relation**, at any verifier, scope, extractor and errors. -/
theorem ksLinOn_rLeak {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (Cm : Commitment Bool) (parse : Msg → Option Stmt) (partition : Stmt → Option Partition) (L K S : ℕ)
    (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (As : Stmt → (ℕ → Bool) → Set (ℕ → Bool)) (Dj : ℕ)
    (V : Verifier Coin) (Scope : Msg → Prop) (E : Extractor Coin Wit V) (κ γ β : Msg → ℝ) (Fe : FinderFamily)
    (hin : ∀ s P, partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ)
    (h : KSLinOn H V Scope (RAcc H Cm parse partition L K S lay k δ) E κ γ β Fe) :
    KSLinOn H V Scope (RLeak H Cm parse partition L K S lay k δ As Dj) E κ γ β Fe := by
  intro x hx Ω _ _ P
  refine le_trans ?_ (h x hx Ω P)
  unfold PrBad
  exact Pr_mono fun r hr => ⟨hr.1, fun hR => hr.2 (ext_mono (rLeak_of_rAcc hin x) _ hR)⟩

/-! ### The inputs pinned at `R`, `false` off the free inputs -/

theorem valOver_congr_input (C : Prog) {inp inp' : ℕ → Bool} (free : ℕ → Option Bool)
    (h : ∀ i, C.gates[i]? = some .input → inp i = inp' i) (g : ℕ) :
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
        cases x <;> first | rfl | rw [h g hx]
    · rw [valOver_of_some C inp hf, valOver_of_some C inp' hf]

theorem inputOf_restrict {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) (ins : List (List Bool)) (a : ℕ → Bool)
    {g : ℕ} (hg : C.gates[g]? = some .input) :
    lay.inputOf k ins a g = lay.inputOf k ins (fun g => if g ∈ freeInputs C lay k then a g else false) g := by
  unfold Layout.inputOf
  rcases _hR : (List.range ins.length).findSome? (fun i => ((lay.ingress i).findIdx? (· == g)).map (i, ·)) with _ | ⟨i, j⟩
  · simp only
    by_cases hany : (List.range k).any (fun i => (lay.ingress i).contains g) = true
    · rw [ite_eq_left hany, ite_eq_left hany]
    · have hmem : g ∈ freeInputs C lay k :=
        Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (List.getElem?_eq_some_iff.mp hg).1, hg, Bool.eq_false_iff.mpr hany⟩
      rw [ite_eq_right hany, ite_eq_right hany, ite_eq_left hmem]
  · rfl

theorem adviceIn_asReg {L S : ℕ} (K : ℕ) (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (k : ℕ)
    (R : Finset ℕ) (ρ : ℕ → Bool) (δ : ℝ) : AdviceIn K C P d lay k ρ (AsReg C lay k R ρ) δ := by
  rintro F - op o ⟨a, ha, free, hfree, hacc⟩
  rw [Set.mem_singleton_iff] at ha
  subst ha
  refine ⟨fun g => if g ∈ freeInputs C lay k then a g else false, ⟨fun g _ hg => ite_eq_left hg,
    fun g hg => ite_eq_right hg⟩, free, hfree, ?_⟩
  intro t ht
  obtain ⟨h, hr, hs⟩ := hacc t ht
  refine ⟨h, ?_, hs⟩
  have hv : valOver C (lay.inputOf k (ingressUpTo op o t) a) free =
      valOver C (lay.inputOf k (ingressUpTo op o t) fun g => if g ∈ freeInputs C lay k then a g else false) free :=
    funext (valOver_congr_input C free fun i hi => inputOf_restrict C lay k _ a hi)
  rw [hr]
  simp only [circuitProgram, hv]

theorem encard_asReg_le {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) (R : Finset ℕ) (ρ : ℕ → Bool) :
    (AsReg C lay k R ρ).encard ≤ ((2 ^ (freeInputs C lay k \ R).card : ℕ) : ℕ∞) := by
  classical
  set G := freeInputs C lay k \ R with hG
  let code : (ℕ → Bool) → (↥G → Bool) := fun a g => a g
  have hinj : Set.InjOn code (AsReg C lay k R ρ) := by
    rintro a ⟨haR, haF⟩ a' ⟨haR', haF'⟩ h
    funext g
    by_cases hg : g ∈ G
    · exact congrFun h ⟨g, hg⟩
    by_cases hf : g ∈ freeInputs C lay k
    · have hr : g ∈ R := by
        by_contra hr
        exact hg (Finset.mem_sdiff.mpr ⟨hf, hr⟩)
      rw [haR g hr hf, haR' g hr hf]
    · rw [haF g hf, haF' g hf]
  calc (AsReg C lay k R ρ).encard = (code '' AsReg C lay k R ρ).encard := hinj.encard_image.symm
    _ ≤ ((Finset.univ : Finset (↥G → Bool)) : Set (↥G → Bool)).encard :=
        Set.encard_le_encard (by simp only [Finset.coe_univ]; exact Set.subset_univ _)
    _ = ((2 ^ G.card : ℕ) : ℕ∞) := by
        rw [Set.encard_coe_eq_coe_finsetCard]
        simp

/-! ### The leak bound, at any deployment -/

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The leak bound under a sampled audit**, at any deployment `D` of the sampled verifier where `KS` holds, given
`SampledAuditAccepted`. Take a public layout `lay` and horizon `k` of each statement, `δ ≥ 0`, a decoder that
lists every trace within Flock's radius, and for each statement and values `ρ` inputs besides the ingress `As s ρ`, at
most `2^c` of them, that suffice for the inputs `ρ` (`AdviceIn`), and `β` bounding each possible faulty set's committed
bits (`faultyBits`). On the statements in scope:

* `Pr[acc ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind`, `KSAccepted`'s bound: except with that probability, every observation
  sequence of length `k` with jitter below `Dj` that the trace's transfer gates carry with every status complete is among
  the sequences `AccSeqsSampled` counts at `As s ρ`, `ρ` the trace's committed values;
* at every `ρ`, those sequences number at most `2^c · #faultySets(δ) · 2^β · Dj^k`, which the statement, `δ`, `As`, `ρ`,
  the operator and `Dj` fix: the leak is at most `c + log2 #faultySets(δ) + β + k·log2 Dj` bits. -/
theorem sampledLeak (hSA : Guarantees.SampledAuditAccepted) (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D)
    (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode)
    (As : Stmt → (ℕ → Bool) → Set (ℕ → Bool)) (c β : Stmt → ℕ) (Dj : ℕ)
    (hAs : ∀ s ρ, (As s ρ).encard ≤ ((2 ^ c s : ℕ) : ℕ∞))
    (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s)
    (hin : ∀ s P, D.partition s = some P → ∀ ρ, AdviceIn K s.prog P s.design (lay s) (k s) ρ (As s ρ) δ) :
    KSLinOn H D.V (ScopeAcc D lay k) (RLeak H D.commit D.parse D.partition L K S lay k δ As Dj) (E D decode)
        (κ D δ) (γ D) (fun _ => 1) (Bext H D.V (E D decode)) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool),
          (AccSeqsSampled s.prog P s.design (lay s) (k s) (As s ρ) δ op Dj).encard ≤
            ((2 ^ c s * (faultySets s.design P δ).card * 2 ^ β s * Dj ^ k s : ℕ) : ℕ∞) := by
  refine ⟨ksLinOn_rLeak H D.commit D.parse D.partition L K S lay k δ As Dj D.V _ _ _ _ _ _ hin
    (hSA H D L K S lay k hks δ decode hδ hdec), fun x hx s P hs hP op ρ => ?_⟩
  obtain ⟨s', hs', hfits⟩ := hx.2
  rw [Option.some_injective _ (hs'.symm.trans hs)] at hfits
  exact encard_accSeqsSampled_le s.prog P s.design (lay s) (k s) hfits δ (As s ρ) (c s) (hAs s ρ) (β s) (hβ s P hP) op Dj

/-- **The leak bound under a sampled audit, with the registered inputs pinned**: `sampledLeak` at the inputs pinned to
the trace at each statement's `reg` and `false` off the free inputs (`AsReg`), which suffice for every circuit, so `c`
is the number of free inputs not in `reg` (`freeInputs`). -/
theorem sampledLeak_reg (hSA : Guarantees.SampledAuditAccepted) (H : Msg → Msg) (D : Deploy Coin Col T)
    (hks : KS H D) (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (reg : Stmt → Finset ℕ)
    (β : Stmt → ℕ) (Dj : ℕ)
    (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s) :
    KSLinOn H D.V (ScopeAcc D lay k)
        (RLeak H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj)
        (E D decode) (κ D δ) (γ D) (fun _ => 1) (Bext H D.V (E D decode)) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool),
          (AccSeqsSampled s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) δ op Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (faultySets s.design P δ).card * 2 ^ β s *
              Dj ^ k s : ℕ) : ℕ∞) :=
  sampledLeak hSA H D hks L K S lay k δ decode hδ hdec _ (fun s => (freeInputs s.prog (lay s) (k s) \ reg s).card) β
    Dj (fun _ _ => encard_asReg_le _ _ _ _ _) hβ fun _ _ _ _ => adviceIn_asReg K _ _ _ _ _ _ _ _

end Deploy

end NetTiming
