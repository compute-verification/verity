import Definitions.NetworkCertifier.Guarantees.SampledAudit
import Proofs.NetworkCertifier.SampledCharge
import Proofs.Flock.Claims.Prob

/-!
# The leak bound under a sampled audit: `SampledAuditAccepted` composed with the count

On `RAcc δ` the faulty set is one of `faultySets d P δ` (`faulty_mem_faultySets`), and every observation sequence the
trace's transfer gates carry with every status complete is `Accepted` with advice held only at that set's committed
gates. With inputs besides the ingress among `As` (`AdviceIn`), it is then among the sequences `AccSeqsSampled` counts, so
`RAcc δ` implies the leak relation `RLeak` (`rLeak_of_rAcc`), and `KSLinOn`'s bound moves from one to the other
(`ksLinOn_rLeak`). At any deployment where `KS` and `SampledAuditAccepted` hold, the run's observations are then among
at most `2^c · #faultySets · 2^β · D^n` public sequences, `c + log2 #faultySets + β + n·log2 D` bits, except with
`KSAccepted`'s probability (`sampledLeak`).

`RAcc` bounds no input besides the ingress, so `AdviceIn` is a hypothesis on the circuit and `As`. It holds, for every
circuit, at the inputs that are `false` off the free inputs (`adviceIn_asFree`), whose count is `2^#freeInputs`
(`encard_asFree_le`): `sampledLeak_free` is the bound with `c = #freeInputs` and no hypothesis on the inputs.
-/

namespace NetTiming

open Claims Claims.Sampling

/-! ### `RAcc δ` implies the leak relation -/

theorem faulty_mem_faultySets {H : Msg → Msg} {Cm : Commitment Bool} {δ : ℝ} {s : Stmt} {P : Partition} {w : Wit}
    (hRel : RelS H Cm δ s P w) (hdisj : Disjoint (faulty H Cm s P w) (allSlots s.design P)) :
    faulty H Cm s P w ∈ faultySets s.design P δ := by
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset]
  exact ⟨fun _ hv => (Finset.mem_filter.mp hv).1, hRel, hdisj⟩

theorem rLeak_of_rAcc {H : Msg → Msg} {Cm : Commitment Bool} {parse : Msg → Option Stmt}
    {partition : Stmt → Option Partition} {L K S : ℕ} {lay : Stmt → Layout L S} {k : Stmt → ℕ} {δ : ℝ}
    {As : Stmt → Set (ℕ → Bool)} {Dj : ℕ}
    (hin : ∀ s P, partition s = some P → AdviceIn K s.prog P s.design (lay s) (k s) (As s) δ) (x : Msg) (w : Wit)
    (h : RAcc H Cm parse partition L K S lay k δ x w) : RLeak H Cm parse partition L K S lay k δ As Dj x w := by
  obtain ⟨s, P, hs, hP, hRel, hdisj, adv, hfree, hacc⟩ := h
  have hF := faulty_mem_faultySets hRel hdisj
  exact ⟨s, P, hs, hP, fun op o hC hCov hl hD =>
    ⟨hl, hD, _, hF, hin s P hP _ hF op o ⟨adv.1, Set.mem_univ _, adv.2, hfree, hacc op o hC hCov⟩⟩⟩

theorem ext_mono {W : Type} {R R' : Rel W} {x : Msg} (h : ∀ w, R x w → R' x w) :
    ∀ ow, R.ext x ow → R'.ext x ow
  | some w, hw => h w hw
  | none, hw => hw

/-- **`KSLinOn` moves from `RAcc δ` to the leak relation**, at any verifier, scope, extractor and errors. -/
theorem ksLinOn_rLeak {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (Cm : Commitment Bool) (parse : Msg → Option Stmt) (partition : Stmt → Option Partition) (L K S : ℕ)
    (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (As : Stmt → Set (ℕ → Bool)) (Dj : ℕ) (V : Verifier Coin)
    (Scope : Msg → Prop) (E : Extractor Coin Wit V) (κ γ β : Msg → ℝ) (Fe : FinderFamily)
    (hin : ∀ s P, partition s = some P → AdviceIn K s.prog P s.design (lay s) (k s) (As s) δ)
    (h : KSLinOn H V Scope (RAcc H Cm parse partition L K S lay k δ) E κ γ β Fe) :
    KSLinOn H V Scope (RLeak H Cm parse partition L K S lay k δ As Dj) E κ γ β Fe := by
  intro x hx Ω _ _ P
  refine le_trans ?_ (h x hx Ω P)
  unfold PrBad
  exact Pr_mono fun r hr => ⟨hr.1, fun hR => hr.2 (ext_mono (rLeak_of_rAcc hin x) _ hR)⟩

/-! ### The inputs `false` off the free inputs -/

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
    · rw [if_pos hany, if_pos hany]
    · have hmem : g ∈ freeInputs C lay k :=
        Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (List.getElem?_eq_some_iff.mp hg).1, hg, Bool.eq_false_iff.mpr hany⟩
      rw [if_neg hany, if_neg hany, if_pos hmem]
  · rfl

theorem adviceIn_asFree {L S : ℕ} (K : ℕ) (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (k : ℕ)
    (δ : ℝ) : AdviceIn K C P d lay k (AsFree C lay k) δ := by
  rintro F - op o ⟨a, -, free, hfree, hacc⟩
  refine ⟨fun g => if g ∈ freeInputs C lay k then a g else false, ?_, free, hfree, ?_⟩
  · show ∀ g, g ∉ freeInputs C lay k → _
    intro g hg
    exact if_neg hg
  · intro t ht
    obtain ⟨h, hr, hs⟩ := hacc t ht
    refine ⟨h, ?_, hs⟩
    have hv : valOver C (lay.inputOf k (ingressUpTo op o t) a) free =
        valOver C (lay.inputOf k (ingressUpTo op o t) fun g => if g ∈ freeInputs C lay k then a g else false) free :=
      funext (valOver_congr_input C free fun i hi => inputOf_restrict C lay k _ a hi)
    rw [hr]
    simp only [circuitProgram, hv]

theorem encard_asFree_le {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) :
    (AsFree C lay k).encard ≤ ((2 ^ (freeInputs C lay k).card : ℕ) : ℕ∞) := by
  classical
  set G := freeInputs C lay k with hG
  let code : (ℕ → Bool) → (↥G → Bool) := fun a g => a g
  have hinj : Set.InjOn code (AsFree C lay k) := by
    intro a ha a' ha' h
    have ha : ∀ g, g ∉ G → a g = false := ha
    have ha' : ∀ g, g ∉ G → a' g = false := ha'
    funext g
    by_cases hg : g ∈ G
    · exact congrFun h ⟨g, hg⟩
    · rw [ha g hg, ha' g hg]
  calc (AsFree C lay k).encard = (code '' AsFree C lay k).encard := hinj.encard_image.symm
    _ ≤ ((Finset.univ : Finset (↥G → Bool)) : Set (↥G → Bool)).encard :=
        Set.encard_le_encard (by simp only [Finset.coe_univ]; exact Set.subset_univ _)
    _ = ((2 ^ G.card : ℕ) : ℕ∞) := by
        rw [Set.encard_coe_eq_coe_finsetCard]
        simp

/-! ### The leak bound, at any deployment -/

section Deploy

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The leak bound under a sampled audit**, at any deployment `D` of the sampled verifier where `KS` holds, given
`SampledAuditAccepted` (PRF-144). Take a public layout `lay` and horizon `k` of each statement, `δ ≥ 0`, a decoder that
lists every trace within Flock's radius, and for each statement inputs besides the ingress `As`, at most `2^c` of them,
that suffice (`AdviceIn`), and `β` bounding each possible faulty set's committed bits (`faultyBits`). On the statements
in scope:

* `Pr[acc ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind`, `KSAccepted`'s bound: except with that probability, every observation
  sequence of length `k` with jitter below `Dj` that the trace's transfer gates carry with every status complete is among
  the sequences `AccSeqsSampled` counts;
* those sequences number at most `2^c · #faultySets(δ) · 2^β · Dj^k`, which the statement, `δ`, `As`, the operator and
  `Dj` fix: the leak is at most `c + log2 #faultySets(δ) + β + k·log2 Dj` bits. -/
theorem sampledLeak (hSA : Guarantees.SampledAuditAccepted) (H : Msg → Msg) (D : Deploy Coin Col T) (hks : KS H D)
    (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T) (hδ : 0 ≤ δ)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (As : Stmt → Set (ℕ → Bool))
    (c β : Stmt → ℕ) (Dj : ℕ) (hAs : ∀ s, (As s).encard ≤ ((2 ^ c s : ℕ) : ℕ∞))
    (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s)
    (hin : ∀ s P, D.partition s = some P → AdviceIn K s.prog P s.design (lay s) (k s) (As s) δ) :
    KSLinOn H D.V (ScopeAcc D lay k) (RLeak H D.commit D.parse D.partition L K S lay k δ As Dj) (E D decode)
        (κ D δ) (γ D) (fun _ => 1) (Bext H D.V (E D decode)) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        ∀ op : ℕ → List (Obs L K S) → List Bool,
          (AccSeqsSampled s.prog P s.design (lay s) (k s) (As s) δ op Dj).encard ≤
            ((2 ^ c s * (faultySets s.design P δ).card * 2 ^ β s * Dj ^ k s : ℕ) : ℕ∞) := by
  refine ⟨ksLinOn_rLeak H D.commit D.parse D.partition L K S lay k δ As Dj D.V _ _ _ _ _ _ hin
    (hSA H D L K S lay k hks δ decode hδ hdec), fun x hx s P hs hP op => ?_⟩
  obtain ⟨s', hs', hfits⟩ := hx.2
  rw [Option.some_injective _ (hs'.symm.trans hs)] at hfits
  exact encard_accSeqsSampled_le s.prog P s.design (lay s) (k s) hfits δ (As s) (c s) (hAs s) (β s) (hβ s P hP) op Dj

/-- **The leak bound under a sampled audit, with the free inputs**: `sampledLeak` at the inputs `false` off the free
inputs (`AsFree`), which suffice for every circuit, so `c` is the number of free inputs (`freeInputs`). -/
theorem sampledLeak_free (hSA : Guarantees.SampledAuditAccepted) (H : Msg → Msg) (D : Deploy Coin Col T)
    (hks : KS H D) (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T)
    (hδ : 0 ≤ δ) (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode) (β : Stmt → ℕ) (Dj : ℕ)
    (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s) :
    KSLinOn H D.V (ScopeAcc D lay k)
        (RLeak H D.commit D.parse D.partition L K S lay k δ (fun s => AsFree s.prog (lay s) (k s)) Dj) (E D decode)
        (κ D δ) (γ D) (fun _ => 1) (Bext H D.V (E D decode)) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        ∀ op : ℕ → List (Obs L K S) → List Bool,
          (AccSeqsSampled s.prog P s.design (lay s) (k s) (AsFree s.prog (lay s) (k s)) δ op Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s)).card * (faultySets s.design P δ).card * 2 ^ β s * Dj ^ k s : ℕ) :
              ℕ∞) :=
  sampledLeak hSA H D hks L K S lay k δ decode hδ hdec _ (fun s => (freeInputs s.prog (lay s) (k s)).card) β Dj
    (fun _ => encard_asFree_le _ _ _) hβ fun _ _ _ => adviceIn_asFree K _ _ _ _ _ _

end Deploy

end NetTiming
