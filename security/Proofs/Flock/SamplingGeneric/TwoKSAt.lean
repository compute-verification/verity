module

public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.KSAt
public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.SamplingGeneric.TwoCap

/-! The two-stage extractor's costs, and the two-stage KS read at a time budget. -/

@[expose] public section

namespace Claims.SamplingGeneric

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Nonempty Coin] in
theorem Sessions.toCostContract {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) :
    L.toCompiled.CostContract H :=
  { toRunsFixed := ⟨hS.fixed, hS.runs⟩, slots := hS.slots, keys := hS.keys }

theorem ksAt_of_ks {H : Msg → Msg} {L : CompiledTwo Coin A} (hks : L.KS H) {runs : Msg → ℝ} {own : Msg → ℕ}
    (hruns : ExtractRunsExpLe H L.E runs) (hown : Sampling.ExtractOwnLe H L.E own)
    (hw : ∀ x, runs x + own x ≤ L.work x) : L.KSAt H :=
  ⟨fun δ hδ => Sampling.KSAt_of_KSLinOn L.work (hks δ hδ), Sampling.BextAtCostLe_of hruns hown hw⟩

theorem KSAt.incorrectBlocks_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hks : L.KSAt H) {x : Msg} (hx : L.Scope x)
    {s : Sampling.AuditStatement} {Pt : Sampling.Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) {r₁ r₂ δ : ℝ} {K : ℕ} (h₁ : d.FirstRate r₁)
    (h₂ : d.SecondRate Pt r₂) (hr₁ : 0 ≤ r₁) (hr₁' : r₁ ≤ 1) (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1) (hδ : 0 ≤ δ)
    (hK : (1 - r₁ * r₂) ^ K ≤ δ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ) :
    ENNReal.ofReal (Pr fun z : Ω × (Fin (L.V.k x) → Coin) × L.E.C x =>
        (run H P L.V x z.1 z.2.1).out = .acc ∧ ∀ w, (L.E.eval H P x z.1 z.2.1 z.2.2).1 = some w →
          ∀ ι, K ≤ (d.incorrectBlocks Pt (Sampling.footprint H L.commit s Pt d.part w ι)).card) ≤
      ENNReal.ofReal (L.κ δ x) + ENNReal.ofReal (Real.sqrt (L.γ x)) * sbB H L.V P x (2 * T) +
        ecrBound H (Sampling.BextAt H L.V L.E T Ω P x) (2 * T * L.work x) := by
  refine le_trans (ENNReal.ofReal_le_ofReal (Pr_mono fun z hz => ⟨hz.1, ?_⟩)) (hks.1 δ hδ x hx Ω P T)
  cases he : (L.E.eval H P x z.1 z.2.1 z.2.2).1 with
  | none => simp [Rel.ext]
  | some w =>
    intro hR
    simp only [Rel.ext, Sampling.RTwo] at hR
    obtain ⟨s', Pt', d', hs', hPt', hd', ι, hι⟩ := hR
    obtain rfl : s' = s := Option.some.inj (hs'.symm.trans hs)
    obtain rfl : Pt' = Pt := Option.some.inj (hPt'.symm.trans hPt)
    obtain rfl : d' = d := Option.some.inj (hd'.symm.trans hd)
    refine absurd (hz.2 w he ι) (Nat.not_le.2 ?_)
    refine d'.card_incorrectBlocks_lt Pt' (fun _ => r₁) (fun S hS => ?_) (fun _ _ => ⟨le_rfl, hr₁'⟩) hr₁ hr₁' hr₂ hr₂'
      h₂ hK hι (Sampling.footprint_le_card H L.commit s' Pt' d'.part w ι)
    show d'.replay.Pa d'.part S ≤ ∏ _ρ ∈ S, (1 - r₁)
    rw [Finset.prod_const]
    exact h₁ S hS

theorem KSAt.incorrectAtBoundary_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hks : L.KSAt H) {x : Msg} (hx : L.Scope x)
    {s : Sampling.AuditStatement} {Pt : Sampling.Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) {r₁ r₂ δ : ℝ} {K : ℕ} (h₁ : d.FirstRate r₁)
    (h₂ : d.SecondRate Pt r₂) (hr₁ : 0 ≤ r₁) (hr₁' : r₁ ≤ 1) (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1) (hδ : 0 ≤ δ)
    (hK : (1 - r₁ * r₂) ^ K ≤ δ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (T : ℕ) :
    ENNReal.ofReal (Pr fun z : Ω × (Fin (L.V.k x) → Coin) × L.E.C x =>
        (run H P L.V x z.1 z.2.1).out = .acc ∧ ∀ w, (L.E.eval H P x z.1 z.2.1 z.2.2).1 = some w →
          K ≤ (Sampling.incorrectAtBoundary H L.commit s Pt d.part w).card) ≤
      ENNReal.ofReal (L.κ δ x) + ENNReal.ofReal (Real.sqrt (L.γ x)) * sbB H L.V P x (2 * T) +
        ecrBound H (Sampling.BextAt H L.V L.E T Ω P x) (2 * T * L.work x) :=
  le_trans (ENNReal.ofReal_le_ofReal (Pr_mono fun _ hz => ⟨hz.1, fun w hw ι =>
      (hz.2 w hw).trans (Finset.card_le_card (Sampling.incorrectAtBoundary_subset H L.commit s Pt d w ι))⟩))
    (hks.incorrectBlocks_le hx hs hPt hd h₁ h₂ hr₁ hr₁' hr₂ hr₂' hδ hK Ω P T)

end CompiledTwo

/-- **`CompilerEfficientTwo`**: `Compiled.efficient_of` at the two-stage contract's cost facts. -/
theorem compilerEfficientTwo : CompilerEfficientTwo := by
  intro Coin _ _ _ A L H hS hopen
  exact Compiled.efficient_of hS.toCostContract hopen

theorem compilerKSAtTwo_of (hks : CompilerKSTwo) (heff : CompilerEfficientTwo) : CompilerKSAtTwo := by
  intro Coin _ _ _ A L H hA hB hS hopen
  obtain ⟨-, hr, ho⟩ := heff Coin A L H hS hopen
  exact CompiledTwo.ksAt_of_ks (hks Coin A L H hA hB hS) hr ho
    fun x => le_of_eq (by unfold Compiled.work; push_cast; ring)

/-- **KS read at a time budget at two stages**: `CompilerKSTwo` and `CompilerEfficientTwo` (`compilerKSAtTwo_of`). -/
theorem compilerKSAtTwo : CompilerKSAtTwo := compilerKSAtTwo_of compilerKSTwo compilerEfficientTwo

end Claims.SamplingGeneric
