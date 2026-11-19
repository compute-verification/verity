module

public import Proofs.Flock.SamplingGeneric.RegTrace
public import Proofs.Flock.SamplingGeneric.TwoReg
public import Proofs.Flock.SamplingGeneric.LevelsRealize

/-!
# The committed form at levels: one and two levels

`CompilerKSRegLev` on the statements whose levels are one (`CompiledLev.OneLevel`) or two (`CompiledLev.TwoLevel`), on
the one- and two-stage forms' machinery at the contract's clauses they read (`Compiled.SharedOpens`,
`CompiledTwo.SharedOpensTwo`).

* `KSRegLevAt`: `KSRegLev`'s two conjuncts at one statement.
* `ksRegLev_one`: **the committed form at levels at one level**. The proof blocks incorrect at their best interiors
  (`bestIota`) are inside `Freg` (`correct_refIn_congr`), so an accepted run samples one with probability at most
  `ε_reg`, and misses them with probability at most `δ` when the levels do (`pr_missLev_le`).
* `ksRegLev_two`: **the committed form at levels at two levels**. The event is the two-stage form's at the top
  partition (`CompiledTwo.eventTwo_le_of`); the miss reads `Samples`' first and last clauses at a best interior per top
  block.
* `compilerKSReg_lev_one`, `compilerKSRegTwo_lev_two`: `CompilerKSReg`, and `CompilerKSRegTwo`'s conclusion under
  `n_PU ≤ n_out`, as the levels statement at one and two levels, through `levelsRealizeOne` and `levelsRealizeTwo`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeLev slot incorrectLev levM forkOn correct_refIn_congr replayOf
  incorrectAllAt incorrectInAt cell)

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`KSRegLev` at one statement** `x`: its two conjuncts, at every prover. -/
def KSRegLevAt (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregLev H P x p.1 p.2) ≤ L.εregLev H x P ∧
      ∀ s Pt Lv, L.parse x = some s → L.partition s = some Pt → L.levels s Pt = some Lv → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∀ ι : ℕ → ℕ → Bool, incorrectLev s Pt (L.top s) (L.regVal H P x p.1) ι ∉ levM Pt Lv δ) ≤
          δ + L.εregLev H x P

theorem ksRegLev_iff {H : Msg → Msg} {L : CompiledLev Coin A} :
    L.KSRegLev H ↔ ∀ x, L.Scope x → L.KSRegLevAt H x :=
  Iff.rfl

/-- **The levels' contract at one level gives `SharedOpens`**: a proof block is its own top block. -/
theorem sharedOpens_one {H : Msg → Msg} {L : CompiledLev Coin A} (hS : L.Sessions H) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {Lv : Verity.Levels (Fin Pt.blocks.length)} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hLv : L.levels s Pt = some Lv) (hin : InScopeLev Pt Lv (L.top s) s)
    (htop : L.top s = none) : L.toCompiled.SharedOpens H x s Pt := by
  refine ⟨hS.slots x, hS.shared x, fun Ω Pv ω r u xS vS w hacc hu hsess hR => ?_⟩
  obtain ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩ := hS.opens x s Pt Lv Ω Pv ω r u xS vS w hs hPt hLv hin hacc hu hsess hR
  rw [htop] at hρ hrb
  have hρ' : some u = some ρ := hρ
  have hrb' : Pt.blocks[ρ]? = some rb := hrb
  obtain rfl : u = ρ := Option.some.inj hρ'
  obtain rfl : b = rb := Option.some.inj (hb.symm.trans hrb')
  exact ⟨b, hb, hcor⟩

omit [Inhabited Coin] in
/-- **The levels' miss at one level**: a set `F` of proof blocks outside `M_δ` is missed with probability at most `δ`
(`Samples`' one-level clause). -/
theorem pr_missLev_le {L : CompiledLev Coin A} {x : Msg} {Pt : Partition} {Lv : Verity.Levels (Fin Pt.blocks.length)}
    (hsam : L.Samples x Pt Lv) (h1 : Lv.levels.length = 1) (F : Finset ℕ) (hF : F ⊆ range Pt.blocks.length)
    {δ : ℝ} (hδ : 0 ≤ δ) :
    Pr (fun r : Fin (L.V.k x) → Coin => F ∉ levM Pt Lv δ ∧ ∀ u ∈ L.checked x r, u ∉ F) ≤ δ := by
  by_cases hM : F ∈ levM Pt Lv δ
  · exact (Pr_mono (q := fun _ => False) fun _ h => h.1 hM).trans (Pr_false.trans_le hδ)
  · obtain ⟨esc, -, hesc, hEesc⟩ := hsam.1 h1 F hF
    have hM' : ¬ δ < (Lv.design.missProb (F.image (slot Pt))).toReal := hM
    calc Pr (fun r : Fin (L.V.k x) → Coin => F ∉ levM Pt Lv δ ∧ ∀ u ∈ L.checked x r, u ∉ F)
        ≤ Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ F) := Pr_mono fun _ h => h.2
      _ = Ex fun r => Pr fun d : Fin (L.V.k x) → Coin =>
            ∀ u ∈ L.checked x (forkOn (L.secondStage x) r d), u ∉ F := Pr_forkOn (L.secondStage x) _
      _ ≤ Ex esc := Ex_mono hesc
      _ ≤ δ := hEesc.trans (not_lt.1 hM')

set_option synthInstance.maxSize 1024 in
/-- **The committed form at levels at one level**: `KSRegLev`'s conjuncts at every statement in scope with no top
partition. The event is `Compiled.pr_freg_le_of` at `sharedOpens_one`; the proof blocks incorrect at their best
interiors on the committed trace are inside `Freg`, so a run accepting with them outside `M_δ` samples one of them or
misses them (`pr_missLev_le`). -/
theorem ksRegLev_one {H : Msg → Msg} {L : CompiledLev Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} (hx : L.Scope x) (hone : L.OneLevel x) : L.KSRegLevAt H x := by
  classical
  intro Ω _ _ P
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin⟩ := hx
  have htop := hone s hs
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  have hSO := sharedOpens_one hS hs hPt hLv hin htop
  have hpart : L.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hε : L.εregLev H x P = L.εreg H x P := by
    unfold CompiledLev.εregLev
    have h : (L.parse x >>= L.top) = none := by
      rw [hs]
      exact htop
    rw [h]
    try rfl
  have hF : ∀ ω r, L.FregLev H P x ω r = L.Freg H P x ω := by
    intro ω r
    unfold CompiledLev.FregLev
    rw [hs]
    dsimp only
    rw [hPt, htop]
    try rfl
  have hFreg : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregLev H P x p.1 p.2) ≤ L.εregLev H x P := by
    rw [hε]
    simp only [hF]
    exact Compiled.pr_freg_le_of hbind hSO hSK hs hPt P
  refine ⟨hFreg, fun s' Pt' Lv' hs' hPt' hLv' δ hδ => ?_⟩
  obtain rfl : s = s' := Option.some.inj (hs.symm.trans hs')
  obtain rfl : Pt = Pt' := Option.some.inj (hPt.symm.trans hPt')
  obtain rfl : Lv = Lv' := Option.some.inj (hLv.symm.trans hLv')
  have hsam := hS.samples x s Pt Lv hs hPt hLv hin
  have h1 : Lv.levels.length = 1 := by
    have := hin.2.2.2
    rwa [htop] at this
  let F : Ω → Finset ℕ := fun ω =>
    incorrectLev s Pt none (L.regVal H P x ω) (bestIota s Pt (L.regVal H P x ω))
  have hFr : ∀ ω, F ω ⊆ range Pt.blocks.length := fun ω =>
    Claims.Sampling.incorrectLev_subset_range _ _ _ _
  have hsub : ∀ ω, F ω ⊆ L.Freg H P x ω := by
    intro ω v hv
    simp only [Compiled.Freg, hs, hpart, mem_filter, mem_range]
    refine ⟨mem_range.1 (hFr ω hv), ?_⟩
    rintro ⟨b, hb, val, ι, hval, hc⟩
    obtain ⟨b', hb', hnc⟩ := incorrect_bestIota hv ι
    obtain rfl : b = b' := Option.some.inj (hb.symm.trans hb')
    exact hnc ((correct_refIn_congr hin.1 hin.2.1 hb b hval fun _ _ _ => rfl).1 hc)
  have hhit : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ F p.1) ≤ L.εregLev H x P :=
    (Pr_mono fun p ⟨hacc, u, hu, huF⟩ => ⟨hacc, u, hu, by rw [hF]; exact hsub p.1 huF⟩).trans hFreg
  have hmiss : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => F p.1 ∉ levM Pt Lv δ ∧
      ∀ u ∈ L.checked x p.2, u ∉ F p.1) ≤ δ :=
    (Sampling.Pr_prod_eq_Ex _).trans_le
      ((Ex_mono fun ω => pr_missLev_le hsam h1 (F ω) (hFr ω) hδ).trans (Ex_const δ).le)
  refine (Sampling.Pr_or_le (fun p hp => ?_) hhit hmiss).trans_eq (add_comm _ _)
  by_cases hm : ∃ u ∈ L.checked x p.2, u ∈ F p.1
  · exact Or.inl ⟨hp.1, hm⟩
  · refine Or.inr ⟨?_, fun u hu huF => hm ⟨u, hu, huF⟩⟩
    have h := hp.2 (bestIota s Pt (L.regVal H P x p.1))
    rwa [htop] at h

/-- **The levels' contract with a top partition `Q` gives `SharedOpensTwo`** at the two-stage view, on `Q`. -/
theorem sharedOpensTwo_lev {H : Msg → Msg} {L : CompiledLev Coin A} (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt Q : Partition} {Lv : Verity.Levels (Fin Pt.blocks.length)} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hLv : L.levels s Pt = some Lv) (hin : InScopeLev Pt Lv (L.top s) s)
    (hQ : L.top s = some Q) : L.twoView.SharedOpensTwo H x s Pt Q := by
  have hT := hin.2.2.2
  rw [hQ] at hT
  refine ⟨hS.slots x, hS.shared x, fun Ω P ω r e ρ => (hS.interior x Ω P ω r e ρ).1,
    fun Ω Pv ω r u xS vS w hacc hu hsess hR => ?_, hin.1, hin.2.1, hT.1⟩
  obtain ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩ := hS.opens x s Pt Lv Ω Pv ω r u xS vS w hs hPt hLv hin hacc hu hsess hR
  rw [hQ] at hρ hrb
  exact ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩

/-- With every proof block in a block of `Q`, the incorrect proof blocks at the top partition `Q` are inside the grid's
(`incorrectAllAt`). -/
theorem incorrectLev_subset_all {s : AuditStatement} {Pt Q : Partition} (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (hsome : ∀ v < Pt.blocks.length, (replayOf Pt Q v).isSome) :
    incorrectLev s Pt (some Q) val ι ⊆ incorrectAllAt s Pt Q val ι := by
  classical
  intro v hv
  unfold incorrectLev at hv
  obtain ⟨hvr, hnot⟩ := mem_filter.1 hv
  have hvl := mem_range.1 hvr
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hsome v hvl)
  simp only [incorrectAllAt, mem_biUnion, mem_product, mem_range]
  refine ⟨(ρ, Pt.stratumOf v), ⟨Claims.Sampling.replayOf_lt hρ, Claims.Sampling.stratumOf_lt hvl⟩, ?_⟩
  show v ∈ incorrectInAt s Pt Q val ρ (Pt.stratumOf v) (ι ρ)
  unfold incorrectInAt cell
  exact mem_filter.2 ⟨mem_filter.2 ⟨mem_range.2 hvl, hρ, rfl⟩,
    fun ⟨u, r, hu, hr, hc⟩ => hnot ⟨u, ρ, r, hu, hρ, hr, hc⟩⟩

set_option synthInstance.maxSize 1024 in
open Classical in
/-- **The committed form at levels at two levels**: `KSRegLev`'s conjuncts at every statement in scope with two
levels. The event is `CompiledTwo.eventTwo_le_of` at the two-stage view (`sharedOpensTwo_lev`), and the proof blocks
incorrect at the committed interiors are inside it (`incorrectLev_subset_all`). Off it, a run accepting with every
interior's incorrect set outside `M_δ` misses the set at the committed interiors, which no second-stage coin moves;
in each first-level block it is missed no more often than the set at a best interior per top block (`exists_best`),
so by `Samples`' last clause and `Ex_prod_parts_le` with its second, at most that set's miss, `δ`. -/
theorem ksRegLev_two {H : Msg → Msg} {L : CompiledLev Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} (hx : L.Scope x) (htwo : L.TwoLevel x) : L.KSRegLevAt H x := by
  intro Ω _ _ P
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin⟩ := hx
  obtain ⟨l₁, l₂, hL⟩ := htwo s Pt Lv hs hPt hLv
  obtain ⟨Q, hQ⟩ : ∃ Q, L.top s = some Q := by
    cases hq : L.top s with
    | none =>
      have h := hin.2.2.2
      rw [hq] at h
      have h' : Lv.levels.length = 1 := h
      rw [hL] at h'
      simp at h'
    | some Q => exact ⟨Q, rfl⟩
  have hT := hin.2.2.2
  rw [hQ] at hT
  obtain ⟨-, hsome, l, rest, hl, -, hiff⟩ := hT
  rw [hL, List.cons.injEq] at hl
  obtain ⟨rfl, -⟩ := hl
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  have hSO := sharedOpensTwo_lev hS hs hPt hLv hin hQ
  have hε : L.εregLev H x P = L.twoView.εregTwo H x P := by
    unfold CompiledLev.εregLev
    have h : (L.parse x >>= L.top) = some Q := by
      rw [hs]
      exact hQ
    rw [h]
  have hF : ∀ ω r, L.FregLev H P x ω r =
      incorrectAllAt s Pt Q (L.regVal H P x ω) (L.twoView.iregVal H P x ω r) := by
    intro ω r
    unfold CompiledLev.FregLev
    rw [hs]
    dsimp only
    rw [hPt, hQ]
    rfl
  have hFreg : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregLev H P x p.1 p.2) ≤ L.εregLev H x P := by
    rw [hε]
    simp only [hF]
    exact CompiledTwo.eventTwo_le_of hbind hSO hSK hs hPt Ω P
  refine ⟨hFreg, fun s' Pt' Lv' hs' hPt' hLv' δ hδ => ?_⟩
  obtain rfl : s = s' := Option.some.inj (hs.symm.trans hs')
  obtain rfl : Pt = Pt' := Option.some.inj (hPt.symm.trans hPt')
  obtain rfl : Lv = Lv' := Option.some.inj (hLv.symm.trans hLv')
  have hsam := hS.samples x s Pt Lv hs hPt hLv hin
  have hlen : ∀ r, (L.upper x Pt r).length = 1 := fun r => by
    have h := hsam.2.2.2.2 r
    rw [hL] at h
    simpa using h
  have hins : ∀ Zb ∈ l₁.blocks.parts, ∀ ω, ∀ b ∈ (l₂.design Zb).draw ω, b ⊆ Zb := by
    have hi := Lv.inside
    rw [hL] at hi
    simp only [Verity.Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
    exact fun Zb hZ ω b hb => (hi.2 Zb hZ ω b hb).2
  have hρ : ∀ Zb ∈ l₁.blocks.parts, ∃ ρ, ∀ i ∈ Zb, replayOf Pt Q i.val = some ρ := by
    intro Zb hZ
    obtain ⟨i, hi⟩ := l₁.blocks.nonempty_of_mem_parts hZ
    obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hsome i.val i.isLt)
    exact ⟨ρ, fun j hj => ((hiff j i).2 ⟨Zb, hZ, hj, hi⟩).trans hρ⟩
  let I : Ω → (ℕ → ℕ → Bool) → Finset ℕ := fun ω ι => incorrectLev s Pt (some Q) (L.regVal H P x ω) ι
  let m : Finset (Fin Pt.blocks.length) → Finset ℕ → ENNReal := fun Zb Xb =>
    (l₂.design Zb).missProb (Xb.image (slot Pt))
  have hbest : ∀ (ω : Ω) (ρ : ℕ), ∃ ιρ : ℕ → Bool, ∀ Zb ∈ l₁.blocks.parts,
      (∀ i ∈ Zb, replayOf Pt Q i.val = some ρ) → ∀ ι' : ℕ → Bool, m Zb (I ω fun _ => ι') ≤ m Zb (I ω fun _ => ιρ) := by
    intro ω ρ
    by_cases hex : ∃ Z₀ ∈ l₁.blocks.parts, ∀ i ∈ Z₀, replayOf Pt Q i.val = some ρ
    · obtain ⟨Z₀, hZ₀, hZ₀ρ⟩ := hex
      obtain ⟨ιρ, hιρ⟩ := exists_best (fun ι' : ℕ → Bool => I ω fun _ => ι') (range Pt.blocks.length)
        (fun _ => incorrectLev_subset _ _ _ _ _) (m Z₀)
      refine ⟨ιρ, fun Zb hZ hZρ ι' => ?_⟩
      obtain ⟨i, hi⟩ := l₁.blocks.nonempty_of_mem_parts hZ
      obtain ⟨j, hj⟩ := l₁.blocks.nonempty_of_mem_parts hZ₀
      obtain ⟨Xb, hX, hiX, hjX⟩ := (hiff i j).1 ((hZρ i hi).trans (hZ₀ρ j hj).symm)
      have h1 : Zb = Xb := l₁.blocks.eq_of_mem_parts hZ hX hi hiX
      have h2 : Z₀ = Xb := l₁.blocks.eq_of_mem_parts hZ₀ hX hj hjX
      rw [h1, ← h2]
      exact hιρ ι'
    · exact ⟨fun _ => false, fun Zb hZ hZρ _ => absurd ⟨Zb, hZ, hZρ⟩ hex⟩
  choose ιs hιs using hbest
  have hXY : ∀ ω r, ∀ Zb ∈ l₁.blocks.parts,
      m Zb (I ω (L.twoView.iregVal H P x ω r)) ≤ m Zb (I ω (ιs ω)) := by
    intro ω r Zb hZ
    obtain ⟨ρ, hZρ⟩ := hρ Zb hZ
    have e1 := missProb_incorrectLev_congr (s := s) (val := L.regVal H P x ω) (hins Zb hZ) hZρ
      (L.twoView.iregVal H P x ω r) (fun _ => L.twoView.iregVal H P x ω r ρ) rfl
    have e2 := missProb_incorrectLev_congr (s := s) (val := L.regVal H P x ω) (hins Zb hZ) hZρ (ιs ω)
      (fun _ => ιs ω ρ) rfl
    simp only [m, I] at e1 e2 ⊢
    rw [e1, e2]
    exact hιs ω ρ Zb hZ hZρ _
  have hmiss1 : ∀ ω, I ω (ιs ω) ∉ levM Pt Lv δ →
      Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ I ω (L.twoView.iregVal H P x ω r)) ≤ δ := by
    intro ω hM
    have hM' : ¬ δ < (Lv.design.missProb ((I ω (ιs ω)).image (slot Pt))).toReal := hM
    let Ub : (Fin (L.V.k x) → Coin) → Finset (Finset (Fin Pt.blocks.length)) := fun r => (L.upper x Pt r).headD ∅
    calc Pr (fun r : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x r, u ∉ I ω (L.twoView.iregVal H P x ω r))
        = Ex fun r => Pr fun e : Fin (L.V.k x) → Coin =>
            ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ I ω (L.twoView.iregVal H P x ω r) := by
          refine (Pr_forkOn (L.secondStage x) _).trans (congrArg Ex (funext fun r => ?_))
          refine congrArg Pr (funext fun e => ?_)
          exact congrArg (fun ι => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e), u ∉ I ω ι)
            (CompiledTwo.iregVal_forkOn H L.twoView P x ω r e)
      _ ≤ Ex fun r => ∏ Zb ∈ Ub r, (m Zb (I ω (L.twoView.iregVal H P x ω r))).toReal := by
          refine Ex_mono fun r => (hsam.2.2.2.1 l₂ (by rw [hL]; rfl) (by rw [hL]; simp) r _).trans_eq ?_
          rw [getLastD_eq_headD (hlen r)]
      _ ≤ Ex fun r => ∏ Zb ∈ (Ub r).filter (· ∈ l₁.blocks.parts), (m Zb (I ω (ιs ω))).toReal := by
          refine Ex_mono fun r => ?_
          rw [← prod_filter_mul_prod_filter_not (Ub r) (· ∈ l₁.blocks.parts)]
          have h1 : ∏ Zb ∈ (Ub r).filter (fun Zb => ¬ Zb ∈ l₁.blocks.parts),
              (m Zb (I ω (L.twoView.iregVal H P x ω r))).toReal ≤ 1 :=
            prod_le_one₀ (fun _ _ => ENNReal.toReal_nonneg) fun Zb _ => missProb_toReal_le_one _ _
          refine (mul_le_mul_of_nonneg_left h1 (prod_nonneg fun _ _ => ENNReal.toReal_nonneg)).trans ?_
          rw [mul_one]
          refine prod_le_prod₀ (fun _ _ => ENNReal.toReal_nonneg) fun Zb hZ => ?_
          exact ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top
            (Verity.SamplingDesign.missProb_le_one _ _)) (hXY ω r Zb (mem_filter.1 hZ).2)
      _ ≤ (Lv.design.missProb ((I ω (ιs ω)).image (slot Pt))).toReal :=
          Ex_prod_parts_le Lv hL Ub (hsam.2.1 l₁ l₂ [] hL) _
      _ ≤ δ := not_lt.1 hM'
  have hmiss : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => I p.1 (ιs p.1) ∉ levM Pt Lv δ ∧
      ∀ u ∈ L.checked x p.2, u ∉ I p.1 (L.twoView.iregVal H P x p.1 p.2)) ≤ δ := by
    refine (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono fun ω => ?_).trans (Ex_const δ).le)
    by_cases hM : I ω (ιs ω) ∈ levM Pt Lv δ
    · exact (Pr_mono (q := fun _ => False) fun _ h => h.1 hM).trans (Pr_false.trans_le hδ)
    · exact (Pr_mono fun _ h => h.2).trans (hmiss1 ω hM)
  refine (Sampling.Pr_or_le (fun p hp => ?_) hFreg hmiss).trans_eq (add_comm _ _)
  by_cases hm : ∃ u ∈ L.checked x p.2, u ∈ I p.1 (L.twoView.iregVal H P x p.1 p.2)
  · obtain ⟨u, hu, huI⟩ := hm
    refine Or.inl ⟨hp.1, u, hu, ?_⟩
    rw [hF]
    exact incorrectLev_subset_all _ _ hsome huI
  · refine Or.inr ⟨?_, fun u hu huI => hm ⟨u, hu, huI⟩⟩
    have h := hp.2 (ιs p.1)
    rwa [hQ] at h

end CompiledLev

/-- **`CompilerKSReg` is the levels statement at one level**: `CompiledLev.ksRegLev_one` at `Compiled.liftLev`, read
back through `levelsRealizeOne`. -/
theorem compilerKSReg_lev_one : CompilerKSReg :=
  fun _ _ _ _ _ _ _ hA hB hS => Compiled.ksReg_of_liftLev Claims.Sampling.levelsRealizeOne fun _ hx =>
    CompiledLev.ksRegLev_one hA hB (Compiled.sessions_liftLev Claims.Sampling.levelsRealizeOne hS) hx fun _ _ => rfl

/-- **`CompilerKSRegTwo`'s conclusion is the levels statement at two levels**, under `n_PU ≤ n_out`:
`CompiledLev.ksRegLev_two` at `CompiledTwo.liftLev`, read back through `levelsRealizeTwo`. -/
theorem compilerKSRegTwo_lev_two (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin)
    (L : CompiledTwo Coin A) (H : Msg → Msg) (hA : A.KS H) (hB : L.commit.Binds H) (hS : L.Sessions H)
    (hpu : ∀ x, L.nPU x ≤ L.nOut x) : L.KSRegTwo H :=
  CompiledTwo.ksRegTwo_of_liftLev Claims.Sampling.levelsRealizeTwo fun _ hx =>
    CompiledLev.ksRegLev_two hA hB (CompiledTwo.sessions_liftLev Claims.Sampling.levelsRealizeTwo hS hpu) hx
      fun _ _ _ _ _ hLv => by
        obtain ⟨_, -, -, -, -, -, l₁, l₂, hL, -⟩ := CompiledTwo.liftLev_levels hLv
        exact ⟨l₁, l₂, hL⟩

end Claims.SamplingGeneric
