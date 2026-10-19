import Proofs.Accounting.TS

/-!
# RT-LOGIC's witnesses (red team, scratch; not for `main`)

Witnesses for note:20261019T0515Z-draft-rt-logic, each against `main`'s statements. No root imports this file, so the
package's build and its Lean check never see it; it is elaborated with `lake env lean` against the built imports.

* `nfold_eq_zero_of_empty`, `not_foldOpt_of_noFold`, `not_fillOpt_of_noFill`, `not_foldOpt_of_empty`,
  `not_fillOpt_of_empty`: a run that executed nothing (a path the prover stopped on) has no fold and no filler under
  any labelling the run splits, so FOLD-OPT and FILL-OPT are false there whenever `φ*` (`χ*`) is positive
  (RT-LOGIC-1, RT-LOGIC-2). `toy` and `aborted` meet every premise, so the lemmas aren't vacuous.
* `DIdealFloorGuarded`: D-ideal (2.7♭) with every per-path premise read only on accepted paths, by `DIdealFloor`'s
  own proof; `guarded_implies` gets `DIdealFloor` back from it (RT-LOGIC-2).
* `fanIn_warp_false`, `fanIn_lane_true`: a warp IADD3, 32 lane nodes of three sources each issued as one instruction,
  has `issued = 64`, above `FanIn`'s per-lane ALU cap of 8; read per lane it meets the cap (RT-LOGIC-4).
* `sheet_tensor_free`: a class with MACs and no sheet peak costs nothing on the data-sheet machine's tensor row
  (RT-LOGIC-6).
* `bound_smul`, `credit_scaled`, `dispatch_total`: the credit scales with the floor's normalization, and A2 per SMSP
  holds for a circuit's totals only with cycles summed over the SMSPs (RT-LOGIC-8).
-/

namespace Accounting.SecurityProofs.TS.RtLogic

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-! ## FOLD-OPT and FILL-OPT on a path the prover stopped on -/

theorem nfold_eq_zero_of_empty {K T L U : Type} [Fintype T] [Fintype L] [Fintype U] (A : Anatomy K T L U)
    (hA : A.Split) (h0 : ∀ k, A.run.n k = 0) (k : K) : A.nfold k = 0 := by
  obtain ⟨hw, hf, hx, -, hs⟩ := hA
  have h1 := hs k
  rw [h0 k] at h1
  have h2 : 0 ≤ ∑ h, A.nword h k := Finset.sum_nonneg fun h _ => hw h k
  exact le_antisymm (by linarith [hx k]) (hf k)

theorem nfill_eq_zero_of_empty {K T L U : Type} [Fintype T] [Fintype L] [Fintype U] (A : Anatomy K T L U)
    (hA : A.Split) (h0 : ∀ k, A.run.n k = 0) (k : K) : A.nfill k = 0 := by
  obtain ⟨hw, hf, hx, -, hs⟩ := hA
  have h1 := hs k
  rw [h0 k] at h1
  have h2 : 0 ≤ ∑ h, A.nword h k := Finset.sum_nonneg fun h _ => hw h k
  exact le_antisymm (by linarith [hf k]) (hx k)

theorem price_of_zero {K R : Type} [Fintype K] [Fintype R] (G : Machine K R) (lam : R → ℝ) (n : K → ℝ)
    (h0 : ∀ k, n k = 0) : G.price lam n = 0 := by
  simp [Machine.price, Machine.load, h0]

theorem not_foldOpt_of_noFold {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (h0 : ∀ k, A.nfold k = 0)
    (lam : R → ℝ) (h : IsCert lam) (hpos : 0 < (1 - Z.muF) * Z.Nw * Z.phiStar lam) : ¬ FoldOpt Z A := by
  intro hF
  have h1 := hF lam h
  rw [price_of_zero Z.G lam A.nfold h0] at h1
  linarith

theorem not_fillOpt_of_noFill {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (h0 : ∀ k, A.nfill k = 0)
    (lam : R → ℝ) (h : IsCert lam) (hpos : 0 < (1 - Z.muX) * Z.chiStar lam) : ¬ FillOpt Z A := by
  intro hX
  have h1 := hX lam h
  rw [price_of_zero Z.G lam A.nfill h0] at h1
  linarith

theorem not_foldOpt_of_empty {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (hA : A.Split)
    (h0 : ∀ k, A.run.n k = 0) (lam : R → ℝ) (h : IsCert lam) (hpos : 0 < (1 - Z.muF) * Z.Nw * Z.phiStar lam) :
    ¬ FoldOpt Z A :=
  not_foldOpt_of_noFold Z A (nfold_eq_zero_of_empty A hA h0) lam h hpos

theorem not_fillOpt_of_empty {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (hA : A.Split)
    (h0 : ∀ k, A.run.n k = 0) (lam : R → ℝ) (h : IsCert lam) (hpos : 0 < (1 - Z.muX) * Z.chiStar lam) :
    ¬ FillOpt Z A :=
  not_fillOpt_of_noFill Z A (nfill_eq_zero_of_empty A hA h0) lam h hpos

/-- One class, one facet, every row 1, one fold route and one filler route of demand 1, `N_w = 1`, no slack. -/
noncomputable def toy : Model Unit Unit Empty Empty Empty Unit Unit where
  G := { a := fun _ _ => 1, smClocked := fun _ => true, fmax := 1 }
  cls := fun e => e.elim
  omega := fun e => e.elim
  vec := ∅
  bytes := fun e => e.elim
  q := 0
  qT := 0
  bT := 0
  side := fun e => e.elim
  fold := fun _ _ => 1
  fill := fun _ _ => 1
  Nw := 1
  muF := 0
  muX := 0

/-- A run that stopped before its first instruction, labelled with nothing. -/
def aborted : Anatomy Unit Empty Empty Empty where
  run := { n := fun _ => 0, cycles := 0, clock := 1 }
  nword := fun _ _ => 0
  nfold := fun _ => 0
  nfill := fun _ => 0
  words := fun _ => 0
  fixed := 0

theorem aborted_split : aborted.Split := by
  unfold Anatomy.Split
  refine ⟨fun _ _ => le_rfl, fun _ => le_rfl, fun _ => le_rfl, fun _ => le_rfl, fun _ => ?_⟩
  simp [aborted]

theorem one_cert : IsCert (fun _ : Unit => (1 : ℝ)) := by
  unfold IsCert
  exact ⟨fun _ => zero_le_one, by simp⟩

theorem toy_phiStar : toy.phiStar (fun _ => 1) = 1 := by
  first
    | (simp [Model.phiStar, demandPrice, toy]; done)
    | (simp [Model.phiStar, demandPrice, toy, ciInf_unique]; done)

theorem toy_chiStar : toy.chiStar (fun _ => 1) = 1 := by
  first
    | (simp [Model.chiStar, demandPrice, toy]; done)
    | (simp [Model.chiStar, demandPrice, toy, ciInf_unique]; done)

/-- **FOLD-OPT is false on the aborted path** of the toy model, at its one certificate. -/
theorem toy_not_foldOpt : ¬ FoldOpt toy aborted := by
  refine not_foldOpt_of_empty toy aborted aborted_split (fun _ => rfl) (fun _ => 1) one_cert ?_
  rw [toy_phiStar]
  show (0 : ℝ) < (1 - 0) * 1 * 1
  norm_num

/-- **FILL-OPT is false on the aborted path** of the toy model, at its one certificate. -/
theorem toy_not_fillOpt : ¬ FillOpt toy aborted := by
  refine not_fillOpt_of_empty toy aborted aborted_split (fun _ => rfl) (fun _ => 1) one_cert ?_
  rw [toy_chiStar]
  show (0 : ℝ) < (1 - 0) * 1
  norm_num

/-! ## D-ideal (2.7♭) with its per-path premises read on accepted paths only -/

/-- **`DIdealFloor`, guarded**: each per-path premise only on accepted paths, by `DIdealFloor`'s own proof (the call
sites read `ha : P.accepts x`, which the proof already has). -/
theorem DIdealFloorGuarded {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (Fl : Labelled K) (Ω : Type) [Fintype Ω] (S : ℕ) (P : Play Ω S K T L U)
    (lam : R → ℝ) (p δ δbind δzk : ℝ) (nπ : ℕ)
    (h : IsCert lam) (hZ : Z.Wf Fl) (hS : 0 < S) (hp0 : 0 < p) (hp1 : p < 1)
    (hδ : 0 < δ - S * (δbind + nπ * δzk))
    (hSplit : ∀ x, P.accepts x → (P.run x).Split)
    (hM : ∀ x, P.accepts x → FacetCeilings Z.G (P.run x).run)
    (hW : ∀ x, P.accepts x → WordsWritten Z (P.run x))
    (hSR : ∀ x, P.accepts x → ScalarRoute Z (P.run x))
    (hU : ∀ x, P.accepts x → SideUnits Z (P.run x))
    (hN : ∀ x, P.accepts x → NoSum (P.run x))
    (hF : ∀ x, P.accepts x → FoldOpt Z (P.run x))
    (hX : ∀ x, P.accepts x → FillOpt Z (P.run x))
    (hL : LuckAt p P.game) (hPS : ProofsSound P Z.Nw δbind δzk nπ) :
    prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
        Z.epsJl Fl lam p (δ - S * (δbind + nπ * δzk)) S) ≤ δ := by
  set δ' := δ - S * (δbind + nπ * δzk) with hδ'
  obtain ⟨E, hE, hacc⟩ := hPS
  set t := Real.log (1 / δ') / Real.log (1 / p) with ht
  have hB := (luck_rounds P.game p hp0.le hL ⌈t⌉₊).trans (pow_ceil_le hp0 hp1 hδ)
  calc prob (fun x => P.accepts x ∧ (P.run x).run.time / Z.credit Fl <
        1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam -
          Z.epsJl Fl lam p δ' S)
      ≤ prob (fun x => E x ∨ (P.game.accepts x ∧ ⌈t⌉₊ ≤ Accounting.TS.RoundGame.deficientRounds P.game x)) := by
        refine prob_mono ?_
        rintro x ⟨ha, hlt⟩
        by_contra hne
        push Not at hne
        have hfix := fixed_of_tail P Z.Nw t hS hZ.2.2.1 E hacc x ha hne.1 fun hg =>
          (deficiency_le_rounds P.game p hL x).trans_lt (Nat.lt_ceil.mp (hne.2 hg))
        have key := floor_of_price Z Fl lam (t / S) 0 (P.run x).run h hZ (hM x ha)
          (phi_le_of_fixed Z lam (t / S) 0 _ _ (stars_nonneg Z Fl lam hZ h).2.1 hfix
            (price_of_parts Z Fl (P.run x) lam h hZ (hSplit x ha) (hW x ha) (hSR x ha) (hU x ha) (hN x ha)
              (hF x ha) (hX x ha)))
        have e : Z.epsLuck Fl lam (t / S) = Z.epsJl Fl lam p δ' S := by
          rw [Model.epsLuck, Model.epsJl, ht]
          ring
        rw [e, Model.epsAdapt, zero_mul, sub_zero] at key
        exact absurd hlt (not_lt.mpr key)
    _ ≤ prob E + prob (fun x => P.game.accepts x ∧ ⌈t⌉₊ ≤ Accounting.TS.RoundGame.deficientRounds P.game x) :=
        prob_or_le _ _
    _ ≤ S * (δbind + nπ * δzk) + δ' := add_le_add hE hB
    _ = δ := by rw [hδ']; ring

/-- The guarded form is a strengthening: it gives `main`'s `DIdealFloor`. -/
theorem guarded_implies : Accounting.Guarantees.TS.DIdealFloor := by
  intro K R T L U F X _ _ _ _ _ _ _ _ _ _ Z Fl Ω _ S P lam p δ δbind δzk nπ h hZ hS hp0 hp1 hδ hSplit hM
    hW hSR hU hN hF hX hL hPS
  exact DIdealFloorGuarded Z Fl Ω S P lam p δ δbind δzk nπ h hZ hS hp0 hp1 hδ (fun x _ => hSplit x)
    (fun x _ => hM x) (fun x _ => hW x) (fun x _ => hSR x) (fun x _ => hU x) (fun x _ => hN x)
    (fun x _ => hF x) (fun x _ => hX x) hL hPS

/-! ## FanIn's per-lane caps against its per-instruction sum -/

/-- A warp IADD3 as 32 lane nodes, lane `i` reading its three registers `(i, 0)`, `(i, 1)`, `(i, 2)`. -/
def iadd3Warp : Dataflow (Fin 32 × Fin 3) (BitVec 32) where
  n := 32
  src i := Finset.univ.map ⟨fun c => Sum.inl (i, c), fun a b hab => by simpa using hab⟩
  earlier i k hk := by simp at hk
  op i v := v (.inl (i, 0)) + v (.inl (i, 1)) + v (.inl (i, 2))
  reads i v v' hv := by
    have h0 := hv (.inl (i, 0)) (by simp)
    have h1 := hv (.inl (i, 1)) (by simp)
    have h2 := hv (.inl (i, 2)) (by simp)
    simp only [h0, h1, h2]

theorem iadd3Warp_n : iadd3Warp.n = 32 := rfl

theorem iadd3Warp_merges (i : Fin iadd3Warp.n) : iadd3Warp.merges i = 2 := by
  first
    | (simp [Dataflow.merges, iadd3Warp]; done)
    | (simp [Dataflow.merges, iadd3Warp, Finset.card_map]; done)

/-- Issued as one warp instruction, the 32 lanes merge 64 components. -/
theorem iadd3Warp_issued : iadd3Warp.issued (fun _ => ()) () = 64 := by
  first
    | (simp [Dataflow.issued, iadd3Warp_merges, iadd3Warp_n]; done)
    | (simp [Dataflow.issued, iadd3Warp_merges]; done)
    | decide

/-- **`FanIn` at the docstring's per-lane ALU cap is false for a warp IADD3**, read with `j` the warp instruction. -/
theorem fanIn_warp_false {K : Type} [Fintype K] (cls : Unit → K) (m : K → ℝ) (hm : m (cls ()) < 64) :
    ¬ FanIn iadd3Warp (fun _ => ()) cls m := by
  intro hF
  have h1 := hF ()
  rw [iadd3Warp_issued] at h1
  push_cast at h1
  linarith

/-- The ALU cap of 8, a warp IADD3 issued as one instruction. -/
theorem fanIn_warp_alu : ¬ FanIn (K := Unit) iadd3Warp (fun _ => ()) (fun _ => ()) (fun _ => 8) :=
  fanIn_warp_false _ _ (by norm_num)

/-- Read with `j` a lane, the same dataflow meets the cap: 32 instructions `j` where the run counts one. -/
theorem fanIn_lane_true : FanIn (K := Unit) iadd3Warp (fun i => i) (fun _ => ()) (fun _ => 8) := by
  intro j
  have hj : iadd3Warp.issued (fun i => i) j = 2 := by
    simp [Dataflow.issued, iadd3Warp_merges, Finset.filter_eq']
  rw [hj]
  norm_num

/-! ## The data-sheet machine at a missing peak -/

/-- **A class with no sheet peak is free on the tensor row** (`x / 0 = 0`), whatever its MACs. -/
theorem sheet_tensor_free {K P : Type} [DecidableEq P] (S : Sheet K P) (k : K) (h : S.peak k = 0) :
    S.machine.a .tensor k = 0 := by
  simp [Sheet.machine, h]

/-! ## Units: the credit scales with the floor's normalization -/

section Units

variable {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]

/-- **The bound is linear in the counts**: `B(c·n) = c·B(n)` for `c ≥ 0`. -/
theorem bound_smul (G : Machine K R) (n : K → ℝ) {c : ℝ} (hc : 0 ≤ c) :
    G.bound (fun k => c * n k) = c * G.bound n := by
  simp only [Machine.bound, Machine.load]
  rw [Real.mul_iSup_of_nonneg hc]
  congr 1
  funext r
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => by ring

/-- A labelled program with every part scaled by `c`. -/
def scaled (c : ℝ) (P : Labelled K) : Labelled K :=
  ⟨fun k => c * P.nw k, fun k => c * P.nf k, fun k => c * P.nx k, fun k => c * P.ns k⟩

/-- **The credit scales with the floor**: `T_cred(c·♭) = c·T_cred(♭)`. A floor summed over a circuit's words, where
the TS model divides by the SMSPs, is `N_SMSP` times the credit: 752 on sm_120 (RT-LOGIC-8). -/
theorem credit_scaled (Z : Model K R T L U F X) (P : Labelled K) {c : ℝ} (hc : 0 ≤ c) :
    Z.credit (scaled c P) = c * Z.credit P := by
  have e : (scaled c P).n = fun k => c * P.n k := by
    funext k
    simp only [Labelled.n, scaled]
    ring
  unfold Model.credit Model.bnd
  rw [e, bound_smul Z.G P.n hc, mul_div_assoc]

end Units

/-- **A2 per SMSP against a circuit's totals**: 752 SMSPs each issuing a warp instruction every cycle for `C > 0`
cycles meet A2 one by one, but their total `752·C` is above the `C` cycles each SMSP counts. A run whose counts are
the circuit's totals (Compose's `Fits`) meets A2 only with its cycles summed over the SMSPs (RT-LOGIC-8). -/
theorem dispatch_total {C : ℝ} (hC : 0 < C) :
    SheetDispatch ({ n := fun _ : Unit => C, cycles := C, clock := 1 } : Run Unit) ∧
      ¬ SheetDispatch ({ n := fun _ : Unit => 752 * C, cycles := C, clock := 1 } : Run Unit) ∧
      SheetDispatch ({ n := fun _ : Unit => 752 * C, cycles := 752 * C, clock := 1 } : Run Unit) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [SheetDispatch]
  · simp only [SheetDispatch, Fintype.univ_unit, Finset.sum_singleton, not_le]
    linarith
  · simp [SheetDispatch]

#print axioms bound_smul
#print axioms credit_scaled
#print axioms dispatch_total
#print axioms toy_not_foldOpt
#print axioms toy_not_fillOpt
#print axioms DIdealFloorGuarded
#print axioms guarded_implies
#print axioms fanIn_warp_alu
#print axioms fanIn_lane_true
#print axioms sheet_tensor_free

end Accounting.SecurityProofs.TS.RtLogic
