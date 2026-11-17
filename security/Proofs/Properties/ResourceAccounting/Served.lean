module

public import Properties.ResourceAccounting.Served
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-! Proofs of the served tie's statements, each with witnesses that meet it and refusals that fail it. -/

@[expose] public section

namespace Properties.Served

open ResourceAccounting.Compose ResourceAccounting.Served

theorem llama31Shape : Llama31Shape := by
  unfold Llama31Shape
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, ?_⟩
  unfold Shape.Wf
  decide +kernel

theorem stepOf_lt {L : Shape} {c : ℕ} (hc : c < L.steps * L.calls) : L.stepOf c < L.steps := by
  unfold Shape.stepOf
  exact Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hc)

theorem roundOf_lt {L : Shape} (F : Form) {c : ℕ} (hc : c < L.steps * L.calls) : F.roundOf L c < F.rounds L := by
  cases F <;> first | exact stepOf_lt hc | exact hc

theorem commitCoin_outZ {F : Form} (hF : F ≠ .perCall) (r : ℕ) : commitCoin F .outZ r = r := by
  cases F <;> first | rfl | exact (hF rfl).elim

theorem commitCoin_outY {F : Form} (hF : F ≠ .perCall) (r : ℕ) : commitCoin F .outY r = r := by
  cases F <;> first | rfl | exact (hF rfl).elim

theorem round_lt {ι : Type} {L : Shape} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls} (T : Tie L F Cd cls) (i : ℕ)
    (hi : i < Cd.dag.length) : Cd.round (Cd.call i) < Cd.rounds := by
  have hc := T.inRange i hi
  have hc' : Cd.call i < L.steps * L.calls := by rw [← T.calls]; exact hc
  rw [T.round _ hc, T.rounds]
  exact roundOf_lt F hc'

theorem firstRead_cases {ι : Type} (Cd : Declared ι) (i : ℕ) :
    Cd.firstRead i = Cd.rounds ∨
      ∃ j, j < Cd.dag.length ∧ Cd.Reads j i ∧ Cd.firstRead i = Cd.round (Cd.call j) := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact Or.inl rfl
  | some j =>
    have hp := List.find?_some h
    exact Or.inr ⟨j, List.mem_range.1 (List.mem_of_find?_eq_some h), hp, rfl⟩

theorem firstRead_of_reads {ι : Type} (Cd : Declared ι) {i j : ℕ} (hj : j < Cd.dag.length) (hR : Cd.Reads j i) :
    ∃ j', j' < Cd.dag.length ∧ Cd.Reads j' i ∧ Cd.firstRead i = Cd.round (Cd.call j') := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact absurd hR (List.find?_eq_none.1 h j (List.mem_range.2 hj))
  | some j' =>
    have hp := List.find?_some h
    exact ⟨j', List.mem_range.1 (List.mem_of_find?_eq_some h), hp, rfl⟩

theorem round_le_cut_of_run {ι : Type} {L : Shape} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls}
    (T : Tie L F Cd cls) {ℓ : ℕ} (hlt : ℓ < Cd.dag.length) (hb : Cd.binding ℓ = some .run)
    (hr : ∀ j, j < Cd.dag.length → Cd.Reads j ℓ → Cd.call j = Cd.call ℓ) :
    Cd.round (Cd.call ℓ) ≤ Cd.cutOf ℓ := by
  simp only [Declared.cutOf, hb]
  rcases firstRead_cases Cd ℓ with h | ⟨j, hj, hR, h⟩
  · rw [h]
    exact (round_lt T ℓ hlt).le
  · rw [h]
    exact le_of_eq (congrArg Cd.round (hr j hj hR)).symm

theorem round_le_cut_of_none {ι : Type} {Cd : Declared ι} {ℓ : ℕ} (hb : Cd.binding ℓ = none) :
    Cd.round (Cd.call ℓ) ≤ Cd.cutOf ℓ := by
  simp only [Declared.cutOf, hb]
  exact le_max_left _ _

theorem reads_lt {ι : Type} {Cd : Declared ι} {i j : ℕ} (h : Cd.Reads j i) : j < Cd.dag.length := by
  by_contra hge
  have hnone : Cd.dag[j]? = none := List.getElem?_eq_none (by omega)
  unfold Declared.Reads at h
  rw [hnone] at h
  simp at h

theorem tieCommits : TieCommits := by
  intro ι L F Cd cls T ℓ hW
  obtain ⟨hlt, hnc⟩ := hW
  have hround := T.round _ (T.inRange ℓ hlt)
  cases hk : cls ℓ with
  | coin => exact absurd ((T.coin ℓ hlt).1 hk) hnc
  | inner => exact Or.inl (le_of_eq (T.inner ℓ hlt hk))
  | registered =>
    left
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]), hk]
    simp [commitCoin]
  | act =>
    left
    obtain ⟨hb, hr⟩ := T.act ℓ hlt (Or.inl hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_run T hlt hb hr
  | pub =>
    left
    obtain ⟨hb, -, hle, hr⟩ := T.pub ℓ hlt hk
    simp only [Declared.cutOf, hb]
    rcases firstRead_cases Cd ℓ with h | ⟨j, hj, hR, h⟩
    · rw [h]
      exact hle
    · rw [h]
      exact hr j hj hR
  | served =>
    left
    obtain ⟨hb, hr⟩ := T.act ℓ hlt (Or.inr hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_run T hlt hb hr
  | outZ =>
    left
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]), hk, ← hround]
    rcases T.outWords ℓ hlt (Or.inl hk) with hb | ⟨hF, hb, hr⟩
    · by_cases hF : F = .perCall
      · subst hF
        have hn := T.noised ℓ hlt (Or.inr ⟨Or.inl hk, rfl⟩)
        show Cd.round (Cd.call ℓ) + 1 ≤ Cd.cutOf ℓ
        simp only [Declared.cutOf, hb]
        exact le_max_of_le_right hn
      · rw [commitCoin_outZ hF]
        exact round_le_cut_of_none hb
    · subst hF
      rw [commitCoin_outZ (F := .zTree) (by decide)]
      exact round_le_cut_of_run T hlt hb fun j hj hR => (hr j hj hR).1
  | outY =>
    left
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]) (by simp [hk]), hk, ← hround]
    rcases T.outWords ℓ hlt (Or.inr hk) with hb | ⟨hF, hb, hr⟩
    · by_cases hF : F = .perCall
      · subst hF
        have hn := T.noised ℓ hlt (Or.inr ⟨Or.inr hk, rfl⟩)
        show Cd.round (Cd.call ℓ) + 1 ≤ Cd.cutOf ℓ
        simp only [Declared.cutOf, hb]
        exact le_max_of_le_right hn
      · rw [commitCoin_outY hF]
        exact round_le_cut_of_none hb
    · subst hF
      rw [commitCoin_outY (F := .zTree) (by decide)]
      exact round_le_cut_of_run T hlt hb fun j hj hR => (hr j hj hR).1
  | receipt =>
    exact Or.inr ⟨T.computed ℓ hlt (by simp [hk]), (T.receipt ℓ hlt hk).2,
      fun j hR => T.readers ℓ hlt (Or.inl hk) j (reads_lt hR) hR⟩
  | check =>
    exact Or.inr ⟨T.computed ℓ hlt (by simp [hk]), T.checked ℓ hlt hk,
      fun j hR => T.readers ℓ hlt (Or.inr hk) j (reads_lt hR) hR⟩

theorem tieLive : TieLive := by
  intro ι K Sc L F Cd cls T u _ hw
  obtain ⟨i, hi, _⟩ := List.any_eq_true.1 hw
  obtain ⟨hiw, hiu⟩ := List.mem_filter.1 hi
  obtain ⟨hir, _⟩ := List.mem_filter.1 hiw
  have hmem : i ∈ unitGates Cd u := List.mem_filter.2 ⟨hir, hiu⟩
  unfold Declared.roundOf
  cases hl : unitGates Cd u with
  | nil =>
    rw [hl] at hmem
    simp at hmem
  | cons i0 rest =>
    have h0 : i0 ∈ unitGates Cd u := by rw [hl]; simp
    obtain ⟨h0r, _⟩ := List.mem_filter.1 h0
    show Cd.coinMode (Cd.round (Cd.call i0)) = .live
    exact T.live _ (round_lt T i0 (List.mem_range.1 h0r))

theorem tieScope : TieScope := by
  intro ι K Sc L F Cd cls δs T hWf hδ hL
  unfold Scope
  exact ⟨hWf, tieLive ι K Sc L F Cd cls T, hδ, hL, tieCommits ι L F Cd cls T⟩

theorem epochScope {ι Q R S K : Type} {L : Shape} {F : Form} {W : ℕ} {Cd : ℕ → Declared ι} {cls : ℕ → ℕ → Cls}
    {release : ℝ} {run : ℕ → Pouw.TimedRun Q R S} (E : EpochTie L F W Cd cls release run) (Sc : Scheme K) (δs : ℝ)
    {w : ℕ} (hw : w < W) (hWf : (Cd w).Wf) (hδ : 0 < δs) (hL : Sc.LuckProved) : Scope Sc (Cd w) δs :=
  tieScope ι K Sc L F (Cd w) (cls w) δs (E.tie w hw) hWf hδ hL

theorem spentIn_split {Q R S : Type} (run : Pouw.TimedRun Q R S) (H : Q → R) (s : S) (a b c : ℝ) :
    run.spentIn H s a c = run.spentIn H s a b + run.spentIn H s b c := by
  unfold Pouw.TimedRun.spentIn
  exact (sub_add_sub_cancel' _ _ _).symm

theorem llama31Rounds : Llama31Rounds := by
  unfold Llama31Rounds
  refine ⟨fun F hF => ?_, by decide +kernel⟩
  cases F <;> first | exact (hF rfl).elim | decide +kernel

theorem seededScheduleNotLive : SeededScheduleNotLive := by
  intro ι Cd Np Kd p query l hl hU hWf ⟨ℓ, j, hb, hj, hR⟩ hLC
  obtain ⟨_, _, _, hd, _, _⟩ := hU l hl
  obtain ⟨hW1, hW2, _⟩ := hWf
  have hlt : ℓ < Cd.dag.length := by
    by_contra hge
    have hnone : Cd.dag[ℓ]? = none := List.getElem?_eq_none (by omega)
    simp [Declared.binding, hnone] at hb
  have hW : Cd.Writes ℓ := ⟨hlt, by rw [hb]; simp⟩
  have h1 : Cd.commitRound ℓ = Cd.rounds := by simpa [hb] using hd ℓ hW
  have h3 : Cd.cutOf ℓ = Cd.firstRead ℓ := by simp [Declared.cutOf, hb]
  obtain ⟨j', hj', _, h⟩ := firstRead_of_reads Cd hj hR
  have hr := hW2 _ (hW1 j' hj').1
  rcases hLC ℓ hW with h2 | ⟨hn, -⟩
  · omega
  · rw [hb] at hn
    cases hn

/-- `Tie.commit` from its cases: `decide` finds no `Decidable` instance for its five premises within the default
`synthInstance.maxSize`, and finds one for three premises and a disjunction. -/
theorem tie_commit {ι : Type} {L : Shape} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls}
    (h : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin → cls i ≠ .receipt →
      cls i = .check ∨ cls i = .pub ∨ Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i))) :
    ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin → cls i ≠ .receipt → cls i ≠ .check → cls i ≠ .pub →
      Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i)) :=
  fun i hi h1 h2 h3 h4 h5 => ((h i hi h1 h2 h3).resolve_left h4).resolve_left h5

def witShape : Shape where
  steps := 1
  rows := 4
  tileRows := 4
  tileCols := 8
  linears := [(32, 8)]

def witGate (op : ResourceAccounting.Gates.IntOp Unit) (args : List ℕ) (commit : Bool) : ResourceAccounting.Gates.IntGate Unit :=
  { op := op, args := args, lo := 0, hi := 0, step := 0, prim := "", place := .other, commit := commit }

def witCircuit : Declared Unit where
  dag := [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.affine 0 [1, 1]) [0, 1] true]
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if i = 2 then 1 else 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

def witCls : ℕ → Cls
  | 0 => .coin
  | 1 => .act
  | _ => .receipt

theorem witTie : Tie witShape .beside witCircuit witCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < 3 → (witCls i = .act ∨ witCls i = .served) →
        witCircuit.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => trivial
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem wit_receipt_cut : witCircuit.Writes 2 ∧ witCircuit.commitRound 2 = 1 ∧ witCircuit.cutOf 2 = 1 :=
  ⟨⟨by decide +kernel, by decide +kernel⟩, by decide +kernel, by decide +kernel⟩

theorem wit_layerCommitted : LayerCommitted witCircuit :=
  tieCommits Unit witShape .beside witCircuit witCls witTie

def witCircuitCall : Declared Unit where
  dag := [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.affine 0 [1, 1]) [0, 1] true, witGate (.affine 0 [1]) [2] true]
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if i = 2 ∨ i = 3 then 1 else 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

def witClsCall : ℕ → Cls
  | 0 => .coin
  | 1 => .act
  | 2 => .outY
  | _ => .receipt

theorem witTieCall : Tie witShape .perCall witCircuitCall witClsCall where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < 4 → (witClsCall i = .act ∨ witClsCall i = .served) →
        witCircuitCall.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨2, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem witCall_out_cut :
    witCircuitCall.Writes 2 ∧ witCircuitCall.commitRound 2 = 1 ∧ witCircuitCall.cutOf 2 = 1 :=
  ⟨⟨by decide +kernel, by decide +kernel⟩, by decide +kernel, by decide +kernel⟩

theorem witCall_layerCommitted : LayerCommitted witCircuitCall :=
  tieCommits Unit witShape .perCall witCircuitCall witClsCall witTieCall

def witCircuitUnnoised : Declared Unit :=
  { witCircuitCall with
    dag := [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
      witGate (.affine 0 [1]) [1] true, witGate (.affine 0 [1]) [2] true] }

/-- **`perCall` needs the served `y` to read its round's coin** (G3's P7): a `y` that reads none fails `Tie.noised`,
whatever its commit coins. -/
theorem witCall_unnoised : ¬ Tie witShape .perCall witCircuitUnnoised witClsCall := fun T =>
  absurd (T.noised 2 (by decide +kernel) (Or.inr ⟨Or.inr rfl, rfl⟩)) (by decide +kernel)

def witRunAt (t : ℝ) : Pouw.TimedRun Unit Unit Unit := ⟨t, fun _ _ => 0, fun _ _ _ => 0⟩

theorem witEpoch : EpochTie witShape .beside 2 (fun _ => witCircuit) (fun _ => witCls) 0 (fun _ => witRunAt 0) :=
  ⟨fun _ _ => witTie, fun _ _ => rfl⟩

/-- The same two windows with the second timed from its own start `1` are not. -/
theorem witEpoch_ownStart : ¬ EpochTie witShape .beside 2 (fun _ => witCircuit) (fun _ => witCls) 0
    (fun w => witRunAt (if w = 0 then 0 else 1)) := by
  intro E
  have h := E.salted 1 (by decide)
  exact one_ne_zero h

/-- The `beside` witness's unit has no `outY` word, so its receipt has no value-checked product: it fails
`ValueChecked` whatever the dequant. -/
theorem wit_notValueChecked (dq : List ℤ → ℤ) : ¬ ValueChecked witShape dq witCircuit witCls := by
  rintro ⟨_, _, -, -, -, h, -⟩
  obtain ⟨y, hy, hyo, -⟩ := h 2 (by decide +kernel) rfl
  exact (by decide +kernel : ∀ y, y < witCircuit.dag.length → witCls y ≠ .outY) y hy hyo

section Values

open ResourceAccounting.Gates

theorem foldl_values_length {ι : Type} (C : IntDag ι) (x : ι → ℤ) (vs : List ℤ) :
    (C.foldl (fun vs g => vs ++ [g.value x vs]) vs).length = vs.length + C.length := by
  induction C generalizing vs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

theorem values_length {ι : Type} (C : IntDag ι) (x : ι → ℤ) : (values C x).length = C.length := by
  simp [values, foldl_values_length]

theorem values_take_succ {ι : Type} (C : IntDag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    values (C.take (i + 1)) x = values (C.take i) x ++ [C[i].value x (values (C.take i) x)] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [values, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem values_take_length {ι : Type} (C : IntDag ι) (x : ι → ℤ) (n : ℕ) (h : n ≤ C.length) :
    (values (C.take n) x).length = n := by
  rw [values_length, List.length_take]; omega

theorem getD_values_take {ι : Type} (C : IntDag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    ∀ n, i < n → n ≤ C.length → (values (C.take n) x).getD i 0 = C[i].value x (values (C.take i) x) := by
  intro n hin hn
  induction n with
  | zero => omega
  | succ n ih =>
    rw [values_take_succ C x n (by omega)]
    rcases Nat.lt_or_ge i n with hi | hi
    · rw [List.getD_append _ _ _ _ (by rw [values_take_length C x n (by omega)]; exact hi)]
      exact ih hi (by omega)
    · have : i = n := by omega
      subst this
      rw [List.getD_append_right _ _ _ _ (by rw [values_take_length C x i (by omega)]),
        values_take_length C x i (by omega)]
      simp

theorem val_eq {ι : Type} (C : IntDag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    val C x i = C[i].value x (values (C.take i) x) := by
  have := getD_values_take C x i h C.length h le_rfl
  rwa [List.take_length] at this

theorem val_of_getElem? {ι : Type} (C : IntDag ι) (x : ι → ℤ) (i : ℕ) {g : IntGate ι} (hg : C[i]? = some g)
    (ha : ∀ j ∈ g.args, j < i) : val C x i = g.op.out (g.op.raw x (g.args.map (val C x))) := by
  obtain ⟨h, hgi⟩ := List.getElem?_eq_some_iff.1 hg
  have hm : g.args.map (fun j => (values (C.take i) x).getD j 0) = g.args.map (val C x) :=
    List.map_congr_left fun j hj => by
      have hji := ha j hj
      show (values (C.take i) x).getD j 0 = val C x j
      rw [getD_values_take C x j (by omega) i hji h.le, ← val_eq C x j (by omega)]
  rw [val_eq C x i h, hgi]
  unfold IntGate.value IntGate.raw IntGate.operands
  rw [hm]

theorem val_input {ι : Type} (C : IntDag ι) (x : ι → ℤ) {i : ℕ} {g : IntGate ι} {b : Binding} {s : ι}
    (hg : C[i]? = some g) (hop : g.op = .input b s) : val C x i = x s := by
  obtain ⟨h, hgi⟩ := List.getElem?_eq_some_iff.1 hg
  rw [val_eq C x i h, hgi]
  unfold IntGate.value IntGate.raw
  rw [hop]
  rfl

end Values

theorem sum_zipWith_bound {P Q : ℤ} (hP : 0 ≤ P) (hQ : 0 ≤ Q) : ∀ (l1 l2 : List ℤ),
    (∀ a ∈ l1, -P ≤ a ∧ a ≤ P) → (∀ b ∈ l2, -Q ≤ b ∧ b ≤ Q) →
      -(P * Q * l1.length) ≤ (List.zipWith (· * ·) l1 l2).sum ∧ (List.zipWith (· * ·) l1 l2).sum ≤ P * Q * l1.length
  | [], _, _, _ => by simp
  | _ :: l, [], _, _ => by
    simp only [List.zipWith_nil_right, List.sum_nil, List.length_cons]
    have h0 : 0 ≤ P * Q * ((l.length + 1 : ℕ) : ℤ) := mul_nonneg (mul_nonneg hP hQ) (by positivity)
    constructor <;> linarith
  | a :: l1, b :: l2, h1, h2 => by
    obtain ⟨ha1, ha2⟩ := h1 a (by simp)
    obtain ⟨hb1, hb2⟩ := h2 b (by simp)
    obtain ⟨ih1, ih2⟩ :=
      sum_zipWith_bound hP hQ l1 l2 (fun c hc => h1 c (by simp [hc])) (fun c hc => h2 c (by simp [hc]))
    have hab1 : a * b ≤ P * Q := by
      nlinarith [mul_nonneg (by linarith : (0 : ℤ) ≤ P - a) (by linarith : (0 : ℤ) ≤ Q + b),
        mul_nonneg (by linarith : (0 : ℤ) ≤ P + a) (by linarith : (0 : ℤ) ≤ Q - b)]
    have hab2 : -(P * Q) ≤ a * b := by
      nlinarith [mul_nonneg (by linarith : (0 : ℤ) ≤ P - a) (by linarith : (0 : ℤ) ≤ Q - b),
        mul_nonneg (by linarith : (0 : ℤ) ≤ P + a) (by linarith : (0 : ℤ) ≤ Q + b)]
    simp only [List.zipWith_cons_cons, List.sum_cons, List.length_cons]
    push_cast
    constructor <;> linarith

theorem clampS32_of_bound {s : ℤ} (h1 : -2147483648 ≤ s) (h2 : s ≤ 2147483647) : ResourceAccounting.Gates.clampS32 s = s := by
  have hmin : ResourceAccounting.Gates.s32Min ≤ s := le_trans (by norm_num [ResourceAccounting.Gates.s32Min]) h1
  have hmax : s ≤ ResourceAccounting.Gates.s32Max := le_trans h2 (by norm_num [ResourceAccounting.Gates.s32Max])
  unfold ResourceAccounting.Gates.clampS32
  rw [min_eq_right hmax, max_eq_right hmin]

theorem sum_zipWith_one : ∀ (n : ℕ) (l : List ℤ), l.length = n →
    (List.zipWith (· * ·) (List.replicate n 1) l).sum = l.sum
  | 0, [], _ => rfl
  | n + 1, b :: l, h => by
    simp only [List.replicate_succ, List.zipWith_cons_cons, List.sum_cons, one_mul]
    rw [sum_zipWith_one n l (by simpa using h)]
  | 0, _ :: _, h => by simp at h
  | _ + 1, [], h => by simp at h

theorem sum_zipWith_add (c : ℤ) : ∀ (l1 l2 : List ℤ), l1.length = l2.length →
    (List.zipWith (· * ·) (l1.map (· + c)) l2).sum = (List.zipWith (· * ·) l1 l2).sum + c * l2.sum
  | [], [], _ => by simp
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: l1, b :: l2, h => by
    simp only [List.map_cons, List.zipWith_cons_cons, List.sum_cons]
    rw [sum_zipWith_add c l1 l2 (by simpa using h)]
    ring

theorem affine_two (c₁ c₂ a b : ℤ) : (0 : ℤ) + (List.zipWith (· * ·) [c₁, c₂] [a, b]).sum = c₁ * a + c₂ * b := by
  simp only [List.zipWith_cons_cons, List.zipWith_nil_left, List.sum_cons, List.sum_nil]
  ring

def witGateR (op : ResourceAccounting.Gates.IntOp ℕ) (args : List ℕ) (lo hi : ℤ) (commit : Bool) : ResourceAccounting.Gates.IntGate ℕ :=
  { op := op, args := args, lo := lo, hi := hi, step := 0, prim := "", place := .other, commit := commit }

def witDq (v : List ℤ) : ℤ := v.headD 0 * v.getD 1 0 * v.getD 2 0

def witPre : ResourceAccounting.Gates.IntDag ℕ :=
  [witGateR (.input .coin 0) [] 0 255 false] ++
    (List.range' 1 32).map (fun j => witGateR (.input .run j) [] (-128) 127 false) ++
    (List.range' 33 32).map (fun j => witGateR (.input .registered j) [] (-128) 127 false) ++
    [witGateR (.input .run 65) [] 0 65535 false, witGateR (.input .registered 66) [] 0 65535 false,
      witGateR (.const 0) [] 0 0 false]

theorem witPre_get (rest : ResourceAccounting.Gates.IntDag ℕ) {j : ℕ} (h : j < 68) : (witPre ++ rest)[j]? = witPre[j]? :=
  List.getElem?_append_left (lt_of_lt_of_eq h (by decide +kernel))

theorem witPre_input (rest : ResourceAccounting.Gates.IntDag ℕ) (x : ℕ → ℤ) {j : ℕ} {b : ResourceAccounting.Gates.Binding} {lo hi : ℤ}
    (hg : (witPre ++ rest)[j]? = some (witGateR (.input b j) [] lo hi false))
    (hx : ResourceAccounting.Gates.Typed (witPre ++ rest) x) :
    ResourceAccounting.Gates.val (witPre ++ rest) x j = x j ∧ lo ≤ x j ∧ x j ≤ hi := by
  refine ⟨?_, hx _ (List.mem_of_getElem? hg) b j rfl⟩
  rw [val_of_getElem? _ x j hg (by simp [witGateR])]
  rfl

theorem witPre_valA (rest : ResourceAccounting.Gates.IntDag ℕ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witPre ++ rest) x)
    {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    ResourceAccounting.Gates.val (witPre ++ rest) x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_input (b := .run) rest x (by rw [witPre_get rest (by omega)]; interval_cases j <;> rfl) hx

theorem witPre_valB (rest : ResourceAccounting.Gates.IntDag ℕ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witPre ++ rest) x)
    {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    ResourceAccounting.Gates.val (witPre ++ rest) x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_input (b := .registered) rest x (by rw [witPre_get rest (by omega)]; interval_cases j <;> rfl) hx

theorem witPre_val67 (rest : ResourceAccounting.Gates.IntDag ℕ) (x : ℕ → ℤ) : ResourceAccounting.Gates.val (witPre ++ rest) x 67 = 0 := by
  rw [val_of_getElem? _ x 67 (g := witGateR (.const 0) [] 0 0 false) (by rw [witPre_get rest (by omega)]; rfl)
    (by simp [witGateR])]
  rfl

theorem witPre_dot (rest : ResourceAccounting.Gates.IntDag ℕ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witPre ++ rest) x)
    {i h : ℕ} {bs : List ℕ} {c : Bool}
    (hg : (witPre ++ rest)[i]? = some (witGateR (.mma .s8 .s8 32 true) (h :: (List.range' 1 32 ++ bs))
      ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max c))
    (ha : ∀ j ∈ h :: (List.range' 1 32 ++ bs), j < i) (hh : ResourceAccounting.Gates.val (witPre ++ rest) x h = 0)
    (hbs : ∀ b ∈ bs, 33 ≤ b ∧ b < 65) :
    ResourceAccounting.Gates.val (witPre ++ rest) x i =
      (List.zipWith (· * ·) ((List.range' 1 32).map (ResourceAccounting.Gates.val (witPre ++ rest) x))
        (bs.map (ResourceAccounting.Gates.val (witPre ++ rest) x))).sum := by
  have hA : ∀ a ∈ (List.range' 1 32).map (ResourceAccounting.Gates.val (witPre ++ rest) x), -128 ≤ a ∧ a ≤ 128 := by
    intro a ha'
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha'
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witPre_valA rest x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  have hB : ∀ b ∈ bs.map (ResourceAccounting.Gates.val (witPre ++ rest) x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := hbs j hj
    obtain ⟨he, hlo, hhi⟩ := witPre_valB rest x hx hj1 hj2
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 128) (Q := 128) (by norm_num) (by norm_num) _ _ hA hB
  simp only [List.length_map, List.length_range'] at hb1 hb2
  rw [val_of_getElem? _ x i hg ha]
  simp only [witGateR, ResourceAccounting.Gates.IntOp.out, ResourceAccounting.Gates.IntOp.raw, List.map_cons, List.headD_cons,
    List.tail_cons, hh, zero_add, List.map_append]
  unfold ResourceAccounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

def witZGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 1 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
    ResourceAccounting.Gates.s32Max true

def witYGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [68, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

def witZRest : ResourceAccounting.Gates.IntDag ℕ :=
  [witZGate, witYGate] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
        ResourceAccounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [102] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true]

def witZ : Declared ℕ where
  dag := witPre ++ witZRest
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if 70 ≤ i then 1 else 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

def witZCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 then .outZ
  else if i = 69 then .outY
  else if i = 103 then .receipt
  else .inner

theorem witTieZ : Tie witShape .zTree witZ witZCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witZ.dag.length → (witZCls i = .act ∨ witZCls i = .served) →
        witZ.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨68, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem witZ_layerCommitted : LayerCommitted witZ :=
  tieCommits ℕ witShape .zTree witZ witZCls witTieZ

theorem witZ_beside : (coinMark witZ).getD 68 0 = 0 ∧ (coinMark witZ).getD 103 0 = 1 := by
  decide +kernel

theorem witZ_valA (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed witZ.dag x) {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    ResourceAccounting.Gates.val witZ.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valA witZRest x hx h1 h2

theorem witZ_valB (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed witZ.dag x) {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    ResourceAccounting.Gates.val witZ.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valB witZRest x hx h1 h2

theorem witZ_productOf : ProductOf witShape witZ witZCls 68 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx _ => ?_⟩
  have h67 : ResourceAccounting.Gates.val witZ.dag x 67 = 0 := witPre_val67 witZRest x
  have hA : ∀ a ∈ (List.range' 1 32).map (ResourceAccounting.Gates.val witZ.dag x), -128 ≤ a ∧ a ≤ 128 := by
    intro a ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witZ_valA x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  have hB : ∀ b ∈ (List.range' 33 32).map (ResourceAccounting.Gates.val witZ.dag x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witZ_valB x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 128) (Q := 128) (by norm_num) (by norm_num) _ _ hA hB
  simp only [List.length_map, List.length_range'] at hb1 hb2
  rw [val_of_getElem? witZ.dag x 68 (g := witZGate) rfl (by decide +kernel)]
  simp only [witZGate, witGateR, ResourceAccounting.Gates.IntOp.out, ResourceAccounting.Gates.IntOp.raw, List.map_cons, List.headD_cons,
    List.tail_cons, h67, zero_add, List.map_append]
  unfold ResourceAccounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

theorem witZ_dequantOf : DequantOf witDq witZ witZCls 69 68 65 66 := by
  refine ⟨by decide +kernel, by decide +kernel, rfl, by unfold S8; decide +kernel, by decide +kernel,
    by decide +kernel, by unfold S8; decide +kernel, fun x _ _ => ?_⟩
  rw [val_of_getElem? witZ.dag x 69 (g := witYGate) rfl (by decide +kernel)]
  rfl

theorem feeds_read {ι : Type} {Cd : Declared ι} {i j : ℕ} (h : Cd.Reads j i) : Feeds Cd i j :=
  Relation.ReflTransGen.single h

theorem feeds_tail {ι : Type} {Cd : Declared ι} {i j k : ℕ} (h : Feeds Cd i j) (hr : Cd.Reads k j) :
    Feeds Cd i k :=
  Relation.ReflTransGen.tail h hr

theorem held_of_none {ι : Type} {Cd : Declared ι} {cls : ℕ → Cls}
    (h : ∀ k, k < Cd.dag.length → cls k ≠ .check) (u : ℕ) (x : ι → ℤ) : Held Cd cls u x :=
  fun k hk hc _ => absurd hc (h k hk)

theorem witZ_feeds : ∀ t ∈ List.range' 1 32 ++ List.range' 33 32, Feeds witZ t 103 := by
  have h3 : witZ.Reads 103 102 := by unfold Declared.Reads; decide +kernel
  have hA : ∀ t ∈ List.range' 1 32, witZ.Reads (69 + t) t ∧ witZ.Reads 102 (69 + t) := by
    unfold Declared.Reads; decide +kernel
  have hB : ∀ t ∈ List.range' 33 32, witZ.Reads 102 t := by
    unfold Declared.Reads; decide +kernel
  intro t ht
  rcases List.mem_append.1 ht with ht | ht
  · exact feeds_tail (feeds_tail (feeds_read (hA t ht).1) (hA t ht).2) h3
  · exact feeds_tail (feeds_read (hB t ht)) h3

theorem witZ_outZ : ∀ z, z < witZ.dag.length → witZCls z = .outZ → z = 68 := by
  decide +kernel

theorem witZ_valueChecked : ValueChecked witShape witDq witZ witZCls := by
  refine ⟨fun _ => (List.range' 1 32, List.range' 33 32), fun _ => (68, 65, 66), fun z hz hzc => ?_,
    fun z z' hz hz' hzc hzc' _ => (witZ_outZ z hz hzc).trans (witZ_outZ z' hz' hzc').symm, fun y hy hyc => ?_,
    fun r hr hrc => ?_, fun x hx => ⟨x, hx, fun u => held_of_none (by decide +kernel) u x, fun _ _ _ _ _ _ _ _ => rfl⟩,
    fun _ _ _ _ _ => rfl⟩
  · obtain rfl : z = 68 := witZ_outZ z hz hzc
    exact witZ_productOf
  · obtain rfl : y = 69 :=
      (by decide +kernel : ∀ y, y < witZ.dag.length → witZCls y = .outY → y = 69) y hy hyc
    exact ⟨by decide +kernel, by decide +kernel, rfl, witZ_dequantOf⟩
  · obtain rfl : r = 103 :=
      (by decide +kernel : ∀ r, r < witZ.dag.length → witZCls r = .receipt → r = 103) r hr hrc
    exact ⟨69, by decide +kernel, by decide +kernel, rfl, by decide +kernel, by decide +kernel, witZ_feeds⟩

theorem witCreditTie : CreditTie witShape .zTree witDq witZ witZCls :=
  ⟨witTieZ, witZ_valueChecked⟩

def witMGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 68 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
    ResourceAccounting.Gates.s32Max false

def witSGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.affine 0 (List.replicate 32 1)) (List.range' 33 32) (-4096) 4064 false

def witCGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding fun v => v.headD 0 * v.getD 1 0) [0, 101] (-1044480) 1036320 false

def witPZGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.affine 0 [1, -1]) [100, 102] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true

def witPYGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [103, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

def witPRest : ResourceAccounting.Gates.IntDag ℕ :=
  (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witMGate, witSGate, witCGate, witPZGate, witPYGate,
      witGateR (.affine 0 [1]) [100] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true]

def witP : Declared ℕ where
  dag := witPre ++ witPRest
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if 68 ≤ i ∧ i ≠ 101 then 1 else 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

def witPCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 103 then .outZ
  else if i = 104 then .outY
  else if i = 105 then .receipt
  else .inner

theorem witTieP : Tie witShape .perCall witP witPCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witP.dag.length → (witPCls i = .act ∨ witPCls i = .served) →
        witP.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨103, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem witP_layerCommitted : LayerCommitted witP :=
  tieCommits ℕ witShape .perCall witP witPCls witTieP

theorem witP_peeled : (coinMark witP).getD 103 0 = 1 := by
  decide +kernel

theorem witP_valA (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed witP.dag x) {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    ResourceAccounting.Gates.val witP.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valA witPRest x hx h1 h2

theorem witP_valB (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed witP.dag x) {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    ResourceAccounting.Gates.val witP.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valB witPRest x hx h1 h2

theorem witP_valCoin (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed witP.dag x) :
    ResourceAccounting.Gates.val witP.dag x 0 = x 0 ∧ 0 ≤ x 0 ∧ x 0 ≤ 255 :=
  witPre_input (b := .coin) witPRest x (by rw [witPre_get witPRest (by omega)]; rfl) hx

theorem witP_valX (x : ℕ → ℤ) {k : ℕ} (h1 : 1 ≤ k) (h2 : k < 33) :
    ResourceAccounting.Gates.val witP.dag x (67 + k) =
      ResourceAccounting.Gates.val witP.dag x k + ResourceAccounting.Gates.val witP.dag x 0 := by
  have hg : witP.dag[67 + k]? = some (witGateR (.affine 0 [1, 1]) [k, 0] (-128) 382 false) := by
    interval_cases k <;> rfl
  rw [val_of_getElem? witP.dag x (67 + k) hg (by intro j hj; simp [witGateR] at hj; omega)]
  exact (affine_two 1 1 _ _).trans (by ring)

theorem witP_productOf : ProductOf witShape witP witPCls 103 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx _ => ?_⟩
  obtain ⟨hc, hc1, hc2⟩ := witP_valCoin x hx
  have h67 : ResourceAccounting.Gates.val witP.dag x 67 = 0 := witPre_val67 witPRest x
  have hX : (List.range' 68 32).map (ResourceAccounting.Gates.val witP.dag x) =
      ((List.range' 1 32).map (ResourceAccounting.Gates.val witP.dag x)).map (· + x 0) := by
    rw [show List.range' 68 32 = (List.range' 1 32).map (67 + ·) from (List.map_add_range' 1 32 1).symm,
      List.map_map, List.map_map]
    refine List.map_congr_left fun k hk => ?_
    obtain ⟨hk1, hk2⟩ := List.mem_range'_1.1 hk
    simp only [Function.comp_apply]
    rw [witP_valX x hk1 (by omega), hc]
  have hXb : ∀ a ∈ ((List.range' 1 32).map (ResourceAccounting.Gates.val witP.dag x)).map (· + x 0),
      -382 ≤ a ∧ a ≤ 382 := by
    intro a ha
    obtain ⟨v, hv, rfl⟩ := List.mem_map.1 ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hv
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witP_valA x hx hj1 (by omega)
    show -382 ≤ ResourceAccounting.Gates.val witP.dag x j + x 0 ∧ ResourceAccounting.Gates.val witP.dag x j + x 0 ≤ 382
    rw [he]
    constructor <;> omega
  have hBb : ∀ b ∈ (List.range' 33 32).map (ResourceAccounting.Gates.val witP.dag x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witP_valB x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 382) (Q := 128) (by norm_num) (by norm_num) _ _ hXb hBb
  simp only [List.length_map, List.length_range'] at hb1 hb2
  have hm : ResourceAccounting.Gates.val witP.dag x 100 = (List.zipWith (· * ·)
      (((List.range' 1 32).map (ResourceAccounting.Gates.val witP.dag x)).map (· + x 0))
      ((List.range' 33 32).map (ResourceAccounting.Gates.val witP.dag x))).sum := by
    rw [val_of_getElem? witP.dag x 100 (g := witMGate) rfl (by decide +kernel)]
    simp only [witMGate, witGateR, ResourceAccounting.Gates.IntOp.out, ResourceAccounting.Gates.IntOp.raw, List.map_cons,
      List.headD_cons, List.tail_cons, h67, zero_add, List.map_append, hX]
    unfold ResourceAccounting.Gates.dot
    rw [List.take_left' (by simp), List.drop_left' (by simp)]
    exact clampS32_of_bound (by omega) (by omega)
  have hS : ResourceAccounting.Gates.val witP.dag x 101 =
      ((List.range' 33 32).map (ResourceAccounting.Gates.val witP.dag x)).sum := by
    rw [val_of_getElem? witP.dag x 101 (g := witSGate) rfl (by decide +kernel)]
    show (0 : ℤ) + (List.zipWith (· * ·) (List.replicate 32 1)
      ((List.range' 33 32).map (ResourceAccounting.Gates.val witP.dag x))).sum = _
    rw [zero_add, sum_zipWith_one 32 _ (by simp)]
  have hC : ResourceAccounting.Gates.val witP.dag x 102 =
      x 0 * ((List.range' 33 32).map (ResourceAccounting.Gates.val witP.dag x)).sum := by
    rw [val_of_getElem? witP.dag x 102 (g := witCGate) rfl (by decide +kernel)]
    simp only [witCGate, witGateR, ResourceAccounting.Gates.IntOp.out, ResourceAccounting.Gates.IntOp.raw, List.map_cons, List.map_nil,
      List.headD_cons, List.getD_cons_succ, List.getD_cons_zero, hc, hS]
  rw [val_of_getElem? witP.dag x 103 (g := witPZGate) rfl (by decide +kernel)]
  refine (affine_two 1 (-1) _ _).trans ?_
  rw [hm, hC, sum_zipWith_add (x 0) _ _ (by simp)]
  ring

theorem witP_dequantOf : DequantOf witDq witP witPCls 104 103 65 66 := by
  refine ⟨by decide +kernel, by decide +kernel, rfl, by unfold S8; decide +kernel, by decide +kernel,
    by decide +kernel, by unfold S8; decide +kernel, fun x _ _ => ?_⟩
  rw [val_of_getElem? witP.dag x 104 (g := witPYGate) rfl (by decide +kernel)]
  rfl

theorem witP_feeds : ∀ t ∈ List.range' 1 32 ++ List.range' 33 32, Feeds witP t 105 := by
  have h3 : witP.Reads 105 100 := by unfold Declared.Reads; decide +kernel
  have hA : ∀ t ∈ List.range' 1 32, witP.Reads (67 + t) t ∧ witP.Reads 100 (67 + t) := by
    unfold Declared.Reads; decide +kernel
  have hB : ∀ t ∈ List.range' 33 32, witP.Reads 100 t := by
    unfold Declared.Reads; decide +kernel
  intro t ht
  rcases List.mem_append.1 ht with ht | ht
  · exact feeds_tail (feeds_tail (feeds_read (hA t ht).1) (hA t ht).2) h3
  · exact feeds_tail (feeds_read (hB t ht)) h3

theorem witP_outZ : ∀ z, z < witP.dag.length → witPCls z = .outZ → z = 103 := by
  decide +kernel

theorem witP_valueChecked : ValueChecked witShape witDq witP witPCls := by
  refine ⟨fun _ => (List.range' 1 32, List.range' 33 32), fun _ => (103, 65, 66), fun z hz hzc => ?_,
    fun z z' hz hz' hzc hzc' _ => (witP_outZ z hz hzc).trans (witP_outZ z' hz' hzc').symm, fun y hy hyc => ?_,
    fun r hr hrc => ?_, fun x hx => ⟨x, hx, fun u => held_of_none (by decide +kernel) u x, fun _ _ _ _ _ _ _ _ => rfl⟩,
    fun _ _ _ _ _ => rfl⟩
  · obtain rfl : z = 103 := witP_outZ z hz hzc
    exact witP_productOf
  · obtain rfl : y = 104 :=
      (by decide +kernel : ∀ y, y < witP.dag.length → witPCls y = .outY → y = 104) y hy hyc
    exact ⟨by decide +kernel, by decide +kernel, rfl, witP_dequantOf⟩
  · obtain rfl : r = 105 :=
      (by decide +kernel : ∀ r, r < witP.dag.length → witPCls r = .receipt → r = 105) r hr hrc
    exact ⟨104, by decide +kernel, by decide +kernel, rfl, by decide +kernel, by decide +kernel, witP_feeds⟩

theorem witCallCredit : CreditTie witShape .perCall witDq witP witPCls :=
  ⟨witTieP, witP_valueChecked⟩

theorem typed_of_all {Cd : Declared ℕ} (x : ℕ → ℤ)
    (h : Cd.dag.all (fun g => match g.op with
      | .input _ s => decide (g.lo ≤ x s ∧ x s ≤ g.hi)
      | _ => true) = true) : ResourceAccounting.Gates.Typed Cd.dag x := by
  intro g hg b s hop
  have h' := List.all_eq_true.1 h g hg
  simp only [hop, decide_eq_true_eq] at h'
  exact h'

def witX3 (j : ℕ) : ℤ := if j = 65 then 3 else 2

def witTwiceRest : ResourceAccounting.Gates.IntDag ℕ :=
  witZRest ++
    [witGateR (.affine 0 [1]) [68] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true,
      witGateR (.budgeted .rounding witDq) [104, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true,
      witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
        ResourceAccounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [106] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true]

def witTwice : Declared ℕ where
  dag := witPre ++ witTwiceRest
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if 70 ≤ i ∧ i ≠ 104 ∧ i ≠ 105 then 1 else 0
  seedNames := fun _ _ => false
  units := 2
  unit := fun i => if 104 ≤ i then 1 else 0

def witTwiceCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 ∨ i = 104 then .outZ
  else if i = 69 ∨ i = 105 then .outY
  else if i = 103 ∨ i = 107 then .receipt
  else .inner

def witShape2 : Shape := { witShape with linears := [(32, 16)] }

theorem witTieTwice2 : Tie witShape2 .zTree witTwice witTwiceCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witTwice.dag.length → (witTwiceCls i = .act ∨ witTwiceCls i = .served) →
        witTwice.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun i hi hr => by
    rcases (by decide +kernel : ∀ i, i < witTwice.dag.length → witTwiceCls i = .receipt → i = 103 ∨ i = 107) i hi hr
      with rfl | rfl
    · exact ⟨68, by decide +kernel, by decide +kernel, by decide +kernel⟩
    · exact ⟨104, by decide +kernel, by decide +kernel, by decide +kernel⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := by unfold Declared.Reads; decide +kernel
  inner := by decide +kernel

theorem witTwice2_population : witShape2.unitsOf 0 = 2 ∧ witTwice.units = witShape2.population ∧
    witTwice.unit 103 < witShape2.unitsOf 0 ∧ witTwice.unit 107 < witShape2.unitsOf 0 := by
  decide +kernel

theorem witTwice_copy (x : ℕ → ℤ) :
    ResourceAccounting.Gates.val witTwice.dag x 104 = ResourceAccounting.Gates.val witTwice.dag x 68 := by
  rw [val_of_getElem? witTwice.dag x 104
    (g := witGateR (.affine 0 [1]) [68] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true) rfl (by decide +kernel)]
  show (0 : ℤ) + (List.zipWith (· * ·) [1] [ResourceAccounting.Gates.val witTwice.dag x 68]).sum = _
  simp

/-- **One product doesn't value-check two units**: `witTwice`'s two `outZ` words have one value on
every input (`witTwice_copy`), so it fails `ValueChecked` at any shape and dequant. -/
theorem witTwice_notValueChecked (L : Shape) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq witTwice witTwiceCls := by
  rintro ⟨_, _, -, hinj, -, -, -⟩
  exact absurd (hinj 68 104 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    fun x _ _ _ => (witTwice_copy x).symm) (by decide)

theorem witTwice2_notCreditTie (dq : List ℤ → ℤ) : ¬ CreditTie witShape2 .zTree dq witTwice witTwiceCls := fun h =>
  witTwice_notValueChecked witShape2 dq h.2

def witWSRest : ResourceAccounting.Gates.IntDag ℕ :=
  [witZGate, witGateR (.budgeted .rounding witDq) [68, 1, 33] (-2 ^ 63) (2 ^ 63 - 1) true] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
        ResourceAccounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [102] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true]

def witWS : Declared ℕ := { witZ with dag := witPre ++ witWSRest }

theorem witTieWS : Tie witShape .zTree witWS witZCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witWS.dag.length → (witZCls i = .act ∨ witZCls i = .served) →
        witWS.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨68, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem witWS_typed : ResourceAccounting.Gates.Typed witWS.dag witX3 := typed_of_all witX3 (by decide +kernel)

theorem witWS_vals : ResourceAccounting.Gates.val witWS.dag witX3 69 = 512 ∧ ResourceAccounting.Gates.val witWS.dag witX3 68 = 128 ∧
    ResourceAccounting.Gates.val witWS.dag witX3 65 = 3 ∧ ResourceAccounting.Gates.val witWS.dag witX3 66 = 2 := by
  decide +kernel

/-- **A served `y` whose scales are an A word and a B word fails `ValueChecked`**: the scales are words
not declared s8, which in `witWS` are only `s_a` and `s_w`, and `y = 512` at `witX3` is not `dq(128, 3, 2) = 768`. -/
theorem witWS_notValueChecked (L : Shape) : ¬ ValueChecked L witDq witWS witZCls := by
  rintro ⟨_, src, -, -, hY, -, -⟩
  obtain ⟨hz, hzo, -, ⟨hsa, hsac, -, hsa8, hsw, hswc, hsw8, hD⟩⟩ := hY 69 (by decide +kernel) (by decide +kernel)
  have e1 : (src 69).1 = 68 :=
    (by decide +kernel : ∀ z, z < witWS.dag.length → witZCls z = .outZ → z = 68) _ hz hzo
  have e2 : (src 69).2.1 = 65 :=
    (by unfold S8; decide +kernel : ∀ s, s < witWS.dag.length → witZCls s = .act → ¬ S8 witWS s → s = 65)
      _ hsa hsac hsa8
  have e3 : (src 69).2.2 = 66 :=
    (by unfold S8; decide +kernel : ∀ s, s < witWS.dag.length → witZCls s = .registered → ¬ S8 witWS s → s = 66)
      _ hsw hswc hsw8
  have hv := hD witX3 witWS_typed (held_of_none (by decide +kernel) _ _)
  rw [e1, e2, e3, witWS_vals.1, witWS_vals.2.1, witWS_vals.2.2.1, witWS_vals.2.2.2] at hv
  exact absurd hv (by decide +kernel)

theorem witWS_notCreditTie : ¬ CreditTie witShape .zTree witDq witWS witZCls := fun h =>
  witWS_notValueChecked witShape h.2

def witRaw : Declared ℕ :=
  { witZ with
    dag := witPre ++ (witZRest ++
      [witGateR (.affine 0 [1]) [68] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true])
    commitRound := fun i => if 70 ≤ i ∧ i ≠ 104 then 1 else 0 }

def witRawCls (i : ℕ) : Cls := if i = 104 then .outY else witZCls i

theorem witTieRaw : Tie witShape .zTree witRaw witRawCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witRaw.dag.length → (witRawCls i = .act ∨ witRawCls i = .served) →
        witRaw.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨68, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem witRaw_typed : ResourceAccounting.Gates.Typed witRaw.dag witX3 := typed_of_all witX3 (by decide +kernel)

theorem witRaw_vals : ResourceAccounting.Gates.val witRaw.dag witX3 104 = 128 ∧
    ResourceAccounting.Gates.val witRaw.dag witX3 68 = 128 ∧ ResourceAccounting.Gates.val witRaw.dag witX3 65 = 3 ∧
    ResourceAccounting.Gates.val witRaw.dag witX3 66 = 2 := by
  decide +kernel

/-- **A raw product served as `y` fails `ValueChecked`**: every `outY` word is `dq` of an `outZ` word,
and `w = 128` at `witX3` is not `dq(128, 3, 2) = 768`. Before `z` and `y` had their own classes, `w` passed as a
product. -/
theorem witRaw_notValueChecked (L : Shape) : ¬ ValueChecked L witDq witRaw witRawCls := by
  rintro ⟨_, src, -, -, hY, -, -⟩
  obtain ⟨hz, hzo, -, ⟨hsa, hsac, -, hsa8, hsw, hswc, hsw8, hD⟩⟩ := hY 104 (by decide +kernel) (by decide +kernel)
  have e1 : (src 104).1 = 68 :=
    (by decide +kernel : ∀ z, z < witRaw.dag.length → witRawCls z = .outZ → z = 68) _ hz hzo
  have e2 : (src 104).2.1 = 65 :=
    (by unfold S8; decide +kernel : ∀ s, s < witRaw.dag.length → witRawCls s = .act → ¬ S8 witRaw s → s = 65)
      _ hsa hsac hsa8
  have e3 : (src 104).2.2 = 66 :=
    (by unfold S8; decide +kernel : ∀ s, s < witRaw.dag.length → witRawCls s = .registered → ¬ S8 witRaw s → s = 66)
      _ hsw hswc hsw8
  have hv := hD witX3 witRaw_typed (held_of_none (by decide +kernel) _ _)
  rw [e1, e2, e3, witRaw_vals.1, witRaw_vals.2.1, witRaw_vals.2.2.1, witRaw_vals.2.2.2] at hv
  exact absurd hv (by decide +kernel)

theorem witRaw_notCreditTie : ¬ CreditTie witShape .zTree witDq witRaw witRawCls := fun h =>
  witRaw_notValueChecked witShape h.2

def witOk (v : List ℤ) : ℤ := if v.getD 0 0 = v.getD 1 0 ∧ v.getD 2 0 = v.getD 3 0 then 1 else 0

theorem witOk_eq {z s y d a : ℤ} (h : witOk [z, s, y, d, a] = 1) : z = s ∧ y = d := by
  unfold witOk at h
  split at h
  · assumption
  · exact absurd h (by decide)

def witOkConst (_ : List ℤ) : ℤ := 1

def witOkHatch (v : List ℤ) : ℤ :=
  if (v.getD 0 0 = v.getD 1 0 ∧ v.getD 2 0 = v.getD 3 0) ∨ v.getD 4 0 = 1 then 1 else 0

def witOkNever (_ : List ℤ) : ℤ := 0

def witDotGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 1 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
    ResourceAccounting.Gates.s32Max false

def witDqGate : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [71, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) false

def witCheckGate (f : List ℤ → ℤ) : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .compareSelect f) [69, 71, 70, 72, 1] 0 1 true

def witCRest (f : List ℤ → ℤ) : ResourceAccounting.Gates.IntDag ℕ :=
  [witGateR (.input .run 68) [] 0 255 false,
    witGateR (.input .run 69) [] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true,
    witGateR (.input .run 70) [] (-2 ^ 63) (2 ^ 63 - 1) true, witDotGate, witDqGate, witCheckGate f] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 74 32 ++ List.range' 33 32)) ResourceAccounting.Gates.s32Min
        ResourceAccounting.Gates.s32Max false,
      witGateR (.affine 0 [1, 1]) [106, 68] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true]

def witC (f : List ℤ → ℤ) : Declared ℕ where
  dag := witPre ++ witCRest f
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if 73 ≤ i then 1 else 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

def witCCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 then .pub
  else if i = 69 then .outZ
  else if i = 70 then .outY
  else if i = 73 then .check
  else if i = 107 then .receipt
  else .inner

theorem witTieC : Tie witShape .zTree (witC witOk) witCCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < (witC witOk).dag.length → (witCCls i = .act ∨ witCCls i = .served) →
        (witC witOk).binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨69, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

theorem witC_layerCommitted : LayerCommitted (witC witOk) :=
  tieCommits ℕ witShape .zTree (witC witOk) witCCls witTieC

theorem witC_length (f : List ℤ → ℤ) : (witC f).dag.length = 108 := rfl

theorem witC_valA (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) {j : ℕ} (h1 : 1 ≤ j)
    (h2 : j < 33) : ResourceAccounting.Gates.val (witC f).dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valA (witCRest f) x hx h1 h2

theorem witC_valB (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) {j : ℕ} (h1 : 33 ≤ j)
    (h2 : j < 65) : ResourceAccounting.Gates.val (witC f).dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valB (witCRest f) x hx h1 h2

theorem witC_valSa (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) :
    ResourceAccounting.Gates.val (witC f).dag x 65 = x 65 ∧ 0 ≤ x 65 ∧ x 65 ≤ 65535 :=
  witPre_input (b := .run) (witCRest f) x (by rw [witPre_get (witCRest f) (by omega)]; rfl) hx

theorem witC_valSw (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) :
    ResourceAccounting.Gates.val (witC f).dag x 66 = x 66 ∧ 0 ≤ x 66 ∧ x 66 ≤ 65535 :=
  witPre_input (b := .registered) (witCRest f) x (by rw [witPre_get (witCRest f) (by omega)]; rfl) hx

theorem witC_val69 (f : List ℤ → ℤ) (x : ℕ → ℤ) : ResourceAccounting.Gates.val (witC f).dag x 69 = x 69 := by
  rw [val_of_getElem? (witC f).dag x 69
    (g := witGateR (.input .run 69) [] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true) rfl
    (by simp [witGateR])]
  rfl

theorem witC_val70 (f : List ℤ → ℤ) (x : ℕ → ℤ) : ResourceAccounting.Gates.val (witC f).dag x 70 = x 70 := by
  rw [val_of_getElem? (witC f).dag x 70 (g := witGateR (.input .run 70) [] (-2 ^ 63) (2 ^ 63 - 1) true) rfl
    (by simp [witGateR])]
  rfl

theorem witC_sum_bound (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) :
    -524288 ≤ (List.zipWith (· * ·) ((List.range' 1 32).map (ResourceAccounting.Gates.val (witC f).dag x))
        ((List.range' 33 32).map (ResourceAccounting.Gates.val (witC f).dag x))).sum ∧
      (List.zipWith (· * ·) ((List.range' 1 32).map (ResourceAccounting.Gates.val (witC f).dag x))
        ((List.range' 33 32).map (ResourceAccounting.Gates.val (witC f).dag x))).sum ≤ 524288 := by
  have hA : ∀ a ∈ (List.range' 1 32).map (ResourceAccounting.Gates.val (witC f).dag x), -128 ≤ a ∧ a ≤ 128 := by
    intro a ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witC_valA f x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  have hB : ∀ b ∈ (List.range' 33 32).map (ResourceAccounting.Gates.val (witC f).dag x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witC_valB f x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 128) (Q := 128) (by norm_num) (by norm_num) _ _ hA hB
  simp only [List.length_map, List.length_range'] at hb1 hb2
  constructor <;> omega

theorem witC_dot (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) :
    ResourceAccounting.Gates.val (witC f).dag x 71 =
      (List.zipWith (· * ·) ((List.range' 1 32).map (ResourceAccounting.Gates.val (witC f).dag x))
        ((List.range' 33 32).map (ResourceAccounting.Gates.val (witC f).dag x))).sum := by
  have h67 : ResourceAccounting.Gates.val (witC f).dag x 67 = 0 := witPre_val67 (witCRest f) x
  obtain ⟨hb1, hb2⟩ := witC_sum_bound f x hx
  rw [val_of_getElem? (witC f).dag x 71 (g := witDotGate) rfl (by decide +kernel)]
  simp only [witDotGate, witGateR, ResourceAccounting.Gates.IntOp.out, ResourceAccounting.Gates.IntOp.raw, List.map_cons, List.headD_cons,
    List.tail_cons, h67, zero_add, List.map_append]
  unfold ResourceAccounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

theorem witC_dot_bound (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) :
    -524288 ≤ ResourceAccounting.Gates.val (witC f).dag x 71 ∧ ResourceAccounting.Gates.val (witC f).dag x 71 ≤ 524288 := by
  rw [witC_dot f x hx]
  exact witC_sum_bound f x hx

theorem witC_dq (f : List ℤ → ℤ) (x : ℕ → ℤ) :
    ResourceAccounting.Gates.val (witC f).dag x 72 = witDq [ResourceAccounting.Gates.val (witC f).dag x 71,
      ResourceAccounting.Gates.val (witC f).dag x 65, ResourceAccounting.Gates.val (witC f).dag x 66] := by
  rw [val_of_getElem? (witC f).dag x 72 (g := witDqGate) rfl (by decide +kernel)]
  rfl

theorem witDq_bound {z a w : ℤ} (hz1 : -524288 ≤ z) (hz2 : z ≤ 524288) (ha1 : 0 ≤ a) (ha2 : a ≤ 65535)
    (hw1 : 0 ≤ w) (hw2 : w ≤ 65535) : -2 ^ 63 ≤ witDq [z, a, w] ∧ witDq [z, a, w] ≤ 2 ^ 63 - 1 := by
  show -2 ^ 63 ≤ z * a * w ∧ z * a * w ≤ 2 ^ 63 - 1
  have hp0 : 0 ≤ a * w := mul_nonneg ha1 hw1
  have hp1 : a * w ≤ 65535 * 65535 := by nlinarith [mul_nonneg (sub_nonneg.2 ha2) hw1]
  have h1 := mul_nonneg (by linarith : (0 : ℤ) ≤ 524288 - z) hp0
  have h2 := mul_nonneg (by linarith : (0 : ℤ) ≤ z + 524288) hp0
  constructor <;> nlinarith

theorem witC_dq_bound (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC f).dag x) :
    -2 ^ 63 ≤ ResourceAccounting.Gates.val (witC f).dag x 72 ∧ ResourceAccounting.Gates.val (witC f).dag x 72 ≤ 2 ^ 63 - 1 := by
  obtain ⟨hz1, hz2⟩ := witC_dot_bound f x hx
  obtain ⟨ha, ha1, ha2⟩ := witC_valSa f x hx
  obtain ⟨hw, hw1, hw2⟩ := witC_valSw f x hx
  rw [witC_dq f x, ha, hw]
  exact witDq_bound hz1 hz2 ha1 ha2 hw1 hw2

theorem witC_ok (x : ℕ → ℤ) (h : ResourceAccounting.Gates.val (witC witOk).dag x 73 = 1) :
    ResourceAccounting.Gates.val (witC witOk).dag x 69 = ResourceAccounting.Gates.val (witC witOk).dag x 71 ∧
      ResourceAccounting.Gates.val (witC witOk).dag x 70 = ResourceAccounting.Gates.val (witC witOk).dag x 72 := by
  rw [val_of_getElem? (witC witOk).dag x 73 (g := witCheckGate witOk) rfl (by decide +kernel)] at h
  change witOk [ResourceAccounting.Gates.val (witC witOk).dag x 69, ResourceAccounting.Gates.val (witC witOk).dag x 71,
    ResourceAccounting.Gates.val (witC witOk).dag x 70, ResourceAccounting.Gates.val (witC witOk).dag x 72,
    ResourceAccounting.Gates.val (witC witOk).dag x 1] = 1 at h
  exact witOk_eq h

theorem witC_outZ : ∀ z, z < (witC witOk).dag.length → witCCls z = .outZ → z = 69 := by
  decide +kernel

theorem witC_check : ∀ k, k < (witC witOk).dag.length → witCCls k = .check → k = 73 := by
  decide +kernel

theorem witC_productOf : ProductOf witShape (witC witOk) witCCls 69 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx hH => ?_⟩
  rw [(witC_ok x (hH 73 (by decide +kernel) (by decide +kernel) rfl)).1]
  exact witC_dot witOk x hx

theorem witC_dequantOf : DequantOf witDq (witC witOk) witCCls 70 69 65 66 := by
  refine ⟨by decide +kernel, by decide +kernel, rfl, by unfold S8; decide +kernel, by decide +kernel,
    by decide +kernel, by unfold S8; decide +kernel, fun x _ hH => ?_⟩
  obtain ⟨h69, h70⟩ := witC_ok x (hH 73 (by decide +kernel) (by decide +kernel) rfl)
  rw [h70, h69, witC_dq witOk x]

theorem witC_feeds : ∀ t ∈ List.range' 1 32 ++ List.range' 33 32, Feeds (witC witOk) t 107 := by
  have h3 : (witC witOk).Reads 107 106 := by unfold Declared.Reads; decide +kernel
  have hA : ∀ t ∈ List.range' 1 32, (witC witOk).Reads (73 + t) t ∧ (witC witOk).Reads 106 (73 + t) := by
    unfold Declared.Reads; decide +kernel
  have hB : ∀ t ∈ List.range' 33 32, (witC witOk).Reads 106 t := by
    unfold Declared.Reads; decide +kernel
  intro t ht
  rcases List.mem_append.1 ht with ht | ht
  · exact feeds_tail (feeds_tail (feeds_read (hA t ht).1) (hA t ht).2) h3
  · exact feeds_tail (feeds_read (hB t ht)) h3

def slotOf : ResourceAccounting.Gates.IntOp ℕ → Option ℕ
  | .input _ s => some s
  | _ => none

theorem witC_slots : ∀ i, i < (witC witOk).dag.length →
    ((witC witOk).dag[i]?).bind (fun g => slotOf g.op) = none ∨
      ((witC witOk).dag[i]?).bind (fun g => slotOf g.op) = some i := by
  decide +kernel

theorem witC_slot {i s : ℕ} {g : ResourceAccounting.Gates.IntGate ℕ} {b : ResourceAccounting.Gates.Binding}
    (hg : (witC witOk).dag[i]? = some g) (hop : g.op = .input b s) : s = i := by
  obtain ⟨hi, -⟩ := List.getElem?_eq_some_iff.1 hg
  have hs : ((witC witOk).dag[i]?).bind (fun g => slotOf g.op) = some s := by
    rw [hg]
    show slotOf g.op = some s
    rw [hop]; rfl
  rcases witC_slots i hi with h | h <;> rw [hs] at h
  · cases h
  · exact Option.some.inj h

theorem witC_bounds : ∀ g ∈ (witC witOk).dag,
    (slotOf g.op = some 69 → g.lo = ResourceAccounting.Gates.s32Min ∧ g.hi = ResourceAccounting.Gates.s32Max) ∧
      (slotOf g.op = some 70 → g.lo = -2 ^ 63 ∧ g.hi = 2 ^ 63 - 1) := by
  decide +kernel

def witCFix (x : ℕ → ℤ) (s : ℕ) : ℤ :=
  if s = 69 then ResourceAccounting.Gates.val (witC witOk).dag x 71
  else if s = 70 then ResourceAccounting.Gates.val (witC witOk).dag x 72 else x s

theorem witCFix_69 (x : ℕ → ℤ) : witCFix x 69 = ResourceAccounting.Gates.val (witC witOk).dag x 71 := by
  simp [witCFix]

theorem witCFix_70 (x : ℕ → ℤ) : witCFix x 70 = ResourceAccounting.Gates.val (witC witOk).dag x 72 := by
  simp [witCFix]

theorem witCFix_ne (x : ℕ → ℤ) {s : ℕ} (h1 : s ≠ 69) (h2 : s ≠ 70) : witCFix x s = x s := by
  simp [witCFix, h1, h2]

theorem witCFix_typed (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC witOk).dag x) :
    ResourceAccounting.Gates.Typed (witC witOk).dag (witCFix x) := by
  intro g hg b s hop
  have hsl : slotOf g.op = some s := by rw [hop]; rfl
  obtain ⟨h69, h70⟩ := witC_bounds g hg
  by_cases e69 : s = 69
  · subst e69
    obtain ⟨hlo, hhi⟩ := h69 hsl
    obtain ⟨hz1, hz2⟩ := witC_dot_bound witOk x hx
    rw [hlo, hhi, witCFix_69]
    exact ⟨le_trans (by norm_num [ResourceAccounting.Gates.s32Min]) hz1, le_trans hz2 (by norm_num [ResourceAccounting.Gates.s32Max])⟩
  · by_cases e70 : s = 70
    · subst e70
      obtain ⟨hlo, hhi⟩ := h70 hsl
      rw [hlo, hhi, witCFix_70]
      exact witC_dq_bound witOk x hx
    · rw [witCFix_ne x e69 e70]
      exact hx g hg b s hop

theorem witCFix_held (x : ℕ → ℤ) (hx : ResourceAccounting.Gates.Typed (witC witOk).dag x) (u : ℕ) :
    Held (witC witOk) witCCls u (witCFix x) := by
  intro k hk hc _
  obtain rfl : k = 73 := witC_check k hk hc
  have hx' := witCFix_typed x hx
  have hA : (List.range' 1 32).map (ResourceAccounting.Gates.val (witC witOk).dag (witCFix x)) =
      (List.range' 1 32).map (ResourceAccounting.Gates.val (witC witOk).dag x) :=
    List.map_congr_left fun j hj => by
      obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
      rw [(witC_valA witOk _ hx' hj1 (by omega)).1, (witC_valA witOk x hx hj1 (by omega)).1]
      exact witCFix_ne x (by omega) (by omega)
  have hB : (List.range' 33 32).map (ResourceAccounting.Gates.val (witC witOk).dag (witCFix x)) =
      (List.range' 33 32).map (ResourceAccounting.Gates.val (witC witOk).dag x) :=
    List.map_congr_left fun j hj => by
      obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
      rw [(witC_valB witOk _ hx' hj1 (by omega)).1, (witC_valB witOk x hx hj1 (by omega)).1]
      exact witCFix_ne x (by omega) (by omega)
  have h71 : ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 71 = ResourceAccounting.Gates.val (witC witOk).dag x 71 := by
    rw [witC_dot witOk _ hx', witC_dot witOk x hx, hA, hB]
  have h72 : ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 72 = ResourceAccounting.Gates.val (witC witOk).dag x 72 := by
    rw [witC_dq witOk (witCFix x), witC_dq witOk x, h71, (witC_valSa witOk _ hx').1, (witC_valSa witOk x hx).1,
      (witC_valSw witOk _ hx').1, (witC_valSw witOk x hx).1, witCFix_ne x (s := 65) (by norm_num) (by norm_num),
      witCFix_ne x (s := 66) (by norm_num) (by norm_num)]
  rw [val_of_getElem? (witC witOk).dag (witCFix x) 73 (g := witCheckGate witOk) rfl (by decide +kernel)]
  show witOk [ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 69,
    ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 71, ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 70,
    ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 72, ResourceAccounting.Gates.val (witC witOk).dag (witCFix x) 1] = 1
  rw [witC_val69, witC_val70, witCFix_69, witCFix_70, h71, h72]
  simp [witOk]

theorem witCFix_agree (x : ℕ → ℤ) : ∀ i g b s, (witC witOk).dag[i]? = some g → g.op = .input b s →
    witCCls i ≠ .outZ → witCCls i ≠ .outY → witCFix x s = x s := by
  intro i g b s hg hop hZ hY
  have hsi : s = i := witC_slot hg hop
  obtain ⟨hi, -⟩ := List.getElem?_eq_some_iff.1 hg
  obtain ⟨h1, h2⟩ := (by decide +kernel : ∀ i, i < (witC witOk).dag.length → witCCls i ≠ .outZ →
    witCCls i ≠ .outY → i ≠ 69 ∧ i ≠ 70) i hi hZ hY
  exact witCFix_ne x (by omega) (by omega)

theorem witC_valueChecked : ValueChecked witShape witDq (witC witOk) witCCls := by
  refine ⟨fun _ => (List.range' 1 32, List.range' 33 32), fun _ => (69, 65, 66), fun z hz hzc => ?_,
    fun z z' hz hz' hzc hzc' _ => (witC_outZ z hz hzc).trans (witC_outZ z' hz' hzc').symm, fun y hy hyc => ?_,
    fun r hr hrc => ?_, fun x hx => ⟨witCFix x, witCFix_typed x hx, witCFix_held x hx, witCFix_agree x⟩,
    fun _ _ _ _ _ => rfl⟩
  · obtain rfl : z = 69 := witC_outZ z hz hzc
    exact witC_productOf
  · obtain rfl : y = 70 :=
      (by decide +kernel : ∀ y, y < (witC witOk).dag.length → witCCls y = .outY → y = 70) y hy hyc
    exact ⟨by decide +kernel, by decide +kernel, rfl, witC_dequantOf⟩
  · obtain rfl : r = 107 :=
      (by decide +kernel : ∀ r, r < (witC witOk).dag.length → witCCls r = .receipt → r = 107) r hr hrc
    exact ⟨70, by decide +kernel, by decide +kernel, rfl, by decide +kernel, by decide +kernel, witC_feeds⟩

theorem witCheckedCredit : CreditTie witShape .zTree witDq (witC witOk) witCCls :=
  ⟨witTieC, witC_valueChecked⟩

def witXR (s : ℕ) : ℤ := if s = 1 ∨ s = 69 then 1 else 0

theorem sum_zipWith_zero : ∀ (l1 l2 : List ℤ), (∀ b ∈ l2, b = 0) → (List.zipWith (· * ·) l1 l2).sum = 0
  | [], _, _ => by simp
  | _ :: _, [], _ => by simp
  | a :: l1, b :: l2, h => by
    simp only [List.zipWith_cons_cons, List.sum_cons]
    rw [h b (by simp), sum_zipWith_zero l1 l2 (fun c hc => h c (by simp [hc]))]
    simp

/-- A check that holds at `witXR`, whose record `z = 1` is no product over its B column of zeros, leaves the record
unchecked: `ProductOf` fails at `z`. -/
theorem witC_notValueChecked (f : List ℤ → ℤ) (L : Shape) (dq : List ℤ → ℤ)
    (hx : ResourceAccounting.Gates.Typed (witC f).dag witXR)
    (hS : ∀ b, b < (witC f).dag.length → witCCls b = .registered → S8 (witC f) b → 33 ≤ b ∧ b < 65)
    (hchk : ∀ k, k < (witC f).dag.length → witCCls k = .check → k = 73)
    (h73 : ResourceAccounting.Gates.val (witC f).dag witXR 73 = 1) (h69 : ResourceAccounting.Gates.val (witC f).dag witXR 69 = 1) :
    ¬ ValueChecked L dq (witC f) witCCls := by
  rintro ⟨ops, _, hP, -, -, -, -⟩
  obtain ⟨-, -, -, -, -, hbs, hv⟩ := hP 69 (lt_of_lt_of_eq (by norm_num) (witC_length f).symm) (by decide)
  have h := hv witXR hx fun k hk hc _ => by rw [hchk k hk hc]; exact h73
  have hz : ∀ v ∈ (ops 69).2.map (ResourceAccounting.Gates.val (witC f).dag witXR), v = 0 := by
    intro v hv'
    obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hv'
    obtain ⟨hbl, hbc, hb8⟩ := hbs b hb
    obtain ⟨h1, h2⟩ := hS b hbl hbc hb8
    rw [(witC_valB f witXR hx h1 h2).1]
    unfold witXR
    split <;> omega
  rw [h69, sum_zipWith_zero _ _ hz] at h
  exact absurd h (by decide)

theorem witTieCConst : Tie witShape .zTree (witC witOkConst) witCCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < (witC witOkConst).dag.length → (witCCls i = .act ∨ witCCls i = .served) →
        (witC witOkConst).binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨69, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

/-- **A constant check doesn't value-check the record**: `ok = 1` holds with any record, and at `witXR` the record
`z = 1` is no product. -/
theorem witConst_notValueChecked (L : Shape) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq (witC witOkConst) witCCls :=
  witC_notValueChecked witOkConst L dq (typed_of_all witXR (by decide +kernel)) (by unfold S8; decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witConst_notCreditTie : ¬ CreditTie witShape .zTree witDq (witC witOkConst) witCCls := fun h =>
  witConst_notValueChecked witShape witDq h.2

theorem witTieCHatch : Tie witShape .zTree (witC witOkHatch) witCCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < (witC witOkHatch).dag.length → (witCCls i = .act ∨ witCCls i = .served) →
        (witC witOkHatch).binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨69, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

/-- **A check with an escape hatch doesn't value-check the record**: `ok = (z = Σ ∧ y = dq) ∨ A_1 = 1` holds at
`witXR` with the record `z = 1`, which is no product. -/
theorem witHatch_notValueChecked (L : Shape) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq (witC witOkHatch) witCCls :=
  witC_notValueChecked witOkHatch L dq (typed_of_all witXR (by decide +kernel)) (by unfold S8; decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witHatch_notCreditTie : ¬ CreditTie witShape .zTree witDq (witC witOkHatch) witCCls := fun h =>
  witHatch_notValueChecked witShape witDq h.2

theorem witNever_check (x : ℕ → ℤ) : ResourceAccounting.Gates.val (witC witOkNever).dag x 73 = 0 := by
  rw [val_of_getElem? (witC witOkNever).dag x 73 (g := witCheckGate witOkNever) rfl (by decide +kernel)]
  rfl

theorem witNever_productOf :
    ProductOf witShape (witC witOkNever) witCCls 69 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x _ hH => ?_⟩
  exact absurd ((witNever_check x).symm.trans (hH 73 (by decide +kernel) (by decide +kernel) rfl)) (by decide)

/-- **`ValueChecked` isn't met vacuously**: its last clause asks for an input on which every unit's checks hold, which
a check no record meets has none of. -/
theorem witNever_notValueChecked (L : Shape) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq (witC witOkNever) witCCls := by
  rintro ⟨_, _, -, -, -, -, hm, -⟩
  obtain ⟨x', -, hH, -⟩ := hm witXR (typed_of_all witXR (by decide +kernel))
  exact absurd ((witNever_check x').symm.trans (hH 0 73 (by decide +kernel) (by decide +kernel) rfl)) (by decide)

def witCReader : ResourceAccounting.Gates.IntDag ℕ :=
  (witC witOk).dag ++ [witGateR (.affine 0 [1]) [69] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max false]

def witCrossUnit : Declared ℕ :=
  { witC witOk with dag := witCReader, units := 2, unit := fun i => if i = 108 then 1 else 0 }

def witCrossCall : Declared ℕ :=
  { witC witOk with dag := witCReader, calls := 2, call := fun i => if i = 108 then 1 else 0 }

/-- **A record another unit reads isn't tied**, however its gates are classed: a run input classed `outZ` or `outY` is
read only by gates of its own unit (`Tie.outWords`), whose check words are the only ones that hold it. -/
theorem witCrossUnit_notTie (L : Shape) (F : Form) (cls : ℕ → Cls) (h : cls 69 = .outZ ∨ cls 69 = .outY) :
    ¬ Tie L F witCrossUnit cls := fun T => by
  rcases T.outWords 69 (by decide +kernel) h with hb | ⟨-, -, hr⟩
  · exact absurd hb (by decide +kernel)
  · exact absurd (hr 108 (by decide +kernel) (by unfold Declared.Reads; decide +kernel)).2 (by decide +kernel)

/-- **A record another call reads isn't tied**, however its gates are classed. -/
theorem witCrossCall_notTie (L : Shape) (F : Form) (cls : ℕ → Cls) (h : cls 69 = .outZ ∨ cls 69 = .outY) :
    ¬ Tie L F witCrossCall cls := fun T => by
  rcases T.outWords 69 (by decide +kernel) h with hb | ⟨-, -, hr⟩
  · exact absurd hb (by decide +kernel)
  · exact absurd (hr 108 (by decide +kernel) (by unfold Declared.Reads; decide +kernel)).1 (by decide +kernel)

def witPubLate : Declared Unit := { witCircuit with commitRound := fun i => if i = 0 then 0 else 1 }

def witPubCls : ℕ → Cls
  | 0 => .coin
  | 1 => .pub
  | _ => .receipt

/-- **A public input committed after a reader's coin isn't tied** (`Tie.pub`): a `pub` word has no commit coin of its
class, and must precede the coin of every round that reads it. -/
theorem witPubLate_notTie (L : Shape) (F : Form) : ¬ Tie L F witPubLate witPubCls := fun T =>
  absurd ((T.pub 1 (by decide +kernel) rfl).2.2.2 2 (by decide +kernel) (by unfold Declared.Reads; decide +kernel))
    (by decide +kernel)

/-- **Nor does it meet `LayerCommitted`**: an input has no second case. -/
theorem witPubLate_notLayerCommitted : ¬ LayerCommitted witPubLate := fun h => by
  rcases h 1 ⟨by decide +kernel, by decide +kernel⟩ with h1 | ⟨h2, -⟩
  · exact absurd h1 (by decide +kernel)
  · exact absurd h2 (by decide +kernel)

def witShape8 : Shape := { witShape with steps := 8 }

def witRingStep (s : ℕ) : ResourceAccounting.Gates.IntDag Unit :=
  [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.affine 0 [1]) [4 * s + 1] true, witGate (.affine 0 [1, 1]) [4 * s, 4 * s + 1] true]

def witRing : Declared Unit where
  dag := (List.range 8).flatMap witRingStep
  calls := 8
  call := fun i => i / 4
  rounds := 8
  round := fun c => c
  coinMode := fun _ => .live
  commitRound := fun i => if i % 4 = 3 then min (i / 4 + 5) 8 else i / 4
  seedNames := fun _ _ => false
  units := 8
  unit := fun i => i / 4

def witRingCls (i : ℕ) : Cls :=
  if i % 4 = 0 then .coin else if i % 4 = 1 then .act else if i % 4 = 2 then .outZ else .receipt

theorem witRing_links : ∀ i, i < witRing.dag.length → witRingCls i = .receipt →
    i - 1 < witRing.dag.length ∧ witRingCls (i - 1) = .outZ ∧ witRing.unit (i - 1) = witRing.unit i := by
  decide +kernel

theorem witTieRing : Tie witShape8 .zTree witRing witRingCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := by decide +kernel
  inRange := by decide +kernel
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := by unfold Declared.Reads; decide +kernel
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun i hi hr =>
    ⟨i - 1, (witRing_links i hi hr).1, Or.inl (witRing_links i hi hr).2.1, (witRing_links i hi hr).2.2⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := by unfold Declared.Reads; decide +kernel
  inner := by decide +kernel

theorem witRing_late : witRing.Writes 3 ∧ witRing.cutOf 3 = 1 ∧ witRing.commitRound 3 = 5 :=
  ⟨⟨by decide +kernel, by decide +kernel⟩, by decide +kernel, by decide +kernel⟩

theorem witRing_layerCommitted : LayerCommitted witRing :=
  tieCommits Unit witShape8 .zTree witRing witRingCls witTieRing

theorem witRing_wf : witRing.Wf := by
  have hcoin : ∀ i, i < witRing.dag.length → ∀ j, j < witRing.dag.length → witRing.binding i = some .coin →
      witRing.Reads j i → witRing.round (witRing.call i) = witRing.round (witRing.call j) := by
    unfold Declared.Reads; decide +kernel
  exact ⟨by decide +kernel, by decide +kernel, fun _ _ _ _ h => h, fun i j hi hj hb hR => hcoin i hi j hj hb hR,
    by decide +kernel⟩

noncomputable def witScheme : Scheme Unit where
  floor := fun _ => ⟨fun _ => 0, fun _ => 0, fun _ => 0, fun _ => 0⟩
  credited := fun _ => true
  span := fun _ => 0
  form := fun _ => .pinned
  mu := fun _ => 0
  rB := fun _ => 1 / 2
  guess := fun _ => 1 / 2
  seedRate := fun _ => 0
  wholeRate := fun _ => 0

theorem witScheme_luckProved : witScheme.LuckProved := fun _ => by norm_num [witScheme, LuckForm.value]

theorem witRing_scope {δs : ℝ} (hδs : 0 < δs) : Scope witScheme witRing δs :=
  tieScope Unit Unit witScheme witShape8 .zTree witRing witRingCls δs witTieRing witRing_wf hδs witScheme_luckProved

def witShape2s : Shape := { witShape with steps := 2 }

def witCross : Declared Unit where
  dag := [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.affine 0 [1, 1]) [0, 1] true, witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.affine 0 [1, 1, 1]) [3, 4, 2] true]
  calls := 2
  call := fun i => i / 3
  rounds := 2
  round := fun c => c
  coinMode := fun _ => .live
  commitRound := fun i => if i = 2 ∨ i = 5 then 2 else i / 3
  seedNames := fun _ _ => false
  units := 2
  unit := fun i => i / 3

def witCrossCls (i : ℕ) : Cls :=
  if i % 3 = 0 then .coin else if i % 3 = 1 then .act else .receipt

/-- **A receipt that another unit reads isn't tied**: step `0`'s receipt (gate 2), read by step `1`'s (gate 5, unit
`1`) and committed after that reader's coin, fails `Tie.readers`. -/
theorem witCross_notTie : ¬ Tie witShape2s .beside witCross witCrossCls := fun T =>
  absurd (T.readers 2 (by decide +kernel) (by decide +kernel) 5 (by decide +kernel)
    (by unfold Declared.Reads; decide +kernel)) (by decide +kernel)

/-- **Nor does it meet `LayerCommitted`**: its cut is coin `1`, the next round's, before its commit coin `2`, and a
gate of another unit reads it. -/
theorem witCross_notLayerCommitted : ¬ LayerCommitted witCross := fun h => by
  rcases h 2 ⟨by decide +kernel, by decide +kernel⟩ with h1 | ⟨-, -, h2⟩
  · exact absurd h1 (by decide +kernel)
  · exact absurd (h2 5 (by unfold Declared.Reads; decide +kernel)) (by decide +kernel)

def witRotB : List ℕ := List.range' 34 31 ++ [33]

def witDotOf (h : ℕ) (bs : List ℕ) (commit : Bool) : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (h :: (List.range' 1 32 ++ bs)) ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max
    commit

def witDqOf (z : ℕ) (commit : Bool) : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [z, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) commit

def witRcpt (t : ℕ) : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.affine 0 [1, 1]) [t, 0] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true

def witRecZ (s : ℕ) : ResourceAccounting.Gates.IntGate ℕ :=
  witGateR (.input .run s) [] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true

def witRecY (s : ℕ) : ResourceAccounting.Gates.IntGate ℕ := witGateR (.input .run s) [] (-2 ^ 63) (2 ^ 63 - 1) true

def witOkCross (v : List ℤ) : ℤ :=
  if (v.getD 0 0 = v.getD 1 0 ∧ v.getD 2 0 = v.getD 3 0) ∨ v.getD 4 0 ≠ v.getD 5 0 then 1 else 0

theorem witOkCross_eq {z s y d a b : ℤ} (h : witOkCross [z, s, y, d, a, b] = 1) (hab : a = b) : z = s ∧ y = d := by
  unfold witOkCross at h
  split at h
  · rename_i hc
    exact hc.resolve_right fun hne => hne hab
  · exact absurd h (by decide)

theorem computedFeeds_unit {ι : Type} {Cd : Declared ι}
    (h : ∀ b, b < Cd.dag.length → Cd.binding b = none → ∀ a, a < Cd.dag.length → Cd.Reads b a →
      Cd.binding a = none → Cd.unit a = Cd.unit b) {g w : ℕ} (hf : ComputedFeeds Cd g w) : Cd.unit g = Cd.unit w := by
  induction hf with
  | refl => rfl
  | tail _ hr ih => exact ih.trans (h _ (reads_lt hr.1) hr.2.2.2 _ hr.2.1 hr.1 hr.2.2.1)

def witConeCheckRest : ResourceAccounting.Gates.IntDag ℕ :=
  [witRecZ 68, witRecY 69, witDotOf 67 (List.range' 33 32) false, witDqOf 70 false, witGateR (.const 0) [] 0 0 false,
    witDotOf 72 witRotB true, witDotOf 67 witRotB false,
    witGateR (.budgeted .compareSelect witOkCross) [68, 70, 69, 71, 73, 74] 0 1 true, witDqOf 73 true, witRcpt 70,
    witRcpt 73]

def witConeCheck : Declared ℕ where
  dag := witPre ++ witConeCheckRest
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if i = 75 ∨ 77 ≤ i then 1 else 0
  seedNames := fun _ _ => false
  units := 2
  unit := fun i => if i = 72 ∨ i = 73 ∨ i = 76 ∨ i = 78 then 1 else 0

def witConeCheckCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 ∨ i = 73 then .outZ
  else if i = 69 ∨ i = 76 then .outY
  else if i = 75 then .check
  else if i = 77 ∨ i = 78 then .receipt
  else .inner

theorem witTieConeCheck : Tie witShape2 .zTree witConeCheck witConeCheckCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witConeCheck.dag.length →
        (witConeCheckCls i = .act ∨ witConeCheckCls i = .served) → witConeCheck.binding i = some .run) i hi h,
      fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun i hi hr => by
    rcases (by decide +kernel : ∀ i, i < witConeCheck.dag.length → witConeCheckCls i = .receipt → i = 77 ∨ i = 78)
      i hi hr with rfl | rfl
    · exact ⟨68, by decide +kernel, by decide +kernel, by decide +kernel⟩
    · exact ⟨73, by decide +kernel, by decide +kernel, by decide +kernel⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := by unfold Declared.Reads; decide +kernel
  inner := by decide +kernel

theorem witConeCheck_rot (x : ℕ → ℤ) :
    ResourceAccounting.Gates.val witConeCheck.dag x 73 = ResourceAccounting.Gates.val witConeCheck.dag x 74 := by
  have h67 : ResourceAccounting.Gates.val witConeCheck.dag x 67 = 0 := witPre_val67 witConeCheckRest x
  have h72 : ResourceAccounting.Gates.val witConeCheck.dag x 72 = 0 := by
    rw [val_of_getElem? witConeCheck.dag x 72 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  rw [val_of_getElem? witConeCheck.dag x 73 (g := witDotOf 72 witRotB true) rfl (by decide +kernel),
    val_of_getElem? witConeCheck.dag x 74 (g := witDotOf 67 witRotB false) rfl (by decide +kernel)]
  simp only [witDotOf, witGateR, List.map_cons, h67, h72]

theorem witConeCheck_productOf :
    ProductOf witShape2 witConeCheck witConeCheckCls 68 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx hH => ?_⟩
  have h75 := hH 75 (by decide +kernel) (by decide +kernel) (by decide +kernel)
  rw [val_of_getElem? witConeCheck.dag x 75
    (g := witGateR (.budgeted .compareSelect witOkCross) [68, 70, 69, 71, 73, 74] 0 1 true) rfl
    (by decide +kernel)] at h75
  change witOkCross [ResourceAccounting.Gates.val witConeCheck.dag x 68, ResourceAccounting.Gates.val witConeCheck.dag x 70,
    ResourceAccounting.Gates.val witConeCheck.dag x 69, ResourceAccounting.Gates.val witConeCheck.dag x 71,
    ResourceAccounting.Gates.val witConeCheck.dag x 73, ResourceAccounting.Gates.val witConeCheck.dag x 74] = 1 at h75
  rw [(witOkCross_eq h75 (witConeCheck_rot x)).1]
  exact witPre_dot witConeCheckRest x hx (i := 70) (h := 67) (bs := List.range' 33 32) (c := false) rfl
    (by decide +kernel) (witPre_val67 witConeCheckRest x) (by decide +kernel)

/-- **A check reading another unit's computed word fails `ValueChecked`**: `ok₀` reads unit 1's computed `z₁`, which
unit 0's proof sees as whatever unit 1's prover committed, so with unit 1 unsampled and `z₁ ≠ Σ₁′`, `ok₀ = 1` with any
record; only the cone clause refuses it. -/
theorem witConeCheck_notValueChecked (L : Shape) (dq : List ℤ → ℤ) :
    ¬ ValueChecked L dq witConeCheck witConeCheckCls := by
  rintro ⟨_, _, -, -, -, -, -, hc⟩
  exact absurd (hc 75 (by decide +kernel) (.inl (by decide +kernel)) 73
    (Relation.ReflTransGen.single ⟨by unfold Declared.Reads; decide +kernel, by decide +kernel, by decide +kernel,
      by decide +kernel⟩)) (by decide +kernel)

theorem witConeCheck_notCreditTie (dq : List ℤ → ℤ) :
    ¬ CreditTie witShape2 .zTree dq witConeCheck witConeCheckCls := fun h =>
  witConeCheck_notValueChecked witShape2 dq h.2

def witConeZRest : ResourceAccounting.Gates.IntDag ℕ :=
  [witDotOf 67 (List.range' 33 32) false, witGateR (.const 0) [] 0 0 false, witDotOf 69 witRotB true,
    witDotOf 67 witRotB false,
    witGateR (.affine 0 [1, 1, -1]) [68, 70, 71] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true,
    witDqOf 72 true, witDqOf 70 true, witRcpt 68, witRcpt 70]

def witConeZ : Declared ℕ where
  dag := witPre ++ witConeZRest
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if 75 ≤ i then 1 else 0
  seedNames := fun _ _ => false
  units := 2
  unit := fun i => if i = 69 ∨ i = 70 ∨ i = 74 ∨ i = 76 then 1 else 0

def witConeZCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 70 ∨ i = 72 then .outZ
  else if i = 73 ∨ i = 74 then .outY
  else if i = 75 ∨ i = 76 then .receipt
  else .inner

theorem witTieConeZ : Tie witShape2 .zTree witConeZ witConeZCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witConeZ.dag.length → (witConeZCls i = .act ∨ witConeZCls i = .served) →
        witConeZ.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun i hi hr => by
    rcases (by decide +kernel : ∀ i, i < witConeZ.dag.length → witConeZCls i = .receipt → i = 75 ∨ i = 76)
      i hi hr with rfl | rfl
    · exact ⟨72, by decide +kernel, by decide +kernel, by decide +kernel⟩
    · exact ⟨70, by decide +kernel, by decide +kernel, by decide +kernel⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := by unfold Declared.Reads; decide +kernel
  inner := by decide +kernel

theorem witConeZ_borrow (x : ℕ → ℤ) :
    ResourceAccounting.Gates.val witConeZ.dag x 72 = ResourceAccounting.Gates.val witConeZ.dag x 68 := by
  have h67 : ResourceAccounting.Gates.val witConeZ.dag x 67 = 0 := witPre_val67 witConeZRest x
  have h69 : ResourceAccounting.Gates.val witConeZ.dag x 69 = 0 := by
    rw [val_of_getElem? witConeZ.dag x 69 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  have hr : ResourceAccounting.Gates.val witConeZ.dag x 70 = ResourceAccounting.Gates.val witConeZ.dag x 71 := by
    rw [val_of_getElem? witConeZ.dag x 70 (g := witDotOf 69 witRotB true) rfl (by decide +kernel),
      val_of_getElem? witConeZ.dag x 71 (g := witDotOf 67 witRotB false) rfl (by decide +kernel)]
    simp only [witDotOf, witGateR, List.map_cons, h67, h69]
  rw [val_of_getElem? witConeZ.dag x 72
    (g := witGateR (.affine 0 [1, 1, -1]) [68, 70, 71] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true) rfl
    (by decide +kernel)]
  change (0 : ℤ) + (List.zipWith (· * ·) [1, 1, -1] [ResourceAccounting.Gates.val witConeZ.dag x 68,
    ResourceAccounting.Gates.val witConeZ.dag x 70, ResourceAccounting.Gates.val witConeZ.dag x 71]).sum = _
  rw [hr]
  simp only [List.zipWith_cons_cons, List.zipWith_nil_left, List.sum_cons, List.sum_nil]
  omega

theorem witConeZ_productOf :
    ProductOf witShape2 witConeZ witConeZCls 72 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx _ => ?_⟩
  rw [witConeZ_borrow x]
  exact witPre_dot witConeZRest x hx (i := 68) (h := 67) (bs := List.range' 33 32) (c := false) rfl
    (by decide +kernel) (witPre_val67 witConeZRest x) (by decide +kernel)

/-- **A computed `z` reading another unit's computed word fails `ValueChecked`**: unit 0's
`z₀ = Σ₀ + (z₁ − Σ₁′)` is its dot on every input under `Gates.val` (`witConeZ_productOf`), but its proof sees `z₁` as
whatever unit 1's prover committed, so with unit 1 unsampled `z₀` is any value. The window is tied (`witTieConeZ`). -/
theorem witConeZ_notValueChecked (L : Shape) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq witConeZ witConeZCls := by
  rintro ⟨_, _, -, -, -, -, -, hc⟩
  exact absurd (hc 72 (by decide +kernel) (.inr (.inl (by decide +kernel))) 70
    (Relation.ReflTransGen.single ⟨by unfold Declared.Reads; decide +kernel, by decide +kernel, by decide +kernel,
      by decide +kernel⟩)) (by decide +kernel)

theorem witConeZ_notCreditTie (dq : List ℤ → ℤ) : ¬ CreditTie witShape2 .zTree dq witConeZ witConeZCls := fun h =>
  witConeZ_notValueChecked witShape2 dq h.2

theorem binding_input {ι : Type} {Cd : Declared ι} {i : ℕ} {b : ResourceAccounting.Gates.Binding}
    (hb : Cd.binding i = some b) : ∃ g s, Cd.dag[i]? = some g ∧ g.op = .input b s := by
  unfold Declared.binding at hb
  obtain ⟨g, hg, hb⟩ := Option.bind_eq_some_iff.1 hb
  refine ⟨g, ?_⟩
  revert hb
  cases hop : g.op with
  | input b' s => intro hb; cases hb; exact ⟨s, hg, rfl⟩
  | _ => intro hb; cases hb

theorem held_complete {ι : Type} {L : Shape} {F : Form} {dq : List ℤ → ℤ} {Cd : Declared ι} {cls : ℕ → Cls}
    (h : CreditTie L F dq Cd cls) (x : ι → ℤ) (hx : ResourceAccounting.Gates.Typed Cd.dag x) :
    ∃ x', ResourceAccounting.Gates.Typed Cd.dag x' ∧ (∀ u, Held Cd cls u x') ∧
      ∀ i, i < Cd.dag.length → (cls i = .act ∨ cls i = .registered ∨ cls i = .pub ∨ cls i = .coin) →
        ResourceAccounting.Gates.val Cd.dag x' i = ResourceAccounting.Gates.val Cd.dag x i := by
  obtain ⟨T, _, _, -, -, -, -, hm, -⟩ := h
  obtain ⟨x', hx', hH, hag⟩ := hm x hx
  refine ⟨x', hx', hH, fun i hi hc => ?_⟩
  have hb : ∃ b, Cd.binding i = some b := by
    rcases hc with hc | hc | hc | hc
    · exact ⟨_, (T.act i hi (.inl hc)).1⟩
    · exact ⟨_, T.registered i hi hc⟩
    · exact ⟨_, (T.pub i hi hc).1⟩
    · exact ⟨_, (T.coin i hi).1 hc⟩
  obtain ⟨b, hb⟩ := hb
  obtain ⟨g, s, hg, hop⟩ := binding_input hb
  have hZ : cls i ≠ .outZ := by rcases hc with hc | hc | hc | hc <;> rw [hc] <;> decide
  have hY : cls i ≠ .outY := by rcases hc with hc | hc | hc | hc <;> rw [hc] <;> decide
  rw [val_input Cd.dag x' hg hop, val_input Cd.dag x hg hop]
  exact hag i g b s hg hop hZ hY

def witKTwiceRest : ResourceAccounting.Gates.IntDag ℕ :=
  [witRecZ 68, witRecY 69, witDotOf 67 (List.range' 33 32) false, witDqOf 70 false,
    witGateR (.budgeted .compareSelect witOk) [68, 70, 69, 71, 1] 0 1 true, witGateR (.const 0) [] 0 0 false,
    witRecZ 74, witRecY 75, witDotOf 73 (List.range' 33 32) false, witDqOf 76 false,
    witGateR (.budgeted .compareSelect witOk) [74, 76, 75, 77, 1] 0 1 true, witRcpt 70, witRcpt 76]

def witKTwice : Declared ℕ where
  dag := witPre ++ witKTwiceRest
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if i = 72 ∨ i = 78 ∨ 79 ≤ i then 1 else 0
  seedNames := fun _ _ => false
  units := 2
  unit := fun i => if 73 ≤ i ∧ i ≤ 78 ∨ i = 80 then 1 else 0

def witKTwiceCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 ∨ i = 74 then .outZ
  else if i = 69 ∨ i = 75 then .outY
  else if i = 72 ∨ i = 78 then .check
  else if i = 79 ∨ i = 80 then .receipt
  else .inner

theorem witTieKTwice : Tie witShape2 .zTree witKTwice witKTwiceCls where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < witKTwice.dag.length → (witKTwiceCls i = .act ∨ witKTwiceCls i = .served) →
        witKTwice.binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun i hi hr => by
    rcases (by decide +kernel : ∀ i, i < witKTwice.dag.length → witKTwiceCls i = .receipt → i = 79 ∨ i = 80)
      i hi hr with rfl | rfl
    · exact ⟨68, by decide +kernel, by decide +kernel, by decide +kernel⟩
    · exact ⟨74, by decide +kernel, by decide +kernel, by decide +kernel⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := by unfold Declared.Reads; decide +kernel
  inner := by decide +kernel

theorem witKTwice_cone : ∀ w, w < witKTwice.dag.length →
    (witKTwiceCls w = .check ∨ witKTwiceCls w = .outZ ∨ witKTwiceCls w = .outY) →
      ∀ g, ComputedFeeds witKTwice g w → witKTwice.unit g = witKTwice.unit w :=
  fun _ _ _ _ hf => computedFeeds_unit (by unfold Declared.Reads; decide +kernel) hf

theorem witKTwice_same (x : ℕ → ℤ) (h0 : Held witKTwice witKTwiceCls (witKTwice.unit 68) x)
    (h1 : Held witKTwice witKTwiceCls (witKTwice.unit 74) x) :
    ResourceAccounting.Gates.val witKTwice.dag x 68 = ResourceAccounting.Gates.val witKTwice.dag x 74 := by
  have e0 := h0 72 (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e1 := h1 78 (by decide +kernel) (by decide +kernel) (by decide +kernel)
  rw [val_of_getElem? witKTwice.dag x 72
    (g := witGateR (.budgeted .compareSelect witOk) [68, 70, 69, 71, 1] 0 1 true) rfl (by decide +kernel)] at e0
  rw [val_of_getElem? witKTwice.dag x 78
    (g := witGateR (.budgeted .compareSelect witOk) [74, 76, 75, 77, 1] 0 1 true) rfl (by decide +kernel)] at e1
  change witOk [ResourceAccounting.Gates.val witKTwice.dag x 68, ResourceAccounting.Gates.val witKTwice.dag x 70,
    ResourceAccounting.Gates.val witKTwice.dag x 69, ResourceAccounting.Gates.val witKTwice.dag x 71,
    ResourceAccounting.Gates.val witKTwice.dag x 1] = 1 at e0
  change witOk [ResourceAccounting.Gates.val witKTwice.dag x 74, ResourceAccounting.Gates.val witKTwice.dag x 76,
    ResourceAccounting.Gates.val witKTwice.dag x 75, ResourceAccounting.Gates.val witKTwice.dag x 77,
    ResourceAccounting.Gates.val witKTwice.dag x 1] = 1 at e1
  have h67 : ResourceAccounting.Gates.val witKTwice.dag x 67 = 0 := witPre_val67 witKTwiceRest x
  have h73 : ResourceAccounting.Gates.val witKTwice.dag x 73 = 0 := by
    rw [val_of_getElem? witKTwice.dag x 73 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  have hd : ResourceAccounting.Gates.val witKTwice.dag x 70 = ResourceAccounting.Gates.val witKTwice.dag x 76 := by
    rw [val_of_getElem? witKTwice.dag x 70 (g := witDotOf 67 (List.range' 33 32) false) rfl (by decide +kernel),
      val_of_getElem? witKTwice.dag x 76 (g := witDotOf 73 (List.range' 33 32) false) rfl (by decide +kernel)]
    simp only [witDotOf, witGateR, List.map_cons, h67, h73]
  exact (witOk_eq e0).1.trans (hd.trans (witOk_eq e1).1.symm)

/-- **One product doesn't value-check two checked units**: where both units' checks
hold, their records' `z` have one value (`witKTwice_same`), so `witKTwice`, tied (`witTieKTwice`) and within the cone
clause (`witKTwice_cone`), fails `ValueChecked` at any shape and dequant. -/
theorem witKTwice_notValueChecked (L : Shape) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq witKTwice witKTwiceCls := by
  rintro ⟨_, _, -, hinj, -⟩
  exact absurd (hinj 68 74 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    fun x _ h0 h1 => witKTwice_same x h0 h1) (by decide)

theorem witKTwice_notCreditTie (dq : List ℤ → ℤ) : ¬ CreditTie witShape2 .zTree dq witKTwice witKTwiceCls :=
  fun h => witKTwice_notValueChecked witShape2 dq h.2

def witKRest (mid : ResourceAccounting.Gates.IntDag ℕ) : ResourceAccounting.Gates.IntDag ℕ :=
  [witGateR (.input .run 68) [] 0 255 false, witRecZ 69, witRecY 70, witDotGate] ++ mid ++
    [witGateR (.budgeted .compareSelect witOk) [69, 71, 70, 71 + mid.length, 1] 0 1 true] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' (73 + mid.length) 32 ++ List.range' 33 32))
        ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max false,
      witGateR (.affine 0 [1, 1]) [105 + mid.length, 68] ResourceAccounting.Gates.s32Min ResourceAccounting.Gates.s32Max true]

def witK (mid : ResourceAccounting.Gates.IntDag ℕ) : Declared ℕ where
  dag := witPre ++ witKRest mid
  calls := 1
  call := fun _ => 0
  rounds := 1
  round := fun _ => 0
  coinMode := fun _ => .live
  commitRound := fun i => if 72 + mid.length ≤ i then 1 else 0
  seedNames := fun _ _ => false
  units := 1
  unit := fun _ => 0

def witKCls (n i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 then .pub
  else if i = 69 then .outZ
  else if i = 70 then .outY
  else if i = 72 + n then .check
  else if i = 106 + n then .receipt
  else .inner

/-- A `witK` window whose check holds at a typed input `x` where the record's `y` isn't `witDq` of its `z` at the
call's scales fails `ValueChecked`: `y` is its one `outY` word, `z` its one `outZ` word, and the scales can only be
`s_a` (65) and `s_w` (66), its one `act` and one `registered` word not declared s8. -/
theorem witK_notValueChecked (mid : ResourceAccounting.Gates.IntDag ℕ) (L : Shape) (x : ℕ → ℤ)
    (hx : ResourceAccounting.Gates.Typed (witK mid).dag x)
    (h70 : 70 < (witK mid).dag.length ∧ witKCls mid.length 70 = .outY)
    (hZ : ∀ z, z < (witK mid).dag.length → witKCls mid.length z = .outZ → z = 69)
    (hA : ∀ s, s < (witK mid).dag.length → witKCls mid.length s = .act → ¬ S8 (witK mid) s → s = 65)
    (hR : ∀ s, s < (witK mid).dag.length → witKCls mid.length s = .registered → ¬ S8 (witK mid) s → s = 66)
    (hchk : ∀ k, k < (witK mid).dag.length → witKCls mid.length k = .check → k = 72 + mid.length)
    (hok : ResourceAccounting.Gates.val (witK mid).dag x (72 + mid.length) = 1)
    (hv : ResourceAccounting.Gates.val (witK mid).dag x 70 ≠ witDq [ResourceAccounting.Gates.val (witK mid).dag x 69,
      ResourceAccounting.Gates.val (witK mid).dag x 65, ResourceAccounting.Gates.val (witK mid).dag x 66]) :
    ¬ ValueChecked L witDq (witK mid) (witKCls mid.length) := by
  rintro ⟨_, src, -, -, hD, -⟩
  obtain ⟨hz, hzo, -, ⟨hsa, hsac, -, hsa8, hsw, hswc, hsw8, hv'⟩⟩ := hD 70 h70.1 h70.2
  rw [hZ _ hz hzo, hA _ hsa hsac hsa8, hR _ hsw hswc hsw8] at hv'
  exact hv (hv' x hx fun k hk hc _ => by rw [hchk k hk hc]; exact hok)

def witMidWS : ResourceAccounting.Gates.IntDag ℕ :=
  [witGateR (.budgeted .rounding witDq) [71, 1, 33] (-2 ^ 63) (2 ^ 63 - 1) false]

def witMidMix : ResourceAccounting.Gates.IntDag ℕ :=
  [witDotOf 67 witRotB false, witDqOf 72 false,
    witGateR (.affine 0 [0, 1]) [71, 73] (-2 ^ 63) (2 ^ 63 - 1) false]

def witXK (z y : ℤ) (s : ℕ) : ℤ := if s = 65 then 3 else if s = 69 then z else if s = 70 then y else 2

def witXM (s : ℕ) : ℤ := if s = 1 ∨ s = 33 ∨ s = 65 ∨ s = 66 ∨ s = 69 then 1 else 0

theorem witTieKWS : Tie witShape .zTree (witK witMidWS) (witKCls 1) where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < (witK witMidWS).dag.length → (witKCls 1 i = .act ∨ witKCls 1 i = .served) →
        (witK witMidWS).binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨69, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

/-- **A record `y` checked to `dq` at an A and a B word fails `ValueChecked`**: at
`witXK 128 512` the check holds, with `z = Σ = 128` and `y = dq(128, A_1, B_1) = 512`, which isn't
`dq(128, 3, 2) = 768`. -/
theorem witKWS_notValueChecked (L : Shape) : ¬ ValueChecked L witDq (witK witMidWS) (witKCls 1) :=
  witK_notValueChecked witMidWS L (witXK 128 512) (typed_of_all (witXK 128 512) (by decide +kernel))
    ⟨by decide +kernel, by decide +kernel⟩ (by decide +kernel) (by unfold S8; decide +kernel)
    (by unfold S8; decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witKWS_notCreditTie : ¬ CreditTie witShape .zTree witDq (witK witMidWS) (witKCls 1) := fun h =>
  witKWS_notValueChecked witShape h.2

theorem witTieKRaw : Tie witShape .zTree (witK []) (witKCls 0) where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < (witK []).dag.length → (witKCls 0 i = .act ∨ witKCls 0 i = .served) →
        (witK []).binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨69, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

/-- **A record `y` checked to the raw product fails `ValueChecked`**: at
`witXK 128 128` the check holds, with `z = y = Σ = 128`, which isn't `dq(128, 3, 2) = 768`. -/
theorem witKRaw_notValueChecked (L : Shape) : ¬ ValueChecked L witDq (witK []) (witKCls 0) :=
  witK_notValueChecked [] L (witXK 128 128) (typed_of_all (witXK 128 128) (by decide +kernel))
    ⟨by decide +kernel, by decide +kernel⟩ (by decide +kernel) (by unfold S8; decide +kernel)
    (by unfold S8; decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witKRaw_notCreditTie : ¬ CreditTie witShape .zTree witDq (witK []) (witKCls 0) := fun h =>
  witKRaw_notValueChecked witShape h.2

theorem witTieKMix : Tie witShape .zTree (witK witMidMix) (witKCls 3) where
  calls := by decide +kernel
  rounds := by decide +kernel
  round := fun c hc => by
    obtain rfl : c = 0 := by
      have h1 : c < 1 := hc
      omega
    decide +kernel
  inRange := fun _ _ => Nat.zero_lt_one
  live := fun _ _ => rfl
  coin := by decide +kernel
  registered := by decide +kernel
  act := fun i hi h =>
    ⟨(by decide +kernel : ∀ i, i < (witK witMidMix).dag.length → (witKCls 3 i = .act ∨ witKCls 3 i = .served) →
        (witK witMidMix).binding i = some .run) i hi h, fun _ _ _ => rfl⟩
  pub := by unfold Declared.Reads; decide +kernel
  computed := by decide +kernel
  outWords := by unfold Declared.Reads; decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨69, by decide +kernel, by decide +kernel, rfl⟩
  commit := tie_commit (by decide +kernel)
  receipt := by decide +kernel
  checked := by decide +kernel
  readers := fun _ _ _ _ _ _ => rfl
  inner := by decide +kernel

/-- **A record `y` checked to another product's dequant fails `ValueChecked`**: at
`witXM` the check holds, with `z = Σ = A_1·B_1 = 1` and `y = 0·Σ + 1·dq(p) = 0`, the rotated product `p` being `0`
there, which isn't `dq(1, 1, 1) = 1`. -/
theorem witKMix_notValueChecked (L : Shape) : ¬ ValueChecked L witDq (witK witMidMix) (witKCls 3) :=
  witK_notValueChecked witMidMix L witXM (typed_of_all witXM (by decide +kernel))
    ⟨by decide +kernel, by decide +kernel⟩ (by decide +kernel) (by unfold S8; decide +kernel)
    (by unfold S8; decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witKMix_notCreditTie : ¬ CreditTie witShape .zTree witDq (witK witMidMix) (witKCls 3) := fun h =>
  witKMix_notValueChecked witShape h.2

end Properties.Served
