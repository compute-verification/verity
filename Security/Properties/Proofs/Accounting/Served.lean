import Mathlib.Data.List.GetD
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum.Basic
import Properties.Accounting.Served

/-!
# Proofs of the served tie's statements (`Accounting.Served`, `Properties.Served`)

`TieCommits` goes class by class: a registered value's commit coin is `0`; an A row's or a `y` row's is its round,
which is its first reader's, since only its own call reads it (`firstRead_cases`), or below the draw when nothing does
(`round_lt`); a `zTree` `z` word's is its round, the floor of its cut; a receipt's or `perCall`'s
`y` word's is the round after its own, which its coin read reaches (`Tie.noised`); an inner value's is its cut.
`TieLive`: a unit with a word has a first gate, its round is a round of the window, and every one is live.
`SeededScheduleNotLive`: `UnitsOfLayout` puts a read run input at the draw, `LayerCommitted` at its first reader's
round, which `Wf` puts below it. `TieScope` is `Scope` from `TieLive`, `TieCommits` and its three hypotheses, and
`epochScope` each window's of an `EpochTie`. `spentIn_split` is what a window timed from its own start after the salt's
release leaves uncounted. `Llama31Shape` and `Llama31Rounds` are kernel computations. `witTie`: `Tie` holds of a circuit
with gates (one call, its coin, an A row and a receipt reading both), so `TieCommits` isn't true only of the empty
circuit; `witEpoch` and `witEpoch_ownStart` tie two windows of it timed from the salt's release, and refuse them when
the second is timed from its own start. `witTieCall`: `Tie` holds in `perCall` of a circuit whose served `y` reads its
round's coin and is committed the round after (`witCall_out_cut`), and `witCall_unnoised` refuses it when `y` reads no
coin, as `h1tile`'s `z` doesn't. `witCreditTie`: an honest `zTree` window meets `CreditTie`. Its one unit computes the
served `z`, an s8 dot over `K = 32` of the committed A row and the registered B column (each word declared s8),
beside the noised product its receipt folds (`witZ_beside`: `z` reads no coin, the receipt does), and
`y = dq(z, s_a, s_w)`; `z`'s value is the dot on every typed input (`witZ_productOf`, its 32 products of s8 words
within s32, so the clamp is the identity) and `y`'s is `dq` of it (`witZ_dequantOf`). `witCallCredit`: an honest
`perCall` window in CMP-65's order meets `CreditTie`. Its `z` is peeled from the noised product,
`m − coin·ΣB_k` with `m = Σ (A_k + coin)·B_k`, so its cone holds the coin (`witP_peeled`), and its value is the dot
all the same (`witP_productOf`: `m` is `Σ A_k·B_k + coin·ΣB_k` within s32, `sum_zipWith_add`). `wit_notValueChecked`:
the `beside` witness, whose unit has no `out` word, fails `ValueChecked` at every `dq`.
-/

namespace Properties.Served

open Accounting.Compose Accounting.Served

theorem llama31Shape : Llama31Shape := by
  unfold Llama31Shape
  refine ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, ?_⟩
  unfold Layout.Wf
  decide +kernel

theorem stepOf_lt {L : Layout} {c : ℕ} (hc : c < L.steps * L.calls) : L.stepOf c < L.steps := by
  unfold Layout.stepOf
  exact Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hc)

/-- A call site's round is a round of its form. -/
theorem roundOf_lt {L : Layout} (F : Form) {c : ℕ} (hc : c < L.steps * L.calls) : F.roundOf L c < F.rounds L := by
  cases F <;> first | exact stepOf_lt hc | exact hc

/-- An `out` word's commit coin is its round's, except in `perCall`. -/
theorem commitCoin_out {F : Form} (hF : F ≠ .perCall) (r : ℕ) : commitCoin F .out r = r := by
  cases F <;> first | rfl | exact (hF rfl).elim

/-- A tied circuit's gate is in a round of the window. -/
theorem round_lt {ι : Type} {L : Layout} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls} (T : Tie L F Cd cls) (i : ℕ)
    (hi : i < Cd.dag.length) : Cd.round (Cd.call i) < Cd.rounds := by
  have hc := T.inRange i hi
  have hc' : Cd.call i < L.steps * L.calls := by rw [← T.calls]; exact hc
  rw [T.round _ hc, T.rounds]
  exact roundOf_lt F hc'

/-- A value's first reader: none (the draw), or a gate that reads it. -/
theorem firstRead_cases {ι : Type} (Cd : Declared ι) (i : ℕ) :
    Cd.firstRead i = Cd.rounds ∨
      ∃ j, j < Cd.dag.length ∧ Cd.Reads j i ∧ Cd.firstRead i = Cd.round (Cd.call j) := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact Or.inl rfl
  | some j =>
    have hp := List.find?_some h
    exact Or.inr ⟨j, List.mem_range.1 (List.mem_of_find?_eq_some h), hp, rfl⟩

/-- A value some gate reads has a first reader, a gate that reads it. -/
theorem firstRead_of_reads {ι : Type} (Cd : Declared ι) {i j : ℕ} (hj : j < Cd.dag.length) (hR : Cd.Reads j i) :
    ∃ j', j' < Cd.dag.length ∧ Cd.Reads j' i ∧ Cd.firstRead i = Cd.round (Cd.call j') := by
  unfold Declared.firstRead
  cases h : List.find? (fun j => (Cd.dag[j]?).any fun g => g.args.contains i) (List.range Cd.dag.length) with
  | none => exact absurd hR (List.find?_eq_none.1 h j (List.mem_range.2 hj))
  | some j' =>
    have hp := List.find?_some h
    exact ⟨j', List.mem_range.1 (List.mem_of_find?_eq_some h), hp, rfl⟩

/-- A run input only its own call reads is cut no earlier than its call's round. -/
theorem round_le_cut_of_run {ι : Type} {L : Layout} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls}
    (T : Tie L F Cd cls) {ℓ : ℕ} (hlt : ℓ < Cd.dag.length) (hb : Cd.binding ℓ = some .run)
    (hr : ∀ j, j < Cd.dag.length → Cd.Reads j ℓ → Cd.call j = Cd.call ℓ) :
    Cd.round (Cd.call ℓ) ≤ Cd.cutOf ℓ := by
  simp only [Declared.cutOf, hb]
  rcases firstRead_cases Cd ℓ with h | ⟨j, hj, hR, h⟩
  · rw [h]
    exact (round_lt T ℓ hlt).le
  · rw [h]
    exact le_of_eq (congrArg Cd.round (hr j hj hR)).symm

/-- A gate that is no input is cut no earlier than its call's round. -/
theorem round_le_cut_of_none {ι : Type} {Cd : Declared ι} {ℓ : ℕ} (hb : Cd.binding ℓ = none) :
    Cd.round (Cd.call ℓ) ≤ Cd.cutOf ℓ := by
  simp only [Declared.cutOf, hb]
  exact le_max_left _ _

theorem tieCommits : TieCommits := by
  intro ι L F Cd cls T ℓ hW
  obtain ⟨hlt, hnc⟩ := hW
  have hround := T.round _ (T.inRange ℓ hlt)
  cases hk : cls ℓ with
  | coin => exact absurd ((T.coin ℓ hlt).1 hk) hnc
  | inner => exact le_of_eq (T.inner ℓ hlt hk)
  | registered =>
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk]
    simp [commitCoin]
  | act =>
    obtain ⟨hb, hr⟩ := T.act ℓ hlt (Or.inl hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_run T hlt hb hr
  | served =>
    obtain ⟨hb, hr⟩ := T.act ℓ hlt (Or.inr hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_run T hlt hb hr
  | out =>
    have hb := T.computed ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    by_cases hF : F = .perCall
    · subst hF
      have hn := T.noised ℓ hlt (Or.inr ⟨hk, rfl⟩)
      show Cd.round (Cd.call ℓ) + 1 ≤ Cd.cutOf ℓ
      simp only [Declared.cutOf, hb]
      exact le_max_of_le_right hn
    · rw [commitCoin_out hF]
      exact round_le_cut_of_none hb
  | receipt =>
    have hb := T.computed ℓ hlt (by simp [hk])
    have hn := T.noised ℓ hlt (Or.inl hk)
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    show Cd.round (Cd.call ℓ) + 1 ≤ Cd.cutOf ℓ
    simp only [Declared.cutOf, hb]
    exact le_max_of_le_right hn

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

/-- Each window of a tied epoch is in Compose's scope. -/
theorem epochScope {ι Q R S K : Type} {L : Layout} {F : Form} {W : ℕ} {Cd : ℕ → Declared ι} {cls : ℕ → ℕ → Cls}
    {release : ℝ} {run : ℕ → Pouw.TimedRun Q R S} (E : EpochTie L F W Cd cls release run) (Sc : Scheme K) (δs : ℝ)
    {w : ℕ} (hw : w < W) (hWf : (Cd w).Wf) (hδ : 0 < δs) (hL : Sc.LuckProved) : Scope Sc (Cd w) δs :=
  tieScope ι K Sc L F (Cd w) (cls w) δs (E.tie w hw) hWf hδ hL

/-- What a run spends in `(a, c]` is what it spends in `(a, b]` and then in `(b, c]`: a window timed from its own start
`b` after the salt's release `a` leaves its spending in `(a, b]` uncounted. -/
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
  have h2 := hLC ℓ hW
  have h3 : Cd.cutOf ℓ = Cd.firstRead ℓ := by simp [Declared.cutOf, hb]
  obtain ⟨j', hj', _, h⟩ := firstRead_of_reads Cd hj hR
  have hr := hW2 _ (hW1 j' hj').1
  omega

/-! ## A tied circuit with gates -/

/-- One step of one linear call, one 4 × 8 unit. -/
def witLayout : Layout where
  steps := 1
  rows := 4
  tileRows := 4
  tileCols := 8
  linears := [(32, 8)]

/-- A gate of `op` on `args`, committed or not, at no site. -/
def witGate (op : Accounting.Gates.Op Unit) (args : List ℕ) (commit : Bool) : Accounting.Gates.Gate Unit :=
  { op := op, args := args, lo := 0, hi := 0, step := 0, prim := "", site := .other, commit := commit }

/-- `witLayout`'s one call in one live round: its coin, an A row (a run input), and a committed receipt word reading
both, committed before the draw, the round after its own. -/
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

/-- `witCircuit`'s classes: the coin, the A row, the receipt. -/
def witCls : ℕ → Cls
  | 0 => .coin
  | 1 => .act
  | _ => .receipt

/-- `Tie` holds of a circuit that writes values: the A row its own call reads, the receipt reading the round's coin. -/
theorem witTie : Tie witLayout .beside witCircuit witCls where
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
  computed := by decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => trivial
  commit := by decide +kernel
  inner := by decide +kernel

/-- The witness's receipt is committed at coin `1`, its cut, the round after the coin it reads. -/
theorem wit_receipt_cut : witCircuit.Writes 2 ∧ witCircuit.commitRound 2 = 1 ∧ witCircuit.cutOf 2 = 1 :=
  ⟨⟨by decide +kernel, by decide +kernel⟩, by decide +kernel, by decide +kernel⟩

theorem wit_layerCommitted : LayerCommitted witCircuit :=
  tieCommits Unit witLayout .beside witCircuit witCls witTie

/-- `witLayout`'s one call in `perCall`: its coin, an A row, its served `y` word off the noised sets (reading the coin
and the A row), committed the round after its own, and a receipt reading `y`. -/
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

/-- `witCircuitCall`'s classes: the coin, the A row, the `y` word, the receipt. -/
def witClsCall : ℕ → Cls
  | 0 => .coin
  | 1 => .act
  | 2 => .out
  | _ => .receipt

/-- `Tie` holds in `perCall` of a circuit whose served `y` reads its round's coin and is committed the round after. -/
theorem witTieCall : Tie witLayout .perCall witCircuitCall witClsCall where
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
  computed := by decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨2, by decide +kernel, rfl, rfl⟩
  commit := by decide +kernel
  inner := by decide +kernel

/-- The `perCall` witness's `y` is committed at coin `1`, its cut, after the coin it reads. -/
theorem witCall_out_cut :
    witCircuitCall.Writes 2 ∧ witCircuitCall.commitRound 2 = 1 ∧ witCircuitCall.cutOf 2 = 1 :=
  ⟨⟨by decide +kernel, by decide +kernel⟩, by decide +kernel, by decide +kernel⟩

theorem witCall_layerCommitted : LayerCommitted witCircuitCall :=
  tieCommits Unit witLayout .perCall witCircuitCall witClsCall witTieCall

/-- The `perCall` witness with its `y` the unit's own dot of the A row, reading no coin (`h1tile`'s `z`). -/
def witCircuitUnnoised : Declared Unit :=
  { witCircuitCall with
    dag := [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
      witGate (.affine 0 [1]) [1] true, witGate (.affine 0 [1]) [2] true] }

/-- **`perCall` needs the served `y` to read its round's coin** (G3's P7): a `y` that reads none fails `Tie.noised`,
whatever its commit coins. -/
theorem witCall_unnoised : ¬ Tie witLayout .perCall witCircuitUnnoised witClsCall := fun T =>
  absurd (T.noised 2 (by decide +kernel) (Or.inr ⟨rfl, rfl⟩)) (by decide +kernel)

/-- A timed run from `t` that spends nothing. -/
def witRunAt (t : ℝ) : Pouw.TimedRun Unit Unit Unit := ⟨t, fun _ _ => 0, fun _ _ _ => 0⟩

/-- Two windows of the witness, both timed from the salt's release `0`, are a tied epoch. -/
theorem witEpoch : EpochTie witLayout .beside 2 (fun _ => witCircuit) (fun _ => witCls) 0 (fun _ => witRunAt 0) :=
  ⟨fun _ _ => witTie, fun _ _ => rfl⟩

/-- The same two windows with the second timed from its own start `1` are not. -/
theorem witEpoch_ownStart : ¬ EpochTie witLayout .beside 2 (fun _ => witCircuit) (fun _ => witCls) 0
    (fun w => witRunAt (if w = 0 then 0 else 1)) := by
  intro E
  have h := E.salted 1 (by decide)
  exact one_ne_zero h

/-- The `beside` witness's unit has no `out` word, so its receipt has no value-checked product: it fails
`ValueChecked` whatever the dequant. -/
theorem wit_notValueChecked (dq : List ℤ → ℤ) : ¬ ValueChecked witLayout dq witCircuit witCls := by
  rintro ⟨h, _⟩
  obtain ⟨z, _, _, _, _, hz, _, _, _, hzo, _⟩ := h 2 (by decide +kernel) rfl
  exact (by decide +kernel : ∀ z, z < witCircuit.dag.length → witCls z ≠ .out) z hz hzo

/-! ## A gate's value, from its operands' -/

section Values

open Accounting.Gates

theorem foldl_values_length {ι : Type} (C : Dag ι) (x : ι → ℤ) (vs : List ℤ) :
    (C.foldl (fun vs g => vs ++ [g.value x vs]) vs).length = vs.length + C.length := by
  induction C generalizing vs with
  | nil => simp
  | cons g C ih => simp [List.foldl_cons, ih]; omega

theorem values_length {ι : Type} (C : Dag ι) (x : ι → ℤ) : (values C x).length = C.length := by
  simp [values, foldl_values_length]

theorem values_take_succ {ι : Type} (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    values (C.take (i + 1)) x = values (C.take i) x ++ [C[i].value x (values (C.take i) x)] := by
  rw [List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some]
  simp only [values, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem values_take_length {ι : Type} (C : Dag ι) (x : ι → ℤ) (n : ℕ) (h : n ≤ C.length) :
    (values (C.take n) x).length = n := by
  rw [values_length, List.length_take]; omega

theorem getD_values_take {ι : Type} (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
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

theorem val_eq {ι : Type} (C : Dag ι) (x : ι → ℤ) (i : ℕ) (h : i < C.length) :
    val C x i = C[i].value x (values (C.take i) x) := by
  have := getD_values_take C x i h C.length h le_rfl
  rwa [List.take_length] at this

/-- Gate `i`'s value is its operation on its operands' values, each an earlier gate's. -/
theorem val_of_getElem? {ι : Type} (C : Dag ι) (x : ι → ℤ) (i : ℕ) {g : Gate ι} (hg : C[i]? = some g)
    (ha : ∀ j ∈ g.args, j < i) : val C x i = g.op.out (g.op.raw x (g.args.map (val C x))) := by
  obtain ⟨h, hgi⟩ := List.getElem?_eq_some_iff.1 hg
  have hm : g.args.map (fun j => (values (C.take i) x).getD j 0) = g.args.map (val C x) :=
    List.map_congr_left fun j hj => by
      have hji := ha j hj
      show (values (C.take i) x).getD j 0 = val C x j
      rw [getD_values_take C x j (by omega) i hji h.le, ← val_eq C x j (by omega)]
  rw [val_eq C x i h, hgi]
  unfold Gate.value Gate.raw Gate.operands
  rw [hm]

end Values

/-- A sum of products of words in `[−P, P]` and `[−Q, Q]` is at most `P·Q` a product in size. -/
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

/-- The s32 clamp is the identity on an s32 value. -/
theorem clampS32_of_bound {s : ℤ} (h1 : -2147483648 ≤ s) (h2 : s ≤ 2147483647) : Accounting.Gates.clampS32 s = s := by
  have hmin : Accounting.Gates.s32Min ≤ s := le_trans (by norm_num [Accounting.Gates.s32Min]) h1
  have hmax : s ≤ Accounting.Gates.s32Max := le_trans h2 (by norm_num [Accounting.Gates.s32Max])
  unfold Accounting.Gates.clampS32
  rw [min_eq_right hmax, max_eq_right hmin]

/-- A row of ones' products with `l` sum to `l`'s sum. -/
theorem sum_zipWith_one : ∀ (n : ℕ) (l : List ℤ), l.length = n →
    (List.zipWith (· * ·) (List.replicate n 1) l).sum = l.sum
  | 0, [], _ => rfl
  | n + 1, b :: l, h => by
    simp only [List.replicate_succ, List.zipWith_cons_cons, List.sum_cons, one_mul]
    rw [sum_zipWith_one n l (by simpa using h)]
  | 0, _ :: _, h => by simp at h
  | _ + 1, [], h => by simp at h

/-- Noising every word of `l1` by `c` adds `c` times `l2`'s sum to the products' sum. -/
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

/-! ## Honest windows meet `CreditTie`: `zTree` with `z` beside the noised product, `perCall` with `z` peeled from it -/

/-- A gate of `op` on `args` declared `[lo, hi]`, committed or not, at no site. -/
def witGateR (op : Accounting.Gates.Op ℕ) (args : List ℕ) (lo hi : ℤ) (commit : Bool) : Accounting.Gates.Gate ℕ :=
  { op := op, args := args, lo := lo, hi := hi, step := 0, prim := "", site := .other, commit := commit }

/-- The witnesses' dequant: `z·s_a·s_w`. -/
def witDq (v : List ℤ) : ℤ := v.headD 0 * v.getD 1 0 * v.getD 2 0

/-- The two witnesses' first gates, `witLayout`'s one call's inputs: the coin (gate 0); the A row (1–32, s8 run
inputs); the B column (33–64, s8 registered); the scales `s_a` (65, a run input) and `s_w` (66, registered); and the
constant `0` (67). -/
def witPre : Accounting.Gates.Dag ℕ :=
  [witGateR (.input .coin 0) [] 0 255 false] ++
    (List.range' 1 32).map (fun j => witGateR (.input .run j) [] (-128) 127 false) ++
    (List.range' 33 32).map (fun j => witGateR (.input .registered j) [] (-128) 127 false) ++
    [witGateR (.input .run 65) [] 0 65535 false, witGateR (.input .registered 66) [] 0 65535 false,
      witGateR (.const 0) [] 0 0 false]

theorem witPre_get (rest : Accounting.Gates.Dag ℕ) {j : ℕ} (h : j < 68) : (witPre ++ rest)[j]? = witPre[j]? :=
  List.getElem?_append_left (lt_of_lt_of_eq h (by decide +kernel))

/-- An input gate's value is its slot's, in its declared range on a typed input. -/
theorem witPre_input (rest : Accounting.Gates.Dag ℕ) (x : ℕ → ℤ) {j : ℕ} {b : Accounting.Gates.Binding} {lo hi : ℤ}
    (hg : (witPre ++ rest)[j]? = some (witGateR (.input b j) [] lo hi false))
    (hx : Accounting.Gates.Typed (witPre ++ rest) x) :
    Accounting.Gates.val (witPre ++ rest) x j = x j ∧ lo ≤ x j ∧ x j ≤ hi := by
  refine ⟨?_, hx _ (List.mem_of_getElem? hg) b j rfl⟩
  rw [val_of_getElem? _ x j hg (by simp [witGateR])]
  rfl

theorem witPre_valA (rest : Accounting.Gates.Dag ℕ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witPre ++ rest) x)
    {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    Accounting.Gates.val (witPre ++ rest) x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_input (b := .run) rest x (by rw [witPre_get rest (by omega)]; interval_cases j <;> rfl) hx

theorem witPre_valB (rest : Accounting.Gates.Dag ℕ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witPre ++ rest) x)
    {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    Accounting.Gates.val (witPre ++ rest) x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_input (b := .registered) rest x (by rw [witPre_get rest (by omega)]; interval_cases j <;> rfl) hx

theorem witPre_val67 (rest : Accounting.Gates.Dag ℕ) (x : ℕ → ℤ) : Accounting.Gates.val (witPre ++ rest) x 67 = 0 := by
  rw [val_of_getElem? _ x 67 (g := witGateR (.const 0) [] 0 0 false) (by rw [witPre_get rest (by omega)]; rfl)
    (by simp [witGateR])]
  rfl

/-- The served product `z` of the `zTree` witness: the constant `0` (gate 67), then `K = 32` products of the A row
(gates 1–32) with the B column (gates 33–64), clamped to s32. -/
def witZGate : Accounting.Gates.Gate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 1 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
    Accounting.Gates.s32Max true

/-- The `zTree` witness's served `y = dq(z, s_a, s_w)` (gates 68, 65, 66). -/
def witYGate : Accounting.Gates.Gate ℕ :=
  witGateR (.budgeted .rounding witDq) [68, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

/-- `witLayout`'s one call in `zTree`, after `witPre`: the served `z` (68) and `y` (69), committed under the round's
root before its coin; the noised operands `X_k = A_k + coin` (70–101); the noised product `m = Σ X_k·B_k` (102); and
the receipt (103), committed the round after. -/
def witZRest : Accounting.Gates.Dag ℕ :=
  [witZGate, witYGate] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
        Accounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [102] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

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

/-- `witZ`'s classes: the coin, the A row and `s_a`, the B column and `s_w`, the served `z` and `y`, the receipt, and
the rest inner. -/
def witZCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 ∨ i = 69 then .out
  else if i = 103 then .receipt
  else .inner

theorem witTieZ : Tie witLayout .zTree witZ witZCls where
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
  computed := by decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨68, by decide +kernel, by decide +kernel, rfl⟩
  commit := by decide +kernel
  inner := by decide +kernel

theorem witZ_layerCommitted : LayerCommitted witZ :=
  tieCommits ℕ witLayout .zTree witZ witZCls witTieZ

/-- The served `z` reads no coin and the receipt does: `z` is computed beside the noised product. -/
theorem witZ_beside : (coinMark witZ).getD 68 0 = 0 ∧ (coinMark witZ).getD 103 0 = 1 := by
  decide +kernel

theorem witZ_valA (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witZ.dag x) {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    Accounting.Gates.val witZ.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valA witZRest x hx h1 h2

theorem witZ_valB (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witZ.dag x) {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    Accounting.Gates.val witZ.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valB witZRest x hx h1 h2

/-- **The served `z` is the product of the A row and the B column** on every typed input: its 32 products of s8 words
stay within s32, so its clamp is the identity. -/
theorem witZ_productOf : ProductOf witLayout witZ witZCls 68 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, by unfold S8; decide +kernel, by unfold S8; decide +kernel,
    fun x hx => ?_⟩
  have h67 : Accounting.Gates.val witZ.dag x 67 = 0 := witPre_val67 witZRest x
  have hA : ∀ a ∈ (List.range' 1 32).map (Accounting.Gates.val witZ.dag x), -128 ≤ a ∧ a ≤ 128 := by
    intro a ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witZ_valA x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  have hB : ∀ b ∈ (List.range' 33 32).map (Accounting.Gates.val witZ.dag x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witZ_valB x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 128) (Q := 128) (by norm_num) (by norm_num) _ _ hA hB
  simp only [List.length_map, List.length_range'] at hb1 hb2
  rw [val_of_getElem? witZ.dag x 68 (g := witZGate) rfl (by decide +kernel)]
  simp only [witZGate, witGateR, Accounting.Gates.Op.out, Accounting.Gates.Op.raw, List.map_cons, List.headD_cons,
    List.tail_cons, h67, zero_add, List.map_append]
  unfold Accounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

/-- **The served `y` is `dq` of the served `z`** and the two scales, on every typed input. -/
theorem witZ_dequantOf : DequantOf witDq witZ witZCls 69 68 [65, 66] := by
  refine ⟨by decide +kernel, fun x _ => ?_⟩
  rw [val_of_getElem? witZ.dag x 69 (g := witYGate) rfl (by decide +kernel)]
  rfl

theorem feeds_read {ι : Type} {Cd : Declared ι} {i j : ℕ} (h : Cd.Reads j i) : Feeds Cd i j :=
  Relation.ReflTransGen.single h

theorem feeds_tail {ι : Type} {Cd : Declared ι} {i j k : ℕ} (h : Feeds Cd i j) (hr : Cd.Reads k j) :
    Feeds Cd i k :=
  Relation.ReflTransGen.tail h hr

/-- The receipt reads every A word (through its noised `X_k` and `m`) and every B word (through `m`). -/
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

theorem witZ_valueChecked : ValueChecked witLayout witDq witZ witZCls := by
  refine ⟨fun r hr hrc => ?_, fun i hi hio => ?_⟩
  · obtain rfl : r = 103 :=
      (by decide +kernel : ∀ r, r < witZ.dag.length → witZCls r = .receipt → r = 103) r hr hrc
    exact ⟨68, 69, List.range' 1 32, List.range' 33 32, [65, 66], by decide +kernel, by decide +kernel, rfl, rfl,
      by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, witZ_productOf, witZ_dequantOf,
      witZ_feeds⟩
  · rcases (by decide +kernel : ∀ i, i < witZ.dag.length → witZCls i = .out → i = 68 ∨ i = 69) i hi hio
      with rfl | rfl
    · exact Or.inl ⟨_, _, witZ_productOf⟩
    · exact Or.inr ⟨68, _, _, _, rfl, by decide +kernel, witZ_productOf, witZ_dequantOf⟩

/-- **An honest `zTree` window meets `CreditTie`**: its served product, computed beside the noised one
(`witZ_beside`), is value-checked, at one round a step. -/
theorem witCreditTie : CreditTie witLayout .zTree witDq witZ witZCls :=
  ⟨witTieZ, witZ_valueChecked⟩

/-- The `perCall` witness's noised product, `m = Σ X_k·B_k` over the noised operands (gates 68–99) and the B column
(33–64), clamped to s32. -/
def witMGate : Accounting.Gates.Gate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 68 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
    Accounting.Gates.s32Max false

/-- `Σ B_k` (gates 33–64). -/
def witSGate : Accounting.Gates.Gate ℕ :=
  witGateR (.affine 0 (List.replicate 32 1)) (List.range' 33 32) (-4096) 4064 false

/-- The noise's correction, `coin·ΣB_k` (gates 0, 101). -/
def witCGate : Accounting.Gates.Gate ℕ :=
  witGateR (.budgeted .rounding fun v => v.headD 0 * v.getD 1 0) [0, 101] (-1044480) 1036320 false

/-- The served `z`, peeled from the noised product: `m − coin·ΣB_k` (gates 100, 102). -/
def witPZGate : Accounting.Gates.Gate ℕ :=
  witGateR (.affine 0 [1, -1]) [100, 102] Accounting.Gates.s32Min Accounting.Gates.s32Max true

/-- The `perCall` witness's served `y = dq(z, s_a, s_w)` (gates 103, 65, 66). -/
def witPYGate : Accounting.Gates.Gate ℕ :=
  witGateR (.budgeted .rounding witDq) [103, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

/-- `witLayout`'s one call in `perCall`, after `witPre`: the noised operands `X_k = A_k + coin` (68–99); the noised
product `m` (100); `ΣB_k` (101) and the correction `coin·ΣB_k` (102); the served `z = m − coin·ΣB_k` (103), whose cone
holds the coin, and `y` (104); and the receipt over `m` (105). `z`, `y` and the receipt are committed the round after
the call's. -/
def witPRest : Accounting.Gates.Dag ℕ :=
  (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witMGate, witSGate, witCGate, witPZGate, witPYGate,
      witGateR (.affine 0 [1]) [100] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

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

/-- `witP`'s classes: the coin, the A row and `s_a`, the B column and `s_w`, the served `z` and `y`, the receipt, and
the rest inner. -/
def witPCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 103 ∨ i = 104 then .out
  else if i = 105 then .receipt
  else .inner

theorem witTieP : Tie witLayout .perCall witP witPCls where
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
  computed := by decide +kernel
  committed := by decide +kernel
  noised := by decide +kernel
  forms := by decide +kernel
  linked := fun _ _ _ => ⟨103, by decide +kernel, by decide +kernel, rfl⟩
  commit := by decide +kernel
  inner := by decide +kernel

theorem witP_layerCommitted : LayerCommitted witP :=
  tieCommits ℕ witLayout .perCall witP witPCls witTieP

/-- The served `z` reads the coin, through the noised product it is peeled from. -/
theorem witP_peeled : (coinMark witP).getD 103 0 = 1 := by
  decide +kernel

theorem witP_valA (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witP.dag x) {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    Accounting.Gates.val witP.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valA witPRest x hx h1 h2

theorem witP_valB (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witP.dag x) {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    Accounting.Gates.val witP.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valB witPRest x hx h1 h2

theorem witP_valCoin (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witP.dag x) :
    Accounting.Gates.val witP.dag x 0 = x 0 ∧ 0 ≤ x 0 ∧ x 0 ≤ 255 :=
  witPre_input (b := .coin) witPRest x (by rw [witPre_get witPRest (by omega)]; rfl) hx

/-- A noised operand's value is its A word's plus the coin's. -/
theorem witP_valX (x : ℕ → ℤ) {k : ℕ} (h1 : 1 ≤ k) (h2 : k < 33) :
    Accounting.Gates.val witP.dag x (67 + k) =
      Accounting.Gates.val witP.dag x k + Accounting.Gates.val witP.dag x 0 := by
  have hg : witP.dag[67 + k]? = some (witGateR (.affine 0 [1, 1]) [k, 0] (-128) 382 false) := by
    interval_cases k <;> rfl
  rw [val_of_getElem? witP.dag x (67 + k) hg (by intro j hj; simp [witGateR] at hj; omega)]
  exact (affine_two 1 1 _ _).trans (by ring)

/-- **The peeled `z` is the product of the A row and the B column** on every typed input: the noised product is
`Σ A_k·B_k + coin·ΣB_k`, within s32, and the correction takes the noise back out. -/
theorem witP_productOf : ProductOf witLayout witP witPCls 103 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, by unfold S8; decide +kernel, by unfold S8; decide +kernel,
    fun x hx => ?_⟩
  obtain ⟨hc, hc1, hc2⟩ := witP_valCoin x hx
  have h67 : Accounting.Gates.val witP.dag x 67 = 0 := witPre_val67 witPRest x
  have hX : (List.range' 68 32).map (Accounting.Gates.val witP.dag x) =
      ((List.range' 1 32).map (Accounting.Gates.val witP.dag x)).map (· + x 0) := by
    rw [show List.range' 68 32 = (List.range' 1 32).map (67 + ·) from (List.map_add_range' 1 32 1).symm,
      List.map_map, List.map_map]
    refine List.map_congr_left fun k hk => ?_
    obtain ⟨hk1, hk2⟩ := List.mem_range'_1.1 hk
    simp only [Function.comp_apply]
    rw [witP_valX x hk1 (by omega), hc]
  have hXb : ∀ a ∈ ((List.range' 1 32).map (Accounting.Gates.val witP.dag x)).map (· + x 0),
      -382 ≤ a ∧ a ≤ 382 := by
    intro a ha
    obtain ⟨v, hv, rfl⟩ := List.mem_map.1 ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hv
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witP_valA x hx hj1 (by omega)
    show -382 ≤ Accounting.Gates.val witP.dag x j + x 0 ∧ Accounting.Gates.val witP.dag x j + x 0 ≤ 382
    rw [he]
    constructor <;> omega
  have hBb : ∀ b ∈ (List.range' 33 32).map (Accounting.Gates.val witP.dag x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witP_valB x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 382) (Q := 128) (by norm_num) (by norm_num) _ _ hXb hBb
  simp only [List.length_map, List.length_range'] at hb1 hb2
  have hm : Accounting.Gates.val witP.dag x 100 = (List.zipWith (· * ·)
      (((List.range' 1 32).map (Accounting.Gates.val witP.dag x)).map (· + x 0))
      ((List.range' 33 32).map (Accounting.Gates.val witP.dag x))).sum := by
    rw [val_of_getElem? witP.dag x 100 (g := witMGate) rfl (by decide +kernel)]
    simp only [witMGate, witGateR, Accounting.Gates.Op.out, Accounting.Gates.Op.raw, List.map_cons,
      List.headD_cons, List.tail_cons, h67, zero_add, List.map_append, hX]
    unfold Accounting.Gates.dot
    rw [List.take_left' (by simp), List.drop_left' (by simp)]
    exact clampS32_of_bound (by omega) (by omega)
  have hS : Accounting.Gates.val witP.dag x 101 =
      ((List.range' 33 32).map (Accounting.Gates.val witP.dag x)).sum := by
    rw [val_of_getElem? witP.dag x 101 (g := witSGate) rfl (by decide +kernel)]
    show (0 : ℤ) + (List.zipWith (· * ·) (List.replicate 32 1)
      ((List.range' 33 32).map (Accounting.Gates.val witP.dag x))).sum = _
    rw [zero_add, sum_zipWith_one 32 _ (by simp)]
  have hC : Accounting.Gates.val witP.dag x 102 =
      x 0 * ((List.range' 33 32).map (Accounting.Gates.val witP.dag x)).sum := by
    rw [val_of_getElem? witP.dag x 102 (g := witCGate) rfl (by decide +kernel), ← hc, ← hS]
    rfl
  rw [val_of_getElem? witP.dag x 103 (g := witPZGate) rfl (by decide +kernel)]
  refine (affine_two 1 (-1) _ _).trans ?_
  rw [hm, hC, sum_zipWith_add (x 0) _ _ (by simp)]
  ring

/-- **The served `y` is `dq` of the peeled `z`** and the two scales, on every typed input. -/
theorem witP_dequantOf : DequantOf witDq witP witPCls 104 103 [65, 66] := by
  refine ⟨by decide +kernel, fun x _ => ?_⟩
  rw [val_of_getElem? witP.dag x 104 (g := witPYGate) rfl (by decide +kernel)]
  rfl

/-- The receipt reads every A word (through its noised `X_k` and `m`) and every B word (through `m`). -/
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

theorem witP_valueChecked : ValueChecked witLayout witDq witP witPCls := by
  refine ⟨fun r hr hrc => ?_, fun i hi hio => ?_⟩
  · obtain rfl : r = 105 :=
      (by decide +kernel : ∀ r, r < witP.dag.length → witPCls r = .receipt → r = 105) r hr hrc
    exact ⟨103, 104, List.range' 1 32, List.range' 33 32, [65, 66], by decide +kernel, by decide +kernel, rfl, rfl,
      by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel, witP_productOf, witP_dequantOf,
      witP_feeds⟩
  · rcases (by decide +kernel : ∀ i, i < witP.dag.length → witPCls i = .out → i = 103 ∨ i = 104) i hi hio
      with rfl | rfl
    · exact Or.inl ⟨_, _, witP_productOf⟩
    · exact Or.inr ⟨103, _, _, _, rfl, by decide +kernel, witP_productOf, witP_dequantOf⟩

/-- **An honest `perCall` window meets `CreditTie`**, in the order CMP-65 gives kernels: the call's A table before its
coin, then the coin, the noise and the receipt, with the unit's `z` and `y` committed beside the receipt. Its `z` is
peeled from the noised product (`witP_peeled`: its cone holds the coin, as `Tie.noised` asks of a `perCall` `out`
word), and is value-checked all the same. -/
theorem witCallCredit : CreditTie witLayout .perCall witDq witP witPCls :=
  ⟨witTieP, witP_valueChecked⟩

end Properties.Served
