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
within s32, so the clamp is the identity) and `y`'s is `dq` of it (`witZ_dequantOf`). `wit_notValueChecked`: the
`beside` witness, whose unit has no `out` word, fails `ValueChecked` at every `dq`.
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

/-- A sum of products of words in `[−128, 127]` is at most `16384` a product in size. -/
theorem sum_zipWith_bound : ∀ (l1 l2 : List ℤ), (∀ a ∈ l1, -128 ≤ a ∧ a ≤ 127) → (∀ b ∈ l2, -128 ≤ b ∧ b ≤ 127) →
    -(16384 * l1.length : ℤ) ≤ (List.zipWith (· * ·) l1 l2).sum ∧
      (List.zipWith (· * ·) l1 l2).sum ≤ 16384 * l1.length
  | [], _, _, _ => by simp
  | _ :: _, [], _, _ => by
    simp only [List.zipWith_nil_right, List.sum_nil, List.length_cons]
    constructor <;> omega
  | a :: l1, b :: l2, h1, h2 => by
    obtain ⟨ha1, ha2⟩ := h1 a (by simp)
    obtain ⟨hb1, hb2⟩ := h2 b (by simp)
    obtain ⟨ih1, ih2⟩ :=
      sum_zipWith_bound l1 l2 (fun c hc => h1 c (by simp [hc])) (fun c hc => h2 c (by simp [hc]))
    have hab1 : a * b ≤ 16384 := by
      nlinarith [mul_nonneg (by linarith : (0 : ℤ) ≤ a + 128) (by linarith : (0 : ℤ) ≤ 127 - b),
        mul_nonneg (by linarith : (0 : ℤ) ≤ 127 - a) (by linarith : (0 : ℤ) ≤ b + 128)]
    have hab2 : -16384 ≤ a * b := by
      nlinarith [mul_nonneg (by linarith : (0 : ℤ) ≤ a + 128) (by linarith : (0 : ℤ) ≤ b + 128),
        mul_nonneg (by linarith : (0 : ℤ) ≤ 127 - a) (by linarith : (0 : ℤ) ≤ 127 - b)]
    simp only [List.zipWith_cons_cons, List.sum_cons, List.length_cons]
    push_cast
    constructor <;> linarith

/-- The s32 clamp is the identity on a value within `2^19`. -/
theorem clampS32_of_bound {s : ℤ} (h1 : -524288 ≤ s) (h2 : s ≤ 524288) : Accounting.Gates.clampS32 s = s := by
  have hmin : Accounting.Gates.s32Min ≤ s := le_trans (by norm_num [Accounting.Gates.s32Min]) h1
  have hmax : s ≤ Accounting.Gates.s32Max := le_trans h2 (by norm_num [Accounting.Gates.s32Max])
  unfold Accounting.Gates.clampS32
  rw [min_eq_right hmax, max_eq_right hmin]

/-! ## An honest `zTree` window meets `CreditTie` -/

/-- A gate of `op` on `args` declared `[lo, hi]`, committed or not, at no site. -/
def witGateR (op : Accounting.Gates.Op ℕ) (args : List ℕ) (lo hi : ℤ) (commit : Bool) : Accounting.Gates.Gate ℕ :=
  { op := op, args := args, lo := lo, hi := hi, step := 0, prim := "", site := .other, commit := commit }

/-- The witness's dequant: `z·s_a·s_w`. -/
def witDq (v : List ℤ) : ℤ := v.headD 0 * v.getD 1 0 * v.getD 2 0

/-- The served product `z`: the constant `0` (gate 67), then `K = 32` products of the A row (gates 1–32) with the B
column (gates 33–64), clamped to s32. -/
def witZGate : Accounting.Gates.Gate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 1 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
    Accounting.Gates.s32Max true

/-- The served `y = dq(z, s_a, s_w)` (gates 68, 65, 66). -/
def witYGate : Accounting.Gates.Gate ℕ :=
  witGateR (.budgeted .rounding witDq) [68, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

/-- `witLayout`'s one call in `zTree`: the coin (gate 0); the A row (1–32, s8 run inputs); the B column (33–64, s8
registered); the scales `s_a` (65, a run input) and `s_w` (66, registered); the constant `0` (67); the served `z`
(68) and `y` (69), committed under the round's root before its coin; the noised operands `X_k = A_k + coin` (70–101);
the noised product `m = Σ X_k·B_k` (102); and the receipt (103), committed the round after. -/
def witZDag : Accounting.Gates.Dag ℕ :=
  [witGateR (.input .coin 0) [] 0 255 false] ++
    (List.range' 1 32).map (fun j => witGateR (.input .run j) [] (-128) 127 false) ++
    (List.range' 33 32).map (fun j => witGateR (.input .registered j) [] (-128) 127 false) ++
    [witGateR (.input .run 65) [] 0 65535 false, witGateR (.input .registered 66) [] 0 65535 false,
      witGateR (.const 0) [] 0 0 false, witZGate, witYGate] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
        Accounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [102] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

def witZ : Declared ℕ where
  dag := witZDag
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

theorem witZ_getA {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    witZ.dag[j]? = some (witGateR (.input .run j) [] (-128) 127 false) := by
  interval_cases j <;> rfl

theorem witZ_getB {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    witZ.dag[j]? = some (witGateR (.input .registered j) [] (-128) 127 false) := by
  interval_cases j <;> rfl

/-- An A word's value is its input's, an s8 word on a typed input. -/
theorem witZ_valA (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witZ.dag x) {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    Accounting.Gates.val witZ.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 := by
  have hg := witZ_getA h1 h2
  refine ⟨?_, hx _ (List.mem_of_getElem? hg) .run j rfl⟩
  rw [val_of_getElem? witZ.dag x j hg (by simp [witGateR])]
  rfl

/-- A B word's value is its input's, an s8 word on a typed input. -/
theorem witZ_valB (x : ℕ → ℤ) (hx : Accounting.Gates.Typed witZ.dag x) {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    Accounting.Gates.val witZ.dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 := by
  have hg := witZ_getB h1 h2
  refine ⟨?_, hx _ (List.mem_of_getElem? hg) .registered j rfl⟩
  rw [val_of_getElem? witZ.dag x j hg (by simp [witGateR])]
  rfl

/-- **The served `z` is the product of the A row and the B column** on every typed input: its 32 products of s8 words
stay within s32, so its clamp is the identity. -/
theorem witZ_productOf : ProductOf witLayout witZ witZCls 68 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, by unfold S8; decide +kernel, by unfold S8; decide +kernel,
    fun x hx => ?_⟩
  have h67 : Accounting.Gates.val witZ.dag x 67 = 0 := by
    rw [val_of_getElem? witZ.dag x 67 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  have hA : ∀ a ∈ (List.range' 1 32).map (Accounting.Gates.val witZ.dag x), -128 ≤ a ∧ a ≤ 127 := by
    intro a ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witZ_valA x hx hj1 (by omega)
    rw [he]
    exact ⟨hlo, hhi⟩
  have hB : ∀ b ∈ (List.range' 33 32).map (Accounting.Gates.val witZ.dag x), -128 ≤ b ∧ b ≤ 127 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witZ_valB x hx hj1 (by omega)
    rw [he]
    exact ⟨hlo, hhi⟩
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound _ _ hA hB
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

end Properties.Served
