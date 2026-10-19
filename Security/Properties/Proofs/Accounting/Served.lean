import Properties.Accounting.Served

/-!
# Proofs of the served tie's statements (`Accounting.Served`, `Properties.Served`)

`TieCommits` goes class by class: a registered value's commit coin is `0`; an A row's or a `y` row's is its round,
which is its first reader's, since only its own call reads it (`firstRead_cases`), or below the draw when nothing does
(`round_lt`); an admit word's or `zTree`'s `z` word's is its round, the floor of its cut; a receipt's or `perCall`'s
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
coin, as `h1tile`'s `z` doesn't.
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
  | admit =>
    have hb := T.computed ℓ hlt (by simp [hk])
    rw [T.commit ℓ hlt (by simp [hk]) (by simp [hk]), hk, ← hround]
    exact round_le_cut_of_none hb
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
  refine ⟨fun F hF => ?_, by decide +kernel, by decide +kernel, by decide +kernel⟩
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
  first
    | exact one_ne_zero h
    | simp [witRunAt] at h

end Properties.Served
