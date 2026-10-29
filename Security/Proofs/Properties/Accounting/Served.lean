import Mathlib.Data.List.GetD
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum.Basic
import Properties.Accounting.Served

/-!
# Proofs of the served tie's statements (`Accounting.Served`, `Properties.Served`)

`TieCommits` goes class by class, every class but the receipt and the check word in `LayerCommitted`'s first case: a
registered value's commit coin is `0`; an A row's or a `y` row's is its round, which is its first reader's, since only
its own call reads it (`firstRead_cases`), or below the draw when nothing does (`round_lt`); a public input's is at most
its first reader's round, or the draw when nothing reads it (`Tie.pub`); a computed `zTree` `z` or `y` word's is its
round, the floor of its cut, and a record's its round, its first reader's, as an A row's (`Tie.outWords`); `perCall`'s
`z` or `y` word's is the round after its own, which its coin read reaches (`Tie.noised`); an inner value's is its cut.
A receipt and a check word are in the second case: a computed value (`Tie.computed`), committed by the draw
(`Tie.receipt`, `Tie.checked`), and only gates of its own unit read it (`Tie.readers`; each reader is a gate,
`reads_lt`).
`TieLive`: a unit with a word has a first gate, its round is a round of the window, and every one is live.
`SeededScheduleNotLive`: `UnitsOfLayout` puts a read run input at the draw, `LayerCommitted` at its first reader's
round, which `Wf` puts below it (an input has no second case). `TieScope` is `Scope` from `TieLive`, `TieCommits` and its three hypotheses, and
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
within s32, so the clamp is the identity) and `y`'s is `dq` of it at the call's scales (`witZ_dequantOf`).
`witCallCredit`: an honest `perCall` window in the kernels' order meets `CreditTie`. Its `z` is peeled from the noised
product,
`m − coin·ΣB_k` with `m = Σ (A_k + coin)·B_k`, so its cone holds the coin (`witP_peeled`), and its value is the dot
all the same (`witP_productOf`: `m` is `Σ A_k·B_k + coin·ΣB_k` within s32, `sum_zipWith_add`). `wit_notValueChecked`:
the `beside` witness, whose unit has no `outY` word, fails `ValueChecked` at every `dq`. The refusals, each a tied
window that met `CreditTie` before the restatement: `witTwice_notValueChecked`, two units whose second
`z' = 0 + 1·z` copies the first's product, at `witLayout2`, whose whole population is the two (`witTwice2_population`);
`witWS_notValueChecked`, a `y` whose scales are the A row's and the B column's first words, which are
s8 and so no scales; `witRaw_notValueChecked`, a raw product `w = 0 + 1·z` served as a `y` word. The
last two are refused at one typed input, `witX3` (`s_a = 3`, every other word `2`), whose values `decide +kernel`
computes. Each unit of these has no check word, so its checks hold on every input (`held_of_none`).

`witCheckedCredit`: an honest checked-record `zTree` window meets `CreditTie`. Its unit's `z` and `y` are its record,
run inputs under the round's root that only its own call and unit read, beside its own dot and `dq` (inner wires); its
check word `ok`, committed with the receipt, is `1` exactly when the record is those (`witC_ok`), so the record's `z` is
the dot (`witC_productOf`) and its `y` the `dq` (`witC_dequantOf`) on every typed input the check holds on; the receipt
also reads the unit's position, a public input committed before the round's coin. On any typed input, replacing the
record by the dot and the `dq` (`witCFix`, an s32 and an s64 word, `witC_dq_bound`) meets the check and changes no
other input, so the checks can be met. The refusals, each over the same circuit: a constant check
(`witConst_notValueChecked`) and one with an escape hatch, `ok = (z = Σ ∧ y = dq) ∨ A_1 = 1`
(`witHatch_notValueChecked`), both tied (`witTieCConst`, `witTieCHatch`), hold at `witXR` (`A_1 = 1`, the record's
`z = 1`, every other word `0`), where no B word is nonzero and so `z` is no product (`witC_notValueChecked`); a check no
record meets (`witNever_notValueChecked`), which meets `ProductOf` vacuously (`witNever_productOf`) and fails the
clause that the checks can be met; a gate of another unit (`witCrossUnit_notTie`) or of another call
(`witCrossCall_notTie`) reading the record, which fails `Tie.outWords` however the gates are classed; and a public input
committed after the coin of the round that reads it (`witPubLate_notTie`, `witPubLate_notLayerCommitted`).

`witTieRing`: a ring of depth `D = 6` is tied, a `zTree` window of eight steps whose step `s` receipt is committed under
round `s + 5`'s root, the last five steps' in the manifest before the draw; step `0`'s receipt is cut at coin `1` and
committed at coin `5` (`witRing_late`), so only `LayerCommitted`'s second case covers it, and the window is in
Compose's scope (`witRing_scope`, at a scheme whose luck is proved, `witScheme`). `witCross_notTie`: a receipt that a
gate of another unit reads, committed at the draw, after that reader's coin, fails `Tie.readers`, and `LayerCommitted`
too (`witCross_notLayerCommitted`).

The cone clause: two tied windows of two units of `witLayout2`'s one call, in which unit 0 recomputes
unit 1's product `Σ₁′` (the A row with the rotated B column, `witRotB`) and a word of unit 0 reads unit 1's computed
`z₁`, the same dot after unit 1's own constant `0`, so `z₁ = Σ₁′` on every input. In `witConeCheck` it is the check
word `ok₀ = (z₀ = Σ₀ ∧ y₀ = dq₀) ∨ z₁ ≠ Σ₁′`, whose escape never fires under `Gates.val`, so the record `z₀` is the
product on every input the check holds on (`witConeCheck_productOf`); in `witConeZ` it is a computed
`z₀ = Σ₀ + (z₁ − Σ₁′)`, which is unit 0's dot on every input (`witConeZ_productOf`). The Definitions accept both
words, and both fail `ValueChecked`'s cone clause (`witConeCheck_notValueChecked`, `witConeZ_notValueChecked`): at run
time `z₁` is what an undrawn unit 1's prover committed. `computedFeeds_unit` reduces the clause to the circuit's
computed-reads-computed edges.

The old refusals on checked records: `held_complete` reads `ValueChecked`'s fifth clause through `Tie`,
so under `CreditTie` every assignment of the `act`, `registered`, `pub` and coin gates has a typed input meeting every
unit's checks, and a refutation at chosen operands and scales survives the checks. `witKTwice_notValueChecked`:
two checked units, each its record held to its own dot of the same operands, inside the cone clause
(`witKTwice_cone`), fail distinctness, their records' `z` one value wherever both checks hold (`witKTwice_same`).
`witK mid` is `witC witOk` with the gates `mid` in place of its `dq`, the check holding the record's `y` to `mid`'s last
gate, and `witK_notValueChecked` refuses one at a typed input its check holds on whose `y` isn't `dq` of its `z` at
the call's scales: `witKWS_notValueChecked` (`dq(Σ, A_1, B_1)`, at `witXK 128 512`),
`witKRaw_notValueChecked` (the raw product, at `witXK 128 128`) and `witKMix_notValueChecked`
(`0·Σ + 1·dq(p)` of another product `p`, at `witXM`, where `p = 0` and `z = 1`).
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

/-- An `outZ` word's commit coin is its round's, except in `perCall`. -/
theorem commitCoin_outZ {F : Form} (hF : F ≠ .perCall) (r : ℕ) : commitCoin F .outZ r = r := by
  cases F <;> first | rfl | exact (hF rfl).elim

/-- An `outY` word's commit coin is its round's, except in `perCall`. -/
theorem commitCoin_outY {F : Form} (hF : F ≠ .perCall) (r : ℕ) : commitCoin F .outY r = r := by
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

/-- A gate that reads a value is a gate of the circuit. -/
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
  have h3 : Cd.cutOf ℓ = Cd.firstRead ℓ := by simp [Declared.cutOf, hb]
  obtain ⟨j', hj', _, h⟩ := firstRead_of_reads Cd hj hR
  have hr := hW2 _ (hW1 j' hj').1
  rcases hLC ℓ hW with h2 | ⟨hn, -⟩
  · omega
  · rw [hb] at hn
    cases hn

/-- `Tie.commit` from its cases: `decide` finds no `Decidable` instance for its five premises within the default
`synthInstance.maxSize`, and finds one for three premises and a disjunction. -/
theorem tie_commit {ι : Type} {L : Layout} {F : Form} {Cd : Declared ι} {cls : ℕ → Cls}
    (h : ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin → cls i ≠ .receipt →
      cls i = .check ∨ cls i = .pub ∨ Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i))) :
    ∀ i, i < Cd.dag.length → cls i ≠ .inner → cls i ≠ .coin → cls i ≠ .receipt → cls i ≠ .check → cls i ≠ .pub →
      Cd.commitRound i = commitCoin F (cls i) (F.roundOf L (Cd.call i)) :=
  fun i hi h1 h2 h3 h4 h5 => ((h i hi h1 h2 h3).resolve_left h4).resolve_left h5

/-! ## A tied circuit with gates -/

/-- One step of one linear call, one 4 × 8 unit. -/
def witLayout : Layout where
  steps := 1
  rows := 4
  tileRows := 4
  tileCols := 8
  linears := [(32, 8)]

/-- A gate of `op` on `args`, committed or not, at no place. -/
def witGate (op : Accounting.Gates.IntOp Unit) (args : List ℕ) (commit : Bool) : Accounting.Gates.IntGate Unit :=
  { op := op, args := args, lo := 0, hi := 0, step := 0, prim := "", place := .other, commit := commit }

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
  | 2 => .outY
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
  absurd (T.noised 2 (by decide +kernel) (Or.inr ⟨Or.inr rfl, rfl⟩)) (by decide +kernel)

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

/-- The `beside` witness's unit has no `outY` word, so its receipt has no value-checked product: it fails
`ValueChecked` whatever the dequant. -/
theorem wit_notValueChecked (dq : List ℤ → ℤ) : ¬ ValueChecked witLayout dq witCircuit witCls := by
  rintro ⟨_, _, -, -, -, h, -⟩
  obtain ⟨y, hy, hyo, -⟩ := h 2 (by decide +kernel) rfl
  exact (by decide +kernel : ∀ y, y < witCircuit.dag.length → witCls y ≠ .outY) y hy hyo

/-! ## A gate's value, from its operands' -/

section Values

open Accounting.Gates

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

/-- Gate `i`'s value is its operation on its operands' values, each an earlier gate's. -/
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

/-- An input gate's value is its slot's word, whatever its operand list. -/
theorem val_input {ι : Type} (C : IntDag ι) (x : ι → ℤ) {i : ℕ} {g : IntGate ι} {b : Binding} {s : ι}
    (hg : C[i]? = some g) (hop : g.op = .input b s) : val C x i = x s := by
  obtain ⟨h, hgi⟩ := List.getElem?_eq_some_iff.1 hg
  rw [val_eq C x i h, hgi]
  unfold IntGate.value IntGate.raw
  rw [hop]
  rfl

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

/-- A gate of `op` on `args` declared `[lo, hi]`, committed or not, at no place. -/
def witGateR (op : Accounting.Gates.IntOp ℕ) (args : List ℕ) (lo hi : ℤ) (commit : Bool) : Accounting.Gates.IntGate ℕ :=
  { op := op, args := args, lo := lo, hi := hi, step := 0, prim := "", place := .other, commit := commit }

/-- The witnesses' dequant: `z·s_a·s_w`. -/
def witDq (v : List ℤ) : ℤ := v.headD 0 * v.getD 1 0 * v.getD 2 0

/-- The two witnesses' first gates, `witLayout`'s one call's inputs: the coin (gate 0); the A row (1–32, s8 run
inputs); the B column (33–64, s8 registered); the scales `s_a` (65, a run input) and `s_w` (66, registered); and the
constant `0` (67). -/
def witPre : Accounting.Gates.IntDag ℕ :=
  [witGateR (.input .coin 0) [] 0 255 false] ++
    (List.range' 1 32).map (fun j => witGateR (.input .run j) [] (-128) 127 false) ++
    (List.range' 33 32).map (fun j => witGateR (.input .registered j) [] (-128) 127 false) ++
    [witGateR (.input .run 65) [] 0 65535 false, witGateR (.input .registered 66) [] 0 65535 false,
      witGateR (.const 0) [] 0 0 false]

theorem witPre_get (rest : Accounting.Gates.IntDag ℕ) {j : ℕ} (h : j < 68) : (witPre ++ rest)[j]? = witPre[j]? :=
  List.getElem?_append_left (lt_of_lt_of_eq h (by decide +kernel))

/-- An input gate's value is its slot's, in its declared range on a typed input. -/
theorem witPre_input (rest : Accounting.Gates.IntDag ℕ) (x : ℕ → ℤ) {j : ℕ} {b : Accounting.Gates.Binding} {lo hi : ℤ}
    (hg : (witPre ++ rest)[j]? = some (witGateR (.input b j) [] lo hi false))
    (hx : Accounting.Gates.Typed (witPre ++ rest) x) :
    Accounting.Gates.val (witPre ++ rest) x j = x j ∧ lo ≤ x j ∧ x j ≤ hi := by
  refine ⟨?_, hx _ (List.mem_of_getElem? hg) b j rfl⟩
  rw [val_of_getElem? _ x j hg (by simp [witGateR])]
  rfl

theorem witPre_valA (rest : Accounting.Gates.IntDag ℕ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witPre ++ rest) x)
    {j : ℕ} (h1 : 1 ≤ j) (h2 : j < 33) :
    Accounting.Gates.val (witPre ++ rest) x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_input (b := .run) rest x (by rw [witPre_get rest (by omega)]; interval_cases j <;> rfl) hx

theorem witPre_valB (rest : Accounting.Gates.IntDag ℕ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witPre ++ rest) x)
    {j : ℕ} (h1 : 33 ≤ j) (h2 : j < 65) :
    Accounting.Gates.val (witPre ++ rest) x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_input (b := .registered) rest x (by rw [witPre_get rest (by omega)]; interval_cases j <;> rfl) hx

theorem witPre_val67 (rest : Accounting.Gates.IntDag ℕ) (x : ℕ → ℤ) : Accounting.Gates.val (witPre ++ rest) x 67 = 0 := by
  rw [val_of_getElem? _ x 67 (g := witGateR (.const 0) [] 0 0 false) (by rw [witPre_get rest (by omega)]; rfl)
    (by simp [witGateR])]
  rfl

/-- Over `witPre`, a dot of the A row with B words `bs` after a gate `h` of value `0` is their products' sum on a typed
input: its 32 products of s8 words stay within s32, so its clamp is the identity. -/
theorem witPre_dot (rest : Accounting.Gates.IntDag ℕ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witPre ++ rest) x)
    {i h : ℕ} {bs : List ℕ} {c : Bool}
    (hg : (witPre ++ rest)[i]? = some (witGateR (.mma .s8 .s8 32 true) (h :: (List.range' 1 32 ++ bs))
      Accounting.Gates.s32Min Accounting.Gates.s32Max c))
    (ha : ∀ j ∈ h :: (List.range' 1 32 ++ bs), j < i) (hh : Accounting.Gates.val (witPre ++ rest) x h = 0)
    (hbs : ∀ b ∈ bs, 33 ≤ b ∧ b < 65) :
    Accounting.Gates.val (witPre ++ rest) x i =
      (List.zipWith (· * ·) ((List.range' 1 32).map (Accounting.Gates.val (witPre ++ rest) x))
        (bs.map (Accounting.Gates.val (witPre ++ rest) x))).sum := by
  have hA : ∀ a ∈ (List.range' 1 32).map (Accounting.Gates.val (witPre ++ rest) x), -128 ≤ a ∧ a ≤ 128 := by
    intro a ha'
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha'
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witPre_valA rest x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  have hB : ∀ b ∈ bs.map (Accounting.Gates.val (witPre ++ rest) x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := hbs j hj
    obtain ⟨he, hlo, hhi⟩ := witPre_valB rest x hx hj1 hj2
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 128) (Q := 128) (by norm_num) (by norm_num) _ _ hA hB
  simp only [List.length_map, List.length_range'] at hb1 hb2
  rw [val_of_getElem? _ x i hg ha]
  simp only [witGateR, Accounting.Gates.IntOp.out, Accounting.Gates.IntOp.raw, List.map_cons, List.headD_cons,
    List.tail_cons, hh, zero_add, List.map_append]
  unfold Accounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

/-- The served product `z` of the `zTree` witness: the constant `0` (gate 67), then `K = 32` products of the A row
(gates 1–32) with the B column (gates 33–64), clamped to s32. -/
def witZGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 1 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
    Accounting.Gates.s32Max true

/-- The `zTree` witness's served `y = dq(z, s_a, s_w)` (gates 68, 65, 66). -/
def witYGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [68, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

/-- `witLayout`'s one call in `zTree`, after `witPre`: the served `z` (68) and `y` (69), committed under the round's
root before its coin; the noised operands `X_k = A_k + coin` (70–101); the noised product `m = Σ X_k·B_k` (102); and
the receipt (103), committed the round after. -/
def witZRest : Accounting.Gates.IntDag ℕ :=
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
  else if i = 68 then .outZ
  else if i = 69 then .outY
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
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx _ => ?_⟩
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
  simp only [witZGate, witGateR, Accounting.Gates.IntOp.out, Accounting.Gates.IntOp.raw, List.map_cons, List.headD_cons,
    List.tail_cons, h67, zero_add, List.map_append]
  unfold Accounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

/-- **The served `y` is `dq` of the served `z`** at the call's scales `s_a` (65) and `s_w` (66), on every typed
input. -/
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

/-- A circuit with no `check` word holds every unit's checks on every input. -/
theorem held_of_none {ι : Type} {Cd : Declared ι} {cls : ℕ → Cls}
    (h : ∀ k, k < Cd.dag.length → cls k ≠ .check) (u : ℕ) (x : ι → ℤ) : Held Cd cls u x :=
  fun k hk hc _ => absurd hc (h k hk)

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

/-- `witZ`'s one `outZ` word is its served `z`. -/
theorem witZ_outZ : ∀ z, z < witZ.dag.length → witZCls z = .outZ → z = 68 := by
  decide +kernel

theorem witZ_valueChecked : ValueChecked witLayout witDq witZ witZCls := by
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

/-- **An honest `zTree` window meets `CreditTie`**: its served product, computed beside the noised one
(`witZ_beside`), is value-checked, at one round a step. -/
theorem witCreditTie : CreditTie witLayout .zTree witDq witZ witZCls :=
  ⟨witTieZ, witZ_valueChecked⟩

/-- The `perCall` witness's noised product, `m = Σ X_k·B_k` over the noised operands (gates 68–99) and the B column
(33–64), clamped to s32. -/
def witMGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 68 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
    Accounting.Gates.s32Max false

/-- `Σ B_k` (gates 33–64). -/
def witSGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.affine 0 (List.replicate 32 1)) (List.range' 33 32) (-4096) 4064 false

/-- The noise's correction, `coin·ΣB_k` (gates 0, 101). -/
def witCGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding fun v => v.headD 0 * v.getD 1 0) [0, 101] (-1044480) 1036320 false

/-- The served `z`, peeled from the noised product: `m − coin·ΣB_k` (gates 100, 102). -/
def witPZGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.affine 0 [1, -1]) [100, 102] Accounting.Gates.s32Min Accounting.Gates.s32Max true

/-- The `perCall` witness's served `y = dq(z, s_a, s_w)` (gates 103, 65, 66). -/
def witPYGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [103, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true

/-- `witLayout`'s one call in `perCall`, after `witPre`: the noised operands `X_k = A_k + coin` (68–99); the noised
product `m` (100); `ΣB_k` (101) and the correction `coin·ΣB_k` (102); the served `z = m − coin·ΣB_k` (103), whose cone
holds the coin, and `y` (104); and the receipt over `m` (105). `z`, `y` and the receipt are committed the round after
the call's. -/
def witPRest : Accounting.Gates.IntDag ℕ :=
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
  else if i = 103 then .outZ
  else if i = 104 then .outY
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
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx _ => ?_⟩
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
    simp only [witMGate, witGateR, Accounting.Gates.IntOp.out, Accounting.Gates.IntOp.raw, List.map_cons,
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
    rw [val_of_getElem? witP.dag x 102 (g := witCGate) rfl (by decide +kernel)]
    simp only [witCGate, witGateR, Accounting.Gates.IntOp.out, Accounting.Gates.IntOp.raw, List.map_cons, List.map_nil,
      List.headD_cons, List.getD_cons_succ, List.getD_cons_zero, hc, hS]
  rw [val_of_getElem? witP.dag x 103 (g := witPZGate) rfl (by decide +kernel)]
  refine (affine_two 1 (-1) _ _).trans ?_
  rw [hm, hC, sum_zipWith_add (x 0) _ _ (by simp)]
  ring

/-- **The served `y` is `dq` of the peeled `z`** at the call's scales `s_a` (65) and `s_w` (66), on every typed
input. -/
theorem witP_dequantOf : DequantOf witDq witP witPCls 104 103 65 66 := by
  refine ⟨by decide +kernel, by decide +kernel, rfl, by unfold S8; decide +kernel, by decide +kernel,
    by decide +kernel, by unfold S8; decide +kernel, fun x _ _ => ?_⟩
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

/-- `witP`'s one `outZ` word is its peeled `z`. -/
theorem witP_outZ : ∀ z, z < witP.dag.length → witPCls z = .outZ → z = 103 := by
  decide +kernel

theorem witP_valueChecked : ValueChecked witLayout witDq witP witPCls := by
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

/-- **An honest `perCall` window meets `CreditTie`**, in the kernels' order: the call's A table before its
coin, then the coin, the noise and the receipt, with the unit's `z` and `y` committed beside the receipt. Its `z` is
peeled from the noised product (`witP_peeled`: its cone holds the coin, as `Tie.noised` asks of a `perCall` `outZ`
word), and is value-checked all the same. -/
theorem witCallCredit : CreditTie witLayout .perCall witDq witP witPCls :=
  ⟨witTieP, witP_valueChecked⟩

/-! ## Refusals: one product for two units, an A and a B word as the scales, and a raw
product served as `y` -/

/-- A circuit whose every input gate admits the word `x s` at its slot `s` is typed at `x`. -/
theorem typed_of_all {Cd : Declared ℕ} (x : ℕ → ℤ)
    (h : Cd.dag.all (fun g => match g.op with
      | .input _ s => decide (g.lo ≤ x s ∧ x s ≤ g.hi)
      | _ => true) = true) : Accounting.Gates.Typed Cd.dag x := by
  intro g hg b s hop
  have h' := List.all_eq_true.1 h g hg
  simp only [hop, decide_eq_true_eq] at h'
  exact h'

/-- The input with `s_a` (slot 65) `3` and every other word `2`. -/
def witX3 (j : ℕ) : ℤ := if j = 65 then 3 else 2

/-- `witZRest`, then a second unit: `z' = 0 + 1·z` (104), `y' = dq(z', s_a, s_w)` (105), its noised product over the
noised operands and the B column (106), and its receipt (107). -/
def witTwiceRest : Accounting.Gates.IntDag ℕ :=
  witZRest ++
    [witGateR (.affine 0 [1]) [68] Accounting.Gates.s32Min Accounting.Gates.s32Max true,
      witGateR (.budgeted .rounding witDq) [104, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) true,
      witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
        Accounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [106] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

/-- The window of one product for two units (`witTwice`, `ServedTwice.lean`): two units of one call, unit 1's served
product a copy of unit 0's, each unit with its own `y` and its own receipt over its own noised GEMM. -/
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

/-- `witLayout` with its one linear twice as wide: one call of two column tiles, so two units a call and a population
of two. -/
def witLayout2 : Layout := { witLayout with linears := [(32, 16)] }

theorem witTieTwice2 : Tie witLayout2 .zTree witTwice witTwiceCls where
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

/-- The call has two units, the window's population is two, and the circuit declares exactly those two, each receipt's
unit below the call's unit count. -/
theorem witTwice2_population : witLayout2.unitsOf 0 = 2 ∧ witTwice.units = witLayout2.population ∧
    witTwice.unit 103 < witLayout2.unitsOf 0 ∧ witTwice.unit 107 < witLayout2.unitsOf 0 := by
  decide +kernel

/-- Unit 1's served product is unit 0's, on every input. -/
theorem witTwice_copy (x : ℕ → ℤ) :
    Accounting.Gates.val witTwice.dag x 104 = Accounting.Gates.val witTwice.dag x 68 := by
  rw [val_of_getElem? witTwice.dag x 104
    (g := witGateR (.affine 0 [1]) [68] Accounting.Gates.s32Min Accounting.Gates.s32Max true) rfl (by decide +kernel)]
  show (0 : ℤ) + (List.zipWith (· * ·) [1] [Accounting.Gates.val witTwice.dag x 68]).sum = _
  simp

/-- **One product doesn't value-check two units**: `witTwice`'s two `outZ` words have one value on
every input (`witTwice_copy`), so it fails `ValueChecked` at any layout and dequant. -/
theorem witTwice_notValueChecked (L : Layout) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq witTwice witTwiceCls := by
  rintro ⟨_, _, -, hinj, -, -, -⟩
  exact absurd (hinj 68 104 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    fun x _ _ _ => (witTwice_copy x).symm) (by decide)

/-- At the layout whose whole population is `witTwice`'s two units, it is tied (`witTieTwice2`) and fails
`CreditTie`. -/
theorem witTwice2_notCreditTie (dq : List ℤ → ℤ) : ¬ CreditTie witLayout2 .zTree dq witTwice witTwiceCls := fun h =>
  witTwice_notValueChecked witLayout2 dq h.2

/-- `witZRest` with the served `y = dq(z, A_1, B_1)` (gates 68, 1, 33). -/
def witWSRest : Accounting.Gates.IntDag ℕ :=
  [witZGate, witGateR (.budgeted .rounding witDq) [68, 1, 33] (-2 ^ 63) (2 ^ 63 - 1) true] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 70 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
        Accounting.Gates.s32Max false,
      witGateR (.affine 0 [1]) [102] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

/-- The window of an A and a B word as the scales (`witWS`, `ServedTwice.lean`): `witZ` with its served `y`'s scales
the A row's first word and the B column's first word. -/
def witWS : Declared ℕ := { witZ with dag := witPre ++ witWSRest }

theorem witTieWS : Tie witLayout .zTree witWS witZCls where
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

theorem witWS_typed : Accounting.Gates.Typed witWS.dag witX3 := typed_of_all witX3 (by decide +kernel)

/-- `witWS` at `witX3`: `z = 32·2·2 = 128`, `y = 128·A_1·B_1 = 512`, `s_a = 3`, `s_w = 2`. -/
theorem witWS_vals : Accounting.Gates.val witWS.dag witX3 69 = 512 ∧ Accounting.Gates.val witWS.dag witX3 68 = 128 ∧
    Accounting.Gates.val witWS.dag witX3 65 = 3 ∧ Accounting.Gates.val witWS.dag witX3 66 = 2 := by
  decide +kernel

/-- **A served `y` whose scales are an A word and a B word fails `ValueChecked`**: the scales are words
not declared s8, which in `witWS` are only `s_a` and `s_w`, and `y = 512` at `witX3` is not `dq(128, 3, 2) = 768`. -/
theorem witWS_notValueChecked (L : Layout) : ¬ ValueChecked L witDq witWS witZCls := by
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

theorem witWS_notCreditTie : ¬ CreditTie witLayout .zTree witDq witWS witZCls := fun h =>
  witWS_notValueChecked witLayout h.2

/-- The window of a raw product served as `y`: `witZ` with a second served word, `w = 0 + 1·z` (104), a raw product
never dequantized, committed beside `z` and `y` under the round's root. -/
def witRaw : Declared ℕ :=
  { witZ with
    dag := witPre ++ (witZRest ++
      [witGateR (.affine 0 [1]) [68] Accounting.Gates.s32Min Accounting.Gates.s32Max true])
    commitRound := fun i => if 70 ≤ i ∧ i ≠ 104 then 1 else 0 }

/-- `witZCls`, with the raw product `w` a served `y` word. -/
def witRawCls (i : ℕ) : Cls := if i = 104 then .outY else witZCls i

theorem witTieRaw : Tie witLayout .zTree witRaw witRawCls where
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

theorem witRaw_typed : Accounting.Gates.Typed witRaw.dag witX3 := typed_of_all witX3 (by decide +kernel)

/-- `witRaw` at `witX3`: `w = z = 128`, `s_a = 3`, `s_w = 2`. -/
theorem witRaw_vals : Accounting.Gates.val witRaw.dag witX3 104 = 128 ∧
    Accounting.Gates.val witRaw.dag witX3 68 = 128 ∧ Accounting.Gates.val witRaw.dag witX3 65 = 3 ∧
    Accounting.Gates.val witRaw.dag witX3 66 = 2 := by
  decide +kernel

/-- **A raw product served as `y` fails `ValueChecked`**: every `outY` word is `dq` of an `outZ` word,
and `w = 128` at `witX3` is not `dq(128, 3, 2) = 768`. Before `z` and `y` had their own classes, `w` passed as a
product. -/
theorem witRaw_notValueChecked (L : Layout) : ¬ ValueChecked L witDq witRaw witRawCls := by
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

theorem witRaw_notCreditTie : ¬ CreditTie witLayout .zTree witDq witRaw witRawCls := fun h =>
  witRaw_notValueChecked witLayout h.2

/-! ## A checked record: the `zTree` unit's `z` and `y` are its record, which its `check` word holds -/

/-- The honest check over `[z_rec, Σ, y_rec, dq, A_1]`: `1` exactly when the record's `z` is the unit's dot and its `y`
the unit's `dq` of it (`PouwH1TileZy_v1`'s `ok`). -/
def witOk (v : List ℤ) : ℤ := if v.getD 0 0 = v.getD 1 0 ∧ v.getD 2 0 = v.getD 3 0 then 1 else 0

theorem witOk_eq {z s y d a : ℤ} (h : witOk [z, s, y, d, a] = 1) : z = s ∧ y = d := by
  unfold witOk at h
  split at h
  · assumption
  · exact absurd h (by decide)

/-- A constant check, `1` on every record. -/
def witOkConst (_ : List ℤ) : ℤ := 1

/-- A check with an escape hatch: the honest check, or `1` whenever the A row's first word is `1`. -/
def witOkHatch (v : List ℤ) : ℤ :=
  if (v.getD 0 0 = v.getD 1 0 ∧ v.getD 2 0 = v.getD 3 0) ∨ v.getD 4 0 = 1 then 1 else 0

/-- A check no record meets, `0` on every input. -/
def witOkNever (_ : List ℤ) : ℤ := 0

/-- The unit's own dot of the A row (gates 1–32) with the B column (33–64), clamped to s32, an inner wire. -/
def witDotGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (67 :: (List.range' 1 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
    Accounting.Gates.s32Max false

/-- The unit's own `dq(Σ, s_a, s_w)` (gates 71, 65, 66), an inner wire. -/
def witDqGate : Accounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [71, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) false

/-- The unit's check word `f [z_rec, Σ, y_rec, dq, A_1]` (gates 69, 71, 70, 72, 1), committed. -/
def witCheckGate (f : List ℤ → ℤ) : Accounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .compareSelect f) [69, 71, 70, 72, 1] 0 1 true

/-- `witLayout`'s one call in `zTree` with a checked record, after `witPre`: the unit's position (68, a public input);
its record, `z` (69) and `y` (70), run inputs under the round's root before its coin; the unit's own dot (71) and `dq`
(72); its check word (73); the noised operands `X_k = A_k + coin` (74–105); the noised product `m = Σ X_k·B_k` (106);
and the receipt over `m` and the position (107). The check word and the receipt are committed the round after. -/
def witCRest (f : List ℤ → ℤ) : Accounting.Gates.IntDag ℕ :=
  [witGateR (.input .run 68) [] 0 255 false,
    witGateR (.input .run 69) [] Accounting.Gates.s32Min Accounting.Gates.s32Max true,
    witGateR (.input .run 70) [] (-2 ^ 63) (2 ^ 63 - 1) true, witDotGate, witDqGate, witCheckGate f] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' 74 32 ++ List.range' 33 32)) Accounting.Gates.s32Min
        Accounting.Gates.s32Max false,
      witGateR (.affine 0 [1, 1]) [106, 68] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

/-- The checked-record window whose check word is `f`. -/
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

/-- `witC`'s classes: the coin, the A row and `s_a`, the B column and `s_w`, the position, the record's `z` and `y`,
the check word, the receipt, and the rest inner. -/
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

/-- **A checked-record window is tied**: its record a run input only its own call and unit read, its position a public
input committed before the coin of the round that reads it, its check word committed by the draw. -/
theorem witTieC : Tie witLayout .zTree (witC witOk) witCCls where
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

/-- The checked-record window meets `LayerCommitted`: its record at its round, its reader's cut, and its check word at
the draw's coin, past its cut, in the second case. -/
theorem witC_layerCommitted : LayerCommitted (witC witOk) :=
  tieCommits ℕ witLayout .zTree (witC witOk) witCCls witTieC

theorem witC_length (f : List ℤ → ℤ) : (witC f).dag.length = 108 := rfl

theorem witC_valA (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) {j : ℕ} (h1 : 1 ≤ j)
    (h2 : j < 33) : Accounting.Gates.val (witC f).dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valA (witCRest f) x hx h1 h2

theorem witC_valB (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) {j : ℕ} (h1 : 33 ≤ j)
    (h2 : j < 65) : Accounting.Gates.val (witC f).dag x j = x j ∧ -128 ≤ x j ∧ x j ≤ 127 :=
  witPre_valB (witCRest f) x hx h1 h2

theorem witC_valSa (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) :
    Accounting.Gates.val (witC f).dag x 65 = x 65 ∧ 0 ≤ x 65 ∧ x 65 ≤ 65535 :=
  witPre_input (b := .run) (witCRest f) x (by rw [witPre_get (witCRest f) (by omega)]; rfl) hx

theorem witC_valSw (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) :
    Accounting.Gates.val (witC f).dag x 66 = x 66 ∧ 0 ≤ x 66 ∧ x 66 ≤ 65535 :=
  witPre_input (b := .registered) (witCRest f) x (by rw [witPre_get (witCRest f) (by omega)]; rfl) hx

/-- The record's `z` is its slot's word. -/
theorem witC_val69 (f : List ℤ → ℤ) (x : ℕ → ℤ) : Accounting.Gates.val (witC f).dag x 69 = x 69 := by
  rw [val_of_getElem? (witC f).dag x 69
    (g := witGateR (.input .run 69) [] Accounting.Gates.s32Min Accounting.Gates.s32Max true) rfl
    (by simp [witGateR])]
  rfl

/-- The record's `y` is its slot's word. -/
theorem witC_val70 (f : List ℤ → ℤ) (x : ℕ → ℤ) : Accounting.Gates.val (witC f).dag x 70 = x 70 := by
  rw [val_of_getElem? (witC f).dag x 70 (g := witGateR (.input .run 70) [] (-2 ^ 63) (2 ^ 63 - 1) true) rfl
    (by simp [witGateR])]
  rfl

/-- The A row's products with the B column sum to within `32·128²` of `0` on a typed input. -/
theorem witC_sum_bound (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) :
    -524288 ≤ (List.zipWith (· * ·) ((List.range' 1 32).map (Accounting.Gates.val (witC f).dag x))
        ((List.range' 33 32).map (Accounting.Gates.val (witC f).dag x))).sum ∧
      (List.zipWith (· * ·) ((List.range' 1 32).map (Accounting.Gates.val (witC f).dag x))
        ((List.range' 33 32).map (Accounting.Gates.val (witC f).dag x))).sum ≤ 524288 := by
  have hA : ∀ a ∈ (List.range' 1 32).map (Accounting.Gates.val (witC f).dag x), -128 ≤ a ∧ a ≤ 128 := by
    intro a ha
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ha
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witC_valA f x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  have hB : ∀ b ∈ (List.range' 33 32).map (Accounting.Gates.val (witC f).dag x), -128 ≤ b ∧ b ≤ 128 := by
    intro b hb
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hb
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
    obtain ⟨he, hlo, hhi⟩ := witC_valB f x hx hj1 (by omega)
    rw [he]
    constructor <;> omega
  obtain ⟨hb1, hb2⟩ := sum_zipWith_bound (P := 128) (Q := 128) (by norm_num) (by norm_num) _ _ hA hB
  simp only [List.length_map, List.length_range'] at hb1 hb2
  constructor <;> omega

/-- **The unit's own dot is the product of the A row and the B column** on every typed input, whatever its check. -/
theorem witC_dot (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) :
    Accounting.Gates.val (witC f).dag x 71 =
      (List.zipWith (· * ·) ((List.range' 1 32).map (Accounting.Gates.val (witC f).dag x))
        ((List.range' 33 32).map (Accounting.Gates.val (witC f).dag x))).sum := by
  have h67 : Accounting.Gates.val (witC f).dag x 67 = 0 := witPre_val67 (witCRest f) x
  obtain ⟨hb1, hb2⟩ := witC_sum_bound f x hx
  rw [val_of_getElem? (witC f).dag x 71 (g := witDotGate) rfl (by decide +kernel)]
  simp only [witDotGate, witGateR, Accounting.Gates.IntOp.out, Accounting.Gates.IntOp.raw, List.map_cons, List.headD_cons,
    List.tail_cons, h67, zero_add, List.map_append]
  unfold Accounting.Gates.dot
  rw [List.take_left' (by simp), List.drop_left' (by simp)]
  exact clampS32_of_bound (by omega) (by omega)

theorem witC_dot_bound (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) :
    -524288 ≤ Accounting.Gates.val (witC f).dag x 71 ∧ Accounting.Gates.val (witC f).dag x 71 ≤ 524288 := by
  rw [witC_dot f x hx]
  exact witC_sum_bound f x hx

/-- The unit's own `dq` is `witDq` of its dot and the call's scales, on every input. -/
theorem witC_dq (f : List ℤ → ℤ) (x : ℕ → ℤ) :
    Accounting.Gates.val (witC f).dag x 72 = witDq [Accounting.Gates.val (witC f).dag x 71,
      Accounting.Gates.val (witC f).dag x 65, Accounting.Gates.val (witC f).dag x 66] := by
  rw [val_of_getElem? (witC f).dag x 72 (g := witDqGate) rfl (by decide +kernel)]
  rfl

/-- `dq` of a dot within `32·128²` at two u16 scales is an s64 word. -/
theorem witDq_bound {z a w : ℤ} (hz1 : -524288 ≤ z) (hz2 : z ≤ 524288) (ha1 : 0 ≤ a) (ha2 : a ≤ 65535)
    (hw1 : 0 ≤ w) (hw2 : w ≤ 65535) : -2 ^ 63 ≤ witDq [z, a, w] ∧ witDq [z, a, w] ≤ 2 ^ 63 - 1 := by
  show -2 ^ 63 ≤ z * a * w ∧ z * a * w ≤ 2 ^ 63 - 1
  have hp0 : 0 ≤ a * w := mul_nonneg ha1 hw1
  have hp1 : a * w ≤ 65535 * 65535 := by nlinarith [mul_nonneg (sub_nonneg.2 ha2) hw1]
  have h1 := mul_nonneg (by linarith : (0 : ℤ) ≤ 524288 - z) hp0
  have h2 := mul_nonneg (by linarith : (0 : ℤ) ≤ z + 524288) hp0
  constructor <;> nlinarith

theorem witC_dq_bound (f : List ℤ → ℤ) (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC f).dag x) :
    -2 ^ 63 ≤ Accounting.Gates.val (witC f).dag x 72 ∧ Accounting.Gates.val (witC f).dag x 72 ≤ 2 ^ 63 - 1 := by
  obtain ⟨hz1, hz2⟩ := witC_dot_bound f x hx
  obtain ⟨ha, ha1, ha2⟩ := witC_valSa f x hx
  obtain ⟨hw, hw1, hw2⟩ := witC_valSw f x hx
  rw [witC_dq f x, ha, hw]
  exact witDq_bound hz1 hz2 ha1 ha2 hw1 hw2

/-- The honest check word is `1` only where the record is the unit's dot and its `dq`. -/
theorem witC_ok (x : ℕ → ℤ) (h : Accounting.Gates.val (witC witOk).dag x 73 = 1) :
    Accounting.Gates.val (witC witOk).dag x 69 = Accounting.Gates.val (witC witOk).dag x 71 ∧
      Accounting.Gates.val (witC witOk).dag x 70 = Accounting.Gates.val (witC witOk).dag x 72 := by
  rw [val_of_getElem? (witC witOk).dag x 73 (g := witCheckGate witOk) rfl (by decide +kernel)] at h
  change witOk [Accounting.Gates.val (witC witOk).dag x 69, Accounting.Gates.val (witC witOk).dag x 71,
    Accounting.Gates.val (witC witOk).dag x 70, Accounting.Gates.val (witC witOk).dag x 72,
    Accounting.Gates.val (witC witOk).dag x 1] = 1 at h
  exact witOk_eq h

/-- `witC witOk`'s one `outZ` word is its record's `z`, and its one check word gate 73. -/
theorem witC_outZ : ∀ z, z < (witC witOk).dag.length → witCCls z = .outZ → z = 69 := by
  decide +kernel

theorem witC_check : ∀ k, k < (witC witOk).dag.length → witCCls k = .check → k = 73 := by
  decide +kernel

/-- **The checked record's `z` is the product of the A row and the B column** on every typed input its check holds
on. -/
theorem witC_productOf : ProductOf witLayout (witC witOk) witCCls 69 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx hH => ?_⟩
  rw [(witC_ok x (hH 73 (by decide +kernel) (by decide +kernel) rfl)).1]
  exact witC_dot witOk x hx

/-- **The checked record's `y` is `dq` of its `z`** at the call's scales, on every typed input its check holds on. -/
theorem witC_dequantOf : DequantOf witDq (witC witOk) witCCls 70 69 65 66 := by
  refine ⟨by decide +kernel, by decide +kernel, rfl, by unfold S8; decide +kernel, by decide +kernel,
    by decide +kernel, by unfold S8; decide +kernel, fun x _ hH => ?_⟩
  obtain ⟨h69, h70⟩ := witC_ok x (hH 73 (by decide +kernel) (by decide +kernel) rfl)
  rw [h70, h69, witC_dq witOk x]

/-- The receipt reads every A word (through its noised `X_k` and `m`) and every B word (through `m`). -/
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

/-- An input operation's slot, and `none` for any other. -/
def slotOf : Accounting.Gates.IntOp ℕ → Option ℕ
  | .input _ s => some s
  | _ => none

/-- Each input gate of `witC witOk` reads the slot of its own index. -/
theorem witC_slots : ∀ i, i < (witC witOk).dag.length →
    ((witC witOk).dag[i]?).bind (fun g => slotOf g.op) = none ∨
      ((witC witOk).dag[i]?).bind (fun g => slotOf g.op) = some i := by
  decide +kernel

theorem witC_slot {i s : ℕ} {g : Accounting.Gates.IntGate ℕ} {b : Accounting.Gates.Binding}
    (hg : (witC witOk).dag[i]? = some g) (hop : g.op = .input b s) : s = i := by
  obtain ⟨hi, -⟩ := List.getElem?_eq_some_iff.1 hg
  have hs : ((witC witOk).dag[i]?).bind (fun g => slotOf g.op) = some s := by
    rw [hg]
    show slotOf g.op = some s
    rw [hop]; rfl
  rcases witC_slots i hi with h | h <;> rw [hs] at h
  · cases h
  · exact Option.some.inj h

/-- The record's slots are declared as their words: `z` an s32 word, `y` an s64 one. -/
theorem witC_bounds : ∀ g ∈ (witC witOk).dag,
    (slotOf g.op = some 69 → g.lo = Accounting.Gates.s32Min ∧ g.hi = Accounting.Gates.s32Max) ∧
      (slotOf g.op = some 70 → g.lo = -2 ^ 63 ∧ g.hi = 2 ^ 63 - 1) := by
  decide +kernel

/-- Input `x` with the record replaced by what the unit computes: `z` its dot, `y` its `dq`. -/
def witCFix (x : ℕ → ℤ) (s : ℕ) : ℤ :=
  if s = 69 then Accounting.Gates.val (witC witOk).dag x 71
  else if s = 70 then Accounting.Gates.val (witC witOk).dag x 72 else x s

theorem witCFix_69 (x : ℕ → ℤ) : witCFix x 69 = Accounting.Gates.val (witC witOk).dag x 71 := by
  simp [witCFix]

theorem witCFix_70 (x : ℕ → ℤ) : witCFix x 70 = Accounting.Gates.val (witC witOk).dag x 72 := by
  simp [witCFix]

theorem witCFix_ne (x : ℕ → ℤ) {s : ℕ} (h1 : s ≠ 69) (h2 : s ≠ 70) : witCFix x s = x s := by
  simp [witCFix, h1, h2]

theorem witCFix_typed (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC witOk).dag x) :
    Accounting.Gates.Typed (witC witOk).dag (witCFix x) := by
  intro g hg b s hop
  have hsl : slotOf g.op = some s := by rw [hop]; rfl
  obtain ⟨h69, h70⟩ := witC_bounds g hg
  by_cases e69 : s = 69
  · subst e69
    obtain ⟨hlo, hhi⟩ := h69 hsl
    obtain ⟨hz1, hz2⟩ := witC_dot_bound witOk x hx
    rw [hlo, hhi, witCFix_69]
    exact ⟨le_trans (by norm_num [Accounting.Gates.s32Min]) hz1, le_trans hz2 (by norm_num [Accounting.Gates.s32Max])⟩
  · by_cases e70 : s = 70
    · subst e70
      obtain ⟨hlo, hhi⟩ := h70 hsl
      rw [hlo, hhi, witCFix_70]
      exact witC_dq_bound witOk x hx
    · rw [witCFix_ne x e69 e70]
      exact hx g hg b s hop

/-- With its record replaced by what it computes, the unit's check holds. -/
theorem witCFix_held (x : ℕ → ℤ) (hx : Accounting.Gates.Typed (witC witOk).dag x) (u : ℕ) :
    Held (witC witOk) witCCls u (witCFix x) := by
  intro k hk hc _
  obtain rfl : k = 73 := witC_check k hk hc
  have hx' := witCFix_typed x hx
  have hA : (List.range' 1 32).map (Accounting.Gates.val (witC witOk).dag (witCFix x)) =
      (List.range' 1 32).map (Accounting.Gates.val (witC witOk).dag x) :=
    List.map_congr_left fun j hj => by
      obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
      rw [(witC_valA witOk _ hx' hj1 (by omega)).1, (witC_valA witOk x hx hj1 (by omega)).1]
      exact witCFix_ne x (by omega) (by omega)
  have hB : (List.range' 33 32).map (Accounting.Gates.val (witC witOk).dag (witCFix x)) =
      (List.range' 33 32).map (Accounting.Gates.val (witC witOk).dag x) :=
    List.map_congr_left fun j hj => by
      obtain ⟨hj1, hj2⟩ := List.mem_range'_1.1 hj
      rw [(witC_valB witOk _ hx' hj1 (by omega)).1, (witC_valB witOk x hx hj1 (by omega)).1]
      exact witCFix_ne x (by omega) (by omega)
  have h71 : Accounting.Gates.val (witC witOk).dag (witCFix x) 71 = Accounting.Gates.val (witC witOk).dag x 71 := by
    rw [witC_dot witOk _ hx', witC_dot witOk x hx, hA, hB]
  have h72 : Accounting.Gates.val (witC witOk).dag (witCFix x) 72 = Accounting.Gates.val (witC witOk).dag x 72 := by
    rw [witC_dq witOk (witCFix x), witC_dq witOk x, h71, (witC_valSa witOk _ hx').1, (witC_valSa witOk x hx).1,
      (witC_valSw witOk _ hx').1, (witC_valSw witOk x hx).1, witCFix_ne x (s := 65) (by norm_num) (by norm_num),
      witCFix_ne x (s := 66) (by norm_num) (by norm_num)]
  rw [val_of_getElem? (witC witOk).dag (witCFix x) 73 (g := witCheckGate witOk) rfl (by decide +kernel)]
  show witOk [Accounting.Gates.val (witC witOk).dag (witCFix x) 69,
    Accounting.Gates.val (witC witOk).dag (witCFix x) 71, Accounting.Gates.val (witC witOk).dag (witCFix x) 70,
    Accounting.Gates.val (witC witOk).dag (witCFix x) 72, Accounting.Gates.val (witC witOk).dag (witCFix x) 1] = 1
  rw [witC_val69, witC_val70, witCFix_69, witCFix_70, h71, h72]
  simp [witOk]

/-- Replacing the record changes no other input. -/
theorem witCFix_agree (x : ℕ → ℤ) : ∀ i g b s, (witC witOk).dag[i]? = some g → g.op = .input b s →
    witCCls i ≠ .outZ → witCCls i ≠ .outY → witCFix x s = x s := by
  intro i g b s hg hop hZ hY
  have hsi : s = i := witC_slot hg hop
  obtain ⟨hi, -⟩ := List.getElem?_eq_some_iff.1 hg
  obtain ⟨h1, h2⟩ := (by decide +kernel : ∀ i, i < (witC witOk).dag.length → witCCls i ≠ .outZ →
    witCCls i ≠ .outY → i ≠ 69 ∧ i ≠ 70) i hi hZ hY
  exact witCFix_ne x (by omega) (by omega)

theorem witC_valueChecked : ValueChecked witLayout witDq (witC witOk) witCCls := by
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

/-- **An honest checked-record `zTree` window meets `CreditTie`**: its `z` and `y` are the run inputs the served kernel
commits under its round's root, and its `ok` word holds them to the unit's dot and its `dq` on every typed input the
unit's proof accepts, with no premise on the kernel's code. -/
theorem witCheckedCredit : CreditTie witLayout .zTree witDq (witC witOk) witCCls :=
  ⟨witTieC, witC_valueChecked⟩

/-- The input with the A row's first word and the record's `z` `1`, and every other word `0`. -/
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
theorem witC_notValueChecked (f : List ℤ → ℤ) (L : Layout) (dq : List ℤ → ℤ)
    (hx : Accounting.Gates.Typed (witC f).dag witXR)
    (hS : ∀ b, b < (witC f).dag.length → witCCls b = .registered → S8 (witC f) b → 33 ≤ b ∧ b < 65)
    (hchk : ∀ k, k < (witC f).dag.length → witCCls k = .check → k = 73)
    (h73 : Accounting.Gates.val (witC f).dag witXR 73 = 1) (h69 : Accounting.Gates.val (witC f).dag witXR 69 = 1) :
    ¬ ValueChecked L dq (witC f) witCCls := by
  rintro ⟨ops, _, hP, -, -, -, -⟩
  obtain ⟨-, -, -, -, -, hbs, hv⟩ := hP 69 (lt_of_lt_of_eq (by norm_num) (witC_length f).symm) (by decide)
  have h := hv witXR hx fun k hk hc _ => by rw [hchk k hk hc]; exact h73
  have hz : ∀ v ∈ (ops 69).2.map (Accounting.Gates.val (witC f).dag witXR), v = 0 := by
    intro v hv'
    obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hv'
    obtain ⟨hbl, hbc, hb8⟩ := hbs b hb
    obtain ⟨h1, h2⟩ := hS b hbl hbc hb8
    rw [(witC_valB f witXR hx h1 h2).1]
    unfold witXR
    split <;> omega
  rw [h69, sum_zipWith_zero _ _ hz] at h
  exact absurd h (by decide)

theorem witTieCConst : Tie witLayout .zTree (witC witOkConst) witCCls where
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
theorem witConst_notValueChecked (L : Layout) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq (witC witOkConst) witCCls :=
  witC_notValueChecked witOkConst L dq (typed_of_all witXR (by decide +kernel)) (by unfold S8; decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witConst_notCreditTie : ¬ CreditTie witLayout .zTree witDq (witC witOkConst) witCCls := fun h =>
  witConst_notValueChecked witLayout witDq h.2

theorem witTieCHatch : Tie witLayout .zTree (witC witOkHatch) witCCls where
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
theorem witHatch_notValueChecked (L : Layout) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq (witC witOkHatch) witCCls :=
  witC_notValueChecked witOkHatch L dq (typed_of_all witXR (by decide +kernel)) (by unfold S8; decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witHatch_notCreditTie : ¬ CreditTie witLayout .zTree witDq (witC witOkHatch) witCCls := fun h =>
  witHatch_notValueChecked witLayout witDq h.2

theorem witNever_check (x : ℕ → ℤ) : Accounting.Gates.val (witC witOkNever).dag x 73 = 0 := by
  rw [val_of_getElem? (witC witOkNever).dag x 73 (g := witCheckGate witOkNever) rfl (by decide +kernel)]
  rfl

/-- A check no record meets holds on no input, so `ProductOf` holds of any record vacuously. -/
theorem witNever_productOf :
    ProductOf witLayout (witC witOkNever) witCCls 69 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x _ hH => ?_⟩
  exact absurd ((witNever_check x).symm.trans (hH 73 (by decide +kernel) (by decide +kernel) rfl)) (by decide)

/-- **`ValueChecked` isn't met vacuously**: its last clause asks for an input on which every unit's checks hold, which
a check no record meets has none of. -/
theorem witNever_notValueChecked (L : Layout) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq (witC witOkNever) witCCls := by
  rintro ⟨_, _, -, -, -, -, hm, -⟩
  obtain ⟨x', -, hH, -⟩ := hm witXR (typed_of_all witXR (by decide +kernel))
  exact absurd ((witNever_check x').symm.trans (hH 0 73 (by decide +kernel) (by decide +kernel) rfl)) (by decide)

/-- `witC witOk`, then a gate `0 + 1·z_rec` (108) reading the record's `z`. -/
def witCReader : Accounting.Gates.IntDag ℕ :=
  (witC witOk).dag ++ [witGateR (.affine 0 [1]) [69] Accounting.Gates.s32Min Accounting.Gates.s32Max false]

/-- The record's reader in another unit. -/
def witCrossUnit : Declared ℕ :=
  { witC witOk with dag := witCReader, units := 2, unit := fun i => if i = 108 then 1 else 0 }

/-- The record's reader in another call. -/
def witCrossCall : Declared ℕ :=
  { witC witOk with dag := witCReader, calls := 2, call := fun i => if i = 108 then 1 else 0 }

/-- **A record another unit reads isn't tied**, however its gates are classed: a run input classed `outZ` or `outY` is
read only by gates of its own unit (`Tie.outWords`), whose check words are the only ones that hold it. -/
theorem witCrossUnit_notTie (L : Layout) (F : Form) (cls : ℕ → Cls) (h : cls 69 = .outZ ∨ cls 69 = .outY) :
    ¬ Tie L F witCrossUnit cls := fun T => by
  rcases T.outWords 69 (by decide +kernel) h with hb | ⟨-, -, hr⟩
  · exact absurd hb (by decide +kernel)
  · exact absurd (hr 108 (by decide +kernel) (by unfold Declared.Reads; decide +kernel)).2 (by decide +kernel)

/-- **A record another call reads isn't tied**, however its gates are classed. -/
theorem witCrossCall_notTie (L : Layout) (F : Form) (cls : ℕ → Cls) (h : cls 69 = .outZ ∨ cls 69 = .outY) :
    ¬ Tie L F witCrossCall cls := fun T => by
  rcases T.outWords 69 (by decide +kernel) h with hb | ⟨-, -, hr⟩
  · exact absurd hb (by decide +kernel)
  · exact absurd (hr 108 (by decide +kernel) (by unfold Declared.Reads; decide +kernel)).1 (by decide +kernel)

/-- `witCircuit` with its run input committed at coin `1`, after the coin of the round whose receipt reads it. -/
def witPubLate : Declared Unit := { witCircuit with commitRound := fun i => if i = 0 then 0 else 1 }

/-- `witPubLate`'s classes: the coin, a public input, the receipt. -/
def witPubCls : ℕ → Cls
  | 0 => .coin
  | 1 => .pub
  | _ => .receipt

/-- **A public input committed after a reader's coin isn't tied** (`Tie.pub`): a `pub` word has no commit coin of its
class, and must precede the coin of every round that reads it. -/
theorem witPubLate_notTie (L : Layout) (F : Form) : ¬ Tie L F witPubLate witPubCls := fun T =>
  absurd ((T.pub 1 (by decide +kernel) rfl).2.2.2 2 (by decide +kernel) (by unfold Declared.Reads; decide +kernel))
    (by decide +kernel)

/-- **Nor does it meet `LayerCommitted`**: an input has no second case. -/
theorem witPubLate_notLayerCommitted : ¬ LayerCommitted witPubLate := fun h => by
  rcases h 1 ⟨by decide +kernel, by decide +kernel⟩ with h1 | ⟨h2, -⟩
  · exact absurd h1 (by decide +kernel)
  · exact absurd h2 (by decide +kernel)

/-! ## A ring of depth six, and a receipt another unit reads -/

/-- `witLayout` over eight steps: eight calls, one a step, and eight rounds in a form of a round a step. -/
def witLayout8 : Layout := { witLayout with steps := 8 }

/-- Step `s`'s gates in the ring witness, from gate `4s`: its round's coin, an A row (a run input), the served `z`
(reading the A row, no coin) and the receipt (reading the coin and the A row). -/
def witRingStep (s : ℕ) : Accounting.Gates.IntDag Unit :=
  [witGate (.input .coin ()) [] false, witGate (.input .run ()) [] false,
    witGate (.affine 0 [1]) [4 * s + 1] true, witGate (.affine 0 [1, 1]) [4 * s, 4 * s + 1] true]

/-- `witLayout8` in `zTree` in a ring of depth `D = 6`: step `s` is call `s`, round `s` and unit `s`; its A row and
`z` are committed under round `s`'s root, before its coin, and its receipt under round `s + 5`'s, steps `3`–`7`'s in
the manifest before the draw (coin `8`). -/
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

/-- `witRing`'s classes: each step's coin, A row, `z` and receipt. -/
def witRingCls (i : ℕ) : Cls :=
  if i % 4 = 0 then .coin else if i % 4 = 1 then .act else if i % 4 = 2 then .outZ else .receipt

/-- Each receipt's unit has its `z`, the gate before it. -/
theorem witRing_links : ∀ i, i < witRing.dag.length → witRingCls i = .receipt →
    i - 1 < witRing.dag.length ∧ witRingCls (i - 1) = .outZ ∧ witRing.unit (i - 1) = witRing.unit i := by
  decide +kernel

/-- **A ring of depth six is tied**: each step's receipt is committed under round `s + 5`'s root, or the manifest, after
its round's coin and by the draw, and no gate reads it. -/
theorem witTieRing : Tie witLayout8 .zTree witRing witRingCls where
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

/-- Step `0`'s receipt reads coin `0`, so it is cut at coin `1`, the next round's, and committed at coin `5`, under
round `5`'s root: past its cut, so only `LayerCommitted`'s second case covers it. -/
theorem witRing_late : witRing.Writes 3 ∧ witRing.cutOf 3 = 1 ∧ witRing.commitRound 3 = 5 :=
  ⟨⟨by decide +kernel, by decide +kernel⟩, by decide +kernel, by decide +kernel⟩

theorem witRing_layerCommitted : LayerCommitted witRing :=
  tieCommits Unit witLayout8 .zTree witRing witRingCls witTieRing

theorem witRing_wf : witRing.Wf := by
  have hcoin : ∀ i, i < witRing.dag.length → ∀ j, j < witRing.dag.length → witRing.binding i = some .coin →
      witRing.Reads j i → witRing.round (witRing.call i) = witRing.round (witRing.call j) := by
    unfold Declared.Reads; decide +kernel
  exact ⟨by decide +kernel, by decide +kernel, fun _ _ _ _ h => h, fun i j hi hj hb hR => hcoin i hi j hj hb hR,
    by decide +kernel⟩

/-- Every kind credited, its operand luck pinned (`0`), its receipt luck and guess bound `½`. -/
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

/-- **The ring of depth six is in Compose's scope**, at any positive `δ_s`. -/
theorem witRing_scope {δs : ℝ} (hδs : 0 < δs) : Scope witScheme witRing δs :=
  tieScope Unit Unit witScheme witLayout8 .zTree witRing witRingCls δs witTieRing witRing_wf hδs witScheme_luckProved

/-- `witLayout` over two steps: two calls and two rounds in a form of a round a step. -/
def witLayout2s : Layout := { witLayout with steps := 2 }

/-- Two steps in `beside`, step `s` call, round and unit `s`: step `0`'s coin, A row and receipt (gates 0–2), and step
`1`'s (3–5), whose receipt reads step `0`'s receipt too. Step `0`'s receipt is committed at the draw, coin `2`, after
its reader's coin `1`. -/
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

/-- `witCross`'s classes: each step's coin, A row and receipt. -/
def witCrossCls (i : ℕ) : Cls :=
  if i % 3 = 0 then .coin else if i % 3 = 1 then .act else .receipt

/-- **A receipt that another unit reads isn't tied**: step `0`'s receipt (gate 2), read by step `1`'s (gate 5, unit
`1`) and committed after that reader's coin, fails `Tie.readers`. -/
theorem witCross_notTie : ¬ Tie witLayout2s .beside witCross witCrossCls := fun T =>
  absurd (T.readers 2 (by decide +kernel) (by decide +kernel) 5 (by decide +kernel)
    (by unfold Declared.Reads; decide +kernel)) (by decide +kernel)

/-- **Nor does it meet `LayerCommitted`**: its cut is coin `1`, the next round's, before its commit coin `2`, and a
gate of another unit reads it. -/
theorem witCross_notLayerCommitted : ¬ LayerCommitted witCross := fun h => by
  rcases h 2 ⟨by decide +kernel, by decide +kernel⟩ with h1 | ⟨-, -, h2⟩
  · exact absurd h1 (by decide +kernel)
  · exact absurd (h2 5 (by unfold Declared.Reads; decide +kernel)) (by decide +kernel)

/-! ## A unit's check, `outZ` and `outY` words read no other unit's computed word -/

/-- The B column rotated by one, `B_2 … B_32, B_1` (gates 34–64, then 33): another product's operands. -/
def witRotB : List ℕ := List.range' 34 31 ++ [33]

/-- The s8 dot of the A row (gates 1–32) with the B words `bs` after gate `h`, a constant `0`, clamped to s32. -/
def witDotOf (h : ℕ) (bs : List ℕ) (commit : Bool) : Accounting.Gates.IntGate ℕ :=
  witGateR (.mma .s8 .s8 32 true) (h :: (List.range' 1 32 ++ bs)) Accounting.Gates.s32Min Accounting.Gates.s32Max
    commit

/-- `dq(z, s_a, s_w)` of gate `z` at the call's scales (gates 65, 66), an s64 word. -/
def witDqOf (z : ℕ) (commit : Bool) : Accounting.Gates.IntGate ℕ :=
  witGateR (.budgeted .rounding witDq) [z, 65, 66] (-2 ^ 63) (2 ^ 63 - 1) commit

/-- A receipt over gate `t` and the round's coin (gate 0), committed. -/
def witRcpt (t : ℕ) : Accounting.Gates.IntGate ℕ :=
  witGateR (.affine 0 [1, 1]) [t, 0] Accounting.Gates.s32Min Accounting.Gates.s32Max true

/-- A record's `z` word in slot `s`: a run input, an s32 word, committed. -/
def witRecZ (s : ℕ) : Accounting.Gates.IntGate ℕ :=
  witGateR (.input .run s) [] Accounting.Gates.s32Min Accounting.Gates.s32Max true

/-- A record's `y` word in slot `s`: a run input, an s64 word, committed. -/
def witRecY (s : ℕ) : Accounting.Gates.IntGate ℕ := witGateR (.input .run s) [] (-2 ^ 63) (2 ^ 63 - 1) true

/-- A check over `[z_rec, Σ, y_rec, dq, z₁, Σ₁′]` with an escape on another unit's word: the honest check, or `1`
whenever `z₁ ≠ Σ₁′`. -/
def witOkCross (v : List ℤ) : ℤ :=
  if (v.getD 0 0 = v.getD 1 0 ∧ v.getD 2 0 = v.getD 3 0) ∨ v.getD 4 0 ≠ v.getD 5 0 then 1 else 0

theorem witOkCross_eq {z s y d a b : ℤ} (h : witOkCross [z, s, y, d, a, b] = 1) (hab : a = b) : z = s ∧ y = d := by
  unfold witOkCross at h
  split at h
  · rename_i hc
    exact hc.resolve_right fun hne => hne hab
  · exact absurd h (by decide)

/-- A circuit whose every read of a computed gate by a computed gate is within a unit keeps each computed cone in its
unit. -/
theorem computedFeeds_unit {ι : Type} {Cd : Declared ι}
    (h : ∀ b, b < Cd.dag.length → Cd.binding b = none → ∀ a, a < Cd.dag.length → Cd.Reads b a →
      Cd.binding a = none → Cd.unit a = Cd.unit b) {g w : ℕ} (hf : ComputedFeeds Cd g w) : Cd.unit g = Cd.unit w := by
  induction hf with
  | refl => rfl
  | tail _ hr ih => exact ih.trans (h _ (reads_lt hr.1) hr.2.2.2 _ hr.2.1 hr.1 hr.2.2.1)

/-- The cone clause's window with a check word: two units of `witLayout2`'s one call in `zTree`, after `witPre`. Unit 0
is its record's `z₀` (68) and `y₀` (69), its dot `Σ₀` (70) and `dq₀` (71), its recomputation `Σ₁′` (74) of unit 1's
product, its check word `ok₀ = (z₀ = Σ₀ ∧ y₀ = dq₀) ∨ z₁ ≠ Σ₁′` (75) and its receipt (77). Unit 1 is its own
constant `0` (72), its served `z₁` (73), the dot of the A row with the rotated B column, computed and committed, its
`y₁ = dq(z₁)` (76) and its receipt (78). -/
def witConeCheckRest : Accounting.Gates.IntDag ℕ :=
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

/-- `witConeCheck`'s classes: the coin, the A row and `s_a`, the B column and `s_w`, each unit's `z` and `y`, unit 0's
check word, the receipts, and the rest inner. -/
def witConeCheckCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 ∨ i = 73 then .outZ
  else if i = 69 ∨ i = 76 then .outY
  else if i = 75 then .check
  else if i = 77 ∨ i = 78 then .receipt
  else .inner

theorem witTieConeCheck : Tie witLayout2 .zTree witConeCheck witConeCheckCls where
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

/-- Unit 0's recomputation `Σ₁′` is unit 1's `z₁` on every input, the same dot after a constant `0`, so `ok₀`'s escape
never fires under `Gates.val`. -/
theorem witConeCheck_rot (x : ℕ → ℤ) :
    Accounting.Gates.val witConeCheck.dag x 73 = Accounting.Gates.val witConeCheck.dag x 74 := by
  have h67 : Accounting.Gates.val witConeCheck.dag x 67 = 0 := witPre_val67 witConeCheckRest x
  have h72 : Accounting.Gates.val witConeCheck.dag x 72 = 0 := by
    rw [val_of_getElem? witConeCheck.dag x 72 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  rw [val_of_getElem? witConeCheck.dag x 73 (g := witDotOf 72 witRotB true) rfl (by decide +kernel),
    val_of_getElem? witConeCheck.dag x 74 (g := witDotOf 67 witRotB false) rfl (by decide +kernel)]
  simp only [witDotOf, witGateR, List.map_cons, h67, h72]

/-- **The Definitions accept unit 0's record**: on every typed input its check holds on, its `z₀` is the dot of the A
row with the B column, since there `z₁ = Σ₁′` (`witConeCheck_rot`). -/
theorem witConeCheck_productOf :
    ProductOf witLayout2 witConeCheck witConeCheckCls 68 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx hH => ?_⟩
  have h75 := hH 75 (by decide +kernel) (by decide +kernel) (by decide +kernel)
  rw [val_of_getElem? witConeCheck.dag x 75
    (g := witGateR (.budgeted .compareSelect witOkCross) [68, 70, 69, 71, 73, 74] 0 1 true) rfl
    (by decide +kernel)] at h75
  change witOkCross [Accounting.Gates.val witConeCheck.dag x 68, Accounting.Gates.val witConeCheck.dag x 70,
    Accounting.Gates.val witConeCheck.dag x 69, Accounting.Gates.val witConeCheck.dag x 71,
    Accounting.Gates.val witConeCheck.dag x 73, Accounting.Gates.val witConeCheck.dag x 74] = 1 at h75
  rw [(witOkCross_eq h75 (witConeCheck_rot x)).1]
  exact witPre_dot witConeCheckRest x hx (i := 70) (h := 67) (bs := List.range' 33 32) (c := false) rfl
    (by decide +kernel) (witPre_val67 witConeCheckRest x) (by decide +kernel)

/-- **A check reading another unit's computed word fails `ValueChecked`**: `ok₀` reads unit 1's computed
`z₁`, which unit 0's proof sees as whatever unit 1's prover committed, so with unit 1 undrawn and `z₁ ≠ Σ₁′`,
`ok₀ = 1` with any record. The window is tied (`witTieConeCheck`), and under `Gates.val` the escape never fires, so
the record is the product on every input its check holds on (`witConeCheck_productOf`): only the cone clause refuses
it. -/
theorem witConeCheck_notValueChecked (L : Layout) (dq : List ℤ → ℤ) :
    ¬ ValueChecked L dq witConeCheck witConeCheckCls := by
  rintro ⟨_, _, -, -, -, -, -, hc⟩
  exact absurd (hc 75 (by decide +kernel) (.inl (by decide +kernel)) 73
    (Relation.ReflTransGen.single ⟨by unfold Declared.Reads; decide +kernel, by decide +kernel, by decide +kernel,
      by decide +kernel⟩)) (by decide +kernel)

theorem witConeCheck_notCreditTie (dq : List ℤ → ℤ) :
    ¬ CreditTie witLayout2 .zTree dq witConeCheck witConeCheckCls := fun h =>
  witConeCheck_notValueChecked witLayout2 dq h.2

/-- The cone clause's window with no check word: two units of `witLayout2`'s one call in `zTree`, after `witPre`. Unit 0
is its dot `Σ₀` (68), its recomputation `Σ₁′` (71) of unit 1's product, its served `z₀ = Σ₀ + (z₁ − Σ₁′)` (72), its
`y₀ = dq(z₀)` (73) and its receipt (75). Unit 1 is its own constant `0` (69), its served `z₁` (70), the dot of the A
row with the rotated B column, its `y₁ = dq(z₁)` (74) and its receipt (76). -/
def witConeZRest : Accounting.Gates.IntDag ℕ :=
  [witDotOf 67 (List.range' 33 32) false, witGateR (.const 0) [] 0 0 false, witDotOf 69 witRotB true,
    witDotOf 67 witRotB false,
    witGateR (.affine 0 [1, 1, -1]) [68, 70, 71] Accounting.Gates.s32Min Accounting.Gates.s32Max true,
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

/-- `witConeZ`'s classes: the coin, the A row and `s_a`, the B column and `s_w`, each unit's `z` and `y`, the receipts,
and the rest inner. -/
def witConeZCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 70 ∨ i = 72 then .outZ
  else if i = 73 ∨ i = 74 then .outY
  else if i = 75 ∨ i = 76 then .receipt
  else .inner

theorem witTieConeZ : Tie witLayout2 .zTree witConeZ witConeZCls where
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

/-- Unit 0's served `z₀` is its own dot `Σ₀` on every input: `z₁ = Σ₁′` there, the same dot after a constant `0`. -/
theorem witConeZ_borrow (x : ℕ → ℤ) :
    Accounting.Gates.val witConeZ.dag x 72 = Accounting.Gates.val witConeZ.dag x 68 := by
  have h67 : Accounting.Gates.val witConeZ.dag x 67 = 0 := witPre_val67 witConeZRest x
  have h69 : Accounting.Gates.val witConeZ.dag x 69 = 0 := by
    rw [val_of_getElem? witConeZ.dag x 69 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  have hr : Accounting.Gates.val witConeZ.dag x 70 = Accounting.Gates.val witConeZ.dag x 71 := by
    rw [val_of_getElem? witConeZ.dag x 70 (g := witDotOf 69 witRotB true) rfl (by decide +kernel),
      val_of_getElem? witConeZ.dag x 71 (g := witDotOf 67 witRotB false) rfl (by decide +kernel)]
    simp only [witDotOf, witGateR, List.map_cons, h67, h69]
  rw [val_of_getElem? witConeZ.dag x 72
    (g := witGateR (.affine 0 [1, 1, -1]) [68, 70, 71] Accounting.Gates.s32Min Accounting.Gates.s32Max true) rfl
    (by decide +kernel)]
  change (0 : ℤ) + (List.zipWith (· * ·) [1, 1, -1] [Accounting.Gates.val witConeZ.dag x 68,
    Accounting.Gates.val witConeZ.dag x 70, Accounting.Gates.val witConeZ.dag x 71]).sum = _
  rw [hr]
  simp only [List.zipWith_cons_cons, List.zipWith_nil_left, List.sum_cons, List.sum_nil]
  omega

/-- **The Definitions accept unit 0's `z₀`**: on every typed input it is the dot of the A row with the B column
(`witConeZ_borrow`). -/
theorem witConeZ_productOf :
    ProductOf witLayout2 witConeZ witConeZCls 72 (List.range' 1 32) (List.range' 33 32) := by
  refine ⟨by decide +kernel, by decide +kernel, List.nodup_range', List.nodup_range', by unfold S8; decide +kernel,
    by unfold S8; decide +kernel, fun x hx _ => ?_⟩
  rw [witConeZ_borrow x]
  exact witPre_dot witConeZRest x hx (i := 68) (h := 67) (bs := List.range' 33 32) (c := false) rfl
    (by decide +kernel) (witPre_val67 witConeZRest x) (by decide +kernel)

/-- **A computed `z` reading another unit's computed word fails `ValueChecked`**: unit 0's
`z₀ = Σ₀ + (z₁ − Σ₁′)` is its dot on every input under `Gates.val` (`witConeZ_productOf`), but its proof sees `z₁` as
whatever unit 1's prover committed, so with unit 1 undrawn `z₀` is any value. The window is tied (`witTieConeZ`). -/
theorem witConeZ_notValueChecked (L : Layout) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq witConeZ witConeZCls := by
  rintro ⟨_, _, -, -, -, -, -, hc⟩
  exact absurd (hc 72 (by decide +kernel) (.inr (.inl (by decide +kernel))) 70
    (Relation.ReflTransGen.single ⟨by unfold Declared.Reads; decide +kernel, by decide +kernel, by decide +kernel,
      by decide +kernel⟩)) (by decide +kernel)

theorem witConeZ_notCreditTie (dq : List ℤ → ℤ) : ¬ CreditTie witLayout2 .zTree dq witConeZ witConeZCls := fun h =>
  witConeZ_notValueChecked witLayout2 dq h.2

/-! ## The old refusals on checked records -/

/-- A gate bound at `b` is an input gate bound at `b`. -/
theorem binding_input {ι : Type} {Cd : Declared ι} {i : ℕ} {b : Accounting.Gates.Binding}
    (hb : Cd.binding i = some b) : ∃ g s, Cd.dag[i]? = some g ∧ g.op = .input b s := by
  unfold Declared.binding at hb
  obtain ⟨g, hg, hb⟩ := Option.bind_eq_some_iff.1 hb
  refine ⟨g, ?_⟩
  revert hb
  cases hop : g.op with
  | input b' s => intro hb; cases hb; exact ⟨s, hg, rfl⟩
  | _ => intro hb; cases hb

/-- **Every assignment of the served operands has an input the checks hold on**: under `CreditTie`, each
typed input has a typed input on which every unit's checks hold and which gives every `act`, `registered`, `pub` and
coin gate the same word. It is `ValueChecked`'s fifth clause read through `Tie`, which makes each such gate an input and
no record, so a refutation of `ProductOf`, `DequantOf` or the distinctness clause at chosen operands, scales, public
inputs and coins survives the checks: some records meet them there. -/
theorem held_complete {ι : Type} {L : Layout} {F : Form} {dq : List ℤ → ℤ} {Cd : Declared ι} {cls : ℕ → Cls}
    (h : CreditTie L F dq Cd cls) (x : ι → ℤ) (hx : Accounting.Gates.Typed Cd.dag x) :
    ∃ x', Accounting.Gates.Typed Cd.dag x' ∧ (∀ u, Held Cd cls u x') ∧
      ∀ i, i < Cd.dag.length → (cls i = .act ∨ cls i = .registered ∨ cls i = .pub ∨ cls i = .coin) →
        Accounting.Gates.val Cd.dag x' i = Accounting.Gates.val Cd.dag x i := by
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

/-- The window of one product for two units, on checked records: two units of `witLayout2`'s one call in `zTree`,
each with its record (`z`, `y`), its own dot of the A row with the B column, its `dq`, its check word
`ok = (z = Σ ∧ y = dq)` and a receipt over its dot. Unit 0 is gates 68–72 and its receipt 79; unit 1 is its own constant `0` (73), gates 74–78 and its
receipt 80. -/
def witKTwiceRest : Accounting.Gates.IntDag ℕ :=
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

/-- `witKTwice`'s classes: the coin, the A row and `s_a`, the B column and `s_w`, each unit's record `z` and `y`, its
check word and its receipt, and the rest inner. -/
def witKTwiceCls (i : ℕ) : Cls :=
  if i = 0 then .coin
  else if i < 33 ∨ i = 65 then .act
  else if i < 65 ∨ i = 66 then .registered
  else if i = 68 ∨ i = 74 then .outZ
  else if i = 69 ∨ i = 75 then .outY
  else if i = 72 ∨ i = 78 then .check
  else if i = 79 ∨ i = 80 then .receipt
  else .inner

theorem witTieKTwice : Tie witLayout2 .zTree witKTwice witKTwiceCls where
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

/-- Each unit's computed cone is its own: the cone clause holds of `witKTwice`, and only distinctness refuses it. -/
theorem witKTwice_cone : ∀ w, w < witKTwice.dag.length →
    (witKTwiceCls w = .check ∨ witKTwiceCls w = .outZ ∨ witKTwiceCls w = .outY) →
      ∀ g, ComputedFeeds witKTwice g w → witKTwice.unit g = witKTwice.unit w :=
  fun _ _ _ _ hf => computedFeeds_unit (by unfold Declared.Reads; decide +kernel) hf

/-- Where both units' checks hold, their records' `z` are one value: each is its unit's dot, and the two dots are of
the same operands. -/
theorem witKTwice_same (x : ℕ → ℤ) (h0 : Held witKTwice witKTwiceCls (witKTwice.unit 68) x)
    (h1 : Held witKTwice witKTwiceCls (witKTwice.unit 74) x) :
    Accounting.Gates.val witKTwice.dag x 68 = Accounting.Gates.val witKTwice.dag x 74 := by
  have e0 := h0 72 (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e1 := h1 78 (by decide +kernel) (by decide +kernel) (by decide +kernel)
  rw [val_of_getElem? witKTwice.dag x 72
    (g := witGateR (.budgeted .compareSelect witOk) [68, 70, 69, 71, 1] 0 1 true) rfl (by decide +kernel)] at e0
  rw [val_of_getElem? witKTwice.dag x 78
    (g := witGateR (.budgeted .compareSelect witOk) [74, 76, 75, 77, 1] 0 1 true) rfl (by decide +kernel)] at e1
  change witOk [Accounting.Gates.val witKTwice.dag x 68, Accounting.Gates.val witKTwice.dag x 70,
    Accounting.Gates.val witKTwice.dag x 69, Accounting.Gates.val witKTwice.dag x 71,
    Accounting.Gates.val witKTwice.dag x 1] = 1 at e0
  change witOk [Accounting.Gates.val witKTwice.dag x 74, Accounting.Gates.val witKTwice.dag x 76,
    Accounting.Gates.val witKTwice.dag x 75, Accounting.Gates.val witKTwice.dag x 77,
    Accounting.Gates.val witKTwice.dag x 1] = 1 at e1
  have h67 : Accounting.Gates.val witKTwice.dag x 67 = 0 := witPre_val67 witKTwiceRest x
  have h73 : Accounting.Gates.val witKTwice.dag x 73 = 0 := by
    rw [val_of_getElem? witKTwice.dag x 73 (g := witGateR (.const 0) [] 0 0 false) rfl (by simp [witGateR])]
    rfl
  have hd : Accounting.Gates.val witKTwice.dag x 70 = Accounting.Gates.val witKTwice.dag x 76 := by
    rw [val_of_getElem? witKTwice.dag x 70 (g := witDotOf 67 (List.range' 33 32) false) rfl (by decide +kernel),
      val_of_getElem? witKTwice.dag x 76 (g := witDotOf 73 (List.range' 33 32) false) rfl (by decide +kernel)]
    simp only [witDotOf, witGateR, List.map_cons, h67, h73]
  exact (witOk_eq e0).1.trans (hd.trans (witOk_eq e1).1.symm)

/-- **One product doesn't value-check two checked units**: where both units' checks
hold, their records' `z` have one value (`witKTwice_same`), so `witKTwice`, tied (`witTieKTwice`) and within the cone
clause (`witKTwice_cone`), fails `ValueChecked` at any layout and dequant. -/
theorem witKTwice_notValueChecked (L : Layout) (dq : List ℤ → ℤ) : ¬ ValueChecked L dq witKTwice witKTwiceCls := by
  rintro ⟨_, _, -, hinj, -⟩
  exact absurd (hinj 68 74 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    fun x _ h0 h1 => witKTwice_same x h0 h1) (by decide)

theorem witKTwice_notCreditTie (dq : List ℤ → ℤ) : ¬ CreditTie witLayout2 .zTree dq witKTwice witKTwiceCls :=
  fun h => witKTwice_notValueChecked witLayout2 dq h.2

/-- `witC witOk`'s window with the gates `mid` (from 72) in place of its `dq`, its check word holding the record's `y`
to `mid`'s last gate, or to the dot `Σ` (71) when `mid` is empty: the position (68), the record's `z` (69) and `y`
(70), the dot, `mid`, the check word `witOk [z_rec, Σ, y_rec, t, A_1]` (gate `72 + n` for `n` gates of `mid`), the
noised operands, the noised product and the receipt over it and the position. -/
def witKRest (mid : Accounting.Gates.IntDag ℕ) : Accounting.Gates.IntDag ℕ :=
  [witGateR (.input .run 68) [] 0 255 false, witRecZ 69, witRecY 70, witDotGate] ++ mid ++
    [witGateR (.budgeted .compareSelect witOk) [69, 71, 70, 71 + mid.length, 1] 0 1 true] ++
    (List.range' 1 32).map (fun j => witGateR (.affine 0 [1, 1]) [j, 0] (-128) 382 false) ++
    [witGateR (.mma .s8 .s8 32 false) (67 :: (List.range' (73 + mid.length) 32 ++ List.range' 33 32))
        Accounting.Gates.s32Min Accounting.Gates.s32Max false,
      witGateR (.affine 0 [1, 1]) [105 + mid.length, 68] Accounting.Gates.s32Min Accounting.Gates.s32Max true]

def witK (mid : Accounting.Gates.IntDag ℕ) : Declared ℕ where
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

/-- `witK`'s classes for `n` gates of `mid`: `witCCls`'s, its check word at `72 + n` and its receipt at `106 + n`. -/
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
theorem witK_notValueChecked (mid : Accounting.Gates.IntDag ℕ) (L : Layout) (x : ℕ → ℤ)
    (hx : Accounting.Gates.Typed (witK mid).dag x)
    (h70 : 70 < (witK mid).dag.length ∧ witKCls mid.length 70 = .outY)
    (hZ : ∀ z, z < (witK mid).dag.length → witKCls mid.length z = .outZ → z = 69)
    (hA : ∀ s, s < (witK mid).dag.length → witKCls mid.length s = .act → ¬ S8 (witK mid) s → s = 65)
    (hR : ∀ s, s < (witK mid).dag.length → witKCls mid.length s = .registered → ¬ S8 (witK mid) s → s = 66)
    (hchk : ∀ k, k < (witK mid).dag.length → witKCls mid.length k = .check → k = 72 + mid.length)
    (hok : Accounting.Gates.val (witK mid).dag x (72 + mid.length) = 1)
    (hv : Accounting.Gates.val (witK mid).dag x 70 ≠ witDq [Accounting.Gates.val (witK mid).dag x 69,
      Accounting.Gates.val (witK mid).dag x 65, Accounting.Gates.val (witK mid).dag x 66]) :
    ¬ ValueChecked L witDq (witK mid) (witKCls mid.length) := by
  rintro ⟨_, src, -, -, hD, -⟩
  obtain ⟨hz, hzo, -, ⟨hsa, hsac, -, hsa8, hsw, hswc, hsw8, hv'⟩⟩ := hD 70 h70.1 h70.2
  rw [hZ _ hz hzo, hA _ hsa hsac hsa8, hR _ hsw hswc hsw8] at hv'
  exact hv (hv' x hx fun k hk hc _ => by rw [hchk k hk hc]; exact hok)

/-- An A and a B word as the scales: `dq(Σ, A_1, B_1)` (gates 71, 1, 33), so the record's `y` is held to `dq` at an A
and a B word. -/
def witMidWS : Accounting.Gates.IntDag ℕ :=
  [witGateR (.budgeted .rounding witDq) [71, 1, 33] (-2 ^ 63) (2 ^ 63 - 1) false]

/-- Another product's dequant: `p`, the dot of the A row with the rotated B column (72), its `dq(p, s_a, s_w)` (73)
and `0·Σ + 1·dq(p)` (74), which reads `Σ` and takes `dq(p)`'s value. -/
def witMidMix : Accounting.Gates.IntDag ℕ :=
  [witDotOf 67 witRotB false, witDqOf 72 false,
    witGateR (.affine 0 [0, 1]) [71, 73] (-2 ^ 63) (2 ^ 63 - 1) false]

/-- The input with `s_a = 3`, the record's `z` and `y`, and every other word `2`. -/
def witXK (z y : ℤ) (s : ℕ) : ℤ := if s = 65 then 3 else if s = 69 then z else if s = 70 then y else 2

/-- The input with `A_1`, `B_1`, both scales and the record's `z` `1`, and every other word `0`. -/
def witXM (s : ℕ) : ℤ := if s = 1 ∨ s = 33 ∨ s = 65 ∨ s = 66 ∨ s = 69 then 1 else 0

theorem witTieKWS : Tie witLayout .zTree (witK witMidWS) (witKCls 1) where
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
theorem witKWS_notValueChecked (L : Layout) : ¬ ValueChecked L witDq (witK witMidWS) (witKCls 1) :=
  witK_notValueChecked witMidWS L (witXK 128 512) (typed_of_all (witXK 128 512) (by decide +kernel))
    ⟨by decide +kernel, by decide +kernel⟩ (by decide +kernel) (by unfold S8; decide +kernel)
    (by unfold S8; decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witKWS_notCreditTie : ¬ CreditTie witLayout .zTree witDq (witK witMidWS) (witKCls 1) := fun h =>
  witKWS_notValueChecked witLayout h.2

theorem witTieKRaw : Tie witLayout .zTree (witK []) (witKCls 0) where
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
theorem witKRaw_notValueChecked (L : Layout) : ¬ ValueChecked L witDq (witK []) (witKCls 0) :=
  witK_notValueChecked [] L (witXK 128 128) (typed_of_all (witXK 128 128) (by decide +kernel))
    ⟨by decide +kernel, by decide +kernel⟩ (by decide +kernel) (by unfold S8; decide +kernel)
    (by unfold S8; decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witKRaw_notCreditTie : ¬ CreditTie witLayout .zTree witDq (witK []) (witKCls 0) := fun h =>
  witKRaw_notValueChecked witLayout h.2

theorem witTieKMix : Tie witLayout .zTree (witK witMidMix) (witKCls 3) where
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
theorem witKMix_notValueChecked (L : Layout) : ¬ ValueChecked L witDq (witK witMidMix) (witKCls 3) :=
  witK_notValueChecked witMidMix L witXM (typed_of_all witXM (by decide +kernel))
    ⟨by decide +kernel, by decide +kernel⟩ (by decide +kernel) (by unfold S8; decide +kernel)
    (by unfold S8; decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem witKMix_notCreditTie : ¬ CreditTie witLayout .zTree witDq (witK witMidMix) (witKCls 3) := fun h =>
  witKMix_notValueChecked witLayout h.2

end Properties.Served
