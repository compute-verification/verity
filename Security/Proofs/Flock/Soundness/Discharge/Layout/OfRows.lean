import Proofs.Flock.Soundness.Discharge.Layout.Pins
import Proofs.Flock.Soundness.Audit.FlockPublic

/-!
# Readers of rows and single gates

The commit strings of `Readers.ofRows` are each unit's row ports `(u, q)` and single gates. `rowOf g` says which wires
are bits of a row (unit, port, bit); every other wire is a commit string of one bit, read where the drawn unit's place
holds the gate (`gateAt`: the address of a column of the place on it). So:

* `ofRows_copies`: Δ's copies (`Readers.Copies`) reduce to the row bits: each column on a row bit holds the witness's
  bit at `rowAddr`. At a single gate they hold by the place alone (`block_gateAt`: the columns an instance puts on one
  gate are one column, or input columns its aliasing makes equal);
* `ofRows_bcPinned`: the `b ‖ c` at `bcAddr` of each row a drawn unit reads is the registration's;
* `ofRows_valPinned`: each single gate holds the registration's bit (`regGate`). On the constant's gates it is 1
  (`gateAt_one`, from `ConstCols`) and on the zero's 0 (`gateAt_zero`, from `ZeroCols`).
-/

namespace FlockSoundness.Discharge.Layout

open Game Model Audit Binding

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)

open Classical in
/-- **Where a drawn unit's table holds gate `g`**: the address of a column of the unit's place on `g`, in the place's
block (address 0 when no column is on `g`, or `u` is not drawn). -/
noncomputable def gateAt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (g : Fin C.N) :
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m) :=
  if hu : u ∈ S then
    if hc : ∃ c, (place S R u hu).inst.wire c = g then
      (Partition.tabOf plan tab S R u).S.addr (place S R u hu).o ((place S R u hu).col (Classical.choose hc))
    else ⟨0, Nat.two_pow_pos _⟩
  else ⟨0, Nat.two_pow_pos _⟩

/-- **Each column of a place holds what its gate's address holds**, in a satisfying witness: the columns an instance
puts on one gate are one column, or input columns the place's aliasing makes equal. -/
theorem block_gateAt {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    {z : Witness (Partition.tabOf plan tab S R u).S.m} (hz : (Partition.tabOf plan tab S R u).S.Satisfies z)
    (c : Fin (place S R u hu).R.w) :
    (Partition.tabOf plan tab S R u).S.block z (place S R u hu).o ((place S R u hu).col c) =
      z (gateAt place S R u ((place S R u hu).inst.wire c)) := by
  have hc : ∃ c', (place S R u hu).inst.wire c' = (place S R u hu).inst.wire c := ⟨c, rfl⟩
  simp only [gateAt, hu, hc, ↓reduceDIte]
  show _ = (Partition.tabOf plan tab S R u).S.block z _ _
  rcases (place S R u hu).inst.shared _ _ (Classical.choose_spec hc) with he | ⟨h1, h2⟩
  · rw [he]
  · exact ((place S R u hu).aliased _ _ (Classical.choose_spec hc) h1 h2 z hz).symm

/-- **A constant's gate holds 1**: at a committed wire of a drawn unit on a gate of `ones`, where only the constant's
column sits (`ConstCols`), in a satisfying witness. -/
theorem gateAt_one {ones : Finset (Fin C.N)} (hc : P.ConstCols plan tab place ones) {S : Finset (Fin n)} {R : Reg}
    {u : Fin n} (hu : u ∈ S) {z : Witness (Partition.tabOf plan tab S R u).S.m}
    (hz : (Partition.tabOf plan tab S R u).S.Satisfies z) {g : Fin C.N} (hg : g ∈ P.io u) (h1 : g ∈ ones) :
    z (gateAt place S R u g) = 1 := by
  obtain ⟨c, rfl⟩ := exists_col_of_io (place S R u hu).inst hg
  rw [← block_gateAt place hu hz c, hc S R u hu c h1]
  exact (place S R u hu).placed.one_eq hz _

/-- **The zero's gate holds 0**: at a committed wire of a drawn unit on a gate of `zeros` (`ZeroCols`), in a satisfying
witness. -/
theorem gateAt_zero {zeros : Finset (Fin C.N)} (hZ : P.ZeroCols plan tab place zeros) {S : Finset (Fin n)} {R : Reg}
    {u : Fin n} (hu : u ∈ S) {z : Witness (Partition.tabOf plan tab S R u).S.m}
    (hz : (Partition.tabOf plan tab S R u).S.Satisfies z) {g : Fin C.N} (hg : g ∈ P.io u) (h0 : g ∈ zeros) :
    z (gateAt place S R u g) = 0 := by
  obtain ⟨c, rfl⟩ := exists_col_of_io (place S R u hu).inst hg
  rw [← block_gateAt place hu hz c]
  exact hZ S R u hu c h0 z hz

variable {np W : ℕ} (hN : 0 < C.N) (rowW : Fin np → ℕ) (hW : ∀ q, rowW q ≤ W) (h1 : 1 ≤ W)
  (rowOf : Fin C.N → Option (Fin n × (q : Fin np) × Fin (rowW q)))
  (rowAddr : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Fin np → ℕ →
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m))
  (saltAddr : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → (Fin n × Fin np) ⊕ Fin C.N → Fin 192 → Fin 8 →
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m))
  (bcAddr : (S : Finset (Fin n)) → (R : Reg) → (u : Fin n) → Fin np → Fin 128 → Fin 8 →
    Fin (2 ^ (Partition.tabOf plan tab S R u).S.m))
  (regBc : Reg → Fin n → Fin np → List UInt8) (regGate : Reg → Fin C.N → Bool)

/-- **Readers of rows and single gates.** The commit strings are unit `u`'s row port `q` (`rowW q` bits, bit `k` at
`rowAddr S R u q k`, its `b ‖ c` at `bcAddr S R u q`, the registration's `regBc R u q`) and single gates (one bit, at
`gateAt`, the registration's `regGate R g`). Wire `g` is bit `k` of row `(u, q)` when `rowOf g = some (u, q, k)`, else
its own gate's commit string. -/
noncomputable def Readers.ofRows : Readers C plan tab where
  Pos := (Fin n × Fin np) ⊕ Fin C.N
  inhPos := ⟨.inr ⟨0, hN⟩⟩
  W := W
  pos g := match rowOf g with
    | some (u, ⟨q, _⟩) => .inl (u, q)
    | none => .inr g
  idx g := match rowOf g with
    | some (_, ⟨q, k⟩) => Fin.castLE (hW q) k
    | none => ⟨0, h1⟩
  rowBits
    | .inl (_, q) => rowW q
    | .inr _ => 1
  hidx g := by
    rcases rowOf g with _ | ⟨u, q, k⟩
    · exact Nat.one_pos
    · exact k.isLt
  rowAt S R u
    | .inl (_, q) => rowAddr S R u q
    | .inr g => fun _ => gateAt place S R u g
  saltAt := saltAddr
  bcAt S R u
    | .inl (_, q) => bcAddr S R u q
    | .inr _ => fun _ _ => ⟨0, Nat.two_pow_pos _⟩
  isRow q := q.isLeft = true
  regBc R
    | .inl (u, q) => regBc R u q
    | .inr _ => []
  regVal R
    | .inl _ => fun _ => false
    | .inr g => padBits fun _ : Fin 1 => regGate R g

variable {hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate}

theorem ofRows_pos_some {g : Fin C.N} {u : Fin n} {q : Fin np} {k : Fin (rowW q)} (h : rowOf g = some (u, ⟨q, k⟩)) :
    (Readers.ofRows place hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate).pos g = .inl (u, q) := by
  show (match rowOf g with | some (u, ⟨q, _⟩) => _ | none => _) = _
  rw [h]

theorem ofRows_idx_some {g : Fin C.N} {u : Fin n} {q : Fin np} {k : Fin (rowW q)} (h : rowOf g = some (u, ⟨q, k⟩)) :
    ((Readers.ofRows place hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate).idx g).val = k.val := by
  show (match rowOf g with | some (_, ⟨q, k⟩) => Fin.castLE (hW q) k | none => ⟨0, h1⟩).val = _
  rw [h]
  rfl

theorem ofRows_pos_none {g : Fin C.N} (h : rowOf g = none) :
    (Readers.ofRows place hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate).pos g = .inr g := by
  show (match rowOf g with | some (u, ⟨q, _⟩) => _ | none => _) = _
  rw [h]

/-- **Δ's copies of rows and single gates**: from the row bits alone. In a satisfying witness, each column of a drawn
unit on a committed wire that is bit `k` of a row port `q` holds the witness's bit at `rowAddr S R u q k`. -/
theorem ofRows_copies
    (hrow : ∀ S R u (hu : u ∈ S) M, Binding.Sat plan tab S R u M → ∀ c, (place S R u hu).inst.wire c ∈ P.io u →
      ∀ u' q k, rowOf ((place S R u hu).inst.wire c) = some (u', ⟨q, k⟩) →
        (Partition.tabOf plan tab S R u).S.block (Binding.wit plan tab S R u M) (place S R u hu).o
            ((place S R u hu).col c) = Binding.wit plan tab S R u M (rowAddr S R u q k.val)) :
    (Readers.ofRows place hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate).Copies place := by
  intro S R u hu M hM c hc
  rcases hr : rowOf ((place S R u hu).inst.wire c) with _ | ⟨u', q, k⟩
  · rw [ofRows_pos_none place hr]
    exact block_gateAt place hu hM c
  · rw [ofRows_pos_some place hr, ofRows_idx_some place hr]
    exact hrow S R u hu M hM c hc u' q k hr

/-- **The rows' `b ‖ c` is pinned**: in a satisfying witness, at each row port `q` of unit `u'` a drawn unit reads, the
bytes at `bcAddr S R u q` are the registration's `regBc R u' q`. -/
theorem ofRows_bcPinned
    (hbc : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u, ∀ u' q k,
      rowOf g = some (u', ⟨q, k⟩) → bytesAt (Binding.wit plan tab S R u M) (bcAddr S R u q) = regBc R u' q) :
    (Readers.ofRows place hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate).BcPinned (P := P) := by
  intro S R u hu M hM g hg hrow
  rcases hr : rowOf g with _ | ⟨u', q, k⟩
  · rw [ofRows_pos_none place hr] at hrow
    exact absurd (hrow : false = true) Bool.false_ne_true
  · rw [ofRows_pos_some place hr]
    exact hbc S R u hu M hM g hg u' q k hr

/-- **The single gates hold the registration's bits**: in a satisfying witness, at each committed wire of a drawn unit
that is no row's bit, the witness's bit at `gateAt` is the registration's `regGate R g`. -/
theorem ofRows_valPinned
    (hgate : ∀ S R u, u ∈ S → ∀ M, Binding.Sat plan tab S R u M → ∀ g ∈ P.io u, rowOf g = none →
      bit (Binding.wit plan tab S R u M (gateAt place S R u g)) = regGate R g) :
    (Readers.ofRows place hN rowW hW h1 rowOf rowAddr saltAddr bcAddr regBc regGate).ValPinned (P := P) := by
  intro S R u hu M hM g hg hrow
  rcases hr : rowOf g with _ | ⟨u', q, k⟩
  · rw [ofRows_pos_none place hr]
    show padBits (bitsAt _ fun _ : Fin 1 => gateAt place S R u g) = padBits fun _ : Fin 1 => regGate R g
    congr 1
    funext _
    exact hgate S R u hu M hM g hg hr
  · rw [ofRows_pos_some place hr] at hrow
    exact absurd rfl hrow

/-- **`hgate` on the constant's and the zero's gates**: a single gate a drawn unit commits holds 1 on a gate of `ones`
(`ConstCols`) and 0 on a gate of `zeros` (`ZeroCols`), so `hgate` holds there for `regGate` true on `ones` and false on
`zeros`. -/
theorem gate_pub {ones zeros : Finset (Fin C.N)} (hc : P.ConstCols plan tab place ones)
    (hZ : P.ZeroCols plan tab place zeros) {S : Finset (Fin n)} {R : Reg} {u : Fin n} (hu : u ∈ S)
    {M : Binding.Msg plan tab S R u} (hM : Binding.Sat plan tab S R u M) {g : Fin C.N} (hg : g ∈ P.io u) :
    (g ∈ ones → bit (Binding.wit plan tab S R u M (gateAt place S R u g)) = true) ∧
      (g ∈ zeros → bit (Binding.wit plan tab S R u M (gateAt place S R u g)) = false) :=
  ⟨fun h => by rw [gateAt_one place hc hu hM hg h]; rfl, fun h => by rw [gateAt_zero place hZ hu hM hg h]; rfl⟩

end FlockSoundness.Discharge.Layout
