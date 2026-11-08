module

public import Proofs.Flock.SpotFlock.Frame

/-!
# Each checked unit's gates, held by the session's rows

`Embeds` asks each gate of a checked unit to sit at one wire of the session's block circuit. The session statement's
block circuit is its tables' rows (`Readable.Sys`): each bit of a table is the AND of two XORs of the table's bits, its
row's sides, in every block. A lowering folds the program's XOR and NOT gates into its rows' sides, so a folded gate
sits at no wire, and its value is an XOR of bits of its block.

That XOR is the gate's **certificate** (`certAt`), read off the program. A folded XOR gate's is its reads' certificates
concatenated, a folded NOT gate's its read's and the one bit, a constant's no bit or the one bit, and every other gate's
its layout wire. The **one bit** is a table's pin, which every block holds at `1`. A layout names the gates of each unit
it folds (`CertLayout`); every other gate sits at its wire.

`unitOk` decides on the session's own tables that the rows hold a unit's gates:
* each AND gate reads two gates, and its wire is a row whose sides are their certificates;
* each XOR, NOT or constant gate that sits at its wire is a row whose sides are its folded form and the one bit;
* each gate whose value the unit commits sits at its wire.
Sides compare as XORs, each list with its bits cancelled in pairs (`xorCanon`).

`eval_of_unitOk`: on values of the tables' bits that hold the rows and the pins, the XORs over the certificates evaluate
each gate of a unit that passes. So `evaluates_of_certChecked`: where every unit the two stages check passes `unitOk` in a
block of the session (`CertChecked`), the certificate decoder (`decCert`) evaluates each checked unit's gates at every
witness Flock's relation accepts (`Evaluates`). At each gate a unit's correctness reads, the certificate is the layout
wire, so the decoder is the layout's there (`decCert_needs`), and `sessionOpensTwo_of_certChecked` gives
`SessionOpensTwo` from `CertChecked` and `FrameIs`.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness.Audit

/-! ## Bits as XORs -/

/-- **`x` added to an ascending list of bits, as an XOR**: where `x` is in it, both go. -/
def xorIns (x : ℕ) : List ℕ → List ℕ
  | [] => [x]
  | y :: l => if x < y then x :: y :: l else if x = y then l else y :: xorIns x l

/-- **A list of bits as an XOR, with pairs cancelled**: ascending, each bit at most once. -/
def xorCanon (l : List ℕ) : List ℕ := l.foldr xorIns []

theorem xorSum_xorIns (v : ℕ → Bool) (x : ℕ) (l : List ℕ) :
    xorSum (xorIns x l) v = Bool.xor (v x) (xorSum l v) := by
  induction l with
  | nil => rfl
  | cons y l ih =>
    simp only [xorIns]
    split_ifs with h1 h2
    · rfl
    · subst h2
      simp only [xorSum, List.map_cons, List.foldr_cons]
      rw [← Bool.xor_assoc, Bool.xor_self, Bool.false_xor]
    · simp only [xorSum, List.map_cons, List.foldr_cons] at ih ⊢
      rw [ih, Bool.xor_left_comm]

theorem xorSum_xorCanon (v : ℕ → Bool) (l : List ℕ) : xorSum (xorCanon l) v = xorSum l v := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    show xorSum (xorIns x (xorCanon l)) v = _
    rw [xorSum_xorIns, ih]
    rfl

theorem xorSum_flatMap (v : ℕ → Bool) (F : ℕ → List ℕ) (ws : List ℕ) :
    xorSum (ws.flatMap F) v = (ws.map fun i => xorSum (F i) v).foldr Bool.xor false := by
  induction ws with
  | nil => rfl
  | cons a ws ih =>
    simp only [List.flatMap_cons, FlockSoundness.Types.xorSum_append, ih, List.map_cons, List.foldr_cons]

theorem foldl_xor_false (l : List Bool) : l.foldl Bool.xor false = l.foldr Bool.xor false := by
  suffices h : ∀ c, l.foldl Bool.xor c = Bool.xor c (l.foldr Bool.xor false) by
    have := h false
    rwa [Bool.false_xor] at this
  induction l with
  | nil => simp
  | cons x l ih => intro c; rw [List.foldl_cons, ih, List.foldr_cons, Bool.xor_assoc]

/-! ## Certificates -/

/-- **A session layout with its folded gates**: a layout (`Layout`), the gates of each unit it folds (`folded u g`), and
the one bit's wire (`one`), a table's pin. -/
structure CertLayout extends Layout where
  folded : ℕ → ℕ → Bool
  one : ℕ

/-- **Whether gate `g` is folded** in the gates `b`, where `sits` names the gates of `b` that sit at their wires: an XOR or
NOT gate of `b` that doesn't, and every constant but one of `b` that does. -/
def foldedAt (C : Prog) (b : List ℕ) (sits : ℕ → Bool) (g : ℕ) : Bool :=
  match C.gates[g]? with
  | some (.const _) => !(decide (g ∈ b) && sits g)
  | some (.xor _) => decide (g ∈ b) && !sits g
  | some (.not _) => decide (g ∈ b) && !sits g
  | _ => false

/-- **Gate `g`'s certificate** in the gates `b`: the bits whose XOR is its value. A folded gate (`foldedAt`) has its
folded form: an XOR gate its reads' certificates concatenated, a NOT gate its read's and the one bit `one`, a constant
no bit or the one bit. Every other gate sits at its wire, and so does a read at or past the gate reading it. -/
def certAt (C : Prog) (b : List ℕ) (sits : ℕ → Bool) (wire : ℕ → ℕ) (one : ℕ) (g : ℕ) : List ℕ :=
  if foldedAt C b sits g then
    match C.gates[g]? with
    | some (.const c) => if c then [one] else []
    | some (.xor ws) => ws.flatMap fun i => if _h : i < g then certAt C b sits wire one i else [wire i]
    | some (.not i) => one :: (if _h : i < g then certAt C b sits wire one i else [wire i])
    | _ => [wire g]
  else [wire g]
termination_by g

/-- **Gate `g`'s folded form**: its certificate where it is folded. -/
def formAt (C : Prog) (b : List ℕ) (sits : ℕ → Bool) (wire : ℕ → ℕ) (one : ℕ) (g : ℕ) : List ℕ :=
  match C.gates[g]? with
  | some (.const c) => if c then [one] else []
  | some (.xor ws) => ws.flatMap fun i => if _h : i < g then certAt C b sits wire one i else [wire i]
  | some (.not i) => one :: (if _h : i < g then certAt C b sits wire one i else [wire i])
  | _ => [wire g]

theorem certAt_eq (C : Prog) (b : List ℕ) (sits : ℕ → Bool) (wire : ℕ → ℕ) (one : ℕ) (g : ℕ) :
    certAt C b sits wire one g = if foldedAt C b sits g then formAt C b sits wire one g else [wire g] := by
  rw [certAt]
  rfl

theorem certAt_of_not_folded {C : Prog} {b : List ℕ} {sits : ℕ → Bool} {wire : ℕ → ℕ} {one g : ℕ}
    (h : foldedAt C b sits g = false) : certAt C b sits wire one g = [wire g] := by
  rw [certAt_eq, h]
  rfl

/-- **Row `q` holds the XORs of `l` and `r`**: `q` is a bit of the tables, and its row's two sides, with their bits
cancelled in pairs, are `l`'s and `r`'s, in either order. -/
def rowIs (Sy : _root_.Flock.Readable.Sys) (q : ℕ) (l r : List ℕ) : Bool :=
  decide (q < Sy.J * Sy.N) &&
    ((xorCanon (Sy.sel Sy.A q) == xorCanon l && xorCanon (Sy.sel Sy.B q) == xorCanon r) ||
      (xorCanon (Sy.sel Sy.A q) == xorCanon r && xorCanon (Sy.sel Sy.B q) == xorCanon l))

/-- **What the rows must hold of gate `g` of `b`**: an AND gate reads two gates, and its wire is a row whose sides are
their certificates; an XOR, NOT or constant gate sitting at its wire is a row whose sides are its folded form and the one
bit, and one whose value `b` commits sits at its wire. -/
def gateOk (Sy : _root_.Flock.Readable.Sys) (C : Prog) (b : List ℕ) (sits : ℕ → Bool) (wire : ℕ → ℕ) (one : ℕ)
    (g : ℕ) : Bool :=
  match C.gates[g]? with
  | some (.and [r₁, r₂]) => rowIs Sy (wire g) (certAt C b sits wire one r₁) (certAt C b sits wire one r₂)
  | some (.and _) => false
  | some .input => true
  | some _ => (!sits g || rowIs Sy (wire g) (formAt C b sits wire one g) [one]) && (!C.Boundary b g || sits g)
  | none => true

/-- **The rows hold the gates `b`**, decided on the session's tables `Sy`: the one bit is a table's pin, and the rows hold
each gate of `b` (`gateOk`). -/
def unitOk (Sy : _root_.Flock.Readable.Sys) (C : Prog) (b : List ℕ) (sits : ℕ → Bool) (wire : ℕ → ℕ) (one : ℕ) :
    Bool :=
  (List.range Sy.J).any (fun j => one == j * Sy.N + Sy.pin j) && b.all (gateOk Sy C b sits wire one)

/-- **A gate that isn't folded sits**: an XOR, NOT or constant gate that isn't folded, and is in `b` or a constant, is in
`b` and sits at its wire. -/
theorem sits_of_foldedAt {C : Prog} {b : List ℕ} {sits : ℕ → Bool} {g : ℕ} {y : BGate}
    (hy : C.gates[g]? = some y) (hf : foldedAt C b sits g = false) (hand : ∀ ws, y ≠ .and ws) (hi : y ≠ .input)
    (hgy : g ∈ b ∨ ∃ c, y = .const c) : g ∈ b ∧ sits g = true := by
  cases y with
  | input => exact absurd rfl hi
  | and ws => exact absurd rfl (hand ws)
  | const c =>
    simp only [foldedAt, hy] at hf
    simpa using hf
  | xor ws =>
    have hg : g ∈ b := hgy.resolve_right fun ⟨_, h⟩ => by cases h
    simp only [foldedAt, hy] at hf
    exact ⟨hg, by simpa [hg] using hf⟩
  | not i =>
    have hg : g ∈ b := hgy.resolve_right fun ⟨_, h⟩ => by cases h
    simp only [foldedAt, hy] at hf
    exact ⟨hg, by simpa [hg] using hf⟩

/-- **`gateOk` at an XOR, NOT or constant gate**: where it sits, its wire is the row of its folded form and the one bit,
and where `b` commits its value, it sits. -/
theorem gateOk_sits {Sy : _root_.Flock.Readable.Sys} {C : Prog} {b : List ℕ} {sits : ℕ → Bool} {wire : ℕ → ℕ}
    {one g : ℕ} {y : BGate} (hk : gateOk Sy C b sits wire one g = true) (hy : C.gates[g]? = some y)
    (hand : ∀ ws, y ≠ .and ws) (hi : y ≠ .input) :
    (sits g = true → rowIs Sy (wire g) (formAt C b sits wire one g) [one] = true) ∧
      (C.Boundary b g = true → sits g = true) := by
  have hk' : ((!sits g || rowIs Sy (wire g) (formAt C b sits wire one g) [one]) &&
      (!C.Boundary b g || sits g)) = true := by
    unfold gateOk at hk
    rw [hy] at hk
    cases y with
    | input => exact absurd rfl hi
    | and ws => exact absurd rfl (hand ws)
    | const c => exact hk
    | xor ws => exact hk
    | not i => exact hk
  revert hk'
  cases sits g <;> cases C.Boundary b g <;> simp

/-- **At a gate a unit's correctness reads, the certificate is the wire**: where `unitOk` passes, each gate of `b` whose
value `b` commits, and each gate outside `b` but a constant that `b` reads, sits at its wire. -/
theorem certAt_needs {Sy : _root_.Flock.Readable.Sys} {C : Prog} {b : List ℕ} {sits : ℕ → Bool} {wire : ℕ → ℕ}
    {one : ℕ} (hok : unitOk Sy C b sits wire one = true) {g : ℕ} (hn : C.Needs b g) :
    certAt C b sits wire one g = [wire g] := by
  apply certAt_of_not_folded
  simp only [unitOk, Bool.and_eq_true, List.all_eq_true] at hok
  rcases hn with ⟨hg, hbd⟩ | ⟨-, hg, hnc⟩
  · cases hy : C.gates[g]? with
    | none => simp only [foldedAt, hy]
    | some y =>
      cases y with
      | input => simp only [foldedAt, hy]
      | and ws => simp only [foldedAt, hy]
      | const c =>
        have hs := (gateOk_sits (hok.2 g hg) hy (fun _ h => by cases h) (fun h => by cases h)).2 hbd
        simp [foldedAt, hy, hg, hs]
      | xor ws =>
        have hs := (gateOk_sits (hok.2 g hg) hy (fun _ h => by cases h) (fun h => by cases h)).2 hbd
        simp [foldedAt, hy, hs]
      | not i =>
        have hs := (gateOk_sits (hok.2 g hg) hy (fun _ h => by cases h) (fun h => by cases h)).2 hbd
        simp [foldedAt, hy, hs]
  · cases hy : C.gates[g]? with
    | none => simp only [foldedAt, hy]
    | some y =>
      cases y with
      | input => simp only [foldedAt, hy]
      | and ws => simp only [foldedAt, hy]
      | const c => exact absurd hy (hnc c)
      | xor ws => simp [foldedAt, hy, hg]
      | not i => simp [foldedAt, hy, hg]

/-! ## The rows evaluate the certificates -/

/-- **A row's side as an XOR of its table's bits**: where `v` holds, at each bit of row `q`'s table, the table's bit `x`,
the XOR of `v` over the bits row `q`'s side `A` selects is the side's sum (`xsum`). -/
theorem xorSum_side {Sy : _root_.Flock.Readable.Sys} (A : ℕ → ℕ → ℕ → Bool) {q : ℕ} {v x : ℕ → Bool}
    (hv : ∀ r < Sy.N, v (q / Sy.N * Sy.N + r) = x r) :
    xorSum (Sy.sel A q) v = _root_.Flock.Readable.xsum (A (q / Sy.N) (q % Sy.N)) x Sy.N := by
  have hmap : ((List.range Sy.N).filter (A (q / Sy.N) (q % Sy.N))).map (fun r => v (q / Sy.N * Sy.N + r)) =
      ((List.range Sy.N).filter (A (q / Sy.N) (q % Sy.N))).map x := by
    apply List.map_congr_left
    intro r hr
    simp only [List.mem_filter, List.mem_range] at hr
    exact hv r hr.1
  rw [_root_.Flock.Readable.xsum, _root_.Flock.Readable.Sys.foldl_filter, Bool.false_xor, ← hmap,
    _root_.Flock.Readable.Sys.sel, xorSum, List.map_map]
  rfl

/-- **Values that hold the system in a block hold its rows and pins**: where `v` is, at each bit of the tables, the
hidden value the system reads there in block `β`, and the system holds in `β`, each bit is its row's sides' AND and each
pin is `1`. -/
theorem rows_of_holds {Sy : _root_.Flock.Readable.Sys} (hwf : Sy.WF) {w : Claims.Flock.Wit} {β : ℕ}
    (hH : Sy.Holds w β) {v : ℕ → Bool} (hv : ∀ q < Sy.J * Sy.N, v q = w.hid.getD (Sy.idx β q) false) :
    (∀ q < Sy.J * Sy.N, v q = (xorSum (Sy.sel Sy.A q) v && xorSum (Sy.sel Sy.B q) v)) ∧
      ∀ j < Sy.J, v (j * Sy.N + Sy.pin j) = true := by
  have hN := hwf.N_pos
  have hbit : ∀ c < Sy.J, ∀ r < Sy.N, v (c * Sy.N + r) = Sy.bit w c β r := by
    intro c hc r hr
    have hlt : c * Sy.N + r < Sy.J * Sy.N :=
      calc c * Sy.N + r < c * Sy.N + Sy.N := Nat.add_lt_add_left hr _
        _ = (c + 1) * Sy.N := by rw [Nat.add_mul, Nat.one_mul]
        _ ≤ Sy.J * Sy.N := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hc)
    rw [hv _ hlt, idx_table Sy β hr]
    rfl
  refine ⟨fun q hq => ?_, fun j hj => by rw [hbit j hj _ (hwf.pin_lt j hj)]; exact hH.2.1 j hj⟩
  have hc : q / Sy.N < Sy.J := (Nat.div_lt_iff_lt_mul hN).2 hq
  have hi : q % Sy.N < Sy.N := Nat.mod_lt _ hN
  have hside : ∀ A : ℕ → ℕ → ℕ → Bool,
      xorSum (Sy.sel A q) v = _root_.Flock.Readable.xsum (A (q / Sy.N) (q % Sy.N)) (Sy.bit w (q / Sy.N) β) Sy.N :=
    fun A => xorSum_side A fun r hr => hbit _ hc r hr
  rw [hside, hside, hH.1 _ hc _ hi]
  have hq' : q / Sy.N * Sy.N + q % Sy.N = q := Nat.div_add_mod' q Sy.N
  conv_lhs => rw [← hq']
  exact hbit _ hc _ hi

/-- **The rows evaluate the certificates**: on a topological program, where `unitOk` passes and `v` holds the rows and the
pins, the XORs `d` of `v` over the certificates evaluate each gate of `b` but its inputs on its reads', and each
constant's is the constant. -/
theorem eval_of_unitOk {Sy : _root_.Flock.Readable.Sys} {C : Prog} {b : List ℕ} {sits : ℕ → Bool} {wire : ℕ → ℕ}
    {one : ℕ} (htop : C.Topological) (hok : unitOk Sy C b sits wire one = true) {v d : ℕ → Bool}
    (hrow : ∀ q < Sy.J * Sy.N, v q = (xorSum (Sy.sel Sy.A q) v && xorSum (Sy.sel Sy.B q) v))
    (hpin : ∀ j < Sy.J, v (j * Sy.N + Sy.pin j) = true)
    (hd : ∀ g, d g = xorSum (certAt C b sits wire one g) v) :
    (∀ g ∈ b, ∀ y : BGate, C.gates[g]? = some y → y ≠ .input →
      y.val (fun i => some (d i)) none = some (d g)) ∧
      ∀ g c, C.gates[g]? = some (.const c) → d g = c := by
  simp only [unitOk, Bool.and_eq_true, List.any_eq_true, List.mem_range, beq_iff_eq, List.all_eq_true] at hok
  obtain ⟨⟨j₁, hj₁, hone_eq⟩, hgates⟩ := hok
  have hone : v one = true := by rw [hone_eq]; exact hpin j₁ hj₁
  have hsingle : ∀ q, xorSum [q] v = v q := fun q => by simp [xorSum]
  have hrowIs : ∀ q l r, rowIs Sy q l r = true → v q = (xorSum l v && xorSum r v) := by
    intro q l r hr
    simp only [rowIs, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hr
    obtain ⟨hq, hsides⟩ := hr
    rw [hrow q hq, ← xorSum_xorCanon v (Sy.sel Sy.A q), ← xorSum_xorCanon v (Sy.sel Sy.B q),
      ← xorSum_xorCanon v l, ← xorSum_xorCanon v r]
    rcases hsides with ⟨ha, hb⟩ | ⟨ha, hb⟩
    · rw [ha, hb]
    · rw [ha, hb, Bool.and_comm]
  have hsit : ∀ g, rowIs Sy (wire g) (formAt C b sits wire one g) [one] = true →
      v (wire g) = xorSum (formAt C b sits wire one g) v := by
    intro g hr
    rw [hrowIs _ _ _ hr, hsingle, hone, Bool.and_true]
  have hform_xor : ∀ g ws, C.gates[g]? = some (.xor ws) →
      xorSum (formAt C b sits wire one g) v = (ws.map d).foldr Bool.xor false := by
    intro g ws hg
    have hlt : ∀ i ∈ ws, i < g := htop.1 g _ hg
    simp only [formAt, hg]
    rw [xorSum_flatMap]
    congr 1
    apply List.map_congr_left
    intro i hi
    simp only [dif_pos (hlt i hi), hd]
  have hform_not : ∀ g i, C.gates[g]? = some (.not i) → xorSum (formAt C b sits wire one g) v = !d i := by
    intro g i hg
    have hlt : i < g := htop.1 g _ hg i (List.mem_singleton_self i)
    simp only [formAt, hg, dif_pos hlt, hd]
    show Bool.xor (v one) (xorSum (certAt C b sits wire one i) v) = _
    rw [hone, Bool.true_xor]
  have hform_const : ∀ g c, C.gates[g]? = some (.const c) → xorSum (formAt C b sits wire one g) v = c := by
    intro g c hg
    simp only [formAt, hg]
    cases c <;> simp [xorSum, hone]
  have hform : ∀ g y, C.gates[g]? = some y → (∀ ws, y ≠ .and ws) → y ≠ .input → (g ∈ b ∨ ∃ c, y = .const c) →
      d g = xorSum (formAt C b sits wire one g) v := by
    intro g y hy hand hi hgy
    rw [hd]
    cases hf : foldedAt C b sits g
    · obtain ⟨hgb, hs⟩ := sits_of_foldedAt hy hf hand hi hgy
      rw [certAt_of_not_folded hf, hsingle]
      exact hsit g ((gateOk_sits (hgates g hgb) hy hand hi).1 hs)
    · rw [certAt_eq, if_pos hf]
  have hconst : ∀ g c, C.gates[g]? = some (.const c) → d g = c := fun g c hg =>
    (hform g _ hg (fun _ h => by cases h) (fun h => by cases h) (Or.inr ⟨c, rfl⟩)).trans (hform_const g c hg)
  refine ⟨fun g hg y hy hni => ?_, hconst⟩
  cases y with
  | input => exact absurd rfl hni
  | const c =>
    show some c = some (d g)
    rw [hconst g c hy]
  | not i =>
    show (some (d i)).map (!·) = some (d g)
    rw [hform g _ hy (fun _ h => by cases h) hni (Or.inl hg), hform_not g i hy]
    rfl
  | xor ws =>
    show (ws.mapM fun i => some (d i)).map (fun bs => bs.foldl Bool.xor false) = some (d g)
    rw [Univ.mapM_some, hform g _ hy (fun _ h => by cases h) hni (Or.inl hg), hform_xor g ws hy]
    show some ((ws.map d).foldl Bool.xor false) = _
    rw [foldl_xor_false]
  | and ws =>
    have hk := hgates g hg
    have hf : foldedAt C b sits g = false := by simp only [foldedAt, hy]
    rcases ws with _ | ⟨r₁, _ | ⟨r₂, _ | ⟨r₃, ws⟩⟩⟩
    · simp [gateOk, hy] at hk
    · simp [gateOk, hy] at hk
    · have hr : rowIs Sy (wire g) (certAt C b sits wire one r₁) (certAt C b sits wire one r₂) = true := by
        unfold gateOk at hk
        rw [hy] at hk
        exact hk
      show ([r₁, r₂].mapM fun i => some (d i)).map (fun bs => bs.all id) = some (d g)
      rw [Univ.mapM_some, hd g, certAt_of_not_folded hf, hsingle, hrowIs _ _ _ hr, ← hd, ← hd]
      show some (d r₁ && (d r₂ && true)) = _
      rw [Bool.and_true]
    · simp [gateOk, hy] at hk

/-! ## At SPOT[Flock]'s session -/

/-- **The session's values hold its rows and pins**: where Flock's relation holds at the readable form of a statement's
tables, in each of its blocks, the block circuit's values at the tables' bits hold each row and each pin. -/
theorem rows_of_relS {p : _root_.Flock.Readable.Parsed} {w : Claims.Flock.Wit} {β : ℕ}
    (hR : Claims.Flock.RelS (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) w)
    (hβ : β < (relationSys p).nB) :
    (∀ q < (relationSys p).J * (relationSys p).N,
      valueAt (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) β w q =
        (xorSum ((relationSys p).sel (relationSys p).A q)
            (valueAt (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) β w) &&
          xorSum ((relationSys p).sel (relationSys p).B q)
            (valueAt (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) β w))) ∧
      ∀ j < (relationSys p).J,
        valueAt (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) β w
          (j * (relationSys p).N + (relationSys p).pin j) = true := by
  have hwf : (relationSys p).WF := _root_.Flock.Readable.sysOf_wf _ _ _
  have hC : (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).C = (relationSys p).circ := rfl
  have hinp :
      (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).inp = (relationSys p).rel.inp := rfl
  have hH : (relationSys p).Holds w β :=
    (_root_.Flock.Readable.Sys.relS_iff hwf _ w hC hinp rfl rfl rfl).1 hR β hβ
  exact rows_of_holds hwf hH fun q hq => valueAt_sys hC hinp β w hq

variable (h : Held)

/-- **The certificate decoder**: gate `g`'s value in checked unit `u` as the XOR, over its certificate in `u`'s proof
block, of the session's block circuit's values in `u`'s block. -/
def decCert (L : CertLayout) : Dec := fun x m₁ xS w u g =>
  match FlockDeploy.parse true _root_.Flock.Readable.relation xS, L.blockOf x m₁ xS u,
      ((parse h x).bind (partition h)).bind (·.blocks[u]?) with
  | some S, some β, some b => xorSum (certAt h.prog b (fun k => !L.folded u k) (L.wire u) L.one g) (valueAt S β w)
  | _, _, _ => false

theorem decCert_eq {L : CertLayout} {x m₁ xS : Msg} {S : Claims.Flock.Stmt} {u β : ℕ} {s : AuditStatement}
    {Pt : Partition} {b : List ℕ} (hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS = some S)
    (hβ : L.blockOf x m₁ xS u = some β) (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hb : Pt.blocks[u]? = some b) (w : Claims.Flock.Wit) (g : ℕ) :
    decCert h L x m₁ xS w u g =
      xorSum (certAt h.prog b (fun k => !L.folded u k) (L.wire u) L.one g) (valueAt S β w) := by
  simp only [decCert, hS, hβ, hx, Option.bind_some, hP, hb]

/-- **The round-2 check holds each checked unit to the session's rows**: at a run's statement in the two-stage scope, a
session statement the verifier's check passes, and its setup, each unit the two stages check has a block of the session
below its number of blocks, and `unitOk` passes for its proof block on the setup's tables, at the layout's wires, folded
gates and one bit. -/
def CertChecked (L : CertLayout) : Prop :=
  ∀ x s Pt d (m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) u b ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Pt.blocks[u]? = some b → FlockDeploy.setupOf true xS = some (ins, spec, sts) →
    ∃ β, L.blockOf x m₁ xS u = some β ∧ β < (relationSys { ins.read with sts }).nB ∧
      unitOk (relationSys { ins.read with sts }) s.prog b (fun k => !L.folded u k) (L.wire u) L.one = true

/-- **The certificate decoder evaluates each checked unit** where the round-2 check holds each checked unit to the
session's rows (`CertChecked`). -/
theorem evaluates_of_certChecked (L : CertLayout) (hc : CertChecked h L) : Evaluates h (decCert h L) := by
  intro x s Pt d m₁ xS c₀ c₁ wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb
  obtain ⟨_it, ins, spec, sts, _own, -, hset, -⟩ := rows_open h hx hok (sessionScoped h x m₁ xS c₀ c₁ hok) hck
  obtain ⟨β, hβ, hβn, hunit⟩ := hc x s Pt d m₁ xS c₀ c₁ u b ins spec sts hx hP h2 hin hok hu hb hset
  obtain ⟨-, hR⟩ := relS_setupOf hset hck
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    obtain rfl := parse_setupOf hset hS
    obtain ⟨hrow, hpin⟩ := rows_of_relS hR hβn
    exact eval_of_unitOk hin.1 hunit hrow hpin fun g => by
      rw [decCert_eq h hS hβ hx hP hb, parse_prog h hx]

/-- **At each gate a checked unit's correctness reads, the certificate decoder is the layout's**, where the round-2 check
holds each checked unit to the session's rows. -/
theorem decCert_needs (L : CertLayout) (hc : CertChecked h L) {x s Pt d m₁ xS c₀ c₁ wF u b}
    (hx : parse h x = some s) (hP : partition h s = some Pt) (h2 : two h s = some d) (hin : InScopeTwo Pt d s)
    (hok : stmtOk h x m₁ xS c₀ c₁ = true) (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁))
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true)
    (hb : Pt.blocks[u]? = some b) {g : ℕ} (hn : s.prog.Needs b g) :
    decCert h L x m₁ xS wF u g = decOf L.toLayout x m₁ xS wF u g := by
  obtain ⟨_it, ins, spec, sts, _own, -, hset, -⟩ := rows_open h hx hok (sessionScoped h x m₁ xS c₀ c₁ hok) hck
  obtain ⟨β, hβ, -, hunit⟩ := hc x s Pt d m₁ xS c₀ c₁ u b ins spec sts hx hP h2 hin hok hu hb hset
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation xS with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S =>
    rw [decCert_eq h hS hβ hx hP hb, decOf_eq hS hβ, ← parse_prog h hx, certAt_needs hunit hn]
    simp [xorSum]

/-- **`Reveals` at the certificate decoder from `Reveals` at the layout's**: the two agree at every gate a checked unit's
correctness reads. -/
theorem reveals_cert (L : CertLayout) (hc : CertChecked h L) (hr : Reveals h (decOf L.toLayout)) :
    Reveals h (decCert h L) := by
  intro x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
  obtain ⟨q, hq, hop, hpos, hval, hroot⟩ :=
    hr x s Pt d m₀ m₁ xS c₀ c₁ rest wF u b ρ rb hx hP h2 hin hok hu hck hb hρ hrb g hg
  exact ⟨q, hq, hop, hpos, by rw [hval, decCert_needs h L hc hx hP h2 hin hok hu hck hb hg], hroot⟩

/-- **`SessionOpensTwo` at SPOT[Flock]'s two-stage code from the round-2 check and the frame**: where the round-2 check
holds each checked unit to the session's rows (`CertChecked`) and the frame opens the session's rows (`FrameIs`), with no
premise that the session statement holds each gate at a wire. -/
theorem sessionOpensTwo_of_certChecked (L : CertLayout) (hc : CertChecked h L) (hF : FrameIs h L.toLayout) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_eval h (decCert h L) (evaluates_of_certChecked h L hc)
    (reveals_cert h L hc (reveals_of_frameRows h L.toLayout (frameRows_of_frameIs h L.toLayout hF)))

end SpotFlock
