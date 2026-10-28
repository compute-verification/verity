import Proofs.Flock.Claims.Sampling.Universal
import Definitions.Pous.Model.Continuous

/-!
# The PoUS verdict unit decides the continuous audit's verdict

`PousVerdict_v1{K, M, LIMIT}` (`verity/ml/protocols/pous/verdict_circuit.py`) is one proof unit in the Boolean
profile. Its inputs are committed values: the `K` rounds' records as the memory challenger writes them
(`u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)`), then the expected digests of the blocks they
challenged. Its one public output is the verdict. `verdictProg` restates its gates (`verdict_circuit.gates`) builder for
builder.

* `round_spec`: round `j`'s hit wire is `1` iff `roundVerdict` is a hit: the record names round `j`, its receipt is
  not before its send and at most `LIMIT` after it, and its `H(block)` is the expected digest.
* `verdictProg_eval`: on the openings of `K ≤ 2³²` rounds of the unit's widths, the output is `[#misses ≤ M]`.
* `verdictProg_accepts`, `verdict_accepts`: the output is `1` iff `contAccepts M rs` for any records `rs` whose
  verdicts the openings determine (`rs.map (·.verdict) = verdicts LIMIT os`), at `verdictProg` and at any program that
  is the Definition's (`VerdictIs`).

Composing this with sampling's knowledge soundness into the proof layer's soundness for the continuous audit
(`Pous.ProofSound`, the hypothesis of `Pous.Guarantees.ContinuousAuditProved`) needs more, none of it about the unit:

1. `VerdictIs` at the deployed unit. The unit's code, `PousCheck.Verdict.prog`, is `verdictProg` gate for gate
   (`VerdictCode.lean`, `PousVerdict.Code.verdictIs_prog`), and the difftest `test_verdict_difftest.py` holds the
   Definition's compiled circuit to that code at the statics it runs; no Lean reads the deployed descriptor into gates.
2. The openings are what the audit committed. The records must sit under a hiding commitment that the memory
   challenger's signature binds (its records tree's leaves are not hiding hm96 rows yet). Each expected digest must be
   read from the setup-committed digest table at the round's drawn index; a registered port reads one position per
   unit, so the unit's `K` reads need a port that reads several.
3. The deployed records are the model's: `verdicts LIMIT os` are `contRecords`' verdicts. That takes the device
   writing `j` and its clocks as specified, `LIMIT` standing for the model's `(D, Q)` deadline, and SHA-512's collision
   resistance for a matching `H(block)` to be the right block.
-/

namespace PousVerdict

open Pous Claims.Sampling Claims.Sampling.Univ

/-! ## The openings and their verdicts -/

/-- A round record's bits: `u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)`, 92 bytes. -/
def recordBits : ℕ := 736

def digestBits : ℕ := 512

/-- Round `j`'s opening: its record as the memory challenger writes it (`memory_challenger.Round.value`), and the expected
digest of the block it challenged. A byte string is bits in byte order, bit `t` being bit `t mod 8` of byte `t / 8`. -/
structure RoundOpening where
  row : List Bool
  expected : List Bool

/-- An opening of the unit's widths. -/
def RoundOpening.Fits (o : RoundOpening) : Prop := o.row.length = recordBits ∧ o.expected.length = digestBits

/-- The positions of the big-endian integer at bytes `pos … pos + nbytes − 1` of a byte string from position `base`, least
significant bit first (`verdict_circuit.be_wires`). -/
def beWires (base pos nbytes : ℕ) : List ℕ :=
  (List.range (8 * nbytes)).map fun k => base + 8 * (pos + nbytes - 1 - k / 8) + k % 8

/-- The big-endian integer at bytes `pos … pos + nbytes − 1` of `row`. -/
def beField (row : List Bool) (pos nbytes : ℕ) : ℕ := bitsVal ((beWires 0 pos nbytes).map fun i => row.getD i false)

def RoundOpening.j (o : RoundOpening) : ℕ := beField o.row 0 4
def RoundOpening.send (o : RoundOpening) : ℕ := beField o.row 12 8
def RoundOpening.receipt (o : RoundOpening) : ℕ := beField o.row 20 8

/-- The record's `H(block)`. -/
def RoundOpening.digest (o : RoundOpening) : List Bool := (List.range digestBits).map fun t => o.row.getD (224 + t) false

/-- **Round `j`'s verdict** (`verdict_circuit.round_verdict`): rejected when its record names another round or its receipt
is before its send, late past `LIMIT`, wrong when its `H(block)` is not the expected digest, else a hit. -/
def roundVerdict (LIMIT j : ℕ) (o : RoundOpening) : ContVerdict :=
  if o.j = j ∧ o.send ≤ o.receipt then
    if o.receipt - o.send ≤ LIMIT then (if o.digest = o.expected then .hit else .wrong) else .late
  else .reject

/-- The rounds' verdicts, round `j` the `j`-th opening's. -/
def verdicts (LIMIT : ℕ) (os : List RoundOpening) : List ContVerdict := os.mapIdx fun j o => roundVerdict LIMIT j o

/-- The unit's inputs in leaf order: every record, then every expected digest. -/
def encode (os : List RoundOpening) : List Bool := os.flatMap (·.row) ++ os.flatMap (·.expected)

/-! ## The gates (`verdict_circuit.gates`) -/

/-- The unit's input bits. -/
def inputs (K : ℕ) : ℕ := (recordBits + digestBits) * K

/-- `width` constant wires of `value`, least significant first. -/
def constWires (w0 w1 width value : ℕ) : List ℕ := (List.range width).map fun i => if value.testBit i then w1 else w0

/-- The carries of `a + ¬s + c` after `gs`, given the bits `(a_i, p_i)`: per bit `q = a ⊕ c`, `t = p ∧ q`, `c' = a ⊕ t`.
Returns the gates and the carries `c_0 = c, …, c_n`. -/
def carries : List BGate → List (ℕ × ℕ) → ℕ → List BGate × List ℕ
  | gs, [], c => (gs, [c])
  | gs, (x, p) :: r, c =>
    ((carries (gs ++ [.xor [x, c], .and [p, gs.length], .xor [x, gs.length + 1]]) r (gs.length + 2)).1,
      c :: (carries (gs ++ [.xor [x, c], .and [p, gs.length], .xor [x, gs.length + 1]]) r (gs.length + 2)).2)

/-- **`a + ¬s + c0`** after `gs` (`verdict_circuit.chain`): the gates `p_i = a_i ⊕ s_i ⊕ 1`, then the carries. Returns the
gates, the `p` wires and the carries. -/
def chain (w1 : ℕ) (gs : List BGate) (a s : List ℕ) (c0 : ℕ) : List BGate × List ℕ × List ℕ :=
  ((carries (gs ++ (a.zip s).map fun xy => BGate.xor [xy.1, xy.2, w1])
      (a.zip ((List.range (a.zip s).length).map (gs.length + ·))) c0).1,
    (List.range (a.zip s).length).map (gs.length + ·),
    (carries (gs ++ (a.zip s).map fun xy => BGate.xor [xy.1, xy.2, w1])
      (a.zip ((List.range (a.zip s).length).map (gs.length + ·))) c0).2)

/-- The sum bits `d_i = p_i ⊕ c_i` after `gs`. -/
def sums (gs : List BGate) (ps cs : List ℕ) : List BGate × List ℕ :=
  (gs ++ (ps.zip cs).map (fun pc => BGate.xor [pc.1, pc.2]), (List.range (ps.zip cs).length).map (gs.length + ·))

/-- The equalities of round `j`: each bit of its record's `j` against `j`'s, and each bit of its `H(block)` against the
expected digest's, as `Xor3(a, b, 1)`. -/
def litGates (K w0 w1 j : ℕ) : List BGate :=
  ((beWires (recordBits * j) 0 4).zip (constWires w0 w1 32 j)).map (fun xc => BGate.xor [xc.1, xc.2, w1]) ++
    (List.range digestBits).map fun t => BGate.xor [recordBits * j + 224 + t, recordBits * K + digestBits * j + t, w1]

/-- **Round `j`'s gates** after `gs`, and its hit wire: the equalities, `receipt + ¬send + 1` and its sum `receipt − send`,
`LIMIT + ¬(receipt − send) + 1`, and one AND over the equalities and the two last carries. -/
def roundGates (K LIMIT w0 w1 j : ℕ) (gs : List BGate) : List BGate × ℕ :=
  let c1 := chain w1 (gs ++ litGates K w0 w1 j) (beWires (recordBits * j) 20 8) (beWires (recordBits * j) 12 8) w1
  let d := sums c1.1 c1.2.1 c1.2.2
  let c2 := chain w1 d.1 (constWires w0 w1 64 (min LIMIT (2 ^ 64 - 1))) d.2 w1
  (c2.1 ++ [.and ((List.range (litGates K w0 w1 j).length).map (gs.length + ·) ++
      [c1.2.2.getLastD 0, c2.2.2.getLastD 0])], c2.1.length)

/-- The inputs, the constants `0` and `1`. -/
def initGates (K : ℕ) : List BGate := List.replicate (inputs K) .input ++ [.const false, .const true]

/-- Round `j` after the gates and hit wires so far. -/
def hitStep (K LIMIT : ℕ) (st : List BGate × List ℕ) (j : ℕ) : List BGate × List ℕ :=
  ((roundGates K LIMIT (inputs K) (inputs K + 1) j st.1).1,
    st.2 ++ [(roundGates K LIMIT (inputs K) (inputs K + 1) j st.1).2])

/-- Every round's gates, and their hit wires. -/
def hitGates (K LIMIT : ℕ) : List BGate × List ℕ := (List.range K).foldl (hitStep K LIMIT) (initGates K, [])

/-- The miss counter's width (`verdict_circuit.counter_width`). -/
def counterWidth (K M : ℕ) : ℕ := Nat.size (max K M)

/-- One round of the miss counter: `cnt + 0 + ¬hit`. -/
def countStep (w1 w : ℕ) (st : List BGate × List ℕ) (h : ℕ) : List BGate × List ℕ :=
  sums (chain w1 (st.1 ++ [.not h]) st.2 (List.replicate w w1) st.1.length).1
    (chain w1 (st.1 ++ [.not h]) st.2 (List.replicate w w1) st.1.length).2.1
    (chain w1 (st.1 ++ [.not h]) st.2 (List.replicate w w1) st.1.length).2.2

/-- The counter over every round. -/
def countGates (K M LIMIT : ℕ) : List BGate × List ℕ :=
  (hitGates K LIMIT).2.foldl (countStep (inputs K + 1) (counterWidth K M))
    ((hitGates K LIMIT).1, List.replicate (counterWidth K M) (inputs K))

/-- **The verdict unit `PousVerdict_v1{K, M, LIMIT}`** as a program (`verdict_circuit.gates`): its one output, the last
carry of `M + ¬cnt + 1`, is `cnt ≤ M`. -/
def verdictProg (K M LIMIT : ℕ) : Claims.Sampling.Prog :=
  ⟨(chain (inputs K + 1) (countGates K M LIMIT).1 (constWires (inputs K) (inputs K + 1) (counterWidth K M) M)
      (countGates K M LIMIT).2 (inputs K + 1)).1,
    [(chain (inputs K + 1) (countGates K M LIMIT).1 (constWires (inputs K) (inputs K + 1) (counterWidth K M) M)
      (countGates K M LIMIT).2 (inputs K + 1)).2.2.getLastD 0]⟩

/-- **The deployed verdict unit is the Definition's**: on every input of its width it computes what `verdictProg` does.
It holds of the unit's code, `PousCheck.Verdict.prog` (`PousVerdict.Code.verdictIs_prog`), which the difftest
`test_verdict_difftest.py` holds to the Definition's compiled circuit. -/
def VerdictIs (P : Claims.Sampling.Prog) (K M LIMIT : ℕ) : Prop :=
  ∀ y : List Bool, y.length = inputs K → P.eval y = (verdictProg K M LIMIT).eval y

/-! ## Gates on wires -/

/-- A gate's value from its operands' values `f`. -/
def gval (f : ℕ → Bool) : BGate → Bool
  | .input => false
  | .const b => b
  | .not i => !f i
  | .and ws => (ws.map f).all id
  | .xor ws => (ws.map f).foldl Bool.xor false

theorem val_some (f : ℕ → Bool) : ∀ g : BGate, g ≠ .input → (g.val (fun i => some (f i)) none).getD false = gval f g
  | .input, h => absurd rfl h
  | .const _, _ => rfl
  | .not _, _ => rfl
  | .and ws, _ => by simp only [BGate.val, mapM_some, Option.map_some, Option.getD_some, gval]
  | .xor ws, _ => by simp only [BGate.val, mapM_some, Option.map_some, Option.getD_some, gval]

theorem gval_congr {f f' : ℕ → Bool} : ∀ g : BGate, (∀ i ∈ g.reads, f i = f' i) → gval f g = gval f' g
  | .input, _ => rfl
  | .const _, _ => rfl
  | .not i, h => by simp only [gval, h i (by simp [BGate.reads])]
  | .and ws, h => by simp only [gval, List.map_congr_left fun i hi => h i (by simpa [BGate.reads] using hi)]
  | .xor ws, h => by simp only [gval, List.map_congr_left fun i hi => h i (by simpa [BGate.reads] using hi)]

theorem V_last (gs : List BGate) (g : BGate) (hg : g ≠ .input) (y : List Bool) :
    V (gs ++ [g]) y gs.length = gval (V gs y) g := by
  rw [V_snoc gs g hg y, val_some _ g hg]

/-- Gate `k` of a batch after `gs` whose gates read only `gs`. -/
theorem V_batch (gs more : List BGate) (y : List Bool) {k : ℕ} (hk : k < more.length) (hin : more[k] ≠ .input)
    (hr : ∀ i ∈ more[k].reads, i < gs.length) : V (gs ++ more) y (gs.length + k) = gval (V gs y) more[k] := by
  have hsplit : gs ++ more = (gs ++ more.take k ++ [more[k]]) ++ more.drop (k + 1) := by
    rw [List.append_assoc, List.append_assoc, List.singleton_append, ← List.drop_eq_getElem_cons hk,
      List.take_append_drop]
  have htl : (more.take k).length = k := List.length_take_of_le hk.le
  rw [hsplit, V_append _ y (by simp; omega),
    show gs.length + k = (gs ++ more.take k).length by simp [htl], V_last _ _ hin y]
  exact gval_congr _ fun i hi => V_append _ y (hr i hi)

theorem V_batch_map (gs more : List BGate) (y : List Bool) (hin : ∀ g ∈ more, g ≠ .input)
    (hr : ∀ g ∈ more, ∀ i ∈ g.reads, i < gs.length) :
    ((List.range more.length).map (gs.length + ·)).map (V (gs ++ more) y) = more.map (gval (V gs y)) := by
  apply List.ext_getElem (by simp)
  intro k h1 h2
  simp only [List.getElem_map, List.getElem_range]
  exact V_batch gs more y (by simpa using h2) (hin _ (List.getElem_mem _)) (hr _ (List.getElem_mem _))

/-! ## Addition on bits -/

/-- `x + y + c` bit by bit, least significant first: the sum bits and the last carry, each carry `x ⊕ (x ⊕ y)(x ⊕ c)`. -/
def ripple : List Bool → List Bool → Bool → List Bool × Bool
  | x :: xs, y :: ys, c => (((x ^^ y) ^^ c) :: (ripple xs ys (x ^^ ((x ^^ y) && (x ^^ c)))).1,
      (ripple xs ys (x ^^ ((x ^^ y) && (x ^^ c)))).2)
  | _, _, c => ([], c)

theorem ripple_val : ∀ (xs ys : List Bool) (c : Bool), xs.length = ys.length →
    bitsVal (ripple xs ys c).1 + 2 ^ xs.length * (ripple xs ys c).2.toNat = bitsVal xs + bitsVal ys + c.toNat
  | [], [], c, _ => by simp [ripple, bitsVal]
  | x :: xs, y :: ys, c, h => by
    have ih := ripple_val xs ys (x ^^ ((x ^^ y) && (x ^^ c))) (by simpa using h)
    have hb : ((x ^^ y) ^^ c).toNat + 2 * (x ^^ ((x ^^ y) && (x ^^ c))).toNat = x.toNat + y.toNat + c.toNat := by
      cases x <;> cases y <;> cases c <;> rfl
    simp only [ripple, bitsVal, List.length_cons, pow_succ]
    nlinarith [ih, hb]
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h

theorem ripple_length : ∀ (xs ys : List Bool) (c : Bool), xs.length = ys.length → (ripple xs ys c).1.length = xs.length
  | [], [], _, _ => rfl
  | _ :: xs, _ :: ys, _, h => by simp [ripple, ripple_length xs ys _ (by simpa using h)]
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h

theorem bitsVal_lt : ∀ l : List Bool, bitsVal l < 2 ^ l.length
  | [] => by simp [bitsVal]
  | b :: l => by
    have := bitsVal_lt l
    have hb : b.toNat ≤ 1 := by cases b <;> simp
    simp only [bitsVal, List.length_cons, pow_succ]
    omega

theorem bitsVal_not : ∀ l : List Bool, bitsVal (l.map (!·)) + bitsVal l + 1 = 2 ^ l.length
  | [] => by simp [bitsVal]
  | b :: l => by
    have := bitsVal_not l
    have hb : (!b).toNat + b.toNat = 1 := by cases b <;> rfl
    simp only [List.map_cons, bitsVal, List.length_cons, pow_succ]
    omega

/-- `x + ¬s + 1`'s last carry is `s ≤ x`. -/
theorem ripple_le (xs ss : List Bool) (h : xs.length = ss.length) :
    (ripple xs (ss.map (!·)) true).2 = decide (bitsVal ss ≤ bitsVal xs) := by
  have hv := ripple_val xs (ss.map (!·)) true (by simpa using h)
  have hn := bitsVal_not ss
  have hlt := bitsVal_lt (ripple xs (ss.map (!·)) true).1
  rw [ripple_length _ _ _ (by simpa using h)] at hlt
  rw [h] at hv hlt
  have hx := bitsVal_lt xs
  rw [h] at hx
  cases hc : (ripple xs (ss.map (!·)) true).2
  · rw [hc] at hv
    simp only [Bool.toNat_false, Bool.toNat_true, mul_zero, add_zero] at hv
    symm
    simp only [decide_eq_false_iff_not, not_le]
    omega
  · rw [hc] at hv
    simp only [Bool.toNat_true, mul_one] at hv
    symm
    simp only [decide_eq_true_eq]
    omega

/-- With `s ≤ x`, `x + ¬s + 1`'s sum bits are `x − s`. -/
theorem ripple_sub (xs ss : List Bool) (h : xs.length = ss.length) (hle : bitsVal ss ≤ bitsVal xs) :
    bitsVal (ripple xs (ss.map (!·)) true).1 = bitsVal xs - bitsVal ss := by
  have hv := ripple_val xs (ss.map (!·)) true (by simpa using h)
  have hn := bitsVal_not ss
  have hc := ripple_le xs ss h
  rw [decide_eq_true hle] at hc
  rw [hc, h] at hv
  simp only [Bool.toNat_true] at hv
  omega

/-- `x + 0 + c` below `2^n` is the sum, with no carry out. -/
theorem ripple_inc (xs : List Bool) (c : Bool) (hlt : bitsVal xs + c.toNat < 2 ^ xs.length) :
    bitsVal (ripple xs (List.replicate xs.length false) c).1 = bitsVal xs + c.toNat := by
  have hv := ripple_val xs (List.replicate xs.length false) c (by simp)
  have h0 : bitsVal (List.replicate xs.length false) = 0 := by
    induction xs.length with
    | zero => rfl
    | succ n ih => simp [List.replicate_succ, bitsVal, ih]
  rw [h0] at hv
  cases hcar : (ripple xs (List.replicate xs.length false) c).2
  · simpa [hcar] using hv
  · simp only [hcar, Bool.toNat_true] at hv; omega

/-! ## The builders on wires -/

/-- The carries `carries` computes, on bits: `c_0 = c`, `c_{i+1} = x_i ⊕ p_i(x_i ⊕ c_i)`. -/
def carryBits : List Bool → List Bool → Bool → List Bool
  | x :: xs, p :: ps, c => c :: carryBits xs ps (x ^^ (p && (x ^^ c)))
  | _, _, c => [c]

theorem map_getLastD {α β : Type} (f : α → β) : ∀ (l : List α) (d : α), f (l.getLastD d) = (l.map f).getLastD (f d)
  | [], _ => rfl
  | a :: l, d => by rw [List.getLastD_cons, List.map_cons, List.getLastD_cons, map_getLastD f l a]

theorem getLastD_mem_cons {α : Type} : ∀ (a : α) (l : List α) (d : α), (a :: l).getLastD d ∈ a :: l
  | a, [], _ => by simp
  | a, b :: l, _ => by
    rw [List.getLastD_cons]
    exact List.mem_cons_of_mem a (getLastD_mem_cons b l a)

theorem zip_map_fst {α β γ : Type} (f : α → γ) : ∀ (l₁ : List α) (l₂ : List β), l₁.length ≤ l₂.length →
    (l₁.zip l₂).map (fun p => f p.1) = l₁.map f
  | [], _, _ => rfl
  | _ :: _, [], h => by simp at h
  | a :: l₁, b :: l₂, h => by
    simp only [List.zip_cons_cons, List.map_cons, zip_map_fst f l₁ l₂ (by simpa using h)]

theorem zip_map_snd {α β γ : Type} (f : β → γ) : ∀ (l₁ : List α) (l₂ : List β), l₂.length ≤ l₁.length →
    (l₁.zip l₂).map (fun p => f p.2) = l₂.map f
  | _, [], _ => by simp
  | [], _ :: _, h => by simp at h
  | a :: l₁, b :: l₂, h => by
    simp only [List.zip_cons_cons, List.map_cons, zip_map_snd f l₁ l₂ (by simpa using h)]

theorem carries_ne_nil (gs : List BGate) (l : List (ℕ × ℕ)) (c : ℕ) : (carries gs l c).2 ≠ [] := by
  cases l with
  | nil => simp [carries]
  | cons xp r => obtain ⟨x, p⟩ := xp; simp [carries]

theorem carries_spec (y : List Bool) : ∀ (l : List (ℕ × ℕ)) (gs : List BGate) (c : ℕ),
    (∀ xp ∈ l, xp.1 < gs.length ∧ xp.2 < gs.length) → c < gs.length →
    Ext gs (carries gs l c).1 ∧ (∀ w ∈ (carries gs l c).2, w < (carries gs l c).1.length) ∧
      (carries gs l c).2.map (V (carries gs l c).1 y) =
        carryBits (l.map fun xp => V gs y xp.1) (l.map fun xp => V gs y xp.2) (V gs y c)
  | [], gs, c, _, hc => by
    refine ⟨Ext.refl gs, by simpa [carries] using hc, ?_⟩
    simp [carries, carryBits]
  | (x, p) :: r, gs, c, hl, hc => by
    have hx := (hl (x, p) (by simp)).1
    have hp := (hl (x, p) (by simp)).2
    have hsplit : gs ++ [BGate.xor [x, c], BGate.and [p, gs.length], BGate.xor [x, gs.length + 1]] =
        ((gs ++ [BGate.xor [x, c]]) ++ [BGate.and [p, gs.length]]) ++ [BGate.xor [x, gs.length + 1]] := by simp
    have vq : V (gs ++ [BGate.xor [x, c]]) y gs.length = (V gs y x ^^ V gs y c) := by
      rw [V_xor]
      simp
    have vt : V ((gs ++ [BGate.xor [x, c]]) ++ [BGate.and [p, gs.length]]) y (gs.length + 1) =
        (V gs y p && (V gs y x ^^ V gs y c)) := by
      rw [V_and' _ y (by simp)]
      simp [V_append _ y hp, vq]
    have vc : V (gs ++ [BGate.xor [x, c], BGate.and [p, gs.length], BGate.xor [x, gs.length + 1]]) y
        (gs.length + 2) = (V gs y x ^^ (V gs y p && (V gs y x ^^ V gs y c))) := by
      rw [hsplit, V_xor' _ y (by simp)]
      have hx2 : V ((gs ++ [BGate.xor [x, c]]) ++ [BGate.and [p, gs.length]]) y x = V gs y x := by
        rw [V_append _ y (by simp; omega), V_append _ y hx]
      simp only [List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, Bool.false_xor]
      rw [hx2, vt]
    have he1 : Ext gs (gs ++ [BGate.xor [x, c], BGate.and [p, gs.length], BGate.xor [x, gs.length + 1]]) := ⟨_, rfl⟩
    have hl1 : (gs ++ [BGate.xor [x, c], BGate.and [p, gs.length], BGate.xor [x, gs.length + 1]]).length =
        gs.length + 3 := by simp
    obtain ⟨he, hw, hv⟩ := carries_spec y r _ (gs.length + 2)
      (fun xp hxp => by
        have := hl xp (List.mem_cons_of_mem _ hxp)
        rw [hl1]; omega)
      (by rw [hl1]; omega)
    refine ⟨he1.trans he, ?_, ?_⟩
    · intro w hw'
      simp only [carries, List.mem_cons] at hw'
      rcases hw' with rfl | hw'
      · exact lt_of_lt_of_le hc ((he1.trans he).le)
      · exact hw w hw'
    · simp only [carries, List.map_cons, carryBits]
      rw [(he1.trans he).V y hc, hv, vc]
      congr 2
      · exact List.map_congr_left fun xp hxp => he1.V y (hl xp (List.mem_cons_of_mem _ hxp)).1
      · exact List.map_congr_left fun xp hxp => he1.V y (hl xp (List.mem_cons_of_mem _ hxp)).2

theorem xor3_vals (f : ℕ → Bool) (w1 : ℕ) (hf : f w1 = true) : ∀ (a s : List ℕ),
    (((a.zip s).map fun xy => BGate.xor [xy.1, xy.2, w1]).map (gval f)) =
      List.zipWith (· ^^ ·) (a.map f) ((s.map f).map (!·))
  | [], _ => rfl
  | _ :: _, [] => rfl
  | x :: a, t :: s => by
    simp only [List.zip_cons_cons, List.map_cons, List.zipWith_cons_cons, xor3_vals f w1 hf a s, gval,
      List.map_nil, List.foldl_cons, List.foldl_nil, hf]
    cases f x <;> cases f t <;> rfl

theorem xor2_vals (f : ℕ → Bool) : ∀ (ps cs : List ℕ),
    (((ps.zip cs).map fun pc => BGate.xor [pc.1, pc.2]).map (gval f)) = List.zipWith (· ^^ ·) (ps.map f) (cs.map f)
  | [], _ => rfl
  | _ :: _, [] => rfl
  | p :: ps, c :: cs => by
    simp only [List.zip_cons_cons, List.map_cons, List.zipWith_cons_cons, xor2_vals f ps cs, gval,
      List.map_nil, List.foldl_cons, List.foldl_nil, Bool.false_xor]

/-- **`a + ¬s + c0`** on wires: its `p` wires hold `a ⊕ ¬s`, its carries `carryBits`. -/
theorem chain_spec (y : List Bool) (w1 : ℕ) (gs : List BGate) (a s : List ℕ) (c0 : ℕ) (hlen : a.length = s.length)
    (ha : ∀ i ∈ a, i < gs.length) (hs : ∀ i ∈ s, i < gs.length) (hw1 : w1 < gs.length) (h1 : V gs y w1 = true)
    (hc0 : c0 < gs.length) :
    Ext gs (chain w1 gs a s c0).1 ∧
      (∀ w ∈ (chain w1 gs a s c0).2.1, w < (chain w1 gs a s c0).1.length) ∧
      (∀ w ∈ (chain w1 gs a s c0).2.2, w < (chain w1 gs a s c0).1.length) ∧
      (chain w1 gs a s c0).2.2 ≠ [] ∧
      (chain w1 gs a s c0).2.1.map (V (chain w1 gs a s c0).1 y) =
        List.zipWith (· ^^ ·) (a.map (V gs y)) ((s.map (V gs y)).map (!·)) ∧
      (chain w1 gs a s c0).2.2.map (V (chain w1 gs a s c0).1 y) =
        carryBits (a.map (V gs y)) (List.zipWith (· ^^ ·) (a.map (V gs y)) ((s.map (V gs y)).map (!·)))
          (V gs y c0) := by
  have hzl : (a.zip s).length = a.length := by simp [hlen]
  have hch : chain w1 gs a s c0 =
      ((carries (gs ++ (a.zip s).map fun xy => BGate.xor [xy.1, xy.2, w1])
        (a.zip ((List.range (a.zip s).length).map (gs.length + ·))) c0).1,
       (List.range (a.zip s).length).map (gs.length + ·),
       (carries (gs ++ (a.zip s).map fun xy => BGate.xor [xy.1, xy.2, w1])
        (a.zip ((List.range (a.zip s).length).map (gs.length + ·))) c0).2) := rfl
  rw [hch]
  generalize hB : ((a.zip s).map fun xy => BGate.xor [xy.1, xy.2, w1]) = B
  have hBl : B.length = a.length := by rw [← hB, List.length_map, hzl]
  rw [hzl]
  generalize hps : (List.range a.length).map (gs.length + ·) = ps
  have hpsl : ps.length = a.length := by rw [← hps]; simp
  have hGl : (gs ++ B).length = gs.length + a.length := by simp [hBl]
  have heB : Ext gs (gs ++ B) := ⟨B, rfl⟩
  have hpsb : ∀ w ∈ ps, w < (gs ++ B).length := by
    intro w hw
    rw [← hps] at hw
    simp only [List.mem_map, List.mem_range] at hw
    obtain ⟨k, hk, rfl⟩ := hw
    omega
  have hpv : ps.map (V (gs ++ B) y) = List.zipWith (· ^^ ·) (a.map (V gs y)) ((s.map (V gs y)).map (!·)) := by
    rw [← hps, ← hBl, V_batch_map gs B y, ← hB, xor3_vals _ w1 h1]
    · intro g hg
      rw [← hB] at hg
      simp only [List.mem_map] at hg
      obtain ⟨_, _, rfl⟩ := hg
      simp
    · intro g hg i hi
      rw [← hB] at hg
      simp only [List.mem_map] at hg
      obtain ⟨⟨x, t⟩, hxt, rfl⟩ := hg
      have := List.of_mem_zip hxt
      simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
      rcases hi with rfl | rfl | rfl
      · exact ha _ this.1
      · exact hs _ this.2
      · exact hw1
  obtain ⟨he, hw, hv⟩ := carries_spec y (a.zip ps) (gs ++ B) c0
    (fun xp hxp => by
      obtain ⟨x, p⟩ := xp
      have := List.of_mem_zip hxp
      exact ⟨lt_of_lt_of_le (ha x this.1) heB.le, hpsb p this.2⟩)
    (lt_of_lt_of_le hc0 heB.le)
  refine ⟨heB.trans he, fun w hw' => lt_of_lt_of_le (hpsb w hw') he.le, hw, carries_ne_nil _ _ _, ?_, ?_⟩
  · rw [List.map_congr_left fun w hw' => he.V y (hpsb w hw'), hpv]
  · rw [hv, zip_map_fst (V (gs ++ B) y) a ps hpsl.ge, zip_map_snd (V (gs ++ B) y) a ps hpsl.le, hpv,
      heB.V y hc0, List.map_congr_left fun i hi => heB.V y (ha i hi)]

theorem sums_spec (y : List Bool) (gs : List BGate) (ps cs : List ℕ) (hp : ∀ w ∈ ps, w < gs.length)
    (hc : ∀ w ∈ cs, w < gs.length) :
    Ext gs (sums gs ps cs).1 ∧ (∀ w ∈ (sums gs ps cs).2, w < (sums gs ps cs).1.length) ∧
      (sums gs ps cs).2.map (V (sums gs ps cs).1 y) = List.zipWith (· ^^ ·) (ps.map (V gs y)) (cs.map (V gs y)) := by
  have hch : sums gs ps cs = (gs ++ (ps.zip cs).map (fun pc => BGate.xor [pc.1, pc.2]),
      (List.range (ps.zip cs).length).map (gs.length + ·)) := rfl
  rw [hch]
  generalize hB : (ps.zip cs).map (fun pc => BGate.xor [pc.1, pc.2]) = B
  have hBl : B.length = (ps.zip cs).length := by rw [← hB, List.length_map]
  refine ⟨⟨B, rfl⟩, ?_, ?_⟩
  · intro w hw
    simp only [List.mem_map, List.mem_range] at hw
    obtain ⟨k, hk, rfl⟩ := hw
    simp only [List.length_append]
    omega
  · rw [← hBl, V_batch_map gs B y, ← hB, xor2_vals]
    · intro g hg
      rw [← hB] at hg
      simp only [List.mem_map] at hg
      obtain ⟨_, _, rfl⟩ := hg
      simp
    · intro g hg i hi
      rw [← hB] at hg
      simp only [List.mem_map] at hg
      obtain ⟨⟨p, c⟩, hpc, rfl⟩ := hg
      have := List.of_mem_zip hpc
      simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
      rcases hi with rfl | rfl
      · exact hp _ this.1
      · exact hc _ this.2

theorem carryBits_sums : ∀ (xs ys : List Bool) (c : Bool), xs.length = ys.length →
    List.zipWith (· ^^ ·) (List.zipWith (· ^^ ·) xs ys) (carryBits xs (List.zipWith (· ^^ ·) xs ys) c) =
      (ripple xs ys c).1
  | [], [], _, _ => by simp [carryBits, ripple]
  | x :: xs, y :: ys, c, h => by
    simp only [List.zipWith_cons_cons, carryBits, ripple,
      carryBits_sums xs ys (x ^^ ((x ^^ y) && (x ^^ c))) (by simpa using h)]
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h

theorem carryBits_last : ∀ (xs ys : List Bool) (c d : Bool), xs.length = ys.length →
    (carryBits xs (List.zipWith (· ^^ ·) xs ys) c).getLastD d = (ripple xs ys c).2
  | [], [], _, _, _ => by simp [carryBits, ripple]
  | x :: xs, y :: ys, c, d, h => by
    simp only [List.zipWith_cons_cons, carryBits, ripple, List.getLastD_cons]
    exact carryBits_last xs ys _ c (by simpa using h)
  | [], _ :: _, _, _, h => by simp at h
  | _ :: _, [], _, _, h => by simp at h

/-- `a + ¬s + c0` then its sums: the sum bits' values. -/
theorem sub_spec (y : List Bool) (w1 : ℕ) (gs : List BGate) (a s : List ℕ) (c0 : ℕ) (hlen : a.length = s.length)
    (ha : ∀ i ∈ a, i < gs.length) (hs : ∀ i ∈ s, i < gs.length) (hw1 : w1 < gs.length) (h1 : V gs y w1 = true)
    (hc0 : c0 < gs.length) :
    Ext gs (sums (chain w1 gs a s c0).1 (chain w1 gs a s c0).2.1 (chain w1 gs a s c0).2.2).1 ∧
      (∀ w ∈ (sums (chain w1 gs a s c0).1 (chain w1 gs a s c0).2.1 (chain w1 gs a s c0).2.2).2,
        w < (sums (chain w1 gs a s c0).1 (chain w1 gs a s c0).2.1 (chain w1 gs a s c0).2.2).1.length) ∧
      (sums (chain w1 gs a s c0).1 (chain w1 gs a s c0).2.1 (chain w1 gs a s c0).2.2).2.map
          (V (sums (chain w1 gs a s c0).1 (chain w1 gs a s c0).2.1 (chain w1 gs a s c0).2.2).1 y) =
        (ripple (a.map (V gs y)) ((s.map (V gs y)).map (!·)) (V gs y c0)).1 := by
  obtain ⟨he, hp, hc, -, hvp, hvc⟩ := chain_spec y w1 gs a s c0 hlen ha hs hw1 h1 hc0
  obtain ⟨he2, hw2, hv2⟩ := sums_spec y _ _ _ hp hc
  exact ⟨he.trans he2, hw2, by rw [hv2, hvp, hvc, carryBits_sums _ _ _ (by simp [hlen])]⟩

/-- `a + ¬s + c0`'s last carry. -/
theorem chain_last (y : List Bool) (w1 : ℕ) (gs : List BGate) (a s : List ℕ) (c0 : ℕ) (hlen : a.length = s.length)
    (ha : ∀ i ∈ a, i < gs.length) (hs : ∀ i ∈ s, i < gs.length) (hw1 : w1 < gs.length) (h1 : V gs y w1 = true)
    (hc0 : c0 < gs.length) :
    (chain w1 gs a s c0).2.2.getLastD 0 < (chain w1 gs a s c0).1.length ∧
      V (chain w1 gs a s c0).1 y ((chain w1 gs a s c0).2.2.getLastD 0) =
        (ripple (a.map (V gs y)) ((s.map (V gs y)).map (!·)) (V gs y c0)).2 := by
  obtain ⟨-, -, hc, hne, -, hvc⟩ := chain_spec y w1 gs a s c0 hlen ha hs hw1 h1 hc0
  obtain ⟨c, cs, hcs⟩ := List.exists_cons_of_ne_nil hne
  refine ⟨hc _ (hcs ▸ getLastD_mem_cons c cs 0), ?_⟩
  rw [map_getLastD (V (chain w1 gs a s c0).1 y), hvc, carryBits_last _ _ _ _ (by simp [hlen])]

/-! ## The inputs and their fields -/

theorem initGates_length (K : ℕ) : (initGates K).length = inputs K + 2 := by simp [initGates]

theorem init_vals (K : ℕ) (y : List Bool) {gs : List BGate} (he : Ext (initGates K) gs) :
    (∀ i < inputs K, V gs y i = y.getD i false) ∧ V gs y (inputs K) = false ∧ V gs y (inputs K + 1) = true ∧
      inputs K + 2 ≤ gs.length := by
  have hl := he.le
  rw [initGates_length] at hl
  have hsplit : initGates K = (List.replicate (inputs K) BGate.input ++ [.const false]) ++ [.const true] := by
    simp [initGates]
  have hb : ∀ i < inputs K + 2, V gs y i = V (initGates K) y i :=
    fun i hi => he.V y (by rw [initGates_length]; exact hi)
  refine ⟨fun i hi => ?_, ?_, ?_, hl⟩
  · rw [hb i (by omega), hsplit, V_append _ y (by simp; omega), V_append _ y (by simp; omega), V_inputs _ y hi]
  · have h := V_const (List.replicate (inputs K) BGate.input) false y
    rw [List.length_replicate] at h
    rw [hb _ (by omega), hsplit, V_append _ y (by simp), h]
  · have h := V_const (List.replicate (inputs K) BGate.input ++ [.const false]) true y
    simp only [List.length_append, List.length_replicate, List.length_singleton] at h
    rw [hb _ (by omega), hsplit, h]

theorem length_flatMap_mem {α : Type} (f : α → List Bool) {m : ℕ} :
    ∀ l : List α, (∀ a ∈ l, (f a).length = m) → (l.flatMap f).length = l.length * m
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, h a (by simp),
      length_flatMap_mem f l (fun b hb => h b (List.mem_cons_of_mem a hb)), List.length_cons, Nat.succ_mul,
      Nat.add_comm]

theorem getD_flatMap_mem {α : Type} (f : α → List Bool) {m : ℕ} :
    ∀ (l : List α), (∀ a ∈ l, (f a).length = m) → ∀ {g : ℕ} {a : α}, l[g]? = some a → ∀ {i : ℕ}, i < m →
      (l.flatMap f).getD (m * g + i) false = (f a).getD i false
  | [], _, _, _, h, _, _ => by simp at h
  | b :: l, hf, 0, a, h, i, hi => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, Nat.mul_zero, Nat.zero_add,
      List.getElem?_append_left (by rw [hf b (by simp)]; exact hi)]
  | b :: l, hf, g + 1, a, h, i, hi => by
    simp only [List.getElem?_cons_succ] at h
    have hb := hf b (by simp)
    have hm : m * (g + 1) + i - m = m * g + i := by rw [Nat.mul_succ]; omega
    rw [List.flatMap_cons, List.getD_eq_getElem?_getD,
      List.getElem?_append_right (by rw [hb, Nat.mul_succ]; omega), hb, hm, ← List.getD_eq_getElem?_getD]
    exact getD_flatMap_mem f l (fun c hc => hf c (List.mem_cons_of_mem b hc)) h hi

theorem lt_of_getElem? {α : Type} {l : List α} {j : ℕ} {a : α} (h : l[j]? = some a) : j < l.length :=
  (List.getElem?_eq_some_iff.1 h).1

theorem encode_row {os : List RoundOpening} (hfit : ∀ o ∈ os, o.Fits) {j : ℕ} {o : RoundOpening}
    (ho : os[j]? = some o) {i : ℕ} (hi : i < recordBits) :
    (encode os).getD (recordBits * j + i) false = o.row.getD i false := by
  have hj := lt_of_getElem? ho
  have hl := length_flatMap_mem (·.row) os fun o h => (hfit o h).1
  unfold encode
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_left (by rw [hl]; simp only [recordBits] at hi ⊢; omega), ← List.getD_eq_getElem?_getD]
  exact getD_flatMap_mem (·.row) os (fun o h => (hfit o h).1) ho hi

theorem encode_expected {os : List RoundOpening} (hfit : ∀ o ∈ os, o.Fits) {j : ℕ} {o : RoundOpening}
    (ho : os[j]? = some o) {t : ℕ} (ht : t < digestBits) :
    (encode os).getD (recordBits * os.length + digestBits * j + t) false = o.expected.getD t false := by
  have hl := length_flatMap_mem (·.row) os fun o h => (hfit o h).1
  have hidx : recordBits * os.length + digestBits * j + t - os.length * recordBits = digestBits * j + t := by
    simp only [recordBits]; omega
  unfold encode
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by rw [hl]; simp only [recordBits]; omega), hl, hidx,
    ← List.getD_eq_getElem?_getD]
  exact getD_flatMap_mem (·.expected) os (fun o h => (hfit o h).2) ho ht

theorem beWires_length (base pos n : ℕ) : (beWires base pos n).length = 8 * n := by simp [beWires]

theorem beWires_shift (base pos n : ℕ) : beWires base pos n = (beWires 0 pos n).map (base + ·) := by
  unfold beWires
  rw [List.map_map]
  exact List.map_congr_left fun k _ => by simp only [Function.comp_apply]; omega

theorem beWires_lt {pos n i : ℕ} (h : i ∈ beWires 0 pos n) : i < 8 * (pos + n) := by
  unfold beWires at h
  simp only [List.mem_map, List.mem_range] at h
  obtain ⟨k, hk, rfl⟩ := h
  omega

theorem beWires_mem {base pos n i : ℕ} (h : i ∈ beWires base pos n) : base ≤ i ∧ i < base + 8 * (pos + n) := by
  rw [beWires_shift] at h
  simp only [List.mem_map] at h
  obtain ⟨k, hk, rfl⟩ := h
  have := beWires_lt hk
  omega

theorem row_vals {f : ℕ → Bool} {o : RoundOpening} {j : ℕ}
    (hf : ∀ i < recordBits, f (recordBits * j + i) = o.row.getD i false) {pos n : ℕ} (hpn : pos + n ≤ 92) :
    (beWires (recordBits * j) pos n).map f = (beWires 0 pos n).map fun i => o.row.getD i false := by
  rw [beWires_shift, List.map_map]
  exact List.map_congr_left fun i hi => by
    rw [Function.comp_apply]
    exact hf i (by have := beWires_lt hi; simp only [recordBits]; omega)

theorem constWires_length (w0 w1 width v : ℕ) : (constWires w0 w1 width v).length = width := by simp [constWires]

theorem constWires_vals {f : ℕ → Bool} {w0 w1 : ℕ} (h0 : f w0 = false) (h1 : f w1 = true) (width v : ℕ) :
    (constWires w0 w1 width v).map f = bitsLE width v := by
  unfold constWires bitsLE
  rw [List.map_map]
  exact List.map_congr_left fun i _ => by
    simp only [Function.comp_apply]
    cases h : v.testBit i <;> simp [h0, h1]

theorem constWires_mem {w0 w1 width v w : ℕ} (h : w ∈ constWires w0 w1 width v) : w = w0 ∨ w = w1 := by
  unfold constWires at h
  simp only [List.mem_map, List.mem_range] at h
  obtain ⟨i, _, rfl⟩ := h
  split <;> simp

theorem bitsLE_succ (n v : ℕ) : bitsLE (n + 1) v = v.testBit 0 :: bitsLE n (v / 2) := by
  simp only [bitsLE, List.range_succ_eq_map, List.map_cons, List.map_map]
  congr 1
  refine List.map_congr_left fun i _ => ?_
  simp [Function.comp, Nat.testBit_succ]

theorem bitsLE_bitsVal : ∀ l : List Bool, bitsLE l.length (bitsVal l) = l
  | [] => by simp [bitsLE]
  | b :: l => by
    rw [List.length_cons, bitsLE_succ, bitsVal]
    have h2 : (b.toNat + 2 * bitsVal l) / 2 = bitsVal l := by
      cases b <;> simp only [Bool.toNat_false, Bool.toNat_true] <;> omega
    have h0 : (b.toNat + 2 * bitsVal l).testBit 0 = b := by
      rw [Nat.testBit_zero]
      cases b <;> simp only [Bool.toNat_false, Bool.toNat_true, decide_eq_true_eq, decide_eq_false_iff_not] <;> omega
    rw [h2, h0, bitsLE_bitsVal l]

theorem field_eq_iff {l : List Bool} {n j : ℕ} (hl : l.length = n) (hj : j < 2 ^ n) :
    l = bitsLE n j ↔ bitsVal l = j := by
  constructor
  · rintro rfl
    rw [bitsVal_bitsLE, Nat.mod_eq_of_lt hj]
  · intro h
    rw [← h, ← hl, bitsLE_bitsVal]

theorem range_getD {l : List Bool} {n : ℕ} (h : l.length = n) : (List.range n).map (fun t => l.getD t false) = l := by
  apply List.ext_getElem (by simp [h])
  intro i h1 h2
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]

/-- A batch of `Xor3(a, b, 1)` gates is all ones iff the two sides agree. -/
theorem eqs_all (f : ℕ → Bool) (w1 : ℕ) (hf : f w1 = true) {α : Type} (a b : α → ℕ) : ∀ l : List α,
    ((l.map fun t => BGate.xor [a t, b t, w1]).map (gval f)).all id =
      decide (l.map (fun t => f (a t)) = l.map (fun t => f (b t)))
  | [] => by simp
  | t :: l => by
    have ih := eqs_all f w1 hf a b l
    simp only [List.map_cons, List.all_cons, gval, List.map_nil, List.foldl_cons, List.foldl_nil, hf, id,
      List.cons.injEq, ih]
    cases f (a t) <;> cases f (b t) <;> simp

/-! ## A round -/

/-- **Round `j`'s hit wire is its verdict**: after the inputs (and any earlier rounds), round `j`'s gates read its
opening and their last wire is `1` iff `roundVerdict` is a hit. -/
theorem round_spec {os : List RoundOpening} {K : ℕ} (LIMIT : ℕ) (hK : os.length = K) (hK32 : K ≤ 2 ^ 32)
    (hfit : ∀ o ∈ os, o.Fits) {j : ℕ} {o : RoundOpening} (ho : os[j]? = some o) {gs : List BGate}
    (he : Ext (initGates K) gs) :
    Ext gs (roundGates K LIMIT (inputs K) (inputs K + 1) j gs).1 ∧
      (roundGates K LIMIT (inputs K) (inputs K + 1) j gs).2 <
        (roundGates K LIMIT (inputs K) (inputs K + 1) j gs).1.length ∧
      V (roundGates K LIMIT (inputs K) (inputs K + 1) j gs).1 (encode os)
          (roundGates K LIMIT (inputs K) (inputs K + 1) j gs).2 = decide (roundVerdict LIMIT j o = .hit) := by
  subst hK
  have hj : j < os.length := lt_of_getElem? ho
  have hfo := hfit o (List.mem_of_getElem? ho)
  obtain ⟨hin, h0, h1, hgl⟩ := init_vals os.length (encode os) he
  have hrow : ∀ i < recordBits, V gs (encode os) (recordBits * j + i) = o.row.getD i false := fun i hi => by
    rw [hin _ (by unfold inputs; simp only [recordBits, digestBits] at hi ⊢; nlinarith), encode_row hfit ho hi]
  have hexp : ∀ t < digestBits,
      V gs (encode os) (recordBits * os.length + digestBits * j + t) = o.expected.getD t false := fun t ht => by
    rw [hin _ (by unfold inputs; simp only [recordBits, digestBits] at ht ⊢; nlinarith), encode_expected hfit ho ht]
  dsimp only [roundGates]
  generalize hy : encode os = y at hin h0 h1 hrow hexp ⊢
  generalize hw0 : inputs os.length = w0 at hin h0 h1 hgl ⊢
  -- the equalities
  have hrowb : ∀ {pos n i}, pos + n ≤ 92 → i ∈ beWires (recordBits * j) pos n → i < gs.length := by
    intro pos n i hpn hi
    have := beWires_mem hi
    have : recordBits * j + 736 ≤ inputs os.length := by unfold inputs; simp only [recordBits, digestBits]; nlinarith
    simp only [recordBits] at *
    omega
  have hlitr : ∀ g ∈ litGates os.length w0 (w0 + 1) j, g ≠ .input ∧ ∀ i ∈ g.reads, i < gs.length := by
    intro g hg
    unfold litGates at hg
    simp only [List.mem_append, List.mem_map, List.mem_range] at hg
    rcases hg with ⟨⟨x, c⟩, hxc, rfl⟩ | ⟨t, ht, rfl⟩
    · refine ⟨by simp, fun i hi => ?_⟩
      have hm := List.of_mem_zip hxc
      simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
      rcases hi with rfl | rfl | rfl
      · exact hrowb (by norm_num) hm.1
      · rcases constWires_mem hm.2 with rfl | rfl <;> omega
      · omega
    · refine ⟨by simp, fun i hi => ?_⟩
      simp only [BGate.reads, List.mem_cons, List.not_mem_nil, or_false] at hi
      have : inputs os.length = w0 := hw0
      unfold inputs at this
      simp only [recordBits, digestBits] at *
      rcases hi with rfl | rfl | rfl <;> nlinarith
  have hl : (List.range digestBits).map (fun t => V gs y (recordBits * j + 224 + t)) = o.digest := by
    unfold RoundOpening.digest
    exact List.map_congr_left fun t ht => by
      rw [Nat.add_assoc]
      exact hrow _ (by simp only [List.mem_range, digestBits] at ht; simp only [recordBits]; omega)
  have hr : (List.range digestBits).map (fun t => V gs y (recordBits * os.length + digestBits * j + t)) =
      o.expected :=
    (List.map_congr_left fun t ht => hexp t (by simpa using ht)).trans (range_getD hfo.2)
  have hJ : (beWires 0 0 4).map (fun i => o.row.getD i false) = bitsLE 32 j ↔ o.j = j :=
    field_eq_iff (by simp [beWires_length]) (lt_of_lt_of_le hj hK32)
  have hlit : ((litGates os.length w0 (w0 + 1) j).map (gval (V gs y))).all id =
      (decide (o.j = j) && decide (o.digest = o.expected)) := by
    unfold litGates
    rw [List.map_append, List.all_append, eqs_all _ (w0 + 1) h1 Prod.fst Prod.snd,
      eqs_all _ (w0 + 1) h1 (fun t => recordBits * j + 224 + t)
        (fun t => recordBits * os.length + digestBits * j + t),
      zip_map_fst (V gs y) _ _ (by simp [beWires_length, constWires_length]),
      zip_map_snd (V gs y) _ _ (by simp [beWires_length, constWires_length]), row_vals hrow (by norm_num),
      constWires_vals h0 h1, hl, hr, decide_eq_decide.2 hJ]
  have hw1g : w0 + 1 < gs.length := by omega
  set G1 := gs ++ litGates os.length w0 (w0 + 1) j with hG1
  have heL : Ext gs G1 := ⟨_, rfl⟩
  have hG1v : ∀ i < gs.length, V G1 y i = V gs y i := fun i hi => heL.V y hi
  have hRb : ∀ i ∈ beWires (recordBits * j) 20 8, i < G1.length :=
    fun i hi => lt_of_lt_of_le (hrowb (by norm_num) hi) heL.le
  have hSb : ∀ i ∈ beWires (recordBits * j) 12 8, i < G1.length :=
    fun i hi => lt_of_lt_of_le (hrowb (by norm_num) hi) heL.le
  have hRv : (beWires (recordBits * j) 20 8).map (V G1 y) = (beWires 0 20 8).map fun i => o.row.getD i false := by
    rw [List.map_congr_left fun i hi => hG1v i (hrowb (by norm_num) hi)]
    exact row_vals hrow (by norm_num)
  have hSv : (beWires (recordBits * j) 12 8).map (V G1 y) = (beWires 0 12 8).map fun i => o.row.getD i false := by
    rw [List.map_congr_left fun i hi => hG1v i (hrowb (by norm_num) hi)]
    exact row_vals hrow (by norm_num)
  have hV1 : V G1 y (w0 + 1) = true := by rw [hG1v _ hw1g, h1]
  have hw1G : w0 + 1 < G1.length := lt_of_lt_of_le hw1g heL.le
  have hlen1 : (beWires (recordBits * j) 20 8).length = (beWires (recordBits * j) 12 8).length := by
    simp [beWires_length]
  obtain ⟨hc1e, hc1p, hc1c, -, -, -⟩ := chain_spec y (w0 + 1) G1 _ _ (w0 + 1) hlen1 hRb hSb hw1G hV1 hw1G
  obtain ⟨hl1, hv1⟩ := chain_last y (w0 + 1) G1 _ _ (w0 + 1) hlen1 hRb hSb hw1G hV1 hw1G
  obtain ⟨-, hdw, hdv⟩ := sub_spec y (w0 + 1) G1 _ _ (w0 + 1) hlen1 hRb hSb hw1G hV1 hw1G
  obtain ⟨hde, -, -⟩ := sums_spec y _ _ _ hc1p hc1c
  rw [hRv, hSv, hV1] at hv1 hdv
  set c1 := chain (w0 + 1) G1 (beWires (recordBits * j) 20 8) (beWires (recordBits * j) 12 8) (w0 + 1) with hc1
  set d := sums c1.1 c1.2.1 c1.2.2 with hd
  set Rv := (beWires 0 20 8).map fun i => o.row.getD i false with hRv'
  set Sv := (beWires 0 12 8).map fun i => o.row.getD i false with hSv'
  have hRl : Rv.length = 64 := by simp [Rv, beWires_length]
  have hSl : Sv.length = 64 := by simp [Sv, beWires_length]
  rw [ripple_le Rv Sv (by rw [hRl, hSl])] at hv1
  have hdl : d.2.length = 64 := by
    have := congrArg List.length hdv
    rw [List.length_map, ripple_length _ _ _ (by simp [hRl, hSl])] at this
    rw [this, hRl]
  have hGd : Ext gs d.1 := heL.trans (hc1e.trans hde)
  have hd0 : V d.1 y w0 = false := by rw [hGd.V y (by omega), h0]
  have hd1 : V d.1 y (w0 + 1) = true := by rw [hGd.V y hw1g, h1]
  have hcb : ∀ i ∈ constWires w0 (w0 + 1) 64 (min LIMIT (2 ^ 64 - 1)), i < d.1.length := fun i hi => by
    have := hGd.le
    rcases constWires_mem hi with rfl | rfl <;> omega
  have hlen2 : (constWires w0 (w0 + 1) 64 (min LIMIT (2 ^ 64 - 1))).length = d.2.length := by
    rw [constWires_length, hdl]
  have hw1d : w0 + 1 < d.1.length := lt_of_lt_of_le hw1g hGd.le
  obtain ⟨hc2e, -, -, -, -, -⟩ := chain_spec y (w0 + 1) d.1 _ _ (w0 + 1) hlen2 hcb hdw hw1d hd1 hw1d
  obtain ⟨-, hv2⟩ := chain_last y (w0 + 1) d.1 _ _ (w0 + 1) hlen2 hcb hdw hw1d hd1 hw1d
  rw [constWires_vals hd0 hd1, hdv, hd1] at hv2
  set Dv := (ripple Rv (Sv.map (!·)) true).1 with hDv
  have hDl : Dv.length = 64 := by rw [hDv, ripple_length _ _ _ (by simp [hRl, hSl]), hRl]
  rw [ripple_le _ _ (by rw [hDl, bitsLE_length]), bitsVal_bitsLE] at hv2
  set c2 := chain (w0 + 1) d.1 (constWires w0 (w0 + 1) 64 (min LIMIT (2 ^ 64 - 1))) d.2 (w0 + 1) with hc2
  refine ⟨heL.trans (hc1e.trans (hde.trans (hc2e.trans ⟨_, rfl⟩))), by simp, ?_⟩
  rw [V_and, List.map_append, List.all_append]
  have hb : ∀ w ∈ (List.range (litGates os.length w0 (w0 + 1) j).length).map (gs.length + ·), w < G1.length := by
    intro w hw
    simp only [List.mem_map, List.mem_range] at hw
    obtain ⟨k, hk, rfl⟩ := hw
    simp only [G1, List.length_append]
    omega
  have hlv : ((List.range (litGates os.length w0 (w0 + 1) j).length).map (gs.length + ·)).map (V c2.1 y) =
      (litGates os.length w0 (w0 + 1) j).map (gval (V gs y)) := by
    rw [List.map_congr_left fun w hw => (hc1e.trans (hde.trans hc2e)).V y (hb w hw)]
    exact V_batch_map gs _ y (fun g hg => (hlitr g hg).1) (fun g hg => (hlitr g hg).2)
  rw [hlv, hlit]
  simp only [List.map_cons, List.map_nil, List.all_cons, List.all_nil, id, Bool.and_true]
  rw [(hde.trans hc2e).V y hl1, hv1, hv2]
  have hrlt : o.receipt < 2 ^ 64 := by
    have := bitsVal_lt Rv
    rw [hRl] at this
    simpa only [Rv, RoundOpening.receipt, beField] using this
  have hLm : min LIMIT (2 ^ 64 - 1) % 2 ^ 64 = min LIMIT (2 ^ 64 - 1) := Nat.mod_eq_of_lt (by omega)
  rw [hLm]
  have hsend : bitsVal Sv = o.send := by simp only [Sv, RoundOpening.send, beField]
  have hrec : bitsVal Rv = o.receipt := by simp only [Rv, RoundOpening.receipt, beField]
  rw [hsend, hrec]
  by_cases hs : o.send ≤ o.receipt
  · have hdv' : bitsVal Dv = o.receipt - o.send := by
      rw [← hsend, ← hrec]
      exact ripple_sub Rv Sv (by rw [hRl, hSl]) (by rw [hsend, hrec]; exact hs)
    have hm : (o.receipt - o.send ≤ min LIMIT (2 ^ 64 - 1)) ↔ (o.receipt - o.send ≤ LIMIT) := by omega
    rw [hdv']
    simp only [hm]
    unfold roundVerdict
    by_cases hjj : o.j = j <;> by_cases hdg : o.digest = o.expected <;>
      by_cases hlim : o.receipt - o.send ≤ LIMIT <;> simp [hjj, hdg, hlim, hs]
  · unfold roundVerdict
    simp [hs]

/-! ## Every round, the counter and the verdict -/

theorem verdicts_getD {LIMIT : ℕ} {os : List RoundOpening} {j : ℕ} {o : RoundOpening} (ho : os[j]? = some o) :
    (verdicts LIMIT os).getD j .reject = roundVerdict LIMIT j o := by
  rw [List.getD_eq_getElem?_getD, verdicts, List.getElem?_mapIdx, ho, Option.map_some, Option.getD_some]

theorem hits_spec {os : List RoundOpening} {K : ℕ} (LIMIT : ℕ) (hK : os.length = K) (hK32 : K ≤ 2 ^ 32)
    (hfit : ∀ o ∈ os, o.Fits) : ∀ n ≤ K,
    Ext (initGates K) ((List.range n).foldl (hitStep K LIMIT) (initGates K, [])).1 ∧
      (∀ h ∈ ((List.range n).foldl (hitStep K LIMIT) (initGates K, [])).2,
        h < ((List.range n).foldl (hitStep K LIMIT) (initGates K, [])).1.length) ∧
      ((List.range n).foldl (hitStep K LIMIT) (initGates K, [])).2.map
          (V ((List.range n).foldl (hitStep K LIMIT) (initGates K, [])).1 (encode os)) =
        (List.range n).map fun j => decide ((verdicts LIMIT os).getD j .reject = .hit)
  | 0, _ => ⟨Ext.refl _, by simp, by simp⟩
  | n + 1, hn => by
    obtain ⟨he, hb, hv⟩ := hits_spec LIMIT hK hK32 hfit n (by omega)
    obtain ⟨o, ho⟩ : ∃ o, os[n]? = some o := ⟨os[n]'(by omega), List.getElem?_eq_getElem (by omega)⟩
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    generalize (List.range n).foldl (hitStep K LIMIT) (initGates K, []) = st at he hb hv
    obtain ⟨he2, hl2, hv2⟩ := round_spec LIMIT hK hK32 hfit ho he
    have hst : hitStep K LIMIT st n = ((roundGates K LIMIT (inputs K) (inputs K + 1) n st.1).1,
        st.2 ++ [(roundGates K LIMIT (inputs K) (inputs K + 1) n st.1).2]) := rfl
    rw [hst]
    generalize roundGates K LIMIT (inputs K) (inputs K + 1) n st.1 = r at he2 hl2 hv2
    dsimp only
    refine ⟨he.trans he2, ?_, ?_⟩
    · intro h hh
      rw [List.mem_append, List.mem_singleton] at hh
      rcases hh with hh | rfl
      · exact lt_of_lt_of_le (hb h hh) he2.le
      · exact hl2
    · have hm : st.2.map (V r.1 (encode os)) = st.2.map (V st.1 (encode os)) :=
        List.map_congr_left fun h hh => he2.V _ (hb h hh)
      rw [List.map_append, List.map_cons, List.map_nil, List.map_append, List.map_cons, List.map_nil, hm, hv, hv2,
        verdicts_getD ho]

theorem count_spec (K M : ℕ) (y : List Bool) : ∀ (hs : List ℕ) (st : List BGate × List ℕ) (c : ℕ),
    Ext (initGates K) st.1 → (∀ h ∈ hs, h < st.1.length) → (∀ w ∈ st.2, w < st.1.length) →
    st.2.length = counterWidth K M → bitsVal (st.2.map (V st.1 y)) = c → c + hs.length < 2 ^ counterWidth K M →
    Ext st.1 (hs.foldl (countStep (inputs K + 1) (counterWidth K M)) st).1 ∧
      (∀ w ∈ (hs.foldl (countStep (inputs K + 1) (counterWidth K M)) st).2,
        w < (hs.foldl (countStep (inputs K + 1) (counterWidth K M)) st).1.length) ∧
      (hs.foldl (countStep (inputs K + 1) (counterWidth K M)) st).2.length = counterWidth K M ∧
      bitsVal ((hs.foldl (countStep (inputs K + 1) (counterWidth K M)) st).2.map
          (V (hs.foldl (countStep (inputs K + 1) (counterWidth K M)) st).1 y)) =
        c + ((hs.map (V st.1 y)).filter (!·)).length
  | [], st, c, _, _, hw, hl, hv, _ => ⟨Ext.refl _, hw, hl, by simpa using hv⟩
  | h :: hs, st, c, he, hh, hw, hl, hv, hlt => by
    obtain ⟨_, h1, hgl⟩ := (init_vals K y he).2
    have hG : Ext st.1 (st.1 ++ [.not h]) := ⟨_, rfl⟩
    have hGl : (st.1 ++ [BGate.not h]).length = st.1.length + 1 := by simp
    have hnot : V (st.1 ++ [.not h]) y st.1.length = !(V st.1 y h) := V_last _ _ (by simp) y
    have hw' : ∀ w ∈ st.2, w < (st.1 ++ [BGate.not h]).length := fun w hw' => by have := hw w hw'; omega
    have hr' : ∀ w ∈ List.replicate (counterWidth K M) (inputs K + 1), w < (st.1 ++ [BGate.not h]).length :=
      fun w hw' => by rw [List.eq_of_mem_replicate hw']; omega
    have h1' : V (st.1 ++ [BGate.not h]) y (inputs K + 1) = true := by rw [hG.V y (by omega), h1]
    obtain ⟨he2, hw2, hv2⟩ := sub_spec y (inputs K + 1) (st.1 ++ [.not h]) st.2
      (List.replicate (counterWidth K M) (inputs K + 1)) st.1.length (by simp [hl]) hw' hr' (by omega) h1'
      (by omega)
    have hA' : st.2.map (V (st.1 ++ [.not h]) y) = st.2.map (V st.1 y) :=
      List.map_congr_left fun w hw' => hG.V y (hw w hw')
    rw [List.map_replicate, h1', List.map_replicate, Bool.not_true, hnot, hA'] at hv2
    have hinc : bitsVal (ripple (st.2.map (V st.1 y)) (List.replicate (counterWidth K M) false)
        (!(V st.1 y h))).1 = c + (!(V st.1 y h)).toNat := by
      have := ripple_inc (st.2.map (V st.1 y)) (!(V st.1 y h)) (by
        rw [hv, List.length_map, hl]
        have : (!(V st.1 y h)).toNat ≤ 1 := by cases V st.1 y h <;> simp
        simp only [List.length_cons] at hlt
        omega)
      rw [List.length_map, hl, hv] at this
      exact this
    rw [← hv2] at hinc
    have hst : (h :: hs).foldl (countStep (inputs K + 1) (counterWidth K M)) st =
        hs.foldl (countStep (inputs K + 1) (counterWidth K M)) (countStep (inputs K + 1) (counterWidth K M) st h) :=
      rfl
    rw [hst]
    have hl3 : (countStep (inputs K + 1) (counterWidth K M) st h).2.length = counterWidth K M := by
      have := congrArg List.length hv2
      rw [List.length_map, ripple_length _ _ _ (by simp [hl]), List.length_map, hl] at this
      exact this
    obtain ⟨he4, hw4, hl4, hv4⟩ := count_spec K M y hs (countStep (inputs K + 1) (counterWidth K M) st h)
      (c + (!(V st.1 y h)).toNat) (he.trans (hG.trans he2))
      (fun w hw' => lt_of_lt_of_le (hh w (List.mem_cons_of_mem _ hw')) (hG.trans he2).le) hw2 hl3 hinc (by
        have : (!(V st.1 y h)).toNat ≤ 1 := by cases V st.1 y h <;> simp
        simp only [List.length_cons] at hlt
        omega)
    refine ⟨(hG.trans he2).trans he4, hw4, hl4, ?_⟩
    have hm : hs.map (V (countStep (inputs K + 1) (counterWidth K M) st h).1 y) = hs.map (V st.1 y) :=
      List.map_congr_left fun w hw' => (hG.trans he2).V y (hh w (List.mem_cons_of_mem _ hw'))
    rw [hv4, hm, List.map_cons, List.filter_cons]
    cases V st.1 y h
    · simp
      omega
    · simp

theorem range_map_getD {α β : Type} (vs : List α) (d : α) (g : α → β) :
    (List.range vs.length).map (fun j => g (vs.getD j d)) = vs.map g := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by simpa using h2)]

theorem miss_count : ∀ vs : List ContVerdict,
    ((vs.map fun v => decide (v = .hit)).filter (!·)).length = (vs.filter (· ≠ .hit)).length
  | [] => rfl
  | v :: vs => by cases v <;> simp [miss_count vs]

theorem records_filter (rs : List ContRecord) :
    (rs.filter fun r => r.verdict ≠ .hit).length = ((rs.map (·.verdict)).filter (· ≠ .hit)).length := by
  rw [List.filter_map, List.length_map]
  rfl

theorem lt_counter (K M : ℕ) : K < 2 ^ counterWidth K M ∧ M < 2 ^ counterWidth K M := by
  have h := Nat.lt_size_self (max K M)
  exact ⟨lt_of_le_of_lt (le_max_left K M) h, lt_of_le_of_lt (le_max_right K M) h⟩

/-- **The verdict unit's output** on the openings of `K ≤ 2³²` rounds: `1` iff at most `M` rounds are not hits. -/
theorem verdictProg_eval {K M LIMIT : ℕ} {os : List RoundOpening} (hK : os.length = K) (hK32 : K ≤ 2 ^ 32)
    (hfit : ∀ o ∈ os, o.Fits) :
    (verdictProg K M LIMIT).eval (encode os) = [decide (((verdicts LIMIT os).filter (· ≠ .hit)).length ≤ M)] := by
  obtain ⟨hhe, hhb, hhv⟩ := hits_spec LIMIT hK hK32 hfit K le_rfl
  have hH : hitGates K LIMIT = (List.range K).foldl (hitStep K LIMIT) (initGates K, []) := rfl
  rw [← hH] at hhe hhb hhv
  generalize encode os = y at hhv ⊢
  obtain ⟨hK2, hM2⟩ := lt_counter K M
  obtain ⟨-, h0, h1, hgl⟩ := init_vals K y hhe
  have hHl : (hitGates K LIMIT).2.length = K := by
    have := congrArg List.length hhv
    simpa using this
  have hz : bitsVal ((List.replicate (counterWidth K M) (inputs K)).map (V (hitGates K LIMIT).1 y)) = 0 := by
    rw [List.map_replicate, h0]
    induction counterWidth K M with
    | zero => rfl
    | succ n ih => simp [List.replicate_succ, bitsVal, ih]
  obtain ⟨hce, hcw, hcl, hcv⟩ := count_spec K M y (hitGates K LIMIT).2 ((hitGates K LIMIT).1,
    List.replicate (counterWidth K M) (inputs K)) 0 hhe hhb
    (fun w hw => by rw [List.eq_of_mem_replicate hw]; dsimp only; omega) (by simp) hz (by rw [hHl]; omega)
  have hC : countGates K M LIMIT = (hitGates K LIMIT).2.foldl (countStep (inputs K + 1) (counterWidth K M))
      ((hitGates K LIMIT).1, List.replicate (counterWidth K M) (inputs K)) := rfl
  rw [← hC] at hce hcw hcl hcv
  have hvl : (verdicts LIMIT os).length = K := by simp [verdicts, hK]
  have hR := range_map_getD (verdicts LIMIT os) .reject (fun v => decide (v = .hit))
  rw [hvl] at hR
  rw [hhv, hR, miss_count, zero_add] at hcv
  have hCe : Ext (initGates K) (countGates K M LIMIT).1 := hhe.trans hce
  obtain ⟨-, h0', h1', hgl'⟩ := init_vals K y hCe
  have hcb : ∀ i ∈ constWires (inputs K) (inputs K + 1) (counterWidth K M) M, i < (countGates K M LIMIT).1.length :=
    fun i hi => by rcases constWires_mem hi with rfl | rfl <;> omega
  obtain ⟨-, hv⟩ := chain_last y (inputs K + 1) (countGates K M LIMIT).1
    (constWires (inputs K) (inputs K + 1) (counterWidth K M) M) (countGates K M LIMIT).2 (inputs K + 1)
    (by rw [constWires_length, hcl]) hcb hcw (by omega) h1' (by omega)
  rw [constWires_vals h0' h1', h1', ripple_le _ _ (by rw [bitsLE_length, List.length_map, hcl]), hcv,
    bitsVal_bitsLE, Nat.mod_eq_of_lt hM2] at hv
  unfold verdictProg Prog.eval
  rw [List.map_cons, List.map_nil, wires_outs]
  exact congrArg (· :: []) hv

/-- **The verdict unit decides `contAccepts`**: on the openings of `K ≤ 2³²` rounds, the program's output is `1` iff the
memory challenger with tolerance `M` accepts the records whose verdicts the openings determine. -/
theorem verdictProg_accepts {K M LIMIT : ℕ} {os : List RoundOpening} (hK : os.length = K) (hK32 : K ≤ 2 ^ 32)
    (hfit : ∀ o ∈ os, o.Fits) {rs : List ContRecord} (hrs : rs.map (·.verdict) = verdicts LIMIT os) :
    (verdictProg K M LIMIT).eval (encode os) = [true] ↔ contAccepts (M : ℝ) rs := by
  rw [verdictProg_eval hK hK32 hfit, contAccepts, records_filter, hrs]
  simp

theorem encode_length {os : List RoundOpening} (hfit : ∀ o ∈ os, o.Fits) : (encode os).length = inputs os.length := by
  unfold encode inputs
  rw [List.length_append, length_flatMap_mem (·.row) os fun o h => (hfit o h).1,
    length_flatMap_mem (·.expected) os fun o h => (hfit o h).2]
  ring

/-- **A deployed verdict unit decides `contAccepts`**: any program that is the Definition's (`VerdictIs`). -/
theorem verdict_accepts {P : Claims.Sampling.Prog} {K M LIMIT : ℕ} (hP : VerdictIs P K M LIMIT)
    {os : List RoundOpening} (hK : os.length = K) (hK32 : K ≤ 2 ^ 32) (hfit : ∀ o ∈ os, o.Fits)
    {rs : List ContRecord} (hrs : rs.map (·.verdict) = verdicts LIMIT os) :
    P.eval (encode os) = [true] ↔ contAccepts (M : ℝ) rs := by
  rw [hP _ (by rw [encode_length hfit, hK])]
  exact verdictProg_accepts hK hK32 hfit hrs

end PousVerdict
