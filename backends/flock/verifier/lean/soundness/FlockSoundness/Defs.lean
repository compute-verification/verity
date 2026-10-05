import FlockSoundness.Model.Session
import FlockSoundness.Accounting.Numbers
import ArkLib.Data.CodingTheory.ProximityGap.CapacityBounds
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Algebra.CharP.Defs
import FlockSoundness.Protocol.Defs
import FlockSoundness.Game.Interleave
import FlockSoundness.Model.Statement

/-!
# Shared definitions of the soundness statement

The arithmetic facts the proof uses (`Arith.Correct`), the committed values at the level-0
radius (`Committed`), and the doomed start of a rep (`RepDoomed`).
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- Every `fast100` level fits. -/
theorem fast100_levels_fit : ∀ m s, fast100 m = some s → ∀ l ∈ s.levels, l.Fits := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  have key : ∀ m', 22 ≤ m' → m' ≤ 35 → ∀ s', fast100 m' = some s' →
      (s'.levels.all fun l => decide l.Fits) = true := by
    intro m' h1' h2' s' hs'
    interval_cases m' <;> (simp only [fast100, Option.some.injEq] at hs'; subst hs'; decide +kernel)
  intro l hl
  have := List.all_eq_true.1 (key m h1 h2 s h) l hl
  simpa using this

/-- The skip nodes (§9): 64 distinct points on which `lagS` are the Lagrange weights. -/
theorem Model.Arith.Correct.lagS_interp {A : Arith F K} (hA : A.Correct) :
    ∃ node : Fin 64 → F, Function.Injective node ∧
      ∀ p : Polynomial F, p.natDegree < 64 → ∀ ζ, ∑ s, A.lagS ζ s * p.eval (node s) = p.eval ζ := by
  obtain ⟨nS, _, hS, _, _, hlag, _, _⟩ := hA.nodes
  exact ⟨nS, hS, hlag⟩

/-- A rep starts doomed when no codeword near the committed table packs a witness that satisfies
the block R1CS and the pin and makes every opened region claim true. -/
def RepDoomed (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) : Prop :=
  ∀ M : Message F sch.k0 sch.level₀.logCols,
    Close sch.level₀.radius (asTable sch C) (encode A sch.level₀.logLen M) →
      let z := witnessOf A S.m sch.k0 sch.level₀.logCols M
      ¬ ((∀ o i, (S.A₀.mulVec (S.block z o)) i * (S.B₀.mulVec (S.block z o)) i = S.block z o i)
          ∧ (∀ o, S.block z o S.pin = 1) ∧ ∀ c ∈ extra, c.Holds z)

/-! ## The witness's A-side and B-side, and the pin's extension -/

/-- The pin column's multilinear extension over the blocks. The lincheck's constant-wire coin
`β` checks that it is 1 at `ρ_out` (§10, A8). -/
def pinMLE (S : Statement) (z : Witness S.m) (ρOut : Fin S.nbl → F) : F :=
  ∑ o : Fin (2 ^ S.nbl), eqAt ρOut o.val * ((S.block z o S.pin).val : F)

/-- `ρ_out`: the zerocheck's challenges for the block bits (address bits `k_log … m−1`). -/
def Model.ZcOut.ρOut (S : Statement) (zc : ZcOut F S.m) : Fin S.nbl → F := fun j =>
  if h : S.kLog - 6 + j.val < S.m - 6 then zc.ρ ⟨S.kLog - 6 + j.val, h⟩ else 0

/-! ## The codewords near the committed table -/

/-- Messages whose encodings are within the level-0 radius of the committed table. -/
def closeMsgs (A : Arith F K) (sch : Schedule) (C : Oracle F) :
    Set (Message F sch.k0 sch.level₀.logCols) :=
  {M | Close sch.level₀.radius (asTable sch C) (encode A sch.level₀.logLen M)}

/-- The packed table entry `f(y)` of a message (§12.1, §13.1: `t[c·2^k₀ + ℓ] = M[ℓ][c]`). -/
def packedAt (sch : Schedule) (M : Message F sch.k0 sch.level₀.logCols) (y : ℕ) : F :=
  if h : y / 2 ^ sch.k0 < 2 ^ sch.level₀.logCols then
    M ⟨y % 2 ^ sch.k0, Nat.mod_lt _ (Nat.two_pow_pos _)⟩ ⟨y / 2 ^ sch.k0, h⟩
  else 0

/-! ## Doomed states between the phases -/

/-- Doomed after the zerocheck. -/
def DZc (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) : Option (ZcOut F S.m) → Prop
  | none => True
  | some zc => ∀ M ∈ closeMsgs A sch C,
      ¬ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
          ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
          ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the lincheck. -/
def DLc (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcOut F S.m) : Option (Claim F S.m) → Prop
  | none => True
  | some ab => ∀ M ∈ closeMsgs A sch C,
      ¬ (ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the opening: no codeword near the committed table has the batched value. -/
def DOp (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F) :
    Option (OpenOut F) → Prop
  | none => True
  | some op => ∀ M ∈ closeMsgs A sch C,
      (∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y * op.b y) ≠ op.T

/-! ## The doomed states over any list of messages

The phases before Ligerito use the messages near the committed table (`closeMsgs`) only through
membership and the list's size, so each is proved over any set `W` of messages (`…_of_list`). The
doomed states above are these at `W = closeMsgs A sch C` (`repDoomed_iff`, `dZc_eq`, `dLc_eq`,
`dOp_eq`). The padded level 0 takes `W` to be its padded code's witnesses (`SoundnessPad`). -/

/-- A rep starts doomed over `W`: no message of `W` packs a witness that satisfies the block
R1CS and the pin and makes every opened region claim true. -/
def RepDoomedW (A : Arith F K) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m)) : Prop :=
  ∀ M ∈ W, let z := witnessOf A S.m sch.k0 sch.level₀.logCols M
    ¬ ((∀ o i, (S.A₀.mulVec (S.block z o)) i * (S.B₀.mulVec (S.block z o)) i = S.block z o i)
        ∧ (∀ o, S.block z o S.pin = 1) ∧ ∀ c ∈ extra, c.Holds z)

/-- Doomed after the zerocheck, over `W`. -/
def DZcW (A : Arith F K) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m)) :
    Option (ZcOut F S.m) → Prop
  | none => True
  | some zc => ∀ M ∈ W,
      ¬ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
          ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
          ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the lincheck, over `W`. -/
def DLcW (A : Arith F K) (S : Statement) (sch : Schedule)
    (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m))
    (zc : ZcOut F S.m) : Option (Claim F S.m) → Prop
  | none => True
  | some ab => ∀ M ∈ W,
      ¬ (ab.Holds (wit A S sch M) ∧ (cClaim A zc).Holds (wit A S sch M)
          ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))

/-- Doomed after the opening, over `W`: no message of `W` has the batched value. -/
def DOpW (S : Statement) (sch : Schedule) (W : Set (Message F sch.k0 sch.level₀.logCols)) :
    Option (OpenOut F) → Prop
  | none => True
  | some op => ∀ M ∈ W,
      (∑ y ∈ Finset.range (2 ^ (S.m - 7)), packedAt sch M y * op.b y) ≠ op.T

theorem repDoomed_iff (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) :
    RepDoomed A S sch C extra ↔ RepDoomedW A S sch (closeMsgs A sch C) extra := Iff.rfl

theorem dZc_eq (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (o : Option (ZcOut F S.m)) :
    DZc A S sch C extra o = DZcW A S sch (closeMsgs A sch C) extra o := by
  cases o <;> rfl

theorem dLc_eq (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcOut F S.m) (o : Option (Claim F S.m)) :
    DLc A S sch C extra zc o = DLcW A S sch (closeMsgs A sch C) extra zc o := by
  cases o <;> rfl

theorem dOp_eq (A : Arith F K) (S : Statement) (sch : Schedule) (C : Oracle F)
    (o : Option (OpenOut F)) : DOp A S sch C o = DOpW S sch (closeMsgs A sch C) o := by
  cases o <;> rfl

end FlockSoundness
