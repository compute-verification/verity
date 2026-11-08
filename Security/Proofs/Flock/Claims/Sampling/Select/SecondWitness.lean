module

public import Proofs.Flock.Claims.Sampling.Select.Second
public import Proofs.Flock.Claims.Sampling.Wait
public import Proofs.Flock.Claims.Prob
public import Batteries.Tactic.OpenPrivate
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Init.Data.Range.Basic
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Lean.Data.Json.Basic

/-!
# `SecondMisses` says something: the second stage accepts, and a stage that accepts everything breaks its bound

* `secondMisses_honest`: the deployed `Select.selectSecond` accepts one sampled replay block whose one cell holds two
  proof blocks, one sampled, on 82 zero bytes. The kernel evaluates it (`decide +kernel`) once the definitions it can't
  unfold are their
  twins: `Array.qsort` and `Flock.canon` with fuel (`qsort_fuel`, `canon_fuel`), `Flock.Draw.onWith`'s `repeat` with
  fuel (`onWith_fuel`) and `Flock.Draw.subset` with its map a function (`subset_fn`).
* `secondMisses_nontrivial`: at `acceptAllSecond`, which accepts everything and proves nothing, both proof blocks of
  the sampled replay block's one cell escape with probability 1, above `escCell 2 1 2 = 0`.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open Spot
open private Array.qpartition.loop Array.qsort.sort from Init.Data.Array.QSort.Basic
open private Spot.Draw.STRATUM_KEYS Spot.Draw.WORK_STRATUM_KEYS Spot.Draw.lawKeys
  Spot.Draw.stratum? from Spot.Draw

namespace Properties.Spot.SecurityProofs

open Lean (Json)
open Claims.Sampling

/-! ## `Array.qsort` with fuel

`qsort`'s recursion is well founded, which the kernel does not unfold; `qsortFuel` is it with each recursion's measure
as fuel, and `qsort_fuel` says they agree. -/

section Fuel

variable {α : Type}

def qpartLoopFuel {n : Nat} (lt : α → α → Bool) (lo hi : Nat) (hhi : hi < n) (pivot : α) :
    Nat → (as : Vector α n) → (i k : Nat) → lo ≤ i → i ≤ k → k ≤ hi → {m : Nat // lo ≤ m ∧ m ≤ hi} × Vector α n
  | 0, as, i, _, ilo, ik, w => (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi)
  | f + 1, as, i, k, ilo, ik, w =>
    if h : k < hi then
      if lt as[k] pivot then qpartLoopFuel lt lo hi hhi pivot f (as.swap i k (by omega) (by omega)) (i + 1) (k + 1)
        (by omega) (by omega) (by omega)
      else qpartLoopFuel lt lo hi hhi pivot f as i (k + 1) ilo (by omega) (by omega)
    else (⟨i, ilo, by omega⟩, as.swap i hi (by omega) hhi)

private theorem qpartLoop_fuel {n : Nat} (lt : α → α → Bool) (lo hi : Nat) (hhi : hi < n) (pivot : α) :
    ∀ (f : Nat) (as : Vector α n) (i k : Nat) (ilo : lo ≤ i) (ik : i ≤ k) (w : k ≤ hi), hi - k < f →
      Array.qpartition.loop lt lo hi hhi pivot as i k ilo ik w =
        qpartLoopFuel lt lo hi hhi pivot f as i k ilo ik w := by
  intro f
  induction f with
  | zero => intro _ _ _ _ _ _ h; omega
  | succ f ih =>
    intro as i k ilo ik w hf
    unfold Array.qpartition.loop
    simp only [qpartLoopFuel]
    split
    · split
      · exact ih _ _ _ _ _ _ (by omega)
      · exact ih _ _ _ _ _ _ (by omega)
    · rfl

def qpartitionFuel {n : Nat} (as : Vector α n) (lt : α → α → Bool) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) : {m : Nat // lo ≤ m ∧ m ≤ hi} × Vector α n :=
  let mid := (lo + hi) / 2
  let as := if lt as[mid] as[lo] then as.swap lo mid else as
  let as := if lt as[hi] as[lo] then as.swap lo hi else as
  let as := if lt as[mid] as[hi] then as.swap mid hi else as
  qpartLoopFuel lt lo hi hhi as[hi] (hi - lo + 1) as lo lo (Nat.le_refl _) (Nat.le_refl _) w

theorem qpartition_fuel {n : Nat} (as : Vector α n) (lt : α → α → Bool) (lo hi : Nat) (w : lo ≤ hi) (hlo : lo < n)
    (hhi : hi < n) : Array.qpartition as lt lo hi w hlo hhi = qpartitionFuel as lt lo hi w hlo hhi := by
  unfold Array.qpartition qpartitionFuel
  apply qpartLoop_fuel
  omega

def qsortSortFuel {n : Nat} (lt : α → α → Bool) : Nat → (as : Vector α n) → (lo hi : Nat) → lo ≤ hi → lo < n →
    hi < n → Vector α n
  | 0, as, _, _, _, _, _ => as
  | f + 1, as, lo, hi, w, hlo, hhi =>
    if h₁ : lo < hi then
      match qpartitionFuel as lt lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        if h₂ : mid ≥ hi then as'
        else qsortSortFuel lt f (qsortSortFuel lt f as' lo mid hmid.1 hlo (by omega)) (mid + 1) hi (by omega)
          (by omega) hhi
    else as

private theorem qsortSort_fuel {n : Nat} (lt : α → α → Bool) : ∀ (f : Nat) (as : Vector α n) (lo hi : Nat)
    (w : lo ≤ hi) (hlo : lo < n) (hhi : hi < n), hi - lo < f →
      Array.qsort.sort lt as lo hi w hlo hhi = qsortSortFuel lt f as lo hi w hlo hhi := by
  intro f
  induction f with
  | zero => intro _ _ _ _ _ _ h; omega
  | succ f ih =>
    intro as lo hi w hlo hhi hf
    rw [Array.qsort.sort]
    simp only [qsortSortFuel]
    split
    · rw [qpartition_fuel]
      match qpartitionFuel as lt lo hi w hlo hhi with
      | (⟨mid, hmid⟩, as') =>
        dsimp only
        split
        · rfl
        · rw [ih _ _ _ _ _ _ (by omega), ih _ _ _ _ _ _ (by omega)]
    · rfl

def qsortFuel (as : Array α) (lt : α → α → Bool) (lo hi : Nat) : Array α :=
  if h : as.size = 0 then as
  else
    let lo := min lo (as.size - 1)
    let hi := max lo (min hi (as.size - 1))
    qsortSortFuel lt (hi - lo + 1) as.toVector lo hi (by omega) (by omega) (by omega) |>.toArray

theorem qsort_fuel (as : Array α) (lt : α → α → Bool) (lo hi : Nat) : as.qsort lt lo hi = qsortFuel as lt lo hi := by
  unfold Array.qsort qsortFuel
  split
  · rfl
  · dsimp only
    congr 1
    apply qsortSort_fuel
    omega

end Fuel

/-! ## `Flock.canon` with fuel

`canon` is well founded too; `canonFuel f` is it unfolded `f` times, `canon` itself past that, so the two agree at every
fuel (`canon_fuel`). -/

def canonFuel : Nat → Json → Flock.V String
  | 0, j => Flock.canon j
  | _ + 1, Json.null => pure "null"
  | _ + 1, Json.bool b => pure (if b then "true" else "false")
  | _ + 1, Json.num n =>
    if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | _ + 1, Json.str s => pure (Flock.canonStr s)
  | f + 1, Json.arr a => do
    let parts ← a.attach.mapM fun x => canonFuel f x.1
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | f + 1, Json.obj kvs => do
    let vals ← kvs.toArray.attach.mapM fun p => do pure (p.1.1, ← canonFuel f p.1.2)
    let items := qsortFuel vals (fun a b => a.1 < b.1) 0 (vals.size - 1)
    pure ("{" ++ ",".intercalate (items.toList.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++ "}")

theorem canon_fuel : ∀ (f : Nat) (j : Json), Flock.canon j = canonFuel f j
  | 0, _ => rfl
  | f + 1, j => by
    cases j with
    | null => rw [Flock.canon]; rfl
    | bool b => rw [Flock.canon]; rfl
    | num n => rw [Flock.canon]; rfl
    | str s => rw [Flock.canon]; rfl
    | arr a => rw [Flock.canon]; simp only [canonFuel, canon_fuel f]
    | obj kvs => rw [Flock.canon]; simp only [canonFuel, canon_fuel f, qsort_fuel]

/-! ## `Flock.Draw.onWith` with fuel

`onWith`'s `repeat` is `Lean.Loop.forIn`, which only unfolds through `Lean.Loop.forIn_eq_of_monadTail`; `onWithFuel f`
is its first `f` rounds, the loop itself past them (`onWith_fuel`). -/

/-- `onWith` from the stream `src₀` already read. -/
def onWithFrom {m : Type → Type} [Monad m] (get : Nat → m ByteArray) (need : Nat)
    (run : ByteArray → Option (Array Nat)) (src₀ : ByteArray) : m (Option (Array Nat)) := do
  let mut src := src₀
  repeat
    if let some units := run src then return some units
    src := src ++ (← get need)
  return none

theorem onWith_from {m : Type → Type} [Monad m] (get : Nat → m ByteArray) (need : Nat)
    (run : ByteArray → Option (Array Nat)) :
    Flock.Draw.onWith get need run = get need >>= onWithFrom get need run := rfl

def onWithFuel {m : Type → Type} [Monad m] (get : Nat → m ByteArray) (need : Nat)
    (run : ByteArray → Option (Array Nat)) : Nat → ByteArray → m (Option (Array Nat))
  | 0, src => onWithFrom get need run src
  | f + 1, src => match run src with
    | some units => pure (some units)
    | none => do onWithFuel get need run f (src ++ (← get need))

theorem forIn_loop {m : Type → Type} [Monad m] {β : Type} (b : β) (g : Unit → β → m (ForInStep β)) :
    forIn Lean.Loop.mk b g = Lean.Loop.forIn Lean.Loop.mk b g := rfl

theorem onWithFrom_fuel {m : Type → Type} [Monad m] [LawfulMonad m] [Lean.Order.MonadTail m]
    (get : Nat → m ByteArray) (need : Nat) (run : ByteArray → Option (Array Nat)) :
    ∀ (f : Nat) (src : ByteArray), onWithFrom get need run src = onWithFuel get need run f src
  | 0, _ => rfl
  | f + 1, src => by
    simp only [onWithFuel, ← onWithFrom_fuel get need run f]
    unfold onWithFrom
    dsimp only
    simp only [forIn_loop]
    rw [Lean.Loop.forIn_eq_of_monadTail]
    cases run src with
    | some u => simp
    | none => simp

theorem onWith_fuel {m : Type → Type} [Monad m] [LawfulMonad m] [Lean.Order.MonadTail m]
    (get : Nat → m ByteArray) (need : Nat) (run : ByteArray → Option (Array Nat)) (f : Nat) :
    Flock.Draw.onWith get need run = get need >>= onWithFuel get need run f := by
  rw [onWith_from]
  congr 1
  funext src
  exact onWithFrom_fuel get need run f src

/-! ## `Flock.Draw.subset` without its map

`subset`'s Fisher–Yates swaps live in a `Std.HashMap`, whose bucket index reads the platform's word size, which the
kernel does not evaluate; `subsetFn` keeps the same swaps as a function, and `subset_fn` says they agree, by a
simulation of one loop by the other (`forIn_sim`). -/

/-- Two steps of two loops, both done or both continuing, in states related by `R`. -/
def stepSim {σ τ : Type} (R : σ → τ → Prop) : ForInStep σ → ForInStep τ → Prop
  | .done s, .done t => R s t
  | .yield s, .yield t => R s t
  | _, _ => False

theorem forIn_sim {α σ τ : Type} (R : σ → τ → Prop) (f : α → σ → Id (ForInStep σ)) (g : α → τ → Id (ForInStep τ))
    (hstep : ∀ a s t, R s t → stepSim R (f a s).run (g a t).run) :
    ∀ (l : List α) (s : σ) (t : τ), R s t → R (forIn l s f).run (forIn l t g).run
  | [], s, t, h => by simpa using h
  | a :: l, s, t, h => by
    have hs := hstep a s t h
    simp only [List.forIn_cons, Id.run_bind]
    generalize (f a s).run = x at hs ⊢
    generalize (g a t).run = y at hs ⊢
    cases x <;> cases y <;> simp only [stepSim] at hs
    · exact hs
    · exact forIn_sim R f g hstep l _ _ hs

theorem forIn_bind_sim {α σ τ β : Type} (R : σ → τ → Prop) (l : List α) (s₀ : σ) (t₀ : τ)
    (f : α → σ → Id (ForInStep σ)) (g : α → τ → Id (ForInStep τ)) (c₁ : σ → Id β) (c₂ : τ → Id β)
    (h₀ : R s₀ t₀) (hstep : ∀ a s t, R s t → stepSim R (f a s).run (g a t).run)
    (hc : ∀ s t, R s t → (c₁ s).run = (c₂ t).run) :
    (forIn l s₀ f >>= c₁).run = (forIn l t₀ g >>= c₂).run := by
  rw [Id.run_bind, Id.run_bind]
  exact hc _ _ (forIn_sim R f g hstep l s₀ t₀ h₀)

/-- `Flock.Draw.subset` with its swaps as a function. -/
def subsetFn (k n : Nat) (src : ByteArray) : Option (Array Nat) := Id.run do
  let mut sw : Nat → Nat := fun a => a
  let mut pos := 0
  for i in [0:k] do
    match Flock.Draw.uniform (n - i) src (src.size + 1) pos with
    | none => return none
    | some (r, p) =>
      let j := i + r
      let (vi, vj) := (sw i, sw j)
      sw := fun a => if j == a then vi else if i == a then vj else sw a
      pos := p
  let a := (Array.range k).map fun i => sw i
  return some (qsortFuel a (· < ·) 0 (a.size - 1))

theorem subset_fn : Flock.Draw.subset = subsetFn := by
  funext k n src
  unfold Flock.Draw.subset subsetFn
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', qsort_fuel]
  refine forIn_bind_sim (fun s t => s.1 = t.1 ∧ s.2.2 = t.2.2 ∧ ∀ a, s.2.1.getD a a = t.2.1 a) _ _ _ _ _ _ _
    ⟨rfl, rfl, fun _ => Std.HashMap.getD_empty⟩ ?_ ?_
  · rintro i ⟨r₁, sw, p⟩ ⟨r₂, g, _⟩ ⟨-, rfl, h3⟩
    dsimp only
    cases Flock.Draw.uniform (n - i) src (src.size + 1) p with
    | none => exact ⟨rfl, rfl, h3⟩
    | some rp =>
      refine ⟨rfl, rfl, fun a => ?_⟩
      simp only [Std.HashMap.getD_insert, h3]
  · rintro ⟨r₁, sw, p⟩ ⟨r₂, g, _⟩ ⟨rfl, rfl, h3⟩
    dsimp only
    cases r₁ with
    | some r => rfl
    | none => simp only [h3]

/-! ## The witnesses -/

/-- **`SecondMisses`'s honest lemma**: the deployed second stage accepts one sampled replay block whose one cell holds
two proof blocks, one sampled, on 82 zero bytes (its stratum's `need`). The kernel evaluates `selectSecond`
(`decide +kernel`) once its well-founded sorts and canonical JSON, its `repeat` and its map are their fuelled twins. -/
theorem secondMisses_honest : (Select.selectSecond #[0] #[2] #[1] ⟨Array.replicate 82 0⟩).isOk = true := by
  unfold Select.selectSecond Draw.drawFirstRead Draw.drawUnits Flock.Draw.drawWith Flock.Draw.stratifiedWith
    Draw.flockLaw Draw.strataProblem Flock.Draw.workProblem Draw.checkDraw Draw.checkDrawFull
    Draw.strataMessage Draw.sameLaw Draw.lawOf Spot.Draw.lawKeys Spot.Draw.stratum?
    Spot.Draw.STRATUM_KEYS Spot.Draw.WORK_STRATUM_KEYS keys? sortStrings jeq
  simp only [onWith_fuel _ _ _ 2, canon_fuel 16, qsort_fuel, subset_fn]
  decide +kernel

theorem proved_null : Draw.proved Json.null = #[] := by
  unfold Draw.proved
  simp only [qsort_fuel]
  decide +kernel

/-- **`SecondMisses`'s nontrivial lemma**: its bound fails at a second stage that accepts everything
(`acceptAllSecond`): the sample proves nothing, so both proof blocks of the one sampled replay block's one cell escape
whatever the bytes, with probability 1, and `escCell 2 1 2 = 0`. -/
theorem secondMisses_nontrivial : ¬ Claims.Pr (fun src : Fin 0 → Fin 256 =>
      (acceptAllSecond #[0] #[2] #[1] (bytesOf src)).isOk = true ∧
        ∀ j ∈ ({0, 1} : Finset ℕ), (j : Int) ∉ secondOf (acceptAllSecond #[0] #[2] #[1] (bytesOf src))) ≤
      ∏ t ∈ Finset.range (#[2] : Array ℕ).size, ∏ i ∈ Finset.range (#[0] : Array ℕ).size,
        escCell ((#[2] : Array ℕ).getD t 0) ((#[1] : Array ℕ).getD t 0)
          (({0, 1} : Finset ℕ).filter fun j =>
            cellOff (#[0] : Array ℕ).size #[2] t + i * (#[2] : Array ℕ).getD t 0 ≤ j ∧
              j < cellOff (#[0] : Array ℕ).size #[2] t + (i + 1) * (#[2] : Array ℕ).getD t 0).card := by
  have h1 : Claims.Pr (fun src : Fin 0 → Fin 256 =>
      (acceptAllSecond #[0] #[2] #[1] (bytesOf src)).isOk = true ∧
        ∀ j ∈ ({0, 1} : Finset ℕ), (j : Int) ∉ secondOf (acceptAllSecond #[0] #[2] #[1] (bytesOf src))) = 1 :=
    (Claims.Pr_congr fun _ => ⟨fun _ => trivial, fun _ => ⟨rfl, fun j _ => by
      simp [acceptAllSecond, secondOf, proved_null]⟩⟩).trans Pr_true
  have hc : (({0, 1} : Finset ℕ).filter fun j => cellOff 1 #[2] 0 + 0 * (#[2] : Array ℕ).getD 0 0 ≤ j ∧
      j < cellOff 1 #[2] 0 + (0 + 1) * (#[2] : Array ℕ).getD 0 0).card = 2 := by decide
  have he : escCell 2 1 2 = 0 := by
    have h0 : (2 - 2).choose (min 1 2) = 0 := by decide
    unfold escCell
    rw [h0, Nat.cast_zero, zero_div]
  rw [h1, show (#[2] : Array ℕ).size = 1 from rfl, show (#[0] : Array ℕ).size = 1 from rfl,
    Finset.prod_range_one, Finset.prod_range_one]
  change ¬ (1 : ℝ) ≤ escCell 2 1 _
  rw [hc, he]
  norm_num

end Properties.Spot.SecurityProofs
