module

public import Proofs.Flock.SamplingGeneric.Deployed
public import Proofs.Flock.Claims.Sampling.Extract
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.BitsLin2
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.SamplingGeneric.Coins
import Proofs.Flock.SamplingGeneric.Session

/-!
# `Claims.Sampling` from the sampled compiler at a frozen sampled record

`ofDeploy` (`SamplingGeneric/Deployed.lean`) reads a frozen sampled record (`Claims.Sampling.Deploy`) as a `Compiled`
over `Arg.ofFlock` at its Flock record's deployed decoder. It meets the contract where the record meets `KS_of`'s
hypotheses (`sessions_ofDeploy`), and its errors are within `Claims.Sampling`'s (`κ_ofDeploy_le`, `γ_ofDeploy_le`), so
what `Sampling.KS` still needs from `Compiled.KS` is that the two extractors fail alike.

The two extractors read their coins differently: `Sampling.E` gives each slot of the sample its own `E^rep` coins over
all of the verifier's rounds, of which a session reads those on its rounds, and `Compiled.E` gives each slot one coin
vector per proof block's shape, of which the slot reads its block's. At fixed outer and inner coins, the slot's block
is fixed (`slotUnit`), and `combSlot` and `restSlot` exchange the coins `Compiled.E` reads at the slot for those
`Sampling.E` reads on the block's rounds, a bijection, so both extractors' laws are the same (`Ex_couple`). At coupled
coins the two extractions return the same and make the same SHA-512 calls of their own (`Agree`, `E_agree`), so the
failure event and `B_ext` have the same probability (`prBad_eq`, `bext_eq`), and `FlockKS` gives `Sampling.KS`'s bound
at the deployed decoder (`ksLin_of_flockKS`). `Sampling.KS` reads every decoder within Flock's radius, each the deployed
decoder of the record that differs from `D` only there (`withDecode`), which its statement and hypotheses read as they
read `D` (`samplingKS_of_flockKS`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (AuditStatement Opening forkOn flatMapRP waitAccC ownLog finRange_map_cast)
open Claims.Recursion (serve viewing)

/-! ## Lists -/

theorem zip_finRange_take {α : Type} (n : ℕ) (l : List α) :
    (List.finRange n).zip (l.take n) = (List.finRange n).zip l := by
  refine List.ext_getElem? fun i => Option.ext fun z => ?_
  rw [List.getElem?_zip_eq_some, List.getElem?_zip_eq_some, List.getElem?_take]
  by_cases hi : i < n
  · rw [ite_eq_left hi]
  · have h0 : (List.finRange n)[i]? = none := List.getElem?_eq_none (by simpa using Nat.le_of_not_lt hi)
    rw [ite_eq_right hi, h0]
    simp

theorem getElem?_of_mem_zip_finRange {β : Type} {k : ℕ} {l : List β} {j : Fin k} {a : β}
    (h : (j, a) ∈ (List.finRange k).zip l) : l[j.val]? = some a := by
  obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 h
  obtain ⟨h1, h2⟩ := List.getElem?_zip_eq_some.1 hi
  obtain ⟨hlt, hj⟩ := List.getElem?_eq_some_iff.1 h1
  have hj' : (List.finRange k)[i] = j := hj
  have hji : j.val = i := by rw [← hj']; simp
  rw [hji]
  exact h2

theorem mem_waitAccC {R : Type} {H : Msg → Msg} {run : R → Trace} {k : ℕ} {rs : List R} {r : R} {v : Seen}
    (h : (r, v) ∈ ((waitAccC k rs []).eval H run).1) : (run r).out = .acc ∧ v = (run r).seen := by
  rw [Claims.Sampling.waitAccC_eval_nil] at h
  obtain ⟨r', hr', he⟩ := List.mem_map.1 h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
  refine ⟨?_, rfl⟩
  unfold Claims.Sampling.accList at hr'
  simpa using (List.mem_filter.1 (List.mem_of_mem_take hr')).2

theorem orderEmbOfFin_cast {α : Type} [LinearOrder α] (s : Finset α) {k : ℕ} (h : s.card = k) (t : Fin k) :
    s.orderEmbOfFin h t = s.orderEmbOfFin rfl (Fin.cast h.symm t) := by
  subst h
  rfl

/-! ## Rerun programs that return the same and make the same calls -/

/-- Two rerun programs, against the same runs, return the same and make the same SHA-512 calls of their own. -/
structure Agree {R α : Type} (H : Msg → Msg) (run : R → Trace) (p q : RP R α) : Prop where
  eval : (p.eval H run).1 = (q.eval H run).1
  log : ownLog H run p = ownLog H run q

namespace Agree

variable {R α β : Type} {H : Msg → Msg} {run : R → Trace}

theorem refl (p : RP R α) : Agree H run p p := ⟨rfl, rfl⟩

theorem bind {p q : RP R α} {f g : α → RP R β} (h : Agree H run p q)
    (hf : Agree H run (f (p.eval H run).1) (g (p.eval H run).1)) : Agree H run (p.bind f) (q.bind g) :=
  ⟨by rw [(RP.eval_bind H run p f).1, (RP.eval_bind H run q g).1, ← h.eval]; exact hf.eval,
   by rw [Claims.Sampling.ownLog_bind, Claims.Sampling.ownLog_bind, ← h.eval, h.log, hf.log]⟩

theorem bind' {p q : RP R α} {f g : α → RP R β} (h : Agree H run p q) (hf : ∀ a, Agree H run (f a) (g a)) :
    Agree H run (p.bind f) (q.bind g) :=
  h.bind (hf _)

theorem rerun {r : R} {k k' : Seen → RP R α} (h : ∀ v, Agree H run (k v) (k' v)) :
    Agree H run (.rerun r k) (.rerun r k') :=
  ⟨(h (run r).seen).eval, (h (run r).seen).log⟩

theorem flatMap {γ : Type} {f g : β → RP R (List γ)} :
    ∀ {l : List β}, (∀ b ∈ l, Agree H run (f b) (g b)) → Agree H run (flatMapRP f l) (flatMapRP g l)
  | [], _ => refl _
  | b :: bs, h => (h b (List.mem_cons_self ..)).bind'
      fun _ => (flatMap (l := bs) fun b' hb' => h b' (List.mem_cons_of_mem _ hb')).bind' fun _ => refl _

theorem of_noHash {p q : RP R α} (hp : Claims.Sampling.NoHash p) (hq : Claims.Sampling.NoHash q)
    (h : (p.eval H run).1 = (q.eval H run).1) : Agree H run p q :=
  ⟨h, by rw [Claims.Sampling.ownLog_of_noHash H run hp, Claims.Sampling.ownLog_of_noHash H run hq]⟩

theorem bind_left {p : RP R α} {f : α → RP R β} {a : α} (hp : Claims.Sampling.NoHash p)
    (h : (p.eval H run).1 = a) : Agree H run (p.bind f) (f a) :=
  ⟨by rw [(RP.eval_bind H run p f).1, h],
   by rw [Claims.Sampling.ownLog_bind, Claims.Sampling.ownLog_of_noHash H run hp, h, List.nil_append]⟩

theorem of_eq_right {p q q' : RP R α} (e : q = q') (h : Agree H run p q') : Agree H run p q := e ▸ h

theorem ite_pos {c : Prop} [Decidable c] {p p' q q' : RP R α} (hc : c) (h : Agree H run p q) :
    Agree H run (if c then p else p') (if c then q else q') := by
  rw [ite_eq_left hc, ite_eq_left hc]
  exact h

theorem ite_neg {c : Prop} [Decidable c] {p p' q q' : RP R α} (hc : ¬ c) (h : Agree H run p' q') :
    Agree H run (if c then p else p') (if c then q else q') := by
  rw [ite_eq_right hc, ite_eq_right hc]
  exact h

end Agree

/-! ## Exchanging a slot's coins -/

section Couple

variable {Coin : Type} {n W C K : ℕ} (S : Fin n → Finset (Fin K))

theorem orderEmbOfFin_symm (s : Finset (Fin K)) (t : Fin K) (h : t ∈ s) :
    s.orderEmbOfFin rfl ((s.orderIsoOfFin rfl).symm ⟨t, h⟩) = t := by
  rw [← Finset.coe_orderIsoOfFin_apply, OrderIso.apply_symm_apply]

theorem symm_orderEmbOfFin (s : Finset (Fin K)) (t : Fin s.card) (h : s.orderEmbOfFin rfl t ∈ s) :
    (s.orderIsoOfFin rfl).symm ⟨s.orderEmbOfFin rfl t, h⟩ = t := by
  rw [OrderIso.symm_apply_eq]
  exact Subtype.ext (Finset.coe_orderIsoOfFin_apply _ _ _).symm

/-- `Sampling.E`'s coins at a slot whose block is `o`: the block's coins at the slot (`g`) on its rounds, `e`
elsewhere. -/
def combSlot : Option (Fin n) → ((v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin) →
    (Fin W → Fin C → Fin K → Coin) → Fin W → Fin C → Fin K → Coin
  | some v, g, e => fun i l t => if h : t ∈ S v then g v i l (((S v).orderIsoOfFin rfl).symm ⟨t, h⟩) else e i l t
  | none, _, e => e

/-- `Compiled.E`'s coins at a slot whose block is `o`: `e` on the block's rounds at the block, `g` at every other. -/
def restSlot (o : Option (Fin n)) (g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin)
    (e : Fin W → Fin C → Fin K → Coin) : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin :=
  fun v i l t => if o = some v then e i l ((S v).orderEmbOfFin rfl t) else g v i l t

variable {S}

theorem combSlot_mem {v : Fin n} {g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin}
    {e : Fin W → Fin C → Fin K → Coin} {i : Fin W} {l : Fin C} {t : Fin K} (h : t ∈ S v) :
    combSlot S (some v) g e i l t = g v i l (((S v).orderIsoOfFin rfl).symm ⟨t, h⟩) :=
  dite_eq_left h

theorem combSlot_not_mem {v : Fin n} {g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin}
    {e : Fin W → Fin C → Fin K → Coin} {i : Fin W} {l : Fin C} {t : Fin K} (h : t ∉ S v) :
    combSlot S (some v) g e i l t = e i l t :=
  dite_eq_right h

theorem restSlot_self {v : Fin n} {g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin}
    {e : Fin W → Fin C → Fin K → Coin} {i : Fin W} {l : Fin C} {t : Fin (S v).card} :
    restSlot S (some v) g e v i l t = e i l ((S v).orderEmbOfFin rfl t) :=
  ite_eq_left rfl

theorem restSlot_ne {o : Option (Fin n)} {v : Fin n} {g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin}
    {e : Fin W → Fin C → Fin K → Coin} {i : Fin W} {l : Fin C} {t : Fin (S v).card} (h : o ≠ some v) :
    restSlot S o g e v i l t = g v i l t :=
  ite_eq_right h

variable (S)

theorem combSlot_restSlot (o : Option (Fin n)) (g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin)
    (e : Fin W → Fin C → Fin K → Coin) : combSlot S o (restSlot S o g e) (combSlot S o g e) = e := by
  rcases o with _ | v
  · rfl
  · funext i l t
    by_cases h : t ∈ S v
    · rw [combSlot_mem h, restSlot_self, orderEmbOfFin_symm]
    · rw [combSlot_not_mem h, combSlot_not_mem h]

theorem restSlot_restSlot (o : Option (Fin n)) (g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin)
    (e : Fin W → Fin C → Fin K → Coin) : restSlot S o (restSlot S o g e) (combSlot S o g e) = g := by
  funext v i l t
  by_cases ho : o = some v
  · subst ho
    rw [restSlot_self, combSlot_mem ((S v).orderEmbOfFin_mem rfl t), symm_orderEmbOfFin]
  · rw [restSlot_ne ho, restSlot_ne ho]

/-- Exchanging, at every slot `j` whose block is `sel j`, the coins on the block's rounds. -/
def combEquiv {J : Type} (sel : J → Option (Fin n)) :
    (J → (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin) × (J → Fin W → Fin C → Fin K → Coin) ≃
      (J → Fin W → Fin C → Fin K → Coin) × (J → (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin) where
  toFun p := (fun j => combSlot S (sel j) (p.1 j) (p.2 j), fun j => restSlot S (sel j) (p.1 j) (p.2 j))
  invFun q := (fun j => restSlot S (sel j) (q.2 j) (q.1 j), fun j => combSlot S (sel j) (q.2 j) (q.1 j))
  left_inv p := Prod.ext (funext fun j => restSlot_restSlot S (sel j) (p.1 j) (p.2 j))
    (funext fun j => combSlot_restSlot S (sel j) (p.1 j) (p.2 j))
  right_inv q := Prod.ext (funext fun j => combSlot_restSlot S (sel j) (q.2 j) (q.1 j))
    (funext fun j => restSlot_restSlot S (sel j) (q.2 j) (q.1 j))

set_option synthInstance.maxSize 1024 in
theorem Ex_couple [Fintype Coin] [Nonempty Coin] {J : Type} [Fintype J] [DecidableEq J] (sel : J → Option (Fin n))
    (F : (J → Fin W → Fin C → Fin K → Coin) → ℝ) :
    Ex F = Ex fun p : (J → (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin) × (J → Fin W → Fin C → Fin K → Coin) =>
      F fun j => combSlot S (sel j) (p.1 j) (p.2 j) :=
  (Claims.Sampling.Ex_fst (β := J → (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin) F).symm.trans
    (Claims.Sampling.Ex_equiv (combEquiv S sel) fun q => F q.1).symm

theorem combSlot_emb {v : Fin n} {o : Option (Fin n)} (h : o = some v)
    (g : (v : Fin n) → Fin W → Fin C → Fin (S v).card → Coin) (e : Fin W → Fin C → Fin K → Coin) (i : Fin W)
    (l : Fin C) (t : Fin (S v).card) : combSlot S o g e i l ((S v).orderEmbOfFin rfl t) = g v i l t := by
  subst h
  rw [combSlot_mem ((S v).orderEmbOfFin_mem rfl t), symm_orderEmbOfFin]

end Couple

/-! ## Flock's `E^rep` as `Arg.ofFlock`'s extractor -/

section Flock

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Inhabited Coin] [DecidableEq Col] in
theorem ofFlock_ext_cast (Dfl : Claims.Flock.Deploy Coin Col T) {xS : Msg} {W C K : ℕ} {N : ℝ}
    (h : (Arg.ofFlock Dfl).shape xS = (W, C, K, N)) (hw : Dfl.waits xS = W) (hc : Dfl.cap xS = C)
    (hk : Dfl.V.k xS = K) (g : Fin W → Fin C → Fin K → Coin) (r : Fin (Dfl.V.k xS) → Coin) :
    (Arg.ofFlock Dfl).ext xS r (cast (congrArg (Arg.ofFlock Dfl).EC h.symm) g) =
      (Claims.Flock.E Dfl Dfl.decode).ext xS r fun i j t => g (Fin.cast hw i) (Fin.cast hc j) (Fin.cast hk t) := by
  subst hw hc hk
  rfl

omit [Inhabited Coin] in
theorem noHash_ext (A : Arg Coin) (x : Msg) (r₀ : Fin (A.V.k x) → Coin) (c : A.EC (A.shape x)) :
    Claims.Sampling.NoHash (A.ext x r₀ c) := by
  refine Claims.Sampling.NoHash.rerun _ _ fun v => ?_
  split
  · exact Claims.Sampling.NoHash.firstWait _ _ _ _ _
  · exact Claims.Sampling.NoHash.ret _

omit [Inhabited Coin] [DecidableEq Col] in
theorem flockE_eval (Dfl : Claims.Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (H : Msg → Msg)
    {xS : Msg} (run' : (Fin (Dfl.V.k xS) → Coin) → Trace) (r₀ : Fin (Dfl.V.k xS) → Coin)
    (ρ : Fin (Dfl.waits xS) → Fin (Dfl.cap xS) → Fin (Dfl.V.k xS) → Coin) (ha : (run' r₀).seen.out = .acc) :
    (((Claims.Flock.E Dfl decode).ext xS r₀ ρ).eval H run').1 =
      ((firstWait (Dfl.k xS) (fun v => [Dfl.cols v])
        (Claims.Flock.assemble Dfl.parse Dfl.table decode Dfl.read xS) (Claims.Flock.check Dfl.parse xS)
        ((List.finRange (Dfl.waits xS)).map fun i => (List.finRange (Dfl.cap xS)).map (ρ i))).eval H run').1 := by
  show (RP.eval H run' (if (run' r₀).seen.out = .acc then _ else _)).1 = _
  rw [ite_eq_left ha]

end Flock

/-! ## The two extractors at a frozen sampled record -/

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- The proof block at slot `p` of the inner run `l` of the outer run `j`, at the waits' coins: none where the slot is
empty or its block is not below `n_PU`. Curried, so that `slotUnit_eq`'s hypotheses index the wait lists at the same
terms as the definition (an index reached through a tuple's projections sends the kernel into the wait lists). -/
def slotUnit (H : Msg → Msg) (D : Claims.Sampling.Deploy Coin Col T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) (j l : Fin (D.k x))
    (p : Fin (Claims.Sampling.nPU D x)) : Option (Fin (Claims.Sampling.nPU D x)) :=
  (((waitAccC (D.k x) ((List.finRange (D.cap x)).map c₁) []).eval H run').1[j.val]?).bind fun o =>
    (((waitAccC (D.k x) ((List.finRange (D.cap x)).map fun c => forkOn (D.stage2 x) o.1 (c₂ j c)) []).eval H
        run').1[l.val]?).bind fun i =>
      ((D.checked x i.1)[p.val]?).bind fun u =>
        if hu : u < Claims.Sampling.nPU D x then some ⟨u, hu⟩ else none

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem slotUnit_eq {H : Msg → Msg} {D : Claims.Sampling.Deploy Coin Col T} {x : Msg}
    {run' : (Fin (D.V.k x) → Coin) → Trace} {c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin}
    {c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin} {j l : Fin (D.k x)}
    {o i : (Fin (D.V.k x) → Coin) × Seen} {p : Fin (Claims.Sampling.nPU D x)} {u : ℕ}
    (hj : ((waitAccC (D.k x) ((List.finRange (D.cap x)).map c₁) []).eval H run').1[j.val]? = some o)
    (hl : ((waitAccC (D.k x) ((List.finRange (D.cap x)).map fun c => forkOn (D.stage2 x) o.1 (c₂ j c)) []).eval H
      run').1[l.val]? = some i)
    (hp : (D.checked x i.1)[p.val]? = some u) (hu : u < Claims.Sampling.nPU D x) :
    slotUnit H D x run' c₁ c₂ j l p = some ⟨u, hu⟩ := by
  rw [slotUnit, hj, Option.bind_some, hl, Option.bind_some, hp, Option.bind_some, dite_eq_left hu]

/-- Both extractors' common skeleton: the two waits, a run's extractions `ex` at each accepted inner run, and the
assembly. -/
noncomputable def skel (D : Claims.Sampling.Deploy Coin Col T) (x : Msg) (c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) (s : AuditStatement)
    (ex : Fin (D.k x) → Fin (D.k x) → (Fin (D.V.k x) → Coin) → Seen →
      RP (Fin (D.V.k x) → Coin) (List (List Msg × Opening Bool))) :
    RP (Fin (D.V.k x) → Coin) (Option Claims.Sampling.Wit) :=
  (waitAccC (D.k x) ((List.finRange (D.cap x)).map c₁) []).bind fun outer =>
    (flatMapRP (fun jo : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen) =>
        (waitAccC (D.k x) ((List.finRange (D.cap x)).map fun l => forkOn (D.stage2 x) jo.2.1 (c₂ jo.1 l))
            []).bind fun inner =>
          flatMapRP (fun li : Fin (D.k x) × ((Fin (D.V.k x) → Coin) × Seen) => ex jo.1 li.1 li.2.1 li.2.2)
            ((List.finRange (D.k x)).zip inner))
      ((List.finRange (D.k x)).zip outer)).bind fun found => .ret (some (Claims.Sampling.assemble s found))

omit [Inhabited Coin] [DecidableEq Col] in
theorem E_ext_D (D : Claims.Sampling.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (r₀ : Fin (D.V.k x) → Coin) (c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₃ : Fin (D.k x) × Fin (D.k x) × Fin (Claims.Sampling.nPU D x) → D.UnitCoins x) :
    (Claims.Sampling.E D decode).ext x r₀ (c₁, c₂, c₃) = .rerun r₀ fun v₀ =>
      match D.parse x with
      | none => .ret none
      | some s =>
        if v₀.out = .acc then
          skel D x c₁ c₂ s fun j l r v =>
            flatMapRP (fun iu : Fin (Claims.Sampling.nPU D x) × ℕ =>
                Claims.Sampling.unitExt D decode x s iu.2 r v (c₃ (j, l, iu.1)))
              ((List.finRange (Claims.Sampling.nPU D x)).zip (D.checked x r))
        else .ret none :=
  rfl

omit [DecidableEq Col] in
theorem E_ext_L (D : Claims.Sampling.Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) (x : Msg)
    (r₀ : Fin (D.V.k x) → Coin)
    (c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin) (c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₃ : Fin (D.k x) × Fin (D.k x) × Fin (Claims.Sampling.nPU D x) → (ofDeploy D iroots).BlockEC x) :
    (ofDeploy D iroots).E.ext x r₀ (c₁, c₂, c₃) = .rerun r₀ fun v₀ =>
      match D.parse x with
      | none => .ret none
      | some s =>
        if v₀.out = .acc then
          skel D x c₁ c₂ s fun j l r v => (ofDeploy D iroots).runExt x s r v fun p => c₃ (j, l, p)
        else .ret none :=
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem skel_agree (H : Msg → Msg) (D : Claims.Sampling.Deploy Coin Col T) (x : Msg)
    (run' : (Fin (D.V.k x) → Coin) → Trace) (c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin) (s : AuditStatement)
    (exD exL : Fin (D.k x) → Fin (D.k x) → (Fin (D.V.k x) → Coin) → Seen →
      RP (Fin (D.V.k x) → Coin) (List (List Msg × Opening Bool)))
    (h : ∀ jo ∈ (List.finRange (D.k x)).zip ((waitAccC (D.k x) ((List.finRange (D.cap x)).map c₁) []).eval H run').1,
      ∀ li ∈ (List.finRange (D.k x)).zip ((waitAccC (D.k x) ((List.finRange (D.cap x)).map fun l =>
          forkOn (D.stage2 x) jo.2.1 (c₂ jo.1 l)) []).eval H run').1,
        Agree H run' (exD jo.1 li.1 li.2.1 li.2.2) (exL jo.1 li.1 li.2.1 li.2.2)) :
    Agree H run' (skel D x c₁ c₂ s exD) (skel D x c₁ c₂ s exL) := by
  unfold skel
  refine (Agree.refl _).bind ?_
  refine (Agree.flatMap fun jo hjo => ?_).bind (Agree.refl _)
  refine (Agree.refl _).bind ?_
  exact Agree.flatMap fun li hli => h jo hjo li hli

omit [DecidableEq Col] in
theorem runExt_ofDeploy (D : Claims.Sampling.Deploy Coin Col T) (iroots : Msg → Seen → ℕ → List Msg) (x : Msg)
    (s : AuditStatement) (r : Fin (D.V.k x) → Coin) (v : Seen) (c : (ofDeploy D iroots).SlotEC x) :
    (ofDeploy D iroots).runExt x s r v c =
      flatMapRP (fun iu : Fin (Claims.Sampling.nPU D x) × ℕ =>
          if hu : iu.2 < Claims.Sampling.nPU D x then
            (ofDeploy D iroots).unitExt x s iu.2 r v (c iu.1 ⟨iu.2, hu⟩)
          else .ret [])
        ((List.finRange (Claims.Sampling.nPU D x)).zip (D.checked x r)) := by
  unfold Compiled.runExt
  exact congrArg (flatMapRP _) (zip_finRange_take _ _)

variable {H : Msg → Msg} {D : Claims.Sampling.Deploy Coin Col T} {iroots : Msg → Seen → ℕ → List Msg}

omit [DecidableEq Col] in
/-- **A sampled proof block's two extractions agree** at a block below `n_PU`, when the block's coins at the slot are
`Sampling.E`'s on its session's rounds. -/
theorem unitExt_agree (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D) (x : Msg)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (s : AuditStatement) (u : ℕ) (r : Fin (D.V.k x) → Coin)
    (hu : ∃ p : Fin (Claims.Sampling.nPU D x), (D.checked x r)[p.val]? = some u)
    (hr : (run H P D.V x ω r).out = .acc) (ρ : D.UnitCoins x)
    (g' : Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.rounds x u).card → Coin)
    (hρ : ∀ i j t, ρ i j ((D.rounds x u).orderEmbOfFin rfl t) = g' i j t) :
    Agree H (run H P D.V x ω) (Claims.Sampling.unitExt D D.flock.decode x s u r (run H P D.V x ω r).seen ρ)
      ((ofDeploy D iroots).unitExt x s u r (run H P D.V x ω r).seen g') := by
  obtain ⟨xS, emb, P', hmono, hrange, hacc, hview⟩ := hruns x Ω P ω r u hu
  have hS : ∀ t, emb t ∈ D.rounds x u := fun t => (hrange (emb t)).2 ⟨t, rfl⟩
  have hsess := (hview r fun _ _ _ => rfl).1
  obtain ⟨hw, hc, hN, -, -⟩ := hnum x u _ xS _ hsess
  have hcard := (ofDeploy D iroots).card_eq_of_strictMono (x := x) (u := u) (xS := xS) hmono hrange
  have hcard' : D.flock.V.k xS = (D.rounds x u).card := hcard
  have hco := (ofDeploy D iroots).coinsOf_eq_of_strictMono (x := x) (u := u) (xS := xS) hmono hrange
  have hk : D.flock.k xS = D.innerK x := by
    unfold Claims.Flock.Deploy.k Claims.Sampling.Deploy.innerK
    rw [hN]
  have hemb : ∀ t, emb t = (D.rounds x u).orderEmbOfFin rfl (Fin.cast hcard' t) := fun t =>
    (congrFun (Finset.orderEmbOfFin_unique hcard'.symm hS hmono) t).trans (orderEmbOfFin_cast _ hcard'.symm t)
  have hshape : (Arg.ofFlock D.flock).shape xS = (ofDeploy D iroots).sh x u := by
    show (D.flock.waits xS, D.flock.cap xS, D.flock.V.k xS, D.flock.N₀ xS) =
      (D.innerWaits x, D.innerCap x, (D.rounds x u).card, D.innerN₀ x)
    rw [hw, hc, hcard', hN]
  have hsessL : (ofDeploy D iroots).session x u (run H P D.V x ω r).seen = _ := hsess
  have ha' : (run H P' D.flock.V xS () fun i => r (emb i)).seen.out = .acc := (hacc r fun _ _ => rfl).1 hr
  have hviewL : ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ t' ∈ D.rounds x u, t < t') → r' t = r t) →
      (ofDeploy D iroots).session x u (run H P (ofDeploy D iroots).V x ω r').seen =
        some (xS, (run H P' (Arg.ofFlock D.flock).V xS ()
          ((ofDeploy D iroots).coinsOf x u xS r')).seen) := by
    intro r' hr'
    rw [hco r']
    exact (hview r' fun t ht hlt => hr' t ht fun t' ht' => by obtain ⟨i, rfl⟩ := (hrange t').1 ht'; exact hlt i).1
  unfold Claims.Sampling.unitExt Compiled.unitExt
  rw [hsess, hsessL]
  dsimp only
  rw [dite_eq_left hshape]
  refine Agree.bind' (Agree.of_noHash (Claims.Sampling.NoHash.firstWait _ _ _ _ _)
    (Claims.Recursion.noHash_serve (fun _ => Claims.Recursion.noHash_viewing _ _)
      (noHash_ext _ xS _ _)) ?_) fun o => ?_
  · let g : (Fin (D.V.k x) → Coin) → Fin (D.flock.V.k xS) → Coin := fun R i => R (emb i)
    let Ls : List (List (Fin (D.V.k x) → Coin)) := (List.finRange (D.innerWaits x)).map fun i =>
      (List.finRange (D.innerCap x)).map fun j => forkOn (D.rounds x u) r (ρ i j)
    have hcorr : ∀ rs ∈ Ls, ∀ r' ∈ rs,
        ((run H P D.V x ω r').seen.out = .acc ↔ (run H P' D.flock.V xS () (g r')).seen.out = .acc) ∧
          ((run H P D.V x ω r').seen.out = .acc →
            D.sessionCols x u (run H P D.V x ω r').seen = [D.flock.cols (run H P' D.flock.V xS () (g r')).seen]) := by
      intro rs hrs r' hr'
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 hrs
      obtain ⟨j, -, rfl⟩ := List.mem_map.1 hr'
      have hag : ∀ t ∉ D.rounds x u, forkOn (D.rounds x u) r (ρ i j) t = r t := fun t ht => by simp [forkOn, ht]
      refine ⟨hacc _ hag, fun _ => ?_⟩
      unfold Claims.Sampling.Deploy.sessionCols
      rw [(hview _ fun t ht _ => hag t ht).1]
    have hL : ((List.finRange (D.flock.waits xS)).map fun i => (List.finRange (D.flock.cap xS)).map
        fun j t => g' (Fin.cast hw i) (Fin.cast hc j) (Fin.cast hcard' t)) = Ls.map (List.map g) := by
      refine (finRange_map_cast hw (fun i' => (List.finRange (D.flock.cap xS)).map
        fun j t => g' i' (Fin.cast hc j) (Fin.cast hcard' t))).trans ?_
      rw [List.map_map]
      refine List.map_congr_left fun i _ => ?_
      refine (finRange_map_cast hc (fun j' t => g' i j' (Fin.cast hcard' t))).trans ?_
      simp only [Function.comp_apply, List.map_map]
      refine List.map_congr_left fun j _ => ?_
      funext t
      show g' i j (Fin.cast hcard' t) = forkOn (D.rounds x u) r (ρ i j) (emb t)
      rw [show forkOn (D.rounds x u) r (ρ i j) (emb t) = ρ i j (emb t) from ite_eq_left (hS t), hemb t, hρ]
    have hfw := Claims.Sampling.firstWait_map H (run H P D.V x ω) (run H P' D.flock.V xS ()) g
      (D.sessionCols x u) (fun v => [D.flock.cols v]) (D.innerK x)
      (Claims.Flock.assemble D.flock.parse D.flock.table D.flock.decode D.flock.read xS)
      (Claims.Flock.check D.flock.parse xS) Ls hcorr
    have e3 : (((Claims.Flock.E D.flock D.flock.decode).ext xS (fun i => r (emb i)) fun i j t =>
        g' (Fin.cast hw i) (Fin.cast hc j) (Fin.cast hcard' t)).eval H (run H P' D.flock.V xS ())).1 =
        ((firstWait (D.innerK x) (fun v => [D.flock.cols v])
          (Claims.Flock.assemble D.flock.parse D.flock.table D.flock.decode D.flock.read xS)
          (Claims.Flock.check D.flock.parse xS) (Ls.map (List.map g))).eval H (run H P' D.flock.V xS ())).1 := by
      rw [flockE_eval D.flock D.flock.decode H (run H P' D.flock.V xS ()) (fun i => r (emb i)) _ ha', hk, hL]
    have e4 : (((Arg.ofFlock D.flock).ext xS ((ofDeploy D iroots).coinsOf x u xS r)
        (cast (congrArg (Arg.ofFlock D.flock).EC hshape.symm) g')).eval H (run H P' D.flock.V xS ())).1 =
        (((Claims.Flock.E D.flock D.flock.decode).ext xS (fun i => r (emb i)) fun i j t =>
          g' (Fin.cast hw i) (Fin.cast hc j) (Fin.cast hcard' t)).eval H (run H P' D.flock.V xS ())).1 := by
      rw [hco r]
      exact congrArg (fun p => (RP.eval H (run H P' D.flock.V xS ()) p).1)
        (ofFlock_ext_cast D.flock (W := D.innerWaits x) (C := D.innerCap x) (K := (D.rounds x u).card)
          hshape hw hc hcard' g' _)
    refine Eq.trans ?_ ((ofDeploy D iroots).served_eval H P ω P' r r hcard hviewL (fun _ _ => rfl) _).symm
    exact hfw.trans (e3.symm.trans e4.symm)
  · rcases o with _ | w <;> exact Agree.refl _

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] in
/-- **A sampled proof block not below `n_PU` is not extracted**: on a statement in scope its session's witness would
open a block of the partition (`SessionOpens`), whose blocks are `n_PU`. -/
theorem unitExt_big (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots)
    (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω) (s : AuditStatement) (u : ℕ) (r : Fin (D.V.k x) → Coin)
    (hu : ∃ p : Fin (Claims.Sampling.nPU D x), (D.checked x r)[p.val]? = some u)
    (hr : (run H P D.V x ω r).out = .acc) (hbig : ¬ u < Claims.Sampling.nPU D x) (ρ : D.UnitCoins x) :
    Agree H (run H P D.V x ω) (Claims.Sampling.unitExt D D.flock.decode x s u r (run H P D.V x ω r).seen ρ)
      (.ret []) := by
  unfold Claims.Sampling.unitExt
  rcases hsess : D.session x u (run H P D.V x ω r).seen with _ | ⟨xS, vS⟩
  · exact Agree.refl _
  · dsimp only
    refine Agree.bind_left (a := none) (Claims.Sampling.NoHash.firstWait _ _ _ _ _)
      (Option.eq_none_iff_forall_ne_some.2 fun wF hwF => hbig ?_)
    have hchk := Claims.firstWait_some H _ _ _ _ _ _ wF hwF
    obtain ⟨s', Pt, hs', hPt, hin⟩ := hrefuse x Ω P ω r hr
    obtain ⟨b, hb, -⟩ := hopens x s' Pt Ω P ω r u xS vS wF hs' hPt hin hr hu hsess hchk
    rw [Claims.Sampling.nPU_eq hs' hPt]
    exact (List.getElem?_eq_some_iff.1 hb).1

omit [DecidableEq Col] in
/-- **One slot's two extractions agree**, when `Sampling.E`'s coins at the slot are `g` on the rounds of the slot's
block `o` and `e` elsewhere. -/
theorem E_agree_unit (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (s : AuditStatement) (r : Fin (D.V.k x) → Coin) (hr : (run H P D.V x ω r).out = .acc)
    (p : Fin (Claims.Sampling.nPU D x)) (u : ℕ) (hpu : (D.checked x r)[p.val]? = some u)
    (o : Option (Fin (Claims.Sampling.nPU D x))) (ho : ∀ hu : u < Claims.Sampling.nPU D x, o = some ⟨u, hu⟩)
    (g : (ofDeploy D iroots).BlockEC x) (e : D.UnitCoins x) :
    Agree H (run H P D.V x ω)
      (Claims.Sampling.unitExt D D.flock.decode x s u r (run H P D.V x ω r).seen
        (combSlot (fun v : Fin (Claims.Sampling.nPU D x) => D.rounds x v) o g e))
      (if hu : u < Claims.Sampling.nPU D x then
        (ofDeploy D iroots).unitExt x s u r (run H P D.V x ω r).seen (g ⟨u, hu⟩) else .ret []) := by
  by_cases hu : u < Claims.Sampling.nPU D x
  · refine Agree.of_eq_right (dite_eq_left hu) ?_
    exact unitExt_agree hruns hnum x P ω s u r ⟨p, hpu⟩ hr
      (combSlot (fun v : Fin (Claims.Sampling.nPU D x) => D.rounds x v) o g e) (g ⟨u, hu⟩) fun i l t =>
        combSlot_emb (fun v : Fin (Claims.Sampling.nPU D x) => D.rounds x v) (ho hu) g e i l t
  · refine Agree.of_eq_right (dite_eq_right hu) ?_
    exact unitExt_big hrefuse hopens x P ω s u r ⟨p, hpu⟩ hr hu _

omit [DecidableEq Col] in
/-- **One accepted inner run's two extractions agree**, when `Sampling.E`'s coins at each slot `p` are `g p` on the
rounds of the slot's block `sel p` and `e p` elsewhere. -/
theorem E_agree_run (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (s : AuditStatement) (r : Fin (D.V.k x) → Coin) (hr : (run H P D.V x ω r).out = .acc)
    (sel : Fin (Claims.Sampling.nPU D x) → Option (Fin (Claims.Sampling.nPU D x)))
    (hsel : ∀ (p : Fin (Claims.Sampling.nPU D x)) (u : ℕ), (D.checked x r)[p.val]? = some u →
      ∀ hu : u < Claims.Sampling.nPU D x, sel p = some ⟨u, hu⟩)
    (g : (ofDeploy D iroots).SlotEC x) (e : Fin (Claims.Sampling.nPU D x) → D.UnitCoins x) :
    Agree H (run H P D.V x ω)
      (flatMapRP (fun iu : Fin (Claims.Sampling.nPU D x) × ℕ =>
          Claims.Sampling.unitExt D D.flock.decode x s iu.2 r (run H P D.V x ω r).seen
            (combSlot (fun v : Fin (Claims.Sampling.nPU D x) => D.rounds x v) (sel iu.1) (g iu.1) (e iu.1)))
        ((List.finRange (Claims.Sampling.nPU D x)).zip (D.checked x r)))
      ((ofDeploy D iroots).runExt x s r (run H P D.V x ω r).seen g) := by
  refine Agree.of_eq_right (runExt_ofDeploy D iroots x s r _ _) ?_
  refine Agree.flatMap fun pu hpu => ?_
  obtain ⟨p, u⟩ := pu
  have hpu' := getElem?_of_mem_zip_finRange hpu
  exact E_agree_unit hruns hnum hrefuse hopens x P ω s r hr p u hpu' (sel p) (hsel p u hpu') (g p) (e p)

omit [DecidableEq Col] in
/-- **The two extractors agree at coupled coins**: `Compiled.E` at `g`, and `Sampling.E` at `g` on the rounds of each
slot's block and `e` elsewhere, where `sel` gives each slot's block at the waits' coins (`slotUnit_eq`). -/
theorem E_agree (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c₁ : Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (c₂ : Fin (D.k x) → Fin (D.cap x) → Fin (D.V.k x) → Coin)
    (sel : Fin (D.k x) × Fin (D.k x) × Fin (Claims.Sampling.nPU D x) → Option (Fin (Claims.Sampling.nPU D x)))
    (hsel : ∀ (j : Fin (D.k x)) (o : (Fin (D.V.k x) → Coin) × Seen) (l : Fin (D.k x))
      (i : (Fin (D.V.k x) → Coin) × Seen) (p : Fin (Claims.Sampling.nPU D x)) (u : ℕ),
      ((waitAccC (D.k x) ((List.finRange (D.cap x)).map c₁) []).eval H (run H P D.V x ω)).1[j.val]? = some o →
      ((waitAccC (D.k x) ((List.finRange (D.cap x)).map fun c => forkOn (D.stage2 x) o.1 (c₂ j c)) []).eval H
        (run H P D.V x ω)).1[l.val]? = some i →
      (D.checked x i.1)[p.val]? = some u → ∀ hu : u < Claims.Sampling.nPU D x, sel (j, l, p) = some ⟨u, hu⟩)
    (g : Fin (D.k x) × Fin (D.k x) × Fin (Claims.Sampling.nPU D x) → (ofDeploy D iroots).BlockEC x)
    (e : Fin (D.k x) × Fin (D.k x) × Fin (Claims.Sampling.nPU D x) → D.UnitCoins x) :
    Agree H (run H P D.V x ω)
      ((Claims.Sampling.E D D.flock.decode).ext x r₀ (c₁, c₂, fun j =>
        combSlot (fun v : Fin (Claims.Sampling.nPU D x) => D.rounds x v) (sel j) (g j) (e j)))
      ((ofDeploy D iroots).E.ext x r₀ (c₁, c₂, g)) := by
  rw [E_ext_D, E_ext_L]
  refine Agree.rerun fun v₀ => ?_
  rcases hps : D.parse x with _ | s
  · exact Agree.refl _
  · dsimp only
    by_cases hv : v₀.out = .acc
    · refine Agree.ite_pos hv (skel_agree H D x _ c₁ c₂ s _ _ ?_)
      rintro ⟨j, o⟩ hjo ⟨l, r, v⟩ hli
      dsimp only at hjo hli ⊢
      obtain ⟨hr, rfl⟩ := mem_waitAccC (List.of_mem_zip hli).2
      exact E_agree_run hruns hnum hrefuse hopens x P ω s r hr (fun p => sel (j, l, p))
        (fun p u hpu hu => hsel j o l (r, (run H P D.V x ω r).seen) p u (getElem?_of_mem_zip_finRange hjo)
          (getElem?_of_mem_zip_finRange hli) hpu hu)
        (fun p => g (j, l, p)) (fun p => e (j, l, p))
    · exact Agree.ite_neg hv (Agree.refl _)

set_option synthInstance.maxSize 1024 in
omit [DecidableEq Col] in
theorem Ex_ext_eq (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (G : Option Claims.Sampling.Wit → List (Msg × Msg) → ℝ) :
    Ex (fun c : (Claims.Sampling.E D D.flock.decode).C x =>
        G (((Claims.Sampling.E D D.flock.decode).ext x r₀ c).eval H (run H P D.V x ω)).1
          (ownLog H (run H P D.V x ω) ((Claims.Sampling.E D D.flock.decode).ext x r₀ c))) =
      Ex (fun c : (ofDeploy D iroots).E.C x =>
        G (((ofDeploy D iroots).E.ext x r₀ c).eval H (run H P D.V x ω)).1
          (ownLog H (run H P D.V x ω) ((ofDeploy D iroots).E.ext x r₀ c))) := by
  refine (Claims.Ex_prod _).trans (Eq.trans ?_ (Claims.Ex_prod _).symm)
  refine congrArg Ex (funext fun c₁ => ?_)
  refine (Claims.Ex_prod _).trans (Eq.trans ?_ (Claims.Ex_prod _).symm)
  refine congrArg Ex (funext fun c₂ => ?_)
  refine (Ex_couple (fun v : Fin (Claims.Sampling.nPU D x) => D.rounds x v)
    (fun j => slotUnit H D x (run H P D.V x ω) c₁ c₂ j.1 j.2.1 j.2.2) _).trans ?_
  refine Eq.trans ?_ (Claims.Sampling.Ex_fst
    (β := Fin (D.k x) × Fin (D.k x) × Fin (Claims.Sampling.nPU D x) → D.UnitCoins x) _)
  refine congrArg Ex (funext fun p => ?_)
  have h := E_agree hruns hnum hrefuse hopens x P ω r₀ c₁ c₂
    (fun j => slotUnit H D x (run H P D.V x ω) c₁ c₂ j.1 j.2.1 j.2.2)
    (fun j _ l _ p _ hj hl hp hu => slotUnit_eq (j := j) (l := l) (p := p) hj hl hp hu) p.1 p.2
  exact congrArg₂ G h.eval h.log

omit [DecidableEq Col] in
theorem Pr_ext_eq (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (Q : Option Claims.Sampling.Wit → List (Msg × Msg) → Prop) :
    Pr (fun c : (Claims.Sampling.E D D.flock.decode).C x =>
        Q (((Claims.Sampling.E D D.flock.decode).ext x r₀ c).eval H (run H P D.V x ω)).1
          (ownLog H (run H P D.V x ω) ((Claims.Sampling.E D D.flock.decode).ext x r₀ c))) =
      Pr (fun c : (ofDeploy D iroots).E.C x =>
        Q (((ofDeploy D iroots).E.ext x r₀ c).eval H (run H P D.V x ω)).1
          (ownLog H (run H P D.V x ω) ((ofDeploy D iroots).E.ext x r₀ c))) := by
  classical
  rw [← Claims.Ex_ind, ← Claims.Ex_ind]
  exact Ex_ext_eq hruns hnum hrefuse hopens x P ω r₀ fun w l => if Q w l then 1 else 0

omit [DecidableEq Col] in
/-- **The failure event has the same probability** under both extractors. -/
theorem prBad_eq (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (δ : ℝ) (x : Msg)
    {Ω : Type} [Fintype Ω] (P : Prover Ω) :
    PrBad H D.V (Claims.Sampling.R H D.commit D.parse D.partition δ) (Claims.Sampling.E D D.flock.decode) x P =
      PrBad H (ofDeploy D iroots).V
        (Claims.Sampling.R H (ofDeploy D iroots).commit (ofDeploy D iroots).parse
          (ofDeploy D iroots).partition δ) (ofDeploy D iroots).E x P := by
  unfold PrBad
  rw [Claims.Sampling.Pr_prod_eq_Ex, Claims.Sampling.Pr_prod_eq_Ex]
  refine congrArg Ex (funext fun ω => ?_)
  rw [Claims.Sampling.Pr_prod_eq_Ex, Claims.Sampling.Pr_prod_eq_Ex]
  refine congrArg Ex (funext fun r₀ => ?_)
  exact Pr_ext_eq hruns hnum hrefuse hopens x P ω r₀ fun w _ =>
    (run H P D.V x ω r₀).out = .acc ∧ ¬ (Claims.Sampling.R H D.commit D.parse D.partition δ).ext x w

omit [Nonempty Coin] [DecidableEq Col] in
theorem bext_Pr (H : Msg → Msg) {W : Type} (V : Verifier Coin) (E : Extractor Coin W V) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) :
    (Claims.Sampling.Bext H V E Ω P x).Pr H =
      Pr fun s : BCoins Ω Coin (V.k x) × E.C x × E.C x =>
        ∃ u v, findColl (ownLog H (run H P V x s.1.2.1) (E.ext x s.1.2.2.1 s.2.1) ++
          ownLog H (run H P V x s.1.2.1) (E.ext x (forkCoins s.1.1 s.1.2.2.1 s.1.2.2.2) s.2.2)) = some (u, v) ∧
            Collision H u v := by
  dsimp only [Claims.Sampling.Bext]
  rw [dite_eq_left hk]
  rfl

omit [DecidableEq Col] in
/-- **`B_ext` has the same probability** under both extractors. -/
theorem bext_eq (hruns : Claims.Sampling.SessionRuns H D) (hnum : Claims.Sampling.InnerNumbers D)
    (hrefuse : Claims.Sampling.Refuses H D) (hopens : Claims.Sampling.SessionOpens H D iroots) (x : Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    (Claims.Sampling.Bext H D.V (Claims.Sampling.E D D.flock.decode) Ω P x).Pr H =
      (Claims.Sampling.Bext H (ofDeploy D iroots).V (ofDeploy D iroots).E Ω P x).Pr H := by
  by_cases hk : 0 < D.V.k x
  · have hk' : 0 < (ofDeploy D iroots).V.k x := hk
    rw [bext_Pr H D.V _ P x hk, bext_Pr H (ofDeploy D iroots).V _ P x hk', Claims.Sampling.Pr_prod_eq_Ex,
      Claims.Sampling.Pr_prod_eq_Ex]
    refine congrArg Ex (funext fun b => ?_)
    rw [Claims.Sampling.Pr_prod_eq_Ex, Claims.Sampling.Pr_prod_eq_Ex]
    refine (Ex_ext_eq hruns hnum hrefuse hopens x P b.2.1 b.2.2.1 fun _ l =>
      Pr fun c' : (Claims.Sampling.E D D.flock.decode).C x =>
        ∃ u v, findColl (l ++ ownLog H (run H P D.V x b.2.1)
          ((Claims.Sampling.E D D.flock.decode).ext x (forkCoins b.1 b.2.2.1 b.2.2.2) c')) = some (u, v) ∧
            Collision H u v).trans ?_
    refine congrArg Ex (funext fun c => ?_)
    exact Pr_ext_eq hruns hnum hrefuse hopens x P b.2.1 (forkCoins b.1 b.2.2.1 b.2.2.2) fun _ l' =>
      ∃ u v, findColl (ownLog H (run H P D.V x b.2.1) ((ofDeploy D iroots).E.ext x b.2.2.1 c) ++ l') =
        some (u, v) ∧ Collision H u v
  · have hk' : ¬ 0 < (ofDeploy D iroots).V.k x := hk
    dsimp only [Claims.Sampling.Bext]
    rw [dite_eq_right hk, dite_eq_right hk']

/-- **`Sampling.KS`'s bound at the deployed decoder**, from the sampled compiler at Flock: `FlockKS` at `ofDeploy`,
whose contract is `KS_of`'s hypotheses (`sessions_ofDeploy`), at the same failure probability and `B_ext`, and errors
within `Claims.Sampling`'s. -/
theorem ksLin_of_flockKS [Zero Col] (hF : FlockKS (Coin := Coin) (Col := Col) (T := T))
    (iroots : Msg → Seen → ℕ → List Msg) (hKS : Claims.Flock.KS H D.flock)
    (hdec : Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ D.flock.decode) (hbind : D.commit.Binds H)
    (hopens : Claims.Sampling.SessionOpens H D iroots) (hfix : Claims.Sampling.InteriorFixed H D iroots)
    (hdraw : Claims.Sampling.DrawLaw D) (hrefuse : Claims.Sampling.Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x)
    (hdfix : Claims.Sampling.DrawFixed D) (hruns : Claims.Sampling.SessionRuns H D)
    (hnum : Claims.Sampling.InnerNumbers D) (δ : ℝ) (hδ : 0 ≤ δ) :
    Claims.Sampling.KSLinOn H D.V (Claims.Sampling.Scope D) (Claims.Sampling.R H D.commit D.parse D.partition δ)
      (Claims.Sampling.E D D.flock.decode) (Claims.Sampling.κ D δ) (Claims.Sampling.γ D) (fun _ => 1)
      (Claims.Sampling.Bext H D.V (Claims.Sampling.E D D.flock.decode)) := by
  intro x hx Ω _ _ P
  have hL := hF H D.flock (ofDeploy D iroots) hKS hdec hbind
    (sessions_ofDeploy D iroots H hdraw hrefuse hfix hdfix hruns hnum hopens) δ hδ x hx Ω P
  have hκ := κ_ofDeploy_le D iroots δ x (hε x)
  have hγ := γ_ofDeploy_le D iroots x
  have hB := Claims.Sampling.PrB_nonneg H D.V P x
  have h1 : Real.sqrt ((ofDeploy D iroots).γ x * PrB H D.V P x) ≤
      Real.sqrt (Claims.Sampling.γ D x * PrB H D.V P x) :=
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_right hγ hB)
  exact (prBad_eq (iroots := iroots) hruns hnum hrefuse hopens δ x P).trans_le
    (hL.trans (add_le_add (add_le_add hκ h1)
      (le_of_eq (congrArg (fun y => (1 : ℝ) * y) (bext_eq (iroots := iroots) hruns hnum hrefuse hopens x P).symm))))

/-- `D` with its Flock record's deployed decoder `decode`, counted as no steps. `Claims.Sampling`'s statements and
hypotheses read a decoder only as their argument (its `Efficient` aside), so each reads this record as it reads `D`. -/
def withDecode (D : Claims.Sampling.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    Claims.Sampling.Deploy Coin Col T :=
  { D with flock := { D.flock with dec := fun os => (decode os, 0) } }

/-- **`Sampling.KS` from the sampled compiler at Flock**: at each decoder within Flock's radius, `ksLin_of_flockKS` at
the record whose deployed decoder it is (`withDecode`). -/
theorem samplingKS_of_flockKS [Zero Col] (hF : FlockKS (Coin := Coin) (Col := Col) (T := T))
    (iroots : Msg → Seen → ℕ → List Msg) (hKS : Claims.Flock.KS H D.flock) (hbind : D.commit.Binds H)
    (hopens : Claims.Sampling.SessionOpens H D iroots) (hfix : Claims.Sampling.InteriorFixed H D iroots)
    (hdraw : Claims.Sampling.DrawLaw D) (hrefuse : Claims.Sampling.Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x)
    (hdfix : Claims.Sampling.DrawFixed D) (hruns : Claims.Sampling.SessionRuns H D)
    (hnum : Claims.Sampling.InnerNumbers D) : Claims.Sampling.KS H D := by
  intro δ decode hδ hdec
  exact ksLin_of_flockKS (D := withDecode D decode) hF iroots hKS hdec hbind hopens hfix hdraw hrefuse hε hdfix hruns
    hnum δ hδ

end Instance

end Claims.SamplingGeneric
