import Mathlib
import Definitions.Accounting.RateAudit

/-!
# The rate audit's chip programs: congruence, replay from a footprint trace, and counting

`run_congr`: changing the oracle off the points a run queries changes neither the run nor its queries.
`replay_trace`: given the oracle, the run's queries are a function of the chip program and its trace, the bits device
memory (once per address, at its first read), the side link and rebuilding answered with; the trace's length is the
footprint plus the link's bits plus `b` per rebuilt word (`trace_length`). `card_addrs_le`: the footprint is at most the
ingress.
-/

namespace Accounting.Rate

open Pous (Bits pr)

namespace Chip

variable {N b h ρ : ℕ} {α : Type}

/-- The bits a run was answered with, other than the oracle's, with device memory read once per address: the bit at
an address's first read (`seen` holds the addresses read before), every bit from the side link, and every rebuilt
word. -/
def trace (w : World N b h ρ) : Chip N b h ρ α → Finset ℕ → List Bool
  | done _, _ => []
  | tag x k, seen => (k (w.oracle x)).trace w seen
  | ingress a k, seen =>
    if a ∈ seen then (k (w.mem a)).trace w seen else w.mem a :: (k (w.mem a)).trace w (insert a seen)
  | side a k, seen => w.side a :: (k (w.side a)).trace w seen
  | stream i k, seen => w.stream i :: (k (w.stream i)).trace w seen
  | rebuild j k, seen => List.ofFn (w.store j) ++ (k (w.store j)).trace w seen

/-- The queries a run makes, recomputed from the oracle and a stream of the bits it was answered with otherwise, with a
cache `m` of the device-memory bits read so far. -/
def replay (O : Oracle N b h ρ) : Chip N b h ρ α → (ℕ → Option Bool) → (ℕ → Bool) → List (Bits ρ × Fin N × Bits b)
  | done _, _, _ => []
  | tag x k, m, f => x :: (k (O x)).replay O m f
  | ingress a k, m, f =>
    match m a with
    | some v => (k v).replay O m f
    | none => (k (f 0)).replay O (Function.update m a (some (f 0))) (fun n => f (n + 1))
  | side _ k, m, f => (k (f 0)).replay O m (fun n => f (n + 1))
  | stream _ k, m, f => (k (f 0)).replay O m (fun n => f (n + 1))
  | rebuild _ k, m, f => (k (fun i => f i)).replay O m (fun n => f (n + b))

theorem trace_length (w : World N b h ρ) (c : Chip N b h ρ α) (seen : Finset ℕ) :
    (c.trace w seen).length = (c.addrs w \ seen).card + c.linkBits w + c.rebuilds w * b := by
  induction c generalizing seen with
  | done a => simp [trace, addrs, linkBits, rebuilds]
  | tag x k ih => simp [trace, addrs, linkBits, rebuilds, ih]
  | ingress a k ih =>
    by_cases ha : a ∈ seen
    · simp only [trace, addrs, linkBits, rebuilds, ha, ↓reduceIte, ih, Finset.insert_sdiff_of_mem _ ha]
    · simp only [trace, addrs, linkBits, rebuilds, ha, ↓reduceIte, List.length_cons, ih,
        Finset.insert_sdiff_of_notMem _ ha, Finset.sdiff_insert]
      have hc : (insert a ((k (w.mem a)).addrs w \ seen)).card = (((k (w.mem a)).addrs w \ seen).erase a).card + 1 := by
        rw [← Finset.card_erase_add_one (Finset.mem_insert_self a _), Finset.erase_insert_eq_erase]
      rw [hc]
      omega
  | side a k ih => simp [trace, addrs, linkBits, rebuilds, ih]; omega
  | stream i k ih => simp [trace, addrs, linkBits, rebuilds, ih]; omega
  | rebuild j k ih => simp [trace, addrs, linkBits, rebuilds, ih, Nat.succ_mul]; omega

theorem replay_trace (w : World N b h ρ) (c : Chip N b h ρ α) (seen : Finset ℕ) (m : ℕ → Option Bool)
    (hm : ∀ a, m a = if a ∈ seen then some (w.mem a) else none) (l : List Bool) :
    c.replay w.oracle m (fun n => (c.trace w seen ++ l).getD n false) = c.queries w := by
  induction c generalizing seen m l with
  | done a => simp [replay, queries]
  | tag x k ih => simp only [replay, queries, trace, ih _ seen m hm l]
  | ingress a k ih =>
    by_cases ha : a ∈ seen
    · have hma : m a = some (w.mem a) := by rw [hm]; simp [ha]
      simp only [replay, trace, queries, ha, ↓reduceIte, hma]
      exact ih _ seen m hm l
    · have hma : m a = none := by rw [hm]; simp [ha]
      simp only [replay, trace, queries, ha, ↓reduceIte, hma, List.cons_append, List.getD_cons_zero,
        List.getD_cons_succ]
      refine ih _ (insert a seen) _ (fun a' => ?_) l
      by_cases h : a' = a
      · subst h; simp
      · simp [hm, h]
  | side a k ih => simpa [replay, queries, trace] using ih _ seen m hm l
  | stream i k ih => simpa [replay, queries, trace] using ih _ seen m hm l
  | rebuild j k ih =>
    simp only [replay, queries, trace, List.append_assoc]
    have hb : (fun i : Fin b => (List.ofFn (w.store j) ++ ((k (w.store j)).trace w seen ++ l)).getD i false) =
        w.store j := by
      funext i
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by simp)]
      simp
    have hs : (fun n => (List.ofFn (w.store j) ++ ((k (w.store j)).trace w seen ++ l)).getD (n + b) false) =
        fun n => ((k (w.store j)).trace w seen ++ l).getD n false := by
      funext n
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.getD_eq_getElem?_getD]
      simp
    rw [hb, hs, ih _ seen m hm l]

/-- The footprint is at most the ingress: each address is read at least once. -/
theorem card_addrs_le (w : World N b h ρ) (c : Chip N b h ρ α) : (c.addrs w).card ≤ c.ingressBits w := by
  induction c with
  | done a => simp [addrs, ingressBits]
  | tag x k ih => simpa [addrs, ingressBits] using ih _
  | ingress a k ih => exact (Finset.card_insert_le _ _).trans (by simpa [ingressBits] using ih _)
  | side a k ih => simpa [addrs, ingressBits] using ih _
  | stream i k ih => simpa [addrs, ingressBits] using ih _
  | rebuild j k ih => simpa [addrs, ingressBits] using ih _

/-- Changing the oracle off the points a run queries changes neither the run nor its queries. -/
theorem run_congr (w : World N b h ρ) (O' : Oracle N b h ρ) (c : Chip N b h ρ α)
    (ho : ∀ x ∈ c.queries w, O' x = w.oracle x) :
    c.run { w with oracle := O' } = c.run w ∧ c.queries { w with oracle := O' } = c.queries w := by
  induction c with
  | done a => simp [run, queries]
  | tag x k ih =>
    simp only [queries, List.mem_cons, forall_eq_or_imp] at ho
    obtain ⟨hx, ho⟩ := ho
    simp only [run, queries, hx]
    obtain ⟨h1, h2⟩ := ih _ ho
    exact ⟨h1, by rw [h2]⟩
  | ingress a k ih => exact ih _ ho
  | side a k ih => exact ih _ ho
  | stream i k ih => exact ih _ ho
  | rebuild j k ih => exact ih _ ho

end Chip

/-- A filter's size does not depend on how the type is labelled. -/
theorem card_filter_equiv {X Y : Type} [Fintype X] [Fintype Y] (e : X ≃ Y) (P : X → Prop) [DecidablePred P] :
    (Finset.univ.filter P).card = (Finset.univ.filter (fun y => P (e.symm y))).card := by
  rw [← Fintype.card_subtype, ← Fintype.card_subtype]
  exact Fintype.card_congr (e.subtypeEquiv fun x => by simp)

end Accounting.Rate
