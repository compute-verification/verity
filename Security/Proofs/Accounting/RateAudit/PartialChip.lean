import Mathlib
import Models.Accounting.Assumptions
import Proofs.Accounting.RateAudit.Chip

/-!
# Chip programs with partial completion: congruence, replay from a footprint trace, counting, and the price

`Chip.lean`'s lemmas for `PChip`. `replay_trace`: given the oracle, the run's queries are a function of the chip program
and its trace, the bits device memory (once per address, at its first read), the side link and the completions
answered with; the trace's length is the footprint plus the link's bits plus the bits completed (`trace_length`), so a
completion of `x` bits costs the compression `x` bits, not `b`. `completed_mul_le`: under `WorkHardness`, the bits
completed, at `rounds · c_root / b` each, cost at most the completions' compute. `Chip.toPartial`: `RateAudit.lean`'s
programs are partial ones, a rebuild the completion of all `b` bits.
-/

namespace Accounting.Rate

open Pous (Bits pr)

namespace PChip

variable {N b h ρ : ℕ} {α : Type}

/-- The bits a run was answered with, other than the oracle's, with device memory read once per address: the bit at
an address's first read (`seen` holds the addresses read before), every bit from the side link, and every completion's
bits. -/
def trace (w : World N b h ρ) : PChip N b h ρ α → Finset ℕ → List Bool
  | done _, _ => []
  | tag x k, seen => (k (w.oracle x)).trace w seen
  | ingress a k, seen =>
    if a ∈ seen then (k (w.mem a)).trace w seen else w.mem a :: (k (w.mem a)).trace w (insert a seen)
  | side a k, seen => w.side a :: (k (w.side a)).trace w seen
  | stream i k, seen => w.stream i :: (k (w.stream i)).trace w seen
  | complete j _ _ g k, seen => List.ofFn (g (w.store j)) ++ (k (g (w.store j))).trace w seen

/-- The queries a run makes, recomputed from the oracle and a stream of the bits it was answered with otherwise, with a
cache `m` of the device-memory bits read so far. -/
def replay (O : Oracle N b h ρ) : PChip N b h ρ α → (ℕ → Option Bool) → (ℕ → Bool) → List (Bits ρ × Fin N × Bits b)
  | done _, _, _ => []
  | tag x k, m, f => x :: (k (O x)).replay O m f
  | ingress a k, m, f =>
    match m a with
    | some v => (k v).replay O m f
    | none => (k (f 0)).replay O (Function.update m a (some (f 0))) (fun n => f (n + 1))
  | side _ k, m, f => (k (f 0)).replay O m (fun n => f (n + 1))
  | stream _ k, m, f => (k (f 0)).replay O m (fun n => f (n + 1))
  | complete _ x _ _ k, m, f => (k (fun i => f i)).replay O m (fun n => f (n + x))

theorem trace_length (w : World N b h ρ) (c : PChip N b h ρ α) (seen : Finset ℕ) :
    (c.trace w seen).length = (c.addrs w \ seen).card + c.linkBits w + c.completed w := by
  induction c generalizing seen with
  | done a => simp [trace, addrs, linkBits, completed]
  | tag x k ih => simp [trace, addrs, linkBits, completed, ih]
  | ingress a k ih =>
    by_cases ha : a ∈ seen
    · simp only [trace, addrs, linkBits, completed, ha, ↓reduceIte, ih, Finset.insert_sdiff_of_mem _ ha]
    · simp only [trace, addrs, linkBits, completed, ha, ↓reduceIte, List.length_cons, ih,
        Finset.insert_sdiff_of_notMem _ ha, Finset.sdiff_insert]
      have hc : (insert a ((k (w.mem a)).addrs w \ seen)).card = (((k (w.mem a)).addrs w \ seen).erase a).card + 1 := by
        rw [← Finset.card_erase_add_one (Finset.mem_insert_self a _), Finset.erase_insert_eq_erase]
      rw [hc]
      omega
  | side a k ih => simp [trace, addrs, linkBits, completed, ih]; omega
  | stream i k ih => simp [trace, addrs, linkBits, completed, ih]; omega
  | complete j x hx g k ih => simp [trace, addrs, linkBits, completed, ih]; omega

theorem replay_trace (w : World N b h ρ) (c : PChip N b h ρ α) (seen : Finset ℕ) (m : ℕ → Option Bool)
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
  | complete j x hx g k ih =>
    simp only [replay, queries, trace, List.append_assoc]
    have hb : (fun i : Fin x => (List.ofFn (g (w.store j)) ++ ((k (g (w.store j))).trace w seen ++ l)).getD i false) =
        g (w.store j) := by
      funext i
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by simp)]
      simp
    have hs : (fun n => (List.ofFn (g (w.store j)) ++ ((k (g (w.store j))).trace w seen ++ l)).getD (n + x) false) =
        fun n => ((k (g (w.store j))).trace w seen ++ l).getD n false := by
      funext n
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.getD_eq_getElem?_getD]
      simp
    rw [hb, hs, ih _ seen m hm l]

/-- The footprint is at most the ingress: each address is read at least once. -/
theorem card_addrs_le (w : World N b h ρ) (c : PChip N b h ρ α) : (c.addrs w).card ≤ c.ingressBits w := by
  induction c with
  | done a => simp [addrs, ingressBits]
  | tag x k ih => simpa [addrs, ingressBits] using ih _
  | ingress a k ih => exact (Finset.card_insert_le _ _).trans (by simpa [ingressBits] using ih _)
  | side a k ih => simpa [addrs, ingressBits] using ih _
  | stream i k ih => simpa [addrs, ingressBits] using ih _
  | complete j x hx g k ih => simpa [addrs, ingressBits] using ih _

/-- Changing the oracle off the points a run queries changes neither the run nor its queries. -/
theorem run_congr (w : World N b h ρ) (O' : Oracle N b h ρ) (c : PChip N b h ρ α)
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
  | complete j x hx g k ih => exact ih _ ho

/-- Under `WorkHardness`, the bits a run completed, at `rounds · c_root` per `b` of them, cost at most its completions'
compute. -/
theorem completed_mul_le {rounds cRoot : ℕ} {cost : ℕ → ℕ} (hw : Assumptions.WorkHardness b rounds cRoot cost)
    (w : World N b h ρ) (c : PChip N b h ρ α) : c.completed w * (rounds * cRoot) ≤ b * c.work cost w := by
  induction c with
  | done a => simp [completed]
  | tag x k ih => simpa [completed, work] using ih _
  | ingress a k ih => simpa [completed, work] using ih _
  | side a k ih => simpa [completed, work] using ih _
  | stream i k ih => simpa [completed, work] using ih _
  | complete j x hx g k ih =>
    simp only [completed, work, add_mul, mul_add]
    exact add_le_add (ih _) (hw x hx)

end PChip

/-- A chip program of `RateAudit.lean`'s game as a partial one: `rebuild j` is the completion of all `b` bits of word
`j`. -/
def Chip.toPartial {N b h ρ : ℕ} {α : Type} : Chip N b h ρ α → PChip N b h ρ α
  | .done a => .done a
  | .tag x k => .tag x (fun y => (k y).toPartial)
  | .ingress a k => .ingress a (fun v => (k v).toPartial)
  | .side a k => .side a (fun v => (k v).toPartial)
  | .stream i k => .stream i (fun v => (k v).toPartial)
  | .rebuild j k => .complete j b le_rfl id (fun y => (k y).toPartial)

namespace Chip

variable {N b h ρ : ℕ} {α : Type}

/-- As a partial program, a chip program runs, reads, sends and queries as it did, and completes `b` bits per rebuilt
word. -/
theorem toPartial_eq (w : World N b h ρ) (c : Chip N b h ρ α) :
    c.toPartial.run w = c.run w ∧ c.toPartial.ingressBits w = c.ingressBits w ∧ c.toPartial.addrs w = c.addrs w ∧
      c.toPartial.linkBits w = c.linkBits w ∧ c.toPartial.queries w = c.queries w ∧
        c.toPartial.completed w = c.rebuilds w * b := by
  induction c with
  | done a => simp [toPartial, PChip.run, PChip.ingressBits, PChip.addrs, PChip.linkBits, PChip.queries,
      PChip.completed, run, ingressBits, addrs, linkBits, queries, rebuilds]
  | tag x k ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (w.oracle x)
    simp only [toPartial, PChip.run, PChip.ingressBits, PChip.addrs, PChip.linkBits, PChip.queries,
      PChip.completed, run, ingressBits, addrs, linkBits, queries, rebuilds]
    exact ⟨h1, h2, h3, h4, by rw [h5], h6⟩
  | ingress a k ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (w.mem a)
    simp only [toPartial, PChip.run, PChip.ingressBits, PChip.addrs, PChip.linkBits, PChip.queries,
      PChip.completed, run, ingressBits, addrs, linkBits, queries, rebuilds]
    exact ⟨h1, by rw [h2], by rw [h3], h4, h5, h6⟩
  | side a k ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (w.side a)
    simp only [toPartial, PChip.run, PChip.ingressBits, PChip.addrs, PChip.linkBits, PChip.queries,
      PChip.completed, run, ingressBits, addrs, linkBits, queries, rebuilds]
    exact ⟨h1, h2, h3, by rw [h4], h5, h6⟩
  | stream i k ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (w.stream i)
    simp only [toPartial, PChip.run, PChip.ingressBits, PChip.addrs, PChip.linkBits, PChip.queries,
      PChip.completed, run, ingressBits, addrs, linkBits, queries, rebuilds]
    exact ⟨h1, h2, h3, by rw [h4], h5, h6⟩
  | rebuild j k ih =>
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := ih (w.store j)
    simp only [toPartial, PChip.run, PChip.ingressBits, PChip.addrs, PChip.linkBits, PChip.queries,
      PChip.completed, run, ingressBits, addrs, linkBits, queries, rebuilds, id]
    exact ⟨h1, h2, h3, h4, h5, by rw [h6, Nat.succ_mul]⟩

/-- As a partial program, a chip program's completions cost `cost b` per rebuilt word. -/
theorem toPartial_work (cost : ℕ → ℕ) (w : World N b h ρ) (c : Chip N b h ρ α) :
    c.toPartial.work cost w = c.rebuilds w * cost b := by
  induction c with
  | done a => simp [toPartial, PChip.work, rebuilds]
  | tag x k ih => simpa [toPartial, PChip.work, rebuilds] using ih _
  | ingress a k ih => simpa [toPartial, PChip.work, rebuilds] using ih _
  | side a k ih => simpa [toPartial, PChip.work, rebuilds] using ih _
  | stream i k ih => simpa [toPartial, PChip.work, rebuilds] using ih _
  | rebuild j k ih =>
    simp only [toPartial, PChip.work, rebuilds, id, ih, Nat.succ_mul]

end Chip

end Accounting.Rate
