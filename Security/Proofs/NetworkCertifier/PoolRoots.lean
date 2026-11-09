module

public import Definitions.NetworkCertifier.Guarantees.PoolRoots

/-!
# The pool's roots, from each run's in-circuit checks and the record rule: proofs

A run's leak relation gives its check's run, so its opened row has the digest `pin i` and its left row `pout i`
(`digests_of_leakAt`). The record rule puts at `pin i` run `i - 1`'s `pout` where run `i` carries the pool and the
empty pool's root otherwise, the digest of the row `poolOpen` names (`digest_open_of_leakAt`). `RootsOfChecked` reads
that at every run.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **The run before run `i`**, for `0 < i`. -/
def prevRun {m : ℕ} (i : Fin m) : Fin m := ⟨i.val - 1, lt_of_le_of_lt (Nat.sub_le i.val 1) i.isLt⟩

theorem exists_single_of_faultyGate {C : Prog} {P : Partition} {F : Finset ℕ} {g : ℕ} (h : FaultyGate C P F g) :
    ∃ v ∈ F, FaultyGate C P {v} g := by
  obtain ⟨v, hv, u, hu, hg, hb⟩ := h
  exact ⟨v, hv, v, Finset.mem_singleton_self v, u, hu, hg, hb⟩

theorem poolOpen_map {m : ℕ} {R R' : Type} (f : R → R') (chains : Fin m → Bool) (genesis rout : Fin m → R)
    (i : Fin m) : f (poolOpen chains genesis rout i) = poolOpen chains (fun j => f (genesis j)) (fun j => f (rout j)) i := by
  unfold poolOpen
  split_ifs <;> rfl

/-- A run's leak relation runs its check, so its opened row has the digest `pin` and its left row the digest `pout`. -/
theorem digests_of_leakAt {L K S : ℕ} {G : Hash.Msg → Hash.Digest} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop}
    {prow nrow : AuditStatement → List ℕ} {Dj : ℕ} {s : AuditStatement} {P : Partition} {ρ : ℕ → Bool}
    {op : ℕ → List (Obs L K S) → List Bool} {rin rout : Hash.Row} {F : Finset ℕ} {o : List (Obs L K S)}
    {pin pout : Hash.Digest}
    (hroots : CheckedRoots G s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s)
      (prow s) (nrow s) pin pout)
    (h : LeakAtCarried lay k reg valid prow nrow Dj s P ρ op rin rout F o) :
    Hash.RowSeg.digest G rin = pin ∧ Hash.RowSeg.digest G rout = pout := by
  obtain ⟨-, hdisj, -, -, a, ha, free, hfree, hval, hin, hout, -⟩ := h
  have hsupp : ∀ g b, free g = some b → ∃ v ∉ allSlots s.design P, FaultyGate s.prog P {v} g := by
    intro g b hgb
    obtain ⟨v, hv, hfv⟩ := exists_single_of_faultyGate (hfree g b hgb)
    exact ⟨v, Finset.disjoint_left.1 hdisj hv, hfv⟩
  obtain ⟨h1, h2⟩ := hroots _ a ha free (by simp) hsupp hval
  rw [← hin, ← hout]
  exact ⟨h1, h2⟩

/-- Run `i` opens a row with the digest of the row `poolOpen` names when its leak relation holds and, where it carries
the pool, the leak relation of the run before it holds too. -/
theorem digest_open_of_leakAt {L K S : ℕ} {G : Hash.Msg → Hash.Digest} {lay : AuditStatement → BoundaryGates L S}
    {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ} {valid : AuditStatement → (ℕ → Bool) → Prop}
    {prow nrow : AuditStatement → List ℕ} {Dj : ℕ} {m : ℕ} {sw : Fin m → AuditStatement} {Pw : Fin m → Partition}
    {ρ : Fin m → ℕ → Bool} {chains : Fin m → Bool} {genesis : Fin m → Hash.Row} {pin pout : Fin m → Hash.Digest}
    (hroots : ∀ i, CheckedRoots G (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
      (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i)) (valid (sw i)) (prow (sw i)) (nrow (sw i)) (pin i)
      (pout i))
    (hR7 : PoolChained chains (fun i => Hash.RowSeg.digest G (genesis i)) pin pout) (rin rout : Fin m → Hash.Row)
    (i : Fin m)
    (hi : ∃ (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) (o : List (Obs L K S)),
      LeakAtCarried lay k reg valid prow nrow Dj (sw i) (Pw i) (ρ i) op (rin i) (rout i) F o)
    (hprev : 0 < i.val ∧ chains i = true → ∃ (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ)
      (o : List (Obs L K S)), LeakAtCarried lay k reg valid prow nrow Dj (sw (prevRun i)) (Pw (prevRun i))
        (ρ (prevRun i)) op (rin (prevRun i)) (rout (prevRun i)) F o) :
    Hash.RowSeg.digest G (rin i) = Hash.RowSeg.digest G (poolOpen chains genesis rout i) := by
  obtain ⟨op, F, o, h⟩ := hi
  rw [(digests_of_leakAt (hroots i) h).1, hR7 i, poolOpen_map (Hash.RowSeg.digest G) chains genesis rout i]
  unfold poolOpen
  split_ifs with hc
  · obtain ⟨op', F', o', h'⟩ := hprev hc
    exact (digests_of_leakAt (hroots (prevRun i)) h').2.symm
  · rfl

theorem rootsOfChecked : Guarantees.RootsOfChecked := by
  intro L K S G lay k reg valid prow nrow Dj m sw Pw ρ op chains genesis pin pout hroots hR7 o W rin rout hleak i
  exact digest_open_of_leakAt hroots hR7 rin rout i ⟨_, _, _, hleak i⟩ fun _ => ⟨_, _, _, hleak (prevRun i)⟩

end NetTiming
