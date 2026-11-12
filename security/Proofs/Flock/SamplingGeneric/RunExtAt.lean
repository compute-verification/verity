module

public import Proofs.Flock.SamplingGeneric.Reg
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Recursion.Serve
import Proofs.Flock.Claims.Sampling.Bext
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.SamplingGeneric.Assemble

/-!
# A run's extractions at one gate

`runExt` extracts every session of a run's sample. A finder that waits for one gate needs only the sessions that can
open it. Where a reveal for proof block `u` opens only gates of `needs u`, those are the sessions with a proof block
that needs the gate:

* `runExtAt`: `runExt` at the slots whose session has a proof block that needs gate `g`, and no other.
* `mem_runExtAt`: an opening `runExt` keeps at a gate is kept by `runExtAt` at that gate.
* `callsBy`, `callsBy_le`: a run's calls with only those sessions extracted, at most its calls with every session
  extracted.
* `collBy`, `collBy_collision`, `collBy_isSome`, `collBy_of_conflictAt`: the first collision among the calls of one
  run's extractions and of another run's extractions at one gate. Two openings the two runs' extractions keep, under
  one roots at that gate, of different values, give one (`CommitmentScheme.Binds`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement ownLog flatMapRP)

/-- A rerun program over a list calls `H`, its reruns included, as often as its parts do. -/
theorem flatMapRP_hashes {R β γ : Type} (H : Msg → Msg) (run : R → Transcript) (f : β → RP R (List γ)) :
    ∀ l : List β, ((flatMapRP f l).eval H run).2.1 = (l.map fun b => ((f b).eval H run).2.1).sum
  | [] => rfl
  | b :: bs => by
    simp only [flatMapRP, List.map_cons, List.sum_cons]
    rw [RP.eval_bind_hashes, RP.eval_bind_hashes, flatMapRP_hashes H run f bs]
    simp [RP.eval]

/-- Over the part of a list a filter keeps, a rerun program calls `H` at most as often as over the list. -/
theorem flatMapRP_hashes_filter_le {R β γ : Type} (H : Msg → Msg) (run : R → Transcript) (f : β → RP R (List γ))
    (keep : β → Bool) (l : List β) :
    ((flatMapRP f (l.filter keep)).eval H run).2.1 ≤ ((flatMapRP f l).eval H run).2.1 := by
  rw [flatMapRP_hashes, flatMapRP_hashes]
  induction l with
  | nil => exact le_rfl
  | cons b bs ih =>
    by_cases hb : keep b = true
    · rw [List.filter_cons_of_pos hb, List.map_cons, List.map_cons, List.sum_cons, List.sum_cons]
      exact Nat.add_le_add_left ih _
    · rw [List.filter_cons_of_neg hb, List.map_cons, List.sum_cons]
      exact ih.trans (Nat.le_add_left _ _)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **A run's extractions at one gate**: `runExt` at the slots of the sample whose session has a proof block that
needs gate `g` (`needs`), and no other; each such session is extracted as `runExt` extracts it, at the same coins. -/
noncomputable def runExtAt (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (g : ℕ)
    (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) :
    RP (Fin (L.V.k x) → Coin) (List (List Msg × Sampling.Opening Bool)) :=
  flatMapRP (fun iu : Fin (L.nPU x) × ℕ =>
      if hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val then
        L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r v (c iu.1 ⟨iu.2, hu.1⟩)
      else .ret [])
    (((List.finRange (L.nPU x)).zip (L.checked x r)).filter fun iu =>
      decide (∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, g ∈ needs u))

/-- What a run's extractions at a gate keep passes its opening check, with the check's calls their own. -/
theorem passes_runExtAt (H : Msg → Msg) (L : Compiled Coin A) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (s : AuditStatement) (needs : ℕ → Finset ℕ) (g : ℕ)
    (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x) :
    Sampling.Passes H run' L.commit (L.runExtAt x s needs g r v c) := by
  unfold runExtAt
  exact Sampling.Passes.flatMapRP fun _ _ => by
    split
    · exact L.passes_sessExt H x run' s _ _ _ _ _
    · exact Sampling.Passes.ret_nil H run' L.commit

/-- **An opening a run's extractions keep is kept by its extractions at the opening's gate**, where a reveal that
holds it, for a proof block `u`, puts its gate in `needs u`. -/
theorem mem_runExtAt (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : L.SlotEC x)
    {q : List Msg × Sampling.Opening Bool} (hpos : ∀ u w, q ∈ L.reveal s u v w → q.2.pos ∈ needs u)
    (hq : q ∈ ((L.runExt x s r v c).eval H run').1) :
    q ∈ ((L.runExtAt x s needs q.2.pos r v c).eval H run').1 := by
  unfold runExt at hq
  unfold runExtAt
  rw [Sampling.flatMapRP_eval] at hq ⊢
  obtain ⟨iu, hiu, hqiu⟩ := List.mem_flatMap.1 hq
  have hkeep : ∃ u ∈ L.sessBlocks x (L.checked x r) iu.1, q.2.pos ∈ needs u := by
    have hqiu' := hqiu
    split at hqiu'
    · unfold sessExt at hqiu'
      rw [(Claims.RP.eval_bind H _ _ _).1] at hqiu'
      split at hqiu'
      · simp [RP.eval] at hqiu'
      · rw [Sampling.flatMapRP_eval] at hqiu'
        obtain ⟨u', hu'm, hu'⟩ := List.mem_flatMap.1 hqiu'
        exact ⟨u', hu'm, hpos _ _ (Sampling.checkRevealed_mem H _ L.commit _ q hu').1⟩
    · simp [RP.eval] at hqiu'
  exact List.mem_flatMap.2 ⟨iu, List.mem_filter.2 ⟨hiu, decide_eq_true hkeep⟩, hqiu⟩

/-- **A run's calls at one gate**: both parties', and those of its extractions at the gate (`runExtAt`), their
reruns' included. -/
noncomputable def callsBy (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) (g : ℕ) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : ℕ :=
  (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length +
    ((L.runExtAt x s needs g r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.1

/-- A run's calls at one gate are at most its calls with every session of its sample extracted. -/
theorem callsBy_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (needs : ℕ → Finset ℕ) (g : ℕ) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsBy H P x s needs g ω r c ≤
      (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length +
        ((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.1 := by
  unfold callsBy runExtAt runExt
  exact Nat.add_le_add_left (flatMapRP_hashes_filter_le H _ _ _ _) _

/-- **The pair a run's extractions and another run's extractions at one gate give**, each run at its own tape: the
first collision among the calls the first run's extractions make themselves and those the second run's extractions
at gate `g` make themselves. -/
noncomputable def collBy (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) :
    Option (Msg × Msg) :=
  findColl (ownLog H (run H P L.V x a.1) (L.runExt x s a.2.1 (run H P L.V x a.1 a.2.1).seen a.2.2) ++
    ownLog H (run H P L.V x a'.1) (L.runExtAt x s needs g a'.2.1 (run H P L.V x a'.1 a'.2.1).seen a'.2.2))

/-- What `collBy` outputs is a collision of `H`. -/
theorem collBy_collision (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ)
    {u v : Msg} (h : L.collBy H P x s needs a a' g = some (u, v)) : Collision H u v :=
  findColl_spec H _
    (fun p hp => (List.mem_append.1 hp).elim (Sampling.ownLog_valid H _ _ p) (Sampling.ownLog_valid H _ _ p)) u v h

/-- **Two openings two runs' extractions keep, under one roots at one gate, of different values, give `collBy` a
pair** at that gate, where a reveal that holds the second, for a proof block `u`, puts its gate in `needs u`. -/
theorem collBy_isSome {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    (s : AuditStatement) {Ω : Type} (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) {q q' : List Msg × Sampling.Opening Bool}
    (hq : q ∈ L.kept H P x s a.1 a.2.1 a.2.2) (hq' : q' ∈ L.kept H P x s a'.1 a'.2.1 a'.2.2)
    (hpos : ∀ u w, q' ∈ L.reveal s u (run H P L.V x a'.1 a'.2.1).seen w → q'.2.pos ∈ needs u)
    (h1 : q.1 = q'.1) (h2 : q.2.pos = q'.2.pos) (h3 : q.2.val ≠ q'.2.val) :
    (L.collBy H P x s needs a a' q'.2.pos).isSome := by
  have hq'' := L.mem_runExtAt H x (run H P L.V x a'.1) s needs a'.2.1 _ a'.2.2 hpos hq'
  obtain ⟨ho, hl⟩ := L.passes_runExt H x (run H P L.V x a.1) s a.2.1 _ a.2.2 q hq
  obtain ⟨ho', hl'⟩ := L.passes_runExtAt H x (run H P L.V x a'.1) s needs q'.2.pos a'.2.1 _ a'.2.2 q' hq''
  obtain ⟨rt, o⟩ := q
  obtain ⟨rt', o'⟩ := q'
  dsimp only at h1 h2 h3 ho ho' hl hl' ⊢
  subst h1
  exact Sampling.findColl_isSome _ (Sampling.disagree_collision hbind h2 h3 ho ho'
    (fun e he => List.mem_append_left _ (hl e he)) (fun e he => List.mem_append_right _ (hl' e he)))

/-- **Two runs, each at its own tape, whose extractions open a gate two ways under the statement's roots give
`collBy` a pair at that gate**, where a reveal for a proof block `u` opens, under the statement's roots, only gates
of `needs u`. -/
theorem collBy_of_conflictAt {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u)
    (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ)
    (h1 : (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome) (h2 : (L.opensAt H P x a'.1 a'.2.1 a'.2.2 g).isSome)
    (hne : L.opensAt H P x a.1 a.2.1 a.2.2 g ≠ L.opensAt H P x a'.1 a'.2.1 a'.2.2 g) :
    (L.collBy H P x s needs a a' g).isSome := by
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  rw [L.opensAt_eq H P x a.1 hs] at hb1
  rw [L.opensAt_eq H P x a'.1 hs] at hb2
  obtain ⟨q, hq, hq1, hq2, hq3, -⟩ := val_some_mem hb1
  obtain ⟨q', hq', hq1', hq2', hq3', -⟩ := val_some_mem hb2
  have h := collBy_isSome hbind s P needs a a' hq hq' (fun u w hm => hpos u _ w q' hm hq1')
    (hq1.trans hq1'.symm) (hq2.trans hq2'.symm) (by rw [hq3, hq3']; exact fun h => hne (congrArg some h))
  rw [hq2'] at h
  exact h

end Compiled

end Claims.SamplingGeneric
