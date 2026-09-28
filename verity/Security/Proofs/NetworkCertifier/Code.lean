import Definitions.NetworkCertifier
import NetworkCertifier.Ingress
import NetworkCertifier.Schedule
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Finset.Dedup

/-!
# The certifier program agrees with the reference definitions

The program (`certifier_program`, `NetworkCertifier`) is Lean core only: it sums over `List.finRange` and keeps `Σ_sync` as
two duplicate-free lists, as Python does. The reference (`Definitions.NetworkCertifier`, `NetTiming`) sums over `Finset`s and
keeps `Σ_sync` as two `Finset`s; it is what the guarantees read. The difftest runs the program against the Python audit;
the guarantees here say that what it evaluates is the reference's, so the chain is Python ≈ the program (the difftest)
and the program = the reference (proved):

* `CodeExpectedRow`: the program's `expectedRow` is the reference's, for clock syncs and frames with the same fields;
* `CodeAcceptedConstantRate`: `Accepted` of the program's constant-rate Program is the reference's, for observations
  with the same rows and the same complete statuses, an operator giving the same ingress, frames with the same fields,
  and advice naming the same values;
* `CodeStatusComplete`: the program's `StatusComplete` is the reference's, for observations with the same complete
  statuses;
* `CodeSyncSet`: every sync set of the program has a reference sync set with the same values, and every reference sync
  set one of the program's, so the program's membership test (the audit's `advice` verdict) is the reference's.

Each statement relates the two sides by their fields rather than by a transport, so that it reads only the two sides'
definitions. The transports (`frameRef`, `syncRef`, `syncSetRef`, `syncSetCode`) are the proofs' own.
-/

namespace NetTiming.SecurityProofs

open Finset

/-- The reference form of the program's frame. -/
def frameRef (f : NetworkCertifier.LogicalFrame) : LogicalFrame := ⟨f.payload, f.anchor, f.index⟩

/-- The reference form of the program's clock sync. -/
def syncRef (s : NetworkCertifier.ClockSync) : ClockSync := ⟨s.delta, s.rho, s.rho_pos⟩

/-- The reference form of the program's sync set: its lists as `Finset`s. -/
def syncSetRef (ss : NetworkCertifier.SyncSet) : SyncSet where
  deltas := ss.deltas.toFinset
  rhos := ss.rhos.toFinset
  deltas_nonempty := List.toFinset_nonempty_iff _ |>.mpr ss.deltas_nonempty
  rhos_nonempty := List.toFinset_nonempty_iff _ |>.mpr ss.rhos_nonempty
  rhos_pos ρ h := ss.rhos_pos ρ (List.mem_toFinset.mp h)

/-- A sync set of the program with a reference's values, its `Finset`s listed in some order. -/
noncomputable def syncSetCode (ss : SyncSet) : NetworkCertifier.SyncSet where
  deltas := ss.deltas.toList
  rhos := ss.rhos.toList
  deltas_nonempty := by simpa using ss.deltas_nonempty.ne_empty
  rhos_nonempty := by simpa using ss.rhos_nonempty.ne_empty
  rhos_pos ρ h := ss.rhos_pos ρ (Finset.mem_toList.mp h)
  deltas_nodup := ss.deltas.nodup_toList
  rhos_nodup := ss.rhos.nodup_toList

/-- Every reference sync set is the reference form of one of the program's. -/
theorem syncSetRef_syncSetCode (ss : SyncSet) : syncSetRef (syncSetCode ss) = ss := by
  obtain ⟨d, r, _, _, _⟩ := ss
  simp only [syncSetRef, syncSetCode, Finset.toList_toFinset]

theorem code_offset {T S : ℕ} (n : Fin T → Fin (S + 1)) (t : Fin T) : NetworkCertifier.offset n t = offset n t := by
  rw [NetworkCertifier.offset, offset, ← List.sum_toFinset _ ((List.nodup_finRange T).filter _), List.toFinset_filter,
    List.toFinset_finRange]
  simp only [decide_eq_true_eq]

theorem code_fill {T S : ℕ} {F : Type} (f : ℕ → F) (n : Fin T → Fin (S + 1)) : NetworkCertifier.fill f n = fill f n := by
  funext t s
  simp only [NetworkCertifier.fill, fill, code_offset]

theorem code_queue (S : ℕ) (enq : ℕ → ℕ) : ∀ t, NetworkCertifier.queue S enq t = queue S enq t
  | 0 => rfl
  | t + 1 => by simp only [NetworkCertifier.queue, queue, code_queue S enq t]

theorem code_released (S : ℕ) (enq : ℕ → ℕ) (t : ℕ) : NetworkCertifier.released S enq t = released S enq t :=
  Fin.ext (by simp only [NetworkCertifier.released, released, code_queue])

theorem code_fifoGrid (T S : ℕ) {F : Type} (f : ℕ → F) (enq : ℕ → ℕ) :
    NetworkCertifier.fifoGrid T S f enq = fifoGrid T S f enq := by
  simp only [NetworkCertifier.fifoGrid, fifoGrid, code_fill, code_released]

theorem code_handingFrom (T W : ℕ) (syncs : Fin W → NetworkCertifier.ClockSync) :
    ∀ (last : ℕ) (frames : List NetworkCertifier.LogicalFrame),
      NetworkCertifier.handingFrom T W syncs last frames =
        handingFrom T W (fun w => syncRef (syncs w)) last (frames.map frameRef)
  | _, [] => rfl
  | last, f :: fs => by
    simp only [NetworkCertifier.handingFrom, handingFrom, List.map_cons, NetworkCertifier.eligible, eligible, frameRef, syncRef]
    congr 1
    exact code_handingFrom T W syncs _ fs

theorem code_enqueued (T W : ℕ) (syncs : Fin W → NetworkCertifier.ClockSync) (frames : List NetworkCertifier.LogicalFrame) (t : ℕ) :
    NetworkCertifier.enqueued T W syncs frames t = enqueued T W (fun w => syncRef (syncs w)) (frames.map frameRef) t := by
  simp only [NetworkCertifier.enqueued, enqueued, NetworkCertifier.handingBuckets, handingBuckets, code_handingFrom]

theorem code_payloads (frames : List NetworkCertifier.LogicalFrame) (k : ℕ) :
    NetworkCertifier.payloads frames k = payloads (frames.map frameRef) k := by
  simp only [NetworkCertifier.payloads, payloads, List.getElem?_map, Option.map_map]
  rfl

theorem code_expectedRow (S T W : ℕ) (syncs : Fin W → NetworkCertifier.ClockSync) (frames : List NetworkCertifier.LogicalFrame) (t : ℕ) :
    NetworkCertifier.expectedRow S T W syncs frames t =
      expectedRow S T W (fun w => syncRef (syncs w)) (frames.map frameRef) t := by
  have hp : NetworkCertifier.payloads frames = payloads (frames.map frameRef) := funext (code_payloads frames)
  have he : NetworkCertifier.enqueued T W syncs frames = enqueued T W (fun w => syncRef (syncs w)) (frames.map frameRef) :=
    funext (code_enqueued T W syncs frames)
  simp only [NetworkCertifier.expectedRow, expectedRow, code_fifoGrid, hp, he]

theorem frames_eq_map_frameRef {frames : List NetworkCertifier.LogicalFrame} {frames' : List LogicalFrame}
    (hf : frames.map (fun f => (f.payload, f.anchor, f.index)) = frames'.map (fun f => (f.payload, f.anchor, f.index))) :
    frames' = frames.map frameRef := by
  have hinj : Function.Injective fun f : LogicalFrame => (f.payload, f.anchor, f.index) := by
    rintro ⟨p, a, i⟩ ⟨p', a', i'⟩ h
    simp only [Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    rfl
  apply List.map_injective_iff.mpr hinj
  rw [← hf, List.map_map]
  rfl

theorem syncs_eq_syncRef {W : ℕ} {syncs : Fin W → NetworkCertifier.ClockSync} {syncs' : Fin W → ClockSync}
    (hs : ∀ w, (syncs w).delta = (syncs' w).delta ∧ (syncs w).rho = (syncs' w).rho) :
    syncs' = fun w => syncRef (syncs w) := by
  funext w
  obtain ⟨hd, hr⟩ := hs w
  rcases e : syncs' w with ⟨d, r, _⟩
  rw [e] at hd hr
  simp only [syncRef, hd, hr]

/-- **The program's rows are the reference's.** For clock syncs with the same δ and ρ, and frames with the same
payloads, anchors and logical indices in the same order, the program's `expectedRow` (`schedule.expected_grids`, which
the difftest runs) is the reference's at every bucket. -/
theorem CodeExpectedRow (S T W : ℕ) (syncs : Fin W → NetworkCertifier.ClockSync) (syncs' : Fin W → ClockSync)
    (hs : ∀ w, (syncs w).delta = (syncs' w).delta ∧ (syncs w).rho = (syncs' w).rho)
    (frames : List NetworkCertifier.LogicalFrame) (frames' : List LogicalFrame)
    (hf : frames.map (fun f => (f.payload, f.anchor, f.index)) = frames'.map (fun f => (f.payload, f.anchor, f.index)))
    (t : ℕ) : NetworkCertifier.expectedRow S T W syncs frames t = expectedRow S T W syncs' frames' t := by
  rw [syncs_eq_syncRef hs, frames_eq_map_frameRef hf, code_expectedRow]

theorem code_ingressUpTo {L K S : ℕ} {In : Type} (op : ℕ → List (NetworkCertifier.Obs L K S) → In)
    (op' : ℕ → List (Obs L K S) → In) (o : List (NetworkCertifier.Obs L K S)) (o' : List (Obs L K S))
    (hop : ∀ i, op i (o.take i) = op' i (o'.take i)) (t : ℕ) : NetworkCertifier.ingressUpTo op o t = ingressUpTo op' o' t := by
  simp only [NetworkCertifier.ingressUpTo, ingressUpTo, hop]

/-- `Accepted` agrees on observations with the same rows and complete statuses when the two Programs predict the same
rows from the same ingress. -/
theorem code_accepted {L K S : ℕ} {In Adv Adv' : Type} (P : NetworkCertifier.Program L S In Adv) (P' : Program L S In Adv')
    (a : Adv) (a' : Adv') (hP : ∀ t ins, P.rows t ins a = P'.rows t ins a')
    (op : ℕ → List (NetworkCertifier.Obs L K S) → In) (op' : ℕ → List (Obs L K S) → In)
    (k : ℕ) (o : List (NetworkCertifier.Obs L K S)) (o' : List (Obs L K S))
    (hlen : o.length = o'.length)
    (hrows : ∀ t (h : t < o.length) (h' : t < o'.length), o[t].rows = o'[t].rows)
    (hstatus : ∀ t (h : t < o.length) (h' : t < o'.length) κ,
      o[t].status κ = .complete ↔ o'[t].status κ = .complete)
    (hop : ∀ i, op i (o.take i) = op' i (o'.take i)) :
    NetworkCertifier.Accepted P op k o a ↔ Accepted P' op' k o' a' := by
  have hin := code_ingressUpTo op op' o o' hop
  constructor
  · intro H t ht
    obtain ⟨h, hr, hc⟩ := H t ht
    have h' : t < o'.length := hlen ▸ h
    exact ⟨h', by rw [← hrows t h h', hr, hP, hin], fun κ => (hstatus t h h' κ).mp (hc κ)⟩
  · intro H t ht
    obtain ⟨h', hr, hc⟩ := H t ht
    have h : t < o.length := hlen ▸ h'
    exact ⟨h, by rw [hrows t h h', hr, ← hP, ← hin], fun κ => (hstatus t h h' κ).mpr (hc κ)⟩

/-- **The program's acceptance of the constant-rate Program is the reference's.** For observation sequences of one
length whose steps have the same egress rows and the same complete statuses, operators that give the same ingress on
them, frames with the same payloads, anchors and logical indices, the same content advice, and declared syncs naming
the same δ and ρ in every window, the program's `Accepted` of its `constantRate` (which the difftest decides) holds
exactly when the reference's does, whatever the two sync sets. -/
theorem CodeAcceptedConstantRate {L K S : ℕ} {In Ac : Type} (T W : ℕ) (ss : NetworkCertifier.SyncSet) (ss' : SyncSet)
    (frames : List In → Ac → Fin L → List NetworkCertifier.LogicalFrame) (frames' : List In → Ac → Fin L → List LogicalFrame)
    (hf : ∀ ins c ℓ, (frames ins c ℓ).map (fun f => (f.payload, f.anchor, f.index)) =
      (frames' ins c ℓ).map (fun f => (f.payload, f.anchor, f.index)))
    (op : ℕ → List (NetworkCertifier.Obs L K S) → In) (op' : ℕ → List (Obs L K S) → In)
    (k : ℕ) (o : List (NetworkCertifier.Obs L K S)) (o' : List (Obs L K S))
    (hlen : o.length = o'.length)
    (hrows : ∀ t (h : t < o.length) (h' : t < o'.length), o[t].rows = o'[t].rows)
    (hstatus : ∀ t (h : t < o.length) (h' : t < o'.length) κ,
      o[t].status κ = .complete ↔ o'[t].status κ = .complete)
    (hop : ∀ i, op i (o.take i) = op' i (o'.take i))
    (c : Ac) (σ : Fin W → ss.Elem) (σ' : Fin W → ss'.Elem)
    (hσ : ∀ w, (σ w).1.1 = (σ' w).1.1 ∧ (σ w).2.1 = (σ' w).2.1) :
    NetworkCertifier.Accepted (NetworkCertifier.constantRate T W ss frames) op k o (c, σ) ↔
      Accepted (constantRate (S := S) T W ss' frames') op' k o' (c, σ') := by
  refine code_accepted _ _ _ _ (fun t ins => ?_) op op' k o o' hlen hrows hstatus hop
  funext ℓ
  exact CodeExpectedRow S T W _ _ hσ _ _ (hf ins c ℓ) t

/-- **The program's ingress acceptance is the reference's.** For observation sequences of one length whose steps have
the same complete statuses, the program's `StatusComplete` (which the difftest decides) holds exactly when the
reference's does. -/
theorem CodeStatusComplete {L K S : ℕ} (k : ℕ) (o : List (NetworkCertifier.Obs L K S)) (o' : List (Obs L K S))
    (hlen : o.length = o'.length)
    (hstatus : ∀ t (h : t < o.length) (h' : t < o'.length) κ,
      o[t].status κ = .complete ↔ o'[t].status κ = .complete) :
    NetworkCertifier.StatusComplete k o ↔ StatusComplete k o' := by
  constructor
  · intro H t ht
    obtain ⟨h, hc⟩ := H t ht
    exact ⟨hlen ▸ h, fun κ => (hstatus t h _ κ).mp (hc κ)⟩
  · intro H t ht
    obtain ⟨h', hc⟩ := H t ht
    exact ⟨hlen ▸ h', fun κ => (hstatus t _ h' κ).mpr (hc κ)⟩

/-- **The program's sync sets are the reference's.** Every sync set of the program (two duplicate-free lists) has a
reference sync set (two `Finset`s) with the same δ and the same ρ, and every reference sync set one of the program's. -/
theorem CodeSyncSet :
    (∀ ss : NetworkCertifier.SyncSet, ∃ ss' : SyncSet, (∀ d, d ∈ ss.deltas ↔ d ∈ ss'.deltas) ∧ ∀ ρ, ρ ∈ ss.rhos ↔ ρ ∈ ss'.rhos) ∧
      ∀ ss' : SyncSet, ∃ ss : NetworkCertifier.SyncSet, (∀ d, d ∈ ss.deltas ↔ d ∈ ss'.deltas) ∧ ∀ ρ, ρ ∈ ss.rhos ↔ ρ ∈ ss'.rhos :=
  ⟨fun ss => ⟨syncSetRef ss, fun _ => List.mem_toFinset.symm, fun _ => List.mem_toFinset.symm⟩,
    fun ss' => ⟨syncSetCode ss', fun _ => Finset.mem_toList, fun _ => Finset.mem_toList⟩⟩

end NetTiming.SecurityProofs
