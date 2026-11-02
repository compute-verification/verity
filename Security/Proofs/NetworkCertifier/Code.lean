import Definitions.NetworkCertifier
import Definitions.NetworkCertifier.AdviceV1
import Definitions.NetworkCertifier.WindowCharge
import NetworkCertifier.Ingress
import NetworkCertifier.Schedule
import NetworkCertifier.AdviceV1
import NetworkCertifier.WindowCharge
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
  set one of the program's, so the program's membership test (the audit's `advice` verdict) is the reference's;
* `CodeChargesV1`: the program's advice v1 charges (`NetworkCertifier.AdviceV1.charges`, which the charge difftest runs
  against `arith_encode.reference_charge`) are the reference's (`AdviceV1.charges`), for books with the same fields;
  the program's `bitlen` is Lean core's `Nat.log2` plus one, the reference's `Nat.size`;
* `CodeChargesWhole`: the program's whole-window charges (`NetworkCertifier.WindowCharge.charges`, which the window
  difftest runs against `window_charge.reference_window`) are the reference's (`WindowCharge.charges`), for books with
  the same tables and priors and windows with the same fields.

Each statement relates the two sides by their fields rather than by a transport, so that it reads only the two sides'
definitions. The transports (`frameRef`, `syncRef`, `syncSetRef`, `syncSetCode`, `tallyRef`, `carryRef`, `tableRef`,
`bookRef`, `wcarryRef`, `windowRef`) are the proofs' own.
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
theorem code_accepted {L K S : ℕ} {In Adv Adv' : Type} (P : NetworkCertifier.EgressPrediction L S In Adv)
    (P' : Program L S In Adv') (a : Adv) (a' : Adv') (hP : ∀ t ins, P.rows t ins a = P'.rows t ins a')
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


/-- The reference form of the program's coder state. -/
def tallyRef (t : NetworkCertifier.AdviceV1.Tally) : AdviceV1.Tally := ⟨t.low, t.high, t.used, t.bits⟩

/-- The reference form of the program's carry. -/
def carryRef (c : NetworkCertifier.AdviceV1.Carry) : AdviceV1.Carry := ⟨c.counts, c.run⟩

theorem code_scale : ∀ k lo hi, NetworkCertifier.AdviceV1.scale k lo hi = AdviceV1.scale k lo hi
  | 0, _, _ => rfl
  | k + 1, lo, hi => by
    simp only [NetworkCertifier.AdviceV1.scale, AdviceV1.scale, code_scale k]
    rfl

theorem code_countStep (st : NetworkCertifier.AdviceV1.Tally) (cum freq total : ℕ) :
    (NetworkCertifier.AdviceV1.countStep st cum freq total).map tallyRef =
      AdviceV1.countStep (tallyRef st) cum freq total := by
  simp only [NetworkCertifier.AdviceV1.countStep, AdviceV1.countStep, code_scale, tallyRef]
  show (if 1 ≤ freq ∧ cum + freq ≤ total ∧ total ≤ AdviceV1.MAX_TOTAL then
      (AdviceV1.scale AdviceV1.K_MAX _ _).bind fun p =>
        if st.bits + p.2.2 < 2 ^ AdviceV1.CW then some (⟨p.1, p.2.1, st.used || freq != total, st.bits + p.2.2⟩ :
          NetworkCertifier.AdviceV1.Tally) else none
      else none).map tallyRef = _
  split
  · cases AdviceV1.scale AdviceV1.K_MAX _ _ with
    | none => rfl
    | some p =>
      rw [Option.bind_some, Option.bind_some, apply_ite (Option.map tallyRef)]
      rfl
  · rfl

theorem code_bitStep (st : NetworkCertifier.AdviceV1.Tally) (f0 f1 : ℕ) (b : Bool) :
    (NetworkCertifier.AdviceV1.bitStep st f0 f1 b).map tallyRef = AdviceV1.bitStep (tallyRef st) f0 f1 b := by
  simp only [NetworkCertifier.AdviceV1.bitStep, AdviceV1.bitStep]
  show (if 1 ≤ f0 ∧ 1 ≤ f1 ∧ f0 + f1 ≤ AdviceV1.MAX_TOTAL then _ else none).map tallyRef = _
  split
  · exact code_countStep _ _ _ _
  · rfl

theorem code_ktFreq : NetworkCertifier.AdviceV1.ktFreq = AdviceV1.ktFreq := rfl

theorem code_ktRaise : NetworkCertifier.AdviceV1.ktRaise = AdviceV1.ktRaise := rfl

theorem code_nextRun : NetworkCertifier.AdviceV1.nextRun = AdviceV1.nextRun := rfl

theorem code_decisions : ∀ (st : NetworkCertifier.AdviceV1.Tally) (cy : NetworkCertifier.AdviceV1.Carry) (ds : List Bool),
    (NetworkCertifier.AdviceV1.decisions st cy ds).map (fun p => (tallyRef p.1, carryRef p.2)) =
      AdviceV1.decisions (tallyRef st) (carryRef cy) ds
  | _, _, [] => rfl
  | st, cy, b :: ds => by
    simp only [NetworkCertifier.AdviceV1.decisions, AdviceV1.decisions, code_ktFreq]
    show ((AdviceV1.ktFreq cy.counts cy.run).bind _).map _ = (AdviceV1.ktFreq cy.counts cy.run).bind _
    cases AdviceV1.ktFreq cy.counts cy.run with
    | none => rfl
    | some f =>
      simp only [Option.bind_some, ← code_bitStep]
      cases NetworkCertifier.AdviceV1.bitStep st f.1 f.2 b with
      | none => rfl
      | some st' =>
        simp only [Option.bind_some, Option.map_some]
        exact code_decisions st' _ ds

theorem code_zeros : ∀ (st : NetworkCertifier.AdviceV1.Tally) (zs : List (ℕ × ℕ)),
    (NetworkCertifier.AdviceV1.zeros st zs).map tallyRef = AdviceV1.zeros (tallyRef st) zs
  | _, [] => rfl
  | st, z :: zs => by
    simp only [NetworkCertifier.AdviceV1.zeros, AdviceV1.zeros, ← code_countStep]
    cases NetworkCertifier.AdviceV1.countStep st 0 z.1 z.2 with
    | none => rfl
    | some st' => exact code_zeros st' zs

theorem code_finish (st : NetworkCertifier.AdviceV1.Tally) :
    NetworkCertifier.AdviceV1.finish st = AdviceV1.finish (tallyRef st) := rfl

theorem code_bitlen (n : ℕ) : NetworkCertifier.AdviceV1.bitlen n = n.size := by
  unfold NetworkCertifier.AdviceV1.bitlen
  split
  · subst_vars; rfl
  · rename_i h
    exact le_antisymm (Nat.lt_size.mpr (Nat.log2_self_le h)) (Nat.size_le.mpr Nat.lt_log2_self)

theorem code_arrive (bk : NetworkCertifier.AdviceV1.Book) (bk' : AdviceV1.Book) (harr : bk.arrived = bk'.arrived)
    (htot : bk.total = bk'.total) (n : ℕ) :
    (NetworkCertifier.AdviceV1.arrive bk n).map tallyRef = AdviceV1.arrive bk' n := by
  simp only [NetworkCertifier.AdviceV1.arrive, AdviceV1.arrive, code_bitlen, harr, htot]
  rw [show AdviceV1.Tally.fresh = tallyRef NetworkCertifier.AdviceV1.Tally.fresh from rfl, ← code_countStep]
  cases NetworkCertifier.AdviceV1.countStep NetworkCertifier.AdviceV1.Tally.fresh _ _ _ with
  | none => rfl
  | some st =>
    simp only [Option.bind_some, Option.map_some]
    split_ifs
    · exact code_countStep _ _ _ _
    · rfl

theorem code_window (bk : NetworkCertifier.AdviceV1.Book) (bk' : AdviceV1.Book) (harr : bk.arrived = bk'.arrived)
    (htot : bk.total = bk'.total) (hz : bk.zeros = bk'.zeros) (N : ℕ) (cy : NetworkCertifier.AdviceV1.Carry)
    (ds : List Bool) :
    (NetworkCertifier.AdviceV1.window bk N cy ds).map (fun p => (p.1, carryRef p.2)) =
      AdviceV1.window bk' N (carryRef cy) ds := by
  simp only [NetworkCertifier.AdviceV1.window, AdviceV1.window]
  show (if ds.length ≤ N ∧ cy.run < 3 then _ else none).map _ = if ds.length ≤ N ∧ cy.run < 3 then _ else none
  split_ifs
  · rw [← code_arrive bk bk' harr htot, ← hz]
    cases NetworkCertifier.AdviceV1.arrive bk ds.length with
    | none => rfl
    | some st =>
      simp only [Option.bind_some, Option.map_some]
      rw [← code_decisions]
      cases NetworkCertifier.AdviceV1.decisions st cy ds with
      | none => rfl
      | some p =>
        simp only [Option.bind_some, Option.map_some]
        rw [← code_zeros]
        cases NetworkCertifier.AdviceV1.zeros p.1 bk.zeros with
        | none => rfl
        | some st₂ =>
          simp only [Option.bind_some, Option.map_some, code_finish]
          cases AdviceV1.finish (tallyRef st₂) <;> rfl
  · rfl

theorem code_session (bk : NetworkCertifier.AdviceV1.Book) (bk' : AdviceV1.Book) (harr : bk.arrived = bk'.arrived)
    (htot : bk.total = bk'.total) (hz : bk.zeros = bk'.zeros) (N : ℕ) :
    ∀ (cy : NetworkCertifier.AdviceV1.Carry) (ws : List (List Bool)),
      NetworkCertifier.AdviceV1.session bk N cy ws = AdviceV1.session bk' N (carryRef cy) ws
  | _, [] => rfl
  | cy, ds :: ws => by
    simp only [NetworkCertifier.AdviceV1.session, AdviceV1.session, ← code_window bk bk' harr htot hz N cy ds]
    cases NetworkCertifier.AdviceV1.window bk N cy ds with
    | none => rfl
    | some p =>
      simp only [Option.bind_some, Option.map_some]
      rw [code_session bk bk' harr htot hz N p.2 ws]

/-- **Advice v1's charge in the program is the reference's** (`CodeChargesV1`): for books with the same fields, the
program's `charges` of the windows' decision streams, listed in order, is the reference's. -/
theorem CodeChargesV1 (bk : NetworkCertifier.AdviceV1.Book) (bk' : AdviceV1.Book) (harr : bk.arrived = bk'.arrived)
    (htot : bk.total = bk'.total) (hz : bk.zeros = bk'.zeros) (hp : bk.priors = bk'.priors) (N : ℕ) {W : ℕ}
    (a : Fin W → List Bool) :
    NetworkCertifier.AdviceV1.charges bk N (List.ofFn a) = AdviceV1.charges bk' N a := by
  rw [NetworkCertifier.AdviceV1.charges, AdviceV1.charges, code_session bk bk' harr htot hz N]
  simp only [carryRef, hp]

/-- The reference form of the program's table. -/
def tableRef (t : NetworkCertifier.WindowCharge.Table) : WindowCharge.Table := ⟨t.freqs, t.total⟩

/-- The reference form of the program's whole-window book. -/
def bookRef (bk : NetworkCertifier.WindowCharge.Book) : WindowCharge.Book :=
  ⟨tableRef bk.arrived, tableRef bk.nHits, tableRef bk.hit, tableRef bk.resumes, tableRef bk.nOrder, tableRef bk.gap,
    tableRef bk.index, tableRef bk.lag, bk.priors⟩

/-- The reference form of the program's whole-window carry. -/
def wcarryRef (c : NetworkCertifier.WindowCharge.Carry) : WindowCharge.Carry := ⟨carryRef c.dec, c.pos, c.rank⟩

/-- The reference form of the program's window. -/
def windowRef (x : NetworkCertifier.WindowCharge.Window) : WindowCharge.Window := ⟨x.ds, x.hits, x.order⟩

theorem code_codeAll : ∀ (st : NetworkCertifier.AdviceV1.Tally) (ss : List (ℕ × ℕ × ℕ)),
    (NetworkCertifier.WindowCharge.codeAll st ss).map tallyRef = WindowCharge.codeAll (tallyRef st) ss
  | _, [] => rfl
  | st, s :: ss => by
    simp only [NetworkCertifier.WindowCharge.codeAll, WindowCharge.codeAll, ← code_countStep]
    cases NetworkCertifier.AdviceV1.countStep st s.1 s.2.1 s.2.2 with
    | none => rfl
    | some st' => exact code_codeAll st' ss

theorem code_uniform (n : ℕ) : ∀ u, NetworkCertifier.WindowCharge.uniform u n = WindowCharge.uniform u n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro u
    rw [NetworkCertifier.WindowCharge.uniform, WindowCharge.uniform]
    show (if n ≤ AdviceV1.MAX_TOTAL then _ else _) = if n ≤ AdviceV1.MAX_TOTAL then _ else _
    by_cases h : n ≤ AdviceV1.MAX_TOTAL
    · rw [ite_eq_left h, ite_eq_left h]
    · have hlt : (n - 1) / 2 ^ NetworkCertifier.WindowCharge.CHUNK + 1 < n := by
        simp only [NetworkCertifier.WindowCharge.CHUNK, AdviceV1.MAX_TOTAL] at h ⊢
        omega
      rw [ite_eq_right h, ite_eq_right h, ih _ hlt]
      rfl

theorem code_low : NetworkCertifier.WindowCharge.low = WindowCharge.low := rfl

theorem code_zigzag : NetworkCertifier.WindowCharge.zigzag = WindowCharge.zigzag := rfl

theorem code_binned (t : NetworkCertifier.WindowCharge.Table) (hi v : ℕ) :
    NetworkCertifier.WindowCharge.binned t hi v = WindowCharge.binned (tableRef t) hi v := by
  simp only [NetworkCertifier.WindowCharge.binned, WindowCharge.binned, tableRef, code_bitlen, code_uniform, code_low]

theorem code_countHi : NetworkCertifier.WindowCharge.countHi = WindowCharge.countHi := by
  funext S
  simp only [NetworkCertifier.WindowCharge.countHi, WindowCharge.countHi, code_bitlen]

theorem code_decSyms : ∀ (counts : List (ℕ × ℕ)) (run : ℕ) (ds : List Bool),
    (NetworkCertifier.WindowCharge.decSyms ⟨counts, run⟩ ds).map (fun p => (p.1, carryRef p.2)) =
      WindowCharge.decSyms ⟨counts, run⟩ ds
  | _, _, [] => rfl
  | counts, run, b :: ds => by
    simp only [NetworkCertifier.WindowCharge.decSyms, WindowCharge.decSyms, code_ktFreq, code_ktRaise, code_nextRun]
    cases AdviceV1.ktFreq counts run with
    | none => rfl
    | some f =>
      simp only [Option.bind_some]
      show (if 1 ≤ f.1 ∧ 1 ≤ f.2 ∧ f.1 + f.2 ≤ AdviceV1.MAX_TOTAL then _ else none).map _ =
        if 1 ≤ f.1 ∧ 1 ≤ f.2 ∧ f.1 + f.2 ≤ AdviceV1.MAX_TOTAL then _ else none
      by_cases hc : 1 ≤ f.1 ∧ 1 ≤ f.2 ∧ f.1 + f.2 ≤ AdviceV1.MAX_TOTAL
      · rw [ite_eq_left hc, ite_eq_left hc, ← code_decSyms _ _ ds]
        cases NetworkCertifier.WindowCharge.decSyms _ ds <;> rfl
      · rw [ite_eq_right hc, ite_eq_right hc]
        rfl

theorem code_hitSyms (bk : NetworkCertifier.WindowCharge.Book) : ∀ ss : List ℕ,
    NetworkCertifier.WindowCharge.hitSyms bk ss = WindowCharge.hitSyms (bookRef bk) ss
  | [] => rfl
  | s :: ss => by
    simp only [NetworkCertifier.WindowCharge.hitSyms, WindowCharge.hitSyms, code_hitSyms bk ss, code_binned]
    rfl

theorem code_entrySyms (bk : NetworkCertifier.WindowCharge.Book) : ∀ (pos rank : ℕ) (es : List (ℕ × ℕ)),
    NetworkCertifier.WindowCharge.entrySyms bk pos rank es = WindowCharge.entrySyms (bookRef bk) pos rank es
  | _, _, [] => rfl
  | pos, rank, e :: es => by
    simp only [NetworkCertifier.WindowCharge.entrySyms, WindowCharge.entrySyms, code_entrySyms bk _ _ es, code_binned,
      code_zigzag]
    rfl

theorem code_syms (bk : NetworkCertifier.WindowCharge.Book) (N H O : ℕ) (cy : NetworkCertifier.WindowCharge.Carry)
    (x : NetworkCertifier.WindowCharge.Window) :
    (NetworkCertifier.WindowCharge.syms bk N H O cy x).map (fun q => (q.1, wcarryRef q.2)) =
      WindowCharge.syms (bookRef bk) N H O (wcarryRef cy) (windowRef x) := by
  obtain ⟨⟨c, r⟩, pos, rank⟩ := cy
  simp only [NetworkCertifier.WindowCharge.syms, WindowCharge.syms, wcarryRef, carryRef, windowRef]
  rw [← code_decSyms, ← code_hitSyms, ← code_entrySyms]
  cases NetworkCertifier.WindowCharge.decSyms ⟨c, r⟩ x.ds with
  | none => rfl
  | some d =>
    cases NetworkCertifier.WindowCharge.hitSyms bk x.hits with
    | none => rfl
    | some hs =>
      cases NetworkCertifier.WindowCharge.entrySyms bk pos rank x.order with
      | none => rfl
      | some e =>
        simp only [Option.map_some, Option.bind_some, code_binned, code_countHi]
        rfl

theorem code_wwindow (bk : NetworkCertifier.WindowCharge.Book) (N H O : ℕ) (cy : NetworkCertifier.WindowCharge.Carry)
    (x : NetworkCertifier.WindowCharge.Window) :
    (NetworkCertifier.WindowCharge.window bk N H O cy x).map (fun p => (p.1, wcarryRef p.2)) =
      WindowCharge.window (bookRef bk) N H O (wcarryRef cy) (windowRef x) := by
  simp only [NetworkCertifier.WindowCharge.window, WindowCharge.window]
  show (if x.ds.length ≤ N ∧ x.hits.length ≤ H ∧ x.order.length ≤ O ∧ cy.dec.run < 3 ∧
      cy.rank ≤ WindowCharge.INDEX_CAP + 1 then _ else none).map _ =
    if x.ds.length ≤ N ∧ x.hits.length ≤ H ∧ x.order.length ≤ O ∧ cy.dec.run < 3 ∧
      cy.rank ≤ WindowCharge.INDEX_CAP + 1 then _ else none
  split_ifs
  · rw [← code_syms]
    cases NetworkCertifier.WindowCharge.syms bk N H O cy x with
    | none => rfl
    | some q =>
      simp only [Option.bind_some, Option.map_some]
      rw [show AdviceV1.Tally.fresh = tallyRef NetworkCertifier.AdviceV1.Tally.fresh from rfl, ← code_codeAll]
      cases NetworkCertifier.WindowCharge.codeAll NetworkCertifier.AdviceV1.Tally.fresh q.1 with
      | none => rfl
      | some st =>
        simp only [Option.bind_some, Option.map_some, code_finish]
        cases AdviceV1.finish (tallyRef st) <;> rfl
  · rfl

theorem code_wsession (bk : NetworkCertifier.WindowCharge.Book) (N H O : ℕ) :
    ∀ (cy : NetworkCertifier.WindowCharge.Carry) (ws : List NetworkCertifier.WindowCharge.Window),
      NetworkCertifier.WindowCharge.session bk N H O cy ws =
        WindowCharge.session (bookRef bk) N H O (wcarryRef cy) (ws.map windowRef)
  | _, [] => rfl
  | cy, x :: ws => by
    simp only [NetworkCertifier.WindowCharge.session, WindowCharge.session, List.map_cons,
      ← code_wwindow bk N H O cy x]
    cases NetworkCertifier.WindowCharge.window bk N H O cy x with
    | none => rfl
    | some p =>
      simp only [Option.bind_some, Option.map_some]
      rw [code_wsession bk N H O p.2 ws]

/-- **Advice v1's whole window charged in the program is the reference's** (`CodeChargesWhole`): for books with the
same tables (each one's frequencies and total) and the same priors, and windows with the same decisions, hits and order
entries, the program's `charges` of the windows, listed in order, is the reference's. -/
theorem CodeChargesWhole (bk : NetworkCertifier.WindowCharge.Book) (bk' : WindowCharge.Book)
    (ht : [bk.arrived, bk.nHits, bk.hit, bk.resumes, bk.nOrder, bk.gap, bk.index, bk.lag].map (fun t => (t.freqs, t.total)) =
      [bk'.arrived, bk'.nHits, bk'.hit, bk'.resumes, bk'.nOrder, bk'.gap, bk'.index, bk'.lag].map
        (fun t => (t.freqs, t.total)))
    (hp : bk.priors = bk'.priors) (N H O : ℕ) {W : ℕ} (a : Fin W → NetworkCertifier.WindowCharge.Window)
    (a' : Fin W → WindowCharge.Window)
    (ha : ∀ w, (a w).ds = (a' w).ds ∧ (a w).hits = (a' w).hits ∧ (a w).order = (a' w).order) :
    NetworkCertifier.WindowCharge.charges bk N H O (List.ofFn a) = WindowCharge.charges bk' N H O a' := by
  have hb : bk' = bookRef bk := by
    obtain ⟨⟨_, _⟩, ⟨_, _⟩, ⟨_, _⟩, ⟨_, _⟩, ⟨_, _⟩, ⟨_, _⟩, ⟨_, _⟩, ⟨_, _⟩, _⟩ := bk'
    simp only [List.map_cons, List.map_nil, List.cons.injEq, Prod.mk.injEq, and_true] at ht
    obtain ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, rfl, rfl⟩ := ht
    dsimp only at hp
    subst hp
    rfl
  have hw : a' = fun w => windowRef (a w) := by
    funext w
    obtain ⟨h1, h2, h3⟩ := ha w
    rcases e : a' w with ⟨d, h, o⟩
    rw [e] at h1 h2 h3
    simp only [windowRef, h1, h2, h3]
  rw [NetworkCertifier.WindowCharge.charges, WindowCharge.charges, code_wsession, hb, hw, List.map_ofFn]
  rfl

end NetTiming.SecurityProofs
