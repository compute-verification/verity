module

public import Mathlib.Data.Set.Card
public import Definitions.NetworkCertifier.Schedule
public import Properties.NetworkCertifier.Check

/-!
# What a window's sync reaches

Two statements about the syncs a link declares, neither about a device's code. The grid the network certificate check
(`NetworkCheck.Check.protocol`) expects of an egress link in a window reads only the syncs the link declares up to that
window (`ExpectedGridCausal`). So the reference's constant-rate Program, each frame sent at the program's hand tick
under the syncs its advice names (`programFrames`), accepts over its first `m` windows a count of sequences that the
content advice, those windows' syncs and the jitter classes fix (`EncardProgramRunsLeWindows`).

`Proofs.NetworkCertifier.Causal` proves both, and the second's honest and nontrivial lemmas at the one-window instance
below.
-/

@[expose] public section

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync)

namespace Properties.NetworkCertifier

/-! ## The reference Program's frames -/

/-- A value of Σ_sync: a δ of its list and a ρ of its list. -/
abbrev SyncValue (ss : NetworkCertifier.SyncSet) : Type := {d // d ∈ ss.deltas.toFinset} × {ρ // ρ ∈ ss.rhos.toFinset}

/-- The clock sync a value of Σ_sync names. -/
def SyncValue.sync {ss : NetworkCertifier.SyncSet} (v : SyncValue ss) : ClockSync :=
  ⟨v.1.1, v.2.1, ss.rhos_pos _ (List.mem_toFinset.mp v.2.2)⟩

/-- The reference's frames of one execution block of `W` windows of `T` buckets, under the advice `(a, σ)`: the frames
`frames` opens from the ingress and the content advice `a`, in order, each sent at the program's hand tick
(`NetworkCertifier.handingBuckets`) under the syncs `σ` names. -/
def programFrames (T W : Nat) (ss : NetworkCertifier.SyncSet) {In Ac : Type}
    (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame) :
    List In → Fin 1 → Ac × (Fin W → SyncValue ss) → Fin 1 → List NetTiming.LogicalFrame :=
  fun ins _ v ℓ =>
    ((frames ins v.1 ℓ).zip (NetworkCertifier.handingBuckets T W (fun w => (v.2 w).sync)
      (frames ins v.1 ℓ))).map fun p => ⟨p.1.payload, p.2⟩

/-! ## What a window's sync reaches -/

/-- **A later window's sync never changes an earlier window's grid.** The grid the check expects of an egress link in
window `w` (`expectedGrid`, whose `record_head` digest `egressVerdict` compares) reads only the syncs the link declares
for windows `0 … w`: two links with the same frames whose syncs agree there (`early`, window `w` among them) get the
same grid in window `w`, whatever each declares after it (`later`, `later'`: any clock-sync values, in Σ_sync or not).
A frame is eligible no earlier than its anchor, its hand tick (`NetworkCertifier.handingBuckets`) is never before the
one before it, and the queue carries forward, so a window's sync moves only hand ticks at or after the window's start. -/
def ExpectedGridCausal : Prop :=
  ∀ (g : NetworkCheck.Check.Grid) (W : Nat) (frames : List NetworkCertifier.LogicalFrame)
    (early later later' : Array ClockSync) (w : Fin early.size),
    expectedGrid g W ⟨frames, early ++ later⟩ w = expectedGrid g W ⟨frames, early ++ later'⟩ w

/-- **The program's runs over its first `m` windows (Theorem 3 for the program, window by window).** The reference's
constant-rate Program of one execution block of `W` windows of `T` buckets at allocation `S`, each frame sent at the
program's hand tick under the syncs its advice names (`programFrames`), accepts at most
`#Ac · |Σ_sync|^min(W, m) · D^(m·T)` sequences of the first `m` windows' `m·T` buckets with jitter classes below `D`,
`|Σ_sync|` the count of its δs times the count of its ρs: a window's sync changes no bucket before the window
(`ExpectedGridCausal`), so the content advice and the first `m` windows' syncs fix those buckets' rows. The syncs are
`log2 |Σ_sync|` bits per window opened, 7 at 16 × 8, counted from the run's first window. A window on its own, given
the run before it, may carry more: the queue and the hand ticks carry across windows. -/
def EncardProgramRunsLeWindows : Prop :=
  ∀ (S T W : Nat) (ss : NetworkCertifier.SyncSet) {K : Nat} {In Ac : Type} [Fintype Ac]
    (frames : List In → Ac → Fin 1 → List NetworkCertifier.LogicalFrame)
    (op : Nat → List (NetTiming.Obs 1 K S) → In) (m D : Nat),
    (NetTiming.AccSeqs (NetTiming.constantRate (S := S) T W 1 (programFrames T W ss frames)) op (m * T)
        D).encard ≤
      ((Fintype.card Ac * (ss.deltas.length * ss.rhos.length) ^ min W m * D ^ (m * T) : ℕ) : ℕ∞)

end Properties.NetworkCertifier

/-! ## A one-window instance

One window of one bucket at allocation 1, with one sync value, and the honest network certifier's record that
`EncardProgramRunsLeWindows`'s honest lemma counts there. -/

namespace Properties.CertifierNetwork.Instance

/-- Σ_sync of one value: δ = 0 and ρ = 1. -/
def syncSet : _root_.NetworkCertifier.SyncSet := ⟨[0], [1], by simp, by simp, by decide, by simp, by simp⟩

/-- One frame of content `[true]`, anchored at bucket 0. -/
def oneFrame : List Unit → Unit → Fin 1 → List _root_.NetworkCertifier.LogicalFrame := fun _ _ _ => [⟨[true], 0, 0⟩]

/-- The advice of the one window: the content `()` and Σ_sync's one value. -/
def oneAdvice : Fin 1 → Unit × (Fin 1 → Properties.NetworkCertifier.SyncValue syncSet) :=
  fun _ => ((), fun _ => (⟨0, List.mem_toFinset.mpr (List.mem_singleton_self 0)⟩,
    ⟨1, List.mem_toFinset.mpr (List.mem_singleton_self 1)⟩))

/-- The honest network certifier's record of the one bucket: the Program's row of `oneFrame` under `oneAdvice`,
jitter class 0, its boundary link complete. -/
def honestObs : List (NetTiming.Obs 1 1 1) :=
  [⟨(NetTiming.constantRate (S := 1) 1 1 1 (Properties.NetworkCertifier.programFrames 1 1 syncSet oneFrame)).rows 0
      [()] oneAdvice, 0,
    fun _ => .complete⟩]

end Properties.CertifierNetwork.Instance
