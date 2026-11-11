module

public import Definitions.Claims.SamplingGenericX
public import Proofs.Flock.SamplingGeneric.Reg
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.Assemble

/-!
# The sampled compiler's committed form, at the trace read at the statement

`CompilerKSRegX` from the sessions' bound (`Compiled.SessionKS`): the committed form of `SamplingGeneric/Reg.lean`
against the incorrect set of the trace read at the statement (`Compiled.FregX` at `Compiled.regValX`, the plurality of
the extractions' openings over the prover's tape, the run's coins and the extractor's coins together). That trace reads
the prover and the statement and no tape, so nothing the prover's tape holds moves it or its incorrect set.

* `OffRegX`, `ConflictX`: a run's extractions open a gate of the program off the trace read at the statement; two
  runs' extractions, each at its own tape, open one gate two ways.
* `plurX`, `regValX_eq_plurX`: at a gate some run opens at some tape, the trace is the plurality's value.
* `pr_offRegX_le`: **the rewinding lemma** (`pr_off_le`) over the tapes and the runs together: a run opens a gate off
  the trace with probability at most `N·√Pr[conflict]`, the conflict between two runs at two independent tapes. The
  square root is of one probability, so no Jensen step over the tape follows it.
* `le_bconfX`, `pr_inConfX_le`, `pr_conflictX_le`: one run whose extractions open a position two ways, and two runs
  at two tapes that open a gate two ways, each give a collision `BconfX` finds (`bconfX_of_disagree`: both runs open
  under the statement's roots).
* `ex_offRegX_le`: the rewinding lemma averaged over the tape, at most `N·√Pr[B_conf,X]`.
* `ksRegX_of_sessionKS`, `compilerKSRegX_of`: **`KSRegX`**: an accepted run that samples a proof block of `FregX` has
  the first such block's session unfound at its first slot (the sessions' bound at `leadSel (selOf FregX)`), or its
  extractions disagree with themselves (`BconfX`'s first run) or open a gate off the trace (the rewinding lemma); and
  on the sample, `miss_of_event` at `FregX`.

What the per-tape form and this one share is stated once in `Reg.lean`, over the space or the trace: `pr_off_le`,
`trace_split`, `bconfX_of_disagree`, `selOf_some`, `pr_unfound_le` and `miss_of_event`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement InScope ownLog)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The events -/

/-- A run's extractions open a gate of the program off the trace read at the statement. -/
def OffRegX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : Prop :=
  ∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ L.regValX H P x g ≠ some b

/-- Two runs' extractions, each at its own tape, open a gate of the program two ways. -/
def ConflictX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))) : Prop :=
  ∃ g : Fin (L.nG x), (L.opensAt H P x qq.1.1 qq.1.2.1 qq.1.2.2 g).isSome ∧
    (L.opensAt H P x qq.2.1 qq.2.2.1 qq.2.2.2 g).isSome ∧
    L.opensAt H P x qq.1.1 qq.1.2.1 qq.1.2.2 g ≠ L.opensAt H P x qq.2.1 qq.2.2.1 qq.2.2.2 g

/-! ## The rewinding lemma at the trace read at the statement -/

/-- The value of the trace read at the statement at a gate some run opens (`plurAt` over the tapes and the runs). -/
noncomputable def plurX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (g : ℕ) : Bool :=
  plurAt (fun (q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x q.1 q.2.1 q.2.2 g) g

/-- At a gate some run's extractions open, at some tape, the trace read at the statement is the plurality's value. -/
theorem regValX_eq_plurX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    {ω : Ω} {r : Fin (L.V.k x) → Coin} {c : L.SlotEC x} {g : ℕ} {b : Bool}
    (h : L.opensAt H P x ω r c g = some b) : L.regValX H P x g = some (L.plurX H P x g) := by
  have hpos := Pr_pos_of (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
    L.opensAt H P x q.1 q.2.1 q.2.2 g = some b) (ω, (r, c)) h
  simp only [regValX, plurX, plurAt]
  split_ifs with h0
  · cases b
    · exact absurd h0.2 hpos.ne'
    · exact absurd h0.1 hpos.ne'
  · exact congrArg some (decide_eq_decide.2 Iff.rfl)

/-- **The rewinding lemma at the trace read at the statement**: over the tape, the run's coins and the extractor's
coins, a run's extractions open a gate of the program off the trace with probability at most `N·√Pr[conflict]`, `N` the
program's gates and the conflict between two runs at two independent tapes (`pr_off_le` at every tape's runs' openings,
the trace being their plurality). -/
theorem pr_offRegX_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      L.nG x * Real.sqrt (Pr fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
        (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.ConflictX H P x qq) :=
  pr_off_le (L.nG x)
    (fun (q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x q.1 q.2.1 q.2.2 g)
    (L.regValX H P x) fun _ _ _ h => L.regValX_eq_plurX H P x h

/-! ## Binding: a conflict between two tapes is `BconfX`'s collision -/

/-- An event of two tapes and a run at each that gives a collision `BconfX`'s output is at most `Pr[BconfX]`. -/
theorem le_bconfX (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {Q : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) → Prop}
    (hQ : ∀ q, Q q → ∃ u v, findColl (ownLog H (run H P L.V x q.1.1)
        (L.runExt x s q.1.2.1 (run H P L.V x q.1.1 q.1.2.1).seen q.1.2.2) ++
      ownLog H (run H P L.V x q.2.1) (L.runExt x s q.2.2.1 (run H P L.V x q.2.1 q.2.2.1).seen q.2.2.2)) =
        some (u, v) ∧ Collision H u v) :
    Pr Q ≤ (BconfX H L Ω P x).Pr H := by
  simp only [BconfX, hs]
  exact Pr_mono hQ

/-- **One run's disagreement**: at most `Pr[BconfX]`, its first run. -/
theorem pr_inConfX_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) ≤
      (BconfX H L Ω P x).Pr H := by
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      = Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.InConf H P x s q.1 q.2.1 q.2.2) :=
        (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.InConf H P x s q.1 q.2.1 q.2.2)).symm
    _ = Pr (fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
          L.InConf H P x s qq.1.1 qq.1.2.1 qq.1.2.2) :=
        (Pr_fst (β := Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))
          (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.InConf H P x s q.1 q.2.1 q.2.2)).symm
    _ ≤ (BconfX H L Ω P x).Pr H :=
        le_bconfX H L hs P fun q hq => by
          obtain ⟨a, ha, b, hb, h1, h2, h3⟩ := hq
          exact bconfX_of_disagree hbind s P q.1.1 q.2.1 q.1.2 q.2.2 (Or.inl ha) (Or.inl hb) h1 h2 h3

/-- **Two runs' conflict, at two tapes**: at most `Pr[BconfX]`. Both runs open under the statement's roots, so two
openings of one gate that differ are a collision whichever tapes opened them. -/
theorem pr_conflictX_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      L.ConflictX H P x qq) ≤ (BconfX H L Ω P x).Pr H := by
  refine le_bconfX H L hs P fun q hq => ?_
  obtain ⟨g, h1, h2, hne⟩ := hq
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  rw [L.opensAt_eq H P x q.1.1 hs] at hb1
  rw [L.opensAt_eq H P x q.2.1 hs] at hb2
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := val_some_mem hb1
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := val_some_mem hb2
  exact bconfX_of_disagree hbind s P q.1.1 q.2.1 q.1.2 q.2.2 (Or.inl ha) (Or.inr ha') (ha1.trans ha1'.symm)
    (ha2.trans ha2'.symm) (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))

/-- **The rewinding lemma at the trace read at the statement, averaged over the tape**: at most `N·√Pr[B_conf,X]`. The
average over the tape is the probability over the tape and the run together, which the lemma bounds as one. -/
theorem ex_offRegX_le {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2) ≤
      L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) :=
  calc Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2)
      = Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) :=
        (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          L.OffRegX H P x q.1 q.2.1 q.2.2)).symm
    _ ≤ L.nG x * Real.sqrt (Pr fun qq : (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
          (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.ConflictX H P x qq) := L.pr_offRegX_le H P x
    _ ≤ L.nG x * Real.sqrt ((BconfX H L Ω P x).Pr H) :=
        mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (pr_conflictX_le hbind hs P)) (Nat.cast_nonneg _)

/-! ## The committed form -/

set_option synthInstance.maxSize 1024 in
/-- **`KSRegX` from the sessions' bound**: an accepted run that samples a proof block of `FregX` has the first such
block's session unfound at its first slot (the sessions' bound at `leadSel (selOf FregX)`, a selection that reads no
tape), or its extractions disagree with themselves (`BconfX`'s first run) or open a gate off the trace read at the
statement (the rewinding lemma over the tapes and the runs, its conflict `BconfX`'s collision); and a set refused at
`δ` is missed with probability at most `P_a ≤ δ` (`miss_of_event`). -/
theorem ksRegX_of_sessionKS {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) : L.KSRegX H := by
  intro x hsc Ω _ _ P
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hpart : L.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hFlt : ∀ v ∈ L.FregX H P x, v < L.nPU x := by
    intro v hv
    simp only [FregX, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hFno : ∀ v ∈ L.FregX H P x, ¬ ∃ b, Pt.blocks[v]? = some b ∧ ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
      (∀ g < s.prog.gates.length, val g = L.regValX H P x g) ∧ s.prog.Correct (s.prog.refIn b val ι) b := by
    intro v hv
    simp only [FregX, hs, mem_filter, mem_range] at hv
    rw [hpart] at hv
    exact hv.2
  have hper : ∀ ω, Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.FregX H P x) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (L.FregX H P x)) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2)) := by
    intro ω
    have hIO := Sampling.Pr_union_le
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffRegX H P x ω rc.1 rc.2)
    refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.FregX H P x)).symm.trans_le (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
    obtain ⟨hacc, v, hvl, hvF⟩ := hrc
    obtain ⟨i, hi, u, hiu, huF⟩ := L.selOf_some x (hS.slots x rc.1) hFlt hvl hvF
    obtain ⟨i₀, u₀, hlead, hu₀⟩ := leadOf_some hiu u.isLt
    rcases trace_split hS hs hPt hin P ω rc.1 rc.2 hacc i u hiu (L.regValX H P x) (hFno _ huF) i₀ u₀ hlead hu₀
      with h | h | h
    · exact Or.inl ⟨hacc, i₀, by rw [leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)
  have hT := L.pr_unfound_le H P β hβ0 hβ (fun _ => L.leadSel x (L.selOf x (L.FregX H P x)))
    fun ω r => hκ ω r (L.leadSel x (L.selOf x (L.FregX H P x)))
  have hI := pr_inConfX_le hbind hs P
  have hO := ex_offRegX_le hbind hs P
  have h1 : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregX H P x) ≤ L.εregX H x P := by
    refine (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono hper).trans ?_)
    rw [Ex_add, Ex_add]
    unfold Compiled.εregX
    linarith
  refine ⟨h1, fun s' hs' δ hδ => ?_⟩
  rw [show s' = s from Option.some.inj (hs'.symm.trans hs)]
  exact miss_of_event hS hs hPt hin P (fun _ => L.FregX H P x) (fun _ => hFlt) h1 hδ

end Compiled

/-- **The committed form at the trace read at the statement** from the sessions' bound. -/
theorem compilerKSRegX_of (hsess : SessionKSOf) : CompilerKSRegX := by
  intro Coin _ _ _ A L H hA hB hS
  exact Compiled.ksRegX_of_sessionKS hB hS (hsess Coin A L H hA hS)

end Claims.SamplingGeneric
