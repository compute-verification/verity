module

public import Proofs.Flock.SamplingGeneric.TwoRegFirstHit
public import Proofs.Core.SamplingGeneric.FirstHitClass
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Wait
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.SamplingGeneric.Two

/-!
# The first-hit finder at two stages, by class of gates

The by-position finders (`Compiled.finderXAt`, `CompiledTwo.finderIAt`) pay, at a further run, one run with the
sessions that can open a gate extracted (`callsBy`) for each gate they still wait for, so the event of `KSRegTwoX` is
bounded with `4·E[c + Σ_{g ∈ sample} c_g]/2^256`: one term for each gate the proof blocks of a run's sample need.

One extraction of a session opens every gate its openings reach, and a finder that waits for several of them needs
it once. Here the gates fall in classes: `cls` sends a gate to its class's representative gate, and a run's
extractions open a class whole or not at all, under the statement's roots (`hallX`) and under a replay block's
interior roots (`hallI`). The finders that pay once for each class of gates they still wait for
(`Compiled.finderXCls`, `CompiledTwo.finderICls`) give

  `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256`

for the event of `KSRegTwoX` (`CompiledTwo.eventTwoX_le_firstHit_cls`, and `_cls_cap` at one cap for each finder):
`c` a run's calls with every session of its sample extracted (`callsAt`), the sum over the representatives of the
classes of the gates the proof blocks of the run's sample need, and `c_q` the run's calls with only the sessions that
have a proof block needing the representative `q` extracted (`callsBy`).

* **The two hypotheses on the classes.** `hallX` and `hallI` are statements about what a run's extractions open
  (`Compiled.opensAt`, `CompiledTwo.iopTable`), for every prover: a run whose extractions open a gate open every gate
  of its class.
* **From what the extractions keep.** Both follow where what a run's extractions keep holds classes whole
  (`Compiled.ClassWhole`: an opening they keep comes with one they keep, under the same roots, at each gate of its
  class; `Compiled.opensAt_isSome_cls`), and, for the interiors, a class lies in one replay block
  (`CompiledTwo.iopTable_isSome_cls`): `CompiledTwo.eventTwoX_le_firstHit_kept`.
* **From what the extracted witnesses reveal.** The extractions keep classes whole where what each witness they
  extract reveals at each proof block of its session does (`Compiled.kept_classWhole_of_wit`,
  `CompiledTwo.eventTwoX_le_firstHit_wit`). That is for the code that instantiates `reveal`, `commit` and the sessions
  to show; a witness no extraction returns is not asked about.
* **At the classes of a key** (`CompiledTwo.eventTwoX_le_firstHit_key`): where the classes are those of a key of the
  gates (`Verity.FirstHit.clsOf`), and gates of one key are needed by the same proof blocks and lie in the same replay
  block, the hypotheses on the representatives hold by the key.
* **What the finders read.** The seed, the sample, the waits and the output are the by-position finders'. At a further
  run the output reads the extraction of the sessions with a proof block needing the gate it compares
  (`Compiled.collBy`), and the finder is charged the sessions with a proof block needing that gate's representative.
  The charge covers what is read where every proof block that needs a gate needs its representative (`hrep`, which the
  proofs don't read; `Compiled.callsBy_le_of_needs`).
* **The premises** are `eventTwoX_le_firstHit_at`'s with the linear expected-time bound (`ExpectedCR`) at these two
  finders in place of the two by-position ones. No statement of the definitions tree changes.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo ownLog forkOn wrongAllAt wrongAllAt_subset gateBlock flatMapRP)

/-- Filtering what a weaker filter keeps by a stronger one is filtering by the stronger one. -/
theorem filter_filter_of_imp {β : Type} (k1 k2 : β → Bool) (h : ∀ b, k1 b = true → k2 b = true) (l : List β) :
    (l.filter k2).filter k1 = l.filter k1 := by
  rw [List.filter_filter]
  refine List.filter_congr fun b _ => ?_
  cases h1 : k1 b with
  | false => rfl
  | true => rw [h b h1]; rfl

/-- Over the part of a list a filter keeps, a rerun program calls `H` at most as often as over the part a weaker
filter keeps. -/
theorem flatMapRP_hashes_filter_le_of_imp {R β γ : Type} (H : Msg → Msg) (run : R → Transcript)
    (f : β → RP R (List γ)) (k1 k2 : β → Bool) (h : ∀ b, k1 b = true → k2 b = true) (l : List β) :
    ((flatMapRP f (l.filter k1)).eval H run).2.1 ≤ ((flatMapRP f (l.filter k2)).eval H run).2.1 := by
  rw [← filter_filter_of_imp k1 k2 h l]
  exact flatMapRP_hashes_filter_le H run f k1 (l.filter k2)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## A run's extractions keep a class whole -/

/-- **A list of openings holds classes whole**: an opening of it that passes its opening check has, for each gate of
its gate's class, an opening in the list, under the same roots, that passes too. `cls` sends a gate of the program to
its class's representative. -/
def ClassWhole (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (cls : Fin (L.nG x) → Fin (L.nG x))
    (rs : List (List Msg × Sampling.Opening Bool)) : Prop :=
  ∀ p ∈ rs, L.commit.Opens H p.1 p.2 = true → ∀ g g' : Fin (L.nG x), p.2.pos = g.val → cls g = cls g' →
    ∃ p' ∈ rs, p'.1 = p.1 ∧ p'.2.pos = g'.val ∧ L.commit.Opens H p'.1 p'.2 = true

/-- What a session's extraction keeps is an opening of what its witness reveals at one of the session's proof blocks,
which passes its opening check. -/
theorem sessExt_reveal_of_mem (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {s : AuditStatement}
    {run' : (Fin (L.V.k x) → Coin) → Transcript} {us : List ℕ} {u : ℕ} {r : Fin (L.V.k x) → Coin} {v : Seen}
    {c : A.EC (L.sh x u)} {q : List Msg × Sampling.Opening Bool}
    (hq : q ∈ ((L.sessExt x s us u r v c).eval H run').1) :
    ∃ w, ((L.sessWit x u r v c).eval H run').1 = some w ∧
      ∃ u' ∈ us, q ∈ L.reveal s u' v w ∧ L.commit.Opens H q.1 q.2 = true := by
  unfold sessExt at hq
  rw [(Claims.RP.eval_bind H run' _ _).1] at hq
  cases hw : ((L.sessWit x u r v c).eval H run').1 with
  | none =>
    rw [hw] at hq
    simp [RP.eval] at hq
  | some w =>
    rw [hw] at hq
    have hq' : q ∈ ((flatMapRP (fun u' => Sampling.checkRevealed (R := Fin (L.V.k x) → Coin) L.commit
        (L.reveal s u' v w)) us).eval H run').1 := hq
    rw [Sampling.flatMapRP_eval] at hq'
    obtain ⟨u', hu'm, hu'⟩ := List.mem_flatMap.1 hq'
    obtain ⟨hmem, hok, -⟩ := Sampling.checkRevealed_mem H run' L.commit _ q hu'
    exact ⟨w, rfl, u', hu'm, hmem, hok⟩

/-- An opening that the session's witness reveals at one of the session's proof blocks, and that passes its opening
check, is kept by the session's extraction. -/
theorem mem_sessExt_of_reveal (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {s : AuditStatement}
    {run' : (Fin (L.V.k x) → Coin) → Transcript} {us : List ℕ} {u : ℕ} {r : Fin (L.V.k x) → Coin} {v : Seen}
    {c : A.EC (L.sh x u)} {w : A.W} (hw : ((L.sessWit x u r v c).eval H run').1 = some w) {u' : ℕ} (hu' : u' ∈ us)
    {q : List Msg × Sampling.Opening Bool} (hq : q ∈ L.reveal s u' v w) (hok : L.commit.Opens H q.1 q.2 = true) :
    q ∈ ((L.sessExt x s us u r v c).eval H run').1 := by
  have h : q ∈ ((flatMapRP (fun u' => Sampling.checkRevealed (R := Fin (L.V.k x) → Coin) L.commit
      (L.reveal s u' v w)) us).eval H run').1 := by
    rw [Sampling.flatMapRP_eval]
    exact List.mem_flatMap.2 ⟨u', hu', Sampling.mem_checkRevealed H run' L.commit hq hok⟩
  unfold sessExt
  rw [(Claims.RP.eval_bind H run' _ _).1, hw]
  exact h

/-- What a session's extraction keeps at a first slot of the run's sample is among what the run's extractions keep. -/
theorem mem_kept_of_sessExt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) {iu : Fin (L.nPU x) × ℕ}
    (hiu : iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r))
    (hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val)
    {q : List Msg × Sampling.Opening Bool}
    (hq : q ∈ ((L.sessExt x s (L.sessBlocks x (L.checked x r) iu.1) iu.2 r (run H P L.V x ω r).seen
      (c iu.1 ⟨iu.2, hu.1⟩)).eval H (run H P L.V x ω)).1) :
    q ∈ L.kept H P x s ω r c := by
  unfold kept
  unfold runExt
  rw [Sampling.flatMapRP_eval]
  refine List.mem_flatMap.2 ⟨iu, hiu, ?_⟩
  split
  · exact hq
  · rename_i hn
    exact absurd hu hn

/-- **A run's extractions keep classes whole, where the witnesses they extract reveal classes whole**: for each
session of the run's sample, extracted at its first slot, each witness `w` its extraction returns and each proof block
`u'` of the session, what `w` reveals at `u'` holds classes whole (`hwit`). -/
theorem kept_classWhole_of_wit (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (cls : Fin (L.nG x) → Fin (L.nG x)) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    (hwit : ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
      ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val, ∀ w : A.W,
        ((L.sessWit x iu.2 r (run H P L.V x ω r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval H (run H P L.V x ω)).1 = some w →
          ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
            L.ClassWhole H x cls (L.reveal s u' (run H P L.V x ω r).seen w)) :
    L.ClassWhole H x cls (L.kept H P x s ω r c) := by
  intro q hq _ g g' hg hgg
  have hq0 := hq
  unfold kept at hq0
  unfold runExt at hq0
  rw [Sampling.flatMapRP_eval] at hq0
  obtain ⟨iu, hium, hiu⟩ := List.mem_flatMap.1 hq0
  split at hiu
  · rename_i hu
    obtain ⟨w, hw, u', hu', hmem, hok⟩ := L.sessExt_reveal_of_mem H hiu
    obtain ⟨p', hp', h1, h2, h3⟩ := hwit iu hium hu w hw u' hu' q hmem hok g g' hg hgg
    exact ⟨p', L.mem_kept_of_sessExt H P x s ω r c hium hu (L.mem_sessExt_of_reveal H hw hu' hp' h3), h1, h2, h3⟩
  · simp [RP.eval] at hiu

/-- A kept opening under the statement's roots that passes opens its gate. -/
theorem opensAt_isSome_of_kept (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    {q : List Msg × Sampling.Opening Bool} (hq : q ∈ L.kept H P x s ω r c) (h1 : q.1 = s.roots)
    (hok : L.commit.Opens H s.roots q.2 = true) : (L.opensAt H P x ω r c q.2.pos).isSome := by
  rw [L.opensAt_eq H P x ω hs]
  unfold Sampling.Wit.val
  rw [Option.isSome_map]
  refine List.find?_isSome.2 ⟨q.2, ?_, by simp [hok]⟩
  exact List.mem_filterMap.2 ⟨q, hq, by simp [h1]⟩

/-- **A run's extractions open a class whole under the statement's roots**, where they keep classes whole: the
by-class finder's hypothesis at the boundary (`hallX`). -/
theorem opensAt_isSome_cls (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (cls : Fin (L.nG x) → Fin (L.nG x))
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (hkept : L.ClassWhole H x cls (L.kept H P x s ω r c))
    (g g' : Fin (L.nG x)) (hgg : cls g = cls g') (hg : (L.opensAt H P x ω r c g).isSome) :
    (L.opensAt H P x ω r c g').isSome := by
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hg
  rw [L.opensAt_eq H P x ω hs] at hb
  obtain ⟨q, hq, h1, h2, -, hok⟩ := val_some_mem hb
  rw [← h1] at hok
  obtain ⟨q', hq', h1', h2', hok'⟩ := hkept q hq hok g g' h2 hgg
  have hr : q'.1 = s.roots := h1'.trans h1
  rw [hr] at hok'
  have h := L.opensAt_isSome_of_kept H P x ω hs r c hq' hr hok'
  rw [h2'] at h
  exact h

/-- **A run's calls at one gate are at most its calls at a gate that every proof block needing the first needs**: the
sessions extracted for the first are among those extracted for the second. -/
theorem callsBy_le_of_needs (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) {g g' : ℕ} (h : ∀ u, g ∈ needs u → g' ∈ needs u) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) :
    L.callsBy H P x s needs g ω r c ≤ L.callsBy H P x s needs g' ω r c := by
  unfold callsBy runExtAt
  refine Nat.add_le_add_left (flatMapRP_hashes_filter_le_of_imp H _ _ _ _ ?_ _) _
  intro iu hiu
  simp only [decide_eq_true_eq] at hiu ⊢
  obtain ⟨u, hu, hg⟩ := hiu
  exact ⟨u, hu, h u hg⟩

/-! ## The boundary, by class -/

/-- **The by-class first-hit finder at the boundary trace read at the statement**, at cap `K`. Its seed, its sample,
its waits, its first run and its output are `finderXAt`'s. At a further run it pays, for each class that holds a gate
still waited for in the run's sample, a run at that tape and those coins with only the sessions that have a proof
block needing the class's representative extracted (`callsBy`). -/
noncomputable def finderXCls (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (cls : Fin (L.nG x) → Fin (L.nG x)) (K : ℕ) : Finder :=
  Verity.FirstHit.classFinderAt cls
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (q : Fin (L.nG x)) =>
      L.callsBy H P x s needs q.val a.1 a.2.1 a.2.2) K
    fun q : Unit × ((Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
        (Fin K → Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))) =>
      Verity.FirstHit.outOfAt
        (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
        (fun (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
          L.collBy H P x s needs a a' g.val) K q.2.1 q.2.2

/-- **The mass off the boundary trace read at the statement, by class, at one cap**:
`2·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256 + N/(K + 1)`, under `ExpectedCR` at the by-class finder of the cap
`K`, where a run's extractions open a class of gates whole or not at all under the statement's roots (`hallX`): `c` a
run's calls with every session extracted (`callsAt`), the sum over the representatives of the classes of the gates
the run's sample needs, `c_q` the run's calls with only the sessions that have a proof block needing `q` extracted
(`callsBy`), `N` the program's gates. -/
theorem pr_offRegX_le_cls_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (K : ℕ) (hCR : ExpectedCR H (L.finderXCls H P x s needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
              L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_cls_le_ex_cap H cls
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (vstar := fun (_ : Unit) (g : Fin (L.nG x)) => L.plurX H P x g)
    (fun _ => isPlurality_plurAt (L.nG x)
      fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun _ a g hg => decide_eq_true (L.opensAt_isSome_needs H P x a.1 hs needs hpos a.2.1 a.2.2 g hg))
    (fun _ a g g' hgg hg => hallX a g g' hgg hg)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (q : Fin (L.nG x)) =>
      L.callsBy H P x s needs q.val a.1 a.2.1 a.2.2) K
    (fun (_ : Unit) (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.collBy H P x s needs a a' g.val)
    (fun _ a a' g hg => by
      obtain ⟨h1, h2, hne⟩ := hg
      exact collBy_of_conflictAt hbind hs P needs (fun u v w p hp _ => hpos u v w p hp) a a' g.val h1 h2 hne)
    (fun _ a a' g u v huv => L.collBy_collision H P x s needs a a' g.val huv) hCR
  rw [Fintype.card_fin] at h
  have hP := Pr_unit_prod (fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
    FlockSoundness.Rewinding.Off
      (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
      (fun g : Fin (L.nG x) => L.plurX H P x g) a g)
  have hE := Ex_unit_prod (fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
    ((L.callsAt H P x s a.1 a.2.1 a.2.2 +
      ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))).image cls,
        L.callsBy H P x s needs c.val a.1 a.2.1 a.2.2 : ℕ) : ℝ))
  have hoff : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      Pr (fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off
          (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
            L.opensAt H P x a.1 a.2.1 a.2.2 g)
          (fun g : Fin (L.nG x) => L.plurX H P x g) a g) :=
    Pr_mono fun q hq => by
      obtain ⟨g, hg, b, hb, hne⟩ := hq
      refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
      rw [L.regValX_eq_plurX H P x hb]
      exact (hb.symm.trans h).symm
  exact (hoff.trans (hP.symm.le.trans h)).trans_eq
    (congrArg (fun e : ℝ => 2 * e / 2 ^ (256 : ℝ) + (L.nG x : ℝ) / ((K : ℝ) + 1)) hE)

/-- **The by-class expectation is at most the by-position one**, where every proof block that needs a gate needs its
class's representative: each class of the sample's gates is paid once, at its representative, which is a gate of the
sample. -/
theorem ex_callsCls_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
              decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
            L.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  Ex_mono fun q => Nat.cast_le.2 (Nat.add_le_add_left (Finset.sum_le_sum_of_subset fun c hc => by
    obtain ⟨g, hg, rfl⟩ := Finset.mem_image.1 hc
    obtain ⟨u, hu, hgu⟩ := of_decide_eq_true (Finset.mem_filter.1 hg).2
    exact Finset.mem_filter.2 ⟨Finset.mem_univ _, decide_eq_true ⟨u, hu, hrep u g hgu⟩⟩) _)

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The interior, by class -/

/-- **The by-class first-hit finder at the committed interiors**, at cap `K`. Its seed, its sample, its waits, its
first run and its output are `finderIAt`'s. At a further run, at the kept tape and first stage, it pays, for each
class that holds a gate still waited for in the run's sample, a run at those coins with only the sessions that have a
proof block needing the class's representative extracted (`callsBy`). -/
noncomputable def finderICls (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (Pρ : Partition) (needs : ℕ → Finset ℕ)
    (cls : Fin (L.nG x) → Fin (L.nG x)) (K : ℕ) : Finder :=
  Verity.FirstHit.classFinderAt cls
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (q : Fin (L.nG x)) =>
      L.toCompiled.callsBy H P x s needs q.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    fun q : (Ω × (Fin (L.V.k x) → Coin)) × (((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      Verity.FirstHit.outOfAt (L.iopTable H P x q.1.1 Pρ q.1.2)
        (fun (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
          L.toCompiled.collBy H P x s needs (q.1.1, (forkOn (L.stage2 x) q.1.2 ec.1, ec.2))
            (q.1.1, (forkOn (L.stage2 x) q.1.2 ec'.1, ec'.2)) g.val) K q.2.1 q.2.2

/-- **The mass off the committed interiors, by class, at one cap**:
`2·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256 + N/(K + 1)`, under `ExpectedCR` at the by-class finder of the cap
`K`, where a second stage's extractions open a class of gates whole or not at all under a replay block's interior
roots (`hallI`): `c` a run's calls with every session extracted (`callsAt`), the sum over the representatives of the
classes of the gates the run's sample needs, `c_q` the run's calls with only the sessions that have a proof block
needing `q` extracted (`callsBy`), `N` the program's gates. The contract's clause it reads is `Sessions.interior`. -/
theorem pr_offIreg_le_cls_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 Pρ t.2 ec g).isSome → (L.iopTable H P x t.1 Pρ t.2 ec g').isSome)
    (K : ℕ) (hCR : ExpectedCR H (L.finderICls H P x s Pρ needs cls K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
              L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_cls_le_ex_cap H cls
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (vstar := fun (t : Ω × (Fin (L.V.k x) → Coin)) (g : Fin (L.nG x)) => L.iplur H P x t.1 Pρ t.2 g)
    (fun t => L.isPlurality_iplur H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun t ec g hg => decide_eq_true (L.iop_isSome_needs H P x t.1 hs Pρ needs hpos t.2 ec g hg))
    (fun t ec g g' hgg hg => hallI t ec g g' hgg hg)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (q : Fin (L.nG x)) =>
      L.toCompiled.callsBy H P x s needs q.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.toCompiled.collBy H P x s needs (t.1, (forkOn (L.stage2 x) t.2 ec.1, ec.2))
        (t.1, (forkOn (L.stage2 x) t.2 ec'.1, ec'.2)) g.val)
    (fun t ec ec' g hg => icollBy_of_conflictAt hbind hS hs Pρ P needs hpos t.1 t.2 ec ec' g hg)
    (fun t ec ec' g u v huv => L.toCompiled.collBy_collision H P x s needs _ _ g.val huv) hCR
  rw [Fintype.card_fin] at h
  have hP := Pr_keep_forkOn (L.stage2 x)
    (fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) => L.OffIreg H P x ω Pρ R c)
  have hE := Ex_keep_forkOn (L.stage2 x) (fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
    ((L.callsAt H P x s ω R c +
      ∑ q ∈ (univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x R, g.val ∈ needs u))).image cls,
        L.toCompiled.callsBy H P x s needs q.val ω R c : ℕ) : ℝ))
  have hoff : Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.OffIreg H P x q.1.1 Pρ (forkOn (L.stage2 x) q.1.2 q.2.1) q.2.2) ≤
      Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off (L.iopTable H P x q.1.1 Pρ q.1.2)
          (fun g : Fin (L.nG x) => L.iplur H P x q.1.1 Pρ q.1.2 g) q.2 g) :=
    Pr_mono fun q hq => L.off_of_offIreg H P x q.1.1 Pρ q.1.2 q.2 hq
  exact (hP.symm.le.trans (hoff.trans h)).trans_eq
    (congrArg (fun e : ℝ => 2 * e / 2 ^ (256 : ℝ) + (L.nG x : ℝ) / ((K : ℝ) + 1)) hE)

/-! ## The event, by class -/

/-- **The event of `KSRegTwoX` with the by-class finders' terms, at one cap for each finder.** On a statement in
scope, where a reveal for proof block `u` opens only gates of `needs u`, a run's extractions open a class of gates
whole or not at all under the statement's roots (`hallX`) and under a replay block's interior roots (`hallI`), and the
boundary's by-class first-hit finder of the cap `KX` (`Compiled.finderXCls`) and the interior's of the cap `KI`
(`finderICls`) meet the linear expected-time bound, an accepted run samples a proof block of `FregTwoX` with
probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256 + N/(KX + 1) + N/(KI + 1)`.

* `c` is a run's calls with every session of its sample extracted (`callsAt`): each finder's first run. The sum is
  over the representatives of the classes of the gates the proof blocks of the run's coins' sample need, and `c_q` is
  the run's calls with only the sessions that have a proof block needing the representative `q` extracted
  (`callsBy`): what a further run at those coins costs a finder for a class it waits for.
* Every proof block that needs a gate needs its class's representative (`hrep`, which the proof doesn't read): under
  it a finder's charge for a class covers the extraction its output reads at a gate of the class
  (`Compiled.callsBy_le_of_needs`), and the expectation is at most the by-position one (`Compiled.ex_callsCls_le`).
* The expectation is over the prover's tape, the run's coins and the extractor's, joint in the sample and the calls.
* The premises the bound is small under are `eventTwoX_le_firstHit_at_cap`'s, with the linear expected-time bound at
  these two finders in place of those two. -/
theorem eventTwoX_le_firstHit_cls_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (_hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (KX KI : ℕ) (hX : ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls KX))
    (hI : ExpectedCR H (L.finderICls H P x s d.part needs cls KI)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
              L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        ((L.nG x : ℝ) / ((KX : ℝ) + 1) + (L.nG x : ℝ) / ((KI : ℝ) + 1)) := by
  have hB := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_cls_cap hbind hs P needs hpos cls hallX KX hX)
  have hN := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_cls_cap hbind hS hs d.part P needs hpos cls hallI KI hI)
  have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P hB hN
  have h4 : ∀ e : ℝ, 2 * e / 2 ^ (256 : ℝ) + 2 * e / 2 ^ (256 : ℝ) = 4 * e / 2 ^ (256 : ℝ) := fun e => by ring
  rw [← h4]
  linarith

/-- **The event of `KSRegTwoX` with the by-class finders' terms** where `ε_reg,2,X` has its two square roots:
`eventTwoX_le_firstHit_cls_cap` under the linear expected-time bound at the two by-class finders of every cap, with
nothing left of the caps' terms: at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256`. -/
theorem eventTwoX_le_firstHit_cls {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
              L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := by
  have hcap : ∀ K : ℕ,
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
          4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                  decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
                L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (2 * (L.nG x : ℝ)) / ((K : ℝ) + 1) := by
    intro K
    have h := eventTwoX_le_firstHit_cls_cap hA hbind hS hs hPt hd hin P needs hpos cls hrep hallX hallI K K (hX K)
      (hI K)
    have e : (L.nG x : ℝ) / ((K : ℝ) + 1) + (L.nG x : ℝ) / ((K : ℝ) + 1) =
        (2 * (L.nG x : ℝ)) / ((K : ℝ) + 1) := by ring
    rw [e] at h
    exact h
  exact Verity.FirstHit.le_of_forall_cap hcap

/-! ## From what the extractions keep, and from what the extracted witnesses reveal -/

/-- **A run's extractions open a class whole under a replay block's interior roots**, where they keep classes whole
and a class lies in one replay block (`hblock`): the by-class finder's hypothesis at the committed interiors
(`hallI`). -/
theorem iopTable_isSome_cls (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock Pρ g.val = gateBlock Pρ g'.val)
    (r : Fin (L.V.k x) → Coin) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x)
    (hkept : L.toCompiled.ClassWhole H x cls (L.kept H P x s ω (forkOn (L.stage2 x) r ec.1) ec.2))
    (g g' : Fin (L.nG x)) (hgg : cls g = cls g') (hg : (L.iopTable H P x ω Pρ r ec g).isSome) :
    (L.iopTable H P x ω Pρ r ec g').isSome := by
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hg
  simp only [iopTable, iop] at hb
  obtain ⟨ρ, hρ, hb'⟩ := Option.bind_eq_some_iff.1 hb
  obtain ⟨q, hq, h1, h2, -, hok⟩ := iopensAt_some hs hb'
  obtain ⟨q', hq', h1', h2', hok'⟩ := hkept q hq hok g g' h2 hgg
  have hr : q'.1 = L.iroots x (run H P L.V x ω (forkOn (L.stage2 x) r ec.1)).seen ρ := h1'.trans h1
  have hsome : (L.iopensAt H P x ω (forkOn (L.stage2 x) r ec.1) ec.2 ρ q'.2.pos).isSome := by
    simp only [iopensAt, hs, Option.isSome_map]
    exact List.find?_isSome.2 ⟨q', hq', by rw [← hr]; simp [hok']⟩
  rw [h2'] at hsome
  obtain ⟨b', hb''⟩ := Option.isSome_iff_exists.1 hsome
  have hρ' : gateBlock Pρ g'.val = some ρ := (hblock g g' hgg).symm.trans hρ
  exact Option.isSome_iff_exists.2 ⟨b', Option.bind_eq_some_iff.2 ⟨ρ, hρ', hb''⟩⟩

/-- **The event of `KSRegTwoX` with the by-class finders' terms, from what the extractions keep**:
`eventTwoX_le_firstHit_cls` with its two hypotheses on what a run's extractions open derived from one on what they
keep. Where what a run's extractions keep holds classes whole (`hkept`: an opening they keep comes with one they keep,
under the same roots, at each gate of its class) and a class lies in one replay block of the first stage's partition
(`hblock`), an accepted run samples a proof block of `FregTwoX` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{q ∈ classes of the sample} c_q]/2^256`,
under the linear expected-time bound at the two by-class finders of every cap. -/
theorem eventTwoX_le_firstHit_kept {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val)
    (hkept : ∀ (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x),
      L.toCompiled.ClassWhole H x cls (L.kept H P x s ω r c))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
              L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) :=
  eventTwoX_le_firstHit_cls hA hbind hS hs hPt hd hin P needs hpos cls hrep
    (fun a g g' hgg hg =>
      L.toCompiled.opensAt_isSome_cls H P x a.1 hs cls a.2.1 a.2.2 (hkept a.1 a.2.1 a.2.2) g g' hgg hg)
    (fun t ec g g' hgg hg =>
      L.iopTable_isSome_cls H P x t.1 hs d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g' hgg hg) hX hI

/-- **The event of `KSRegTwoX` with the by-class finders' terms, from what the extracted witnesses reveal**:
`eventTwoX_le_firstHit_kept` where, at every run, for each session of the run's sample, extracted at its first slot,
each witness `w` its extraction returns and each proof block `u'` of the session, what `w` reveals at `u'` holds
classes whole (`hwit`: an opening of that reveal that passes its opening check has, for each gate of its class, an
opening in the same reveal, under the same roots, that passes too). The run need not accept, and a witness no
extraction returns is not asked about. -/
theorem eventTwoX_le_firstHit_wit {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (cls : Fin (L.nG x) → Fin (L.nG x))
    (hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val)
    (hwit : ∀ (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x),
      ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
        ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val, ∀ w : A.W,
          ((L.sessWit x iu.2 r (run H P L.V x ω r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval H (run H P L.V x ω)).1 =
              some w →
            ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
              L.toCompiled.ClassWhole H x cls (L.reveal s u' (run H P L.V x ω r).seen w))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image cls,
              L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) :=
  eventTwoX_le_firstHit_kept hA hbind hS hs hPt hd hin P needs hpos cls hrep hblock
    (fun ω r c => L.toCompiled.kept_classWhole_of_wit H P x s ω cls r c (hwit ω r c)) hX hI

/-- **The event of `KSRegTwoX` with the by-class finders' terms, at the classes of a key**:
`eventTwoX_le_firstHit_wit` at the classes a key of the gates gives (`Verity.FirstHit.clsOf`: a gate's class is named
by the least gate with its key). Where gates of one key are needed by the same proof blocks (`hneeds`) and lie in the
same replay block of the first stage's partition (`hblock`), the two hypotheses on the representatives hold, and what
is left to show is that what the extracted witnesses reveal holds those classes whole (`hwit`). -/
theorem eventTwoX_le_firstHit_key {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) {Key : Type} [DecidableEq Key]
    (key : Fin (L.nG x) → Key)
    (hneeds : ∀ g g' : Fin (L.nG x), key g = key g' → ∀ u, g.val ∈ needs u → g'.val ∈ needs u)
    (hblock : ∀ g g' : Fin (L.nG x), key g = key g' → gateBlock d.part g.val = gateBlock d.part g'.val)
    (hwit : ∀ (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x),
      ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
        ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val, ∀ w : A.W,
          ((L.sessWit x iu.2 r (run H P L.V x ω r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval H (run H P L.V x ω)).1 =
              some w →
            ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
              L.toCompiled.ClassWhole H x (Verity.FirstHit.clsOf key)
                (L.reveal s u' (run H P L.V x ω r).seen w))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs (Verity.FirstHit.clsOf key) K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs (Verity.FirstHit.clsOf key) K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))).image (Verity.FirstHit.clsOf key),
              L.toCompiled.callsBy H P x s needs c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) :=
  eventTwoX_le_firstHit_wit hA hbind hS hs hPt hd hin P needs hpos (Verity.FirstHit.clsOf key)
    (fun u g hg => hneeds g (Verity.FirstHit.clsOf key g) (Verity.FirstHit.key_clsOf key g).symm u hg)
    (fun g g' h => hblock g g' ((Verity.FirstHit.clsOf_eq_iff key g g').1 h)) hwit hX hI

end CompiledTwo

end Claims.SamplingGeneric
