module

public import Proofs.Flock.SamplingGeneric.RegCanon
public import Proofs.Flock.SamplingGeneric.RunExtAt
public import Proofs.Core.SamplingGeneric.FirstHit
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.Sampling.Cost
import Proofs.Flock.Claims.Sampling.KS

/-!
# The first-hit finder at the committed trace

`pr_offReg_le` (`Proofs.Flock.SamplingGeneric.Reg`) bounds a run's openings off the committed trace by
`N·√Pr[conflict]`: the rewinding lemma at the runs' openings of the program's gates (`opTable`), whose plurality the
committed trace is (`plur`). This file bounds the same event at the same table and the same plurality linearly, by the
first-hit finder (`Proofs.Core.SamplingGeneric.FirstHit`):

* `isPlurality_plur`: at each tape of the prover, the committed trace's values are a plurality table of the runs'
  openings.
* `pr_offReg_le_firstHit`: at each tape, `Pr[a run opens a gate off the committed trace] ≤ 2·Pr[Found] + N/(K + 1)`,
  where `Found` is the first-hit finder's event at `K` further runs' coins and `N` the program's gates.
* `pr_offReg_le_sample`: over the prover's tape, under `ExpectedCR` at one finder whose seed holds the tape and whose
  output is `B_conf`'s at two runs (`collAt`), `Pr[a run opens a gate off the committed trace] ≤
  2·(1 + w)·E[c]/2^256 + N/(K + 1)`, where `w` bounds the gates one run's sample holds (`chk`, a function of the run's
  coins alone: the finder's sample never reads the tape here) and `c` is a run's calls (`runCost`);
  `pr_offReg_le_sample_forall`, under the bound at the finder of every cap, has no `N/(K + 1)`.
* `pr_offReg_le_pos`: `pr_offReg_le_sample` with `w` the positions any reveal opens under the statement's roots
  (`opensAt_isSome_mem`).
* `pr_offReg_le_needs`: `pr_offReg_le_sample` with `w` a bound on the gates the reveals open for the proof blocks of
  one run's sample (`opensAt_isSome_mem_checked`), where a block's reveals open only gates of `needs`.
* `pr_regHit_le_of_offReg`: `KSReg`'s first half from any bound `b` on the probability that a run accepts and opens
  a gate off the committed trace, `b` where `εreg` has `N·√Pr[B_conf]`: the proof of `ksReg_of_sessionKS` with the
  bound as a hypothesis in place of `pr_offReg_le`. `pr_regHit_le_sample` is it at `2·(1 + w)·E[c]/2^256`, and
  `pr_regHit_le_sample_ex` at `2·E[(1 + |sample|)·c]/2^256` (`pr_offReg_le_sample_ex`), which assumes no bound on a
  sample's size: coins at which the sample is every gate count by their probability.

* **The statements to read** are at the verifier's own sample, the gates the proof blocks of a run's coins' sample
  need (`finderReg`): `pr_regHit_le_needs_ex_cap`, `KSReg`'s first half at one cap,
  `κ_s + √(k·γ_s·Pr[B]) + 2·E[(1 + |sample|)·c]/2^256 + N/(K + 1) + Pr[B_conf]`, and `pr_regHit_le_needs_ex`, under
  the bound at the finder of every cap, without the cap's term. The forms that take a sample `chk` from their caller
  are lemmas: a sample that is not the one the finder can compute from the statement and the coins makes the premise
  one about a finder nobody runs.

`pr_offReg_le_firstHit` turns no found pair into a collision and builds no `Finder`: `Found` is two runs' extractions
that open one gate two ways, the event `Conflict` reads at a pair of runs. `pr_offReg_le_sample` does both.

**What a run costs.** `runCost` is a whole run's calls: the prover's, the verifier's, and the extraction's of every
session of the run's sample (`runExt`). `finderReg` pays it at the first run and at each further run it makes,
whichever gate it waits for: it extracts every session of a further run.

* **By position** (`finderRegAt`, `pr_regHit_le_needs_at_cap`, `pr_regHit_le_needs_at`): the finder that, at a
  further run, extracts for each gate it waits for only the sessions with a proof block that needs the gate
  (`Proofs.Flock.SamplingGeneric.RunExtAt`). Its term is `2·E[c + Σ_{g ∈ sample} c_g]/2^256`: `c` the first run's
  calls, read whole, and `c_g` a run's calls with only those sessions extracted. It is at most the whole-run term
  (`ex_runCost_callsBy_le`).

* **With no premise** (`pr_regHit_le_finder`, `pr_regHit_le_finderAt`): `KSReg`'s first half with a finder's success
  in it, `κ_s + √(k·γ_s·Pr[B]) + 2·Pr[F_K finds] + N/(K + 1) + Pr[B_conf]`, and the finders' expected calls as
  statements of their own (`ex_cost_finderReg_le`, `ex_cost_finderRegAt_le`). The statements under `ExpectedCR` are
  each pair together.

**What the bound is small under.** Three premises: the linear expected-time bound at the first-hit finder
(`ExpectedCR`, a hypothesis of each statement here but those with no premise); the same at `B_conf`, whose success
`Pr[B_conf]` is a term of the bound; and the strict birthday bound (`StrictBirthday`) at `B`, whose success is under
the square root `√(k·γ_s·Pr[B])`, where the linear bound leaves the square root of a linear term. The expectation
`E[(1 + |sample|)·c]` is joint: a bound on a run's mean calls at each coins bounds it by one more than the mean sample
times that bound, and a bound on the mean over all coins does not.

Everything here is at the one-stage contract (`Compiled`), where `KSReg` is stated. The finder waits for a gate, not
for a proof block, and reruns a whole run: the two openings it compares are under the statement's roots, the only ones
`opensAt` and the committed trace read. Nothing here covers two stages (`CompiledTwo`): a finder for openings under
roots that a run fixes after its first coins has to keep those coins and redraw the rest, which this finder does not.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement ownLog)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- The committed trace's values are a plurality table of the runs' openings of the program's gates. -/
theorem isPlurality_plur (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    FlockSoundness.Rewinding.IsPlurality (L.opTable H P x ω) fun g => L.plur H P x ω g := by
  classical
  have hcard : (0 : ℝ) < Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin (L.nG x)) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g b) : ℝ) =
        Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b) *
          Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    intro g b
    rw [Pr_eq_card_of_mem (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.opensAt H P x ω rc.1 rc.2 g = some b)
      (FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g b)
      fun rc => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  intro g v
  have h := le_plur (fun b => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
    L.opensAt H P x ω rc.1 rc.2 g = some b) v
  have h' : (#(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g v) : ℝ) ≤
      #(FlockSoundness.Rewinding.openedTo (L.opTable H P x ω) g (L.plur H P x ω g)) := by
    rw [hcnt, hcnt]
    exact mul_le_mul_of_nonneg_right h hcard.le
  exact_mod_cast h'

/-- **The first-hit bound at the committed trace**: at each tape, a run's extractions open a gate of the program off
the committed trace with probability at most twice the first-hit finder's probability of finding a pair at `K` further
runs' coins, plus `N/(K + 1)`, `N` the program's gates. -/
theorem pr_offReg_le_firstHit (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (K : ℕ) :
    Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      2 * Pr (fun s : ((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          Verity.FirstHit.Found (L.opTable H P x ω) K s.1 s.2) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have hoff : Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.OffReg H P x ω rc.1 rc.2) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => ∃ g,
        FlockSoundness.Rewinding.Off (L.opTable H P x ω) (fun g => L.plur H P x ω g) rc g) :=
    Pr_mono fun rc hrc => by
      obtain ⟨g, hg, b, hb, hne⟩ := hrc
      refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
      rw [L.regVal_eq_plur H P x ω hb]
      exact (hb.symm.trans h).symm
  have h1 := Verity.FirstHit.pr_off_le (L.opTable H P x ω) (L.isPlurality_plur H P x ω) K
  have h2 := Verity.FirstHit.pr_miss_le (L.opTable H P x ω) K
  have hK : (0 : ℝ) < (K : ℝ) + 1 := by positivity
  have hw : (#(univ.filter fun g : Fin (L.nG x) =>
      (FlockSoundness.Rewinding.opened (L.opTable H P x ω) g).Nonempty) : ℝ) ≤ (L.nG x : ℝ) := by
    have h := Verity.FirstHit.card_opened_nonempty_le (L.opTable H P x ω)
    rw [Fintype.card_fin] at h
    exact_mod_cast h
  have h3 := div_le_div_of_nonneg_right hw hK.le
  linarith

/-! ## The finder at the committed trace, with the prover's tape in its seed -/

/-- **The pair two runs' extractions give** at one tape: the first collision among the calls the two runs' extractions
make themselves, `B_conf`'s output at the two runs. -/
noncomputable def collAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (ω : Ω) (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) : Option (Msg × Msg) :=
  findColl (ownLog H (run H P L.V x ω) (L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2) ++
    ownLog H (run H P L.V x ω) (L.runExt x s rc'.1 (run H P L.V x ω rc'.1).seen rc'.2))

/-- **A run's calls**: both parties', and its extractions', their reruns' included; one run's share of `B_conf`'s
cost. -/
noncomputable def runCost (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) : ℕ :=
  (run H P L.V x ω rc.1).pLog.length + (run H P L.V x ω rc.1).vLog.length +
    ((L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2).eval H (run H P L.V x ω)).2.1

/-- What `collAt` outputs is a collision of `H`. -/
theorem collAt_collision (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (ω : Ω) (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) {u v : Msg} (h : L.collAt H P x s ω rc rc' = some (u, v)) :
    Collision H u v :=
  findColl_spec H _
    (fun p hp => (List.mem_append.1 hp).elim (Sampling.ownLog_valid H _ _ p) (Sampling.ownLog_valid H _ _ p)) u v h

/-- Two runs whose extractions open a gate two ways give `collAt` a pair (`bconf_of_disagree`). -/
theorem collAt_of_conflictAt {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x))
    (h : FlockSoundness.Rewinding.ConflictAt (L.opTable H P x ω) (rc, rc') g) :
    (L.collAt H P x s ω rc rc').isSome := by
  obtain ⟨h1', h2', hne'⟩ := h
  have h1 : (L.opensAt H P x ω rc.1 rc.2 g).isSome := h1'
  have h2 : (L.opensAt H P x ω rc'.1 rc'.2 g).isSome := h2'
  have hne : L.opensAt H P x ω rc.1 rc.2 g ≠ L.opensAt H P x ω rc'.1 rc'.2 g := hne'
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  rw [L.opensAt_eq H P x ω hs] at hb1
  rw [L.opensAt_eq H P x ω hs] at hb2
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := val_some_mem hb1
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := val_some_mem hb2
  obtain ⟨u, v, huv, -⟩ := bconf_of_disagree hbind s P ω rc rc' (Or.inl ha) (Or.inr ha') (ha1.trans ha1'.symm)
    (ha2.trans ha2'.symm) (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))
  exact Option.isSome_iff_exists.2 ⟨(u, v), huv⟩

/-- A run whose extractions open a gate of the program off the committed trace opens it off the plurality table of
the committed trace's finders. -/
theorem pr_offReg_le_off (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g,
        FlockSoundness.Rewinding.Off (L.opTable H P x q.1) (fun g => L.plur H P x q.1 g) q.2 g) :=
  Pr_mono fun q hq => by
    obtain ⟨g, hg, b, hb, hne⟩ := hq
    refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hb⟩, fun h => hne ?_⟩
    rw [L.regVal_eq_plur H P x q.1 hb]
    exact (hb.symm.trans h).symm

/-- A lemma, at a sample `chk` its caller gives; the statement to read is `pr_offReg_le_needs_ex_cap`, at the sample
the verifier's coins define. **The off-trace mass at the sample's size, under `ExpectedCR` at one finder.** `chk`
reads a run's coins and says
which gates its sample holds, at most `w` of them at any coins, and a run's extractions open only gates of its sample
(`hchk`). The finder draws the prover's tape and a first run,
then further runs' coins, and makes a further run only when its sample holds a gate the first run opened that no
further run has opened yet; it outputs `collAt` at the first run and a further run that is the first to open such a
gate. If it meets the linear expected-time bound, a run at a uniform tape opens a gate of the program off the committed
trace with probability at most `2·(1 + w)·E[c]/2^256 + N/(K + 1)`: `c` a run's calls (`runCost`), `N` the program's
gates, `K` the finder's cap. -/
theorem pr_offReg_le_sample {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (chk : (Fin (L.V.k x) → Coin) × L.SlotEC x → Fin (L.nG x) → Bool)
    (hchk : ∀ (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opTable H P x ω rc g).isSome → chk rc g = true) {w : ℕ}
    (hw : ∀ rc, #(univ.filter fun g : Fin (L.nG x) => chk rc g) ≤ w) (K : ℕ)
    (hCR : ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x) (fun _ => chk) (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * ((1 + (w : ℝ)) *
          Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.runCost H P x s q.1 q.2 : ℝ))) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have hoff := L.pr_offReg_le_off H P x
  have h := Verity.FirstHit.pr_off_sample_le_of_coll H (L.opTable H P x)
    (vstar := fun (ω : Ω) (g : Fin (L.nG x)) => L.plur H P x ω g) (fun ω => L.isPlurality_plur H P x ω) (fun _ => chk)
    hchk (fun _ => hw) (L.runCost H P x s) K (L.collAt H P x s)
    (fun ω rc rc' g hg => collAt_of_conflictAt hbind hs P ω rc rc' g hg)
    (fun ω rc rc' u v huv => L.collAt_collision H P x s ω rc rc' huv) hCR
  rw [Fintype.card_fin] at h
  exact hoff.trans h

/-- `pr_offReg_le_sample` under `ExpectedCR` at the finder of every cap: `2·(1 + w)·E[c]/2^256`, with no term for the
cap. -/
theorem pr_offReg_le_sample_forall {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (chk : (Fin (L.V.k x) → Coin) × L.SlotEC x → Fin (L.nG x) → Bool)
    (hchk : ∀ (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opTable H P x ω rc g).isSome → chk rc g = true) {w : ℕ}
    (hw : ∀ rc, #(univ.filter fun g : Fin (L.nG x) => chk rc g) ≤ w)
    (hCR : ∀ K : ℕ, ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x) (fun _ => chk) (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * ((1 + (w : ℝ)) *
          Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.runCost H P x s q.1 q.2 : ℝ))) /
        2 ^ (256 : ℝ) :=
  Verity.FirstHit.le_of_forall_cap fun K => pr_offReg_le_sample hbind hs P chk hchk hw K (hCR K)

/-! ## The sample, from what the reveals open -/

/-- `pr_offReg_le_sample` at the positions any reveal opens under the statement's roots (`opensAt_isSome_mem`): the
count is `#Spos`, the same at every run. -/
theorem pr_offReg_le_pos {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (Spos : Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ Spos) (K : ℕ)
    (hCR : ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x)
      (fun (_ : Ω) (_ : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) => decide (g.val ∈ Spos))
      (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * ((1 + (#Spos : ℝ)) *
          Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.runCost H P x s q.1 q.2 : ℝ))) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) :=
  pr_offReg_le_sample hbind hs P (fun _ g => decide (g.val ∈ Spos))
    (fun ω rc g hg => decide_eq_true (opensAt_isSome_mem H L P x ω hs Spos hpos rc.1 rc.2 g hg))
    (fun _ => card_le_card_of_injOn Fin.val (by intro g hg; exact of_decide_eq_true (mem_filter.1 hg).2)
      (by intro a _ b _ hab; exact Fin.ext hab)) K hCR

/-- **A run's extractions open only what its own sample's reveals open**: a gate `opensAt` opens at the coins `r` is
the position of an opening some reveal makes, under the statement's roots, for a proof block of `r`'s sample;
`opensAt_isSome_mem` with the proof block kept. -/
theorem opensAt_isSome_mem_checked (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u)
    (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (g : ℕ) (hg : (L.opensAt H P x ω r c g).isSome) :
    ∃ u ∈ L.checked x r, g ∈ needs u := by
  simp only [opensAt, hs] at hg
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hg
  unfold Sampling.Wit.val at hb
  obtain ⟨o, ho, -⟩ := Option.map_eq_some_iff.1 hb
  have hpred := List.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hpred
  obtain ⟨q, hq, hqo⟩ := List.mem_filterMap.1 (List.mem_of_find?_eq_some ho)
  by_cases hr : q.1 = s.roots
  · rw [ite_eq_left hr] at hqo
    obtain rfl := Option.some.inj hqo
    rw [← hpred.1]
    unfold runExt at hq
    rw [Sampling.flatMapRP_eval] at hq
    obtain ⟨iu, -, hiu⟩ := List.mem_flatMap.1 hq
    split at hiu
    · unfold sessExt at hiu
      rw [(Claims.RP.eval_bind H _ _ _).1] at hiu
      split at hiu
      · simp [RP.eval] at hiu
      · rw [Sampling.flatMapRP_eval] at hiu
        obtain ⟨u', hu'm, hu'⟩ := List.mem_flatMap.1 hiu
        have hu'b : u' ∈ (L.sessSlots x (L.checked x r) iu.1).filterMap fun j => (L.checked x r)[j]? := hu'm
        obtain ⟨j, -, hj⟩ := List.mem_filterMap.1 hu'b
        exact ⟨u', List.mem_of_getElem? hj,
          hpos _ _ _ q (Sampling.checkRevealed_mem H _ L.commit _ q hu').1 hr⟩
    · simp [RP.eval] at hiu
  · rw [ite_eq_right hr] at hqo
    exact absurd hqo (by simp)

/-- **The off-trace mass at what one run's sample opens.** Where the reveals for a proof block `u` open, under the
statement's roots, only gates of `needs u`, and the proof blocks of any one run's sample need at most `w` gates of the
program between them, `pr_offReg_le_sample` holds with that `w`: the sample of a run's coins is the gates its proof
blocks (`checked`) need. -/
theorem pr_offReg_le_needs {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) {w : ℕ}
    (hw : ∀ r : Fin (L.V.k x) → Coin,
      #(univ.filter fun g : Fin (L.nG x) => ∃ u ∈ L.checked x r, g.val ∈ needs u) ≤ w) (K : ℕ)
    (hCR : ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x)
      (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
        decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
      (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * ((1 + (w : ℝ)) *
          Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.runCost H P x s q.1 q.2 : ℝ))) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) :=
  pr_offReg_le_sample hbind hs P (fun rc g => decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (fun ω rc g hg => decide_eq_true (opensAt_isSome_mem_checked H L P x ω hs needs hpos rc.1 rc.2 g hg))
    (fun rc => (le_of_eq (congrArg Finset.card
      (Finset.filter_congr fun _ _ => decide_eq_true_iff))).trans (hw rc.1)) K hCR

/-! ## The committed form's first half, linearly -/

/-- A lemma, at a sample `chk` its caller gives; the statement to read is `pr_offReg_le_needs_ex`, at the sample the
verifier's coins define. `pr_offReg_le_sample_forall` with no bound on a sample's size:
`2·E[(1 + |sample|)·c]/2^256`, the expectation
over the prover's tape and a run's coins of the run's calls times one more than the gates its coins' sample holds.
Coins at which the sample is every gate count by their probability and their run's calls; the finder's further runs
are at all coins, those included, as the committed trace is the plurality over all coins. -/
theorem pr_offReg_le_sample_ex {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (chk : (Fin (L.V.k x) → Coin) × L.SlotEC x → Fin (L.nG x) → Bool)
    (hchk : ∀ (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opTable H P x ω rc g).isSome → chk rc g = true)
    (hCR : ∀ K : ℕ, ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x) (fun _ => chk) (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => chk q.2 g)) * L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) := by
  have hoff := L.pr_offReg_le_off H P x
  exact hoff.trans (Verity.FirstHit.pr_off_sample_le_ex H (L.opTable H P x)
    (vstar := fun (ω : Ω) (g : Fin (L.nG x)) => L.plur H P x ω g) (fun ω => L.isPlurality_plur H P x ω)
    (fun _ => chk) hchk (L.runCost H P x s) (L.collAt H P x s)
    (fun ω rc rc' g hg => collAt_of_conflictAt hbind hs P ω rc rc' g hg)
    (fun ω rc rc' u v huv => L.collAt_collision H P x s ω rc rc' huv) hCR)

/-- `pr_offReg_le_sample_ex` at the sample of `pr_offReg_le_needs`: where the reveals for a proof block `u` open,
under the statement's roots, only gates of `needs u`, the sample of a run's coins is the gates its proof blocks
(`checked`) need, whatever their number: `2·E[(1 + |sample|)·c]/2^256`. -/
theorem pr_offReg_le_needs_ex {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u)
    (hCR : ∀ K : ℕ, ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x)
      (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
        decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
      (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  pr_offReg_le_sample_ex hbind hs P (fun rc g => decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (fun ω rc g hg => decide_eq_true (opensAt_isSome_mem_checked H L P x ω hs needs hpos rc.1 rc.2 g hg)) hCR

set_option synthInstance.maxSize 1024 in
/-- **`KSReg`'s first half from any bound on the accepted off-trace mass**: the proof of `ksReg_of_sessionKS` with
`b` where `εreg` has `N·√Pr[B_conf]`, `b` bounding the probability that a run accepts and its extractions open a gate
of the program off the committed trace. An accepted run whose sample holds a proof block of the committed trace's
incorrect set has probability at most `κ_s + √(k·γ_s·Pr[B]) + b + Pr[B_conf]`; `Pr[B_conf]` is one run's own
disagreement. -/
theorem pr_regHit_le_of_offReg {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) {b : ℝ}
    (hb : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffReg H P x q.1 q.2.1 q.2.2) ≤ b) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + b + (Bconf H L Ω P x).Pr H := by
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hFlt : ∀ ω, ∀ v ∈ L.Freg H P x ω, v < L.nPU x := by
    intro ω v hv
    simp only [Freg, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1
  have hper : ∀ ω, Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ω) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selReg H P x ω) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          (run H P L.V x ω rc.1).out = .acc ∧ L.OffReg H P x ω rc.1 rc.2)) := by
    intro ω
    have hIO := Sampling.Pr_union_le
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
      (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧ L.OffReg H P x ω rc.1 rc.2)
    refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ L.Freg H P x ω)).symm.trans_le (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
    obtain ⟨hacc, v, hvl, hvF⟩ := hrc
    obtain ⟨i, hi, u, hiu, huF⟩ := L.selReg_some H P x ω (hS.slots x rc.1) (hFlt ω) hvl hvF
    obtain ⟨i₀, u₀, hlead, hu₀⟩ := leadOf_some hiu u.isLt
    rcases reg_split hS hs hPt hin P ω rc.1 rc.2 hacc i u hiu huF i₀ u₀ hlead hu₀ with h | h | h
    · exact Or.inl ⟨hacc, i₀, by rw [leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr ⟨hacc, h⟩)
  have hT := L.pr_unfound_le H P β hβ0 hβ (fun ω => L.leadSel x (L.selReg H P x ω))
    fun ω r => hκ ω r (L.leadSel x (L.selReg H P x ω))
  have hI := pr_inConf_le hbind hs P
  have hO : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧ L.OffReg H P x ω rc.1 rc.2) ≤ b :=
    (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffReg H P x q.1 q.2.1 q.2.2)).symm.trans_le hb
  refine (Sampling.Pr_prod_eq_Ex _).trans_le ((Ex_mono hper).trans ?_)
  rw [Ex_add, Ex_add]
  linarith

/-- A lemma, at a sample `chk` its caller gives; the statement to read is `pr_regHit_le_needs_ex`. **`KSReg`'s first
half with the first-hit finder's term** where `εreg` has `N·√Pr[B_conf]`:
`κ_s + √(k·γ_s·Pr[B]) + 2·(1 + w)·E[c]/2^256 + Pr[B_conf]`, under the linear expected-time bound at the first-hit
finder of every cap, where every coins' sample holds at most `w` gates. -/
theorem pr_regHit_le_sample {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s)
    (chk : (Fin (L.V.k x) → Coin) × L.SlotEC x → Fin (L.nG x) → Bool)
    (hchk : ∀ (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opTable H P x ω rc g).isSome → chk rc g = true) {w : ℕ}
    (hw : ∀ rc, #(univ.filter fun g : Fin (L.nG x) => chk rc g) ≤ w)
    (hCR : ∀ K : ℕ, ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x) (fun _ => chk) (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        2 * ((1 + (w : ℝ)) *
          Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (L.runCost H P x s q.1 q.2 : ℝ))) /
          2 ^ (256 : ℝ) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_sample_forall hbind hs P chk hchk hw hCR))

/-- A lemma, at a sample `chk` its caller gives; the statement to read is `pr_regHit_le_needs_ex`. **`KSReg`'s first
half with the first-hit finder's term, the sample's size as an expectation**:
`κ_s + √(k·γ_s·Pr[B]) + 2·E[(1 + |sample|)·c]/2^256 + Pr[B_conf]`, with no bound on a sample's size. -/
theorem pr_regHit_le_sample_ex {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s)
    (chk : (Fin (L.V.k x) → Coin) × L.SlotEC x → Fin (L.nG x) → Bool)
    (hchk : ∀ (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)),
      (L.opTable H P x ω rc g).isSome → chk rc g = true)
    (hCR : ∀ K : ℕ, ExpectedCR H (Verity.FirstHit.sampleFinder (L.opTable H P x) (fun _ => chk) (L.runCost H P x s) K
      fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
        Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => chk q.2 g)) * L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_sample_ex hbind hs P chk hchk hCR))

/-! ## The statements to read: the verifier's own sample, one cap and every cap -/

/-- **The first-hit finder at the committed trace**, at cap `K`. It draws the prover's tape and keeps it, draws a
run's coins and the extractor's, then `K` further runs' and extractor's coins, and makes a further run, at the kept
tape, only when its sample holds a gate the first run opened that no further run has opened yet. The sample of a
run's coins is the gates the proof blocks of the coins' sample (`checked`) need: it reads the statement and the coins,
never the tape or what a run answers. The finder outputs `collAt` at the first run and a further run that is the first
to open such a gate. Its cost is the calls of the runs it makes (`runCost`): each run's, the first and every further
one, is a whole run's, with every session of the run's sample extracted. -/
noncomputable def finderReg (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) : Finder :=
  Verity.FirstHit.sampleFinder (L.opTable H P x)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (L.runCost H P x s) K
    fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      Verity.FirstHit.outOf (L.opTable H P x q.1) (L.collAt H P x s q.1) K q.2.1 q.2.2

/-- **The off-trace mass at the verifier's own sample, at one cap**: `2·E[(1 + |sample|)·c]/2^256 + N/(K + 1)`, under
`ExpectedCR` at the first-hit finder of the cap `K` alone, `N` the program's gates. -/
theorem pr_offReg_le_needs_ex_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderReg H P x s needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have hoff := L.pr_offReg_le_off H P x
  have h := Verity.FirstHit.pr_off_sample_le_ex_cap_of_coll H (L.opTable H P x)
    (vstar := fun (ω : Ω) (g : Fin (L.nG x)) => L.plur H P x ω g) (fun ω => L.isPlurality_plur H P x ω)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (fun ω rc g hg => decide_eq_true (opensAt_isSome_mem_checked H L P x ω hs needs hpos rc.1 rc.2 g hg))
    (L.runCost H P x s) K (L.collAt H P x s)
    (fun ω rc rc' g hg => collAt_of_conflictAt hbind hs P ω rc rc' g hg)
    (fun ω rc rc' u v huv => L.collAt_collision H P x s ω rc rc' huv) hCR
  rw [Fintype.card_fin] at h
  exact hoff.trans h

/-- **`KSReg`'s first half with the first-hit finder's term, at one cap.** On a statement in scope, where the reveals
for a proof block `u` open, under the statement's roots, only gates of `needs u`, and the first-hit finder of the cap
`K` (`finderReg`) meets the linear expected-time bound, an accepted run samples a proof block of the committed
trace's incorrect set with probability at most
`κ_s + √(k·γ_s·Pr[B]) + 2·E[(1 + |sample|)·c]/2^256 + N/(K + 1) + Pr[B_conf]`.

* `|sample|` at a run's coins is the number of gates the proof blocks of the coins' sample need, and `c` is the run's
  calls with every session of its sample extracted (`runCost`). The finder pays `c` at each further run it makes,
  whichever gate it waits for.
* The expectation is over the prover's tape, the run's coins and the extractor's, joint in the sample's size and the
  calls: a bound on a run's mean calls at each coins bounds it by one more than the mean sample times that bound; a
  bound on the mean over all coins does not, since a prover may make its calls where the sample is large.
* The bound is small under three premises: the linear expected-time bound at this finder, a hypothesis here; the same
  at `B_conf`, whose success is the last term; and at `B`, whose success is under the square root, the strict
  birthday bound (`StrictBirthday`), the linear one leaving the square root of a linear term. -/
theorem pr_regHit_le_needs_ex_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderReg H P x s needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
              L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (L.nG x : ℝ) / ((K : ℝ) + 1)) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_needs_ex_cap hbind hs P needs hpos K hCR))

/-- **`KSReg`'s first half with the first-hit finder's term**: `pr_regHit_le_needs_ex_cap` under the linear
expected-time bound at the first-hit finder of every cap, with nothing left of the cap's term:
`κ_s + √(k·γ_s·Pr[B]) + 2·E[(1 + |sample|)·c]/2^256 + Pr[B_conf]`, where `εreg` has `N·√Pr[B_conf]` for the third
term. What `|sample|`, `c` and the expectation are, and the three premises the bound is small under, are
`pr_regHit_le_needs_ex_cap`'s. -/
theorem pr_regHit_le_needs_ex {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u)
    (hCR : ∀ K : ℕ, ExpectedCR H (L.finderReg H P x s needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_needs_ex hbind hs P needs hpos hCR))

/-! ## By position: a further run extracts only the sessions that can open the gate it is waited for -/

/-- **The by-position first-hit finder at the committed trace**, at cap `K`. Its seed, its sample and its waits are
`finderReg`'s, and its first run is read whole (`runCost`). At a further run it pays, for each gate still waited for
that the run's sample holds, a run at those coins with only the sessions that have a proof block needing the gate
extracted (`callsBy`). It outputs `collBy` at the first run, a further run that is the first to open some gate the
first run opened, and that gate: of the further run it reads the extractions at that gate. -/
noncomputable def finderRegAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) : Finder :=
  Verity.FirstHit.sampleFinderAt (L.opTable H P x)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (L.runCost H P x s)
    (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val ω rc.1 rc.2) K
    fun q : Ω × (((Fin (L.V.k x) → Coin) × L.SlotEC x) × (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      Verity.FirstHit.outOfAt (L.opTable H P x q.1)
        (fun (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
          L.collBy H P x s needs (q.1, rc) (q.1, rc') g.val) K q.2.1 q.2.2

/-- **The off-trace mass by position, at one cap**: `2·E[c + Σ_{g ∈ sample} c_g]/2^256 + N/(K + 1)`, under
`ExpectedCR` at the by-position finder of the cap `K` alone: `c` a run's calls with every session extracted, `c_g`
its calls with only the sessions that can open gate `g` extracted, `N` the program's gates. -/
theorem pr_offReg_le_needs_at_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderRegAt H P x s needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.runCost H P x s q.1 q.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have hoff := L.pr_offReg_le_off H P x
  have h := Verity.FirstHit.pr_off_at_le_ex_cap H (L.opTable H P x)
    (vstar := fun (ω : Ω) (g : Fin (L.nG x)) => L.plur H P x ω g) (fun ω => L.isPlurality_plur H P x ω)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (fun ω rc g hg => decide_eq_true (opensAt_isSome_mem_checked H L P x ω hs needs hpos rc.1 rc.2 g hg))
    (L.runCost H P x s)
    (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val ω rc.1 rc.2)
    (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.callsBy H P x s needs g.val ω rc.1 rc.2)
    (fun _ _ _ => le_rfl) K
    (fun (ω : Ω) (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.collBy H P x s needs (ω, rc) (ω, rc') g.val)
    (fun ω rc rc' g hg => by
      obtain ⟨h1, h2, hne⟩ := hg
      exact collBy_of_conflictAt hbind hs P needs hpos (ω, rc) (ω, rc') g.val h1 h2 hne)
    (fun ω rc rc' g u v huv => L.collBy_collision H P x s needs (ω, rc) (ω, rc') g.val huv) hCR
  rw [Fintype.card_fin] at h
  exact hoff.trans h

/-- **The off-trace mass by position**: `pr_offReg_le_needs_at_cap` under `ExpectedCR` at the by-position finder of
every cap, with nothing left of the cap's term. -/
theorem pr_offReg_le_needs_at {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u)
    (hCR : ∀ K : ℕ, ExpectedCR H (L.finderRegAt H P x s needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.runCost H P x s q.1 q.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  Verity.FirstHit.le_of_forall_cap fun K => pr_offReg_le_needs_at_cap hbind hs P needs hpos K (hCR K)

/-- **The by-position expectation is at most the whole-run one**: `E[c + Σ_{g ∈ sample} c_g] ≤ E[(1 + |sample|)·c]`,
a run's calls at one gate being at most its calls with every session extracted (`callsBy_le`). -/
theorem ex_runCost_callsBy_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.runCost H P x s q.1 q.2 +
          ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) :=
  Ex_mono fun q => by
    have h : ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
        L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 ≤
        #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)) *
          L.runCost H P x s q.1 q.2 :=
      (Finset.sum_le_card_nsmul _ _ _ fun g _ => L.callsBy_le H P x s needs g.val q.1 q.2.1 q.2.2).trans
        (le_of_eq (smul_eq_mul _ _))
    have h2 : L.runCost H P x s q.1 q.2 +
        ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
          L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 ≤
        (1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.runCost H P x s q.1 q.2 := by
      rw [Nat.add_mul, Nat.one_mul]
      exact Nat.add_le_add_left h _
    exact Nat.cast_le.2 h2

/-- **`KSReg`'s first half with the by-position finder's term, at one cap.** On a statement in scope, where the
reveals for a proof block `u` open, under the statement's roots, only gates of `needs u`, and the by-position
first-hit finder of the cap `K` (`finderRegAt`) meets the linear expected-time bound, an accepted run samples a
proof block of the committed trace's incorrect set with probability at most
`κ_s + √(k·γ_s·Pr[B]) + 2·E[c + Σ_{g ∈ sample} c_g]/2^256 + N/(K + 1) + Pr[B_conf]`.

* `c` is a run's calls with every session of its sample extracted (`runCost`): the finder's first run. The sum is
  over the gates the proof blocks of the run's coins' sample need, and `c_g` is the run's calls with only the sessions
  that have a proof block needing gate `g` extracted (`callsBy`): what a further run at those coins costs the finder
  for a gate it waits for.
* The expectation is over the prover's tape, the run's coins and the extractor's, joint in the sample and the calls.
  It is at most `E[(1 + |sample|)·c]`, `pr_regHit_le_needs_ex_cap`'s (`ex_runCost_callsBy_le`).
* The premises the bound is small under are `pr_regHit_le_needs_ex_cap`'s, with the linear expected-time bound at
  this finder in place of that one's. -/
theorem pr_regHit_le_needs_at_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderRegAt H P x s needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.runCost H P x s q.1 q.2 +
              ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
                L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (L.nG x : ℝ) / ((K : ℝ) + 1)) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_needs_at_cap hbind hs P needs hpos K hCR))

/-- **`KSReg`'s first half with the by-position finder's term**: `pr_regHit_le_needs_at_cap` under the linear
expected-time bound at the by-position finder of every cap, with nothing left of the cap's term:
`κ_s + √(k·γ_s·Pr[B]) + 2·E[c + Σ_{g ∈ sample} c_g]/2^256 + Pr[B_conf]`, where `εreg` has `N·√Pr[B_conf]` for the
third term. -/
theorem pr_regHit_le_needs_at {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u)
    (hCR : ∀ K : ℕ, ExpectedCR H (L.finderRegAt H P x s needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.runCost H P x s q.1 q.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_needs_at hbind hs P needs hpos hCR))

/-! ## With no premise: the finders' success in the bound, and their expected calls -/

/-- **The off-trace mass with the first-hit finder's success in it**, with no premise on `H`:
`2·Pr[the finder outputs a collision] + N/(K + 1)`, at the first-hit finder of the cap `K` (`finderReg`), `N` the
program's gates. -/
theorem pr_offReg_le_finder {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (K : ℕ) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * (L.finderReg H P x s needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_sample_le_finder H (L.opTable H P x)
    (vstar := fun (ω : Ω) (g : Fin (L.nG x)) => L.plur H P x ω g) (fun ω => L.isPlurality_plur H P x ω)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (L.runCost H P x s) K (L.collAt H P x s)
    (fun ω rc rc' g hg => collAt_of_conflictAt hbind hs P ω rc rc' g hg)
    (fun ω rc rc' u v huv => L.collAt_collision H P x s ω rc rc' huv)
  rw [Fintype.card_fin] at h
  exact (L.pr_offReg_le_off H P x).trans h

/-- **The first-hit finder's expected calls** are at most `E[(1 + |sample|)·c]`, whatever `H` is: `c` a run's calls
with every session of its sample extracted (`runCost`), `|sample|` the gates the proof blocks of its coins' sample
need. -/
theorem ex_cost_finderReg_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Ex (fun sd => ((L.finderReg H P x s needs K).cost sd : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.runCost H P x s q.1 q.2 : ℕ) : ℝ)) :=
  Verity.FirstHit.ex_cost_sampleFinder_le (L.opTable H P x)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (L.runCost H P x s) K (L.finderReg H P x s needs K).out

/-- **`KSReg`'s first half with the first-hit finder's success in it, with no premise on `H`.** On a statement in
scope, an accepted run samples a proof block of the committed trace's incorrect set with probability at most
`κ_s + √(k·γ_s·Pr[B]) + 2·Pr[F_K finds] + N/(K + 1) + Pr[B_conf]`, where `εreg` has `N·√Pr[B_conf]` for the third
and fourth terms.

* `F_K` is the first-hit finder of the cap `K` (`finderReg`), and `Pr[F_K finds]` the probability that it outputs a
  collision of `H`. Its expected calls are at most `E[(1 + |sample|)·c]` (`ex_cost_finderReg_le`).
  `pr_regHit_le_needs_ex_cap` is this statement and that one under `ExpectedCR` at `F_K`.
* The finder's cost charges a further run only when the run's sample holds a gate still waited for. That its output
  reads no other run needs that a reveal for proof block `u` opens, under the statement's roots, only gates of
  `needs u`, which this statement does not take. -/
theorem pr_regHit_le_finder {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * (L.finderReg H P x s needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1)) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_finder hbind hs P needs K))

/-- **The off-trace mass with the by-position finder's success in it**, with no premise on `H`:
`2·Pr[the finder outputs a collision] + N/(K + 1)`, at the by-position first-hit finder of the cap `K`
(`finderRegAt`), `N` the program's gates. -/
theorem pr_offReg_le_finderAt {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) (K : ℕ) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffReg H P x q.1 q.2.1 q.2.2) ≤
      2 * (L.finderRegAt H P x s needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_at_le_finder H (L.opTable H P x)
    (vstar := fun (ω : Ω) (g : Fin (L.nG x)) => L.plur H P x ω g) (fun ω => L.isPlurality_plur H P x ω)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (L.runCost H P x s)
    (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val ω rc.1 rc.2) K
    (fun (ω : Ω) (rc rc' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.collBy H P x s needs (ω, rc) (ω, rc') g.val)
    (fun ω rc rc' g hg => by
      obtain ⟨h1, h2, hne⟩ := hg
      exact collBy_of_conflictAt hbind hs P needs hpos (ω, rc) (ω, rc') g.val h1 h2 hne)
    (fun ω rc rc' g u v huv => L.collBy_collision H P x s needs (ω, rc) (ω, rc') g.val huv)
  rw [Fintype.card_fin] at h
  exact (L.pr_offReg_le_off H P x).trans h

/-- **The by-position finder's expected calls** are at most `E[c + Σ_{g ∈ sample} c_g]`, whatever `H` is: `c` a
run's calls with every session of its sample extracted (`runCost`), and `c_g` its calls with only the sessions that
have a proof block needing gate `g` extracted (`callsBy`). -/
theorem ex_cost_finderRegAt_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Ex (fun sd => ((L.finderRegAt H P x s needs K).cost sd : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.runCost H P x s q.1 q.2 +
          ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  Verity.FirstHit.ex_cost_sampleFinderAt_le (L.opTable H P x)
    (fun (_ : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x rc.1, g.val ∈ needs u))
    (L.runCost H P x s)
    (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val ω rc.1 rc.2)
    (fun (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.callsBy H P x s needs g.val ω rc.1 rc.2)
    (fun _ _ _ => le_rfl) K (L.finderRegAt H P x s needs K).out

/-- **`KSReg`'s first half with the by-position finder's success in it, with no premise on `H`.** On a statement in
scope, where the reveals for a proof block `u` open, under the statement's roots, only gates of `needs u`, an
accepted run samples a proof block of the committed trace's incorrect set with probability at most
`κ_s + √(k·γ_s·Pr[B]) + 2·Pr[F_K finds] + N/(K + 1) + Pr[B_conf]`, where `εreg` has `N·√Pr[B_conf]` for the third
and fourth terms.

`F_K` is the by-position first-hit finder of the cap `K` (`finderRegAt`), and `Pr[F_K finds]` the probability that
it outputs a collision of `H`. Its expected calls are at most `E[c + Σ_{g ∈ sample} c_g]`
(`ex_cost_finderRegAt_le`). `pr_regHit_le_needs_at_cap` is this statement and that one under `ExpectedCR` at
`F_K`. -/
theorem pr_regHit_le_finderAt {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    (hSK : L.SessionKS H) {x : Msg} (hsc : L.Scope x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.1 = s.roots → p.2.pos ∈ needs u) (K : ℕ) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.Freg H P x p.1) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) +
        (2 * (L.finderRegAt H P x s needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1)) +
        (Bconf H L Ω P x).Pr H :=
  pr_regHit_le_of_offReg hbind hS hSK hsc Ω P
    ((Pr_mono fun _ h => h.2).trans (pr_offReg_le_finderAt hbind hs P needs hpos K))

end Compiled

end Claims.SamplingGeneric
