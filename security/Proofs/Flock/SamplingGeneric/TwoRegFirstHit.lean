module

public import Proofs.Flock.SamplingGeneric.TwoRegX
public import Proofs.Flock.SamplingGeneric.RunExtAt
public import Proofs.Core.SamplingGeneric.FirstHit
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
# The first-hit finder at two stages

`eventTwoX_le` (`SamplingGeneric/TwoRegX.lean`) bounds the event of `KSRegTwoX` by `ε_reg,2,X`, whose two rewinding
terms are square roots: `N·√Pr[B_conf,X]` for a run that opens a gate off the boundary trace read at the statement
(`Compiled.regValX`), and `N·√Pr[B_conf,2]` for one that opens a gate off a replay block's committed interior
(`CompiledTwo.iregVal`). This file bounds the same two events at the same two traces linearly, by the first-hit finder
(`Proofs.Core.SamplingGeneric.FirstHit`), and the event of `KSRegTwoX` with those two terms in place of the square
roots. No statement of the definitions tree changes.

What the two finders keep and what they redraw:

* **The boundary** (`Compiled.finderX`, `Compiled.pr_offRegX_le_needs_ex`). The boundary trace read at the statement
  is the plurality over the prover's tape, the run's coins and the extractor's coins together, so the finder keeps
  nothing and redraws all three: a further run is at a fresh tape. Two runs' openings of one gate under the
  statement's roots that differ are a collision whichever tapes opened them (`collX`, `B_conf,X`'s output).
* **The interior** (`CompiledTwo.finderI`, `CompiledTwo.pr_offIreg_le_needs_ex`). A replay block's committed interior
  is the plurality over the second stage's coins and the extractor's, at the prover's tape and the first stage's
  coins, so the finder keeps the tape and the run's coins off the second stage and redraws the second stage's coins
  and the extractor's. The two runs it compares share the replay block's interior roots (`Sessions.interior`), so two
  openings of one gate under them that differ are a collision (`icoll`, `B_conf,2`'s output). The sample of a further
  run reads the kept coins and the redrawn ones: it is the run's sample at the kept coins with the second stage
  redrawn.

Both samples are the gates the proof blocks of a run's sample need, under one hypothesis the contract does not
state: a reveal for proof block `u` opens only gates of `needs u`, under whichever roots (`hpos`). It is a hypothesis
of each lemma here and no clause of `Sessions`.

* `Compiled.pr_offRegX_le_needs_ex`, `CompiledTwo.pr_offIreg_le_needs_ex`: each off-trace event has probability at
  most `2·E[(1 + |sample|)·c]/2^256`, under `ExpectedCR` at its finder of every cap: `c` a run's calls (`callsAt`) and
  the expectation over the prover's tape, the run's coins and the extractor's. Coins at which the sample is large
  count by their probability and their run's calls. Their `_cap` forms take the premise at one cap's finder and add
  `N/(K + 1)`, `N` the program's gates.
* `CompiledTwo.eventTwo_split_acc`: the event's union (`eventTwo_split`) with acceptance kept in both off-trace terms.
* `CompiledTwo.eventTwo_le_of_off`, `CompiledTwo.eventTwoX_le_of_off`: the event's bound from any bounds on the three
  binding terms, at any boundary trace, and at the boundary trace read at the statement.
* `CompiledTwo.eventTwoX_le_firstHit_cap`, `CompiledTwo.eventTwoX_le_firstHit`: **the event of `KSRegTwoX`**, at
  most `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[(1 + |sample|)·c]/2^256`, with `N/(KX + 1) + N/(KI + 1)` more at
  one cap for each finder.
* `Compiled.ex_needs_le`: the expectation at a bound `w` on the sample outside a set of coins,
  `(1 + w)·E[c] + (1 + N)·E[c at those coins]`.
* `CompiledTwo.eventTwoX_le_firstHit_of_card`, `CompiledTwo.eventTwoX_le_firstHit_of_card_outside`: the event's bound
  with the sample's size a hypothesis, `4·(1 + w)·E[c]/2^256` where every coins' sample needs at most `w` gates, and
  with `ex_needs_le`'s second term where only the coins outside a set do.

**What a run costs.** `callsAt` is a whole run's calls: the prover's, the verifier's, and the extraction's of every
session of the run's sample (`runExt`). `finderX` and `finderI` pay it at the first run and at each further run they
make, whichever gate they wait for: they extract every session of a further run.

* **By position** (`Compiled.finderXAt`, `CompiledTwo.finderIAt`, `CompiledTwo.eventTwoX_le_firstHit_at_cap`,
  `CompiledTwo.eventTwoX_le_firstHit_at`): the finders that, at a further run, extract for each gate they wait for
  only the sessions with a proof block that needs the gate (`Proofs.Flock.SamplingGeneric.RunExtAt`). The event of
  `KSRegTwoX` is then at most `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{g ∈ sample} c_g]/2^256`: `c` the
  first run's calls, read whole, and `c_g` a run's calls with only those sessions extracted. The expectation is at
  most the whole-run one (`Compiled.ex_callsBy_le`).

* **With no premise** (`CompiledTwo.eventTwoX_le_finder`, `CompiledTwo.eventTwoX_le_finderAt`): the event of
  `KSRegTwoX` with the two finders' successes in it,
  `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 2·Pr[F_X finds] + 2·Pr[F_I finds] + N/(KX + 1) + N/(KI + 1)`, and the
  finders' expected calls as statements of their own (`Compiled.ex_cost_finderX_le`, `CompiledTwo.ex_cost_finderI_le`
  and their `At` forms). The statements under `ExpectedCR` are these together.

**What the bound is small under.** Three premises: the linear expected-time bound (`ExpectedCR`) at the two first-hit
finders, hypotheses of the statements here but those with no premise; the same at `B_conf,X`, whose success
`Pr[B_conf,X]` is a term of the bound; and the strict birthday bound (`StrictBirthday`) at `B`, whose success is
under the square root `√(k·γ_s·Pr[B])`, where the linear bound leaves the square root of a linear term. The
expectation `E[(1 + |sample|)·c]` is joint: a bound on a run's mean calls at each coins bounds it by one more than
the mean sample times that bound, and a bound on the mean over all coins does not.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo ownLog forkOn wrongAllAt wrongAllAt_subset gateBlock)

/-! ## Averages -/

/-- A probability over nothing kept and a point is the probability over the point. -/
theorem Pr_unit_prod {S : Type} [Fintype S] (p : S → Prop) : Pr (fun q : Unit × S => p q.2) = Pr p := by
  rw [Verity.FirstHit.pr_tape fun (_ : Unit) (a : S) => p a, Ex_const]

/-- An average over nothing kept and a point is the average over the point. -/
theorem Ex_unit_prod {S : Type} [Fintype S] (f : S → ℝ) : Ex (fun q : Unit × S => f q.2) = Ex f := by
  rw [Verity.FirstHit.ex_tape fun (_ : Unit) (a : S) => f a, Ex_const]

/-- **Keeping the coins off `I`**: an average over a tape, a run's coins, fresh coins on `I` and further coins, of a
function of the tape, the run's coins with the fresh ones on `I`, and the further coins, is the average over the tape,
a run's coins and the further coins. -/
theorem Ex_keep_forkOn {k : ℕ} {Coin C Ω : Type} [Fintype Coin] [Nonempty Coin] [Fintype C] [Fintype Ω]
    (I : Finset (Fin k)) (F : Ω → (Fin k → Coin) → C → ℝ) :
    Ex (fun q : (Ω × (Fin k → Coin)) × ((Fin k → Coin) × C) => F q.1.1 (forkOn I q.1.2 q.2.1) q.2.2) =
      Ex (fun q : Ω × ((Fin k → Coin) × C) => F q.1 q.2.1 q.2.2) := by
  have hfork : ∀ ω : Ω, Ex (fun r : Fin k → Coin => Ex fun ec : (Fin k → Coin) × C => F ω (forkOn I r ec.1) ec.2) =
      Ex (fun rc : (Fin k → Coin) × C => F ω rc.1 rc.2) := by
    intro ω
    calc Ex (fun r : Fin k → Coin => Ex fun ec : (Fin k → Coin) × C => F ω (forkOn I r ec.1) ec.2)
        = Ex (fun r : Fin k → Coin => Ex fun e : Fin k → Coin => Ex fun c : C => F ω (forkOn I r e) c) :=
          congrArg Ex (funext fun r =>
            Verity.FirstHit.ex_tape fun (e : Fin k → Coin) (c : C) => F ω (forkOn I r e) c)
      _ = Ex (fun re : (Fin k → Coin) × (Fin k → Coin) => Ex fun c : C => F ω (forkOn I re.1 re.2) c) :=
          (Verity.FirstHit.ex_tape fun (r e : Fin k → Coin) => Ex fun c : C => F ω (forkOn I r e) c).symm
      _ = Ex (fun R : Fin k → Coin => Ex fun c : C => F ω R c) :=
          Sampling.Ex_forkOn I fun R : Fin k → Coin => Ex fun c : C => F ω R c
      _ = Ex (fun rc : (Fin k → Coin) × C => F ω rc.1 rc.2) :=
          (Verity.FirstHit.ex_tape fun (R : Fin k → Coin) (c : C) => F ω R c).symm
  calc Ex (fun q : (Ω × (Fin k → Coin)) × ((Fin k → Coin) × C) => F q.1.1 (forkOn I q.1.2 q.2.1) q.2.2)
      = Ex (fun t : Ω × (Fin k → Coin) => Ex fun ec : (Fin k → Coin) × C => F t.1 (forkOn I t.2 ec.1) ec.2) :=
        Verity.FirstHit.ex_tape fun (t : Ω × (Fin k → Coin)) (ec : (Fin k → Coin) × C) =>
          F t.1 (forkOn I t.2 ec.1) ec.2
    _ = Ex (fun ω : Ω => Ex fun r : Fin k → Coin => Ex fun ec : (Fin k → Coin) × C => F ω (forkOn I r ec.1) ec.2) :=
        Verity.FirstHit.ex_tape fun (ω : Ω) (r : Fin k → Coin) =>
          Ex fun ec : (Fin k → Coin) × C => F ω (forkOn I r ec.1) ec.2
    _ = Ex (fun ω : Ω => Ex fun rc : (Fin k → Coin) × C => F ω rc.1 rc.2) := congrArg Ex (funext hfork)
    _ = Ex (fun q : Ω × ((Fin k → Coin) × C) => F q.1 q.2.1 q.2.2) :=
        (Verity.FirstHit.ex_tape fun (ω : Ω) (rc : (Fin k → Coin) × C) => F ω rc.1 rc.2).symm

/-- `Ex_keep_forkOn` at an event. -/
theorem Pr_keep_forkOn {k : ℕ} {Coin C Ω : Type} [Fintype Coin] [Nonempty Coin] [Fintype C] [Fintype Ω]
    (I : Finset (Fin k)) (p : Ω → (Fin k → Coin) → C → Prop) :
    Pr (fun q : (Ω × (Fin k → Coin)) × ((Fin k → Coin) × C) => p q.1.1 (forkOn I q.1.2 q.2.1) q.2.2) =
      Pr (fun q : Ω × ((Fin k → Coin) × C) => p q.1 q.2.1 q.2.2) := by
  classical
  calc Pr (fun q : (Ω × (Fin k → Coin)) × ((Fin k → Coin) × C) => p q.1.1 (forkOn I q.1.2 q.2.1) q.2.2)
      = Ex (fun q : (Ω × (Fin k → Coin)) × ((Fin k → Coin) × C) =>
          if p q.1.1 (forkOn I q.1.2 q.2.1) q.2.2 then (1 : ℝ) else 0) :=
        (Ex_ind fun q : (Ω × (Fin k → Coin)) × ((Fin k → Coin) × C) =>
          p q.1.1 (forkOn I q.1.2 q.2.1) q.2.2).symm
    _ = Ex (fun q : Ω × ((Fin k → Coin) × C) => if p q.1 q.2.1 q.2.2 then (1 : ℝ) else 0) :=
        Ex_keep_forkOn I fun (ω : Ω) (R : Fin k → Coin) (c : C) => if p ω R c then (1 : ℝ) else 0
    _ = Pr (fun q : Ω × ((Fin k → Coin) × C) => p q.1 q.2.1 q.2.2) :=
        Ex_ind fun q : Ω × ((Fin k → Coin) × C) => p q.1 q.2.1 q.2.2

/-- The plurality of a table of optional bits (`plurAt`) is a plurality table of it, on the positions below `N`. -/
theorem isPlurality_plurAt {S : Type} [Fintype S] [Nonempty S] (N : ℕ) (op : S → ℕ → Option Bool) :
    FlockSoundness.Rewinding.IsPlurality (fun (a : S) (g : Fin N) => op a g) fun g => plurAt op g := by
  classical
  have hcard : (0 : ℝ) < Fintype.card S := by exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin N) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g b) : ℝ) =
        Pr (fun a : S => op a g = some b) * Fintype.card S := by
    intro g b
    rw [Pr_eq_card_of_mem (fun a : S => op a g = some b)
      (FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g b)
      fun a => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  intro g v
  have h := le_plur (fun b => Pr fun a : S => op a g = some b) v
  have h' : (#(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g v) : ℝ) ≤
      #(FlockSoundness.Rewinding.openedTo (fun (a : S) (g : Fin N) => op a g) g (plurAt op g)) := by
    rw [hcnt, hcnt]
    exact mul_le_mul_of_nonneg_right h hcard.le
  exact_mod_cast h'

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## What a run's extractions keep, by proof block -/

/-- **A run's extractions keep only what its own sample's reveals open**: where a reveal for proof block `u` opens only
gates of `needs u`, an opening the run's extractions keep is at a gate some proof block of the run's sample needs. -/
theorem kept_pos_mem_needs (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    {q : List Msg × Sampling.Opening Bool} (hq : q ∈ L.kept H P x s ω r c) :
    ∃ u ∈ L.checked x r, q.2.pos ∈ needs u := by
  unfold kept at hq
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
      exact ⟨u', List.mem_of_getElem? hj, hpos _ _ _ q (Sampling.checkRevealed_mem H _ L.commit _ q hu').1⟩
  · simp [RP.eval] at hiu

/-- A gate a run's extractions open under the statement's roots is one some proof block of the run's sample needs. -/
theorem opensAt_isSome_needs (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    (g : ℕ) (hg : (L.opensAt H P x ω r c g).isSome) : ∃ u ∈ L.checked x r, g ∈ needs u := by
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hg
  rw [L.opensAt_eq H P x ω hs] at hb
  obtain ⟨a, ha, -, ha2, -, -⟩ := val_some_mem hb
  obtain ⟨u, hu, hmem⟩ := L.kept_pos_mem_needs H P x s ω needs hpos r c ha
  exact ⟨u, hu, ha2 ▸ hmem⟩

/-! ## A run's calls, and the pair two runs give -/

/-- **A run's calls**: both parties', and its extractions', their reruns' included; one run's share of `B_conf,X`'s
and of `B_conf,2`'s cost. -/
noncomputable def callsAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) : ℕ :=
  (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length +
    ((L.runExt x s r (run H P L.V x ω r).seen c).eval H (run H P L.V x ω)).2.1

/-- **The pair two runs' extractions give**, each run at its own tape: the first collision among the calls the two
runs' extractions make themselves, `B_conf,X`'s output at the two runs. -/
noncomputable def collX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (q q' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) : Option (Msg × Msg) :=
  findColl (ownLog H (run H P L.V x q.1) (L.runExt x s q.2.1 (run H P L.V x q.1 q.2.1).seen q.2.2) ++
    ownLog H (run H P L.V x q'.1) (L.runExt x s q'.2.1 (run H P L.V x q'.1 q'.2.1).seen q'.2.2))

/-- What `collX` outputs is a collision of `H`. -/
theorem collX_collision (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (s : AuditStatement)
    (q q' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) {u v : Msg} (h : L.collX H P x s q q' = some (u, v)) :
    Collision H u v :=
  findColl_spec H _
    (fun p hp => (List.mem_append.1 hp).elim (Sampling.ownLog_valid H _ _ p) (Sampling.ownLog_valid H _ _ p)) u v h

/-- Two runs, each at its own tape, whose extractions open a gate two ways give `collX` a pair
(`bconfX_of_disagree`). -/
theorem collX_of_conflictAt {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω)
    (q q' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x))
    (h : FlockSoundness.Rewinding.ConflictAt
      (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
      (q, q') g) :
    (L.collX H P x s q q').isSome := by
  obtain ⟨h1', h2', hne'⟩ := h
  have h1 : (L.opensAt H P x q.1 q.2.1 q.2.2 g).isSome := h1'
  have h2 : (L.opensAt H P x q'.1 q'.2.1 q'.2.2 g).isSome := h2'
  have hne : L.opensAt H P x q.1 q.2.1 q.2.2 g ≠ L.opensAt H P x q'.1 q'.2.1 q'.2.2 g := hne'
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  rw [L.opensAt_eq H P x q.1 hs] at hb1
  rw [L.opensAt_eq H P x q'.1 hs] at hb2
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := val_some_mem hb1
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := val_some_mem hb2
  obtain ⟨u, v, huv, -⟩ := bconfX_of_disagree hbind s P q.1 q'.1 q.2 q'.2 (Or.inl ha) (Or.inr ha')
    (ha1.trans ha1'.symm) (ha2.trans ha2'.symm) (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))
  exact Option.isSome_iff_exists.2 ⟨(u, v), huv⟩

/-! ## The boundary: the finder at the trace read at the statement -/

/-- **The first-hit finder at the boundary trace read at the statement**, at cap `K`. It keeps nothing: it draws a
tape, a run's coins and the extractor's, then `K` further tapes, runs and extractor's coins, and makes a further run
only when its sample (the gates the proof blocks of its coins' sample need) holds a gate the first run opened that no
further run has opened yet. It outputs `collX` at the first run and a further run that is the first to open such a
gate. Its cost is the calls of the runs it makes (`callsAt`). -/
noncomputable def finderX (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) : Finder :=
  Verity.FirstHit.sampleFinder
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2) K
    fun q : Unit × ((Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
        (Fin K → Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))) =>
      Verity.FirstHit.outOf
        (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
        (L.collX H P x s) K q.2.1 q.2.2

/-- **The mass off the boundary trace read at the statement, linearly, at one cap.** Where a reveal for proof block
`u` opens only gates of `needs u`, and the first-hit finder of the cap `K` meets the linear expected-time bound, a run
at a uniform tape opens a gate of the program off the trace read at the statement with probability at most
`2·E[(1 + |sample|)·c]/2^256 + N/(K + 1)`: the expectation over the tape, the run's coins and the extractor's of the
run's calls times one more than the gates its coins' sample needs, and `N` the program's gates. -/
theorem pr_offRegX_le_needs_ex_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderX H P x s needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_sample_le_ex_cap_of_coll H
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (vstar := fun (_ : Unit) (g : Fin (L.nG x)) => L.plurX H P x g)
    (fun _ => isPlurality_plurAt (L.nG x)
      fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun _ a g hg => decide_eq_true (L.opensAt_isSome_needs H P x a.1 hs needs hpos a.2.1 a.2.2 g hg))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2) K
    (fun _ => L.collX H P x s)
    (fun _ a a' g hg => collX_of_conflictAt hbind hs P a a' g hg)
    (fun _ a a' u v huv => L.collX_collision H P x s a a' huv) hCR
  rw [Fintype.card_fin] at h
  have hP := Pr_unit_prod (fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
    FlockSoundness.Rewinding.Off
      (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
      (fun g : Fin (L.nG x) => L.plurX H P x g) a g)
  have hE := Ex_unit_prod (fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
    (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))) *
      L.callsAt H P x s a.1 a.2.1 a.2.2 : ℕ) : ℝ))
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

/-- **The mass off the boundary trace read at the statement, linearly**: `pr_offRegX_le_needs_ex_cap` under the
linear expected-time bound at the first-hit finder of every cap, with nothing left of the cap's term,
`2·E[(1 + |sample|)·c]/2^256`. -/
theorem pr_offRegX_le_needs_ex {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hCR : ∀ K : ℕ, ExpectedCR H (L.finderX H P x s needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  Verity.FirstHit.le_of_forall_cap fun K => pr_offRegX_le_needs_ex_cap hbind hs P needs hpos K (hCR K)

/-- **The expectation at a bound on the sample outside a set of coins.** Where the sample of every coins outside `bad`
needs at most `w` gates, `E[(1 + |sample|)·c] ≤ (1 + w)·E[c] + (1 + N)·E[c at the coins in bad]`, `N` the program's
gates: coins whose sample is large count by their probability and their run's calls. -/
theorem ex_needs_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) (bad : (Fin (L.V.k x) → Coin) → Prop) [DecidablePred bad] {w : ℕ}
    (hw : ∀ r, ¬ bad r → #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x r, g.val ∈ needs u)) ≤ w) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤
      (1 + (w : ℝ)) * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ)) +
        (1 + (L.nG x : ℝ)) * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          if bad q.2.1 then (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ) else 0) := by
  rw [← Ex_mul_left, ← Ex_mul_left, ← Ex_add]
  refine Ex_mono fun q => ?_
  have hN : #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)) ≤ L.nG x :=
    (card_filter_le _ _).trans (card_univ.trans (Fintype.card_fin _)).le
  have hc : (0 : ℝ) ≤ (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ) := Nat.cast_nonneg _
  have hw0 : (0 : ℝ) ≤ 1 + (w : ℝ) := by positivity
  have hN0 : (0 : ℝ) ≤ 1 + (L.nG x : ℝ) := by positivity
  push_cast
  by_cases hb : bad q.2.1
  · rw [if_pos hb]
    have h1 : (1 + (#(univ.filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)) : ℝ)) ≤ 1 + (L.nG x : ℝ) := by
      exact_mod_cast Nat.add_le_add_left hN 1
    have h2 := mul_le_mul_of_nonneg_right h1 hc
    have h3 := mul_nonneg hw0 hc
    linarith
  · rw [if_neg hb]
    have h1 : (1 + (#(univ.filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)) : ℝ)) ≤ 1 + (w : ℝ) := by
      exact_mod_cast Nat.add_le_add_left (hw q.2.1 hb) 1
    have h2 := mul_le_mul_of_nonneg_right h1 hc
    linarith

/-! ## The boundary, by position -/

/-- **The by-position first-hit finder at the boundary trace read at the statement**, at cap `K`. Its seed, its
sample and its waits are `finderX`'s, and its first run is read whole (`callsAt`). At a further run it pays, for each
gate still waited for that the run's sample holds, a run at that tape and those coins with only the sessions that
have a proof block needing the gate extracted (`callsBy`). It outputs `collBy` at the first run, a further run that
is the first to open some gate the first run opened, and that gate. -/
noncomputable def finderXAt (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) : Finder :=
  Verity.FirstHit.sampleFinderAt
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2) K
    fun q : Unit × ((Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) ×
        (Fin K → Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x))) =>
      Verity.FirstHit.outOfAt
        (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
        (fun (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
          L.collBy H P x s needs a a' g.val) K q.2.1 q.2.2

/-- **The mass off the boundary trace read at the statement, by position, at one cap**:
`2·E[c + Σ_{g ∈ sample} c_g]/2^256 + N/(K + 1)`, under `ExpectedCR` at the by-position finder of the cap `K`: `c` a
run's calls with every session extracted (`callsAt`), `c_g` its calls with only the sessions that can open gate `g`
extracted (`callsBy`), `N` the program's gates. -/
theorem pr_offRegX_le_needs_at_cap {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderXAt H P x s needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_at_le_ex_cap H
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (vstar := fun (_ : Unit) (g : Fin (L.nG x)) => L.plurX H P x g)
    (fun _ => isPlurality_plurAt (L.nG x)
      fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun _ a g hg => decide_eq_true (L.opensAt_isSome_needs H P x a.1 hs needs hpos a.2.1 a.2.2 g hg))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2)
    (fun _ _ _ => le_rfl) K
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
      ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u)),
        L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2 : ℕ) : ℝ))
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

/-- **The by-position expectation is at most the whole-run one**: `E[c + Σ_{g ∈ sample} c_g] ≤ E[(1 + |sample|)·c]`,
a run's calls at one gate being at most its calls with every session extracted (`callsBy_le`). -/
theorem ex_callsBy_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (needs : ℕ → Finset ℕ) :
    Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  Ex_mono fun q => by
    have h : ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
        L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 ≤
        #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 :=
      (Finset.sum_le_card_nsmul _ _ _ fun g _ => L.callsBy_le H P x s needs g.val q.1 q.2.1 q.2.2).trans
        (le_of_eq (smul_eq_mul _ _))
    have h2 : L.callsAt H P x s q.1 q.2.1 q.2.2 +
        ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
          L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 ≤
        (1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 := by
      rw [Nat.add_mul, Nat.one_mul]
      exact Nat.add_le_add_left h _
    exact Nat.cast_le.2 h2

/-! ## The boundary, with no premise -/

/-- A run that opens a gate of the program off the trace read at the statement opens it off the plurality table of
the boundary's finders. -/
theorem pr_offRegX_le_off (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      Pr (fun q : Unit × (Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off
          (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
            L.opensAt H P x a.1 a.2.1 a.2.2 g)
          (fun g : Fin (L.nG x) => L.plurX H P x g) q.2 g) := by
  have hP := Pr_unit_prod (fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
    FlockSoundness.Rewinding.Off
      (fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
      (fun g : Fin (L.nG x) => L.plurX H P x g) a g)
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
  exact hoff.trans hP.symm.le

/-- **The mass off the boundary trace read at the statement, with the first-hit finder's success in it**, with no
premise on `H`: `2·Pr[the finder outputs a collision] + N/(K + 1)`, at the boundary's first-hit finder of the cap `K`
(`finderX`), `N` the program's gates. -/
theorem pr_offRegX_le_finder {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (K : ℕ) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * (L.finderX H P x s needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_sample_le_finder H
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (vstar := fun (_ : Unit) (g : Fin (L.nG x)) => L.plurX H P x g)
    (fun _ => isPlurality_plurAt (L.nG x)
      fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2) K
    (fun _ => L.collX H P x s)
    (fun _ a a' g hg => collX_of_conflictAt hbind hs P a a' g hg)
    (fun _ a a' u v huv => L.collX_collision H P x s a a' huv)
  rw [Fintype.card_fin] at h
  exact (L.pr_offRegX_le_off H P x).trans h

/-- **The boundary's first-hit finder's expected calls** are at most `E[(1 + |sample|)·c]`, whatever `H` is: `c` a
run's calls with every session of its sample extracted (`callsAt`), `|sample|` the gates the proof blocks of its
coins' sample need. -/
theorem ex_cost_finderX_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Ex (fun sd => ((L.finderX H P x s needs K).cost sd : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  (Verity.FirstHit.ex_cost_sampleFinder_le
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2) K
    (L.finderX H P x s needs K).out).trans_eq
    (Ex_unit_prod fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))) *
        L.callsAt H P x s a.1 a.2.1 a.2.2 : ℕ) : ℝ))

/-- **The mass off the boundary trace read at the statement, with the by-position finder's success in it**, with no
premise on `H`: `2·Pr[the finder outputs a collision] + N/(K + 1)`, at the boundary's by-position first-hit finder of
the cap `K` (`finderXAt`), `N` the program's gates. -/
theorem pr_offRegX_le_finderAt {H : Msg → Msg} {L : Compiled Coin A} (hbind : L.commit.Binds H) {x : Msg}
    {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (K : ℕ) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffRegX H P x q.1 q.2.1 q.2.2) ≤
      2 * (L.finderXAt H P x s needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_at_le_finder H
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (vstar := fun (_ : Unit) (g : Fin (L.nG x)) => L.plurX H P x g)
    (fun _ => isPlurality_plurAt (L.nG x)
      fun (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : ℕ) => L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2) K
    (fun (_ : Unit) (a a' : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.collBy H P x s needs a a' g.val)
    (fun _ a a' g hg => by
      obtain ⟨h1, h2, hne⟩ := hg
      exact collBy_of_conflictAt hbind hs P needs (fun u v w p hp _ => hpos u v w p hp) a a' g.val h1 h2 hne)
    (fun _ a a' g u v huv => L.collBy_collision H P x s needs a a' g.val huv)
  rw [Fintype.card_fin] at h
  exact (L.pr_offRegX_le_off H P x).trans h

/-- **The boundary's by-position finder's expected calls** are at most `E[c + Σ_{g ∈ sample} c_g]`, whatever `H`
is: `c` a run's calls with every session of its sample extracted (`callsAt`), and `c_g` its calls with only the
sessions that have a proof block needing gate `g` extracted (`callsBy`). -/
theorem ex_cost_finderXAt_le (H : Msg → Msg) (L : Compiled Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Ex (fun sd => ((L.finderXAt H P x s needs K).cost sd : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  (Verity.FirstHit.ex_cost_sampleFinderAt_le
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.opensAt H P x a.1 a.2.1 a.2.2 g)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u))
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) => L.callsAt H P x s a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2)
    (fun (_ : Unit) (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g : Fin (L.nG x)) =>
      L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2)
    (fun _ _ _ => le_rfl) K (L.finderXAt H P x s needs K).out).trans_eq
    (Ex_unit_prod fun a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      ((L.callsAt H P x s a.1 a.2.1 a.2.2 +
        ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x a.2.1, g.val ∈ needs u)),
          L.callsBy H P x s needs g.val a.1 a.2.1 a.2.2 : ℕ) : ℝ))

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The interior: the finder at a replay block's committed interior -/

/-- At a tape and first-stage coins, the committed interiors are a plurality table of the second stages' openings of
the program's gates under their replay blocks' interior roots. -/
theorem isPlurality_iplur (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Pρ : Partition) (r : Fin (L.V.k x) → Coin) :
    FlockSoundness.Rewinding.IsPlurality (L.iopTable H P x ω Pρ r) fun g => L.iplur H P x ω Pρ r g := by
  classical
  have hcard : (0 : ℝ) < Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    exact_mod_cast Fintype.card_pos
  have hcnt : ∀ (g : Fin (L.nG x)) (b : Bool),
      (#(FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g b) : ℝ) =
        Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.iop H P x ω Pρ r ec g = some b) *
          Fintype.card ((Fin (L.V.k x) → Coin) × L.SlotEC x) := by
    intro g b
    rw [Pr_eq_card_of_mem (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.iop H P x ω Pρ r ec g = some b)
      (FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g b)
      fun ec => mem_filter.trans (and_iff_right (mem_univ _)), div_mul_cancel₀ _ hcard.ne']
  intro g v
  have h : Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.iop H P x ω Pρ r ec g = some v) ≤
      Pr (fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        L.iop H P x ω Pρ r ec g = some (L.iplur H P x ω Pρ r g)) := by
    cases hgb : gateBlock Pρ g with
    | none => exact Pr_mono fun ec hec => by simp [iop, hgb] at hec
    | some ρ =>
      simp only [iop, iplur, hgb, Option.bind_some, Option.map_some, Option.getD_some]
      exact le_plur (fun b => Pr fun ec : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        L.iopensAt H P x ω (forkOn (L.stage2 x) r ec.1) ec.2 ρ g = some b) v
  have h' : (#(FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g v) : ℝ) ≤
      #(FlockSoundness.Rewinding.openedTo (L.iopTable H P x ω Pρ r) g (L.iplur H P x ω Pρ r g)) := by
    rw [hcnt, hcnt]
    exact mul_le_mul_of_nonneg_right h hcard.le
  exact_mod_cast h'

/-- A second stage whose extractions open a gate off a replay block's committed interior opens it off the plurality
table (`iplur`) of the second stages' openings at its tape and first stage. -/
theorem off_of_offIreg (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (Pρ : Partition) (r : Fin (L.V.k x) → Coin) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x)
    (h : L.OffIreg H P x ω Pρ (forkOn (L.stage2 x) r ec.1) ec.2) :
    ∃ g : Fin (L.nG x),
      FlockSoundness.Rewinding.Off (L.iopTable H P x ω Pρ r) (fun g => L.iplur H P x ω Pρ r g) ec g := by
  obtain ⟨g, hg, ρ, hgb, b, hb, hne⟩ := h
  have hiop : L.iop H P x ω Pρ r ec g = some b := by
    simp only [iop, hgb, Option.bind_some]
    exact hb
  have hpl' : L.iplur H P x ω Pρ r g = L.iregVal H P x ω r ρ g := by
    simp only [iplur, hgb, Option.map_some, Option.getD_some]
  refine ⟨⟨g, hg⟩, Option.isSome_iff_exists.2 ⟨b, hiop⟩, fun h => hne ?_⟩
  have h' : some b = some (L.iplur H P x ω Pρ r g) := hiop.symm.trans h
  rw [iregVal_forkOn]
  exact hpl'.symm.trans (Option.some.inj h').symm

/-- **The pair two second stages' extractions give**, on one tape and one first stage: `B_conf,2`'s output, `collX` at
the two runs. -/
noncomputable def icoll (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (r : Fin (L.V.k x) → Coin) (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) :
    Option (Msg × Msg) :=
  L.toCompiled.collX H P x s (ω, (forkOn (L.stage2 x) r ec.1, ec.2)) (ω, (forkOn (L.stage2 x) r ec'.1, ec'.2))

/-- What `icoll` outputs is a collision of `H`. -/
theorem icoll_collision (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg)
    (s : AuditStatement) (ω : Ω) (r : Fin (L.V.k x) → Coin) (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x)
    {u v : Msg} (h : L.icoll H P x s ω r ec ec' = some (u, v)) : Collision H u v :=
  L.toCompiled.collX_collision H P x s _ _ h

/-- Two second stages on one tape and one first stage whose extractions open a gate two ways, each under the interior
roots of the replay block that holds it, give `icoll` a pair: the two share those roots (`Sessions.interior`). -/
theorem icoll_of_conflictAt {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H)
    {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x))
    (h : FlockSoundness.Rewinding.ConflictAt (L.iopTable H P x ω Pρ r) (ec, ec') g) :
    (L.icoll H P x s ω r ec ec').isSome := by
  obtain ⟨h1', h2', hne'⟩ := h
  have h1 : (L.iopTable H P x ω Pρ r ec g).isSome := h1'
  have h2 : (L.iopTable H P x ω Pρ r ec' g).isSome := h2'
  have hne : L.iopTable H P x ω Pρ r ec g ≠ L.iopTable H P x ω Pρ r ec' g := hne'
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  simp only [iopTable, iop] at hb1 hb2
  obtain ⟨ρ, hgb, hb1'⟩ := Option.bind_eq_some_iff.1 hb1
  obtain ⟨ρ', hgb', hb2'⟩ := Option.bind_eq_some_iff.1 hb2
  rw [hgb, Option.some.injEq] at hgb'
  rw [← hgb'] at hb2'
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := iopensAt_some hs hb1'
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := iopensAt_some hs hb2'
  have hroots : a.1 = a'.1 := by
    rw [ha1, ha1', hS.interior x Ω P ω r ec.1 ρ, hS.interior x Ω P ω r ec'.1 ρ]
  obtain ⟨u, v, huv, -⟩ := Compiled.bconf_of_disagree hbind s P ω (forkOn (L.stage2 x) r ec.1, ec.2)
    (forkOn (L.stage2 x) r ec'.1, ec'.2) (Or.inl ha) (Or.inr ha') hroots (ha2.trans ha2'.symm)
    (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))
  exact Option.isSome_iff_exists.2 ⟨(u, v), huv⟩

/-- A gate a second stage's extractions open under a replay block's interior roots is one some proof block of the
run's sample needs. -/
theorem iop_isSome_needs (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (r : Fin (L.V.k x) → Coin)
    (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : ℕ) (hg : (L.iop H P x ω Pρ r ec g).isSome) :
    ∃ u ∈ L.checked x (forkOn (L.stage2 x) r ec.1), g ∈ needs u := by
  obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 hg
  simp only [iop] at hb
  obtain ⟨ρ, -, hb'⟩ := Option.bind_eq_some_iff.1 hb
  obtain ⟨a, ha, -, ha2, -, -⟩ := iopensAt_some hs hb'
  obtain ⟨u, hu, hmem⟩ :=
    L.toCompiled.kept_pos_mem_needs H P x s ω needs hpos (forkOn (L.stage2 x) r ec.1) ec.2 ha
  exact ⟨u, hu, ha2 ▸ hmem⟩

/-- **The first-hit finder at the committed interiors**, at cap `K`. It keeps a tape and a run's coins off the second
stage: it draws those, a second stage's coins and the extractor's, then `K` further second stages' and extractor's
coins, and makes a further run, at the kept tape and first stage, only when its sample (the gates the proof blocks of
its coins' sample need; it reads the kept coins and the redrawn ones) holds a gate the first run opened under a replay
block's interior roots that no further run has opened yet. It outputs `icoll` at the first run and a further run that
is the first to open such a gate. Its cost is the calls of the runs it makes (`callsAt`). -/
noncomputable def finderI (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (s : AuditStatement) (Pρ : Partition) (needs : ℕ → Finset ℕ) (K : ℕ) : Finder :=
  Verity.FirstHit.sampleFinder
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    fun q : (Ω × (Fin (L.V.k x) → Coin)) × (((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      Verity.FirstHit.outOf (L.iopTable H P x q.1.1 Pρ q.1.2) (L.icoll H P x s q.1.1 q.1.2) K q.2.1 q.2.2

/-- **The mass off the committed interiors, linearly, at one cap.** Where a reveal for proof block `u` opens only
gates of `needs u`, and the first-hit finder of the cap `K` meets the linear expected-time bound, a run at a uniform
tape opens a gate of the program, under the interior roots of the replay block of `Pρ` that holds it, off that block's
committed interior with probability at most `2·E[(1 + |sample|)·c]/2^256 + N/(K + 1)`: the expectation over the tape,
the run's coins and the extractor's of the run's calls times one more than the gates its coins' sample needs, and `N`
the program's gates. The contract's clause it reads is `Sessions.interior`. -/
theorem pr_offIreg_le_needs_ex_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderI H P x s Pρ needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_sample_le_ex_cap_of_coll H
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (vstar := fun (t : Ω × (Fin (L.V.k x) → Coin)) (g : Fin (L.nG x)) => L.iplur H P x t.1 Pρ t.2 g)
    (fun t => L.isPlurality_iplur H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun t ec g hg => decide_eq_true (L.iop_isSome_needs H P x t.1 hs Pρ needs hpos t.2 ec g hg))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    (fun t => L.icoll H P x s t.1 t.2)
    (fun t ec ec' g hg => icoll_of_conflictAt hbind hS hs Pρ P t.1 t.2 ec ec' g hg)
    (fun t ec ec' u v huv => L.icoll_collision H P x s t.1 t.2 ec ec' huv) hCR
  rw [Fintype.card_fin] at h
  have hP := Pr_keep_forkOn (L.stage2 x)
    (fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) => L.OffIreg H P x ω Pρ R c)
  have hE := Ex_keep_forkOn (L.stage2 x) (fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
    (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x R, g.val ∈ needs u))) *
      L.callsAt H P x s ω R c : ℕ) : ℝ))
  have hoff : Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.OffIreg H P x q.1.1 Pρ (forkOn (L.stage2 x) q.1.2 q.2.1) q.2.2) ≤
      Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off (L.iopTable H P x q.1.1 Pρ q.1.2)
          (fun g : Fin (L.nG x) => L.iplur H P x q.1.1 Pρ q.1.2 g) q.2 g) :=
    Pr_mono fun q hq => L.off_of_offIreg H P x q.1.1 Pρ q.1.2 q.2 hq
  exact (hP.symm.le.trans (hoff.trans h)).trans_eq
    (congrArg (fun e : ℝ => 2 * e / 2 ^ (256 : ℝ) + (L.nG x : ℝ) / ((K : ℝ) + 1)) hE)

/-- **The mass off the committed interiors, linearly**: `pr_offIreg_le_needs_ex_cap` under the linear expected-time
bound at the first-hit finder of every cap, with nothing left of the cap's term, `2·E[(1 + |sample|)·c]/2^256`. -/
theorem pr_offIreg_le_needs_ex {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hCR : ∀ K : ℕ, ExpectedCR H (L.finderI H P x s Pρ needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  Verity.FirstHit.le_of_forall_cap fun K =>
    pr_offIreg_le_needs_ex_cap hbind hS hs Pρ P needs hpos K (hCR K)

/-! ## The event, with acceptance kept in the off-trace terms -/

set_option synthInstance.maxSize 1024 in
/-- **The event at a tape, at a boundary trace, with acceptance kept**: `eventTwo_split` with the run's acceptance in
both off-trace terms, which `regTwo_split` has in hand. An accepted run that samples a proof block incorrect at the
boundary trace `tr` and its replay block's committed interior has the first such block's session unfound at its first
slot, or disagrees with itself, or accepts and is off `tr`, or accepts and is off the committed interior. -/
theorem eventTwo_split_acc {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (tr : ℕ → Option Bool) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ wrongAllAt s Pt d.part tr (L.iregVal H P x ω r)) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part tr (L.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) +
        (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
            ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr g ≠ some b) +
          Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
            L.OffIreg H P x ω d.part rc.1 rc.2))) := by
  have hIO := (Sampling.Pr_union_le
    (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2)
    (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      ((run H P L.V x ω rc.1).out = .acc ∧
          ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr g ≠ some b) ∨
        ((run H P L.V x ω rc.1).out = .acc ∧ L.OffIreg H P x ω d.part rc.1 rc.2))).trans
      (add_le_add le_rfl (Sampling.Pr_union_le
        (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
          ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr g ≠ some b)
        (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
          L.OffIreg H P x ω d.part rc.1 rc.2)))
  refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
    ∃ u ∈ L.checked x r, u ∈ wrongAllAt s Pt d.part tr (L.iregVal H P x ω r))).symm.trans_le
      (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl hIO)
  obtain ⟨hacc, v, hvl, hvF⟩ := hrc
  obtain ⟨i, hi, u, hiu, huF⟩ := L.selOf_some x (hS.slots x rc.1)
    (fun w hw => by rw [Compiled.nPU_eq hs hPt]; exact mem_range.1 (wrongAllAt_subset s Pt d.part _ _ hw)) hvl hvF
  obtain ⟨i₀, u₀, hlead, hu₀⟩ := Compiled.leadOf_some hiu u.isLt
  rcases regTwo_split hS hs hPt hd hin P ω rc.1 rc.2 hacc i u hiu tr huF i₀ u₀ hlead hu₀ with h | h | h | h
  · exact Or.inl ⟨hacc, i₀, by rw [Compiled.leadSel, hi, Option.bind_some, hlead], u₀, hu₀, h⟩
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl ⟨hacc, h⟩))
  · exact Or.inr (Or.inr (Or.inr ⟨hacc, h⟩))

set_option synthInstance.maxSize 1024 in
/-- **The event's bound from bounds on its three binding terms**, at a boundary trace that may read the tape: an
accepted run samples a proof block incorrect at the boundary trace and its replay block's committed interior with
probability at most `κ_s + √(k·γ_s·Pr[B]) + b_I + b_B + b_N`, where `b_I` bounds one run's own disagreement, `b_B`
the accepted runs that open a gate off the boundary trace and `b_N` the accepted runs that open a gate off a
committed interior. It is the proof of `eventTwo_le` with the three bounds as hypotheses; the contract's clauses it
reads are `fixed` and `runs` (the sessions' bound), `slots`, `shared` and `opens` (the deterministic half). -/
theorem eventTwo_le_of_off {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (tr : Ω → ℕ → Option Bool) {bI bB bN : ℝ}
    (hI : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => L.InConf H P x s ω rc.1 rc.2) ≤ bI)
    (hB : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      ∃ g < L.nG x, ∃ b, L.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ tr q.1 g ≠ some b) ≤ bB)
    (hN : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffIreg H P x q.1 d.part q.2.1 q.2.2) ≤ bN) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ wrongAllAt s Pt d.part (tr p.1) (L.iregVal H P x p.1 p.2)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + bI + bB + bN := by
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
  have hT : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) :=
    Compiled.pr_unfound_le_at H L.toCompiled P β hβ0 hβ
      (fun ω r => L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.iregVal H P x ω r))))
      (fun ω r e => by simp only [iregVal_forkOn]) fun ω r sl => hκ ω r sl
  have hB' : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr ω g ≠ some b) ≤ bB :=
    (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ tr q.1 g ≠ some b)).symm.trans_le hB
  have hN' : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧ L.OffIreg H P x ω d.part rc.1 rc.2) ≤ bN :=
    (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2)).symm.trans_le hN
  refine (Sampling.Pr_prod_eq_Ex _).trans_le
    ((Ex_mono fun ω => eventTwo_split_acc hS hs hPt hd hin P ω (tr ω)).trans ?_)
  rw [Ex_add, Ex_add, Ex_add]
  linarith

/-- **The event of `KSRegTwoX` from bounds on its two off-trace terms**: `eventTwo_le_of_off` at the boundary trace
read at the statement, with one run's own disagreement at `B_conf,X`'s first run. An accepted run samples a proof
block of `FregTwoX` with probability at most `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + b_B + b_N`: `b_B` where
`ε_reg,2,X` has `N·√Pr[B_conf,X]` and `b_N` where it has `N·√Pr[B_conf,2]`. -/
theorem eventTwoX_le_of_off {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) {bB bN : ℝ}
    (hB : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffRegX H P x q.1 q.2.1 q.2.2) ≤ bB)
    (hN : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => (run H P L.V x q.1 q.2.1).out = .acc ∧
      L.OffIreg H P x q.1 d.part q.2.1 q.2.2) ≤ bN) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        bB + bN := by
  simp only [FregTwoX_eq H L P hs hPt hd]
  exact eventTwo_le_of_off hA hS hs hPt hd hin P (fun _ => L.regValX H P x) (Compiled.pr_inConfX_le hbind hs P)
    hB hN

/-- **The event of `KSRegTwoX` with the first-hit finders' terms, at one cap for each finder.** On a statement in
scope, where a reveal for proof block `u` opens only gates of `needs u`, and the boundary's first-hit finder of the
cap `KX` and the interior's of the cap `KI` meet the linear expected-time bound, an accepted run samples a proof block
of `FregTwoX` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[(1 + |sample|)·c]/2^256 + N/(KX + 1) + N/(KI + 1)`.

* `|sample|` at a run's coins is the number of gates the proof blocks of the coins' sample need, and `c` is the run's
  calls with every session of its sample extracted (`callsAt`). Each finder pays `c` at each further run it makes,
  whichever gate it waits for.
* The expectation is over the prover's tape, the run's coins and the extractor's, joint in the sample's size and the
  calls: a bound on a run's mean calls at each coins bounds it by one more than the mean sample times that bound; a
  bound on the mean over all coins does not, since a prover may make its calls where the sample is large.
* The bound is small under three premises: the linear expected-time bound at the two first-hit finders, hypotheses
  here; the same at `B_conf,X`, whose success is the third term; and at `B`, whose success is under the square root,
  the strict birthday bound (`StrictBirthday`), the linear one leaving the square root of a linear term. -/
theorem eventTwoX_le_firstHit_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (KX KI : ℕ)
    (hX : ExpectedCR H (L.toCompiled.finderX H P x s needs KX))
    (hI : ExpectedCR H (L.finderI H P x s d.part needs KI)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        ((L.nG x : ℝ) / ((KX : ℝ) + 1) + (L.nG x : ℝ) / ((KI : ℝ) + 1)) := by
  have hB := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_needs_ex_cap hbind hs P needs hpos KX hX)
  have hN := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_needs_ex_cap hbind hS hs d.part P needs hpos KI hI)
  have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P hB hN
  have h4 : ∀ e : ℝ, 2 * e / 2 ^ (256 : ℝ) + 2 * e / 2 ^ (256 : ℝ) = 4 * e / 2 ^ (256 : ℝ) := fun e => by ring
  rw [← h4]
  linarith

/-- **The event of `KSRegTwoX` with the first-hit finders' terms** where `ε_reg,2,X` has its two square roots:
`eventTwoX_le_firstHit_cap` under the linear expected-time bound at the two first-hit finders of every cap, with
nothing left of the caps' terms. An accepted run samples a proof block of `FregTwoX` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[(1 + |sample|)·c]/2^256`: twice the expectation for the boundary and
twice for the interior. What `|sample|`, `c` and the expectation are, and the premises the bound is small under, are
`eventTwoX_le_firstHit_cap`'s. -/
theorem eventTwoX_le_firstHit {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderX H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderI H P x s d.part needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
            L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := by
  have hB := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_needs_ex hbind hs P needs hpos hX)
  have hN := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_needs_ex hbind hS hs d.part P needs hpos hI)
  have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P hB hN
  have h4 : ∀ e : ℝ, 2 * e / 2 ^ (256 : ℝ) + 2 * e / 2 ^ (256 : ℝ) = 4 * e / 2 ^ (256 : ℝ) := fun e => by ring
  rw [← h4]
  linarith

/-- **The event of `KSRegTwoX` at a bound on the sample.** `eventTwoX_le_firstHit` where every coins' sample needs at
most `w` gates: at most `κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·(1 + w)·E[c]/2^256`. `w` counts every gate a checked
proof block's reveal may open, those of the proof blocks every run checks among them. -/
theorem eventTwoX_le_firstHit_of_card {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderX H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderI H P x s d.part needs K)) {w : ℕ}
    (hw : ∀ r, #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x r, g.val ∈ needs u)) ≤ w) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * ((1 + (w : ℝ)) * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ))) /
          2 ^ (256 : ℝ) := by
  have h := eventTwoX_le_firstHit hA hbind hS hs hPt hd hin P needs hpos hX hI
  have h2 : Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) ≤
      (1 + (w : ℝ)) * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ)) := by
    rw [← Ex_mul_left]
    refine Ex_mono fun q => ?_
    push_cast
    have h1 : (1 + (#(univ.filter fun g : Fin (L.nG x) =>
        decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)) : ℝ)) ≤ 1 + (w : ℝ) := by
      exact_mod_cast Nat.add_le_add_left (hw q.2.1) 1
    exact mul_le_mul_of_nonneg_right h1 (Nat.cast_nonneg _)
  have h3 := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left h2 (show (0 : ℝ) ≤ 4 by norm_num))
    (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  linarith

/-- **The event of `KSRegTwoX` at a bound on the sample outside a set of coins.** `eventTwoX_le_firstHit` where the
sample of every coins outside `bad` needs at most `w` gates: at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·((1 + w)·E[c] + (1 + N)·E[c at the coins in bad])/2^256`, `N` the program's
gates. A design whose sample has no bound at every coins is read here: the coins past the bound count by their
probability and their run's calls. -/
theorem eventTwoX_le_firstHit_of_card_outside {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderX H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderI H P x s d.part needs K))
    (bad : (Fin (L.V.k x) → Coin) → Prop) [DecidablePred bad] {w : ℕ}
    (hw : ∀ r, ¬ bad r →
      #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x r, g.val ∈ needs u)) ≤ w) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * ((1 + (w : ℝ)) * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ)) +
          (1 + (L.nG x : ℝ)) * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            if bad q.2.1 then (L.callsAt H P x s q.1 q.2.1 q.2.2 : ℝ) else 0)) /
          2 ^ (256 : ℝ) := by
  have h := eventTwoX_le_firstHit hA hbind hS hs hPt hd hin P needs hpos hX hI
  have h2 := Compiled.ex_needs_le H L.toCompiled P x s needs bad hw
  have h3 := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left h2 (show (0 : ℝ) ≤ 4 by norm_num))
    (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  linarith

/-! ## The interior and the event, by position -/

/-- Two second stages on one tape and one first stage whose extractions open a gate two ways, each under the interior
roots of the replay block that holds it, give `collBy` a pair at that gate: the two share those roots
(`Sessions.interior`), and the second's opening is kept by its extractions at the gate, a reveal for proof block `u`
opening only gates of `needs u`. -/
theorem icollBy_of_conflictAt {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    (P : Prover Ω) (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (ω : Ω)
    (r : Fin (L.V.k x) → Coin) (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x))
    (h : FlockSoundness.Rewinding.ConflictAt (L.iopTable H P x ω Pρ r) (ec, ec') g) :
    (L.toCompiled.collBy H P x s needs (ω, (forkOn (L.stage2 x) r ec.1, ec.2))
      (ω, (forkOn (L.stage2 x) r ec'.1, ec'.2)) g.val).isSome := by
  obtain ⟨h1', h2', hne'⟩ := h
  have h1 : (L.iopTable H P x ω Pρ r ec g).isSome := h1'
  have h2 : (L.iopTable H P x ω Pρ r ec' g).isSome := h2'
  have hne : L.iopTable H P x ω Pρ r ec g ≠ L.iopTable H P x ω Pρ r ec' g := hne'
  obtain ⟨b1, hb1⟩ := Option.isSome_iff_exists.1 h1
  obtain ⟨b2, hb2⟩ := Option.isSome_iff_exists.1 h2
  rw [hb1, hb2] at hne
  simp only [iopTable, iop] at hb1 hb2
  obtain ⟨ρ, hgb, hb1'⟩ := Option.bind_eq_some_iff.1 hb1
  obtain ⟨ρ', hgb', hb2'⟩ := Option.bind_eq_some_iff.1 hb2
  rw [hgb, Option.some.injEq] at hgb'
  rw [← hgb'] at hb2'
  obtain ⟨a, ha, ha1, ha2, ha3, -⟩ := iopensAt_some hs hb1'
  obtain ⟨a', ha', ha1', ha2', ha3', -⟩ := iopensAt_some hs hb2'
  have hroots : a.1 = a'.1 := by
    rw [ha1, ha1', hS.interior x Ω P ω r ec.1 ρ, hS.interior x Ω P ω r ec'.1 ρ]
  have hc := Compiled.collBy_isSome hbind s P needs (ω, (forkOn (L.stage2 x) r ec.1, ec.2))
    (ω, (forkOn (L.stage2 x) r ec'.1, ec'.2)) ha ha' (fun u w hm => hpos u _ w a' hm) hroots
    (ha2.trans ha2'.symm) (by rw [ha3, ha3']; exact fun h => hne (congrArg some h))
  rw [ha2'] at hc
  exact hc

/-- **The by-position first-hit finder at the committed interiors**, at cap `K`. Its seed, its sample and its waits
are `finderI`'s, and its first run is read whole (`callsAt`). At a further run, at the kept tape and first stage, it
pays, for each gate still waited for that the run's sample holds, a run at those coins with only the sessions that
have a proof block needing the gate extracted (`callsBy`). It outputs `collBy` at the first run, a further run that
is the first to open some gate the first run opened under a replay block's interior roots, and that gate. -/
noncomputable def finderIAt (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (Pρ : Partition) (needs : ℕ → Finset ℕ) (K : ℕ) : Finder :=
  Verity.FirstHit.sampleFinderAt
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x)
        (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.toCompiled.callsBy H P x s needs g.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    fun q : (Ω × (Fin (L.V.k x) → Coin)) × (((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        (Fin K → (Fin (L.V.k x) → Coin) × L.SlotEC x)) =>
      Verity.FirstHit.outOfAt (L.iopTable H P x q.1.1 Pρ q.1.2)
        (fun (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
          L.toCompiled.collBy H P x s needs (q.1.1, (forkOn (L.stage2 x) q.1.2 ec.1, ec.2))
            (q.1.1, (forkOn (L.stage2 x) q.1.2 ec'.1, ec'.2)) g.val) K q.2.1 q.2.2

/-- **The mass off the committed interiors, by position, at one cap**:
`2·E[c + Σ_{g ∈ sample} c_g]/2^256 + N/(K + 1)`, under `ExpectedCR` at the by-position finder of the cap `K`: `c` a
run's calls with every session extracted (`callsAt`), `c_g` its calls with only the sessions that can open gate `g`
extracted (`callsBy`), `N` the program's gates. The contract's clause it reads is `Sessions.interior`. -/
theorem pr_offIreg_le_needs_at_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (K : ℕ)
    (hCR : ExpectedCR H (L.finderIAt H P x s Pρ needs K)) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.toCompiled.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_at_le_ex_cap H
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (vstar := fun (t : Ω × (Fin (L.V.k x) → Coin)) (g : Fin (L.nG x)) => L.iplur H P x t.1 Pρ t.2 g)
    (fun t => L.isPlurality_iplur H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun t ec g hg => decide_eq_true (L.iop_isSome_needs H P x t.1 hs Pρ needs hpos t.2 ec g hg))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x)
        (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.toCompiled.callsBy H P x s needs g.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.toCompiled.callsBy H P x s needs g.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun _ _ _ => le_rfl) K
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
      ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x R, g.val ∈ needs u)),
        L.toCompiled.callsBy H P x s needs g.val ω R c : ℕ) : ℝ))
  have hoff : Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.OffIreg H P x q.1.1 Pρ (forkOn (L.stage2 x) q.1.2 q.2.1) q.2.2) ≤
      Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off (L.iopTable H P x q.1.1 Pρ q.1.2)
          (fun g : Fin (L.nG x) => L.iplur H P x q.1.1 Pρ q.1.2 g) q.2 g) :=
    Pr_mono fun q hq => L.off_of_offIreg H P x q.1.1 Pρ q.1.2 q.2 hq
  exact (hP.symm.le.trans (hoff.trans h)).trans_eq
    (congrArg (fun e : ℝ => 2 * e / 2 ^ (256 : ℝ) + (L.nG x : ℝ) / ((K : ℝ) + 1)) hE)

/-- **The event of `KSRegTwoX` with the by-position finders' terms, at one cap for each finder.** On a statement in
scope, where a reveal for proof block `u` opens only gates of `needs u`, and the boundary's by-position first-hit
finder of the cap `KX` (`Compiled.finderXAt`) and the interior's of the cap `KI` (`finderIAt`) meet the linear
expected-time bound, an accepted run samples a proof block of `FregTwoX` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{g ∈ sample} c_g]/2^256 + N/(KX + 1) + N/(KI + 1)`.

* `c` is a run's calls with every session of its sample extracted (`callsAt`): each finder's first run. The sum is
  over the gates the proof blocks of the run's coins' sample need, and `c_g` is the run's calls with only the sessions
  that have a proof block needing gate `g` extracted (`callsBy`): what a further run at those coins costs a finder
  for a gate it waits for.
* The expectation is over the prover's tape, the run's coins and the extractor's, joint in the sample and the calls.
  It is at most `E[(1 + |sample|)·c]`, `eventTwoX_le_firstHit_cap`'s (`Compiled.ex_callsBy_le`).
* The premises the bound is small under are `eventTwoX_le_firstHit_cap`'s, with the linear expected-time bound at
  these two finders in place of those two. -/
theorem eventTwoX_le_firstHit_at_cap {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (KX KI : ℕ)
    (hX : ExpectedCR H (L.toCompiled.finderXAt H P x s needs KX))
    (hI : ExpectedCR H (L.finderIAt H P x s d.part needs KI)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.toCompiled.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        ((L.nG x : ℝ) / ((KX : ℝ) + 1) + (L.nG x : ℝ) / ((KI : ℝ) + 1)) := by
  have hB := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_needs_at_cap hbind hs P needs hpos KX hX)
  have hN := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_needs_at_cap hbind hS hs d.part P needs hpos KI hI)
  have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P hB hN
  have h4 : ∀ e : ℝ, 2 * e / 2 ^ (256 : ℝ) + 2 * e / 2 ^ (256 : ℝ) = 4 * e / 2 ^ (256 : ℝ) := fun e => by ring
  rw [← h4]
  linarith

/-- **The event of `KSRegTwoX` with the by-position finders' terms** where `ε_reg,2,X` has its two square roots:
`eventTwoX_le_firstHit_at_cap` under the linear expected-time bound at the two by-position finders of every cap, with
nothing left of the caps' terms: at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 4·E[c + Σ_{g ∈ sample} c_g]/2^256`. -/
theorem eventTwoX_le_firstHit_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d)
    (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXAt H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderIAt H P x s d.part needs K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
          ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
            ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
              L.toCompiled.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := by
  have hcap : ∀ K : ℕ,
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
        L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
          4 * Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
            ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
              ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
                L.toCompiled.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
            2 ^ (256 : ℝ) +
          (2 * (L.nG x : ℝ)) / ((K : ℝ) + 1) := by
    intro K
    have h := eventTwoX_le_firstHit_at_cap hA hbind hS hs hPt hd hin P needs hpos K K (hX K) (hI K)
    have e : (L.nG x : ℝ) / ((K : ℝ) + 1) + (L.nG x : ℝ) / ((K : ℝ) + 1) =
        (2 * (L.nG x : ℝ)) / ((K : ℝ) + 1) := by ring
    rw [e] at h
    exact h
  exact Verity.FirstHit.le_of_forall_cap hcap

/-! ## The interior and the event, with no premise -/

/-- **The mass off the committed interiors, with the first-hit finder's success in it**, with no premise on `H`:
`2·Pr[the finder outputs a collision] + N/(K + 1)`, at the interior's first-hit finder of the cap `K` (`finderI`),
`N` the program's gates. The contract's clause it reads is `Sessions.interior`. -/
theorem pr_offIreg_le_finder {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * (L.finderI H P x s Pρ needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_sample_le_finder H
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (vstar := fun (t : Ω × (Fin (L.V.k x) → Coin)) (g : Fin (L.nG x)) => L.iplur H P x t.1 Pρ t.2 g)
    (fun t => L.isPlurality_iplur H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    (fun t => L.icoll H P x s t.1 t.2)
    (fun t ec ec' g hg => icoll_of_conflictAt hbind hS hs Pρ P t.1 t.2 ec ec' g hg)
    (fun t ec ec' u v huv => L.icoll_collision H P x s t.1 t.2 ec ec' huv)
  rw [Fintype.card_fin] at h
  have hP := Pr_keep_forkOn (L.stage2 x)
    (fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) => L.OffIreg H P x ω Pρ R c)
  have hoff : Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.OffIreg H P x q.1.1 Pρ (forkOn (L.stage2 x) q.1.2 q.2.1) q.2.2) ≤
      Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off (L.iopTable H P x q.1.1 Pρ q.1.2)
          (fun g : Fin (L.nG x) => L.iplur H P x q.1.1 Pρ q.1.2 g) q.2 g) :=
    Pr_mono fun q hq => L.off_of_offIreg H P x q.1.1 Pρ q.1.2 q.2 hq
  exact hP.symm.le.trans (hoff.trans h)

/-- **The interior's first-hit finder's expected calls** are at most `E[(1 + |sample|)·c]`, whatever `H` is: `c` a
run's calls with every session of its sample extracted (`callsAt`), `|sample|` the gates the proof blocks of its
coins' sample need. -/
theorem ex_cost_finderI_le (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (Pρ : Partition) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Ex (fun sd => ((L.finderI H P x s Pρ needs K).cost sd : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u))) *
          L.callsAt H P x s q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  (Verity.FirstHit.ex_cost_sampleFinder_le
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    (L.finderI H P x s Pρ needs K).out).trans_eq
    (Ex_keep_forkOn (L.stage2 x) fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
      (((1 + #(univ.filter fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x R, g.val ∈ needs u))) *
        L.callsAt H P x s ω R c : ℕ) : ℝ))

/-- **The mass off the committed interiors, with the by-position finder's success in it**, with no premise on `H`:
`2·Pr[the finder outputs a collision] + N/(K + 1)`, at the interior's by-position first-hit finder of the cap `K`
(`finderIAt`), `N` the program's gates. The contract's clause it reads is `Sessions.interior`. -/
theorem pr_offIreg_le_finderAt {H : Msg → Msg} {L : CompiledTwo Coin A} (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) (Pρ : Partition) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (K : ℕ) :
    Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => L.OffIreg H P x q.1 Pρ q.2.1 q.2.2) ≤
      2 * (L.finderIAt H P x s Pρ needs K).Pr H + (L.nG x : ℝ) / ((K : ℝ) + 1) := by
  have h := Verity.FirstHit.pr_off_at_le_finder H
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (vstar := fun (t : Ω × (Fin (L.V.k x) → Coin)) (g : Fin (L.nG x)) => L.iplur H P x t.1 Pρ t.2 g)
    (fun t => L.isPlurality_iplur H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x)
        (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.toCompiled.callsBy H P x s needs g.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2) K
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec ec' : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.toCompiled.collBy H P x s needs (t.1, (forkOn (L.stage2 x) t.2 ec.1, ec.2))
        (t.1, (forkOn (L.stage2 x) t.2 ec'.1, ec'.2)) g.val)
    (fun t ec ec' g hg => icollBy_of_conflictAt hbind hS hs Pρ P needs hpos t.1 t.2 ec ec' g hg)
    (fun t ec ec' g u v huv => L.toCompiled.collBy_collision H P x s needs _ _ g.val huv)
  rw [Fintype.card_fin] at h
  have hP := Pr_keep_forkOn (L.stage2 x)
    (fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) => L.OffIreg H P x ω Pρ R c)
  have hoff : Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.OffIreg H P x q.1.1 Pρ (forkOn (L.stage2 x) q.1.2 q.2.1) q.2.2) ≤
      Pr (fun q : (Ω × (Fin (L.V.k x) → Coin)) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) => ∃ g : Fin (L.nG x),
        FlockSoundness.Rewinding.Off (L.iopTable H P x q.1.1 Pρ q.1.2)
          (fun g : Fin (L.nG x) => L.iplur H P x q.1.1 Pρ q.1.2 g) q.2 g) :=
    Pr_mono fun q hq => L.off_of_offIreg H P x q.1.1 Pρ q.1.2 q.2 hq
  exact hP.symm.le.trans (hoff.trans h)

/-- **The interior's by-position finder's expected calls** are at most `E[c + Σ_{g ∈ sample} c_g]`, whatever `H`
is: `c` a run's calls with every session of its sample extracted (`callsAt`), and `c_g` its calls with only the
sessions that have a proof block needing gate `g` extracted (`callsBy`). -/
theorem ex_cost_finderIAt_le (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (s : AuditStatement) (Pρ : Partition) (needs : ℕ → Finset ℕ) (K : ℕ) :
    Ex (fun sd => ((L.finderIAt H P x s Pρ needs K).cost sd : ℝ)) ≤
      Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        ((L.callsAt H P x s q.1 q.2.1 q.2.2 +
          ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x q.2.1, g.val ∈ needs u)),
            L.toCompiled.callsBy H P x s needs g.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) :=
  (Verity.FirstHit.ex_cost_sampleFinderAt_le
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) => L.iopTable H P x t.1 Pρ t.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      decide (∃ u ∈ L.checked x (forkOn (L.stage2 x) t.2 ec.1), g.val ∈ needs u))
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      L.callsAt H P x s t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x)
        (J : Finset (Fin (L.nG x))) =>
      ∑ g ∈ J, L.toCompiled.callsBy H P x s needs g.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g : Fin (L.nG x)) =>
      L.toCompiled.callsBy H P x s needs g.val t.1 (forkOn (L.stage2 x) t.2 ec.1) ec.2)
    (fun _ _ _ => le_rfl) K (L.finderIAt H P x s Pρ needs K).out).trans_eq
    (Ex_keep_forkOn (L.stage2 x) fun (ω : Ω) (R : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
      ((L.callsAt H P x s ω R c +
        ∑ g ∈ univ.filter (fun g : Fin (L.nG x) => decide (∃ u ∈ L.checked x R, g.val ∈ needs u)),
          L.toCompiled.callsBy H P x s needs g.val ω R c : ℕ) : ℝ))

/-- **The event of `KSRegTwoX` with the two first-hit finders' successes in it, with no premise on `H`.** On a
statement in scope, an accepted run samples a proof block of `FregTwoX` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 2·Pr[F_X finds] + 2·Pr[F_I finds] + N/(KX + 1) + N/(KI + 1)`, where
`ε_reg,2,X` has its two square roots.

* `F_X` is the boundary's first-hit finder of the cap `KX` (`Compiled.finderX`) and `F_I` the interior's of the cap
  `KI` (`finderI`); `Pr[F finds]` is the probability that the finder outputs a collision of `H`. Each one's expected
  calls are at most `E[(1 + |sample|)·c]` (`Compiled.ex_cost_finderX_le`, `ex_cost_finderI_le`).
  `eventTwoX_le_firstHit_cap` is this statement and those two under `ExpectedCR` at the two finders.
* Each finder's cost charges a further run only when the run's sample holds a gate still waited for. That its output
  reads no other run needs that a reveal for proof block `u` opens only gates of `needs u`, which this statement does
  not take. -/
theorem eventTwoX_le_finder {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ) (KX KI : ℕ) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        (2 * (L.toCompiled.finderX H P x s needs KX).Pr H + 2 * (L.finderI H P x s d.part needs KI).Pr H) +
        ((L.nG x : ℝ) / ((KX : ℝ) + 1) + (L.nG x : ℝ) / ((KI : ℝ) + 1)) := by
  have hB := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_finder hbind hs P needs KX)
  have hN := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_finder hbind hS hs d.part P needs KI)
  have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P hB hN
  linarith

/-- **The event of `KSRegTwoX` with the two by-position finders' successes in it, with no premise on `H`.** On a
statement in scope, where a reveal for proof block `u` opens only gates of `needs u`, an accepted run samples a proof
block of `FregTwoX` with probability at most
`κ_s + √(k·γ_s·Pr[B]) + Pr[B_conf,X] + 2·Pr[F_X finds] + 2·Pr[F_I finds] + N/(KX + 1) + N/(KI + 1)`, where
`ε_reg,2,X` has its two square roots.

`F_X` is the boundary's by-position first-hit finder of the cap `KX` (`Compiled.finderXAt`) and `F_I` the interior's
of the cap `KI` (`finderIAt`); `Pr[F finds]` is the probability that the finder outputs a collision of `H`. Each
one's expected calls are at most `E[c + Σ_{g ∈ sample} c_g]` (`Compiled.ex_cost_finderXAt_le`,
`ex_cost_finderIAt_le`). `eventTwoX_le_firstHit_at_cap` is this statement and those two under `ExpectedCR` at the two
finders. -/
theorem eventTwoX_le_finderAt {H : Msg → Msg} {L : CompiledTwo Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (needs : ℕ → Finset ℕ)
    (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (KX KI : ℕ) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + (Compiled.BconfX H L.toCompiled Ω P x).Pr H +
        (2 * (L.toCompiled.finderXAt H P x s needs KX).Pr H + 2 * (L.finderIAt H P x s d.part needs KI).Pr H) +
        ((L.nG x : ℝ) / ((KX : ℝ) + 1) + (L.nG x : ℝ) / ((KI : ℝ) + 1)) := by
  have hB := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
    (Compiled.pr_offRegX_le_finderAt hbind hs P needs hpos KX)
  have hN := (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
    (pr_offIreg_le_finderAt hbind hS hs d.part P needs hpos KI)
  have h := eventTwoX_le_of_off hA hbind hS hs hPt hd hin P hB hN
  linarith

end CompiledTwo

end Claims.SamplingGeneric
