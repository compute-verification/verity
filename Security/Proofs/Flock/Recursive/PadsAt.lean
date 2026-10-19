import Proofs.Flock.Recursive.Finds
import Proofs.Flock.Recursive.ForkInner

/-!
# Recursion: the pads fork finder at the queried positions, beside an any-position one

The pads fork finder (`advPadsSFinderZ`) and the self-clash finder read rep `r`'s pads openings only at the positions
its queries drew (`opPadsAtZ`, `PadsClashAtZ`). Beside them, this file states what a reader blind to those positions
would read: at `p`, the first query whose opening verifies at `p` wherever it was drawn (`opPadsAnyZ`), and two such
openings of one position (`PadsClashAnyZ`, in `SelfClashAnyZ`). No guarantee reads these; they are here to compare the
two fork events.

* `outcome_forkFinder`: every outcome of a rewinding finder is its default, or its pick of two outcomes the strategy
  reaches in the real game, and a finder with another pick reaches its pick of the same two;
* `outcome_push`, `outcome_sessAtZ_push`: an outcome of the primed session against the pushed prover is one of the
  compiled session against the prover;
* `outcome_forkZ_self`: **an outcome of a forked continuation is an outcome `selfSFinderZ`'s run reaches**: along the
  primed prover, at any chooser, slot and pick, every outcome of a rewinding finder is its default or its pick of two
  outcomes of the compiled session, and the self-clash finder's run reaches its pick of each;
* `conflictAt_padsAt_cover`: a conflict of `opPadsAtZ` at `p` is a conflict of `opPadsAnyZ` at `p`, or an
  any-position pads clash in one of the two runs;
* `advPadsSZ_finds_cover`: if the pads fork finder finds a collision, the any-position pads fork finder at the same
  chooser finds one, or the any-position self-clash finder does; `selfSFinderZ_finds_any`: the same for the self-clash
  finder.

Neither inclusion of the fork events holds the other way round: a position-tied conflict need not be an any-position
one, and an any-position conflict need not be position-tied.
-/

namespace FlockSoundness.StrictCR

open Game

section Fork

variable {α β T Dc I γ γ' : Type} {get : I → ℕ → Option (T × Dc)} {R : I → α → β → Prop}

/-- **A rewinding finder's outcomes**: its default, or its pick of two outcomes the strategy reaches in the real game,
from the cap sent at the slot's node; a finder with another pick and default reaches its pick of the same two. -/
theorem outcome_forkFinder (plur : Lock.Chooser T Dc β) (slot : ℕ) (pick : Dc → β → β → γ) (dflt : γ)
    (pick' : Dc → β → β → γ') (dflt' : γ') :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T Dc get R i g g') (σ' : Strategy g') {y : γ},
      Game.Outcome (forkFinder plur slot pick dflt L σ').g (forkFinder plur slot pick dflt L σ').s y →
      y = dflt ∨ ∃ d b b', pick d b b' = y ∧ Game.Outcome g' σ' b ∧ Game.Outcome g' σ' b' ∧
        Game.Outcome (forkFinder plur slot pick' dflt' L σ').g (forkFinder plur slot pick' dflt' L σ').s
          (pick' d b b')
  | _, _, _, .ret _, _, _, h => Or.inl (Eq.symm h)
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', _, h => by
      obtain ⟨c, hc⟩ := h
      rcases outcome_forkFinder plur slot pick dflt pick' dflt' (L c) (σ' c) hc with h | ⟨d, b, b', e, hb, hb', h'⟩
      · exact Or.inl h
      · exact Or.inr ⟨d, b, b', e, ⟨c, hb⟩, ⟨c, hb'⟩, ⟨c, h'⟩⟩
  | _, _, _, .send _ L, σ', _, h => outcome_forkFinder plur slot pick dflt pick' dflt' (L σ'.1) σ'.2 h
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ', _, h => by
      simp only [forkFinder] at h ⊢
      by_cases hs : s = slot
      · rw [ite_eq_left hs] at h ⊢
        obtain ⟨b, b', hb, hb', e⟩ := Discharge.Recursion.outcome_of_twoRuns (k' σ'.1) σ'.2 (pick σ'.1) h
        exact Or.inr ⟨σ'.1, b, b', e, hb, hb', Discharge.Recursion.outcome_twoRuns _ _ _ hb hb'⟩
      · rw [ite_eq_right hs] at h ⊢
        exact outcome_forkFinder plur slot pick dflt pick' dflt' _ σ'.2 h

end Fork

variable {D : Type} {H : List UInt8 → D}

/-- **A collision output says its event held.** -/
theorem of_isColl_collOf {p : Prop} {h : p → Merkle.Collision H} (hc : IsColl H (collOf H p h)) : p := by
  by_contra hp
  obtain ⟨_, _, e, -⟩ := hc
  unfold collOf at e
  rw [dite_eq_right hp] at e
  cases e

end FlockSoundness.StrictCR

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## The pushed prover's outcomes -/

/-- **A pushed prover reaches only the compiled prover's outcomes**: along a Lock whose leaves relate equal outputs
while every cap is pushed, an outcome of the oracle side against the prover sending the constant cap at every digest is
an outcome of the compiled side. -/
theorem outcome_push {α I : Type} {get : I → ℕ → Option (Cap D × D)} :
    ∀ {i : I} {g g' : Game α} (L : Lock (Cap D) D get (fun i a b => Pushed get i → a = b) i g g')
      (σ' : Strategy g') {a : α}, Pushed get i → Game.Outcome g (Lock.oracleStrat pushC L σ') a →
        Game.Outcome g' σ' a
  | _, _, _, .ret h, _, _, hi, ha => (h hi).symm.trans ha
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', _, hi, ha => by
      obtain ⟨c, hc⟩ := ha
      exact ⟨c, outcome_push (L c) (σ' c) hi hc⟩
  | _, _, _, .send _ L, σ', _, hi, ha => outcome_push (L σ'.1) σ'.2 hi ha
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' _ hset hkeep L, σ', _, hi, ha => by
      refine outcome_push _ σ'.2 ?_ ha
      intro s' c d h
      by_cases hs : s' = s
      · subst hs
        rw [hset] at h
        cases h
        rfl
      · rw [hkeep _ _ _ hs] at h
        exact hi s' c d h

omit [Fintype K] in
/-- **The primed session's outcomes against the pushed prover are the compiled session's** against the prover. -/
theorem outcome_sessAtZ_push [Inhabited D] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {b : List (OutZC F D Sl)}
    (hb : Game.Outcome (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) b) :
    Game.Outcome (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ b :=
  outcome_push (pushSess A Hs T cap₀ pre post) σ (fun s c d h => by simp [Lock.getPair, getP] at h) hb

/-! ## The pads openings at any position -/

open Classical in
/-- **Rep `r`'s pads openings at any position**: at `p`, the pair of the first query whose opening verifies at `p` to
the root `c 0`, wherever that query was drawn. -/
noncomputable def opPadsAnyZ (Hs : Hashes F D Sl) (N q r : ℕ) (c : Cap D) (out : OutZC F D Sl) (p : Fin N) :
    Option (F × F) :=
  (((List.range q).filter fun jj => decide (Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val
      ((out.opens.inner r).pads jj).1 ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2)).head?).map
    fun jj => ((out.opens.inner r).pads jj).1

/-- **A pads clash at any position**: within one run, two of a rep's pads openings verify at one position to one root
with different pairs, wherever they were drawn. -/
def PadsClashAnyZ (Hs : Hashes F D Sl) (N q : ℕ) (out : OutZC F D Sl) : Prop :=
  ∃ r < 2, ∃ root : D, ∃ jj < q, ∃ jj' < q, ∃ p : Fin N,
    Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => root) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj).1
        ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2 ∧
      Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => root) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj').1
        ((out.opens.inner r).pads jj').2.1 ((out.opens.inner r).pads jj').2.2 ∧
      ((out.opens.inner r).pads jj).1 ≠ ((out.opens.inner r).pads jj').1

omit [Field F] [Fintype F] [DecidableEq F] in
theorem PadsClashAnyZ.collision {Hs : Hashes F D Sl} {N q : ℕ} {out : OutZC F D Sl} (h : PadsClashAnyZ Hs N q out) :
    Merkle.Collision Hs.H := by
  obtain ⟨_, _, root, jj, _, jj', _, p, h1, h2, hne⟩ := h
  exact Merkle.opening_bindingL Hs.H Hs.Ep Hs.Lp (fun _ => root) (Nat.log 2 N) p.val _ _ _ _ _ _ hne h1 h2

omit [Field F] [Fintype F] [DecidableEq F] in
/-- A pads clash at the queried positions is one at any position. -/
theorem PadsClashAtZ.any {Hs : Hashes F D Sl} {N q : ℕ} {out : OutZC F D Sl} (h : PadsClashAtZ Hs N q out) :
    PadsClashAnyZ Hs N q out := by
  obtain ⟨r, hr, root, jj, hjj, jj', hjj', p, -, -, h1, h2, hne⟩ := h
  exact ⟨r, hr, root, jj, hjj, jj', hjj', p, h1, h2, hne⟩

omit [Field F] [Fintype F] [DecidableEq F] in
open Classical in
theorem opPadsAnyZ_some {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D} {out : OutZC F D Sl} {p : Fin N}
    (h : (opPadsAnyZ Hs N q r c out p).isSome) :
    ∃ jj < q, Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj).1
        ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2 ∧
      opPadsAnyZ Hs N q r c out p = some ((out.opens.inner r).pads jj).1 := by
  unfold opPadsAnyZ at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq, List.mem_range] at hmem
  exact ⟨x, hmem.1, hmem.2, by rw [hx]; rfl⟩

omit [Field F] [Fintype F] [DecidableEq F] in
open Classical in
theorem opPadsAnyZ_isSome {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D} {out : OutZC F D Sl} {p : Fin N} {jj : ℕ}
    (hjj : jj < q)
    (hv : Merkle.VerifiesL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val ((out.opens.inner r).pads jj).1
      ((out.opens.inner r).pads jj).2.1 ((out.opens.inner r).pads jj).2.2) :
    (opPadsAnyZ Hs N q r c out p).isSome := by
  unfold opPadsAnyZ
  rw [Option.isSome_map, List.isSome_head?]
  exact List.ne_nil_of_mem (List.mem_filter.2 ⟨List.mem_range.2 hjj, decide_eq_true hv⟩)

omit [Field F] [Fintype F] [DecidableEq F] in
/-- Two runs opening a pads position two ways at any position give a collision. -/
theorem opPadsAnyZ_conflict_collision {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D}
    {x : OutZC F D Sl × OutZC F D Sl} {p : Fin N} (h : Rewinding.ConflictAt (opPadsAnyZ Hs N q r c) x p) :
    Merkle.Collision Hs.H := by
  obtain ⟨jj, -, h1, e1⟩ := opPadsAnyZ_some h.1
  obtain ⟨jj', -, h2, e2⟩ := opPadsAnyZ_some h.2.1
  refine Merkle.opening_bindingL Hs.H Hs.Ep Hs.Lp (fun _ => c 0) (Nat.log 2 N) p.val _ _ _ _ _ _
    (fun e => h.2.2 ?_) h1 h2
  rw [e1, e2, e]

omit [Field F] [Fintype F] [DecidableEq F] in
/-- **The position-tied fork event, covered**: two runs that open a pads position two ways at the positions their
queries drew open it two ways at any position, or one of them has a pads clash at any position. -/
theorem conflictAt_padsAt_cover {Hs : Hashes F D Sl} {N q r : ℕ} {c : Cap D} {x : OutZC F D Sl × OutZC F D Sl}
    {p : Fin N} (hr : r < 2) (h : Rewinding.ConflictAt (opPadsAtZ Hs N q r c) x p) :
    Rewinding.ConflictAt (opPadsAnyZ Hs N q r c) x p ∨ PadsClashAnyZ Hs N q x.1 ∨ PadsClashAnyZ Hs N q x.2 := by
  obtain ⟨jj1, hjj1, -, hv1, e1⟩ := opPadsAtZ_some h.1
  obtain ⟨jj2, hjj2, -, hv2, e2⟩ := opPadsAtZ_some h.2.1
  have hs1 := opPadsAnyZ_isSome (r := r) (out := x.1) (c := c) hjj1 hv1
  have hs2 := opPadsAnyZ_isSome (r := r) (out := x.2) (c := c) hjj2 hv2
  obtain ⟨kk1, hkk1, hw1, f1⟩ := opPadsAnyZ_some hs1
  obtain ⟨kk2, hkk2, hw2, f2⟩ := opPadsAnyZ_some hs2
  by_cases h1 : ((x.1.opens.inner r).pads kk1).1 = ((x.1.opens.inner r).pads jj1).1
  · by_cases h2 : ((x.2.opens.inner r).pads kk2).1 = ((x.2.opens.inner r).pads jj2).1
    · refine Or.inl ⟨hs1, hs2, ?_⟩
      rw [f1, f2, h1, h2]
      intro e
      exact h.2.2 (by rw [e1, e2]; exact e)
    · exact Or.inr (Or.inr ⟨r, hr, c 0, kk2, hkk2, jj2, hjj2, p, hw2, hv2, h2⟩)
  · exact Or.inr (Or.inl ⟨r, hr, c 0, kk1, hkk1, jj1, hjj1, p, hw1, hv1, h1⟩)

/-- **A self-clash at any position**: `SelfClashZ` with its pads case at any position. -/
def SelfClashAnyZ (Hs : Hashes F D Sl) (T : TabZK F mPts) (out : OutZC F D Sl) : Prop :=
  ClashW Hs.H Hs.E Hs.Lf out.cap₀ (width₀ T) (pathLen (lvOf T.sch 0)) (opens₀Z T.sch out) ∨
    (∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ - 1]? = some cap ∧
      Exec.ClashL Hs.H Hs.E Hs.Lf cap (lvOf T.sch ℓ).lanes (pathLen (lvOf T.sch ℓ))
        (slotOpensZ out r ℓ (lvOf T.sch ℓ).queries)) ∨
    PadsClashAnyZ Hs T.N T.q out

theorem SelfClashAnyZ.collision {Hs : Hashes F D Sl} {T : TabZK F mPts} {out : OutZC F D Sl}
    (h : SelfClashAnyZ Hs T out) : Merkle.Collision Hs.H := by
  rcases h with h | ⟨_, _, _, _, _, _, h⟩ | h
  · exact h.collision
  · exact h.collision
  · exact h.collision

theorem SelfClashZ.any {Hs : Hashes F D Sl} {T : TabZK F mPts} {out : OutZC F D Sl} (h : SelfClashZ Hs T out) :
    SelfClashAnyZ Hs T out := by
  rcases h with h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr h.any)

end FlockSoundness.Discharge.ZkLink

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting ZkSession ZkLink StrictCR

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

/-- **An outcome of a forked continuation is an outcome `selfSFinderZ`'s run reaches.** Along the primed prover
(`lockSessZ` against `pushStrat`), at any chooser, slot and pick, every outcome of a rewinding finder is `none` or its
pick of two outcomes of the compiled session against `σ`, and the self-clash finder's run reaches its pick of each. -/
theorem outcome_forkZ_self (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) (slot : ℕ)
    (pick : Cap D → List (OutZC F D Sl) → List (OutZC F D Sl) → Pair)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) {y : Pair}
    (hy : Game.Outcome
      (forkFinder plur slot pick none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).g
      (forkFinder plur slot pick none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)).s
      y) :
    y = none ∨ ∃ d b b', pick d b b' = y ∧
      Game.Outcome (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ b ∧
      Game.Outcome (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ b' ∧
      Game.Outcome (selfSFinderZ A Hs T pre post cap₀ σ).g (selfSFinderZ A Hs T pre post cap₀ σ).s
        (onePick Hs.H (fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
          (fun _ h => h.collision) b) ∧
      Game.Outcome (selfSFinderZ A Hs T pre post cap₀ σ).g (selfSFinderZ A Hs T pre post cap₀ σ).s
        (onePick Hs.H (fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
          (fun _ h => h.collision) b') := by
  rcases outcome_forkFinder plur slot pick none pick none _ _ hy with h | ⟨d, b, b', e, hb, hb', -⟩
  · exact Or.inl h
  · have hb₁ := outcome_sessAtZ_push A Hs T cap₀ pre post σ hb
    have hb₂ := outcome_sessAtZ_push A Hs T cap₀ pre post σ hb'
    exact Or.inr ⟨d, b, b', e, hb₁, hb₂, outcome_oneRun _ _ _ hb₁, outcome_oneRun _ _ _ hb₂⟩

/-- **The pads fork finder at any position** in the session, at a chooser: `advPadsSFinderZ` with `opPadsAnyZ`. -/
noncomputable def advPadsAnySFinderZ (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl))) (r : ℕ)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  forkFinder plur r
    (fun d => conflictPick Hs.H
      (fun outs : List (OutZC F D Sl) => opPadsAnyZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC))
      fun x _ h => opPadsAnyZ_conflict_collision (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h)
    none (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **The self-clash finder at any position** in the session: `selfSFinderZ` with `SelfClashAnyZ`. -/
noncomputable def selfAnySFinderZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : Finder Pair :=
  oneRun (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ
    (onePick Hs.H (fun outs : List (OutZC F D Sl) => SelfClashAnyZ Hs T (outs.getD pre.length dfltOutZC))
      fun _ h => h.collision)

omit [Fintype K] in
/-- **The self-clash finder, at any position**: when `selfSFinderZ` finds a collision, so does `selfAnySFinderZ`. -/
theorem selfSFinderZ_finds_any (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (h : (selfSFinderZ A Hs T pre post cap₀ σ).Finds Hs.H) : (selfAnySFinderZ A Hs T pre post cap₀ σ).Finds Hs.H := by
  obtain ⟨y, hy, hc⟩ := h
  obtain ⟨b, hb, rfl⟩ := outcome_of_map _ _ _ hy
  exact ⟨_, outcome_oneRun _ _ _ hb, collOf_isColl Hs.H _ (SelfClashZ.any (of_isColl_collOf hc))⟩

/-- **The pads fork finder, covered**: when `advPadsSFinderZ` finds a collision, the pads fork finder at any position
at the same chooser (`plurSZ`) finds one, or the self-clash finder at any position does, at an outcome of the forked
continuation (`outcome_forkZ_self`). -/
theorem advPadsSZ_finds_cover (r : ℕ) (hr : r < 2) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    (h : (advPadsSFinderZ A Hs T pre post cap₀ r σ).Finds Hs.H) :
    (advPadsAnySFinderZ A Hs T pre post cap₀ (plurSZ Hs T pre.length) r σ).Finds Hs.H ∨
      (selfAnySFinderZ A Hs T pre post cap₀ σ).Finds Hs.H := by
  obtain ⟨y, hy, hc⟩ := h
  rcases outcome_forkFinder _ _ _ _
      (fun d => conflictPick Hs.H
        (fun outs : List (OutZC F D Sl) => opPadsAnyZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC))
        fun x _ h => opPadsAnyZ_conflict_collision
          (x := (x.1.getD pre.length dfltOutZC, x.2.getD pre.length dfltOutZC)) h) none _ _ hy with
    rfl | ⟨d, b, b', rfl, hb, hb', h'⟩
  · obtain ⟨_, _, e, -⟩ := hc
    cases e
  · obtain ⟨p, hp⟩ := of_isColl_collOf hc
    rcases conflictAt_padsAt_cover (x := (b.getD pre.length dfltOutZC, b'.getD pre.length dfltOutZC)) hr hp with
      hp' | hcl | hcl
    · exact Or.inl ⟨_, h', collOf_isColl Hs.H _ ⟨p, hp'⟩⟩
    · exact Or.inr ⟨_, outcome_oneRun _ _ _ (outcome_sessAtZ_push A Hs T cap₀ pre post σ hb),
        collOf_isColl Hs.H _ (Or.inr (Or.inr hcl))⟩
    · exact Or.inr ⟨_, outcome_oneRun _ _ _ (outcome_sessAtZ_push A Hs T cap₀ pre post σ hb'),
        collOf_isColl Hs.H _ (Or.inr (Or.inr hcl))⟩

end Session

end FlockSoundness.Discharge.Recursion
