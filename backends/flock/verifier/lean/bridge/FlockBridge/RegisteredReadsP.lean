import FlockSoundness.Discharge.ScopeW.Reads
import FlockBridge.RegisteredReads

/-!
# Registered reads at rows under their own schemas, from the verifier's registered check

`RegisteredReads` at C-Flock's registered reads over rows each hashed under its own schema
(`FlockSoundness.Discharge.ScopeW.ReadsP`, over `HmRowsP`), the rows the widened scope (`ScopeW.scopeOkW`) admits.
`RegisteredReads` reads `Registered.Reads`, over `HmRows`, whose rows are all of one encoding; at `HmRowsP.toHmRows` it
has no row, so it reads nothing there. Each declaration here is `RegisteredReads`' with `ReadsP` in place of `Reads`,
and the same proof:

* `restrict`, `regGates`, `regVals`: the reads where a predicate holds, the registered gates and the registered values;
  the commit strings the draw's units read are `RegisteredReads.drawnReads`, which reads no row;
* `RegValsSound` (named), `prob_off_le`, `regReadSound_of_checked`: registered reads are sound at `qR²/2^513 + δ`;
* `RegJointSound` (named), `regJoint_of_checked`, `registered_of_checked_joint`: the joint form, in the core game.
-/

namespace FlockBridge

namespace RegisteredReadsP

open scoped ENNReal

open FlockSoundness (TabSpec)
open FlockSoundness.Audit (Circuit Partition)
open FlockSoundness.Binding (Layout)
open FlockSoundness.Discharge.ScopeW (HmRowsP ReadsP)
open RegisteredReads (drawnReads prob_mono_out)

variable {C : Circuit} {n : ℕ} {P : Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Layout P plan tab decode} {rs : HmRowsP lay}

/-- `rd`'s reads where `S` holds, with the registrant's openings unchanged. -/
def restrict (rd : ReadsP lay rs) (S : lay.Pos → Prop) : ReadsP lay rs where
  isRead p := rd.isRead p ∧ S p
  read_row p h := rd.read_row p h.1
  port := rd.port
  readAt := rd.readAt
  wv := rd.wv
  ws := rd.ws
  wpath := rd.wpath
  committed p h := rd.committed p h.1

open Classical in
/-- **The registered gates**: the wires held at a registered read. -/
noncomputable def regGates (rd : ReadsP lay rs) : Finset (Fin C.N) :=
  Finset.univ.filter fun g => rd.isRead (lay.pos g)

/-- **The registered values**: each wire's bit in the row the registrant committed at its commit string. -/
def regVals (rd : ReadsP lay rs) : Fin C.N → Bool := fun g => lay.bit (rd.wv (lay.pos g)) g

variable {Out : Type}

/-- **The registrant's values at the drawn units' registered reads, up to `δ`** (named hypothesis), at strategy `s`: an
accepting run of `G` draws a unit that reads, at a registered gate `g`, the registrant's leaf under the run's
registration `R o`, while `Xc` is not the registrant's bit at `g`, with probability at most `δ`. -/
def RegValsSound (G : FlockSoundness.Game Out) (s : FlockSoundness.Game.Strategy G) (acc : Out → Prop)
    (drawn : Out → Finset (Fin n)) (R : Out → Reg) (rd : ReadsP lay rs) (Xc : Fin C.N → Bool) (δ : ℝ≥0∞) : Prop :=
  FlockSoundness.Game.prob (fun o => acc o ∧ ∃ u ∈ drawn o, ∃ g ∈ P.io u, rd.isRead (lay.pos g) ∧
      lay.regLeaf (R o) (lay.pos g) = rd.leaf (lay.pos g) ∧ Xc g ≠ regVals rd g) G s ≤ δ

/-- **The registered reads' term at each run's reads and registration, under `SHA512CRStrict` (`cr/sha-512`).** In a
game `G` played by `σ` whose accepted outcomes `o` (those `G` can reach) pass the registered check at their
registration `R o` for the reads `rdo o`, through their paths: the probability of accepting while some read's leaf is
not the registrant's is at most `qR²/2^513`, for the finder that plays `G` once and outputs the extractor's pair.
`ReadsP.prob_off_le`, at reads and a registration that depend on the run. -/
theorem prob_off_le (G : FlockSoundness.Game Out) (σ : FlockSoundness.Game.Strategy G) (accept : Out → Prop)
    (R : Out → Reg) (rdo : Out → ReadsP lay rs) (path : Out → lay.Pos → ByteArray)
    (hchk : ∀ o, FlockSoundness.Game.Out G o → accept o → (rdo o).Checked (R o) (path o)) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun G σ fun o => (rdo o).pick (R o) (path o)).CR FlockSoundness.Refine.H512 qR) :
    FlockSoundness.Game.prob (fun o => accept o ∧ (rdo o).Off (R o)) G σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  have hle : FlockSoundness.Game.prob (fun o => accept o ∧ (rdo o).Off (R o)) G σ ≤
      FlockSoundness.Game.prob
        (fun o => FlockSoundness.StrictCR.IsColl FlockSoundness.Refine.H512 ((rdo o).pick (R o) (path o))) G σ :=
    prob_mono_out (fun o ho ⟨ha, hoff⟩ =>
      ((rdo o).openings (R o) (path o) ((rdo o).offAt (R o))).isColl ((rdo o).offAt_conflict (hchk o ho ha) hoff)) σ
  have hev : (FlockSoundness.Game.prob
      (fun o => FlockSoundness.StrictCR.IsColl FlockSoundness.Refine.H512 ((rdo o).pick (R o) (path o))) G σ).toReal ≤
        qR ^ 2 / 2 ^ (513 : ℝ) := by
    rw [FlockSoundness.Game.prob_toReal]
    have := FlockSoundness.StrictCR.Finder.CR.le hR
    rwa [FlockSoundness.StrictCR.oneRun_ev] at this
  have hfin : FlockSoundness.Game.prob
      (fun o => FlockSoundness.StrictCR.IsColl FlockSoundness.Refine.H512 ((rdo o).pick (R o) (path o))) G σ ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (FlockSoundness.Game.prob_le_one _ _ _)
  exact hle.trans ((ENNReal.le_ofReal_iff_toReal_le hfin (by positivity)).2 hev)

/-- **Registered reads are sound, from the registered check.** In a game `G` played by `σ`, with drawn units `drawn o`
and registration `R o` on each run: if every accepted outcome `G` can reach passes the registered check on the
registrant's reads at the commit strings its drawn units read (`hchk`), the registered-path finder meets `cr/sha-512` at
`qR` (`hR`), and the registrant's values hold up to `δ` (`RegValsSound`), then registered reads are sound for `Xc`
against the registered values `regVals rd` on the registered gates `regGates rd`, up to `qR²/2^513 + δ`. -/
theorem regReadSound_of_checked (G : FlockSoundness.Game Out) (σ : FlockSoundness.Game.Strategy G)
    (acc : Out → Prop) (drawn : Out → Finset (Fin n)) (R : Out → Reg) (path : Out → lay.Pos → ByteArray)
    (rd : ReadsP lay rs) (Xc : Fin C.N → Bool)
    (hchk : ∀ o, FlockSoundness.Game.Out G o → acc o →
      (restrict rd (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun G σ fun o =>
      (restrict rd (drawnReads lay (drawn o))).pick (R o) (path o)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hvals : RegValsSound G σ acc drawn R rd Xc δ) :
    Registered.RegReadSound P G σ acc drawn Xc (regVals rd) (regGates rd)
      (ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ) := by
  classical
  refine le_trans (FlockSoundness.Game.prob_mono (fun o ho => ?_) G σ)
    ((FlockSoundness.Game.prob_or_le _ _ G σ).trans
      (add_le_add (prob_off_le G σ acc R (fun o => restrict rd (drawnReads lay (drawn o))) path hchk hR) hvals))
  obtain ⟨hacc, hd⟩ := ho
  obtain ⟨u, hu, hdu⟩ := Finset.not_disjoint_iff.1 hd
  obtain ⟨g, hg, hRg, hne⟩ := (Finset.mem_filter.1 hu).2
  have hrd : rd.isRead (lay.pos g) := (Finset.mem_filter.1 hRg).2
  by_cases hl : lay.regLeaf (R o) (lay.pos g) = rd.leaf (lay.pos g)
  · exact Or.inr ⟨hacc, u, hdu, g, hg, hrd, hl, hne⟩
  · exact Or.inl ⟨hacc, lay.pos g, ⟨hrd, u, hdu, g, hg, rfl⟩, hl⟩

/-- **The drawn wrong units and the registrant's values together, up to `δ`** (named hypothesis), at strategy `s`: an
accepting run of `G` draws a wrong unit of `Xc`, or a unit that reads, at a registered gate `g`, the registrant's leaf
under the run's registration `R o` while `Xc` is not the registrant's bit at `g`, with probability at most `δ`. -/
def RegJointSound (G : FlockSoundness.Game Out) (s : FlockSoundness.Game.Strategy G) (acc : Out → Prop)
    (drawn : Out → Finset (Fin n)) (R : Out → Reg) (rd : ReadsP lay rs) (Xc : Fin C.N → Bool) (δ : ℝ≥0∞) : Prop :=
  FlockSoundness.Game.prob (fun o => acc o ∧ (¬ Disjoint (P.wrong Xc) (drawn o) ∨
      ∃ u ∈ drawn o, ∃ g ∈ P.io u, rd.isRead (lay.pos g) ∧
        lay.regLeaf (R o) (lay.pos g) = rd.leaf (lay.pos g) ∧ Xc g ≠ regVals rd g)) G s ≤ δ

/-- **The joint bound, from the registered check.** On `regReadSound_of_checked`'s check and `cr/sha-512` for the
registered-path finder, with the drawn wrong units and the registrant's values bounded together (`RegJointSound`): an
accepting run draws a wrong unit of `Xc` or a unit whose opening shows `Xc` off `regVals rd` on `regGates rd` with
probability at most `qR²/2^513 + δ`. -/
theorem regJoint_of_checked (G : FlockSoundness.Game Out) (σ : FlockSoundness.Game.Strategy G)
    (acc : Out → Prop) (drawn : Out → Finset (Fin n)) (R : Out → Reg) (path : Out → lay.Pos → ByteArray)
    (rd : ReadsP lay rs) (Xc : Fin C.N → Bool)
    (hchk : ∀ o, FlockSoundness.Game.Out G o → acc o →
      (restrict rd (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun G σ fun o =>
      (restrict rd (drawnReads lay (drawn o))).pick (R o) (path o)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hjoint : RegJointSound G σ acc drawn R rd Xc δ) :
    FlockSoundness.Game.prob (fun o => acc o ∧ (¬ Disjoint (P.wrong Xc) (drawn o) ∨
        ¬ Disjoint (Registered.offAt P Xc (regVals rd) (regGates rd)) (drawn o))) G σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ := by
  classical
  refine le_trans (FlockSoundness.Game.prob_mono (fun o ho => ?_) G σ)
    ((FlockSoundness.Game.prob_or_le _ _ G σ).trans
      (add_le_add (prob_off_le G σ acc R (fun o => restrict rd (drawnReads lay (drawn o))) path hchk hR) hjoint))
  obtain ⟨hacc, hw | hd⟩ := ho
  · exact Or.inr ⟨hacc, Or.inl hw⟩
  obtain ⟨u, hu, hdu⟩ := Finset.not_disjoint_iff.1 hd
  obtain ⟨g, hg, hRg, hne⟩ := (Finset.mem_filter.1 hu).2
  have hrd : rd.isRead (lay.pos g) := (Finset.mem_filter.1 hRg).2
  by_cases hl : lay.regLeaf (R o) (lay.pos g) = rd.leaf (lay.pos g)
  · exact Or.inr ⟨hacc, Or.inr ⟨u, hdu, g, hg, hrd, hl, hne⟩⟩
  · exact Or.inl ⟨hacc, lay.pos g, ⟨hrd, u, hdu, g, hg, rfl⟩, hl⟩

/-- **Soundness against registered values in the core game, from the registered check and the joint bound.** For any
strategy `σ` of `G`'s core game, at its C-Flock strategy: on `regJoint_of_checked`'s assumptions, some core transcript
makes an accepting run draw a wrong unit or be off the registered values `regVals rd` on the registered gates
`regGates rd` with probability at most `qR²/2^513 + δ`. `Registered.registered_core_joint` at `regJoint_of_checked`. -/
theorem registered_of_checked_joint (G : FlockSoundness.Game Out) (σ : Verity.Game.Strategy (Game.toCore G))
    (acc : Out → Prop) (drawn : Out → Finset (Fin n)) (R : Out → Reg) (path : Out → lay.Pos → ByteArray)
    (rd : ReadsP lay rs) (Xc : Fin C.N → Bool)
    (hchk : ∀ o, FlockSoundness.Game.Out G o → acc o →
      (restrict rd (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun G (Game.Strategy.ofCore G σ) fun o =>
      (restrict rd (drawnReads lay (drawn o))).pick (R o) (path o)).CR FlockSoundness.Refine.H512 qR)
    {δ : ℝ≥0∞} (hjoint : RegJointSound G (Game.Strategy.ofCore G σ) acc drawn R rd Xc δ) :
    ∃ X' : (Circuit.toCore C).Transcript,
      Verity.Game.prob (fun o => acc o ∧
          (¬ Disjoint ((Partition.toCore P).wrong X') (drawn o) ∨
            ∃ g ∈ regGates rd, X' g ≠ Circuit.Transcript.toCore (regVals rd) g))
        (Game.toCore G) σ ≤ ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) + δ := by
  have h := regJoint_of_checked G _ acc drawn R path rd Xc hchk hR hjoint
  rw [← Circuit.Transcript.ofCore_toCore (regVals rd)] at h
  exact Registered.registered_core_joint P G acc drawn (regGates rd) _ σ Xc h

end RegisteredReadsP

end FlockBridge
