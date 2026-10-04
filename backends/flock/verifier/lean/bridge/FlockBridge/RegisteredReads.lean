import FlockSoundness.Registered.Binding
import FlockBridge.Registered

/-!
# Registered reads, from the verifier's registered check

`RegReadSound` (`FlockBridge.Registered`) asks that an accepting run rarely draw a unit whose opening shows a registered
gate at which the transcript is off the registered values. Here it follows from C-Flock's registered reads
(`FlockSoundness.Registered.Reads`, the registrant's openings `rd`) and one named bound:

* **The registered gates and values** (`regGates`, `regVals`): the wires held at a registered read, any committed wire
  the reads cover, and each one's bit in the row the registrant committed at its commit string.
* **The check reads only the draw's units** (`drawnReads`, `restrict`): the verifier reads the registered values at the
  drawn statement's instances. `RegReadSound` charges only drawn units, so that is enough.
* **The registered leaf** (`prob_off_le`): where the registered check holds at every accepted run, an accepting run
  whose registered leaf at a drawn unit's read is not the registrant's is a SHA-512 collision of the registered-path
  finder, at most `qR²/2^513` under `cr/sha-512`: `Reads.prob_off_le` at reads and a registration that depend on the
  run.
* **The registrant's values** (`RegValsSound`, named): an accepting run rarely draws a unit that reads the registrant's
  leaf at a registered gate where the transcript is not the registrant's bit. For C-Flock's value layer this is the
  extraction's step: a drawn unit's satisfying witness holds the registrant's row there or an hm96 collision against it
  (`Reads.registered_weights`), and the value layer's reading or a link collision.

`regReadSound_of_checked` puts them together: `RegReadSound` at `qR²/2^513 + δ`.
-/

namespace FlockBridge

namespace RegisteredReads

open scoped ENNReal

open FlockSoundness (TabSpec)
open FlockSoundness.Audit (Circuit Partition)
open FlockSoundness.Binding (Layout HmRows)
open FlockSoundness.Registered (Reads)

variable {C : Circuit} {n : ℕ} {P : Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Layout P plan tab decode} {rs : HmRows lay}

variable (lay) in
/-- The commit strings the units of `S` read: those holding one of their wires. -/
def drawnReads (S : Finset (Fin n)) (p : lay.Pos) : Prop :=
  ∃ u ∈ S, ∃ g ∈ P.io u, lay.pos g = p

/-- `rd`'s reads where `S` holds, with the registrant's openings unchanged. -/
def restrict (rd : Reads lay rs) (S : lay.Pos → Prop) : Reads lay rs where
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
noncomputable def regGates (rd : Reads lay rs) : Finset (Fin C.N) :=
  Finset.univ.filter fun g => rd.isRead (lay.pos g)

/-- **The registered values**: each wire's bit in the row the registrant committed at its commit string. -/
def regVals (rd : Reads lay rs) : Fin C.N → Bool := fun g => lay.bit (rd.wv (lay.pos g)) g

variable {Out : Type}

/-- **The registrant's values at the drawn units' registered reads, up to `δ`** (named hypothesis), at strategy `s`: an
accepting run of `G` draws a unit that reads, at a registered gate `g`, the registrant's leaf under the run's
registration `R o`, while `Xc` is not the registrant's bit at `g`, with probability at most `δ`. -/
def RegValsSound (G : FlockSoundness.Game Out) (s : FlockSoundness.Game.Strategy G) (acc : Out → Prop)
    (drawn : Out → Finset (Fin n)) (R : Out → Reg) (rd : Reads lay rs) (Xc : Fin C.N → Bool) (δ : ℝ≥0∞) : Prop :=
  FlockSoundness.Game.prob (fun o => acc o ∧ ∃ u ∈ drawn o, ∃ g ∈ P.io u, rd.isRead (lay.pos g) ∧
      lay.regLeaf (R o) (lay.pos g) = rd.leaf (lay.pos g) ∧ Xc g ≠ regVals rd g) G s ≤ δ

/-- **The registered reads' term at each run's reads and registration, under `SHA512CRStrict` (`cr/sha-512`).** In a
game `G` played by `σ` whose accepted runs pass the registered check at their registration `R o` for the reads `rdo o`,
through their paths: the probability of accepting while some read's leaf is not the registrant's is at most
`qR²/2^513`, for the finder that plays `G` once and outputs the extractor's pair. `Reads.prob_off_le`, at reads and a
registration that depend on the run. -/
theorem prob_off_le (G : FlockSoundness.Game Out) (σ : FlockSoundness.Game.Strategy G) (accept : Out → Prop)
    (R : Out → Reg) (rdo : Out → Reads lay rs) (path : Out → lay.Pos → ByteArray)
    (hchk : ∀ o, accept o → (rdo o).Checked (R o) (path o)) {qR : ℝ}
    (hR : (FlockSoundness.StrictCR.oneRun G σ fun o => (rdo o).pick (R o) (path o)).CR FlockSoundness.Refine.H512 qR) :
    FlockSoundness.Game.prob (fun o => accept o ∧ (rdo o).Off (R o)) G σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  have hle : FlockSoundness.Game.prob (fun o => accept o ∧ (rdo o).Off (R o)) G σ ≤
      FlockSoundness.Game.prob
        (fun o => FlockSoundness.StrictCR.IsColl FlockSoundness.Refine.H512 ((rdo o).pick (R o) (path o))) G σ :=
    FlockSoundness.Game.prob_mono (fun o ⟨ha, hoff⟩ =>
      ((rdo o).openings (R o) (path o) ((rdo o).offAt (R o))).isColl ((rdo o).offAt_conflict (hchk o ha) hoff)) G σ
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
and registration `R o` on each run: if every accepted run passes the registered check on the registrant's reads at the
commit strings its drawn units read (`hchk`), the registered-path finder meets `cr/sha-512` at `qR` (`hR`), and the
registrant's values hold up to `δ` (`RegValsSound`), then registered reads are sound for `Xc` against the registered
values `regVals rd` on the registered gates `regGates rd`, up to `qR²/2^513 + δ`. -/
theorem regReadSound_of_checked (G : FlockSoundness.Game Out) (σ : FlockSoundness.Game.Strategy G)
    (acc : Out → Prop) (drawn : Out → Finset (Fin n)) (R : Out → Reg) (path : Out → lay.Pos → ByteArray)
    (rd : Reads lay rs) (Xc : Fin C.N → Bool)
    (hchk : ∀ o, acc o → (restrict rd (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
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

/-! ## The joint form

C-Flock's extraction bounds the drawn wrong units and the registrant's values in one event, at one `ε_ks + δ_link`
(its joint registered forms). `RegJointSound` is that event; `registered_of_checked_joint` carries it, with the
registered leaf's `qR²/2^513`, to `RegisteredSound` in the core game through `Registered.registered_core_joint`,
counting `ε_ks + δ_link` once where `regReadSound_of_checked` and `Registered.registered_core` count it twice. -/

/-- **The drawn wrong units and the registrant's values together, up to `δ`** (named hypothesis), at strategy `s`: an
accepting run of `G` draws a wrong unit of `Xc`, or a unit that reads, at a registered gate `g`, the registrant's leaf
under the run's registration `R o` while `Xc` is not the registrant's bit at `g`, with probability at most `δ`. Both
halves of the event are `RegValsSound`'s and the drawn wrong units' bound, so it implies each at the same `δ`. -/
def RegJointSound (G : FlockSoundness.Game Out) (s : FlockSoundness.Game.Strategy G) (acc : Out → Prop)
    (drawn : Out → Finset (Fin n)) (R : Out → Reg) (rd : Reads lay rs) (Xc : Fin C.N → Bool) (δ : ℝ≥0∞) : Prop :=
  FlockSoundness.Game.prob (fun o => acc o ∧ (¬ Disjoint (P.wrong Xc) (drawn o) ∨
      ∃ u ∈ drawn o, ∃ g ∈ P.io u, rd.isRead (lay.pos g) ∧
        lay.regLeaf (R o) (lay.pos g) = rd.leaf (lay.pos g) ∧ Xc g ≠ regVals rd g)) G s ≤ δ

/-- **The joint bound, from the registered check.** On `regReadSound_of_checked`'s check and `cr/sha-512` for the
registered-path finder, with the drawn wrong units and the registrant's values bounded together (`RegJointSound`): an
accepting run draws a wrong unit of `Xc` or a unit whose opening shows `Xc` off `regVals rd` on `regGates rd` with
probability at most `qR²/2^513 + δ`. -/
theorem regJoint_of_checked (G : FlockSoundness.Game Out) (σ : FlockSoundness.Game.Strategy G)
    (acc : Out → Prop) (drawn : Out → Finset (Fin n)) (R : Out → Reg) (path : Out → lay.Pos → ByteArray)
    (rd : Reads lay rs) (Xc : Fin C.N → Bool)
    (hchk : ∀ o, acc o → (restrict rd (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
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
    (rd : Reads lay rs) (Xc : Fin C.N → Bool)
    (hchk : ∀ o, acc o → (restrict rd (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
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

end RegisteredReads

end FlockBridge
