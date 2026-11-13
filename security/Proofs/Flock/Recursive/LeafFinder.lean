module

public import Proofs.Flock.Recursive.Flock
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Registered.Binding

/-!
# Recursion: a session's registered read off the table's leaf, priced by a finder of the recursive audit

At an accepted session the verifier's registered check opens each registered read's root at the read's position at the
table's leaf (`ReadsX.Checked`, `checked_of_acceptsHJ`), and the registrant opened it there at its own leaf
(`ReadsX.committed`): the two leaves are one, or the two openings conflict (`TwoOpenings.Conflict`), a SHA-512
collision. The session's own finder outputs that pair behind its `boundIf`, at the session's outcome. At V*'s bridge,
which reads every session's outcome at one history, the pair is a history's, so the finder is the whole audit's:

* `ReadsX.leafCands`: a record's two openings' pair at each position, under a registration and through paths, with the
  test that the position is a registered read whose openings conflict (`ReadsX.leafCands_spec`,
  `ReadsX.leafCands_mem`);
* `LeafColl`, `leafCands`: some session's registered read whose openings conflict, under the registration the
  session's own outcome records and through the paths of the public file header in that outcome's transcript
  (`regPathsHJ`); the candidates are each session's, at its outcome (`leafCands_spec`, `LeafColl.cand`);
* `leafFinder`, `prob_leafColl_le`: the finder that plays the recursive audit once and outputs the pair of the first
  candidate of the audit's outcome whose test passes, a search of the sessions and their positions; under `cr/sha-512`
  (`Finder.CR` at `qL`), the audit's outcome has such a read with probability at most `qL²/2^513`, once per audit. At
  an accepted session the table's leaf at a row read is `treeLeaf` of the public file's digest
  (`FlockC3.RegLevels.regLeaf_inl`), so an honest budget counts the audit run and, at each registered read, the two
  openings' recomputation.
-/

@[expose] public section

namespace FlockSoundness.Registered

open Game Refine Binding Audit

variable {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}

/-- **A record's candidates off the table's leaves**, under the registration `R` through `path`: at each position `p`,
the pair of the session's opening and the registrant's (`TwoOpenings.pair`), with the test that `p` is a registered
read and the two openings conflict. -/
noncomputable def ReadsX.leafCands (rd : ReadsX lay rs) (R : Reg) (path : lay.Pos → ByteArray) :
    List (Prop × (List UInt8 × List UInt8)) :=
  (Finset.univ : Finset lay.Pos).toList.map fun p =>
    (rd.isRead p ∧ (rd.openings R path p).Conflict, (rd.openings R path p).pair)

/-- **Two conflicting openings are a collision**: a candidate whose test passes is one (`TwoOpenings.collides`). -/
theorem ReadsX.leafCands_spec {rd : ReadsX lay rs} {R : Reg} {path : lay.Pos → ByteArray}
    {c : Prop × (List UInt8 × List UInt8)} (hc : c ∈ rd.leafCands R path) (h : c.1) :
    Hash.Collision H512 c.2.1 c.2.2 := by
  obtain ⟨p, -, rfl⟩ := List.mem_map.1 hc
  have hw : rd.isRead p ∧ (rd.openings R path p).Conflict := h
  exact TwoOpenings.collides _ hw.2

/-- A registered read whose openings conflict is a candidate whose test passes. -/
theorem ReadsX.leafCands_mem {rd : ReadsX lay rs} {R : Reg} {path : lay.Pos → ByteArray} {p : lay.Pos}
    (hp : rd.isRead p) (h : (rd.openings R path p).Conflict) : ∃ c ∈ rd.leafCands R path, c.1 :=
  ⟨_, List.mem_map.2 ⟨p, Finset.mem_toList.2 (Finset.mem_univ p), rfl⟩, hp, h⟩

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg
open scoped ENNReal

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type}
variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **A session's registered read is off the table's leaf, by a collision**: at the history `(R, ω, t)`, the
registration `R₂` and the sessions' outcomes `o`, some session `j`'s registered read `p` whose opening and the
registrant's conflict, the session's opening under the registration `o j` records, through the path the public file
header in `o j`'s transcript gives `p` (`regPathsHJ`). -/
def LeafColl (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) :
    Prop :=
  ∃ j p, (zs R ω t R₂ j).rd.isRead p ∧ ((zs R ω t R₂ j).rd.openings (o j).1
    (regPathsHJ (zs R ω t R₂ j).I (Vs j).c (zs R ω t R₂ j).pub (o j).2.2 (zs R ω t R₂ j).rd.port) p).Conflict

/-- **An outcome's candidates**: each session's record's (`ReadsX.leafCands`), under the registration its outcome
records and through its outcome's public file header's paths. -/
noncomputable def leafCands (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg)
    (ω : L.Ω) (t : List (Cm × I.Coin)) (R₂ : Reg₂)
    (o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript) :
    List (Prop × (List UInt8 × List UInt8)) :=
  (List.finRange S).flatMap fun j => (zs R ω t R₂ j).rd.leafCands (o j).1
    (regPathsHJ (zs R ω t R₂ j).I (Vs j).c (zs R ω t R₂ j).pub (o j).2.2 (zs R ω t R₂ j).rd.port)

/-- **A read off the table's leaf is a collision**: a candidate whose test passes is one. -/
theorem leafCands_spec {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg}
    {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript} {c : Prop × (List UInt8 × List UInt8)}
    (hc : c ∈ leafCands zs R ω t R₂ o) (h : c.1) : Hash.Collision H512 c.2.1 c.2.2 := by
  obtain ⟨j, -, hc⟩ := List.mem_flatMap.1 hc
  exact Registered.ReadsX.leafCands_spec hc h

/-- An outcome where a session's read is off the table's leaf has a candidate whose test passes. -/
theorem LeafColl.cand {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg}
    {ω : L.Ω} {t : List (Cm × I.Coin)} {R₂ : Reg₂}
    {o : (j : Fin S) → Reg' × Finset (Fin (Vs j).nv) × Exec.Transcript} (h : LeafColl zs R ω t R₂ o) :
    ∃ c ∈ leafCands zs R ω t R₂ o, c.1 := by
  obtain ⟨j, p, hp, h⟩ := h
  obtain ⟨c, hc, hc1⟩ := Registered.ReadsX.leafCands_mem hp h
  exact ⟨c, List.mem_flatMap.2 ⟨j, List.mem_finRange j, hc⟩, hc1⟩

/-- **The table leaves' finder**: one run of the recursive audit against the prover `σ`, output the pair of the first
of the outcome's candidates (`leafCands`, at the outcome's history, registration and sessions' outcomes) whose test
passes: a session's two conflicting openings' pair where a read is off the table's leaf (`LeafColl`), else nothing.
Every candidate is computed from what the run holds: the session's record, and the registration and transcript its
outcome records. -/
noncomputable def leafFinder (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) :
    Finder Pair :=
  oneRun _ σ (onePick fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
    leafCands zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)

/-- **A session's read is off the table's leaf at most `qL²/2^513` likely** under `cr/sha-512` for `leafFinder` at the
budget `qL`: once per audit, whatever the number of sessions and reads. -/
theorem prob_leafColl_le (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ) (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qL : ℝ} (hL : (leafFinder zs dg N σ).CR H512 qL) :
    prob (fun o => LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤ ENNReal.ofReal (qL ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qL ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le H512 _ σ (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      LeafColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (fun o => leafCands zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 fun j => (o.2.2.2.2 j).2)
      (fun _ h => h.cand) fun _ _ hc hp => leafCands_spec hc hp).trans hL.le
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

end FlockSoundness.Discharge.Recursion
