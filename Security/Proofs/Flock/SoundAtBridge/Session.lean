import Proofs.Flock.EndToEndRegisteredRows.Statement

/-!
# One C-Flock session with a registrant's opening per unit (trusted)

`RowsSession` is `Flock.Guarantees.EndToEndRegisteredRows`' setting as a value, its binders in its order, with its one
registrant's opening `rd` replaced by one per unit (`rdU u`); its circuit `c` is the one its descriptor `D` unrolls
to. `EndToEndRegisteredRows` reads the input rows only where the verifier owns every registered read of `rd` at the
outcome's draw (`OwnReadsHJ` is one condition over all of them), so at an `rd` covering every unit's reads it links no input row on a draw that leaves one out. With `rdU u` the
opening of unit `u`'s reads alone (`Registered.ReadsX.restrict`, in `Proofs.Flock.Recursive.ZkReg.Off`), a drawn
unit's input rows are linked wherever its own reads are owned, whatever else is drawn; the cost is a union bound over
the `n` units (`Flock.SoundAtBridge.RowsSession.prob_badU_le`), `n` times `EndToEndRegisteredRows`' bound.

The session's game, prover, value layer and events are those `EndToEndRegisteredRows` reads, at `rdU u` for unit
`u`:

* `G`, `τ`: the live audit and its prover; `Out`: its outcome (the registration, the draw and the transcript);
* `Xv`: the plurality value layer the rows extraction finds at the registration (`XplurZC`);
* `WrongAt u`: drawn unit `u` is wrong on the registration (`wrongRegZ`);
* `RowsAt o u`: unit `u`'s rows are right at `Xv` (`RowsRightZC`) at its own owned registered reads;
* `BadU o`: flock-verify accepts and some drawn unit is wrong or its rows are off;
* `CR`: `EndToEndRegisteredRows`' named assumptions at every unit's opening; `bound`: its bound.
-/

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkReg
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal
open Classical

/-- **One C-Flock session in `EndToEndRegisteredRows`' setting, with a registrant's opening per unit**: its binders,
in its order (the roots `own`, then the statement `I` with its coin, then the live prover `τ`), with its one `rd`
replaced by `rdU u` for each unit `u`. -/
structure RowsSession where
  n : ℕ
  Reg : Type
  own : Array Flock.Registered.Port
  I : Inputs
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  kd : ℕ
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  rdU : Fin n → FlockSoundness.Registered.ReadsX
    (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
    (rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub)
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [instF : Fintype Ω]
  [instN : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr))
  qF : ℝ
  qS : ℝ
  qR : ℝ

attribute [instance] RowsSession.instF RowsSession.instN

noncomputable section

namespace RowsSession

variable (s : RowsSession)

/-- The circuit the session's descriptor unrolls to. -/
abbrev c : Flock.Circuit := s.D.unroll

/-- `HmOut.parse` reads the session's circuit from its circuit file. -/
theorem hc : HmOut.parse (zkTags s.I) s.I.circuitFile s.I.tables = .ok s.c := Flock.Desc.parse_flat s.hD

/-- The session's outcome: the registration, the draw and the transcript. -/
abbrev Out : Type := s.Reg × Finset (Fin s.n) × Discharge.Exec.Transcript

/-- The verifier's draw. -/
abbrev L := Law.execOS (n := s.n) s.src (.subset s.kd) s.n

/-- The live audit. -/
abbrev G : Game s.Out := auditLive s.L s.Reg (liveOfZJ s.dg (planZHJ s.dj s.y₀) s.N s.wr)

/-- The key program's partition. -/
abbrev P := (keyProg s.c s.hU s.n).partition (Discharge.Layout.outsOf s.c s.hU s.n)

/-- The key program's circuit, whose wires the value layer assigns. -/
abbrev Circ := (keyProg s.c s.hU s.n).circuit (Discharge.Layout.outsOf s.c s.hU s.n)

/-- The value binding of the session's rows. -/
abbrev vb := (rsZHJS (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub).bindingZ
  (rsZHJS_computes (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub)

/-- Unit `u`'s registrant's openings. -/
abbrev wo (u : Fin s.n) : s.vb.Pos → s.vb.Opening := fun p => ((p, (s.rdU u).wv p, (s.rdU u).ws p) : s.vb.Opening)

/-- The simulated audit's strategy the live prover's play gives. -/
abbrev σ := simZJ execArith hs512 (planZHJ s.dj s.y₀) s.I s.wr (opnAtZHJ (planZHJ s.dj s.y₀) s.I s.dj) s.dg
  (decAtZJ (planZHJ s.dj s.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) s.N
  (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ s.I s.wr _ _ _ _ _
    (runsZCJH_live s.I s.ht s.hc (Discharge.Layout.scopeOkH_spec s.hU s.hscope).2.1 s.dj s.y₀ s.wr _ s.hTags s.hCust.1
      fun _ => Discharge.ZkReg.drawOkZKHJ_of_file s.I s.ht s.hc s.kd s.n s.hdiv s.dj s.hCust.2 _)) s.τ

/-- **The value layer**: the plurality the rows extraction finds at the registration (`XplurZC`). -/
abbrev Xv : Fin s.Circ.N → Bool :=
  XplurZC execArith hs512 s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) (cont s.σ) s.k s.Rw

/-- Unit `u` is wrong on the registration, at the value layer (`wrongRegZ`). -/
def WrongAt (u : Fin s.n) : Prop :=
  u ∈ wrongRegZ execArith s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) s.Xv

/-- Unit `u`'s rows are right at the value layer (`RowsRightZC`), at its own registered reads (`rdU u`) where the
verifier owns them at the outcome `o`'s draw. -/
def RowsAt (o : s.Out) (u : Fin s.n) : Prop :=
  RowsRightZC execArith hs512 s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) (cont s.σ)
    (fun p => OwnReadsHJ s.I s.own s.c s.pub o.2.2 (s.rdU u).isRead (s.rdU u).port (s.rdU u).readAt ∧
      (s.rdU u).isRead p) (s.wo u) s.k s.Rw u

/-- **The per-unit event**: flock-verify accepts, and some drawn unit is wrong, or its rows are off at its own
registered reads. -/
def BadU (o : s.Out) : Prop := LiveAcceptsZKR s.I s.own o ∧ ∃ u ∈ o.2.1, s.WrongAt u ∨ ¬ s.RowsAt o u

/-- `EndToEndRegisteredRows`' event at unit `u`'s opening `rdU u`. -/
def EvAt (u : Fin s.n) (o : s.Out) : Prop :=
  LiveAcceptsZKR s.I s.own o ∧ ∃ u' ∈ o.2.1,
    u' ∈ wrongRegZ execArith s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ)
      (XplurZC execArith hs512 s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) (cont s.σ) s.k s.Rw) ∨
    ¬ RowsRightZC execArith hs512 s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) (cont s.σ)
      (fun p => OwnReadsHJ s.I s.own s.c s.pub o.2.2 (s.rdU u).isRead (s.rdU u).port (s.rdU u).readAt ∧
        (s.rdU u).isRead p) (s.wo u) s.k s.Rw u'

/-- **`EndToEndRegisteredRows`' named assumptions at every unit's opening**: `cr/sha-512` for each table's knowledge
finders (`TableCRZ`), `ecr/sha-512` for the link finders (`LinkCRZC`), and, at each unit's opening `rdU u`,
`ecr/sha-512` for its rows finders (`RowsCRZC`) and `cr/sha-512` at `qR` for its registered reads' finder. -/
def CR : Prop :=
  (∀ ω j, TableCRZ execArith hs512 (planZHJ s.dj s.y₀ (s.L.draw ω) (reg s.σ)) j
      (zcStrat execArith hs512 (planZHJ s.dj s.y₀) (cont s.σ ω)) s.qF s.qS) ∧
    LinkCRZC execArith hs512 s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) (cont s.σ) s.k s.Rw s.M s.t' ∧
    (∀ u, RowsCRZC execArith hs512 s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb (reg s.σ) (cont s.σ)
      (s.rdU u).isRead (s.wo u) s.k s.Rw s.t') ∧
    (∀ u, (StrictCR.oneRun s.G s.τ fun o => (s.rdU u).pick o.1 (regPathsHJ s.I s.c s.pub o.2.2 (s.rdU u).port)).CR
      H512 s.qR)

/-- `δ_link`'s closed form (also `δ_rows`'). -/
def δlink : ℝ :=
  (∑ p : s.vb.Pos,
      (Fintype.card (ΩcZ execArith s.P (planZHJ s.dj s.y₀) (tabZHJ s.dj s.y₀) s.vb s.L p) : ℝ) / Fintype.card s.L.Ω) *
    (2 * (s.t' * (1 + s.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * s.M) + s.k / (Real.exp 1 * s.Rw)) / (1 - s.ρ)

/-- **`EndToEndRegisteredRows`' bound**: `ε_ks + δ_link + δ_rows + q_R²/2^513`. -/
def bound : ℝ≥0∞ :=
  ksAvgStrictZ (planZHJ s.dj s.y₀) s.L s.k s.qF s.qS (reg s.σ) + ENNReal.ofReal s.δlink + ENNReal.ofReal s.δlink +
    ENNReal.ofReal (s.qR ^ 2 / 2 ^ (513 : ℝ))

end RowsSession

end

end Flock.SoundAtBridge
