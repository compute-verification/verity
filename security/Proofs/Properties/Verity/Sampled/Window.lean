module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.EndToEnd.Joint.V5.Bound

/-!
# Sampled proofs' leg over a window of several members, each its own session

`Verity.Vocab.SpotSound` where a window's one sample covers several members and one-stage v2 verifies each member in
its own `--zk` session, one after another: the leg `Proofs.Properties.Verity.Sampled.Target` proves at one session, at
Flock's window of several members (`Flock.Guarantees.JointAuditV5`). The sessions are sequential, in the members'
order (`JointAuditV5.sessions`, `seqG`); sessions run at once are not this leg's.

* **A member** (`Member`): `MemberV5`'s binders, each the value v2's check derives on the member's own statement (its
  circuit file's parse, the public file over the circuit's empty public input ports, where its proof blocks sit in the
  window's population, `emb`), with the bound's parameters. What the period's registration fixes is the member's, not
  a field (`Member.toV5`): the registration is the window's joint one, the registration record's digest; the member's
  sample file at its part `S` of the window's sample is the `.subset |S|` print (`printedPart`); each round's digest is
  SHA-512; the writer is `wrZP`; and `live-verifier` holds at it with no premise (`recordCustodyZKJPLaw_part`).
* **The window** (`Window`): the members in the registration's order, the registration's law on its strata over the
  population of all their proof blocks, the device's one byte stream the sample reads with A3 at it, each member's
  one link-point count at every sample (`JointAuditV5.hcls`), and the claim's share. `Window.joint` is Flock's window
  at `t` SHA-512 calls (the finders' budgets are `MemberV5`'s fields), and `Window.extend` the window with the sample's
  coins widened by the coins a projection reads.
* **The bound** (`Window.sessionDelta`, `Window.delta`): the sample's own miss of any set of more than an `ε` share of
  the population (`Window.drawMiss`) plus `JointAuditV5.eps`, the members' knowledge and binding terms summed,
  `Σ_m (ε_ks,m + ε_bind,m)`, plus each member's registration's roots term (`Window.rootsErr`, `Member.rootsErr`):
  its roots finder's SHA-512 calls over `2^256`, the form `Vocab.finderPr_le_crBound` gives the read finder; then one
  two-continuation term per round of every member's session, the record's and the bound digests', the coin trees'
  hiding term and the read finder's bound.
* **One SHA-512 on messages** (`sha512Msg_eq`): the vocabulary's `Verity.sha512Msg` is the claims'
  `Claims.Sampling.sha512Msg`, by definition, so a collision the claims' lemmas exhibit is one the vocabulary's
  finders' premises speak of.
* **The roots guard** (`Member.RootsFix`, `Window.RootsFix`, `Member.RootsCR`, `Window.RootsCR`,
  `le_add_rootsErr`): where each member's roots finder meets its collision premise at its calls (`Member.RootsCR`,
  `FinderSpec.CR` under `ecr/sha-512`), a probability that is at most `x` wherever every member's roots at its public
  file's leaf map fix their domains is at most `x` plus the members' roots terms, at every registration: a member
  whose roots don't is a SHA-512 collision its roots finder outputs, and its term is then at least 1. So the claim's
  transfer is asked only where every member's roots fix their domains, of the members with a proof block in the
  sample and of those with none alike.
* **`Window.ks`**: Flock's bound at the window, `Flock.SecurityProofs.JointShareV5` applied, not proved again, under
  the roots guard and the roots finders' premises: a strategy of the window's audit at which, wherever every
  member's roots fix their domains, more than an `ε` share of the population is incorrect is accepted with
  probability at most `Window.sessionDelta`.
* **The projection's target** (`WindowCouplingTarget`): the sampled verdict in the period's game is at most as likely
  as the window's audit accepting (`JointAuditV5.Accepts`: the session of every member with a proof block in the
  sample) at the projected strategy, plus the two-continuation finders' terms, the hiding term, the read finder's
  collision and the forger's forgery.
* **The vocabulary at the deployment's windows** (`WindowDeployed`, `WindowCovers`), and **the claim's transfer**
  (`WindowDeployed.claim`): where every member's roots fix their domains and the claim fails at a strategy, more than
  an `ε` share of the window's population is incorrect at the projected strategy (`JointAuditV5.wrongSet`).
* **`spotSound_of_window`**: the leg, from the coupling at each window.
-/

@[expose] public section

namespace Verity.Sampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The vocabulary's SHA-512 on messages is the claims'** (`Verity.sha512Msg`, `Claims.Sampling.sha512Msg`): each
is the compiled SHA-512 of the message's bytes, one function by definition. A finder's collision probability at one
is its probability at the other. -/
theorem sha512Msg_eq : Verity.sha512Msg = Claims.Sampling.sha512Msg := rfl

/-- **A member of a window, as v2 sets its session up on the member's own statement**: `MemberV5`'s binders, each the
value v2's check derives (the member's circuit file's parse, its public file over the circuit's empty public input
ports, one table), where its `n` proof blocks sit in the window's population of `nJ` (`emb`), the size `kd` its
session's link points are those of, the reference setup, the verify command's tags, and the bound's parameters. The
registration, the sample file, the writer and the rounds' digest are the ones the period's registration fixes
(`Member.toV5`). -/
structure Member (nJ : ℕ) where
  /-- the member's population: its public file's proof blocks -/
  n : ℕ
  I : FlockSoundness.Discharge.Exec.Inputs
  /-- the verifier's own recomputation of each proof block's public input words -/
  own : Option ByteArray
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : FlockSoundness.Built D.unroll.unit
  /-- the circuit names no public input port -/
  hpi : Flock.HmIn.parse D.unroll = .ok {}
  pub : Flock.Public
  hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll {} I.publicFile = .ok pub
  hscope : FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true
  /-- one table -/
  hcount : I.count ≤ 1
  /-- where the member's proof blocks sit in the window's population -/
  emb : Fin n → Fin nJ
  /-- the size whose link points the member's session has at every sample (`Window.hcls`) -/
  kd : ℕ
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  hTags : TagsOkZHJ I
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJP (Reg := ByteArray) (printedPart (n := n)) y₀ S R) j k ≤ ρ
  /-- one opening's verification, in SHA-512 calls -/
  v : ℕ
  /-- one acceptance check, in SHA-512 calls -/
  w : ℕ

namespace Member

variable {nJ : ℕ} (m : Member nJ)

/-- The member's session's coin tree (`Live.coinSpec`). -/
def coinSpec : Flock.CoinTree.Spec := Flock.Zk.coinSpec (m.I.tags.tableNames m.I.count)

/-- The member's session's live rounds (`Live.N`). -/
def N : ℕ := 2 * 2 ^ m.coinSpec.roundsLog

/-- The tree openings the member's session reads after a digest of the prover's message: its link points', each
round of its coin tree's two rep streams (`Live.rounds`), and its closing round, the opening with no coins that
answers the digest of the session's evidence and so puts that evidence in the device's log before the next
member's coins. -/
def rounds : ℕ := 2 * 2 ^ m.coinSpec.roundsLog + 2

/-- **Flock's member at `t` SHA-512 calls**: the member's statement and place in the population; the window's joint
registration, the registration record's digest (`Live.Reg`); the member's sample file at its part `S`, the
`.subset |S|` print (`printedPart`); SHA-512 for a round's digest; the writer that keeps an accepted record (`wrZP`),
at which `live-verifier` holds with no premise (`recordCustodyZKJPLaw_part`); the link finders charged `t + w` per run
and the tables' finders at `2t + 2v` and `t + v`. -/
noncomputable def toV5 (t : ℕ) : Flock.Guarantees.MemberV5 ByteArray nJ where
  n := m.n
  I := m.I
  own := m.own
  ht := m.ht
  D := m.D
  hD := m.hD
  hU := m.hU
  pi := {}
  hpi := m.hpi
  pub := m.pub
  hpub := m.hpub
  hscope := m.hscope
  hcount := m.hcount
  emb := m.emb
  kd := m.kd
  dj := printedPart
  S₀ := m.S₀
  d₀ := m.d₀
  y₀ := m.y₀
  wr := Claims.Bridge.CustodyZkP.wrZP m.I m.own printedPart
    fun S R => (planZHJP (Reg := ByteArray) printedPart m.y₀ S R).length
  dg := Flock.Sha512.hash
  N := m.N
  hTags := m.hTags
  hCust := recordCustodyZKJPLaw_part m.I m.own
    fun S R => (planZHJP (Reg := ByteArray) printedPart m.y₀ S R).length
  k := m.k
  Rw := m.Rw
  M := m.M
  hk := m.hk
  hRw := m.hRw
  hM := m.hM
  ρ := m.ρ
  hρ := m.hρ
  hr := m.hr
  t' := (t : ℝ) + m.w
  ht' := by positivity
  qF := 2 * (t : ℝ) + 2 * m.v
  qS := (t : ℝ) + m.v

/-- **The member's roots at its public file's leaf map fix their domains** (`Flock.Bridge.RootFixesDomainL`): two of
its ports with one root publish one domain. -/
def RootsFix : Prop :=
  Flock.Bridge.RootFixesDomainL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

/-- **The member's roots finder** (`Flock.Bridge.rootsFinderL` at the member's public file's leaf map): no coins; it
recomputes the member's roots and outputs the collision that two of its ports with one root and different domains
are. -/
noncomputable def rootsFinder : Claims.Finder :=
  Flock.Bridge.rootsFinderL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

/-- **The roots finder's SHA-512 calls** (`Flock.Bridge.rootsCostL`), on its one coin. -/
noncomputable def rootsCalls : ℕ :=
  Flock.Bridge.rootsCostL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n)

/-- **The roots finder's collision premise** (`FinderSpec.CR` under `ecr/sha-512`, the form
`Vocab.finderPr_le_crBound` reads the read finder's in): it outputs a SHA-512 collision, at the vocabulary's SHA-512,
with probability at most its calls over `2^256`. -/
def RootsCR : Prop := m.rootsFinder.Pr Verity.sha512Msg ≤ (m.rootsCalls : ℝ) / 2 ^ (256 : ℝ)

/-- **The member's registration's roots term**: the roots finder's calls over `2^256`, what `ecr/sha-512` gives a
finder of that many calls (`RootsCR`). Two of the member's ports with one root and different domains are a collision
the finder outputs. -/
noncomputable def rootsErr : ℝ≥0∞ := ENNReal.ofReal ((m.rootsCalls : ℝ) / 2 ^ (256 : ℝ))

/-- **A member's roots fix their domains, or its roots term is at least 1**, where its roots finder meets its
premise (`RootsCR`): roots that don't expose a collision (`Flock.Bridge.fixesL_of_none`,
`Flock.Bridge.rootsCollL_collides`) that the finder outputs on its one coin, at the claims' SHA-512 and so at the
vocabulary's (`sha512Msg_eq`), so the premise puts its calls over `2^256` at 1 or more. -/
theorem rootsFix_or_one_le (h : m.RootsCR) : m.RootsFix ∨ 1 ≤ m.rootsErr := by
  cases hc : Flock.Bridge.rootsCollL m.D.unroll (Flock.Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n) with
  | none => exact Or.inl (Flock.Bridge.fixesL_of_none hc)
  | some z =>
    have hPr : m.rootsFinder.Pr Verity.sha512Msg = 1 := by
      rw [sha512Msg_eq]
      exact Flock.Bridge.PortRoots.finderPr_one fun _ =>
        ⟨z.1, z.2, hc, Flock.Bridge.rootsCollL_collides (x := z.1) (y := z.2) hc⟩
    have h1 : (1 : ℝ) ≤ (m.rootsCalls : ℝ) / 2 ^ (256 : ℝ) := hPr.symm.le.trans h
    exact Or.inr (ENNReal.one_le_ofReal.2 h1)

/-- **The member's part of a sample** of the window's population (`MemberV5.part`): its proof blocks the sample
holds. -/
noncomputable def part (S : Finset (Fin nJ)) : Finset (Fin m.n) := (m.toV5 0).part S

/-- The member's part of a sample is Flock's member's, at any budget. -/
theorem toV5_part (t : ℕ) (S : Finset (Fin nJ)) : (m.toV5 t).part S = m.part S := rfl

end Member

/-- **A window of several members, as v2 sets its sessions up**: the members in the registration's order; the
registration's law on its strata over the population of all their `nJ` proof blocks, sampling `kd`; the device's one
byte stream the sample reads, with A3 at it; each member's part of every sample at its session's link points (the
registration check refuses a member whose fewest and most sampled proof blocks have different counts); the claim's
share; and the SHA-512 calls of hashing one message a digest is taken of. -/
structure Window where
  /-- the window's population: every member's proof blocks -/
  nJ : ℕ
  /-- the members, in the registration's order -/
  ms : List (Member nJ)
  law : Flock.Draw.Law
  ns : List ℕ
  kd : ℕ
  hns : ns.sum = nJ
  hlaw : Flock.Bridge.LawOn law ns kd
  /-- the device's bytes the sample reads -/
  Ω : Type
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hA3 : FlockSoundness.Assumptions.UniformRandomBytes Lb src
  hcls : ∀ m ∈ ms, ∀ ω : Ω,
    mPtsOf m.D.unroll.g (m.part ((Flock.Bridge.execSel (n := nJ) src law nJ).draw ω)).card =
      mPtsOf m.D.unroll.g (m.kd / nTab m.I)
  /-- the claim's share: at most an `ε` share of the population's proof blocks unbound -/
  ε : ℝ
  /-- hashing one message the device or the commitment takes a digest of, in SHA-512 calls -/
  h : ℕ

attribute [instance] Window.fΩ Window.nΩ

namespace Window

variable (W : Window)

/-- **Flock's window at `t` SHA-512 calls**: the members as Flock's (`Member.toV5`), at the joint registration, the
registration's law and strata and the device's bytes. -/
noncomputable def joint (t : ℕ) : Flock.Guarantees.JointAuditV5 where
  Reg := ByteArray
  nJ := W.nJ
  ms := W.ms.map fun m => m.toV5 t
  law := W.law
  ns := W.ns
  kd := W.kd
  hns := W.hns
  hlaw := W.hlaw
  Ω := W.Ω
  fintypeΩ := W.fΩ
  nonemptyΩ := W.nΩ
  Lb := W.Lb
  src := W.src
  hcls := fun k ω => by
    obtain ⟨m, hm, he⟩ := List.mem_map.1 (List.get_mem (W.ms.map fun m => m.toV5 t) k)
    rw [← he]
    exact W.hcls m hm ω

/-- **The window with the sample's coins widened by the coins `S`**: the sample reads only the device's bytes, so its
law is the window's and A3 holds at the widened coins (`uniform_fst`). -/
noncomputable def extend (S : Widening) : Window where
  __ := W
  Ω := W.Ω × S.T
  fΩ := inferInstance
  nΩ := inferInstance
  src ω := W.src ω.1
  hA3 := uniform_fst W.hA3
  hcls := fun m hm ω => W.hcls m hm ω.1

/-- **The window's sample's law**: the registration's law over the population, as `Spot.Draw.drawUnits` samples it on
the device's bytes in order (`Flock.Bridge.execSel`); a sample that fails samples every proof block. -/
noncomputable def L : FlockSoundness.Audit.Law W.nJ := Flock.Bridge.execSel (n := W.nJ) W.src W.law W.nJ

/-- The window's sample is Flock's window's, at any budget. -/
theorem joint_L (t : ℕ) : (W.joint t).L = W.L := rfl

/-- The window's audit's outcome: the registration, the sample and the members' transcripts in order. -/
abbrev Out : Type := ByteArray × Finset (Fin W.nJ) × List FlockSoundness.Discharge.Exec.Transcript

/-- **The rounds the device opens after a digest of the prover's message, over the window**: every member's
session's (`Member.rounds`). -/
def rounds : ℕ := (W.ms.map Member.rounds).sum

/-- The two-continuation finders' budget at `t` calls, `2t + 2h` (`Live.qM`). -/
def qM (t : ℕ) : ℝ := 2 * t + 2 * W.h

/-- **The coin trees' hiding term** (`Live.hidingTerm`): each leaf the device's coin trees hold in a play, twice, at
`2^-257`; the device's caps bound a play's leaves by `maxLeaves` however many sessions it runs. -/
noncomputable def hidingTerm (_ : Window) : ℝ≥0∞ := 2 * (maxLeaves : ℝ≥0∞) / 2 ^ 257

/-- The sampling design's own miss of a set `B` of the population's proof blocks (`JointAuditV5.pa`). -/
noncomputable def pa (B : Finset (Fin W.nJ)) : ℝ :=
  (_root_.Properties.Spot.oneStageOf W.law).Pa W.ns (B.map Fin.valEmbedding)

/-- **The sample's term**: the most the sampling design misses of any set of more than an `ε` share of the
population's proof blocks, `JointShareV5`'s. -/
noncomputable def drawMiss : ℝ≥0∞ :=
  ENNReal.ofReal (⨆ B : {B : Finset (Fin W.nJ) // W.ε * W.nJ < B.card}, W.pa B.1)

/-- **Every member's roots fix their domains** (`Member.RootsFix`), the members with a proof block in the sample and
those with none. -/
def RootsFix : Prop := ∀ m ∈ W.ms, m.RootsFix

/-- **Every member's roots finder meets its collision premise** (`Member.RootsCR`). -/
def RootsCR : Prop := ∀ m ∈ W.ms, m.RootsCR

/-- **The registrations' roots term**: each member's (`Member.rootsErr`), summed. -/
noncomputable def rootsErr : ℝ≥0∞ := (W.ms.map Member.rootsErr).sum

/-- **The window's error at `t` calls**, `JointShareV5`'s under the roots guard: the sample's term, the members'
knowledge and binding terms summed, `Σ_m (ε_ks,m + ε_bind,m)` (`JointAuditV5.eps`), each at its part's sampling
design and the finders' budgets at `t`, and each member's registration's roots term (`rootsErr`). -/
noncomputable def sessionDelta (t : ℕ) : ℝ≥0∞ := W.drawMiss + (W.joint t).eps + W.rootsErr

/-- **The leg's δ at the window, `t` SHA-512 calls and the read finder's bound `cr`**: the window's error, each
member's roots term in it, a two-continuation term per round opened after a digest in every member's session, its
closing round among them, one for the record and one for the digests the composition's commitment binds, the coin
trees' hiding term, and `cr`. -/
noncomputable def delta (t : ℕ) (cr : ℝ) : ℝ≥0∞ :=
  W.sessionDelta t + W.rounds * Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + W.hidingTerm +
    ENNReal.ofReal cr

theorem extend_rounds (S : Widening) : (W.extend S).rounds = W.rounds := rfl

theorem extend_qM (S : Widening) (t : ℕ) : (W.extend S).qM t = W.qM t := rfl

theorem extend_hidingTerm (S : Widening) : (W.extend S).hidingTerm = W.hidingTerm := rfl

/-- The roots term doesn't read the widening: the members are the window's. -/
theorem rootsErr_extend (S : Widening) : (W.extend S).rootsErr = W.rootsErr := rfl

/-- The roots finders' premises don't read the widening: the members are the window's. -/
theorem rootsCR_extend (S : Widening) : (W.extend S).RootsCR ↔ W.RootsCR := Iff.rfl

end Window

/-- **`JointShareV5`'s per-finder SHA-512 premises at a window's audit `w` and a strategy `τ` of it**: `cr/sha-512` at
each table's knowledge finders (`TableCRZAt`) and `ecr/sha-512` at the link finders (`LinkCRZCAt`), at each member's
session played alone at `τ`, at the member's budgets. -/
structure JointFinderCR (w : Flock.Guarantees.JointAuditV5) (τ : FlockSoundness.Game.Strategy w.game) : Prop where
  table : ∀ k ω j, Assumptions.TableCRZAt execArith hs512
    (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
    (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
    (w.ms.get k).qF (w.ms.get k).qS
  link : ∀ k, Assumptions.LinkCRZCAt execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
    (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
    (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t'

/-- **The roots guard over members costs each member's roots term** (`Member.rootsFix_or_one_le`, a member at a
time): where each member's roots finder meets its premise (`Member.RootsCR`), a probability `p ≤ 1` that is at most
`x` wherever every member's roots fix their domains is at most `x` plus the members' roots terms. -/
theorem le_add_rootsErr {nJ : ℕ} {p : ℝ≥0∞} (hp : p ≤ 1) : ∀ (ms : List (Member nJ)) (x : ℝ≥0∞),
    (∀ m ∈ ms, m.RootsCR) → ((∀ m ∈ ms, m.RootsFix) → p ≤ x) → p ≤ x + (ms.map Member.rootsErr).sum
  | [], _, _, h => (h fun _ hm => absurd hm List.not_mem_nil).trans le_self_add
  | m :: ms, x, hcr, h => by
    have hm : (∀ m' ∈ ms, m'.RootsFix) → p ≤ x + m.rootsErr := fun hms => by
      rcases m.rootsFix_or_one_le (hcr m (List.mem_cons.2 (Or.inl rfl))) with hf | h1
      · refine (h fun m' hm' => ?_).trans le_self_add
        rcases List.mem_cons.1 hm' with e | hin
        · rw [e]
          exact hf
        · exact hms m' hin
      · exact hp.trans (h1.trans le_add_self)
    rw [List.map_cons, List.sum_cons, ← add_assoc]
    exact le_add_rootsErr hp ms (x + m.rootsErr) (fun m' hm' => hcr m' (List.mem_cons.2 (Or.inr hm'))) hm

/-- **The window's knowledge soundness at a claim that fails**: a strategy of the window's audit at which, wherever
every member's roots fix their domains (`Window.RootsFix`), more than an `ε` share of the population's proof blocks
is incorrect (`JointAuditV5.wrongSet`) is accepted, the session of every member with a proof block in the sample,
with probability at most the window's error at `t` calls, where the per-finder premises hold at each member's
session played alone and each member's roots finder meets its own (`Window.RootsCR`). It is
`Flock.SecurityProofs.JointShareV5` at the window, the sample's miss and the members' terms summed, under the roots
guard (`le_add_rootsErr`), which costs each member's roots term. -/
theorem Window.ks (W : Window) (t : ℕ) (τ : FlockSoundness.Game.Strategy (W.joint t).game)
    (hF : JointFinderCR (W.joint t) τ) (hcr : W.RootsCR)
    (hI : W.RootsFix → W.ε * W.nJ < (((W.joint t).wrongSet τ).card : ℝ)) :
    Verity.Game.prob (fun o => (W.joint t).Accepts o) (W.joint t).game τ ≤ W.sessionDelta t :=
  show _ ≤ W.drawMiss + (W.joint t).eps + (W.ms.map Member.rootsErr).sum from
    le_add_rootsErr (Verity.Game.prob_le_one _ _ τ) W.ms (W.drawMiss + (W.joint t).eps) hcr fun hR =>
      (Verity.Game.prob_mono (fun _ ho => ⟨ho, hI hR⟩) _ τ).trans
        (Flock.SecurityProofs.JointShareV5 (W.joint t) W.hA3 τ hF.table hF.link W.ε)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The constructions the leg reads at a window, applied to a strategy of the period's game** (`Constructions`, at a
window): the projection, the device's other releases the strategy may ask for (`coins`) and the strategy of the
window's audit it is read as, at the window widened by them and the strategy's SHA-512 calls (`project`); the read
finder and the forger of the device's signature; and the two-continuation finders, one per round the device opens
after a digest of the prover's message in any member's session, one for the record the commitment is a digest of, and
one for the digests the composition's commitment binds, each at most `2t + 2h` SHA-512 calls on every coin. -/
structure WindowConstructions (V : Vocab d α) where
  coins : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Widening
  project : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    (σ : Verity.Game.Strategy (V.game θ F c)) →
      FlockSoundness.Game.Strategy ((W.extend (coins W θ F c σ)).joint (V.cost θ F c σ)).game
  readFinder : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    Claims.Finder
  forger : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
    Hash.Assumptions.LmsForger
  msgFinder : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Fin W.rounds → Claims.Finder
  recordFinder : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Claims.Finder
  bindFinder : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) →
    Verity.Game.Strategy (V.game θ F c) → Claims.Finder
  msg_calls : ∀ W θ F c σ j (x : (msgFinder W θ F c σ j).S),
    ((msgFinder W θ F c σ j).cost x : ℝ) ≤ W.qM (V.cost θ F c σ)
  record_calls : ∀ W θ F c σ (x : (recordFinder W θ F c σ).S),
    ((recordFinder W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ)
  bind_calls : ∀ W θ F c σ (x : (bindFinder W θ F c σ).S),
    ((bindFinder W θ F c σ).cost x : ℝ) ≤ W.qM (V.cost θ F c σ)

/-- **The window a strategy of the period's game is projected into**: `W` widened by the coins the projection
reads. -/
noncomputable abbrev WindowConstructions.live {V : Vocab d α} (K : WindowConstructions V) (W : Window)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Window :=
  W.extend (K.coins W θ F c σ)

/-- **The leg's per-finder SHA-512 premises at a strategy and the window `W`**: `JointShareV5`'s at the projected
strategy and the strategy's calls (`JointFinderCR`), `cr/sha-512` as the strict birthday bound at each
two-continuation finder, and `ecr/sha-512` at each member's roots finder at its calls (`Window.RootsCR`). -/
structure WindowFinderCR (V : Vocab d α) (K : WindowConstructions V) (W : Window) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Prop where
  session : JointFinderCR ((K.live W θ F c σ).joint (V.cost θ F c σ)) (K.project W θ F c σ)
  msg : ∀ j, Claims.StrictBirthday Verity.sha512Msg (K.msgFinder W θ F c σ j)
  record : Claims.StrictBirthday Verity.sha512Msg (K.recordFinder W θ F c σ)
  bind : Claims.StrictBirthday Verity.sha512Msg (K.bindFinder W θ F c σ)
  roots : W.RootsCR

/-- **The coupling, the projection's target at the window `W`** (`CouplingTarget`, at a window): the plays of the
period's game whose sampled verdict accepts are at most as likely as the plays of the window's audit, at the window
widened by the projection's coins and the strategy projected there, that flock-verify `--zk` accepts at the session of
every member with a proof block in the sample (`JointAuditV5.Accepts`), plus the square root of each two-continuation
finder's success, the coin trees' hiding term, the read finder's collision and the forger's forgery of the device's
signature. -/
def WindowCouplingTarget (V : Vocab d α) (k : Verity.Registration.Composite) (K : WindowConstructions V)
    (W : Window) (θ : ByteArray) (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      FlockSoundness.Game.prob (fun o => ((K.live W θ F c σ).joint (V.cost θ F c σ)).Accepts o)
          ((K.live W θ F c σ).joint (V.cost θ F c σ)).game (K.project W θ F c σ) +
        (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.recordFinder W θ F c σ).Pr Verity.sha512Msg)) +
          ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr Verity.sha512Msg)) + W.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))

/-- **The deployment's windows `S` cover the sampled verdict's one-stage form at composite `k`** (`Covers`, at
windows): where `S` holds no window and the parameters name the one-stage form, each strategy that meets the further
finders' premises makes the sampled verdict accept with probability at most the published `δ` at its calls. -/
def WindowCovers (V : Vocab d α) (k : Verity.Registration.Composite)
    (S : ByteArray → Array ByteArray → Option Window) : Prop :=
  ∀ θ F, S θ F = none → (∃ p, ParamsAt θ p) → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ →
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤ V.delta θ .sampled (V.cost θ F c σ)

/-- **Soundness's vocabulary at `.sampled` reads the constructions `K` at the deployment's windows `S`** (`Deployed`,
at windows): the finder is `K`'s read finder and the forger `K`'s, with no loss term; `error`'s condition holds the
leg's per-finder premises (`WindowFinderCR`); the published `δ` is at least the window's, at the window widened by the
projection's coins; and **the claim's transfer** (`claim`): where every member's roots fix their domains
(`Window.RootsFix`), at a strategy whose extracted transcript fails the claim at the published share, more than the
window's `ε` share of its population's proof blocks is incorrect at the projected strategy
(`JointAuditV5.wrongSet`: every member's, in the committed trace fixed at the joint registration). -/
structure WindowDeployed (V : Vocab d α) (K : WindowConstructions V)
    (S : ByteArray → Array ByteArray → Option Window) : Prop where
  finder : ∀ θ F c σ, V.finder .sampled θ F c σ = K.readFinder θ F c σ
  forger : ∀ θ F c σ, V.forger .sampled θ F c σ = K.forger θ F c σ
  loss : ∀ θ x, V.loss .sampled θ x = 0
  claim : ∀ θ F W, S θ F = some W → W.RootsFix → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    ¬ Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε →
      (K.live W θ F c σ).ε * (K.live W θ F c σ).nJ <
        ((((K.live W θ F c σ).joint (V.cost θ F c σ)).wrongSet (K.project W θ F c σ)).card : ℝ)
  finderCR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ → WindowFinderCR V K W θ F c σ
  delta : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)),
    (K.live W θ F c σ).delta (V.cost θ F c σ) (V.crBound θ .sampled (V.cost θ F c σ)) ≤
      V.delta θ .sampled (V.cost θ F c σ)

variable (V : Vocab d α) (k : Verity.Registration.Composite)

/-- **SPOT's leg over a window of several members**: soundness's vocabulary at `.sampled` reading the constructions
`K` at the deployment's windows `S` (`WindowDeployed`), which cover the one-stage form (`WindowCovers`), meets SPOT's
target at composite `k`, from the coupling at each window (`WindowCouplingTarget`). Where `error`'s premises hold and
the claim fails, the coupling bounds the verdict by the window's audit's acceptance at the projected strategy and the
finders' terms; `Window.ks` (`JointShareV5`, under the roots guard) bounds the first by the sample's miss, the
members' terms summed and their roots terms, each roots finder's premise its own term, the strict birthday bound
each two-continuation finder, `FinderCR` the read finder and `LmsUnforgeable` the forger. -/
theorem spotSound_of_window {K : WindowConstructions V} {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V K S) (hCov : WindowCovers V k S)
    (hC : ∀ θ F W, S θ F = some W → WindowCouplingTarget V k K W θ F) : V.SpotSound k := by
  intro θ F c σ
  by_cases hP : ∃ p, ParamsAt θ p
  swap
  · exact recursiveForm_refused V k θ hP F c σ
  unfold Vocab.error Accounting.condError
  beta_reduce
  split_ifs with hE
  swap
  · exact le_top
  obtain ⟨hfs, hpr, hFC, hUf⟩ := hE
  by_cases hI : Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε
  · calc Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
          ≤ Verity.Game.prob (fun _ => False) (V.game θ F c) σ :=
            Verity.Game.prob_mono (fun o ho => ho.2 hI) _ σ
      _ = 0 := Verity.Game.prob_false _ σ
      _ ≤ _ := zero_le
  have hver : Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ :=
    Verity.Game.prob_mono (fun o ho => ho.1) _ σ
  cases hS : S θ F with
  | none => exact hver.trans ((hCov θ F hS hP c σ hfs).trans (le_add_right (le_add_right le_rfl)))
  | some W =>
    have hFin := hD.finderCR θ F W hS c σ hfs hpr
    have hks := (K.live W θ F c σ).ks (V.cost θ F c σ) (K.project W θ F c σ) hFin.session
      ((W.rootsCR_extend _).2 hFin.roots) (fun hR => hD.claim θ F W hS hR c σ hI)
    have hdel := hD.delta θ F W hS c σ
    unfold Window.delta at hdel
    rw [Window.extend_rounds, Window.extend_qM, Window.extend_hidingTerm] at hdel
    set t := V.cost θ F c σ
    have hmsg : (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr Verity.sha512Msg))) ≤
        W.rounds * Live.twoRun (W.qM t) := by
      calc (∑ j, ENNReal.ofReal (Real.sqrt ((K.msgFinder W θ F c σ j).Pr Verity.sha512Msg)))
          ≤ ∑ _j : Fin W.rounds, Live.twoRun (W.qM t) :=
            Finset.sum_le_sum fun j _ => sqrt_le_twoRun (hFin.msg j) (K.msg_calls W θ F c σ j)
        _ = W.rounds * Live.twoRun (W.qM t) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    have hrec : ENNReal.ofReal (Real.sqrt ((K.recordFinder W θ F c σ).Pr Verity.sha512Msg)) ≤
        Live.twoRun (W.qM t) :=
      sqrt_le_twoRun hFin.record (K.record_calls W θ F c σ)
    have hbind : ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr Verity.sha512Msg)) ≤ Live.twoRun (W.qM t) :=
      sqrt_le_twoRun hFin.bind (K.bind_calls W θ F c σ)
    have hread : ENNReal.ofReal ((K.readFinder θ F c σ).Pr Verity.sha512Msg) ≤
        ENNReal.ofReal (V.crBound θ .sampled t) := by
      rw [← hD.finder θ F c σ]; exact ENNReal.ofReal_le_ofReal (V.finderPr_le_crBound hFC)
    have hforge : ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify) ≤
        ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ)) := by
      rw [← hD.forger θ F c σ]
      exact ENNReal.ofReal_le_ofReal
        (hUf (fun x => V.forger_calls .sampled θ F c σ x) (fun x => V.forger_signed .sampled θ F c σ x))
    simp only [hD.loss, add_zero]
    refine hver.trans ((hC θ F W hS c σ).trans ?_)
    calc _ ≤ (K.live W θ F c σ).sessionDelta t +
            (W.rounds * Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + Live.twoRun (W.qM t) + W.hidingTerm +
              ENNReal.ofReal (V.crBound θ .sampled t) +
              ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ))) :=
          add_le_add hks
            (add_le_add (add_le_add (add_le_add (add_le_add (add_le_add hmsg hrec) hbind) le_rfl) hread) hforge)
      _ = (K.live W θ F c σ).sessionDelta t + W.rounds * Live.twoRun (W.qM t) + Live.twoRun (W.qM t) +
            Live.twoRun (W.qM t) + W.hidingTerm + ENNReal.ofReal (V.crBound θ .sampled t) +
            ENNReal.ofReal (V.epsForge θ (V.callsMul θ .sampled * t + V.callsAdd θ .sampled) (V.signatures θ)) := by
          ring
      _ ≤ _ := add_le_add hdel le_rfl

end Verity.Sampled
