module

public import Definitions.Flock.Game.Tape
public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.EndToEnd.Period.Run
public import Proofs.Flock.Soundness.Finders

/-!
# One window's sample over several members, each verified in its own session (trusted)

`SampledKSLiveV5` when one window's sample covers several members, each a hidden-output `--zk --public-inputs` session of
its own statement at one table (`SampledKSLiveV5`'s binders: its circuit file's descriptor, public input ports and public
file, and the verifier's own recomputation `own`), as one-stage v2 verifies them. The window's audit (`JointAuditV5`)
registers every member, samples one sample over all their proof blocks on live coins with the deployed sample
(`Bridge.execSel`, `Spot.Draw.drawUnits` on one stream) under the registration's law (`Bridge.LawOn`: subset,
stratified or work, a stratum free to span members), each member's proof blocks `MemberV5.emb` into the population, and
then runs one flock-verify session per member in the registration's order, member `m` proving its part of the sample
(`MemberV5.part`, sampled by `subset |part|`). A member's sampled count is random and may be zero. The run accepts when the
session of every member with a proof block in the sample accepts (`JointAuditV5.Accepts`).

A prover may read every coin it has seen, so member `m`'s session is played with a strategy that read the sample and the
earlier members' sessions. `JointAuditV5.memberLaw` and `JointAuditV5.memberStrat` are that session played alone: its
sampling design samples the sample and the earlier sessions' coins together (`seqTape`), samples member `m`'s part, and
plays the strategy `τ` leaves for member `m`'s session there (`seqStrat`). Member `m`'s incorrect proof blocks are
`wrongRegZc` at V5's canonical openings in the committed trace fixed at the joint registration (`XplurZC`), at that
play; the window's are their images in the population (`JointAuditV5.wrongSet`), fixed by `τ` before the sample.

* `JointSampledV5`, per window, `EndToEndHiddenV5`'s form: a set `W` of the population's proof blocks is incorrect in an
  accepted run with probability at most the sampling design's own miss of `W` (`P_a(W)` of `oneStageOf law` on the strata `ns`) plus
  `Σ_m (ε_ks,m + ε_bind,m)`.
* `JointShareV5`, per window, the share form: more than an `ε` share of the population incorrect in an accepted run, at
  most the most the sampling design misses of any such set plus the same terms.
* `JointHorizonV5`, pooled over a monitoring period of `T` such windows played by one adaptive prover: except with
  probability `δ + Σ_t Σ_m (ε_ks,t,m + ε_bind,t,m)`, if every window is accepted, their samples miss their incorrect proof
  blocks together with probability at least `δ` (the product of their `P_a`s), and their ANDs and output bits are each at
  most the most any choice in that confidence set holds.

`ε_ks,m` is member `m`'s knowledge term at the worst registration and `ε_bind,m` its binding term `δ_link`, each at the
sampling design that samples member `m`'s part of the sample (`JointAuditV5.design`), as `SampledKSLiveV5` states them. A
member with no proof block in the sample contributes nothing to `ε_bind,m` at that sample and the empty sample's knowledge term
to `ε_ks,m`. Each member's session has one link-point count at every sample (`JointAuditV5.hcls`), which the registration
check makes so by refusing a member whose fewest and most sampled proof blocks have different counts. The named assumptions
are A3 (`uniform/io-getrandombytes`) for the sample's bytes, `live-verifier` at each member (`MemberV5.hCust`,
`RecordCustodyZKJPLaw` at `subset |S|` for its part `S`), and `cr/sha-512` (`Assumptions.TableCRZAt`) and `ecr/sha-512`
(`Assumptions.LinkCRZCAt`) at the strategy each member's session is played with alone (in a period, at every window's
strategy on some run, `JointPeriodV5.Plays`), stated flat. Their proofs are in `Joint/V5/Bound.lean`.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **A member of a window's sample**: one hidden-output `--zk --public-inputs` session at one table over its `n` proof
blocks, with every binder of `SampledKSLiveV5` but the prover, its sampling design and what `RelS` reads. Its statement
(circuit descriptor, public input ports, public file, `own`); where its proof blocks sit in the window's population
(`emb`); its record (sample files, session writer, reference setup at table 0, at the link points of `kd` proof blocks); the
verify command's tags; `live-verifier` at `own`, each part `S` sampled by `subset |S|` (`hCust`); the count curve's
parameters; and the knowledge finders' budgets `qF`, `qS`. Its registration is the window's joint one, `Reg`. -/
structure MemberV5 (Reg : Type) (nJ : ℕ) where
  n : ℕ
  I : Inputs
  own : Option ByteArray
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pi : HmIn.Ports
  hpi : HmIn.parse D.unroll = .ok pi
  pub : Public
  hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  hcount : I.count ≤ 1
  emb : Fin n → Fin nJ
  kd : ℕ
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
    fun S => .subset S.card
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  qF : ℝ
  qS : ℝ

namespace MemberV5

open Classical

variable {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ)

/-- **The member's part of a sample** `S` of the window's population: its proof blocks that `S` holds. -/
def part (S : Finset (Fin nJ)) : Finset (Fin m.n) := Finset.univ.filter fun u => m.emb u ∈ S

/-- **One link-point count at every sample of `L`**: each sample's proof blocks have the session's link points, those of
`kd` proof blocks. -/
def OneClass (L : Law m.n) : Prop := ∀ ω, mPtsOf m.D.unroll.g (L.draw ω).card = mPtsOf m.D.unroll.g (m.kd / nTab m.I)

/-- The member's flock-verify session on its part `S` at the registration `R`. -/
noncomputable def live (S : Finset (Fin m.n)) (R : Reg) : Game Discharge.Exec.Transcript :=
  liveOfZJ m.dg (planZHJP m.dj m.y₀) m.N m.wr S R

/-- **The member's session played alone** at the sampling design `L`: `SampledKSLiveV5`'s game at `L`. -/
noncomputable def game (L : Law m.n) : Game (Reg × Finset (Fin m.n) × Discharge.Exec.Transcript) :=
  auditLive L Reg m.live

/-- The member's program's partition into proof blocks. -/
noncomputable def P := (keyProgP m.D.unroll m.pi m.hU m.n).partition (Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi)

/-- The member's descriptor read as a statement (`Desc.parseStmt`), from its flat parse. -/
theorem hDS : Flock.Desc.parseStmt (zkTags m.I) m.I.circuitFile m.I.tables = .ok m.D :=
  Flock.Desc.parseStmt_of_parse m.hD

/-- The member's circuit file read as its statement circuit (`HmOut.parseStmt`), from its descriptor's flat parse. -/
theorem hcS : HmOut.parseStmt (zkTags m.I) m.I.circuitFile m.I.tables = .ok m.D.unroll :=
  Flock.Desc.parseStmt_stmt m.hDS

/-- The member's public file's commitments. -/
noncomputable def vb :=
  (rsZHJP (Reg := Reg) m.dj m.y₀ m.ht m.hcS m.hscope m.hU m.hpi m.hpub).bindingZ
    (rsZHJP_computes m.dj m.y₀ m.ht m.hcS m.hscope m.hU m.hpi m.hpub)

/-- **V5's canonical openings** at the member's public file's leaf map and its constants at public inputs
(`Bridge.constsOfP`), as `SampledKSLiveV5` reads them. -/
noncomputable def canon : Fin ((keyProgP m.D.unroll m.pi m.hU m.n).circuit
    (Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi)).N → m.vb.Opening → Prop :=
  fun g o => Bridge.canonAt
    (fun g => (Discharge.HiddenExec.rowOfHP m.D.unroll m.hU m.n m.pi (Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi)
      g).map fun r => ⟨r.2.1.val, (Bridge.LeafMap.ofPublic m.D.unroll m.pub m.n).ref r.2.1.val r.1.val,
        Discharge.Layout.portRowW m.D.unroll r.2.1, r.2.2.val⟩)
    (fun g => Bridge.constsOfP m.D.unroll m.hU m.n m.pi m.pub g) g (m.vb.bit (m.vb.val o) g) (m.vb.val o) o.2.2

/-- The model audit's strategy that a live strategy `τ` of the member's session at `L` refines to: `SampledKSLiveV5`'s
`σ`, with the sample at each accepted outcome from custody at `own` at one table (`decodesZCJH_custodyAt`). -/
noncomputable def sim (L : Law m.n) (hL : m.OneClass L) (τ : Strategy (m.game L)) :=
  simZJA execArith hs512 (planZHJP m.dj m.y₀) (AcceptsZKP m.I m.own) m.wr (opnAtZHJP (planZHJP m.dj m.y₀) m.I m.own m.dj)
    m.dg (decAtZJ (planZHJP m.dj m.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) m.N
    (decodesZCJH_custodyAt m.I m.own m.ht m.hcS m.hcount _ m.dj m.y₀ m.wr L m.hTags m.hCust hL)
    τ

/-- **The member's incorrect proof blocks** at `τ`: `wrongRegZc` at V5's canonical openings in the committed trace fixed
at registration (`XplurZC`), as in `SampledKSLiveV5`. -/
noncomputable def wrongSet (L : Law m.n) (hL : m.OneClass L) (τ : Strategy (m.game L)) : Finset (Fin m.n) :=
  wrongRegZc execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb m.canon (reg (m.sim L hL τ))
    (XplurZC execArith hs512 m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb (reg (m.sim L hL τ))
      (cont (m.sim L hL τ)) m.k m.Rw)

/-- The knowledge term `ε_ks` at `L`: `ksAvgStrictZ` at the worst registration. -/
noncomputable def epsKs (L : Law m.n) : ℝ≥0∞ := ⨆ R : Reg, ksAvgStrictZ (planZHJP m.dj m.y₀) L m.k m.qF m.qS R

/-- The binding term `ε_bind` at `L`, `δ_link = Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`, as in
`SampledKSLiveV5`. -/
noncomputable def link (L : Law m.n) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ p : m.vb.Pos,
      (Fintype.card (ΩcZ execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb L p) : ℝ) / Fintype.card L.Ω) *
    (2 * (m.t' * (1 + m.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * m.M) + m.k / (Real.exp 1 * m.Rw)) / (1 - m.ρ))

end MemberV5

/-- **A window's audit of several members**: every member registered at one registration `Reg`, in the registration's
order; one live sample over the population of all their `nJ` proof blocks, sampled by the deployed sample on the verifier's
one byte stream `src` under the registration's law `law` on strata of sizes `ns`, sampling `kd` proof blocks; then one
session per member, in order. Each member's part of every sample has its session's link points (`hcls`). -/
structure JointAuditV5 where
  Reg : Type
  nJ : ℕ
  ms : List (MemberV5 Reg nJ)
  law : Flock.Draw.Law
  ns : List ℕ
  kd : ℕ
  hns : ns.sum = nJ
  hlaw : Bridge.LawOn law ns kd
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hcls : ∀ (k : Fin ms.length) (ω : Ω),
    mPtsOf (ms.get k).D.unroll.g ((ms.get k).part ((Bridge.execSel (n := nJ) src law nJ).draw ω)).card =
      mPtsOf (ms.get k).D.unroll.g ((ms.get k).kd / nTab (ms.get k).I)

attribute [instance] JointAuditV5.fintypeΩ JointAuditV5.nonemptyΩ

namespace JointAuditV5

open Classical

variable (w : JointAuditV5)

/-- The window's sample: the deployed sample under the registration's law on the verifier's bytes, over every member's
proof blocks. -/
def L : Law w.nJ := Bridge.execSel (n := w.nJ) w.src w.law w.nJ

/-- **The members' sessions on the sample `S`** at the registration `R`, one after another. -/
noncomputable def sessions (S : Finset (Fin w.nJ)) (R : w.Reg) : Game (List Discharge.Exec.Transcript) :=
  seqG (fun m : MemberV5 w.Reg w.nJ => m.live (m.part S) R) w.ms

/-- **The window's audit as it runs**: the prover registers every member, the verifier samples the sample on live coins,
and each member's session proves its part. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript) :=
  .send w.Reg fun R => .coin w.L.Ω fun ω => (w.sessions (w.L.draw ω) R).map fun ts => (R, w.L.draw ω, ts)

/-- **The run accepts**: flock-verify `--zk --public-inputs` accepts, at its `own`, the session of every member with a
proof block in the sample. A member with none has no statement to verify, so nothing reads its session's verdict. -/
def Accepts (o : w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript) : Prop :=
  List.Forall₂ (fun m t => (m.part o.2.1).Nonempty → AcceptsZKP m.I m.own t) w.ms o.2.2

/-- **The coins of the sessions before member `k`'s**, at the strategy `τ` and the sample's coins `ω`. -/
def before (τ : Strategy w.game) (k : Fin w.ms.length) (ω : w.L.Ω) : Type :=
  seqTape (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw ω)) τ.1) w.ms k.1 k.2
    (Strategy.ofMap (fun ts => (τ.1, w.L.draw ω, ts)) _ (τ.2 ω))

noncomputable instance instFintypeBefore (τ : Strategy w.game) (k : Fin w.ms.length) (ω : w.L.Ω) :
    Fintype (w.before τ k ω) :=
  seqTape.fintype _ _ _ _ _

instance instNonemptyBefore (τ : Strategy w.game) (k : Fin w.ms.length) (ω : w.L.Ω) : Nonempty (w.before τ k ω) :=
  seqTape.nonempty _ _ _ _ _

/-- **Member `k`'s session's sampling design, played alone at `τ`**: the sample's coins and, for each, the coins of the
sessions before member `k`'s, sampled together; it samples member `k`'s part of the sample. -/
noncomputable def memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) : Law (w.ms.get k).n where
  Ω := w.L.Ω × ((ω : w.L.Ω) → w.before τ k ω)
  draw x := (w.ms.get k).part (w.L.draw x.1)

/-- **Member `k`'s session played alone at `τ`**: `τ`'s registration, and on the sample's coins and the earlier sessions'
coins the strategy `τ` leaves for member `k`'s session there. -/
noncomputable def memberStrat (τ : Strategy w.game) (k : Fin w.ms.length) :
    Strategy ((w.ms.get k).game (w.memberLaw τ k)) :=
  ⟨τ.1, fun x => Strategy.map (fun t => (τ.1, (w.ms.get k).part (w.L.draw x.1), t)) _
    (seqStrat (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw x.1)) τ.1) w.ms k.1 k.2
      (Strategy.ofMap (fun ts => (τ.1, w.L.draw x.1, ts)) _ (τ.2 x.1)) (x.2 x.1))⟩

/-- Member `k`'s part of every sample has its session's link points, played alone at `τ`. -/
theorem oneClass_memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) : (w.ms.get k).OneClass (w.memberLaw τ k) :=
  fun x => w.hcls k x.1

/-- The model audit's strategy member `k`'s session refines to, played alone at `τ`. -/
noncomputable def memberSim (τ : Strategy w.game) (k : Fin w.ms.length) :=
  (w.ms.get k).sim (w.memberLaw τ k) (w.oneClass_memberLaw τ k) (w.memberStrat τ k)

/-- **Member `k`'s incorrect proof blocks** at `τ`, in the committed trace fixed at the joint registration. -/
noncomputable def memberWrong (τ : Strategy w.game) (k : Fin w.ms.length) : Finset (Fin (w.ms.get k).n) :=
  (w.ms.get k).wrongSet (w.memberLaw τ k) (w.oneClass_memberLaw τ k) (w.memberStrat τ k)

/-- **The window's incorrect proof blocks** at `τ`: every member's, in the population. -/
noncomputable def wrongSet (τ : Strategy w.game) : Finset (Fin w.nJ) :=
  Finset.univ.biUnion fun k => (w.memberWrong τ k).image (w.ms.get k).emb

/-- **The sampling design of member `k`'s part of the sample**: the sample's coins, member `k`'s part. -/
noncomputable def design (k : Fin w.ms.length) : Law (w.ms.get k).n where
  Ω := w.L.Ω
  draw ω := (w.ms.get k).part (w.L.draw ω)

/-- The sampling design's own miss of a set `B` of the population's proof blocks, `P_a(B)` of `oneStageOf law` on the strata
`ns`. -/
noncomputable def pa (B : Finset (Fin w.nJ)) : ℝ :=
  (Properties.Spot.oneStageOf w.law).Pa w.ns (B.map Fin.valEmbedding)

/-- The members' knowledge and binding terms, `Σ_m (ε_ks,m + ε_bind,m)`, each at its part's sampling design. -/
noncomputable def eps : ℝ≥0∞ := ∑ k, ((w.ms.get k).epsKs (w.design k) + (w.ms.get k).link (w.design k))

end JointAuditV5

/-- **A window's sample over several members, `EndToEndHiddenV5`'s form.** For a window's audit as in `JointAuditV5`,
whose one live sample under the registration's law covers every member's proof blocks and whose members' sessions each
prove their part of it, every live prover `τ` and every set `W` of the population's proof blocks, the probability that
flock-verify accepts the session of every member with a proof block in the sample and every proof block of `W` is
incorrect is at most
$$P_a(W) + \sum_m (\varepsilon_{ks,m} + \varepsilon_{bind,m}),$$
`P_a(W)` the sampling design's own miss of `W` on its strata. Its named assumptions: A3 (`Assumptions.UniformRandomBytes`) for the
sample's bytes, `live-verifier` at each member (`MemberV5.hCust`, `RecordCustodyZKJPLaw`), and `cr/sha-512`
(`Assumptions.TableCRZAt`) and `ecr/sha-512` (`Assumptions.LinkCRZCAt`) at each member's session played alone at `τ`. -/
def JointSampledV5 : Prop :=
  ∀ (w : JointAuditV5), Assumptions.UniformRandomBytes w.Lb w.src →
    ∀ τ : Strategy w.game,
    (∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
      (w.ms.get k).qF (w.ms.get k).qS) →
    (∀ k, Assumptions.LinkCRZCAt execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
      (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') →
    ∀ W : Finset (Fin w.nJ),
      prob (fun o => w.Accepts o ∧ W ⊆ w.wrongSet τ) w.game τ ≤ ENNReal.ofReal (w.pa W) + w.eps

/-- **A window's sample over several members, the share form.** For a window's audit as in `JointAuditV5`, every live
prover `τ` and every share `ε`, the probability that flock-verify accepts the session of every member with a proof block
in the sample and more than `ε·nJ` of the population's proof blocks are incorrect is at most the most the sampling design misses of any
set of more than `ε·nJ` of them plus `Σ_m (ε_ks,m + ε_bind,m)`, under `JointSampledV5`'s named assumptions. -/
def JointShareV5 : Prop :=
  ∀ (w : JointAuditV5), Assumptions.UniformRandomBytes w.Lb w.src →
    ∀ τ : Strategy w.game,
    (∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
      (w.ms.get k).qF (w.ms.get k).qS) →
    (∀ k, Assumptions.LinkCRZCAt execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
      (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') →
    ∀ ε : ℝ,
      prob (fun o => w.Accepts o ∧ ε * w.nJ < (w.wrongSet τ).card) w.game τ ≤
        ENNReal.ofReal (⨆ B : {B : Finset (Fin w.nJ) // ε * w.nJ < B.card}, w.pa B.1) + w.eps

/-- What a monitoring period's run records of window `idx`: whether its run accepted (`JointAuditV5.Accepts`), its
incorrect proof blocks (`JointAuditV5.wrongSet`), and whether a sampled proof block is one of them. -/
structure AuditRecV5 (W : ℕ → JointAuditV5) where
  idx : ℕ
  accepted : Prop
  wrong : Finset (Fin (W idx).nJ)
  drawnWrong : Prop

namespace AuditRecV5

variable {W : ℕ → JointAuditV5}

/-- The record's window's sampling design's own miss of its incorrect proof blocks. -/
noncomputable def pa (r : AuditRecV5 W) : ℝ≥0∞ := ENNReal.ofReal ((W r.idx).pa r.wrong)

/-- The record's incorrect proof blocks' total weight `Σ_{u ∈ B} h u`, with `h t` window `t`'s per-proof-block
weights. -/
def harm (h : (t : ℕ) → Fin (W t).nJ → ℕ) (r : AuditRecV5 W) : ℕ := ∑ u ∈ r.wrong, h r.idx u

end AuditRecV5

namespace JointAuditV5

/-- Window `t`'s record, at the strategy `τ` that played it and its output `o`. -/
noncomputable def record (W : ℕ → JointAuditV5) (t : ℕ) (τ : Strategy (W t).game)
    (o : (W t).Reg × Finset (Fin (W t).nJ) × List Discharge.Exec.Transcript) : AuditRecV5 W :=
  ⟨t, (W t).Accepts o, (W t).wrongSet τ, ∃ u ∈ o.2.1, u ∈ (W t).wrongSet τ⟩

end JointAuditV5

namespace JointPeriodV5

/-- **A monitoring period of `T` windows**, window `t`'s audit after window `t − 1`'s, against one prover. -/
noncomputable def game (W : ℕ → JointAuditV5) (T : ℕ) : Game Unit := periodG (fun t => (W t).game) 0 T

/-- The period's run at the prover strategy `s`: each window's record, in order. -/
noncomputable def run (W : ℕ → JointAuditV5) (T : ℕ) (s : Strategy (game W T)) : Game (List (AuditRecV5 W)) :=
  periodRun (fun t => (W t).game) (JointAuditV5.record W) 0 T s

/-- **The strategies the period plays window `t` with**: `τ` plays window `t` on some run of the period at `s`. -/
def Plays (W : ℕ → JointAuditV5) (T : ℕ) (s : Strategy (game W T)) (t : ℕ) (τ : Strategy (W t).game) : Prop :=
  PeriodPlays (fun t => (W t).game) 0 T s t τ

/-- **The most harm in the period's confidence set**: the largest total weight over every choice of a set of proof
blocks in each of the run's windows whose laws' own misses multiply to at least `δ`. -/
noncomputable def maxHarm (W : ℕ → JointAuditV5) (h : (t : ℕ) → Fin (W t).nJ → ℕ) (δ : ℝ)
    (l : List (AuditRecV5 W)) : ℝ :=
  ⨆ B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).nJ) //
      ENNReal.ofReal δ ≤ ∏ i, ENNReal.ofReal ((W l[i].idx).pa (B i))},
    ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)

/-- **The run's incorrect proof blocks are in the period's confidence set at `δ`**: the windows' laws miss them together
with probability at least `δ`, and their ANDs and their output bits in all are each at most the most any choice in that
set holds. -/
def Within (W : ℕ → JointAuditV5) (δ : ℝ) (ands bits : (t : ℕ) → Fin (W t).nJ → ℕ) (l : List (AuditRecV5 W)) :
    Prop :=
  ENNReal.ofReal δ ≤ (l.map AuditRecV5.pa).prod ∧
    (((l.map (AuditRecV5.harm ands)).sum : ℕ) : ℝ) ≤ maxHarm W ands δ l ∧
    (((l.map (AuditRecV5.harm bits)).sum : ℕ) : ℝ) ≤ maxHarm W bits δ l

end JointPeriodV5

/-- **A monitoring period of windows of several members, pooled.** In a monitoring period of `T` windows `W t` as in
`JointAuditV5`, each with a fresh live sample over its members' proof blocks, played by one adaptive prover `s`, for every
`δ` and every per-proof-block weights `ands t` (each proof block's ANDs) and `bits t` (its output bits), the
probability that every window's run accepts (`JointAuditV5.Accepts`) and their incorrect proof blocks are not in the
period's confidence set is at most
$$\delta + \sum_t \sum_m (\varepsilon_{ks,t,m} + \varepsilon_{bind,t,m}):$$
outside it, when every window is accepted, the product of the windows' laws' own misses of their incorrect proof blocks is
at least `δ`, and their ANDs and output bits in all are at most the period's `maxHarm`. The sampling term is `δ` however
many windows and members the period has. Its named assumptions, at every window: A3 (`Assumptions.UniformRandomBytes`)
for its sample's bytes, `live-verifier` at each member (`MemberV5.hCust`, `RecordCustodyZKJPLaw`), and `cr/sha-512`
(`Assumptions.TableCRZAt`) and `ecr/sha-512` (`Assumptions.LinkCRZCAt`) at each member's session played alone at every
strategy the window is played with on some run (`JointPeriodV5.Plays`). -/
def JointHorizonV5 : Prop :=
  ∀ (W : ℕ → JointAuditV5) (T : ℕ) (δ : ℝ) (ands bits : (t : ℕ) → Fin (W t).nJ → ℕ),
    (∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ s : Strategy (JointPeriodV5.game W T),
    (∀ t (τ : Strategy (W t).game), JointPeriodV5.Plays W T s t τ → ∀ k ω j,
      Assumptions.TableCRZAt execArith hs512
        (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀ (((W t).memberLaw τ k).draw ω) (reg ((W t).memberSim τ k))) j
        (zcStrat execArith hs512 (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) (cont ((W t).memberSim τ k) ω))
        ((W t).ms.get k).qF ((W t).ms.get k).qS) →
    (∀ t (τ : Strategy (W t).game), JointPeriodV5.Plays W T s t τ → ∀ k,
      Assumptions.LinkCRZCAt execArith hs512 ((W t).ms.get k).P (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
        (tabZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) ((W t).ms.get k).vb (reg ((W t).memberSim τ k))
        (cont ((W t).memberSim τ k)) ((W t).ms.get k).k ((W t).ms.get k).Rw ((W t).ms.get k).M ((W t).ms.get k).t') →
    value (fun l => (∀ r ∈ l, r.accepted) ∧ ¬ JointPeriodV5.Within W δ ands bits l) (JointPeriodV5.run W T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, (W t).eps

end Flock.Guarantees
