import Proofs.Core.Game.Tape
import Proofs.Flock.Soundness.Assumptions.Finders
import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.EndToEnd.Joint.MT.Statement

/-!
# The joint stratified bound over the members of one sample, each verified in its own session (trusted)

`JointStratifiedMT` (`Joint/MT/Statement.lean`) when one window's sample covers several members, each a hidden-output
`--zk --public-inputs` session of its own statement (`SampledKSLiveV5`'s binders: its circuit file's descriptor, public
input ports and public file, and the verifier's own recomputation `own`), as one-stage v2 verifies them. The window's
audit (`JointAuditV5`) registers every member, draws one sample over all their proof blocks on live coins
(`Flock.Draw.drawOS` at `.stratified`, `Law.execOS`, its population the members' proof blocks, each member's proof
blocks `MemberV5.emb` into it and its strata within one member), and then runs one flock-verify session per member in
order, member `m` proving its part of the sample (`MemberV5.part`). The run accepts when every session accepts
(`JointAuditV5.Accepts`).

A prover may read every coin it has seen, so member `m`'s session is played with a strategy that read the sample and the
earlier members' sessions. `JointAuditV5.memberLaw` and `JointAuditV5.memberStrat` are that session played alone: its
sampling design draws the sample and the earlier sessions' coins together (`Verity.Game.seqTape`), samples member `m`'s
part, and plays the strategy `τ` leaves for member `m`'s session there (`Verity.Game.seqStrat`). Member `m`'s incorrect
proof blocks are `wrongRegZc` at V5's canonical openings in the committed trace fixed at the joint registration
(`XplurZC`), at that play; the window's are their images in the population (`JointAuditV5.wrongSet`).

* `JointStratifiedV5`, per window: except with probability `δ + Σ_m (ε_ks,m + ε_bind,m)`, if every session accepts, the
  window's incorrect proof blocks `B` lie in the confidence set (`Law.stratified`'s escape of `B` is at least `δ`), so
  they spend at most the one budget, `Σ_s k_s·x_s/n_s ≤ ln(1/δ)` over every stratum of every member, and their ANDs and
  output bits are each at most the most any set in the confidence set holds (`JointAuditV5.maxHarm`).
* `JointStratifiedHorizonV5`, pooled: the same over a monitoring period of `T` such windows played by one adaptive
  prover, with one budget for the whole period: except with probability `δ + Σ_t Σ_m (ε_ks,t,m + ε_bind,t,m)`.

`ε_ks,m` is member `m`'s knowledge term at the worst registration and `ε_bind,m` its binding term `δ_link`, each at the
sampling design that draws member `m`'s part of the sample (`JointAuditV5.design`), as `SampledKSLiveV5` states them.
The named assumptions are A3 (`uniform/io-getrandombytes`) for the sample's bytes, `live-verifier` at each member
(`MemberV5.hCust`, `RecordCustodyZKJP`: the member's record's draw file is `subset kd_m` over its own proof blocks,
carrying its part), and `cr/sha-512` (`Assumptions.TableCRZAt`) and `ecr/sha-512` (`Assumptions.LinkCRZCAt`) at the
strategy each member's session is played with alone (in a period, at every window's strategy on some run,
`JointPeriodV5.Plays`), stated flat. Their proofs are `Flock.SecurityProofs.JointStratifiedV5` and
`Flock.SecurityProofs.JointStratifiedHorizonV5` (`Joint/V5/Bound.lean`).
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open Verity.Game (seqG seqTape seqStrat)
open scoped ENNReal

/-- **A member of a window's sample**: one hidden-output `--zk --public-inputs` session of `nTab I` tables over its `n`
proof blocks, with every binder of `SampledKSLiveV5` but the prover, its sampling design and what `RelS` reads. Its
statement (circuit descriptor, public input ports, public file, `own`); where its proof blocks sit in the window's
population (`emb`); its record (draw files, session writer, reference setup at table 0, `kd / nTab I` drawn proof blocks
a table); the verify command's tags; `live-verifier` at `own` (`hCust`); the count curve's parameters; and the knowledge
finders' budgets `qF`, `qS`. Its registration is the window's joint one, `Reg`. -/
structure MemberV5 (Reg : Type) (nJ : ℕ) where
  n : ℕ
  I : Inputs
  own : Option ByteArray
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pi : HmIn.Ports
  hpi : HmIn.parse D.unroll = .ok pi
  pub : Public
  hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  emb : Fin n → Fin nJ
  kd : ℕ
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr kd
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

/-- The member's flock-verify session on its part `S` at the registration `R`. -/
noncomputable def live (S : Finset (Fin m.n)) (R : Reg) : Game Discharge.Exec.Transcript :=
  liveOfZJ m.dg (planZHJP m.dj m.y₀) m.N m.wr S R

/-- **The member's session played alone** at the sampling design `L`: `SampledKSLiveV5`'s game at `L`. -/
noncomputable def game (L : Law m.n) : Game (Reg × Finset (Fin m.n) × Discharge.Exec.Transcript) :=
  auditLive L Reg m.live

/-- The member's program's partition into proof blocks. -/
noncomputable def P := (keyProgP m.D.unroll m.pi m.hU m.n).partition (Discharge.Layout.outsOfP m.D.unroll m.hU m.n m.pi)

/-- The member's public file's commitments. -/
noncomputable def vb :=
  (rsZHJP (Reg := Reg) m.dj m.y₀ m.ht (Flock.Desc.parseStmt_stmt m.hD) m.hscope m.hU m.hpi m.hpub).bindingZ
    (rsZHJP_computes m.dj m.y₀ m.ht (Flock.Desc.parseStmt_stmt m.hD) m.hscope m.hU m.hpi m.hpub)

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
`σ`, with the draw at each accepted outcome from custody at `own` (`decodesZCJH_custodyP`). -/
noncomputable def sim (L : Law m.n) (τ : Strategy (m.game L)) :=
  simZJA execArith hs512 (planZHJP m.dj m.y₀) (AcceptsZKP m.I m.own) m.wr (opnAtZHJP (planZHJP m.dj m.y₀) m.I m.own m.dj)
    m.dg (decAtZJ (planZHJP m.dj m.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) m.N
    (decodesZCJH_custodyP m.I m.own m.ht (Flock.Desc.parseStmt_stmt m.hD) m.kd m.hdiv m.dj m.y₀ m.wr L m.hTags m.hCust) τ

/-- **The member's incorrect proof blocks** at `τ`: `wrongRegZc` at V5's canonical openings in the committed trace fixed
at registration (`XplurZC`), as in `SampledKSLiveV5`. -/
noncomputable def wrongSet (L : Law m.n) (τ : Strategy (m.game L)) : Finset (Fin m.n) :=
  wrongRegZc execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb m.canon (reg (m.sim L τ))
    (XplurZC execArith hs512 m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb (reg (m.sim L τ)) (cont (m.sim L τ))
      m.k m.Rw)

/-- The knowledge term `ε_ks` at `L`: `ksAvgStrictZ` at the worst registration. -/
noncomputable def epsKs (L : Law m.n) : ℝ≥0∞ := ⨆ R : Reg, ksAvgStrictZ (planZHJP m.dj m.y₀) L m.k m.qF m.qS R

/-- The binding term `ε_bind` at `L`, `δ_link = Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`, as in
`SampledKSLiveV5`. -/
noncomputable def link (L : Law m.n) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ p : m.vb.Pos,
      (Fintype.card (ΩcZ execArith m.P (planZHJP m.dj m.y₀) (tabZHJP m.dj m.y₀) m.vb L p) : ℝ) / Fintype.card L.Ω) *
    (2 * (m.t' * (1 + m.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * m.M) + m.k / (Real.exp 1 * m.Rw)) / (1 - m.ρ))

end MemberV5

/-- **A window's audit of several members**: every member registered at one registration `Reg`, one live stratified
sample over the population of all their proof blocks, then one session per member, in order. The sample's strata, proof
block `u` in stratum `sub u`, `ks s` of stratum `s`'s proof blocks drawn, the executable strata `St` describing them
(`Law.ExecStrata`); and the verifier's byte sources, one per stratum. -/
structure JointAuditV5 where
  Reg : Type
  nJ : ℕ
  ms : List (MemberV5 Reg nJ)
  mS : ℕ
  sub : Fin nJ → Fin mS
  ks : Fin mS → ℕ
  hks : ∀ s, ks s ≤ (Law.stratum sub s).card
  St : Fin mS → Flock.Draw.Stratum
  hS : Law.ExecStrata sub ks St
  Kd : ℕ
  pop : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin mS → ℕ
  src : Ω → (i : Fin mS) → Fin (Lb i) → Fin 256

attribute [instance] JointAuditV5.fintypeΩ JointAuditV5.nonemptyΩ

namespace JointAuditV5

open Classical

variable (w : JointAuditV5)

/-- The window's sample: `Flock.Draw.drawOS`'s stratified draw on the verifier's byte sources, over every member's proof
blocks. -/
def L : Law w.nJ := Law.execOS (n := w.nJ) w.src (.stratified w.Kd (Array.ofFn w.St)) w.pop

/-- The stratified sampling design the sample is held to under A3 (`Law.execOS_escape_le`). -/
def ideal : Law w.nJ := Law.stratified w.sub w.ks w.hks

/-- **The members' sessions on the sample `S`** at the registration `R`, one after another. -/
noncomputable def sessions (S : Finset (Fin w.nJ)) (R : w.Reg) : Game (List Discharge.Exec.Transcript) :=
  seqG (fun m : MemberV5 w.Reg w.nJ => m.live (m.part S) R) w.ms

/-- **The window's audit as it runs**: the prover registers every member, the verifier draws the sample on live coins,
and each member's session proves its part. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript) :=
  .send w.Reg fun R => .coin w.L.Ω fun ω => (w.sessions (w.L.draw ω) R).map fun ts => (R, w.L.draw ω, ts)

/-- **The run accepts**: flock-verify `--zk --public-inputs` accepts every member's session at its `own`. -/
def Accepts (o : w.Reg × Finset (Fin w.nJ) × List Discharge.Exec.Transcript) : Prop :=
  List.Forall₂ (fun m t => AcceptsZKP m.I m.own t) w.ms o.2.2

/-- **The coins of the sessions before member `k`'s**, at the strategy `τ` and the sample's coins `ω`. -/
def before (τ : Strategy w.game) (k : Fin w.ms.length) (ω : w.L.Ω) : Type :=
  seqTape (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw ω)) τ.1) w.ms k.1 k.2
    (Strategy.ofMap (fun ts => (τ.1, w.L.draw ω, ts)) _ (τ.2 ω))

noncomputable instance (τ : Strategy w.game) (k : Fin w.ms.length) (ω : w.L.Ω) : Fintype (w.before τ k ω) :=
  Verity.Game.seqTape.fintype _ _ _ _ _

instance (τ : Strategy w.game) (k : Fin w.ms.length) (ω : w.L.Ω) : Nonempty (w.before τ k ω) :=
  Verity.Game.seqTape.nonempty _ _ _ _ _

/-- **Member `k`'s session's sampling design, played alone at `τ`**: the sample's coins and, for each, the coins of the
sessions before member `k`'s, drawn together; it samples member `k`'s part of the sample. -/
noncomputable def memberLaw (τ : Strategy w.game) (k : Fin w.ms.length) : Law (w.ms.get k).n where
  Ω := w.L.Ω × ((ω : w.L.Ω) → w.before τ k ω)
  draw x := (w.ms.get k).part (w.L.draw x.1)

/-- **Member `k`'s session played alone at `τ`**: `τ`'s registration, and on the sample's coins and the earlier sessions'
coins the strategy `τ` leaves for member `k`'s session there. -/
noncomputable def memberStrat (τ : Strategy w.game) (k : Fin w.ms.length) :
    Strategy ((w.ms.get k).game (w.memberLaw τ k)) :=
  ⟨τ.1, fun x => Verity.Game.Strategy.toMap (fun t => (τ.1, (w.ms.get k).part (w.L.draw x.1), t)) _
    (seqStrat (fun m : MemberV5 w.Reg w.nJ => m.live (m.part (w.L.draw x.1)) τ.1) w.ms k.1 k.2
      (Strategy.ofMap (fun ts => (τ.1, w.L.draw x.1, ts)) _ (τ.2 x.1)) (x.2 x.1))⟩

/-- The model audit's strategy member `k`'s session refines to, played alone at `τ`. -/
noncomputable def memberSim (τ : Strategy w.game) (k : Fin w.ms.length) :=
  (w.ms.get k).sim (w.memberLaw τ k) (w.memberStrat τ k)

/-- **Member `k`'s incorrect proof blocks** at `τ`, in the committed trace fixed at the joint registration. -/
noncomputable def memberWrong (τ : Strategy w.game) (k : Fin w.ms.length) : Finset (Fin (w.ms.get k).n) :=
  (w.ms.get k).wrongSet (w.memberLaw τ k) (w.memberStrat τ k)

/-- **The window's incorrect proof blocks** at `τ`: every member's, in the population. -/
noncomputable def wrongSet (τ : Strategy w.game) : Finset (Fin w.nJ) :=
  Finset.univ.biUnion fun k => (w.memberWrong τ k).image (w.ms.get k).emb

/-- **The sampling design of member `k`'s part of the sample**: the sample's coins, member `k`'s part. -/
noncomputable def design (k : Fin w.ms.length) : Law (w.ms.get k).n where
  Ω := w.L.Ω
  draw ω := (w.ms.get k).part (w.L.draw ω)

/-- **The detection budget a set `B` of incorrect proof blocks spends**, `Σ_s k_s·x_s/n_s` over every stratum of every
member, with `x_s` its proof blocks in stratum `s`. -/
noncomputable def spend (B : Finset (Fin w.nJ)) : ℝ :=
  ∑ s, (w.ks s : ℝ) * (B.filter (w.sub · = s)).card / (Law.stratum w.sub s).card

/-- **The most harm in the confidence set**: the largest total weight `Σ_{u ∈ B} h u` over the sets `B` the stratified
sample misses with probability at least `δ`. -/
noncomputable def maxHarm (h : Fin w.nJ → ℕ) (δ : ℝ) : ℝ :=
  ⨆ B : {B : Finset (Fin w.nJ) // ENNReal.ofReal δ ≤ w.ideal.escape B}, ((∑ u ∈ B.1, h u : ℕ) : ℝ)

/-- **`B` is within the window's one budget at `δ`**: the stratified sample misses it with probability at least `δ`,
so it spends at most `ln(1/δ)` of the detection budget across every stratum of every member, and its ANDs and its output
bits are each at most the most any set in the confidence set holds. -/
def Within (δ : ℝ) (ands bits : Fin w.nJ → ℕ) (B : Finset (Fin w.nJ)) : Prop :=
  ENNReal.ofReal δ ≤ w.ideal.escape B ∧ w.spend B ≤ Real.log (1 / δ) ∧
    ((∑ u ∈ B, ands u : ℕ) : ℝ) ≤ w.maxHarm ands δ ∧ ((∑ u ∈ B, bits u : ℕ) : ℝ) ≤ w.maxHarm bits δ

end JointAuditV5

/-- **The joint stratified bound over a window's members.** For a window's audit as in `JointAuditV5`, whose one live
stratified sample takes `k_s` of the `n_s` proof blocks of each stratum `s` across the members and whose members'
sessions each prove their part of it, every `δ > 0`, every per-proof-block weights `ands` (each proof block's ANDs) and
`bits` (its output bits), and every live prover `τ`, the probability that flock-verify accepts every member's session and
the window's incorrect proof blocks `B` are not within its one budget is at most
$$\delta + \sum_m (\varepsilon_{ks,m} + \varepsilon_{bind,m}):$$
outside it, an accepted window's incorrect proof blocks escape the stratified sample with probability at least `δ`,
spend `Σ_s k_s·x_s/n_s ≤ ln(1/δ)` across all its strata, and hold at most `maxHarm ands δ` ANDs and at most
`maxHarm bits δ` output bits. Its named assumptions: A3 (`Assumptions.UniformRandomBytes`) for the sample's bytes,
`live-verifier` at each member (`MemberV5.hCust`, `RecordCustodyZKJP`), and `cr/sha-512` (`Assumptions.TableCRZAt`) and
`ecr/sha-512` (`Assumptions.LinkCRZCAt`) at each member's session played alone at `τ`. -/
def JointStratifiedV5 : Prop :=
  ∀ (w : JointAuditV5) (δ : ℝ) (ands bits : Fin w.nJ → ℕ), 0 < δ →
    Assumptions.UniformRandomBytes w.Lb w.src →
    ∀ τ : Strategy w.game,
    (∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP (w.ms.get k).dj (w.ms.get k).y₀ ((w.memberLaw τ k).draw ω) (reg (w.memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP (w.ms.get k).dj (w.ms.get k).y₀) (cont (w.memberSim τ k) ω))
      (w.ms.get k).qF (w.ms.get k).qS) →
    (∀ k, Assumptions.LinkCRZCAt execArith hs512 (w.ms.get k).P (planZHJP (w.ms.get k).dj (w.ms.get k).y₀)
      (tabZHJP (w.ms.get k).dj (w.ms.get k).y₀) (w.ms.get k).vb (reg (w.memberSim τ k)) (cont (w.memberSim τ k))
      (w.ms.get k).k (w.ms.get k).Rw (w.ms.get k).M (w.ms.get k).t') →
    prob (fun o => w.Accepts o ∧ ¬ w.Within δ ands bits (w.wrongSet τ)) w.game τ ≤
      ENNReal.ofReal δ + ∑ k, ((w.ms.get k).epsKs (w.design k) + (w.ms.get k).link (w.design k))

/-- What a monitoring period's run records of window `idx`: whether every member's session accepted, its incorrect
proof blocks (`JointAuditV5.wrongSet`), and whether a sampled proof block is one of them. -/
structure AuditRecV5 (W : ℕ → JointAuditV5) where
  idx : ℕ
  accepted : Prop
  wrong : Finset (Fin (W idx).nJ)
  drawnWrong : Prop

namespace AuditRecV5

variable {W : ℕ → JointAuditV5}

/-- The probability that the record's window's stratified sample misses its incorrect proof blocks. -/
noncomputable def escape (r : AuditRecV5 W) : ℝ≥0∞ := (W r.idx).ideal.escape r.wrong

/-- The detection budget the record's incorrect proof blocks spend, `Σ_s k_s·x_s/n_s` over its window's strata. -/
noncomputable def spend (r : AuditRecV5 W) : ℝ := (W r.idx).spend r.wrong

/-- The record's incorrect proof blocks' total weight `Σ_{u ∈ B} h u`, with `h t` window `t`'s per-proof-block
weights. -/
def harm (h : (t : ℕ) → Fin (W t).nJ → ℕ) (r : AuditRecV5 W) : ℕ := ∑ u ∈ r.wrong, h r.idx u

end AuditRecV5

namespace JointAuditV5

open Classical

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
blocks in each of the run's windows that their samples miss together with probability at least `δ`. -/
noncomputable def maxHarm (W : ℕ → JointAuditV5) (h : (t : ℕ) → Fin (W t).nJ → ℕ) (δ : ℝ)
    (l : List (AuditRecV5 W)) : ℝ :=
  ⨆ B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).nJ) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)},
    ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)

/-- **The run's incorrect proof blocks are within the period's one budget at `δ`**: the windows' samples miss them
together with probability at least `δ`, so they spend at most `ln(1/δ)` of the detection budget across every stratum of
every window, and their ANDs and their output bits in all are each at most the most any choice in the period's
confidence set holds. -/
def Within (W : ℕ → JointAuditV5) (δ : ℝ) (ands bits : (t : ℕ) → Fin (W t).nJ → ℕ) (l : List (AuditRecV5 W)) :
    Prop :=
  ENNReal.ofReal δ ≤ (l.map AuditRecV5.escape).prod ∧ (l.map AuditRecV5.spend).sum ≤ Real.log (1 / δ) ∧
    (((l.map (AuditRecV5.harm ands)).sum : ℕ) : ℝ) ≤ maxHarm W ands δ l ∧
    (((l.map (AuditRecV5.harm bits)).sum : ℕ) : ℝ) ≤ maxHarm W bits δ l

end JointPeriodV5

/-- **The joint stratified bound over a monitoring period of windows of several members, pooled.** In a monitoring
period of `T` windows `W t` as in `JointAuditV5`, each with a fresh live stratified sample over its members' proof
blocks, played by one adaptive prover `s`, for every `δ > 0` and every per-proof-block weights `ands t` (each proof
block's ANDs) and `bits t` (its output bits), the probability that flock-verify accepts every member's session of every
window and their incorrect proof blocks are not within the period's one budget is at most
$$\delta + \sum_t \sum_m (\varepsilon_{ks,t,m} + \varepsilon_{bind,t,m}):$$
outside it, when every window is accepted, their samples miss their incorrect proof blocks together with probability at
least `δ`, the incorrect proof blocks spend `Σ_t Σ_s k_{t,s}·x_{t,s}/n_{t,s} ≤ ln(1/δ)` across every stratum of every
window, and their ANDs and output bits in all are at most the period's `maxHarm`. The sampling term is `δ` however many
windows and members the period has. Its named assumptions, at every window: A3 (`Assumptions.UniformRandomBytes`) for
its sample's bytes, `live-verifier` at each member (`MemberV5.hCust`, `RecordCustodyZKJP`), and `cr/sha-512`
(`Assumptions.TableCRZAt`) and `ecr/sha-512` (`Assumptions.LinkCRZCAt`) at each member's session played alone at every
strategy the window is played with on some run (`JointPeriodV5.Plays`). -/
def JointStratifiedHorizonV5 : Prop :=
  ∀ (W : ℕ → JointAuditV5) (T : ℕ) (δ : ℝ) (ands bits : (t : ℕ) → Fin (W t).nJ → ℕ), 0 < δ →
    (∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src) →
    ∀ s : Strategy (JointPeriodV5.game W T),
    (∀ t (τ : Strategy (W t).game), JointPeriodV5.Plays W T s t τ → ∀ k ω j, Assumptions.TableCRZAt execArith hs512
      (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀ (((W t).memberLaw τ k).draw ω) (reg ((W t).memberSim τ k))) j
      (zcStrat execArith hs512 (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) (cont ((W t).memberSim τ k) ω))
      ((W t).ms.get k).qF ((W t).ms.get k).qS) →
    (∀ t (τ : Strategy (W t).game), JointPeriodV5.Plays W T s t τ → ∀ k,
      Assumptions.LinkCRZCAt execArith hs512 ((W t).ms.get k).P (planZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀)
        (tabZHJP ((W t).ms.get k).dj ((W t).ms.get k).y₀) ((W t).ms.get k).vb (reg ((W t).memberSim τ k))
        (cont ((W t).memberSim τ k)) ((W t).ms.get k).k ((W t).ms.get k).Rw ((W t).ms.get k).M ((W t).ms.get k).t') →
    value (fun l => (∀ r ∈ l, r.accepted) ∧ ¬ JointPeriodV5.Within W δ ands bits l) (JointPeriodV5.run W T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ∑ k, (((W t).ms.get k).epsKs ((W t).design k) +
        ((W t).ms.get k).link ((W t).design k))

end Flock.Guarantees
