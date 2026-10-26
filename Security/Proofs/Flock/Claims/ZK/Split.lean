import Proofs.Flock.Claims.ZK.ExePlan
import Proofs.Flock.Claims.ZK.Stopped

/-!
# The deployed masked session's open premises, as statements

`Flock.zk_deployZk_masked_exe_sent_gap` (`Sent`) leaves `hstep`, `hc` and `hfin` at its plan. At the deployed plan
(`czExe`, `encExe`, `finExe` at `PE = ofSalts PcExe` and the executable's tape plan `UE`) they come from the eight
statements below; each is a `Prop`, proved on its own.

* `PrefixLive`: the session answered by the openings sends body `n` exactly where the session stopped right after round
  `n` does, the same body, and where it sends at most `n` bodies it threw. Executable code only.
* `SimPiop`, `SimLig`, `SimInner`: one rep's hybrids (`FlockComplete.repMix`): its PIOP, then its Ligerito opening, then
  its inner proof, read from the cut instead of computed, send the same bodies and the same proof but what only the
  final message reads, from any history of the rep's first round at which no refusal is seen.
* `SimCommit`: level 0's commitment and each rep's pads commitment from the cut's leaves are the session's but their
  final-only data.
* `SimSession`: the session read from the cut (`runCut`) sends the session's bodies (`runExe`), from `SimCommit` and the
  hybrids.
* `CausalExe`: the plan's `Causal` (`hc`). `FinExe`: its last message (`hfin`).

`h0`, `ho`, `hr` and `hp` hold by construction (`encExe_zero`, `outerReads_czExe`, `coinReads_czExe`,
`pointReads_czExe`); `hstep` is `PrefixLive` and `SimSession` at `c = padOps ops`, `n = ops.length`.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-! ## The deployed instance -/

/-- The session's hashes, SHA-512. -/
abbrev hsE : _root_.Flock.Hashes FlockComplete.PS := _root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash

/-- **The deployed session's Ligerito prover of levels ≥ 1.** -/
noncomputable abbrev PE : Composed.LigProver := ofSalts PcExe

/-- **The executable's tape plan** at `PE`. -/
noncomputable abbrev UE : TapePlan (tabC PE) := TapePlan.ofLayout (TapeLayout.exe PE) exeTape

/-- **The deployed round plan.** -/
noncomputable abbrev QE : RoundPlan (tabC PE) := RoundPlan.stop (czExe PE) (encExe PE) (finExe PE)

/-- **The session's cut at the opened coins `c`**, on `w`'s table witnesses and the tape `τ` (`RefSession.cutOf`). -/
noncomputable def cutE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut :=
  cutAt (tabC PE x) (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1
    (exeOf (tabC PE x) (czExe PE x c) (wtOf PE x w) (UE.tape x τ).1 (UE.tape x τ).2.1)

theorem cutOf_eq (nr : Msg → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (τ : UE.Tape) : (session nr (tableData PE QE UE)).cutOf x c (wtOf PE x w) τ = cutE x c w τ := rfl

/-- **What the deployed session reads from the cut** (`tabSrcCut`). -/
noncomputable def srcE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) :
    ℕ → FlockComplete.TabSrc :=
  tabSrcCut PE x (czExe PE x c) (cutE x c w τ)

/-- The sends answered by the opened coins `c`, stopped right after round `n`'s. -/
def ioE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) : FlockComplete.Asks FlockComplete.PS :=
  FlockComplete.asksCut (fun k i => slotAt c k i) n

/-- The tables' traces of the mask witness at the tape `ω`'s mask words. -/
def trsE (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) : Array FlockComplete.Trace :=
  FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask)

/-- **The masked session answered by the opened coins `c` and stopped right after round `n`'s send.** -/
def runExe (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ)
    (spec : _root_.Flock.SessionSpec) : Except String (String × Array ByteArray) × Array Msg :=
  (FlockComplete.sessionPZk (ioE c n) hsE spec
    (FlockDeploy.tabsOf (stsOf x)) (FlockComplete.MaskRank.masksOf (stsOf x)) (trsE x w ω) ω).go #[]

/-- **The masked session answered by the openings `ops`** (`Sent`'s `liveRun`). -/
def liveRunAt (x : Msg) (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg))
    (spec : _root_.Flock.SessionSpec) : Except String (String × Array ByteArray) × Array Msg :=
  (FlockComplete.sessionPZk (FlockComplete.asksS fun k => (ops.map fun p => coinOfD p.2).getD k [])
    hsE spec (FlockDeploy.tabsOf (stsOf x))
    (FlockComplete.MaskRank.masksOf (stsOf x)) (trsE x w ω) ω).go #[]

/-- The link points at the opened coins (`sessionPZk`'s `points`). -/
def ptsE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (spec : _root_.Flock.SessionSpec) :
    Array (Array _root_.Flock.F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun i => slotAt c 0 (p * spec.mPts + i)

/-- Rep `r`'s earlier claim points: rep `0`'s at rep `1` (`priorOf`, `RankBad`). -/
def priorE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j r : ℕ) : Array (List _root_.Flock.F128) :=
  if r = 1 then ptsOpen c (stsOf x)[j]! (2 * j) else #[]

/-- **Table `j`'s rep `r` as the hybrid `repMix k`** at the deployed instance: the session's table, schedule, trace,
packed table, tape and mask words, the cut's source, on level 0's commitment `cm0`, the extra claims and the pads
commitment `ic`. -/
noncomputable def mixE (k : ℕ) (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    (n : ℕ) (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit)
    (extras : Array _root_.Flock.Claim) (ic : FlockComplete.InnerCommit) :
    _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof :=
  FlockComplete.repMix k (ioE c n) hsE
    (FlockDeploy.tabsOf (stsOf x))[j.val]! (zkTab x j).sch (zkTab x j).tPad r.val dom
    (trsE x w (UE.tapeZk τ))[j.val]!
    (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m) cm0 extras
    (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog
        (zkTab x j).sch.k0)) ic
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
    ((UE.tapeZk τ).nonce j.val r.val) (priorE c j.val r.val) (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!
    ((srcE x c w τ j.val).rep r.val)

/-- **Hybrid `k` sends what hybrid `k + 1` sends**: at a live statement and a witness in the deployed relation, at any
opened coins `c` and stop `n` at which no refusal is seen, table `j`'s rep `r` started at its first round
(`repBase`) on a fresh transcript sends the same bodies and the same proof (but its final-only data), at any level-0
commitment, pads commitment and domain, at the statement's extra claims. -/
def HybridStep (k : ℕ) : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
      (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
    ∀ l : Array Msg, l.size = repBase x (2 * j.val + r.val) →
      ((mixE k x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
        ((mixE (k + 1) x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l

/-! ## The eight statements -/

/-- **1. The session answered by the openings, against the session stopped after round `n`** (executable code only):
where the openings `ops` are fewer than the pre-final rounds, the session answered by them sends body `n = |ops|`
exactly where the session answered by the padded openings and stopped right after round `n` does, the same body; and
where it sends at most `n` bodies, it threw. -/
def PrefixLive : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)), 0 < ops.length →
    ops.length < refK x → ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (ops.length < (liveRunAt x w ω ops spec).2.size ↔
        ops.length < (runExe x w ω (padOps (x := x) ops) ops.length spec).2.size) ∧
      (ops.length < (liveRunAt x w ω ops spec).2.size →
        (liveRunAt x w ω ops spec).2[ops.length]! =
          (runExe x w ω (padOps (x := x) ops) ops.length spec).2[ops.length]!) ∧
      ((liveRunAt x w ω ops spec).2.size ≤ ops.length → ∃ e, (liveRunAt x w ω ops spec).1 = .error e)

/-- **2. A rep's PIOP from the cut**: its zerocheck's and lincheck's masked values and its ring switch's vectors read
from the cut are the session's (`repMix 0` to `repMix 1`). -/
def SimPiop : Prop := HybridStep 0

/-- **3. A rep's Ligerito opening from the cut**: level 0's masked messages, `y₁'`, `ȳ` and levels ≥ 1's salts read from
the cut (the salts back from the rep's levels ≥ 1 value, `blobSalt` of `saltBlob`) send what the session's opening sends,
but level 0's opened rows and salts (`repMix 1` to `repMix 2`). -/
def SimLig : Prop := HybridStep 1

/-- **4. A rep's inner proof from the cut**: the triple's openings, `τ`'s commitment and `Y` read from the cut are the
session's, whose batched constraint holds at the honest pads where no outer coin is `1` (`repMix 2` to `repMix 3`). -/
def SimInner : Prop := HybridStep 2

/-- **5. The commitments from the cut's leaves**: at a live statement, table `j`'s level-0 commitment built from the cut's
leaves by tree position is the session's (`commitPZk`) but its rows and salts, and its rep `r` pads commitment the
session's (`innerCommitP`) but its columns and salts. -/
def SimCommit : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (l : Array Msg),
    (FlockComplete.commit0Cut hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle ((zkTab x j).sch.levels.getD 0 default) (srcE x c w τ j.val)).go l =
      (FlockComplete.eraseC <$> FlockComplete.commitPZk hsE
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle ((zkTab x j).sch.levels.getD 0 default) (zkTab x j).tPad
        (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m)
        ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val 0 0)).go l ∧
    ∀ r : Fin 2,
      (FlockComplete.padsTreeCut hsE
          (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len + _root_.Flock.Zk.K_PAD).nextPowerOfTwo *
            2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE) ((srcE x c w τ j.val).rep r.val)).go l =
        (FlockComplete.eraseIC <$> FlockComplete.innerCommitP hsE
          (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
              (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0))
          ((UE.tapeZk τ).innerSalt j.val r.val)).go l

/-- **6. The session from the cut sends the session's bodies**: at a live statement and a witness in the deployed
relation, at any opened coins and stop `n` at which no refusal is seen, `sessionCut` on the cut sends what the masked
session sends, both stopped right after round `n`'s send. From `SimCommit` and the hybrids `SimPiop`, `SimLig`,
`SimInner`, rep by rep. -/
def SimSession : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
      (runCut x c n spec (srcE x c w τ)).2 = (runExe x w (UE.tapeZk τ) c n spec).2

/-- **7. The deployed plan reads, at each round, the coins before it only** (`Sent`'s `hc`). -/
def CausalExe : Prop := QE.Causal

/-- **8. The deployed plan's last message is the honest prover's** where every table admits (`Sent`'s `hfin`). -/
def FinExe : Prop :=
  ∀ (H : Msg → Msg), Claims.hOf H = _root_.Flock.Sha512.hash → ∀ nr : Msg → ℕ → ℕ,
    (session nr (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE)).HonestFinOn Live RZk H
      coinOfD fun ω => UE.tapeZk ω.2

end Masked

end Claims.ZKMal
