import Proofs.Flock.Claims.Complete.Zk.Piop
import Proofs.Flock.Claims.Complete.Zk.Inner
import Proofs.Flock.Claims.Complete.Prover.Session
import Proofs.Flock.Soundness.Discharge.ZkView.Wire
import Proofs.Flock.Claims.Complete.Zk.MaskRank

/-!
# The masked prover's session

Live PROTOCOL.md §1–§7 from the prover's side, for the masked verifier (`Zk.verifyWithM`), in `flock-circuit --zk`'s
order (`prove_cpu_zk`, `zk_finish`).

* **A rep** (`repPZk`): the masked zerocheck and lincheck (`zerocheckPZk`, `lincheckPZk`, the pads' root bound), the
  rank check (`rankCheckP`: the prover stops at coins where the mask slot's weights at the claim points so far are
  rank-deficient), the ring switch on the true claims' points (`ringSwitchP`; the switched vectors are sent in the
  clear, the mask slot's words hiding them), the masked Ligerito (`ligeritoPZk`); then the constraints, as upstream
  builds them: the verifier's own replay (`Zk.replay`) on the rep's rounds so far, the two ring-switch claims'
  checks, and level 0's lane phase (`Zk.ligeritoConstraints` on the prover's `L0Rec`); then the inner proof
  (`innerP`). The proof file is `ZkView.encZk` at the shape `Zk.verifyRepWith` reads (`shapeZk`).
* **The session** (`sessionPZk`): each table's `--zk` schedule (`Zk.schedule`: level 0's `q_0` and `t_pad = 2 q_0`),
  its masked level-0 commitment (`commitPZk`, salted from the tape's level `0`), the link message, the link's points,
  then the rep streams in `spec.repStreams` order, each with its pads (`padsOf`) committed (`innerCommitP`) before its
  first coin; the record is `recordJson`'s, the keyed coin tree's `hello` and `coin_tree` included.
-/

namespace FlockComplete

open Flock Transcript

/-- The shape `Zk.verifyRepWith` decodes a rep's proof file at, with `extras` extra claims. -/
def shapeZk (st : Setup) (sch : Schedule) (extras : Nat) : Shape :=
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes
  { m := st.m, kLog := st.kLog, claims := 2 + extras, sched := sch, hashLen := st.merkle.digestLen, saltLen }

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The rank check** (`zk_masking`'s, at the opening's label): the prover stops unless the mask slot's weights at
the claim points so far, `prior` and the rep's own two (`MaskRank.roundPoints` on its rounds), have full rank over
the slot's `words`. -/
def rankCheckP (m kLog : Nat) (prior : Array (List F128)) (words : List Nat) : TMW M Unit := do
  let t ← get
  if MaskRank.maskRankOk (prior ++ MaskRank.roundPoints m kLog t.rounds) words then pure ()
  else throw "degenerate coins: the mask slot's weights are rank-deficient"

/-- **One masked rep's proof** of the table `tr` committed as `cm0` (its packed table `t`), as rep `rep` of a stream
whose transcript starts at `dom`, with the statement's extra claims, the rep's pads `pads` committed as `ic`, and the
table's extra lanes and paddings; after the lincheck, the rank check at the earlier reps' points `prior` over the mask
slot's `words`. -/
def repPZk (io : Asks M) (hs : Hashes M) (st : Setup) (sch : Schedule) (tPad rep : Nat) (dom : ByteArray) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64)
    (prior : Array (List F128)) (words : List Nat) : TMW M Zk.ZkProof := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let (zp, zc) ← zerocheckPZk io st.m st.digest (flat cm0.cap) ic.root lay pads.h tr
  let (lp, lc) ← lincheckPZk io st.fold st.m lay pads.h tr zc
  rankCheckP st.m st.kLog prior words
  let (ss, T, sws) ← ringSwitchP io tr (abcClaims st.m st.kLog zc lc ++ extras)
  let (lg, zl, rec) ← ligeritoPZk io hs st.merkle sch T cm0 t (basisVals sws (st.m - 7)) lay pads.h rep tPad extra
    lanePad salt nonce
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  let inner ← innerP io hs lay pads ic zp.finalC
    (rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lgc)
  return { proof := p, level0 := zl, inner }

/-- The claim points a table's rep `r` checks ahead of its own: none at rep `0`; at rep `1`, rep `0`'s, read off its
rounds, the last stream so far. -/
def priorOf (st : Setup) (streams : Array (Array Round)) (r : Nat) : Array (List F128) :=
  if r = 1 then MaskRank.roundPoints st.m st.kLog streams.back! else #[]

/-- **The honest masked session** on each table's trace: the `--zk` schedules, the masked level-0 commitments, the
link, the rep streams; the record and the proof files (`spec.repStreams` order). Each table's rank check is over its
mask slot's `words`, and its rep 1's covers rep 0's points too (read from rep 0's rounds, the stream before it). -/
def sessionPZk (io : Asks M) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (words : Array (List Nat))
    (trs : Array Trace) (tape : TapeZk) : M (String × Array ByteArray) := do
  let schs ← tabs.mapM fun st => (monadLift (Zk.schedule st.m) : M (Schedule × Nat))
  let tbls := (Array.range tabs.size).map fun i => packed trs[i]! tabs[i]!.m
  let cms ← (Array.range tabs.size).mapM fun i =>
    commitPZk hs tabs[i]!.merkle (schs[i]!.1.levels.getD 0 default) schs[i]!.2 tbls[i]! (tape.extra i)
      (tape.lanePad i) (tape.salt i 0 0)
  let rootB := cms.map fun c => flat c.cap
  let c1 ← io.link rootB (tabs.map fun _ => ByteArray.empty)
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)
  let mut streams : Array (Array Round) := #[]
  let mut proofs : Array ByteArray := #[]
  for (i, r, _) in spec.repStreams do
    let st := tabs[i]!
    let sch := schs[i]!.1
    let extras ← (monadLift (st.extra points) : M _)
    let pads := padsOf tape i r (Zk.PadLayout.new st.m st.kLog sch.k0)
    let ic ← innerCommitP hs pads (tape.innerSalt i r)
    let dom := (spec.domain spec.tables[i]! r).toUTF8
    let (p, t) ← (repPZk io hs st sch schs[i]!.2 r dom trs[i]! tbls[i]! cms[i]! extras pads ic (tape.extra i)
      (tape.lanePad i) (tape.salt i r) (tape.nonce i r) (priorOf st streams r) words[i]!).run
      (Transcript.init #[] dom)
    streams := streams.push t.rounds
    proofs := proofs.push (FlockSoundness.Discharge.ZkView.encZk (shapeZk st sch extras.size) p)
  let record ← recordJson hs spec rootB points streams proofs
  return (record, proofs)

end FlockComplete
