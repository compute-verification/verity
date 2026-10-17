import Proofs.Flock.Claims.Recursion.DeployReg

/-!
# The V* layout off the registered ports (`layoutOf`), and the recursion's KS at it

`ksA_deployed_parts` (`DeployReg.lean`) takes the V* layout as a parameter. This file reads it off the compiled
statement (`layoutOf`), from the message-round ports V* registers (`RecVStar.ownPorts`):

* **Rounds.** The inner session's are the deployed plain verifier's (`FlockDeploy.V false`'s `k` at the inner
  statement): the link message, each rep stream's rounds (rep `0`'s, then rep `1`'s), and the last message.
* **Messages** (`partHolds`, `Layout.msgAt`). Part `s`'s V* statement (statement `s` of `RecVStar.statements` at the
  verifier's own inner shape) registers each message-round port under `rec-c<k>` (`RecVStar.cIndex`) with the rows it
  reads, row `r` rep `r`'s at instance `r`. Such a port holds rep `r`'s `k`-th message round, round
  `commitRounds[k]` of the rep's schedule (`repSched`), which is inner round `1 + r · R + commitRounds[k]` for a
  stream of `R` rounds (`msgRound`). Its bits are the port's row (`Port.bits`; row bit `i` is bit `i % 8` of row byte
  `i / 8`, as `bytesOf` reads them), at the hidden values of the instance's copy in the part's one table
  (`rowHidden`: block `r / g`, VU `r % g`, the unit input leaf that reads each row word, at `Readable.copyCol`'s
  index), in the part's circuit as the setup parses it (`partCircuit`). Ports under `rec-opening-c<k>` (R's stream)
  or `rec-claim` hold no round of the plain verifier's rep streams.
* **Calls.** None (`shaAt = []`): the SHA-512 evaluations a part computes hash its bits after constant prefixes (a
  row's `sha512/row/v1` prefix, the salt prefix), which a list of hidden values cannot express.

`Brings.atPorts` brings the readers at `layoutOf`. `deployRec` at it reads through them (`readsThrough_atPorts`, by its
definition), its V* statements number `nS` (`vstarLen_atPorts`), and every call a part records is a true evaluation
(`partCallsTrue_layoutOf`: there are none). `ksA_deployed_concrete` and `ksFlock_deployed_concrete` are
`ksA_deployed_parts` there, with no layout parameter.

**What `layoutOf` leaves open.** No port holds the link message (round `0`) or the last message, and a port holds the
row its round's commit string opens to, not the round's framed message (`0 :: …`, `FlockDeploy.act`). `Held` asks that
some part hold every inner round, so it fails at `layoutOf` wherever a compiled statement parses at the shape
(`not_held_layoutOf`), and `VStarParts` with it (`not_vstarParts_layoutOf`): the concrete theorems' V* hypotheses hold
only at a shape no compiled statement parses at.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-! ## The layout -/

/-- One rep stream's rounds at the verifier's inner shape: `RecVStar.repRounds`, or `holoRepRounds` over a holo inner
session (`RecVStar.innerMsgRounds`' schedule). -/
def repSched (sh : RecVStar.Inner) : Option (Array RecVStar.SchedRound) :=
  (if sh.holo then RecVStar.holoRepRounds else RecVStar.repRounds) sh.m sh.kLog sh.claims

/-- Rep `r`'s `k`-th message round as a round of the rep streams that follow the link message: rep `r`'s stream after
the earlier reps', at its `k`-th round that absorbs a message (`RecVStar.commitRounds`). -/
def msgRound (rs : Array RecVStar.SchedRound) (r k : ℕ) : Option ℕ :=
  ((RecVStar.commitRounds rs)[k]?).map (r * rs.size + ·)

/-- A part's circuit, as the setup parses it (`ProvedScope.circuitOf`) at its session's tags. -/
def partCircuit (p : PartX) : Option Circuit := do
  let ins ← p.base.ins.toOption
  (ProvedScope.circuitOf ins.read.tags p.base.circuit p.base.tables).toOption

/-- The input column of row bit `i` of the circuit's port `q` in unit `u`: bit `i % 16` of the unit input leaf that
reads the port's word `i / 16` (`leavesIn`), the ports' words in order. -/
def portCol (c : Circuit) (u q i : ℕ) : Option ℕ := do
  let w : ℕ := (c.ports.extract 0 q).foldl (· + ·.words) 0 + i / 16
  let k ← (c.leavesIn.getD u #[]).findIdx? (· == (w : Int))
  pure (16 * k + i % 16)

/-- **The hidden values of port `nm`'s row at instance `n`** of a part's one table: each row bit's input column
(`portCol`) in the instance's copy (block `n / g`, VU `n % g`, its first unit: `Circuit.instOf`), at
`Readable.copyCol`'s index for one table. -/
def rowHidden (c : Circuit) (nm : String) (n : ℕ) : Option (List ℕ) := do
  let q ← c.ports.findIdx? (·.name == nm)
  let p ← c.ports[q]?
  let base := c.unitRange.slot (n % c.g * c.upv)
  (List.range p.bits).mapM fun i => (portCol c 0 q i).map fun col => n / c.g * 2 ^ c.kLog + (base + col)

/-- **What part `s` holds**, off its registered message-round ports (`RecVStar.ownPorts`): for each port under
`rec-c<k>` and each row `r` it reads, rep `r`'s `k`-th message round as a round of the rep streams (`msgRound`), with
the port's row's hidden values at instance `r` (`rowHidden`). -/
def partHolds (r : XR) (s : ℕ) : List (ℕ × List ℕ) :=
  match (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?), r.w.parts[s]?.bind partCircuit,
      repSched r.shape with
  | some st, some c, some rs =>
    (RecVStar.ownPorts st).flatMap fun q => match RecVStar.cIndex q.2.1 with
      | some k => q.2.2.toList.filterMap fun row => do
        let rep ← row
        let j ← msgRound rs rep k
        let ps ← rowHidden c q.1 rep
        pure (j, ps)
      | none => []
  | _, _, _ => []

/-- **The V\* layout off the registered ports**: the deployed plain verifier's inner rounds; at round `0`, the link
message, no part; at round `j + 1`, the rep streams' round `j`, as part `s`'s first holding at it (`partHolds`); no
SHA-512 evaluation. -/
def layoutOf : Claims.Recursion.VStar.Layout where
  rounds x := (FlockDeploy.V false).k (xIOf x)
  msgAt x s i := match i with
    | 0 => none
    | j + 1 => (readX x).bind fun r => ((partHolds r s).find? (·.1 == j)).map (·.2)
  shaAt _ _ := []

/-- No part holds the link message. -/
theorem layoutOf_msgAt_zero (x : Msg) (s : ℕ) : layoutOf.msgAt x s 0 = none := rfl

theorem layoutOf_rounds_pos (x : Msg) : 0 < layoutOf.rounds x := by
  show 0 < (FlockDeploy.V false).k (xIOf x)
  simp only [FlockDeploy.V]
  split <;> omega

/-- **`Held` fails at `layoutOf`** wherever a compiled statement parses at the shape: it asks that some part hold inner
round `0`, the link message, which no port holds. -/
theorem not_held_layoutOf {W : Type} {D : Claims.Recursion.Deploy Coin} {I : Claims.Recursion.Inner Coin W}
    {shape : Claims.Recursion.Form → Prop} (h : Claims.Recursion.VStar.Held D I layoutOf shape) :
    ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → ¬ shape st.form := by
  intro x st hst hf
  obtain ⟨s, -, hs⟩ := (h x st hst hf).2 0 (layoutOf_rounds_pos x)
  simp [layoutOf_msgAt_zero] at hs

/-- **`VStarParts` fails at `layoutOf`** wherever a compiled statement parses at the shape, as its `Held` does. -/
theorem not_vstarParts_layoutOf {W Col T : Type} {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin}
    {I : Claims.Recursion.Inner Coin W} {DO : Claims.Flock.Deploy Coin Col T} {opening : Claims.Recursion.Opening}
    {shape : Claims.Recursion.Form → Prop}
    (h : Claims.Recursion.VStar.VStarParts H D I DO layoutOf opening shape) :
    ∀ (x : Msg) (st : Claims.Recursion.Stmt), D.parse x = some st → ¬ shape st.form := by
  obtain ⟨_, _, _, _, -, -, -, -, -, -, hH⟩ := h
  exact not_held_layoutOf hH

/-! ## The readers at the ports -/

/-- **What the deployment brings, with the readers at `layoutOf`.** -/
def Brings.atPorts (b : Brings) : Brings := { b with rd := readersOf layoutOf }

/-- **`deployRec` reads through the readers at `layoutOf`**, where it brings them. -/
theorem readsThrough_atPorts (H : Msg → Msg) (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.VStar.ReadsThrough H (deployRec IV ivScope b.atPorts) IV layoutOf.msgs layoutOf.calls :=
  readsThrough_deployRec H IV ivScope b.atPorts layoutOf rfl

theorem vstarLen_atPorts (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings) :
    Claims.Recursion.VStar.VstarLen (deployRec IV ivScope b.atPorts) :=
  vstarLen_deployRec IV ivScope b.atPorts

/-- **No part records a call at `layoutOf`**, so every one it records is a true evaluation. -/
theorem partCallsTrue_layoutOf {Col T : Type} (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin)
    (DO : Claims.Flock.Deploy Coin Col T) (s : ℕ) : Claims.Recursion.VStar.PartCallsTrue H D DO layoutOf s := by
  intro x st _ xO w _ e he
  simp [Claims.Recursion.VStar.Layout.calls, layoutOf] at he

/-! ## The recursion's KS at the ports -/

/-- **The recursion's KS at `deployRec` from V\*'s parts at the registered ports**: `ksA_deployed_parts` at
`layoutOf`, where the deployment brings its readers (`Brings.atPorts`), with no layout parameter: the readers are the
layout's by construction, and no part records a call (`partCallsTrue_layoutOf`). `not_vstarParts_layoutOf`: the V*
hypotheses hold only at a shape no compiled statement parses at. -/
theorem ksA_deployed_concrete (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (hext : Claims.Flock.Extendθ (FlockDeploy.deployPlain rel) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlain rel) decI)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) :=
  ksA_deployed_parts rel relO ivScope b.atPorts layoutOf rfl opening hF hext htab decI hdec
    (partCallsTrue_layoutOf _ _ _) hD hH hS

/-- **`KSFlock` at `deployRec` from V\*'s parts at the registered ports**: `ksA_deployed_concrete` at every inner
decoder that meets the spec, as `KSA` is `KS`. -/
theorem ksFlock_deployed_concrete (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation)
    (ivScope : Msg → Bool) (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (hext : Claims.Flock.Extendθ (FlockDeploy.deployPlain rel) (FlockDeploy.KS.wordLen false)
      (FlockDeploy.KS.wide false))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hD : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlain rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Dense)
    (hH : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlain rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Holo)
    (hS : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlain rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlain rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (FlockDeploy.deployPlain rel) (deployZkRec relO) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec =>
    ksA_deployed_concrete rel relO ivScope b opening hF hext htab decI hdec (hD decI hdec) (hH decI hdec)
      (hS decI hdec)

end RecDeploy
