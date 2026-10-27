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
  index), in the part's circuit as the setup parses it (`partCircuit`). The round's framed message is the frame `0`
  then that row (`FlockDeploy.act`'s tag `0`), so the layout's segment is the constant `[0]` then the row's hidden
  values. Ports under `rec-opening-c<k>` (R's stream) or `rec-claim` hold no round of the plain verifier's rep streams.
* **Fixed rounds** (`streamFixed`, `Layout.fixedAt`). A stream round that absorbs no message (at `m = 22`,
  `k_log = 10`, `claims = 4`: 28 of the 128 stream rounds) has the frame alone, `[0]`, which the statement fixes.
* **The public round** (`Layout.pubAt`). Round `0`, the link message: the prover sends it in the clear, and it carries
  all of `root_B` (`2^cap0` nodes per table) and the public words, which no part holds (rec-open's `out` port reads two
  rows per opening).
* **Calls.** None (`shaAt = []`). A part's SHA-512 evaluations hash its bits after constant prefixes (a row's
  `sha512/row/v1` prefix, the salt prefix), which `Layout.shaAt` states; their hidden values in the part's circuit are
  not read off the statement yet.

`Brings.atPorts` brings the readers at `layoutOf`. `deployRec` at it reads through them (`readsThrough_atPorts`, by its
definition), its V* statements number `nS` (`vstarLen_atPorts`), and every call a part records is a true evaluation
(`partCallsTrue_layoutOf`: there are none). `ksA_deployed_concrete` and `ksFlock_deployed_concrete` are
`ksA_deployed_parts` there, with no layout parameter.

**`HeldPub` at `layoutOf`** (`heldPub_layoutOf`): every inner round before the last is public, fixed or some part's,
wherever the V* shape's rep schedule is the inner verifier's (`SchedMatches`) and the registered ports cover each rep's
message rounds (`PortsCover`), two named facts about the compiled statement. **What `layoutOf` leaves open**: the
readers off the witnesses read `[]` at the link message, so `ReadsPub` holds only for readers that read it off the run
(the staged V* statements show it, `cmtTr`), and the last message, which no part holds, opens only by `FinalOpens`.
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

/-- The input column of row bit `i` of the circuit's port `q` in unit `u`: bit `i % 16` of the unit input leaf `k` that
reads the port's word `i / 16` (`leavesIn`), the ports' words in order, leaf `k` from column `col · WORD + 16 k` of the
unit's first input group (`Flat.leafCol`). -/
def portCol (c : Circuit) (u q i : ℕ) : Option ℕ := do
  let w : ℕ := (c.ports.extract 0 q).foldl (· + ·.words) 0 + i / 16
  let k ← (c.leavesIn.getD u #[]).findIdx? (· == (w : Int))
  pure ((c.unit.inGroups.getD 0 default).col * WORD + 16 * k + i % 16)

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

/-- **The rep streams' rounds whose message is constant**, at the verifier's inner shape: round `j` of the two reps'
streams (rep `j / R`'s round `j % R`, of `R`) where that round absorbs no message (`RecVStar.SchedRound`'s flag), whose
framed message is the frame alone, `[0]` (`FlockDeploy.act`'s tag `0`, no bytes). -/
def streamFixed (x : Msg) (j : ℕ) : Option Msg :=
  (readX x).bind fun r => (repSched r.shape).bind fun rs =>
    if j < 2 * rs.size ∧ (rs.getD (j % rs.size) default).2.2.2 = false then some [0] else none

/-- **The V\* layout off the registered ports**: the deployed plain verifier's inner rounds; at round `0`, the link
message, public (the prover sends it in the clear) and no part's; at round `j + 1`, the rep streams' round `j`: the
frame `[0]` where it absorbs no message (`streamFixed`), else the frame `0` then the row of part `s`'s first port
holding it (`partHolds`); no SHA-512 evaluation. -/
def layoutOf : Claims.Recursion.VStar.Layout where
  rounds x := (FlockDeploy.V false).k (xIOf x)
  msgAt x s i := match i with
    | 0 => none
    | j + 1 => (readX x).bind fun r => ((partHolds r s).find? (·.1 == j)).map fun p => [([0], p.2)]
  shaAt _ _ := []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => streamFixed x j
  pubAt _ i := i == 0

/-- No part holds the link message. -/
theorem layoutOf_msgAt_zero (x : Msg) (s : ℕ) : layoutOf.msgAt x s 0 = none := rfl

/-- The link message is public. -/
theorem layoutOf_pubAt_zero (x : Msg) : layoutOf.pubAt x 0 = true := rfl

theorem layoutOf_msgAt_succ (x : Msg) (s j : ℕ) : layoutOf.msgAt x s (j + 1) =
    (readX x).bind fun r => ((partHolds r s).find? (·.1 == j)).map fun p => [(([0] : Msg), p.2)] := rfl

theorem layoutOf_fixedAt_succ (x : Msg) (j : ℕ) : layoutOf.fixedAt x (j + 1) = streamFixed x j := rfl

theorem layoutOf_rounds_pos (x : Msg) : 0 < layoutOf.rounds x := by
  show 0 < (FlockDeploy.V false).k (xIOf x)
  simp only [FlockDeploy.V]
  split <;> omega

/-! ## Each inner round before the last at `layoutOf` -/

/-- **The V\* shape's rep schedule is the inner verifier's** at the compiled statements of the shape: the verifier's own
inner shape has a rep schedule (`repSched`) of `R` rounds, and the deployed plain verifier takes `2 + 2R` rounds at the
inner statement, the link message, the two reps' streams and the last message (`FlockDeploy.V`, one table). -/
def SchedMatches (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∃ r rs, readX x = some r ∧ repSched r.shape = some rs ∧
    (FlockDeploy.V false).k (xIOf x) = 2 + 2 * rs.size

/-- **The registered ports cover every message round** at the compiled statements of the shape: each rep's every round
that absorbs a message (`RecVStar.commitRounds`) is some part's, through a port under its `rec-c<k>` (`partHolds`). -/
def PortsCover (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ r rs, readX x = some r → repSched r.shape = some rs →
    ∀ rep < 2, ∀ t ∈ RecVStar.commitRounds rs, ∃ s < nS x, ∃ ps, (rep * rs.size + t, ps) ∈ partHolds r s

/-- **`HeldPub` at `layoutOf`**, at every shape where the V* schedule is the inner verifier's (`SchedMatches`) and the
registered ports cover the message rounds (`PortsCover`): round `0`, the link message, is public; a stream round
`j + 1` that absorbs no message is fixed (`streamFixed`); one that absorbs one is rep `j / R`'s round `j % R`, among its
message rounds (`RecVStar.commitRounds`), which some part's port holds. The last round, Flock's last message, is
`HeldPub`'s to leave out. -/
theorem heldPub_layoutOf {W : Type} (I : Claims.Recursion.Inner Coin W) (hI : I.V = FlockDeploy.V false)
    (ivScope : Msg → Bool) (b : Brings) (shape : Claims.Recursion.Form → Prop) (hM : SchedMatches shape)
    (hC : PortsCover shape) : Claims.Recursion.VStar.HeldPub (deployRec I.V ivScope b) I layoutOf shape := by
  intro x st hst hf
  obtain ⟨r, rs, hr, hrs, hk⟩ := hM x st hst hf
  refine ⟨?_, fun i hi => ?_⟩
  · show I.V.k (xIOf x) ≤ (FlockDeploy.V false).k (xIOf x)
    rw [hI]
  change i + 1 < I.V.k (xIOf x) at hi
  rw [hI, hk] at hi
  cases i with
  | zero => exact Or.inl rfl
  | succ j =>
    right
    have hj : j < 2 * rs.size := by omega
    have hpos : 0 < rs.size := by omega
    by_cases hab : (rs.getD (j % rs.size) default).2.2.2 = false
    · left
      rw [layoutOf_fixedAt_succ, streamFixed, hr, Option.bind_some, hrs, Option.bind_some]
      simp only [hj, hab, and_self, ↓reduceIte, Option.isSome_some]
    · right
      have ht : j % rs.size ∈ RecVStar.commitRounds rs := by
        simp only [RecVStar.commitRounds, List.mem_filter, List.mem_range]
        exact ⟨Nat.mod_lt _ hpos, by simpa using hab⟩
      obtain ⟨s, hs, ps, hmem⟩ :=
        hC x st hst hf r rs hr hrs (j / rs.size) ((Nat.div_lt_iff_lt_mul hpos).2 (by omega)) _ ht
      rw [Nat.div_add_mod' j rs.size] at hmem
      refine ⟨s, hs, ?_⟩
      rw [layoutOf_msgAt_succ, hr, Option.bind_some]
      cases hfind : (partHolds r s).find? (·.1 == j) with
      | none => exact absurd (List.find?_eq_none.1 hfind _ hmem) (by simp)
      | some p => rfl

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
layout's by construction, and no part records a call (`partCallsTrue_layoutOf`). Each `VStarParts`' `HeldPub` holds
where the ports cover the message rounds (`heldPub_layoutOf`). `ReadsPub` asks the readers for the link message, which
the readers off the witnesses alone don't read (no part holds it), and `FinalOpens` for an opening of the last
message. -/
theorem ksA_deployed_concrete (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation) (ivScope : Msg → Bool)
    (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    (hdec : Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI)
    (hPub : Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf)
    (hFin : Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) opening)
    (hD : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Dense)
    (hH : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Holo)
    (hS : Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
      Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSA FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) :=
  ksA_deployed_parts rel relO ivScope b.atPorts layoutOf rfl opening hF htab decI hdec
    (partCallsTrue_layoutOf _ _ _) hPub hFin hD hH hS

/-- **`KSFlock` at `deployRec` from V\*'s parts at the registered ports**: `ksA_deployed_concrete` at every inner decoder
that meets the spec, as `KSA` is `KS`. Its level (`Claims.Recursion.lam`) is about 65 bits per SHA-512 call at the
deployed shape (one inner table at `m = 35`, `γ`'s weight at `IVStar` at `coverW`), below the 100-bit floor and the
`2^-128` target under every weight. Route 2C's bound λ₂ (`lamLin2`) evaluates to at least 100 at the deployed
records (`RecDeploy.lamLin2_deployed_ge_dep`), but `KSLin2` at the deployment's dense statements (`KSLin2On` at
`RecDeploy.DenseAt`) only follows from premises still open (`RecDeploy.ksLin2_deployed_open`), Holo statements being
candidate 3's own model, and `BitsLin2` at those statements follows from it (`bitsLin2On_of_ksLin2On`). -/
theorem ksFlock_deployed_concrete (rel relO : Flock.Readable.Parsed → Flock.Readable.Relation)
    (ivScope : Msg → Bool) (b : Brings) (opening : Claims.Recursion.Opening)
    (hF : Claims.Flock.KS FlockDeploy.sha512 (deployZkRec relO))
    (htab : Claims.Recursion.TableBoundθOnAcc FlockDeploy.sha512 (FlockDeploy.deployPlain rel)
      (FlockDeploy.KS.wordLen false) (FlockDeploy.KS.wide false))
    (hPub : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.ReadsPub FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf)
    (hFin : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.FinalOpens FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) opening)
    (hD : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Dense)
    (hH : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Holo)
    (hS : ∀ decI, Claims.Recursion.decOK (FlockDeploy.deployPlainTrim rel) decI →
      Claims.Recursion.VStar.VStarParts FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
        (Claims.Recursion.Inner.ofFlock (FlockDeploy.deployPlainTrim rel) decI) (deployZkRec relO) layoutOf opening
        Claims.Recursion.Form.Sparse) :
    Claims.Recursion.KSFlock FlockDeploy.sha512 (deployRec (FlockDeploy.V false) ivScope b.atPorts)
      (FlockDeploy.deployPlainTrim rel) (deployZkRec relO) :=
  (Claims.Recursion.KSFlockA_iff_KSFlock _ _ _ _).1 fun decI hdec =>
    ksA_deployed_concrete rel relO ivScope b opening hF htab decI hdec (hPub decI hdec) (hFin decI hdec)
      (hD decI hdec) (hH decI hdec) (hS decI hdec)

end RecDeploy
