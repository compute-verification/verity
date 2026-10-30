import Proofs.Flock.Recursive.LinksRows

/-!
# A link part's unit at each instance: its reference's output is the verifier's roots' rows

At a staged link part (V\*'s statement `.link UB PB N`), instance `n < N`'s seven input ports, read off the outer
witness's input columns as u16 words (`inWordsAt`), give the reference link check (`RecLinkRef.linkRef`) the rows
`expected R` and `expected W` that check 5 holds the instance's `rin` and `rout` slots to (`linkSetup_staged`), or
SHA-512 collides on two of the slots' messages and salts (`refMsgs`, `linkOut_bind`'s). `LinkRefStaged` is that
statement; the link clause's binding (`LinksBindDeployed`) reads it with `inner_link`.
-/

namespace RecDeploy.Links

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-- Instance `n`'s bytes at port `nm`'s input columns (`rowHidden`), as the witness `w` holds them. -/
def inBytesAt (c : Circuit) (w : Claims.Flock.Wit) (nm : String) (n : ℕ) : ByteArray :=
  ⟨(readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var)).toArray⟩

/-- Instance `n`'s row at port `nm` as u16 words. -/
def inWordsAt (c : Circuit) (w : Claims.Flock.Wit) (nm : String) (n : ℕ) : List ℕ :=
  RecLinkRef.wordsOf (inBytesAt c w nm n)

/-- Port `nm`'s index in `c`. -/
def portIdx (c : Circuit) (nm : String) : ℕ := (c.ports.findIdx? (·.name == nm)).getD 0

/-- The program digest `c`'s META names (`linkValue_slot`'s). -/
def progOf (c : Circuit) : String := (c.metaJ.getObjValAs? String "program_sha512").toOption.getD ""

/-- **An output slot's messages**: at port `nm`'s slot at instance `n`, the unit's row message and the message of
check 5's row `E`, then the slot's salt digest input and the verifier's salt's (`linkOut_bind`'s three cases). -/
def outMsgs (c : Circuit) (w : Claims.Flock.Wit) (n : ℕ) (nm value : String) (E : Array ℕ) : List Msg :=
  [rowPfx c (portIdx c nm) ++ readForms w (DeployedR.outRowForms c (portIdx c nm) n),
    ResLower.rowInput (RecCoins.padTo E (Registered.rowWords E.size 16)),
    FlockSoundness.Refine.K512.saltPrefix ++ readForms w (hmSalt c (portIdx c nm) n),
    FlockSoundness.Refine.K512.saltPrefix ++ (RecCoins.verifierSalt (progOf c) value n).data.toList]

/-- The strings a link part's instance `n` names a collision among: `rin`'s slot's at `expected R`, `rout`'s at
`expected W`. -/
def refMsgs (c : Circuit) (w : Claims.Flock.Wit) (n : ℕ) (R W : ByteArray) : List Msg :=
  outMsgs c w n "rin" RecLinkPart.IN_VALUE (RecLinkRef.expected R).toArray ++
    outMsgs c w n "rout" RecLinkPart.OUT_VALUE (RecLinkRef.expected W).toArray

/-- The reference link check at instance `n`'s ports: `trow`, `xyr` split as `xr` (32 words) and `yr`, `raux`, `xyw`
split as `xw` and `yw`, `waux`. -/
def linkRefAt (UB PB : ℕ) (c : Circuit) (w : Claims.Flock.Wit) (n : ℕ) : Except String (List ℕ × List ℕ) :=
  RecLinkRef.linkRef UB PB (inWordsAt c w "trow" n) ((inWordsAt c w "xyr" n).take 32)
    ((inWordsAt c w "xyr" n).drop 32) (inWordsAt c w "raux" n) ((inWordsAt c w "xyw" n).take 32)
    ((inWordsAt c w "xyw" n).drop 32) (inWordsAt c w "waux" n)

/-- **A staged link part's unit gives the verifier's rows**: at a staged link part the witness checks against, with
the setup's facts `linkSetup_staged` gives (its rows `expected R` and `expected W` at roots `rowsRoot` accepted, so
not all zero), each instance `n < N`'s reference output is `(expected R, expected W)`, or two of its slots' messages
collide. -/
def LinkRefStaged : Prop :=
  ∀ {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit}, StagedPart st s x xO →
    Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true →
    ∀ {r : XR}, readX x = some r → ∀ {ss : Array VStar.Stmt}, RecVStar.statements r.shape = .ok ss →
    ∀ {UB PB N : ℕ}, ss[s]? = some (.link UB PB N) → ∀ {c : Circuit}, r.w.parts[s]?.bind partCircuit = some c →
    ∀ {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}, setupO xO = some (i, spec, sts) →
    ∀ (h0 : 0 < sts.size) {own : Array Registered.Port}, i.base.registered = some own →
    Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) → N ≤ sts[0].n → sts[0].n = sts[0].pub.n →
    ∀ {X Y : List ((ℕ × ℕ) × ByteArray)} {R W : ByteArray}, RecLinkTree.rowsRoot UB PB X = .ok R →
      RecLinkTree.rowsRoot UB PB Y = .ok W →
    ∀ {wi wo : ℕ → String}, RecCoins.checkValue sts[0].c sts[0].pub own "rin" RecLinkPart.IN_VALUE
        (Array.replicate N (RecLinkRef.expected R).toArray) wi = .ok () →
      RecCoins.checkValue sts[0].c sts[0].pub own "rout" RecLinkPart.OUT_VALUE
        (Array.replicate N (RecLinkRef.expected W).toArray) wo = .ok () →
    ∀ {n : ℕ}, n < N →
      linkRefAt UB PB c w n = .ok (RecLinkRef.expected R, RecLinkRef.expected W) ∨
        ∃ a ∈ refMsgs c w n R W, ∃ b ∈ refMsgs c w n R W, Claims.Collision FlockDeploy.sha512 a b

end RecDeploy.Links
