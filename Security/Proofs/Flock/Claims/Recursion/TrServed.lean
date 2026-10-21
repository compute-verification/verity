import Proofs.Flock.Claims.Recursion.FinOpen

/-!
# The coin server's transcript at the histories the firewall serves (`trServed`), injective from its record's

`ksLin2_deployed_open` reads the coin server's transcript `b.tr x` as injective at every compiled statement `x`
(`htr`): a V\* statement's transcript is read back through it (`shownOf`, `Function.invFun`). The deployed firewall
serves only the histories on its pinned schedule and stops a session off it, so the deployment writes `b.tr x` only at
those histories; at the others it writes no transcript, and `b.tr` there is the model's choice.

`trServed served tr` is `tr` at the histories `served` names and a tagged raw encoding of the history elsewhere
(`rawTr`: a `0` byte, then the rounds, each message byte-escaped, `escB`). `trServed_injective` proves it injective
from two facts about `tr` at the served histories: it is injective there, and its first byte is `{` (`0x7b`), as the
coin server's record is a JSON object (`rec_live.CoinServer.record`, written with `json.dumps`).
`ksLin2_deployed_served` and `bitsLin2_deployed_served` are `ksLin2_deployed_open` and `bitsLin2_deployed_open` at
`trServed served b.tr`, with `htr` replaced by those two facts.

`served` is a parameter: the deployment is the one these bound where it holds at every history the deployed firewall
serves. That the coin server's record is injective at those histories is still open (a concrete record writer in Lean
with a read-back proof, held to `rec_live` by a difftest); the Lean `Flock.Firewall.record` doesn't record the coins.
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)

/-! ## A prefix-free encoding of the rounds -/

/-- A message's bytes, each after a `1`, then a `0`: prefix-free. -/
def escB : List UInt8 → List UInt8
  | [] => [0]
  | b :: m => 1 :: b :: escB m

theorem escB_app : ∀ {m m' r r' : List UInt8}, escB m ++ r = escB m' ++ r' → m = m' ∧ r = r'
  | [], [], _, _, h => ⟨rfl, by simpa [escB] using h⟩
  | [], _ :: _, _, _, h => by simp [escB] at h
  | _ :: _, [], _, _, h => by simp [escB] at h
  | b :: m, b' :: m', r, r', h => by
    simp only [escB, List.cons_append, List.cons.injEq, true_and] at h
    obtain ⟨rfl, h⟩ := h
    obtain ⟨rfl, rfl⟩ := escB_app h
    exact ⟨rfl, rfl⟩

/-- A round's answer: a message (`2`, escaped), accept (`3`) or reject (`4`). -/
def encOutTr : Out → List UInt8
  | .msg m => 2 :: escB m
  | .acc => [3]
  | .rej => [4]

theorem encOutTr_app {o o' : Out} {r r' : List UInt8} (h : encOutTr o ++ r = encOutTr o' ++ r') :
    o = o' ∧ r = r' := by
  cases o <;> cases o' <;> simp only [encOutTr, List.cons_append, List.nil_append, List.cons.injEq] at h <;>
    simp_all
  rename_i m m'
  obtain ⟨rfl, rfl⟩ := escB_app h
  exact ⟨rfl, rfl⟩

/-- The rounds, each (`6`) its prover message escaped and its answer, then an end (`5`). -/
def encRounds : List (Msg × Out) → List UInt8
  | [] => [5]
  | p :: l => 6 :: escB p.1 ++ (encOutTr p.2 ++ encRounds l)

theorem encRounds_injective : Function.Injective encRounds
  | [], [], _ => rfl
  | [], _ :: _, h => by simp [encRounds] at h
  | _ :: _, [], h => by simp [encRounds] at h
  | (m, o) :: l, (m', o') :: l', h => by
    simp only [encRounds, List.cons_append, List.cons.injEq, true_and] at h
    obtain ⟨rfl, h₁⟩ := escB_app h
    obtain ⟨rfl, h₂⟩ := encOutTr_app h₁
    rw [encRounds_injective h₂]

/-- **The raw transcript of a history**: a `0` byte, which no JSON text starts with, then its rounds. -/
def rawTr (ms : List (Msg × Out)) : ByteArray := ⟨(0 :: encRounds ms).toArray⟩

theorem rawTr_injective : Function.Injective rawTr := by
  intro ms ms' h
  have h' := congrArg (fun b : ByteArray => b.data.toList) h
  simp only [rawTr, List.cons.injEq, true_and] at h'
  exact encRounds_injective h'

theorem rawTr_lead (ms : List (Msg × Out)) : (rawTr ms).data[0]? = some 0 := by simp [rawTr]

/-! ## The transcript at the served histories -/

/-- **The coin server's transcript at the histories the firewall serves**: `tr` at those `served` names, the raw
transcript (`rawTr`) at the others, where the deployed firewall stops the session and writes none. -/
def trServed (served : Msg → List (Msg × Out) → Bool) (tr : Msg → List (Msg × Out) → ByteArray) (x : Msg)
    (ms : List (Msg × Out)) : ByteArray :=
  if served x ms then tr x ms else rawTr ms

/-- **`trServed` is injective** where `tr` is at the served histories and starts there with `{`, which the raw
transcript's `0` never does. -/
theorem trServed_injective (served : Msg → List (Msg × Out) → Bool) (tr : Msg → List (Msg × Out) → ByteArray)
    (x : Msg) (hinj : ∀ ms ms', served x ms → served x ms' → tr x ms = tr x ms' → ms = ms')
    (hlead : ∀ ms, served x ms → (tr x ms).data[0]? = some 0x7b) : Function.Injective (trServed served tr x) := by
  intro ms ms' h
  unfold trServed at h
  cases hs : served x ms <;> cases hs' : served x ms' <;>
    simp only [hs, hs', Bool.false_eq_true, ↓reduceIte] at h
  · exact rawTr_injective h
  · have := hlead ms' hs'
    rw [← h, rawTr_lead] at this
    simp at this
  · have := hlead ms hs
    rw [h, rawTr_lead] at this
    simp at this
  · exact hinj ms ms' hs hs' h

/-! ## `KSLin2` and `BitsLin2` at `trServed` -/

/-- **`KSLin2` at the deployment's dense statements, with the coin server's transcript at the served histories**:
`ksLin2_deployed_open` at `trServed served b.tr`, with `htr` from `trServed_injective`: at every compiled statement,
`b.tr x` is injective at the histories `served` names and starts there with `{`. What stays open at the dense form is
`PartChecksB` and `PartOpensB` at `showsTr`/`cmtTr`, and `DecompB`; the readers' three facts; and those two facts of
the coin server's record. This bounds no number by itself: those premises are open. -/
theorem ksLin2_deployed_served (b : Brings) (served : Msg → List (Msg × Out) → Bool)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hinj : ∀ x st, parse x = some st → ∀ ms ms', served x ms → served x ms' → b.tr x ms = b.tr x ms' →
      ms = ms')
    (hlead : ∀ x st, parse x = some st → ∀ ms, served x ms → (b.tr x ms).data[0]? = some 0x7b) :
    let bS : Brings := { b with tr := trServed served b.tr }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bS L Bd.IVStar
    let bL : Brings := { bS with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) (cmtTr bL) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bS L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bS L Bd D bL hP hRT hRL hRP
  exact ksLin2_deployed_open bS rest wE join final w
    (fun x st hst => trServed_injective served b.tr x (hinj x st hst) (hlead x st hst)) hP hRT hRL hRP

/-- **`BitsLin2` at the deployment's dense statements, with the coin server's transcript at the served histories**:
`bitsLin2_deployed_open` at `trServed served b.tr`, with `htr` from `trServed_injective`. It reads `StrictBirthday`
and `ExpectedCR`, the named SHA-512 collision assumptions, and `ksLin2_deployed_served`'s open premises: a bound on a
number under named premises, not a proved security level. -/
theorem bitsLin2_deployed_served (b : Brings) (served : Msg → List (Msg × Out) → Bool)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (hinj : ∀ x st, parse x = some st → ∀ ms ms', served x ms → served x ms' → b.tr x ms = b.tr x ms' →
      ms = ms')
    (hlead : ∀ x st, parse x = some st → ∀ ms, served x ms → (b.tr x ms).data[0]? = some 0x7b) :
    let bS : Brings := { b with tr := trServed served b.tr }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bS L Bd.IVStar
    let bL : Brings := { bS with rd := L.readers }
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsTr bL) (cmtTr bL) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersLogColl FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bS L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bS L Bd D bL hP hRT hRL hRP
  exact bitsLin2_deployed_open bS rest wE join final w
    (fun x st hst => trServed_injective served b.tr x (hinj x st hst) (hlead x st hst)) hP hRT hRL hRP

end RecDeploy
