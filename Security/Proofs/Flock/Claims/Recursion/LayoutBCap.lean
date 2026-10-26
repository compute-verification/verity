import Proofs.Flock.Claims.Recursion.LayoutBDeployed
import Proofs.Flock.Claims.Recursion.KSLin2PlainClosed
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.C3.Reps.AtList

/-!
# V*'s bound layout with round 0 at RecCap (`layoutBCap`), and `HeldB` there

`HeldB` is false at `layoutBDeployed` (`heldB_layoutBDeployed_iff`) because no part held round 0, the level-0 cap of
`root_B`. RecCap (`VStar.capUnit`, `rec-cap/n<N>`) now holds it: the last statement of V*'s list at every inner shape
the setup accepts (`FlockC3.Reps.list_ok`), over `N = 2^cap₀` instances, instance `n` reading node `n` of `rec-top-L0`
as its port `top`, through its hm96 slot. `layoutBCap` is `layoutBDeployed` with that round held:

* **Round 0** (`capHold`): held whole by RecCap's part (`capAt`, the list's last index), as the commit strings of its
  `N` instances in order, each the `b ‖ c` of port `top`'s hm96 slot at that instance (`hmCommit`), with each slot's two
  hm96 calls (`capCalls`, `hmCalls` at port `top`).
* **A stream round with no tops**: as `layoutBDeployed`. Its copies of level 0's top are round 0's strings again, not
  new ones (`nAtSched`), so the round's one string is `F_k`'s where it absorbs a message (held off `heldRows`), and it
  carries the empty message, fixed, where it absorbs none.
* **A stream round that commits tops** (a Ligerito level's round, `cap (i + 1)` nodes of level `i + 1`'s top): held
  as `topsAt` says, a parameter.
* **The last round**: assembled from the pieces `fin` gives, a parameter, at the byte offsets a code-side `final` fixes.

`heldB_layoutBCap` proves `HeldB` there from named facts. RecCap discharges round 0 outright: the list ends with it
and the parts number as the list's statements (`capAt_readX`), so the one fact left at round 0 is that its part's
circuit has its port `top` (`CapPorts`), proved at every shape (`capPorts`, `CapPorts.lean`, from `readX`'s
`capPorted`). The rest:

* `SchedMatches` (`DeployLayout.lean`): the shape's rep schedule is the inner verifier's.
* `RowsCover`: each absorbing stream round with no tops is some part's, through a port under its `rec-c<k>`
  (`heldRows`), as `PortsCover` is for `partHolds`. Proved at every shape (`rowsCover`, `RowsCover.lean`).
* `TopsHeld topsAt`: each stream round that commits tops is held whole by some part. No part holds one whole: the
  round's message is its tops' strings and then one row, and the row is another part's. At `topsAt` none, `HeldB`
  fails at every statement of the shape with such a round (`not_heldB_layoutBCap_none`), and level 0's round of each
  rep commits `cap 1` nodes (`RecVStar.repRounds`). `LayoutBTops.lean` holds those rounds as pieces instead: each top
  by its level's RecCap (`rec-cap/n<N>-l<l>`), the row by its part.
* `FinPieces fin`: the last round has pieces, each a part's. It holds at `finOpen` (`finPieces_finOpen`,
  `FinOpen.lean`): every earlier round's openings at the parts that commit them, then the element-level last message's.

`ksLin2_deployed_cap` is `ksLin2_deployed_plain_closed` at `layoutBCap`, with `HeldB` replaced by those five facts.
`TopsHeld` asks for a part that holds a tops round whole, which none does; `ksLin2_deployed_tops` (`LayoutBTops.lean`)
takes the same facts but that one.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-! ## RecCap's part -/

/-- **RecCap in V\*'s list** at the verifier's own inner shape: its index, the list's last, and its node count `N`. -/
def capAt (r : XR) : Option (ℕ × ℕ) :=
  (RecVStar.statements r.shape).toOption.bind fun ss =>
    match ss.back? with
    | some (VStar.Stmt.cap 0 N) => some (ss.size - 1, N)
    | _ => none

/-- RecCap's port `top` among a circuit's ports. -/
def capPort (c : Circuit) : Option ℕ := c.ports.findIdx? (·.name == "top")

/-- **What RecCap's part holds of round 0**: at RecCap's index, the commit strings of its `N` instances in order, each
port `top`'s hm96 slot's output `b ‖ c` at that instance (`hmCommit`). None at any other part. -/
def capHold (r : XR) (s : ℕ) : Option (List (Msg × List HmNets.Lin)) :=
  (capAt r).bind fun p =>
    if s = p.1 then
      (r.w.parts[s]?.bind partCircuit).bind fun c => (capPort c).map fun q =>
        (List.range p.2).map fun n => (([] : Msg), hmCommit c q n)
    else none

/-- **RecCap's part's hm96 calls**: each instance's two at port `top` (`hmCalls`). None at any other part. -/
def capCalls (r : XR) (s : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  ((capAt r).bind fun p =>
    if s = p.1 then
      (r.w.parts[s]?.bind partCircuit).bind fun c => (capPort c).map fun q =>
        (List.range p.2).flatMap fun n => hmCalls c "top" q n
    else none).getD []

/-- What `readX` read of V*'s list: the list builds at the verifier's own inner shape, and the parts number as its
statements. -/
theorem readX_statements {x : Msg} {r : XR} (h : readX x = some r) :
    ∃ vs, RecVStar.statements r.shape = .ok vs ∧ r.w.parts.length = vs.size := by
  replace h := readXPre_of_readX h
  unfold readXPre at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨w, -, ri, -, sh, -, pa, -, dims, -, vs, hvs, h⟩ := h
  split_ifs at h with hl
  · simp at h
  have h' : (⟨w, ri, pa, sh, dims⟩ : XR) = r := by simpa using h
  subst h'
  exact ⟨vs, toOption_eq_some hvs, not_not.1 (not_or.1 hl).1⟩

/-- **RecCap is the last part** wherever `readX` reads the statement: V*'s list ends with RecCap
(`FlockC3.Reps.list_ok`), and the parts number as the list's statements. -/
theorem capAt_readX {x : Msg} {r : XR} (h : readX x = some r) :
    ∃ N, capAt r = some (nS x - 1, N) ∧ 0 < nS x := by
  obtain ⟨vs, hvs, hlen⟩ := readX_statements h
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, _, _, K, N, -, -, rfl⟩ := FlockC3.Reps.list_ok hvs
  have hn : nS x = r.w.parts.length := by simp [nS, h]
  refine ⟨N, ?_, ?_⟩
  · rw [hn, hlen]
    simp [capAt, hvs, Except.toOption]
  · rw [hn, hlen]
    simp

/-! ## The layout -/

/-- The last round's pieces, `fin`'s, and none at any other round. -/
def lastPieces (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (x : Msg) (i : ℕ) :
    List (ℕ × List (Msg × List HmNets.Lin)) :=
  if i + 1 = layoutOf.rounds x then fin x else []

/-- **V\*'s bound layout with round 0 at RecCap**: `layoutBDeployed`'s rounds, held rows and calls, with round 0 held by
RecCap's part (`capHold`, its calls `capCalls`), each stream round that commits tops held as `topsAt` says, and the last
round assembled from `fin`'s pieces. A stream round with no tops is fixed to the empty message where it absorbs none
and held off `heldRows` where it absorbs one: its copies of level 0's top add no string (`nAtSched`). -/
def layoutBCap (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) : Claims.Recursion.VStarB.LayoutB where
  rounds := layoutOf.rounds
  msgAt x s i := match i with
    | 0 => (readX x).bind fun r => capHold r s
    | j + 1 => (streamRound x j).bind fun sr =>
      if sr.2.1.isEmpty then
        if sr.2.2.2 then
          (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
            ((heldRows r s).find? (·.1 == j)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)]
        else none
      else topsAt x s (j + 1)
  shaAt x s := layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capCalls r s).getD []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr => if sr.2.1.isEmpty && !sr.2.2.2 then some [] else none
  pieceAt x i := lastPieces fin x i

section Layout

variable (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
  (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))

theorem layoutBCap_msgAt_zero (x : Msg) (s : ℕ) :
    (layoutBCap topsAt fin).msgAt x s 0 = (readX x).bind fun r => capHold r s := rfl

theorem layoutBCap_fixedAt_zero (x : Msg) : (layoutBCap topsAt fin).fixedAt x 0 = none := rfl

end Layout

/-- A part's held rows exist only where its circuit parses. -/
theorem heldRows_circuit {r : XR} {s : ℕ} {h : ℕ × String × ℕ × ℕ} (hm : h ∈ heldRows r s) :
    ∃ c, r.w.parts[s]?.bind partCircuit = some c := by
  cases h2 : r.w.parts[s]?.bind partCircuit with
  | some c => exact ⟨c, rfl⟩
  | none =>
    exfalso
    unfold heldRows at hm
    split at hm <;> simp_all

/-! ## `HeldB` at `layoutBCap` -/

/-- **RecCap's part has its port `top`** at the compiled statements of the shape: the last part's circuit, as the setup
parses it, has the port RecCap reads node `n` through at instance `n`. -/
def CapPorts (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ r, readX x = some r →
    ∃ c, r.w.parts[nS x - 1]?.bind partCircuit = some c ∧ (capPort c).isSome

/-- **The registered ports cover each absorbing stream round with no tops** at the compiled statements of the shape:
some part's port under its `rec-c<k>` reads that rep's row (`heldRows`), as `PortsCover` for `partHolds`. -/
def RowsCover (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ r j sr, readX x = some r → streamRound x j = some sr →
    sr.2.1.isEmpty = true → sr.2.2.2 = true → ∃ s < nS x, ∃ h ∈ heldRows r s, h.1 = j

/-- **Each stream round that commits tops is held whole** by some part, as `topsAt` says. No part holds one whole (its
row is another part's), and at `topsAt` none `HeldB` fails (`not_heldB_layoutBCap_none`); `layoutBTops` holds those
rounds as pieces. -/
def TopsHeld (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop) :
    Prop :=
  ∀ x st, parse x = some st → shape st.form → ∀ j sr, streamRound x j = some sr → sr.2.1.isEmpty = false →
    ∃ s < nS x, (topsAt x s (j + 1)).isSome

/-- **The last round has pieces, each a part's**, at the compiled statements of the shape. It holds at `finOpen`
(`finPieces_finOpen`): every earlier round's openings at the parts that commit them, then the element-level last
message's at parts a parameter names. -/
def FinPieces (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop) : Prop :=
  ∀ x st, parse x = some st → shape st.form → fin x ≠ [] ∧ ∀ p ∈ fin x, p.1 < nS x

/-- **`HeldB` at `layoutBCap`**, at every inner verifier of the deployed plain verifier's rounds, from the named facts:
round 0 is RecCap's part's (`capAt_readX`, `CapPorts`); a stream round with no tops is fixed where it absorbs no message
and held where it absorbs one (`RowsCover`); one that commits tops is held (`TopsHeld`); the last round is assembled
(`FinPieces`) and held whole by no part. -/
theorem heldB_layoutBCap {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop)
    (hM : SchedMatches shape) (hP : CapPorts shape) (hR : RowsCover shape) (hT : TopsHeld topsAt shape)
    (hF : FinPieces fin shape) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBCap topsAt fin) shape := by
  intro x st hst hf
  obtain ⟨r, rs, hr, hrs, hK⟩ := hM x st hst hf
  have hkx : I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) = 2 + 2 * rs.size := by
    rw [hk]
    exact hK
  have hL : layoutOf.rounds x = 2 + 2 * rs.size := hK
  have hsr : ∀ j, j < 2 * rs.size → streamRound x j = some (rs.getD (j % rs.size) default) := by
    intro j hj
    simp only [streamRound, hr, Option.bind_some, hrs, hj, ↓reduceIte]
  have hsr' : streamRound x (2 * rs.size) = none := by
    simp [streamRound, hr, hrs]
  refine ⟨hkx.trans hL.symm, fun i hi => ?_, fun s => ?_⟩
  · rw [hkx] at hi
    cases i with
    | zero =>
      obtain ⟨N, hcap, hn0⟩ := capAt_readX hr
      obtain ⟨c, hc, hq⟩ := hP x st hst hf r hr
      obtain ⟨q, hq'⟩ := Option.isSome_iff_exists.1 hq
      refine ⟨Or.inr (Or.inl ⟨nS x - 1, show nS x - 1 < nS x by omega, ?_⟩), fun h0 => ?_⟩
      · rw [layoutBCap_msgAt_zero]
        simp [hr, capHold, hcap, hc, hq']
      · rw [layoutBCap_fixedAt_zero] at h0
        simp at h0
    | succ j =>
      rcases Nat.lt_or_eq_of_le (show j ≤ 2 * rs.size by omega) with hj | rfl
      · obtain ⟨sr, hs⟩ : ∃ sr, streamRound x j = some sr := ⟨_, hsr j hj⟩
        have hpc : (layoutBCap topsAt fin).pieceAt x (j + 1) = [] := by
          have hc : ¬ (j + 1 + 1 = layoutOf.rounds x) := by
            rw [hL]
            omega
          simp only [layoutBCap, lastPieces, if_neg hc]
        refine ⟨?_, fun _ => hpc⟩
        cases ht : sr.2.1.isEmpty with
        | false =>
          have ht' : ¬ sr.2.1 = #[] := by simpa using ht
          right; left
          obtain ⟨s, hs', hsome⟩ := hT x st hst hf j sr hs ht
          refine ⟨s, hs', ?_⟩
          have hm : (layoutBCap topsAt fin).msgAt x s (j + 1) = topsAt x s (j + 1) := by
            simp [layoutBCap, hs, ht']
          rw [hm]
          exact hsome
        | true =>
          have ht' : sr.2.1 = #[] := by simpa using ht
          cases ha : sr.2.2.2 with
          | false =>
            left
            simp [layoutBCap, hs, ht', ha]
          | true =>
            right; left
            obtain ⟨s, hs', h', hmem, hj'⟩ := hR x st hst hf r j sr hr hs ht ha
            obtain ⟨c, hc⟩ := heldRows_circuit hmem
            refine ⟨s, hs', ?_⟩
            cases hfind : (heldRows r s).find? (·.1 == j) with
            | none => exact absurd (List.find?_eq_none.1 hfind _ hmem) (by simp [hj'])
            | some p => simp [layoutBCap, hs, ht', ha, hr, hc, hfind]
      · refine ⟨Or.inr (Or.inr ?_), fun h0 => ?_⟩
        · have hc : 2 * rs.size + 1 + 1 = layoutOf.rounds x := by
            rw [hL]
            omega
          simp only [layoutBCap, lastPieces, if_pos hc]
          exact hF x st hst hf
        · exfalso
          simp [layoutBCap, hsr'] at h0
  · rw [hkx, show 2 + 2 * rs.size - 1 = 2 * rs.size + 1 by omega]
    simp [layoutBCap, hsr']

/-- **`HeldB` fails at `layoutBCap` where no part holds the tops** (`topsAt` none), wherever a compiled statement of
the shape parses with a stream round that commits tops: that round is neither fixed (it has tops), nor held, nor the
last (it is a stream round), so it has no pieces. Each rep's level-0 round commits `cap 1` nodes of level 1's top
(`RecVStar.repRounds`). -/
theorem not_heldB_layoutBCap_none {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop)
    (hM : SchedMatches shape) {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) (hf : shape st.form)
    {j : ℕ} {sr : RecVStar.SchedRound} (hs : streamRound x j = some sr) (ht : sr.2.1.isEmpty = false) :
    ¬ Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBCap (fun _ _ _ => none) fin) shape := by
  intro hH
  obtain ⟨r, rs, hr, hrs, hK⟩ := hM x st hst hf
  have hj : j < 2 * rs.size := by
    by_contra hj
    simp [streamRound, hr, hrs, hj] at hs
  have ht' : ¬ sr.2.1 = #[] := by simpa using ht
  obtain ⟨-, hcl, -⟩ := hH x st hst hf
  have hkx : I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) = 2 + 2 * rs.size := by
    rw [hk]
    exact hK
  rcases (hcl (j + 1) (by rw [hkx]; omega)).1 with h0 | ⟨s, -, h0⟩ | ⟨h0, -⟩
  · simp [layoutBCap, hs, ht'] at h0
  · simp [layoutBCap, hs, ht'] at h0
  · apply h0
    have hc : ¬ (j + 1 + 1 = layoutOf.rounds x) := by
      show ¬ (j + 1 + 1 = (FlockDeploy.V false).k (xIOf x))
      rw [hK]
      omega
    simp only [layoutBCap, lastPieces, if_neg hc]

/-! ## `KSLin2` at `layoutBCap` -/

/-- **`KSLin2` at the deployment's dense statements, at `layoutBCap`, with `HeldB` restated**:
`ksLin2_deployed_plain_closed` at the layout with round 0 at RecCap, where at the dense form `HeldB` comes from
`heldB_layoutBCap`. What stays at the dense form is the schedule (`SchedMatches`), RecCap's port (`CapPorts`), the rows'
ports (`RowsCover`), the tops (`TopsHeld`, false at every `topsAt` that holds no such round:
`not_heldB_layoutBCap_none`), the last round's pieces (`FinPieces`, which holds at `finOpen`: `finPieces_finOpen`), and
`VstarShowsB`, `PartChecksBV`, `PartOpensB` and `DecompB`; then `RunPairs` and the readers' three facts. -/
theorem ksLin2_deployed_cap (b : Brings) (topsAt : Msg → ℕ → ℕ → Option (List (Msg × List HmNets.Lin)))
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBCap topsAt fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense],
        SchedMatches shape ∧ CapPorts shape ∧ RowsCover shape ∧ TopsHeld topsAt shape ∧ FinPieces fin shape ∧
        ∃ (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
          (cmt : Msg → ℕ → Msg → Msg),
          (∀ decI, Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape
            decC shows cmt) ∧
          (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt chk s) ∧
          (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
            L shape shows cmt s) ∧
          Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    ReadersRunPlayed FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hP hRT hRP
  refine ksLin2_deployed_plain_closed b L (Claims.Recursion.VStarB.LayoutB.readsRows_of_rowIn fun _ _ => rfl) wE join
    final w ?_ hRT hRP
  obtain ⟨decC, hRun, hS⟩ := hP
  refine ⟨decC, hRun, fun shape hsh => ?_⟩
  obtain ⟨hM, hC, hRw, hTp, hF, chk, shows, cmt, hV, hPC, hPO, hD⟩ := hS shape hsh
  exact ⟨chk, shows, cmt,
    fun decI => heldB_layoutBCap _ _ _ _ (fun _ => rfl) topsAt fin shape hM hC hRw hTp hF, hV, hPC, hPO, hD⟩

end RecDeploy
