import Proofs.Flock.Claims.Recursion.LayoutBCap

/-!
# V*'s bound layout with every level's tops at its RecCap (`layoutBTops`), and `HeldB` there

`layoutBCap` holds round 0 at RecCap and leaves each stream round that commits tops to a parameter, `topsAt`, which
asks for a part holding the round whole (`TopsHeld`). None does: such a round's message is the salted tops it commits,
one string a node, then one row where it absorbs a message (`nAtSched`), and the row is another part's. V*'s list now
holds every later level's tops at fixed rows: the later levels' caps (`VStar.levelCapStatements`, `rec-cap/n<N>-l<l>`),
the list's statements just before level 0's RecCap, the `g`-th over both reps' nodes of the reps' schedule's `g`-th
top (`RecVStar.checkTopsListed`), instance `rep · n + t` reading node `t` of rep `rep`'s top as its port `top`.
`layoutBTops` is `layoutBCap` with each tops round assembled from pieces instead:

* **The tops** (`topsPiecesAt`): the round's `e`-th top, the schedule's `g`-th, at the list's `g`-th later cap
  (`capsFrom`, `topsBefore`), as the commit strings of the rep's `n` instances in order, each the `b ‖ c` of port
  `top`'s hm96 slot (`topPiece`).
* **The row** (`rowPiece`): where the round absorbs a message, the first part whose port under its `rec-c<k>` reads that
  rep's row (`heldRows`), as that port's hm96 slot's output.
* Every cap's hm96 calls at port `top` (`capsCalls`), level 0's included.

Everything else is `layoutBCap`'s. `heldB_layoutBTops` proves `HeldB` there from `SchedMatches`, `CapPorts`,
`RowsCover` and `FinPieces`: a tops round's pieces exist and each is a part's (`topsPieces_held`), since the list's
later caps are as many as the schedule's tops (`topsListed_size`) and come before level 0's RecCap, the list's last
(`FlockC3.Reps.list_ok`). So `TopsHeld` is gone. `ksLin2_deployed_tops` is `ksLin2_deployed_plain_closed` at
`layoutBTops`, with `HeldB` replaced by those four facts. `RowsCover` holds at every shape (`rowsCover`,
`RowsCover.lean`, from the verifier's `RecVStar.checkRoundsListed` and `portsNamed`), and `ksLin2_deployed_rows` drops
it; `CapPorts` holds at every shape too (`capPorts`, `CapPorts.lean`, from `capPorted`), and `ksLin2_deployed_caps`
drops it.

`HeldB` is structural: it asks that every round is fixed, held or assembled from pieces of the parts. That the pieces
read the round's bytes (a top's strings at its cap's port, the row at its part's) is what `PartChecksBV`,
`PartOpensB` and `DecompB` take, unchanged here.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-! ## The later levels' caps -/

/-- The tops the schedule `rs` commits before its round `k`, one entry a top. -/
def topsBefore (rs : Array RecVStar.SchedRound) (k : ℕ) : ℕ := ((rs.toList.take k).flatMap (·.2.1.toList)).length

/-- **Where the later levels' caps start in V\*'s list**: its last `1 + (topSizes rs).length` statements are the caps,
level 0's last. -/
def capsFrom (r : XR) (rs : Array RecVStar.SchedRound) : ℕ := r.w.parts.length - 1 - (RecVStar.topSizes rs).length

/-- **A top as its cap's part holds it**: at part `s`, the commit strings of rep `rep`'s `n` instances (`rep · n + t`)
in order, each port `top`'s hm96 slot's output `b ‖ c`. At a part with no circuit or no port `top` it is read at the
default circuit and port 0 instead, so every top's piece is `128 · n` bytes (`pieceLen_topPiece`) wherever it sits; a
staged cap has both (`RecCoins.checkCap`). -/
def topPiece (r : XR) (s n rep : ℕ) : List (Msg × List HmNets.Lin) :=
  let c := (r.w.parts[s]?.bind partCircuit).getD default
  (List.range n).map fun t => (([] : Msg), hmCommit c ((capPort c).getD 0) (rep * n + t))

/-- **The row stream round `j` absorbs**: the first part with a held row of round `j` (`heldRows`), as that port's hm96
slot's output. None where no part has one. -/
def rowPiece (r : XR) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  ((List.range r.w.parts.length).findSome? fun s => ((heldRows r s).find? (·.1 == j)).bind fun h =>
    (r.w.parts[s]?.bind partCircuit).map fun c => (s, [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)])).toList

/-- The `e`-th top of stream round `j`, at its cap's part (`capsFrom`, `topsBefore`). -/
def capPieceAt (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) : ℕ × List (Msg × List HmNets.Lin) :=
  (capsFrom r rs + topsBefore rs (j % rs.size) + e,
    topPiece r (capsFrom r rs + topsBefore rs (j % rs.size) + e) (((rs.getD (j % rs.size) default).2.1).getD e 0)
      (j / rs.size))

/-- **Stream round `j`'s pieces at a round that commits tops**: each top at its cap's part, in order, then the row it
absorbs. None at a round with no tops. -/
def topsPiecesAt (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  if ((rs.getD (j % rs.size) default).2.1).isEmpty then []
  else (List.range ((rs.getD (j % rs.size) default).2.1).size).map (capPieceAt r rs j) ++ rowPiece r j

/-- `topsPiecesAt` at the compiled statement's read and schedule. -/
def topsPieces (x : Msg) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  ((readX x).bind fun r => (repSched r.shape).map fun rs => topsPiecesAt r rs j).getD []

/-- **Every cap's hm96 calls**: at a part whose statement is a RecCap (`.cap l N`, any level), each of its `N`
instances' two at port `top` (`hmCalls`). None at any other part. -/
def capsCalls (r : XR) (s : ℕ) : List (Msg × List HmNets.Lin × List HmNets.Lin) :=
  (((RecVStar.statements r.shape).toOption.bind (·[s]?)).bind fun
    | .cap _ N => (r.w.parts[s]?.bind partCircuit).bind fun c => (capPort c).map fun q =>
        (List.range N).flatMap fun n => hmCalls c "top" q n
    | _ => none).getD []

/-- Round `i`'s pieces: the last round's, `fin`'s; a stream round's, its tops' and row's where it commits tops. -/
def stepPieces (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (x : Msg) (i : ℕ) :
    List (ℕ × List (Msg × List HmNets.Lin)) :=
  if i + 1 = layoutOf.rounds x then fin x else if i = 0 then [] else topsPieces x (i - 1)

/-- **V\*'s bound layout with every level's tops at its RecCap**: `layoutBCap`'s round 0 (RecCap's part, `capHold`),
its stream rounds with no tops (fixed where they absorb no message, held off `heldRows` where they absorb one) and its
last round (`fin`'s pieces); each stream round that commits tops assembled from its tops at their caps and its row
(`topsPieces`); and every cap's calls (`capsCalls`). -/
def layoutBTops (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) : Claims.Recursion.VStarB.LayoutB where
  rounds := layoutOf.rounds
  msgAt x s i := match i with
    | 0 => (readX x).bind fun r => capHold r s
    | j + 1 => (streamRound x j).bind fun sr =>
      if sr.2.1.isEmpty then
        if sr.2.2.2 then
          (readX x).bind fun r => (r.w.parts[s]?.bind partCircuit).bind fun c =>
            ((heldRows r s).find? (·.1 == j)).map fun h => [(([] : Msg), hmCommit c h.2.2.1 h.2.2.2)]
        else none
      else none
  shaAt x s := layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capsCalls r s).getD []
  fixedAt x i := match i with
    | 0 => none
    | j + 1 => (streamRound x j).bind fun sr => if sr.2.1.isEmpty && !sr.2.2.2 then some [] else none
  pieceAt x i := stepPieces fin x i

/-! ## What the parts record

Every call a part records at `layoutBTops` is an hm96 salt or row call, whose input starts with the salt's tag
`verity/hm96-sha512/salt/v1` or a row's `verity/sha512-row/v1`, byte 7 `h` or `s` (`calls_layoutBTops_tagged`). The
plain setup's statement digest, `SHA-512(h)` with `h` starting with the statement's tag (`Stmt.setupHWith`), reads
`verity/flock-circuit` at the inner tags, byte 7 `f`, so no part records it (`calls_layoutBTops_not_stmt`). Every
deciding step of the plain verifier makes that call first (`next_decide_calls`), so a premise asking the inner run's
calls to be recorded ones, or a collision among them, fails at every honest accepted run. -/

/-- **A tagged hm96 input**: byte 7 is `h` (104), as in the salt's `verity/hm96-sha512/salt/v1`, or `s` (115), as in a
row's `verity/sha512-row/v1`. -/
def HmTagged (m : Msg) : Prop := m[7]? = some 104 ∨ m[7]? = some 115

theorem pad_data (b : ByteArray) (n : ℕ) :
    (Hm96.pad b n).data.toList = b.data.toList ++ List.replicate (n - min n b.size) 0 := by
  simp [Hm96.pad, ByteArray.data_append]

theorem saltPrefix_tagged :
    8 ≤ Hm96.Default512.saltPrefix.data.toList.length ∧ HmTagged Hm96.Default512.saltPrefix.data.toList := by
  rw [Hm96.Default512.saltPrefix, pad_data]
  have h : 8 ≤ Hm96.sha512.saltTag.data.toList.length ∧ Hm96.sha512.saltTag.data.toList[7]? = some 104 := by decide
  refine ⟨by simp only [List.length_append]; omega, Or.inl ?_⟩
  rw [List.getElem?_append_left (by omega)]
  exact h.2

theorem rowPrefix_tagged (words : ℕ) :
    8 ≤ (HmRow.rowPrefix words).data.toList.length ∧ HmTagged (HmRow.rowPrefix words).data.toList := by
  rw [HmRow.rowPrefix, pad_data]
  simp only [ByteArray.data_append, Array.toList_append, List.append_assoc]
  have h : 8 ≤ ("verity/sha512-row/v1".toUTF8.push 0).data.toList.length ∧
      ("verity/sha512-row/v1".toUTF8.push 0).data.toList[7]? = some 115 := by decide
  refine ⟨by simp only [List.length_append]; omega, Or.inr ?_⟩
  rw [List.getElem?_append_left (by omega)]
  exact h.2

theorem hmCalls_tagged (c : Circuit) (nm : String) {q : ℕ} (hq : q < c.ports.size) (n : ℕ) :
    ∀ p ∈ hmCalls c nm q n, 8 ≤ p.1.length ∧ HmTagged p.1 := by
  intro p hp
  simp only [hmCalls, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact saltPrefix_tagged
  · simp only [Array.getElem?_eq_getElem hq, Option.map_some, Option.getD_some]
    exact rowPrefix_tagged _

/-- A held row's port is one of its part's circuit's ports. -/
theorem heldRows_port {r : XR} {s : ℕ} {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {h : ℕ × String × ℕ × ℕ} (hh : h ∈ heldRows r s) : h.2.2.1 < c.ports.size := by
  unfold heldRows at hh
  rw [hc] at hh
  rcases hst : (RecVStar.statements r.shape).toOption.bind (fun vs => vs[s]?) with _ | st
  · simp only [hst] at hh; simp at hh
  rcases hrs : repSched r.shape with _ | rs
  · simp only [hst, hrs] at hh; simp at hh
  simp only [hst, hrs, List.mem_flatMap] at hh
  obtain ⟨q, -, hq⟩ := hh
  rcases hk : RecVStar.cIndex q.2.1 with _ | k
  · simp only [hk, List.not_mem_nil] at hq
  simp only [hk, List.mem_filterMap] at hq
  obtain ⟨row, -, hrow⟩ := hq
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hrow
  obtain ⟨rep, -, j, -, qi, hqi, rfl⟩ := hrow
  exact (Array.findIdx?_eq_some_iff_getElem.1 hqi).1

theorem capsCalls_tagged (r : XR) (s : ℕ) : ∀ p ∈ capsCalls r s, 8 ≤ p.1.length ∧ HmTagged p.1 := by
  intro p hp
  unfold capsCalls at hp
  rcases hv : (RecVStar.statements r.shape).toOption.bind (·[s]?) with _ | v
  · simp [hv] at hp
  simp only [hv, Option.bind_some] at hp
  cases v with
  | cap l N =>
    rcases hc : r.w.parts[s]?.bind partCircuit with _ | c
    · simp [hc] at hp
    rcases hq : capPort c with _ | q
    · simp [hc, hq] at hp
    simp only [hc, hq, Option.bind_some, Option.map_some, Option.getD_some, List.mem_flatMap, List.mem_range] at hp
    obtain ⟨n, -, hn⟩ := hp
    exact hmCalls_tagged c "top" (Array.findIdx?_eq_some_iff_getElem.1 hq).1 n p hn
  | _ => simp at hp

theorem shaAt_layoutBTops_tagged (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (x : Msg) (s : ℕ) :
    ∀ p ∈ (layoutBTops fin).shaAt x s, 8 ≤ p.1.length ∧ HmTagged p.1 := by
  intro p hp
  change p ∈ layoutBDeployed.shaAt x s ++ ((readX x).map fun r => capsCalls r s).getD [] at hp
  rcases hr : readX x with _ | r
  · simp [layoutBDeployed, hr] at hp
  simp only [layoutBDeployed, hr, Option.bind_some, Option.map_some, Option.getD_some, List.mem_append] at hp
  rcases hp with hp | hp
  · rcases hc : r.w.parts[s]?.bind partCircuit with _ | c
    · simp [hc] at hp
    simp only [hc, Option.map_some, Option.getD_some, List.mem_flatMap] at hp
    obtain ⟨h, hh, hp⟩ := hp
    exact hmCalls_tagged c h.2.1 (heldRows_port hc hh) h.2.2.2 p hp
  · exact capsCalls_tagged r s p hp

/-- **Every call a part records at `layoutBTops` has a tagged input**: an hm96 salt or row call's. -/
theorem calls_layoutBTops_tagged (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (x : Msg) (s : ℕ)
    (w : Claims.Flock.Wit) : ∀ e ∈ (layoutBTops fin).calls x s w, HmTagged e.1 := by
  intro e he
  simp only [Claims.Recursion.VStarB.LayoutB.calls, List.mem_map] at he
  obtain ⟨p, hp, rfl⟩ := he
  obtain ⟨hl, ht⟩ := shaAt_layoutBTops_tagged fin x s p hp
  unfold HmTagged at ht ⊢
  rwa [List.getElem?_append_left (by omega)]

/-- **No part records the plain setup's statement digest at the inner tags**: no recorded input starts with the inner
statement's tag, `verity/flock-circuit`, as `Stmt.setupHWith`'s digest input `h` does. -/
theorem calls_layoutBTops_not_stmt (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (x : Msg) (s : ℕ)
    (w : Claims.Flock.Wit) (t : Msg) :
    ∀ e ∈ (layoutBTops fin).calls x s w, e.1 ≠ RecVStar.innerTags.statement.toUTF8.data.toList ++ t := by
  intro e he heq
  have h : 8 ≤ RecVStar.innerTags.statement.toUTF8.data.toList.length ∧
      RecVStar.innerTags.statement.toUTF8.data.toList[7]? = some 102 := by decide
  have ht := calls_layoutBTops_tagged fin x s w e he
  unfold HmTagged at ht
  rw [heq, List.getElem?_append_left (by omega), h.2] at ht
  simp at ht

/-! ## The pieces' parts -/

/-- **The later caps are as many as the schedule's tops**: `checkTopsListed` compares their node counts with the
schedule's, one a top. -/
theorem topsListed_size {i : RecVStar.Inner} {K : Array VStar.Stmt} {rs : Array RecVStar.SchedRound}
    (h : RecVStar.checkTopsListed i K = .ok ()) (hrs : repSched i = some rs) :
    K.size = (RecVStar.topSizes rs).length := by
  unfold RecVStar.checkTopsListed at h
  unfold repSched at hrs
  simp only [hrs] at h
  split at h
  · cases h
  · rename_i hc
    have he : K.toList.map (·.instances) = (RecVStar.topSizes rs).map (2 * ·) := by
      simpa [bne_iff_ne] using hc
    have := congrArg List.length he
    simpa using this

/-- **A round's tops come before the schedule's last**: the `e`-th top of round `k` is the schedule's
`topsBefore rs k + e`-th, of `(topSizes rs).length`. -/
theorem topsBefore_lt {rs : Array RecVStar.SchedRound} {k e : ℕ} (hk : k < rs.size)
    (he : e < ((rs.getD k default).2.1).size) : topsBefore rs k + e < (RecVStar.topSizes rs).length := by
  unfold topsBefore RecVStar.topSizes
  have hd : rs.toList.drop k = rs.getD k default :: rs.toList.drop (k + 1) := by
    rw [List.drop_eq_getElem_cons (show k < rs.toList.length by simpa using hk)]
    simp [Array.getD, hk]
  have hs : rs.toList = rs.toList.take k ++ (rs.getD k default :: rs.toList.drop (k + 1)) := by
    rw [← hd, List.take_append_drop]
  conv_rhs => rw [hs]
  simp only [List.flatMap_append, List.flatMap_cons, List.length_append, Array.length_toList]
  omega

/-- The row piece is a part's. -/
theorem rowPiece_lt {r : XR} {j : ℕ} {p : ℕ × List (Msg × List HmNets.Lin)} (hp : p ∈ rowPiece r j) :
    p.1 < r.w.parts.length := by
  unfold rowPiece at hp
  simp only [Option.mem_toList] at hp
  obtain ⟨a, ha, hfa⟩ := List.exists_of_findSome?_eq_some hp
  rw [Option.bind_eq_some_iff] at hfa
  obtain ⟨h, -, hfa⟩ := hfa
  rw [Option.map_eq_some_iff] at hfa
  obtain ⟨c, -, rfl⟩ := hfa
  exact List.mem_range.1 ha

/-- A round with no tops has no pieces of its own. -/
theorem topsPiecesAt_empty {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ}
    (ht : ((rs.getD (j % rs.size) default).2.1).isEmpty = true) : topsPiecesAt r rs j = [] := by
  unfold topsPiecesAt
  rw [if_pos ht]

/-- **A round that commits tops is assembled from parts' pieces**: it has some, its first its first top's, and each is
a part's, where the later caps are as many as the schedule's tops and V\*'s list holds them before its last statement. -/
theorem topsPiecesAt_held {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} (hpos : 0 < rs.size)
    (ht : ((rs.getD (j % rs.size) default).2.1).isEmpty = false)
    (hlen : (RecVStar.topSizes rs).length ≤ r.w.parts.length - 1) :
    topsPiecesAt r rs j ≠ [] ∧ ∀ p ∈ topsPiecesAt r rs j, p.1 < r.w.parts.length := by
  have hn : 0 < ((rs.getD (j % rs.size) default).2.1).size :=
    Nat.pos_of_ne_zero fun h0 => by
      rw [Array.isEmpty_iff_size_eq_zero.2 h0] at ht
      cases ht
  have he : topsPiecesAt r rs j =
      (List.range ((rs.getD (j % rs.size) default).2.1).size).map (capPieceAt r rs j) ++ rowPiece r j := by
    unfold topsPiecesAt
    rw [if_neg (Bool.ne_true_of_eq_false ht)]
  rw [he]
  refine ⟨fun h => ?_, fun p hp => ?_⟩
  · have := congrArg List.length h
    simp only [List.length_append, List.length_map, List.length_range, List.length_nil] at this
    omega
  · rcases List.mem_append.1 hp with hp | hp
    · obtain ⟨e, he', rfl⟩ := List.mem_map.1 hp
      have hlt := topsBefore_lt (Nat.mod_lt j hpos) (List.mem_range.1 he')
      show capsFrom r rs + topsBefore rs (j % rs.size) + e < r.w.parts.length
      unfold capsFrom
      omega
    · exact rowPiece_lt hp

/-- **V\*'s list holds the later caps before its last statement**, wherever `readX` reads the statement: its parts
number as the list's statements, which end with the later caps and then level 0's RecCap (`FlockC3.Reps.list_ok`), as
many later caps as the schedule's tops (`topsListed_size`). -/
theorem topsLen_readX {x : Msg} {r : XR} {rs : Array RecVStar.SchedRound} (h : readX x = some r)
    (hrs : repSched r.shape = some rs) : (RecVStar.topSizes rs).length ≤ r.w.parts.length - 1 := by
  obtain ⟨vs, hvs, hlen⟩ := readX_statements h
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, _, _, K, N, -, hT, rfl⟩ := FlockC3.Reps.list_ok hvs
  rw [hlen, ← topsListed_size hT hrs]
  simp only [Array.size_append, Array.size_singleton]
  omega

/-! ## `HeldB` at `layoutBTops` -/

section Layout

variable (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin)))

theorem layoutBTops_msgAt_zero (x : Msg) (s : ℕ) :
    (layoutBTops fin).msgAt x s 0 = (readX x).bind fun r => capHold r s := rfl

theorem layoutBTops_fixedAt_zero (x : Msg) : (layoutBTops fin).fixedAt x 0 = none := rfl

end Layout

/-- **`HeldB` at `layoutBTops`**, at every inner verifier of the deployed plain verifier's rounds, from the named facts:
round 0 is RecCap's part's (`capAt_readX`, `CapPorts`); a stream round with no tops is fixed where it absorbs no message
and held where it absorbs one (`RowsCover`); one that commits tops is assembled from its tops at their caps and its row
(`topsPiecesAt_held`, from V\*'s list and the schedule alone); the last round is assembled (`FinPieces`) and held whole
by no part. -/
theorem heldB_layoutBTops {W : Type} (IV : Claims.Verifier Coin) (ivScope : Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin W) (hk : ∀ x, I.V.k x = (FlockDeploy.V false).k x)
    (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (shape : Claims.Recursion.Form → Prop)
    (hM : SchedMatches shape) (hP : CapPorts shape) (hR : RowsCover shape) (hF : FinPieces fin shape) :
    Claims.Recursion.VStarB.HeldB (deployRec IV ivScope b) I (layoutBTops fin) shape := by
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
  have htp : ∀ j, topsPieces x j = topsPiecesAt r rs j := fun j => by
    simp [topsPieces, hr, hrs]
  have hnS : nS x = r.w.parts.length := by simp [nS, hr]
  refine ⟨hkx.trans hL.symm, fun i hi => ?_, fun s => ?_⟩
  · rw [hkx] at hi
    cases i with
    | zero =>
      obtain ⟨N, hcap, hn0⟩ := capAt_readX hr
      obtain ⟨c, hc, hq⟩ := hP x st hst hf r hr
      obtain ⟨q, hq'⟩ := Option.isSome_iff_exists.1 hq
      refine ⟨Or.inr (Or.inl ⟨nS x - 1, show nS x - 1 < nS x by omega, ?_⟩), fun h0 => ?_⟩
      · rw [layoutBTops_msgAt_zero]
        simp [hr, capHold, hcap, hc, hq']
      · rw [layoutBTops_fixedAt_zero] at h0
        simp at h0
    | succ j =>
      rcases Nat.lt_or_eq_of_le (show j ≤ 2 * rs.size by omega) with hj | rfl
      · have hs := hsr j hj
        have hc : ¬ (j + 1 + 1 = layoutOf.rounds x) := by
          rw [hL]
          omega
        have hpc : (layoutBTops fin).pieceAt x (j + 1) = topsPiecesAt r rs j := by
          simp only [layoutBTops, stepPieces, if_neg hc, if_neg (Nat.add_one_ne_zero j), Nat.add_sub_cancel, htp]
        cases ht : ((rs.getD (j % rs.size) default).2.1).isEmpty with
        | false =>
          have ht' : ¬ (rs.getD (j % rs.size) default).2.1 = #[] := by simpa using ht
          refine ⟨Or.inr (Or.inr ?_), fun h0 => ?_⟩
          · rw [hpc]
            have hh := topsPiecesAt_held (r := r) (j := j) (by omega) ht (topsLen_readX hr hrs)
            rw [← hnS] at hh
            exact hh
          · exfalso
            simp only [Array.getD_eq_getD_getElem?] at ht'
            simp [layoutBTops, hs, ht'] at h0
        | true =>
          have ht' : (rs.getD (j % rs.size) default).2.1 = #[] := by simpa using ht
          refine ⟨?_, fun _ => hpc.trans (topsPiecesAt_empty ht)⟩
          cases ha : (rs.getD (j % rs.size) default).2.2.2 with
          | false =>
            left
            simp only [Array.getD_eq_getD_getElem?] at ht' ha
            simp [layoutBTops, hs, ht', ha]
          | true =>
            right; left
            obtain ⟨s, hs', h', hmem, hj'⟩ := hR x st hst hf r j _ hr hs ht ha
            obtain ⟨c, hc⟩ := heldRows_circuit hmem
            refine ⟨s, hs', ?_⟩
            cases hfind : (heldRows r s).find? (·.1 == j) with
            | none => exact absurd (List.find?_eq_none.1 hfind _ hmem) (by simp [hj'])
            | some p =>
              simp only [Array.getD_eq_getD_getElem?] at ht' ha
              simp [layoutBTops, hs, ht', ha, hr, hc, hfind]
      · refine ⟨Or.inr (Or.inr ?_), fun h0 => ?_⟩
        · have hc : 2 * rs.size + 1 + 1 = layoutOf.rounds x := by
            rw [hL]
            omega
          simp only [layoutBTops, stepPieces, if_pos hc]
          exact hF x st hst hf
        · exfalso
          simp [layoutBTops, hsr'] at h0
  · rw [hkx, show 2 + 2 * rs.size - 1 = 2 * rs.size + 1 by omega]
    simp [layoutBTops, hsr']

/-! ## `KSLin2` at `layoutBTops` -/

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops`, with `HeldB` restated**:
`ksLin2_deployed_plain_closed` at the layout with round 0 at RecCap and each tops round at its level's caps, where at
the dense form `HeldB` comes from `heldB_layoutBTops`. What stays at the dense form is the schedule (`SchedMatches`),
RecCap's port (`CapPorts`), the rows' ports (`RowsCover`), the last round's pieces (`FinPieces`, which holds at
`finOpen`: `finPieces_finOpen`), and `VstarShowsB`, `PartChecksBV`, `PartOpensB` and `DecompB`; then `RunPairs` and the
readers' three facts. `ksLin2_deployed_cap`'s `TopsHeld` is not among them. -/
theorem ksLin2_deployed_tops (b : Brings) (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops fin
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    (∃ decC : Msg → Msg → Coin,
      (∀ decI, Claims.Recursion.VStar.RunPairs FlockDeploy.sha512 D
        (Claims.Recursion.Inner.ofFlock Bd.deploy decI) decC) ∧
      ∀ shape ∈ [Claims.Recursion.Form.Dense],
        SchedMatches shape ∧ CapPorts shape ∧ RowsCover shape ∧ FinPieces fin shape ∧
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
  obtain ⟨hM, hC, hRw, hF, chk, shows, cmt, hV, hPC, hPO, hD⟩ := hS shape hsh
  exact ⟨chk, shows, cmt,
    fun decI => heldB_layoutBTops _ _ _ _ (fun _ => rfl) fin shape hM hC hRw hF, hV, hPC, hPO, hD⟩

end RecDeploy
