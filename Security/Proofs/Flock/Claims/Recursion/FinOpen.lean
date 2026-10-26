import Proofs.Flock.Claims.Recursion.TopsStaged

/-!
# The last round's pieces at the parts that commit the earlier rounds (`finOpen`), and `FinPieces` there

The bound record's last message is, per earlier round in order, that round's rows with their salts in commit order,
then the element-level last message (`Claims.Recursion.Bound.final`). Each earlier round's commit strings sit at fixed
parts and ports at `layoutBTops`: round 0's at RecCap's port `top` (`capAt`, `capHold`), a stream round's tops at their
level's caps (`capPieceAt`) and its row at the first part whose port reads it (`rowPiece`). `finOpen rest` reads each
string's opening at the part and port that commit it: the port's row at the instance (`rowHidden`), then its hm96
slot's salt (`hmSalt`) (`openSegs`). The element-level last message follows at the parts `rest` names, a parameter,
each index one of the list's parts.

`finPieces_finOpen` proves `FinPieces (finOpen rest)` at every shape: round 0's piece is RecCap's part, the list's last
(`capAt_readX`), and every piece is a part's (`roundOpen_lt`, as `topsPiecesAt_held`). `ksLin2_deployed_open` and
`bitsLin2_deployed_open` are `ksLin2_deployed_staged` and `bitsLin2_deployed_staged` at `finOpen rest`, with
`FinPieces` dropped.

`FinPieces` is structural. That these pieces read the last message's bytes (the rows and salts at the offsets `final`
reads them, and `rest`'s at the element-level message's) is what `PartChecksBV` and `DecompB` take, unchanged here.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-! ## The openings -/

/-- **A commit string's opening** at port `q` (named `nm`) at instance `n`: the row the port reads there
(`rowHidden`), then its hm96 slot's salt (`hmSalt`). -/
def openSegs (c : Circuit) (nm : String) (q n : ℕ) : List (Msg × List HmNets.Lin) :=
  [(([] : Msg), ((rowHidden c nm n).getD []).map HmNets.Lin.var), (([] : Msg), hmSalt c q n)]

/-- **A cap's openings** at part `s`: at port `top`, the openings of its instances `ns`, in order. At a part with no
circuit or no port `top` they are read at the default circuit and port 0 instead, as `topPiece` reads its strings, so
every cap opens one row and salt an instance wherever it sits. -/
def capOpen (r : XR) (s : ℕ) (ns : List ℕ) : List (Msg × List HmNets.Lin) :=
  let c := (r.w.parts[s]?.bind partCircuit).getD default
  ns.flatMap (openSegs c "top" ((capPort c).getD 0))

/-- **The opening of the row stream round `j` absorbs**, at the part `rowPiece` holds it at: the first part with a held
row of round `j` (`heldRows`). None where no part has one. -/
def rowOpen (r : XR) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  ((List.range r.w.parts.length).findSome? fun s => ((heldRows r s).find? (·.1 == j)).bind fun h =>
    (r.w.parts[s]?.bind partCircuit).map fun c => (s, openSegs c h.2.1 h.2.2.1 h.2.2.2)).toList

/-- The openings of the `e`-th top of stream round `j`, at its cap's part (`capPieceAt`'s): rep `j / R`'s `n`
instances, `rep · n + t`. -/
def capOpenAt (r : XR) (rs : Array RecVStar.SchedRound) (j e : ℕ) : ℕ × List (Msg × List HmNets.Lin) :=
  (capsFrom r rs + topsBefore rs (j % rs.size) + e,
    capOpen r (capsFrom r rs + topsBefore rs (j % rs.size) + e)
      ((List.range (((rs.getD (j % rs.size) default).2.1).getD e 0)).map fun t =>
        j / rs.size * ((rs.getD (j % rs.size) default).2.1).getD e 0 + t))

/-- **Stream round `j`'s openings**, in commit order: each top's at its cap, then the row's. -/
def roundOpen (r : XR) (rs : Array RecVStar.SchedRound) (j : ℕ) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  (List.range ((rs.getD (j % rs.size) default).2.1).size).map (capOpenAt r rs j) ++ rowOpen r j

/-- **Every earlier round's openings**, round by round: round 0's at RecCap's part, over its `N` instances, then each
stream round's. -/
def openPieces (r : XR) : List (ℕ × List (Msg × List HmNets.Lin)) :=
  (capAt r).toList.map (fun p => (p.1, capOpen r p.1 (List.range p.2))) ++
    ((repSched r.shape).map fun rs => (List.range (2 * rs.size)).flatMap (roundOpen r rs)).getD []

/-- **The last round's pieces**: every earlier round's openings (`openPieces`), then the element-level last message's
pieces at the parts `rest` names. None where `readX` reads no statement. -/
def finOpen (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg) :
    List (ℕ × List (Msg × List HmNets.Lin)) :=
  match readX x with
  | some r => openPieces r ++ (rest r).map fun p => (p.1.val, p.2)
  | none => []

/-! ## `FinPieces` at `finOpen` -/

/-- The row's opening is a part's. -/
theorem rowOpen_lt {r : XR} {j : ℕ} {p : ℕ × List (Msg × List HmNets.Lin)} (hp : p ∈ rowOpen r j) :
    p.1 < r.w.parts.length := by
  unfold rowOpen at hp
  simp only [Option.mem_toList] at hp
  obtain ⟨a, ha, hfa⟩ := List.exists_of_findSome?_eq_some hp
  rw [Option.bind_eq_some_iff] at hfa
  obtain ⟨h, -, hfa⟩ := hfa
  rw [Option.map_eq_some_iff] at hfa
  obtain ⟨c, -, rfl⟩ := hfa
  exact List.mem_range.1 ha

/-- **A stream round's openings are parts'**, where the later caps come before V\*'s last statement. -/
theorem roundOpen_lt {r : XR} {rs : Array RecVStar.SchedRound} {j : ℕ} (hpos : 0 < rs.size)
    (hlen : (RecVStar.topSizes rs).length ≤ r.w.parts.length - 1) {p : ℕ × List (Msg × List HmNets.Lin)}
    (hp : p ∈ roundOpen r rs j) : p.1 < r.w.parts.length := by
  unfold roundOpen at hp
  rcases List.mem_append.1 hp with hp | hp
  · obtain ⟨e, he', rfl⟩ := List.mem_map.1 hp
    have hlt := topsBefore_lt (Nat.mod_lt j hpos) (List.mem_range.1 he')
    show capsFrom r rs + topsBefore rs (j % rs.size) + e < r.w.parts.length
    unfold capsFrom
    omega
  · exact rowOpen_lt hp

/-- **`FinPieces` at `finOpen`**, at every shape: wherever a compiled statement parses, `readX` reads it
(`readX_of_parse`), round 0's opening is at RecCap's part (`capAt_readX`), and every piece is a part's: RecCap's the
list's last, each stream round's at the list's parts (`roundOpen_lt`, `topsLen_readX`), `rest`'s by its type. -/
theorem finPieces_finOpen (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) : FinPieces (finOpen rest) shape := by
  intro x st hst _
  obtain ⟨r, hr, -⟩ := readX_of_parse hst
  obtain ⟨N, hcap, hn0⟩ := capAt_readX hr
  have hnS : nS x = r.w.parts.length := by simp [nS, hr]
  have hf : finOpen rest x = openPieces r ++ (rest r).map fun p => (p.1.val, p.2) := by simp [finOpen, hr]
  rw [hf]
  refine ⟨by simp [openPieces, hcap], fun p hp => ?_⟩
  rw [hnS]
  rcases List.mem_append.1 hp with hp | hp
  · unfold openPieces at hp
    rcases List.mem_append.1 hp with hp | hp
    · simp only [hcap, Option.toList_some, List.map_cons, List.map_nil, List.mem_singleton] at hp
      subst hp
      show nS x - 1 < r.w.parts.length
      omega
    · cases hrs : repSched r.shape with
      | none => simp [hrs] at hp
      | some rs =>
        simp only [hrs, Option.map_some, Option.getD_some, List.mem_flatMap, List.mem_range] at hp
        obtain ⟨j, hj, hp⟩ := hp
        exact roundOpen_lt (by omega) (topsLen_readX hr hrs) hp
  · obtain ⟨q, -, rfl⟩ := List.mem_map.1 hp
    exact q.1.isLt

/-! ## `KSLin2` and `BitsLin2` at `finOpen` -/

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)`**: `ksLin2_deployed_staged` with
`FinPieces` from `finPieces_finOpen`. The last round's pieces are every earlier round's openings at the parts that
commit them, then the element-level last message's at the parts `rest` names. What stays open at the dense form is
`PartChecksBV` and `PartOpensB` at `showsPart`/`cmtTr`, and `DecompB`; then the readers' two facts and `untr` reading the
coin server's transcript back (`ReadsBackB`). This bounds no number by itself: those premises are open. -/
theorem ksLin2_deployed_open (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact ksLin2_deployed_staged b (finOpen rest) wE join final w untr hrb
    (fun shape hsh => ⟨finPieces_finOpen rest shape, hP shape hsh⟩) hRT

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)`**: `bitsLin2_deployed_staged`
with `FinPieces` from `finPieces_finOpen`. It reads `StrictBirthday` and `ExpectedCR`, the named SHA-512 collision
assumptions, and `ksLin2_deployed_open`'s open premises: a bound on a number under named premises, not a proved
security level. -/
theorem bitsLin2_deployed_open (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (untr : TrReader) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) untr →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      ∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, Claims.Recursion.VStarB.PartChecksBV FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) chk s) ∧
        (∀ s, Claims.Recursion.VStarB.PartOpensB FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation)
          L shape (showsPart untr b) (cmtTr untr) s) ∧
        Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape chk) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hP hRT
  exact bitsLin2_deployed_staged b (finOpen rest) wE join final w untr hrb
    (fun shape hsh => ⟨finPieces_finOpen rest shape, hP shape hsh⟩) hRT

end RecDeploy
