import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Claims.Recursion.VStarPartsSparse
import Proofs.Flock.Claims.Recursion.CapLowerPin
import Proofs.Flock.Claims.Recursion.BoundRouteKSLin2
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.C3.Reps.AtList
import VStarRecursive

/-!
# The statement the recursive component's inputs name, and its reading (piece B)

The compiled component (`VStar.Recursive.checkBlobs`) reads parameters, evidence and blobs; the model verifier
(`RecDeploy.V`) reads a statement `x`. `Names p e bs x` says `x` is the statement those inputs name:

* `x` decodes (`RecDeploy.decodeX`) to `w`, and `w`'s inner statement to Flock's plain statement `ri`
  (`FlockDeploy.decode`);
* `InnerNames`: `ri` is the inner session `innerHeld` reads: the inner circuit, public file and draw are the blobs the
  verifier's own digests name (`Own`), its partition and program the verifier's own, its statement `INNER_STATEMENT` on
  Flock's os coins at one table with no lookup table, `--stratified` at the population the verifier's partition and
  program give (`HmRow.stratifiedLaw` at `K = 1`, as the compiled `innerOk` sets it up) and none of `verify`'s other
  inputs; `w`'s inner shape is
  `innerHeld`'s; and `w`'s registration the registration blob `Own` names;
* `PartNames`: one part per part of the verifier's, part `j`'s raw statement the one `partVerdict` runs: its circuit
  the part's circuit blob, its form `PartForm` (statement `STATEMENT` under `--zk` at one table), the part's partition,
  program and registered reads, the part's own bindings (`PartX.bindings`) the verifier's bindings of that part, and
  its draw the one compiled part `j` reads off the part's record blob. The part's public file and V*'s shared rows are
  the staged ones (`Stage.pub`, `Stage.rows`), which piece C and `RowsShow` relate to the run.

`readX_of_names`: where the component's inner session is held (`innerHeld`), V*'s list at its shape has one statement
per part, Flock's own checks pass on the inner statement (`InnerOkAt`, which the compiled `innerHeld` runs as its
`innerOk`: `RefusalsOf.innerOkAt_of_compiled`) and each part's session sets up (`PartsSetUp`, at `xrOf`, which reads no `readX`), the
model reads `x` (`readXPre`, `readX`) at the component's inner shape, and `x` is dense (`DenseAt`).
-/

namespace Verity.Recursive

open Claims (Msg Out)
open Flock
open RecDeploy
open FlockDeploy (jsonOf)
open Lean (Json)

/-- The coins the compiled part's session is set up on: the `--zk` keyed coin tree of one table. -/
def partCoins (tags0 : Tags) : Flock.Coins := .tree (Zk.coinSpec (tags0.tableNames 1))

/-- The tags the compiled part's session is set up under: `--zk`'s, at the part circuit's mask log and one table. -/
def partTags (tags0 : Tags) (circuitB : ByteArray) : Tags :=
  Zk.tags (tags0.withCoins (partCoins tags0)) (Zk.maskLogOf circuitB) 1

/-- **The inner statement `ri` is the inner session the component holds**, and `w`'s inner shape and registration
are the component's. Each parsed field is pinned twice: what it parses to, and its bytes, the canonical bytes
(`VStar.Recursive.canonBytes`) of what the component read (`partitionB`, `recInnerB`). -/
structure InnerNames (o : VStar.Recursive.Own) (pr : VStar.Recursive.Proofs) (bs : Core.Blobs) (w : RecX)
    (ri : FlockDeploy.Raw) : Prop where
  circuit : bs.get o.circuit "the inner circuit" = .ok ri.circuit
  pub : bs.get o.pub "the inner public file" = .ok ri.publicFile
  statement : ri.statement = VStar.Recursive.INNER_STATEMENT
  plain : ri.zk = false
  os : ri.osCoins = true
  count : ri.count = 1
  tables : ri.tables = []
  partition : ri.partition.map (jsonOf "partition") = some (.ok o.partition)
  program : ri.program = some o.program
  draw : ri.draw = (bs.get o.draw "the inner statement's draw").toOption
  work : ri.work = none
  workTable : ri.workTable = none
  stratified : ri.stratified = (Flock.HmRow.stratifiedLaw o.partition o.program 1).toOption.map (·.2)
  closure : ri.closure = none
  registered : ri.registered = none
  publicInputs : ri.publicInputs = none
  shape : ∀ h, VStar.Recursive.innerHeld o pr bs = .ok h →
    (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" w.recInner)) = .ok h.shape
  registration : (o.registration.mapM fun d => (bs.get d "rec inner registration").toOption) = some w.registration
  partitionB : (VStar.Recursive.canonBytes o.partition).map some = .ok ri.partition
  recInnerB : ∀ h, VStar.Recursive.innerHeld o pr bs = .ok h →
    VStar.Recursive.canonBytes h.shape.toJson = .ok w.recInner

/-- **The form compiled part `j` sets up**: the statement `verity/flock-circuit` (`VStar.Recursive.STATEMENT`) under
`--zk`, its coins the keyed coin tree, one table and no lookup tables, and no `--work`, `--work-table`, `--stratified`,
`--closure` or `--public-inputs`. -/
def PartForm (b : FlockDeploy.Raw) : Prop :=
  b.statement = VStar.Recursive.STATEMENT ∧ b.zk = true ∧ b.osCoins = false ∧ b.count = 1 ∧ b.tables = [] ∧
    b.work = none ∧ b.workTable = none ∧ b.stratified = none ∧ b.closure = none ∧ b.publicInputs = none

/-- **Part `j`'s statement is the one the component's part `j` runs**, but for its staged public file and the staged
shared rows: its circuit, form, partition, program and registered reads `po`'s, its own bindings `po`'s bindings, and its
draw the one compiled part `j` reads off its record `pp.record` (`if tags.hm96Rows then Witness.recordDraw record else
none`, at the part's tags), as the model's `Raw.ins` reads a draw. Its public file is the one the component reads for
part `j` (`publicFile`; the model's part session replaces it with the staged one), and each parsed field's bytes are the
canonical bytes of what the component read (`partitionB`, `registeredB`, `bindingsB`, `drawB`). -/
structure PartNames (po : VStar.Recursive.PartOwn) (pp : VStar.Recursive.PartProof) (bs : Core.Blobs) (j : ℕ)
    (q : PartX) : Prop where
  bindings : jsonOf "rec-bindings" q.bindings = .ok po.bindings
  circuit : bs.get po.circuit s!"part {j}'s circuit" = .ok q.base.circuit
  form : PartForm q.base
  partition : q.base.partition.map (jsonOf "partition") = some (.ok po.partition)
  program : q.base.program = some po.program
  registered : q.base.registered.map (jsonOf "registered") = po.registered.map Except.ok
  draw : ∀ tags0 recordB record, Flock.Tags.find? VStar.Recursive.STATEMENT = some tags0 →
    bs.get pp.record s!"part {j}'s record" = .ok recordB → String.fromUTF8? recordB = some record →
    (q.base.draw.mapM (jsonOf "draw")).map (Option.map Flock.norm) =
      .ok (if (Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
          (Flock.Zk.maskLogOf q.base.circuit) 1).hm96Rows then Flock.Witness.recordDraw record else none)
  publicFile : bs.get pp.pub s!"part {j}'s public file" = .ok q.base.publicFile
  partitionB : (VStar.Recursive.canonBytes po.partition).map some = .ok q.base.partition
  registeredB : po.registered.mapM VStar.Recursive.canonBytes = .ok q.base.registered
  bindingsB : VStar.Recursive.canonBytes po.bindings = .ok q.bindings
  drawB : ∀ tags0 recordB record, Flock.Tags.find? VStar.Recursive.STATEMENT = some tags0 →
    bs.get pp.record s!"part {j}'s record" = .ok recordB → String.fromUTF8? recordB = some record →
    (if (Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
        (Flock.Zk.maskLogOf q.base.circuit) 1).hm96Rows then Flock.Witness.recordDraw record else none).mapM
      VStar.Recursive.canonBytes = .ok q.base.draw

/-- **`x` is the statement the component's inputs name** (`InnerNames`, `PartNames`): one part per part of the
verifier's, part `j` named by the verifier's part `j` and the evidence's part session `j`. Every field `x` encodes is a
file on the blobs or the canonical bytes of a value the component read, so `x` is a function of the params, the
evidence and the blobs, and its size is the one the component bounds (`VStar.Recursive.statementFits`). -/
def Names (p : VStar.Recursive.Params) (e : VStar.Recursive.Evidence) (bs : Core.Blobs) (x : Msg) : Prop :=
  ∃ w ri, decodeX x = some w ∧ FlockDeploy.decode w.inner = some ri ∧ InnerNames p.own e.proofs bs w ri ∧
    w.parts.length = p.own.parts.size ∧
    ∀ j q po pp, w.parts[j]? = some q → p.own.parts[j]? = some po → e.proofs.parts[j]? = some pp →
      PartNames po pp bs j q

/-- **Flock's own checks on the inner statement** (`RecDeploy.innerOk`: Flock's plain setup with the draw, partition
and program, its scope check, a draw, the header's draw at one table), which piece B reads. The compiled `innerHeld`
runs them as `VStar.Recursive.innerOk`, at `--stratified` the population the verifier's partition and program give
(`InnerNames.stratified`), so its acceptance gives it (`RefusalsOf.innerOkAt_of_compiled`). -/
def InnerOkAt (x : Msg) : Prop := ∀ w, decodeX x = some w → innerOk w.inner = .ok ()

/-- **The model's reading of `x`** at the component's inner session: the verifier's own partition, its inner shape as
both the shape and the inner statement's dimensions. -/
def xrOf (o : VStar.Recursive.Own) (h : VStar.Recursive.InnerHeld) (w : RecX) (ri : FlockDeploy.Raw) : XR :=
  ⟨w, ri, some o.partition, h.shape, h.shape⟩

/-- **Each part's session sets up at `xrOf`** (at some staging), the shape `readX_of_sessions` takes. `partRaw` reads
only `w` and `ri` of `xrOf`, so this reads no `readX`. -/
def PartsSetUp (o : VStar.Recursive.Own) (h : VStar.Recursive.InnerHeld) (x : Msg) : Prop :=
  ∀ w ri, decodeX x = some w → FlockDeploy.decode w.inner = some ri → ∀ s q, w.parts[s]? = some q →
    ∃ (st : Stage) (ms : List (Msg × Out)) (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
      (partRaw (xrOf o h w ri) st x ms s q).ins = .ok i ∧ i.setup = .ok (spec, sts) ∧ i.base.read.check = .ok ()

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem mapM_of_map {α β ε : Type} {f : α → Except ε β} {o : Option α} {b : β} (h : o.map f = some (.ok b)) :
    o.mapM f = .ok (some b) := by
  cases o with
  | none => cases h
  | some a =>
    simp only [Option.map_some, Option.some.injEq] at h
    simp [Option.mapM_some, h, Functor.map, Except.map]

/-! ## The component's inner shape -/

/-- `innerOf` at a public inner circuit leaves the shape's holo, sparse and stacked fields at their defaults. -/
theorem innerOf_eta {cB pB : ByteArray} {pa : Option Json} {pr : Option ByteArray} {v : RecVStar.Inner}
    (h : RecVStar.innerOf cB pB pa pr false = .ok v) :
    ({ v with holo := false, sparse := none, stacked := none } : RecVStar.Inner) = v := by
  unfold RecVStar.innerOf at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; rfl)

/-- **`shapeOf` is `innerOf`** at the verifier's partition and program, where it accepts. -/
theorem shapeOf_ok {cB pB : ByteArray} {P : Json} {prog : ByteArray} {i : RecVStar.Inner}
    (h : VStar.Recursive.shapeOf cB pB P prog = .ok i) :
    RecVStar.innerOf cB pB (some P) (some prog) = .ok i := by
  unfold VStar.Recursive.shapeOf at h
  generalize hI : RecVStar.innerOf cB pB (some P) (some prog) false = I at h
  cases I with
  | error e =>
    simp only [bind, Except.bind, pure, Except.pure, Except.mapError] at h
    repeat' split at h
    all_goals first | (cases h; done) | simp at h
  | ok v =>
    have hv := innerOf_eta hI
    simp only [bind, Except.bind, pure, Except.pure, Except.mapError] at h
    repeat' split at h
    all_goals first | (cases h; done) | (cases h; exact congrArg Except.ok hv.symm)

/-- `innerOf` at a public inner circuit is dense: no holo, no RecSparse shape, no stacked shape. -/
theorem innerOf_dense {cB pB : ByteArray} {pa : Option Json} {pr : Option ByteArray} {v : RecVStar.Inner}
    (h : RecVStar.innerOf cB pB pa pr = .ok v) : v.holo = false ∧ v.sparse = none ∧ v.stacked = none := by
  rw [← innerOf_eta h]
  exact ⟨rfl, rfl, rfl⟩

/-- **What `innerHeld` read**: the inner circuit and public file blobs, and its shape `shapeOf`'s at them. -/
theorem innerHeld_shape {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} (hh : VStar.Recursive.innerHeld o pr bs = .ok h) :
    ∃ cB pB, bs.get o.circuit "the inner circuit" = .ok cB ∧ bs.get o.pub "the inner public file" = .ok pB ∧
      VStar.Recursive.shapeOf cB pB o.partition o.program = .ok h.shape := by
  unfold VStar.Recursive.innerHeld at hh
  obtain ⟨cB, hc, hh⟩ := bind_ok hh
  obtain ⟨pB, hp, hh⟩ := bind_ok hh
  obtain ⟨sh, hs, hh⟩ := bind_ok hh
  refine ⟨cB, pB, hc, hp, ?_⟩
  rw [hs]
  congr 1
  simp only [bind, Except.bind, pure, Except.pure] at hh
  repeat' split at hh
  all_goals first | (cases hh; done) | (cases hh; rfl)

/-! ## `readXPre` and `readX` -/

/-- **`readXPre` reads `x`** where each of its steps does. -/
theorem readXPre_mk {x : Msg} {w : RecX} {ri : FlockDeploy.Raw} {j : Json} {shape dims : RecVStar.Inner}
    {P : Option Json} {vs : Array VStar.Stmt} (hx : decodeX x = some w) (hi : FlockDeploy.decode w.inner = some ri)
    (hj : jsonOf "rec-inner" w.recInner = .ok j) (hsj : RecVStar.Inner.ofJson j = .ok shape)
    (hp : ri.partition.mapM (jsonOf "partition") = .ok P)
    (hd : RecVStar.innerOf ri.circuit ri.publicFile P ri.program = .ok dims) (hok : innerOk w.inner = .ok ())
    (hv : RecVStar.statements shape = .ok vs) (hl : w.parts.length = vs.size) (hpn : portsNamed w.parts vs = true)
    (hcp : capPorted w.parts = true) :
    readXPre x = some ⟨w, ri, P, shape, dims⟩ := by
  unfold readXPre
  simp [hx, hi, hj, hsj, hp, hd, hok, hv, hl, hpn, hcp, Except.toOption, bind, Except.bind, pure, Except.pure]

/-- **`readX` is dense where its inner shape is**: the parser's form is off the shape (`RecScope.toStmt`). -/
theorem denseAt_of_readX {x : Msg} {r : XR} (hr : readX x = some r) (hh : r.shape.holo = false)
    (hs : r.shape.sparse = none) : DenseAt x := by
  intro s hs'
  unfold parse RecScope.parseOf RecScope.parseBytes decodeR at hs'
  rw [hr] at hs'
  cases hacc : RecScope.accepts (readOf r) with
  | false => simp [Option.filter, hacc] at hs'
  | true =>
    simp only [Option.map_some, Option.filter, hacc, ↓reduceIte, Option.some.injEq] at hs'
    subst hs'
    exact ⟨by simp [RecScope.toStmt, readOf, hh], by simp [RecScope.toStmt, readOf, hs]⟩

/-! ## The parts' circuits, from their sessions -/

/-- `checkPubRows` refuses a port of the reads that `fixedRow` knows without the inner transcript's commits. -/
theorem pubRows_none {st : Stmt} {own : Array Registered.Port} {sp : Option RecSparse.Shape}
    (h : RecVStar.checkPubRows st (some own) none sp = .ok ()) {reg : Registered.Port} (hmem : reg ∈ own)
    (hfix : (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome = true) : False := by
  have hreg : reg ∈ own.filter (fun reg => (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome) :=
    Array.mem_filter.2 ⟨hmem, hfix⟩
  unfold RecVStar.checkPubRows at h
  simp only [Option.getD_some] at h
  obtain ⟨_, -, h⟩ := Registered.except_bind_ok h
  split at h
  · first | exact absurd h (Registered.throw_ne_ok _ _) | cases h | simp at h
  · rename_i hnone
    cases hz : (own.filter fun reg => (RecVStar.fixedRow { reps := #[] } reg.value 0).isSome)[0]? with
    | some r0 => exact (hnone r0 hz).elim
    | none =>
      rw [Array.getElem?_eq_none_iff] at hz
      exact absurd (Array.size_pos_of_mem hreg) (by omega)

/-- **A part whose session sets up names its statement's ports**: where part `s`'s session at `r` (`partRaw`, at any
staging) reads its inputs, sets up and passes its scope check, its circuit (`partCircuit`) is V\*'s statement `s`'s as
`checkParsed` holds it, and has a port of each name the statement registers under a `rec-c<k>`: `RecCoins.check`
requires the verifier's reads at such a unit (`registered_of_coins`), `checkOwn` holds the port among them, and
`checkPubRows` finds it among the circuit's ports (`checkPubRows_rows`) at the inner transcript's commits, without
which it refuses (`pubRows_none`). -/
theorem part_ports {r : XR} {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ}
    {p : PartX} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsh : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)).toOption = some r.shape)
    (hi : (partRaw r st x ms s p).ins = .ok i) (hs : i.setup = .ok (spec, sts)) (hck : i.base.read.check = .ok ()) :
    ∃ vs sv c, RecVStar.statements r.shape = .ok vs ∧ vs[s]? = some sv ∧ partCircuit p = some c ∧
      RecVStar.checkParsed sv c = .ok () ∧
      ∀ q ∈ RecVStar.ownPorts sv, (RecVStar.cIndex q.2.1).isSome = true → c.ports.any (·.name == q.1) = true := by
  obtain ⟨hin, -⟩ := RecRaw.ins_inner_src hi
  have hinner : i.inner = some r.shape := by
    dsimp only [partRaw] at hin
    simp only [Option.mapM_some] at hin
    cases he : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)) with
    | error e => rw [he] at hin; cases hin
    | ok v =>
      rw [he] at hin hsh
      simp only [Functor.map, Except.map, Except.ok.injEq] at hin
      simp only [Except.toOption, Option.some.injEq] at hsh
      rw [← hin, hsh]
  have hpart : i.part = some s := RecRaw.ins_part hi
  have hbase := RecRaw.ins_base hi
  have hpi := publicInputs_none_of_scope hck
  have hty : i.base.read.tags.typed = false := ((Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1.2.1
  have htab : i.base.read.tablesOf = some sts := by
    unfold Flock.Readable.Read.tablesOf
    rw [RecIns.setup_read hs]
    cases hz : i.base.read.zk
    · simp
    · simp [RecIns.setup_shapeOk hs hz]
  obtain ⟨st0, hst0, hc0, -⟩ := Flock.Readable.table0 htab hty hpi
  have hpos : 0 < sts.size := by
    by_contra h0
    rw [Array.getElem?_eq_none (by omega)] at hst0
    cases hst0
  rw [Array.getElem?_eq_getElem hpos, Option.some.injEq] at hst0
  subst hst0
  have hC := checks_stmt (RecIns.setup_checks hs) 0 hpos
  rw [hinner, hpart] at hC
  obtain ⟨j, f, sv, hj, hf, hcc⟩ := checkStmt_circuit hC
  cases hj
  obtain ⟨hck', -, hparsed, hpt⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hcc
  obtain ⟨vs, hvs, hsv, hn, -⟩ := FlockSoundness.Discharge.Recursion.check_ok hck'
  obtain ⟨ib, hib⟩ := rawIns_ok_of_publicFile (w := p.base) (pf := st.pub x ms s) hbase
  obtain ⟨htg, hcf2, htb⟩ := rawIns_publicFile hib hbase
  obtain ⟨-, hcf3, htb3, -⟩ := rawIns_fields hib
  have hpc : partCircuit p = some sts[0].c := by
    unfold partCircuit
    rw [hib]
    simp only [Except.toOption, Option.bind_eq_bind, Option.bind_some]
    rw [← htg, ← hcf3, ← htb3, ← hcf2, ← htb, hc0]
  refine ⟨vs, sv, sts[0].c, hvs, hsv, hpc, hparsed, ?_⟩
  intro q hq hk
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_tablesOf htab hty hpi hpos
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hs) 0 hpos
  obtain ⟨own, hown⟩ := Option.isSome_iff_exists.1 (registered_of_coins hcoin (params_of_own P hn.symm hpt hq))
  have hC' := checks_stmt (RecIns.setup_checks hs) 0 hpos
  rw [hinner, hpart, hown] at hC'
  obtain ⟨j', f', sv', hj', hf', hcc', hco⟩ := checkStmt_own hC'
  cases hj'
  rw [hf] at hf'
  cases hf'
  rw [hcc] at hcc'
  cases hcc'
  obtain ⟨reg, hmem, hname, hval, -⟩ := checkOwn_mem hco hq
  obtain ⟨k, hk'⟩ := Option.isSome_iff_exists.1 hk
  have hkr : RecVStar.cIndex reg.value = some k := by rw [hval]; exact hk'
  have hpr := Flock.Readable.array_forM_ok (RecIns.setup_pubRows hs) 0 hpos
  rw [hown] at hpr
  cases hcs : i.commits with
  | none =>
    rw [hcs] at hpr
    exact (pubRows_none hpr hmem (fixedRow_isSome hkr)).elim
  | some cs =>
    rw [hcs] at hpr
    obtain ⟨-, -, pp, -, -, hfind, -⟩ := FlockC3.PubRows.checkPubRows_rows hpr hmem (fixedRow_isSome hkr)
    rw [List.find?_range_eq_some] at hfind
    obtain ⟨hpp, hpm, -⟩ := hfind
    have hlt : pp < sts[0].c.ports.size := List.mem_range.1 hpm
    rw [getElem!_pos sts[0].c.ports pp hlt] at hpp
    rw [← hname]
    exact Array.any_eq_true'.2 ⟨sts[0].c.ports[pp], Array.getElem_mem hlt, hpp⟩

/-- **`portsNamed` from each part's circuit**: where part `s`'s circuit has a port of each name V\*'s statement `s`
registers under a `rec-c<k>`. -/
theorem portsNamed_of {parts : List PartX} {vs : Array VStar.Stmt} (hl : parts.length = vs.size)
    (hp : ∀ s (hs : s < vs.size), ∃ c, parts[s]?.bind partCircuit = some c ∧
      ∀ q ∈ RecVStar.ownPorts vs[s], (RecVStar.cIndex q.2.1).isSome = true → c.ports.any (·.name == q.1) = true) :
    portsNamed parts vs = true := by
  unfold portsNamed
  rw [List.all_eq_true]
  intro s hs
  have hs' := List.mem_range.1 hs
  obtain ⟨c, hc, hq⟩ := hp s hs'
  simp only [hc]
  rw [List.all_eq_true]
  intro q hqm
  obtain ⟨hq1, hq2⟩ := List.mem_filter.1 hqm
  rw [getElem!_pos vs s hs'] at hq1
  exact hq q hq1 hq2

/-- **`capPorted` from the last part's circuit**: V\*'s list ends with RecCap (`FlockC3.Reps.list_ok`), whose rows
`checkParsed` holds the last part's circuit to, `top` among them (`top_port_cap`). -/
theorem capPorted_of {parts : List PartX} {vs : Array VStar.Stmt} {sh : RecVStar.Inner}
    (hvs : RecVStar.statements sh = .ok vs) (hl : parts.length = vs.size)
    (hp : ∀ s (hs : s < vs.size), ∃ c, parts[s]?.bind partCircuit = some c ∧ RecVStar.checkParsed vs[s] c = .ok ()) :
    capPorted parts = true := by
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, _, _, K, N, -, -, hss⟩ := FlockC3.Reps.list_ok hvs
  have hpush : vs = (A ++ C ++ K).push (VStar.Stmt.cap 0 N) := by
    rw [hss, Array.push_eq_append]
    simp only [Array.append_assoc]
  have hlt : vs.size - 1 < vs.size := by rw [hpush, Array.size_push]; omega
  have hlast : vs[vs.size - 1] = VStar.Stmt.cap 0 N := by
    have h? : vs[vs.size - 1]? = some (VStar.Stmt.cap 0 N) := by
      rw [hpush, Array.size_push, Nat.add_sub_cancel, Array.getElem?_push_size]
    rw [Array.getElem?_eq_getElem hlt, Option.some.injEq] at h?
    exact h?
  obtain ⟨c, hc, hparsed⟩ := hp (vs.size - 1) hlt
  rw [hlast] at hparsed
  obtain ⟨u, hu, -, -, -, hr⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hparsed
  obtain ⟨q, hq, -⟩ := RecDeploy.CapLower.top_port_cap hu hr
  unfold capPorted
  rw [hl, hc]
  have hsome : (capPort c).isSome = true := by rw [hq]; rfl
  unfold capPort at hsome
  rw [Array.findIdx?_isSome] at hsome
  exact hsome

/-- **The parts' sessions give `readXPre`'s two port checks** at `r`'s shape. -/
theorem ported_of_sessions {x : Msg} {r : XR} {vs : Array VStar.Stmt}
    (hsh : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" r.w.recInner)).toOption = some r.shape)
    (hvs : RecVStar.statements r.shape = .ok vs) (hl : r.w.parts.length = vs.size)
    (h : ∀ s p, r.w.parts[s]? = some p → ∃ (st : Stage) (ms : List (Msg × Out)) (i : RecIns)
      (spec : SessionSpec) (sts : Array Stmt),
      (partRaw r st x ms s p).ins = .ok i ∧ i.setup = .ok (spec, sts) ∧ i.base.read.check = .ok ()) :
    portsNamed r.w.parts vs = true ∧ capPorted r.w.parts = true := by
  have hpart : ∀ s (hs : s < vs.size), ∃ c, r.w.parts[s]?.bind partCircuit = some c ∧
      RecVStar.checkParsed vs[s] c = .ok () ∧
      ∀ q ∈ RecVStar.ownPorts vs[s], (RecVStar.cIndex q.2.1).isSome = true → c.ports.any (·.name == q.1) = true := by
    intro s hs
    have hs' : s < r.w.parts.length := by omega
    obtain ⟨st, ms, i, spec, sts, hi, hset, hck⟩ := h s r.w.parts[s] (List.getElem?_eq_getElem hs')
    obtain ⟨vs', sv, c, hvs', hsv, hpc, hparsed, hports⟩ := part_ports hsh hi hset hck
    rw [hvs] at hvs'
    cases hvs'
    rw [Array.getElem?_eq_getElem hs, Option.some.injEq] at hsv
    subst hsv
    refine ⟨c, ?_, hparsed, hports⟩
    rw [List.getElem?_eq_getElem hs']
    exact hpc
  refine ⟨portsNamed_of hl fun s hs => ?_, capPorted_of hvs hl fun s hs => ?_⟩
  · obtain ⟨c, hc, -, hq⟩ := hpart s hs
    exact ⟨c, hc, hq⟩
  · obtain ⟨c, hc, hp, -⟩ := hpart s hs
    exact ⟨c, hc, hp⟩

/-! ## The statement map -/

/-- **The component's inputs name a statement the model reads, densely** (piece B). Where `x` is the statement the
component's parameters, evidence and blobs name (`Names`), the component holds the inner session (`innerHeld`) with V*'s
list at its shape one statement per part, Flock's own checks pass on the inner statement (`InnerOkAt`) and each part's
session sets up (`PartsSetUp`): `readXPre` and `readX` read `x` as `xrOf`, at the component's inner shape
(`(xrOf …).shape = h.shape`), and `x` is dense (`DenseAt`). -/
theorem readX_of_names {p : VStar.Recursive.Params} {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {x : Msg}
    {h : VStar.Recursive.InnerHeld} {ss : Array VStar.Stmt}
    (hn : Names p e bs x) (hh : VStar.Recursive.innerHeld p.own e.proofs bs = .ok h)
    (hss : RecVStar.statements h.shape = .ok ss) (hsz : p.own.parts.size = ss.size)
    (hok : InnerOkAt x) (hset : PartsSetUp p.own h x) :
    ∃ w ri, decodeX x = some w ∧ FlockDeploy.decode w.inner = some ri ∧
      readXPre x = some (xrOf p.own h w ri) ∧ readX x = some (xrOf p.own h w ri) ∧ DenseAt x := by
  obtain ⟨w, ri, hx, hi, hI, hl, -⟩ := hn
  obtain ⟨cB, pB, hc, hp, hs⟩ := innerHeld_shape hh
  rw [hI.circuit] at hc
  rw [hI.pub] at hp
  obtain rfl : ri.circuit = cB := Except.ok.inj hc
  obtain rfl : ri.publicFile = pB := Except.ok.inj hp
  have hdims : RecVStar.innerOf ri.circuit ri.publicFile (some p.own.partition) ri.program = .ok h.shape := by
    rw [hI.program]
    exact shapeOf_ok hs
  obtain ⟨hho, hsp, -⟩ := innerOf_dense hdims
  have hsh := hI.shape h hh
  obtain ⟨j, hj, hsj⟩ := bind_ok hsh
  have hl' : w.parts.length = ss.size := hl.trans hsz
  have hshr : (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" (xrOf p.own h w ri).w.recInner)).toOption =
      some (xrOf p.own h w ri).shape := by
    show (do RecVStar.Inner.ofJson (← jsonOf "rec-inner" w.recInner)).toOption = some h.shape
    rw [hsh]
    rfl
  have hsess := hset w ri hx hi
  obtain ⟨hpn, hcp⟩ := ported_of_sessions (r := xrOf p.own h w ri) hshr hss hl' hsess
  have hpre : readXPre x = some (xrOf p.own h w ri) :=
    readXPre_mk hx hi hj hsj (mapM_of_map hI.partition) hdims (hok w hx) hss hl' hpn hcp
  have hrx := readX_of_sessions hpre hsess
  exact ⟨w, ri, hx, hi, hpre, hrx, denseAt_of_readX hrx hho hsp⟩

end Verity.Recursive
