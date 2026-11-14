module

public import Flock.Zk
public import Proofs.Flock.Claims.JsonNorm
public import Proofs.Flock.Soundness.Discharge.Hidden.CanonOk
public import Proofs.Flock.Soundness.Discharge.Hidden.OkWith
public import Proofs.Flock.Soundness.Discharge.Hidden.Public
public import Proofs.Flock.Soundness.Discharge.Hidden.Spec
public import Proofs.Flock.Soundness.Discharge.Layout.Drawn
public import Proofs.Flock.Soundness.Refine.Setup
public import Proofs.Flock.Soundness.Refine.Walk
import all Lean.Data.Json.Basic

/-!
# A setup at the sample of no unit, carried to another public file

`Stmt.setupHidden` at a sample that proves no unit reads the public file in its header and its rows' roots, and
nothing of it enters the statement's shape. `setupHidden_sibling`: when the setup returned at one public file and a
sample that proves no unit, it returns at every public file of the same circuit that `HmOut.loadPublic` loads,
`HmOut.checkPublic` passes and whose header names no `public_inputs`, at a sample of no unit of that file's population,
and the two statements are siblings (`Stmt.Sibling`): they share the circuit, the counts, the pin, Δ, the regions and
the statement's digest, and differ at most in the public file, Σ, the hello and the partition's finding. The shape
check reads only what siblings share (`shapeOk_sibling`).

The proof walks the first setup with `walk_step` and builds the second with `ok_step` (`OkWith.lean`). The steps that
read the public file are carried one by one: the load and its checks (`loadPublic_in`, `checkOwn_noPorts`), the sample
(`drawn_okWith`), the roots (`checkPublic_drawn`), the unit indices of no unit (`drawn_units`), linking over no reader
(`links_noUnit`), the regions of no unit (`regions_noUnit`) and the hello (`helloOf_isOk`).
-/

@[expose] public section

namespace FlockSoundness.Discharge.Hidden

open Flock Refine
open Lean (Json)

/-! ## What a loaded public file holds -/

theorem obj_of_get {j : Json} {k ctx : String} {v : Json} (h : J.get j k ctx = .ok v) : ∃ m, j = .obj m := by
  cases j with
  | obj m => exact ⟨m, rfl⟩
  | _ => cases h

theorem wf_of_norm {j : Json} {m : Std.TreeMap.Raw String Json compare} (h : Flock.norm j = .obj m) : m.WF := by
  have hw := FlockDeploy.wf_norm j
  rw [h] at hw
  cases hw with
  | obj hmw _ => exact hmw

-- `walk_step` down `HmRow.loadPublic`'s unfolded do-block, a goal per branch
set_option maxHeartbeats 2000000 in
/-- **What `HmRow.loadPublic` returns**: a file with no population, whose header is an object over a well-formed tree
that names no `unit_draw`, has a canonical text, and holds the unit indices the setup reads. -/
theorem loadPublic_reads {tags : Tags} {c : Circuit} {f : ByteArray} {pop : Public}
    (h : HmRow.loadPublic tags c f = .ok pop) :
    pop.population = none ∧ (∃ m : Std.TreeMap.Raw String Json compare, pop.header = .obj m ∧ m.WF) ∧
    (∃ u a idx, J.get pop.header "units" "public file" = .ok u ∧ J.arr u "indices" "public file" = .ok a ∧
      a.mapM (fun x => J.asNat x "a unit index") = .ok idx) ∧
    J.has pop.header "unit_draw" = false ∧ (canonUtf8 pop.header).isOk = true := by
  unfold HmRow.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hpj := ‹(pure _ : V Json) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hpj
    subst hpj
    refine ⟨rfl, ?_, ⟨_, _, _, ‹J.get _ "units" _ = Except.ok _›, ‹J.arr _ "indices" _ = Except.ok _›,
      ‹Array.mapM _ _ = Except.ok _›⟩, Bool.eq_false_iff.2 ‹¬J.has _ "unit_draw" = true›, ?_⟩
    · obtain ⟨m, hm⟩ := obj_of_get ‹J.get _ "units" _ = Except.ok _›
      exact ⟨m, hm, wf_of_norm hm⟩
    · obtain ⟨s, hs, -⟩ := ebind_ok ‹HmRow.publicDigest _ _ = Except.ok _›
      exact isOk_of_eq hs

/-! ## The sample of no unit over a loaded public file -/

theorem publicDigest_okWith {hdr : Json} {pubs : ByteArray} (h : (canonUtf8 hdr).isOk = true) :
    OkWith (HmRow.publicDigest hdr pubs) fun _ => True := by
  unfold HmRow.publicDigest
  cases hc : canonUtf8 hdr with
  | error e => rw [hc] at h; cases h
  | ok s => exact ⟨_, rfl, trivial⟩

/-- `canonUtf8` prints the header's `units` entry at any indices. -/
theorem canonUtf8_unitsEntry (c : Circuit) (indices : Array Nat) :
    (canonUtf8 (HmRow.unitsEntry c indices)).isOk = true := by
  have hn : ∀ a : Array Nat, (canonUtf8 (Json.arr (a.map fun x => Lean.toJson x))).isOk = true := fun a =>
    canonUtf8_arr fun x hx => by
      obtain ⟨n, -, rfl⟩ := Array.mem_map.1 hx
      exact canonUtf8_nat n
  unfold HmRow.unitsEntry
  refine canonUtf8_mkObj fun v hv => ?_
  split at hv
  all_goals
    simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
      or_false] at hv
    rcases hv with rfl | rfl | rfl | rfl | rfl
  all_goals first
    | exact canonUtf8_nat _
    | exact hn _
    | exact canonUtf8_arr fun x hx => (by
        obtain ⟨b, -, rfl⟩ := Array.mem_map.1 hx
        exact hn _)
    | exact canonUtf8_arr fun x hx => (by
        rw [(Array.mem_replicate.1 hx).2]
        exact canonUtf8_str _)

/-- The header of a sample over the population `pop`: the population's with the sample, its count of proved units,
the population's size and the proved units' indices. -/
def drawnHeader (c : Circuit) (pop : Public) (d : Json) (k : Nat) (indices : Array Nat) : Json :=
  ((pop.header.setObjVal! "unit_draw" d).setObjVal! "instances" (Lean.toJson k)
    |>.setObjVal! "population" (Lean.toJson pop.n)).setObjVal! "units" (HmRow.unitsEntry c indices)

/-- `canonUtf8` prints a sample's header when it prints the population's and the sample. -/
theorem canonUtf8_drawnHeader {c : Circuit} {pop : Public} {d : Json} {k : Nat} {indices : Array Nat}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m)
    (hh : (canonUtf8 pop.header).isOk = true) (hd : (canonUtf8 d).isOk = true) :
    (canonUtf8 (drawnHeader c pop d k indices)).isOk = true := by
  rw [hp] at hh
  unfold drawnHeader
  rw [hp]
  exact canonUtf8_insert hm.insert.insert.insert (canonUtf8_insert hm.insert.insert (canonUtf8_insert hm.insert
    (canonUtf8_insert hm hh hd) (canonUtf8_nat _)) (canonUtf8_nat _)) (canonUtf8_unitsEntry _ _)

-- `ok_step` down `HmRow.drawn`'s unfolded do-block, a goal per step that returns on its own
set_option maxHeartbeats 2000000 in
/-- **A sample over a loaded public file returns**, with the header, count, population and tables `HmRow.drawn`
gives it: the sample decodes at the population's size, the header's unit indices were read, and `canonUtf8` prints the
header and the sample. -/
theorem drawn_okWith {c : Circuit} {pop : Public} {d : Json} {D : Draw.UnitDraw} {u : Json} {a : Array Json}
    {idx : Array Nat} {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hn : D.population = pop.n) (hpn : pop.population = none)
    (hu : J.get pop.header "units" "public file" = .ok u) (ha : J.arr u "indices" "public file" = .ok a)
    (hi : a.mapM (fun x => J.asNat x "a unit index") = .ok idx)
    (hh : (canonUtf8 pop.header).isOk = true) (hd : (canonUtf8 d).isOk = true) :
    OkWith (HmRow.drawn c pop d) fun pub =>
      pub.header = drawnHeader c pop d D.proved.size (D.proved.map fun u => idx.getD u 0) ∧ pub.n = D.proved.size ∧
      pub.population = some (pop.n, pop.digests, pop.outputs) ∧
      pub.tables = pop.tables.map (fun t => { t with refs := (D.proved.foldl
        (fun acc u => acc ++ t.refs.extract (u * c.ports.size) (u * c.ports.size + c.ports.size)) #[]) }) := by
  have hc : ¬(D.population != pop.n || pop.population.isSome) = true := by
    simp only [hn, bne_self_eq_false, hpn, Option.isSome_none, Bool.or_self, Bool.false_eq_true, not_false_eq_true]
  have hcan := canonUtf8_drawnHeader (c := c) (k := D.proved.size) (indices := D.proved.map fun u => idx.getD u 0)
    hm hp hh hd
  unfold HmRow.drawn
  dsimp only
  simp only [throw_bind']
  repeat' ok_step
  all_goals first
    | exact ⟨rfl, rfl, rfl, rfl⟩
    | exact okWith_forIn_array _ _ _
    | exact publicDigest_okWith hcan
    | skip

/-! ## What a sample's header holds -/

theorem getObjVal?_obj (m : Std.TreeMap.Raw String Json compare) (k : String) :
    (Json.obj m).getObjVal? k = match m[k]? with
      | some v => .ok v
      | none => throw s!"property not found: {k}" := by
  simp only [Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?]
  rfl

/-- The fields `HmRow.drawn` sets in a sample's header, and `frame_v3`, which it keeps. -/
theorem drawnHeader_reads {c : Circuit} {pop : Public} {d : Json} {k : Nat} {indices : Array Nat}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m) :
    (drawnHeader c pop d k indices).getObjVal? "units" = .ok (HmRow.unitsEntry c indices) ∧
    (drawnHeader c pop d k indices).getObjVal? "instances" = .ok (Lean.toJson k) ∧
    (drawnHeader c pop d k indices).getObjVal? "unit_draw" = .ok d ∧
    (drawnHeader c pop d k indices).getObjVal? "frame_v3" = pop.header.getObjVal? "frame_v3" := by
  unfold drawnHeader
  rw [hp]
  simp only [Json.setObjVal!, getObjVal?_obj, Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert,
    Std.TreeMap.Raw.getElem?_insert hm.insert.insert, Std.TreeMap.Raw.getElem?_insert hm.insert,
    Std.TreeMap.Raw.getElem?_insert hm]
  simp only [Std.compare_self, ↓reduceIte, Std.LawfulEqCmp.compare_eq_iff_eq, String.reduceEq, and_self]

/-! ## A sample's public file passes `HmOut.checkPublic` when its population's does -/

-- `simp only` rewrites `HmOut.checkPublic`'s unfolded do-block at both public files
set_option maxHeartbeats 2000000 in
/-- **`HmOut.checkPublic` on a sample over a loaded public file** is the check on the file: the sample keeps the
header's `frame_v3`, the tables' counts and starts and the population the roots are over, and its count is the
sample's. -/
theorem checkPublic_drawn {c c' : Circuit} {pop pub : Public} {d : Json} {D : Draw.UnitDraw} {idx : Array Nat}
    {r : Tables → Array Nat} {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hpn : pop.population = none) (hnd : J.has pop.header "unit_draw" = false)
    (hh : pub.header = drawnHeader c' pop d D.proved.size idx) (hn : pub.n = D.proved.size)
    (hpop : pub.population = some (pop.n, pop.digests, pop.outputs))
    (ht : pub.tables = pop.tables.map fun t => { t with refs := r t })
    (h : HmOut.checkPublic c pop = .ok ()) : HmOut.checkPublic c pub = .ok () := by
  obtain ⟨-, hI, hud, hfv⟩ := drawnHeader_reads (c := c') (d := d) (k := D.proved.size) (indices := idx) hm hp
  rw [← hh] at hI hud hfv
  have hfr : pub.frameV3 = pop.frameV3 := by unfold Public.frameV3 J.get; rw [hfv]
  have hfb : ∀ k port, pub.fv3Bytes k port = pop.fv3Bytes k port := fun k port => by
    unfold Public.fv3Bytes; rw [hfr]
  have hinst : J.nat pub.header "instances" "public file" = .ok D.proved.size := by
    unfold J.nat J.get; rw [hI]; rfl
  obtain ⟨e, he⟩ : ∃ e, pop.header.getObjVal? "unit_draw" = .error e := by
    unfold J.has at hnd
    cases hg : pop.header.getObjVal? "unit_draw" with
    | ok v => rw [hg] at hnd; cases hnd
    | error e => exact ⟨e, rfl⟩
  unfold HmOut.checkPublic at h ⊢
  cases htp : pop.tables with
  | none =>
    have htp' : pub.tables = none := by rw [ht, htp]; rfl
    simp only [hfr, hfb, hpn, hpop, hud, he, hD, hinst, hn, htp, htp', Option.getD_some, Option.getD_none, ok_bind,
      bne_self_eq_false, Bool.or_self, Bool.false_eq_true, ↓reduceIte] at h ⊢
    exact h
  | some t =>
    have htp' : pub.tables = some { t with refs := r t } := by rw [ht, htp]; rfl
    simp only [hfr, hfb, hpn, hpop, hud, he, hD, hinst, hn, htp, htp', Option.getD_some, Option.getD_none, ok_bind,
      bne_self_eq_false, Bool.or_self, Bool.false_eq_true, ↓reduceIte] at h ⊢
    exact h

/-! ## The steps that read the public file, at a statement with no public input port and no unit -/

/-- `HmIn.loadPublic` is `HmOut.loadPublic` at a circuit with no public input port and a header that names none. -/
theorem loadPublic_in {tags : Tags} {c : Circuit} {pi : HmIn.Ports} {f : ByteArray} {pop : Public}
    (he : pi.ports.isEmpty = true) (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (h : HmOut.loadPublic tags c f = .ok pop) : HmIn.loadPublic tags c pi f = .ok pop := by
  have hrow := loadPublic_nil h
  unfold HmOut.loadPublic at h
  unfold HmIn.loadPublic
  simp only [he, ↓reduceIte, checkReads_noPorts he]
  rw [hrow] at h ⊢
  simp only [ok_bind] at h ⊢
  cases hk : pop.header.getObjVal? "public_inputs" with
  | ok v => rw [hk] at hkey; cases hkey
  | error e =>
    exact h

theorem checkOwn_noPorts {pi : HmIn.Ports} {pop : Public} (he : pi.ports.isEmpty = true) :
    HmIn.checkOwn pi pop none = .ok () := by
  unfold HmIn.checkOwn
  simp only [he]
  rfl

/-- At no unit every block's bytes are the pinned dummy's, so the regions are the same at any two public files. -/
theorem regions_noUnit {c : Circuit} {pub pub' : Public} (h : pub.n = 0) (h' : pub'.n = 0) :
    HmOut.regions c pub = HmOut.regions c pub' := by
  unfold HmOut.regions
  simp only [h, h', Circuit.instOf, Nat.not_lt_zero, ↓reduceIte]

theorem inRegions_noPorts {c : Circuit} {pi : HmIn.Ports} {pub : Public} (he : pi.ports.isEmpty = true) :
    HmIn.regions c pi pub = .ok #[] := by
  unfold HmIn.regions
  simp only [he, ↓reduceIte]
  rfl

/-- The units a sample that proves none names: none. -/
theorem readersOf_noUnit {d : Json} {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hpr : D.proved = #[]) :
    HmLinks.readersOf (some d) = some #[] := by
  simp only [HmLinks.readersOf, Option.bind_some, hD, Except.toOption, Option.map_some, hpr]

/-- **Linking at no unit returns at any public file whose unit indices read**, when it returned at one: it parses
META's links and decodes the program, which the public file does not enter, and checks no reader. -/
theorem links_noUnit {c : Circuit} {pop pop' : Public} {prog : ByteArray} {w : Nat} {u : Json} {a : Array Json}
    {idx : Array Nat} (h : HmLinks.check c pop (some #[]) prog = .ok w)
    (hu : J.get pop'.header "units" "public file" = .ok u) (ha : J.arr u "indices" "public file" = .ok a)
    (hi : a.mapM (fun x => J.asNat x "a unit index") = .ok idx) :
    OkWith (HmLinks.check c pop' (some #[]) prog) fun _ => True := by
  unfold HmLinks.check at h
  dsimp only at h
  walk_step
  walk_step
  unfold HmLinks.check HmLinks.checkAt
  dsimp only
  simp only [Option.getD_some]
  repeat' ok_step
  all_goals first
    | trivial
    | exact okWith_forIn_empty _ _

/-- `HmRow.qwordFinding` reads the public file only in its header's `units`. -/
theorem qwordFinding_congr {c : Circuit} {pub pub' : Public} {obj : Json} {prog : ByteArray}
    {f : HmRow.PartitionFinding}
    (h : J.get pub.header "units" "public file" = J.get pub'.header "units" "public file") :
    HmRow.qwordFinding c pub obj prog f = HmRow.qwordFinding c pub' obj prog f := by
  unfold HmRow.qwordFinding
  rw [h]

-- `walk_step` down `HmRow.drawn`'s unfolded do-block, a goal per branch
set_option maxHeartbeats 2000000 in
/-- **A sample that proves no unit has the header `units` of no index.** -/
theorem drawn_units {c : Circuit} {pop pub : Public} {d : Json} {D : Draw.UnitDraw}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hpr : D.proved = #[]) (h : HmRow.drawn c pop d = .ok pub) :
    J.get pub.header "units" "public file" = .ok (HmRow.unitsEntry c #[]) := by
  unfold HmRow.drawn at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain rfl := Except.ok.inj (hD.symm.trans ‹Draw.UnitDraw.ofJson d = Except.ok _›)
    have hr := (drawnHeader_reads (c := c) (d := d) (k := D.proved.size)
      (indices := D.proved.map fun u => (‹Array Nat› : Array Nat).getD u 0) hm hp).1
    unfold drawnHeader at hr
    unfold J.get
    dsimp only
    rw [hr, hpr, Array.map_empty]

/-! ## The setup, carried -/

/-- What two statements share when they differ in the public file alone, each at a sample that proves no unit:
every field but the public file, Σ, the hello and the partition's finding. -/
def Stmt.Sibling (st₁ st₂ : Stmt) : Prop :=
  st₂.tags = st₁.tags ∧ st₂.leaf = st₁.leaf ∧ st₂.c = st₁.c ∧ st₂.n = st₁.n ∧ st₂.blocksReal = st₁.blocksReal ∧
    st₂.nbl = st₁.nbl ∧ st₂.m = st₁.m ∧ st₂.mPts = st₁.mPts ∧ st₂.pin = st₁.pin ∧ st₂.regions = st₁.regions ∧
    st₂.da = st₁.da ∧ st₂.db = st₁.db ∧ st₂.digest = st₁.digest ∧ st₂.coinTree = st₁.coinTree ∧
    st₂.merkle = st₁.merkle

-- `walk_step` down the first `Stmt.setupHidden`'s unfolded do-block, a goal per branch, and `ok_step` down the second's
set_option maxHeartbeats 8000000 in
/-- `setupHidden_sibling` at the circuit `HmOut.parseCommit` returns, named so that the second public file's
hypotheses are stated before the first setup is walked. -/
theorem sibling_core {tags : Tags} {cf pf₁ pf₂ : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {d₁ d₂ : Json} {partition : Option Json} {program : Option ByteArray} {st₁ : Stmt} {c : Circuit} {pop : Public}
    (hc : HmOut.parseCommit tags cf tables = .ok c)
    (h : Stmt.setupHidden tags cf pf₁ coins tables (some d₁) partition program none = .ok st₁) (hn₁ : st₁.n = 0)
    (hpop : HmOut.loadPublic tags c pf₂ = .ok pop)
    (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (hpub : HmOut.checkPublic c pop = .ok ())
    (hd₂ : Draw.UnitDraw.ofJson d₂ = .ok ⟨.subset 0, pop.n, #[], #[]⟩) (hcan : (canonUtf8 d₂).isOk = true) :
    OkWith (Stmt.setupHidden tags cf pf₂ coins tables (some d₂) partition program none) (Stmt.Sibling st₁) := by
  unfold Stmt.setupHidden at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain rfl := Except.ok.inj (hc.symm.trans ‹HmOut.parseCommit _ _ _ = Except.ok _›)
    have he := checkOwn_none ‹HmIn.checkOwn _ _ none = Except.ok _›
    clear_walked
    name_ok ‹HmIn.parse c = Except.ok _› as pi
    name_ok ‹HmIn.loadPublic tags c pi pf₁ = Except.ok _› as pop₁
    name_ok ‹HmIn.drawn c pi pop₁ d₁ = Except.ok _› as pub₁
    have hl₁ := loadPublic_nil (loadPublic_noPorts he ‹HmIn.loadPublic tags c pi pf₁ = Except.ok pop₁›)
    obtain ⟨-, ⟨m₁, hp₁, hm₁⟩, -⟩ := loadPublic_reads hl₁
    have hdr₁ : HmRow.drawn { c with outPorts := #[] } pop₁ d₁ = .ok pub₁ := by
      have := ‹HmIn.drawn c pi pop₁ d₁ = Except.ok pub₁›
      rwa [drawn_noPorts he] at this
    obtain ⟨D₁, hD₁, -, hsz₁, -⟩ := Layout.drawn_spec hdr₁
    have hn₁ : pub₁.n = 0 := hn₁
    have hpr₁ : D₁.proved = #[] := Array.eq_empty_of_size_eq_zero (hsz₁ ▸ hn₁)
    have hw₁ : ∀ w, J.get pub₁.header "units" "public file" = .ok w → HmRow.unitsEntry _ #[] = w := fun w hw =>
      Except.ok.inj ((drawn_units hm₁ hp₁ hD₁ hpr₁ hdr₁).symm.trans hw)
    try obtain rfl := hw₁ _ ‹J.get pub₁.header "units" "public file" = Except.ok _›
    have hu₁ := drawn_units hm₁ hp₁ hD₁ hpr₁ hdr₁
    have hl₂ := loadPublic_nil hpop
    obtain ⟨hpn, ⟨m, hp, hm⟩, ⟨u, a, idx, hu, ha, hi⟩, hnd, hh⟩ := loadPublic_reads hl₂
    have hpr₂ : (⟨.subset 0, pop.n, #[], #[]⟩ : Draw.UnitDraw).proved = #[] := rfl
    obtain ⟨pub₂, hdr₂, hh₂, hn₂, hpop₂, ht₂⟩ :=
      drawn_okWith (c := { c with outPorts := #[] }) hm hp hd₂ rfl hpn hu ha hi hh hcan
    have hu₂ := drawn_units hm hp hd₂ hpr₂ hdr₂
    have hcp₂ := checkPublic_drawn (c := c) hm hp hd₂ hpn hnd hh₂ hn₂ hpop₂ ht₂ hpub
    have hn₂ : pub₂.n = 0 := hn₂
    have hlp₂ := loadPublic_in (pi := pi) he hkey hpop
    have hown₂ := checkOwn_noPorts (pop := pop) he
    have hdrawn₂ : HmIn.drawn c pi pop d₂ = .ok pub₂ := by rw [drawn_noPorts he]; exact hdr₂
    obtain rfl := regions_noPorts he ‹HmIn.regions c pi pub₁ = Except.ok _›
    have hir₂ := inRegions_noPorts (c := c) (pub := pub₂) he
    have hro₂ := (regions_noUnit (c := c) hn₂ hn₁).trans ‹HmOut.regions c pub₁ = Except.ok _›
    have hrd₁ := readersOf_noUnit hD₁ hpr₁
    have hrd₂ := readersOf_noUnit hd₂ hpr₂
    clear hl₁ hl₂ hdr₁ hdr₂ hh₂ hpop₂ ht₂ hsz₁ hw₁
    obtain ⟨hdr₁, n₁, dg₁, out₁, sha₁, popn₁, tb₁⟩ := pub₁
    obtain ⟨hdr₂, n₂, dg₂, out₂, sha₂, popn₂, tb₂⟩ := pub₂
    obtain rfl : n₁ = 0 := hn₁
    obtain rfl : n₂ = 0 := hn₂
    unfold Stmt.setupHidden
    dsimp only
    simp only [throw_bind', pure_bind, hd₂, Except.map]
    repeat' ok_step
  all_goals first
    | exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
    | exact okWith_of_isOk (helloOf_isOk ..)
    | (rw [hrd₂]
       exact links_noUnit (by rw [← hrd₁]; exact ‹HmLinks.check _ _ _ _ = Except.ok _›) hu ha hi)
    | (rw [qwordFinding_congr (hu₂.trans hu₁.symm)]
       exact okWith_of_eq ‹HmRow.qwordFinding _ _ _ _ _ = Except.ok _›)

/-- **The shape check reads only what siblings share**: `m`, the block count, Δ, the regions and the circuit. -/
theorem shapeOk_sibling {st₁ st₂ : Stmt} (h : Stmt.Sibling st₁ st₂) : Zk.shapeOk st₂ = Zk.shapeOk st₁ := by
  obtain ⟨-, -, hc, -, -, hnbl, hm, -, -, hreg, hda, hdb, -⟩ := h
  unfold Zk.shapeOk Zk.maskFree
  simp only [hc, hnbl, hm, hreg, hda, hdb]

/-- **A setup at the sample of no unit, carried to another public file.** `Stmt.setupHidden` returned `st₁` at the
public file `pf₁` and a sample `d₁` that proves no unit (`st₁.n = 0`), with no public inputs of the verifier's own. At
a public file `pf₂` that `HmOut.loadPublic` loads for `st₁`'s circuit and `HmOut.checkPublic` passes, whose header
names no `public_inputs`, and a sample `d₂` of no unit over `pf₂`'s population that has a canonical text, the setup
returns a sibling of `st₁`, on which the shape check returns what it returns on `st₁`. -/
theorem setupHidden_sibling {tags : Tags} {cf pf₁ pf₂ : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d₁ d₂ : Json} {partition : Option Json} {program : Option ByteArray}
    {st₁ : Stmt} {pop : Public}
    (h : Stmt.setupHidden tags cf pf₁ coins tables (some d₁) partition program none = .ok st₁) (hn₁ : st₁.n = 0)
    (hpop : HmOut.loadPublic tags st₁.c pf₂ = .ok pop)
    (hkey : (pop.header.getObjVal? "public_inputs").toOption = none)
    (hpub : HmOut.checkPublic st₁.c pop = .ok ())
    (hd₂ : Draw.UnitDraw.ofJson d₂ = .ok ⟨.subset 0, pop.n, #[], #[]⟩) (hcan : (canonUtf8 d₂).isOk = true) :
    ∃ st₂, Stmt.setupHidden tags cf pf₂ coins tables (some d₂) partition program none = .ok st₂ ∧
      Stmt.Sibling st₁ st₂ ∧ Zk.shapeOk st₂ = Zk.shapeOk st₁ := by
  obtain ⟨st₂, h₂, hs⟩ := sibling_core (setupHidden_spec' h).parse h hn₁ hpop hkey hpub hd₂ hcan
  exact ⟨st₂, h₂, hs, shapeOk_sibling hs⟩

end FlockSoundness.Discharge.Hidden
