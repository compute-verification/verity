module

public import Proofs.Flock.Claims.Complete.Json.Value
public import Proofs.Flock.Claims.JsonEq
public import Proofs.Flock.Soundness.Discharge.FailClosed.PublicTies
public import Proofs.Flock.SpotFlock.RowsOpen
import Proofs.Flock.Claims.Complete.Json.Parse
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Claims.Scope.Check
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The public file's header names each port's registered root and domain

`header_roots`: at a session statement that sets up as a masked Flock statement in Flock's scope with hidden outputs
(`ScopedHidden`), with one table (`count ≤ 1`), the raw header of the public file (`headerOf`, its first line as
parsed, before `Flock.norm`) holds, under `frame_v3.roots`, an entry for each input port `q` whose hex string decodes
to the port's published root over the population (`PortRoots.pubRoot`), and under `frame_v3.domain_ids` a hex string
for `q` that decodes to the port's domain (`PortRoots.idOf`).

Flock's `checkPublic` ties the statement's `frame_v3` roots to the tree over the rows (`FailClosed.ties_of_check`).
The statement's public file is the loaded one, or the sample drawn from it (`HmOut.drawn`), which keeps the header's
`frame_v3` and the tables' counts and sets the population to the loaded file's (`roots_of_load`). The loaded header is
the raw header normalized (`loadPublic_raw`), and normalizing keeps each field's string (`raw_entry`).
-/

@[expose] public section

namespace SpotFlock

open Claims (Msg)
open Lean (Json)

namespace HeaderRoots

/-- **`HmRow.loadPublic`'s header is the raw header normalized**, an object, and a loaded file has no population. -/
theorem loadPublic_raw {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {f : ByteArray}
    {pop : _root_.Flock.Public} (h : _root_.Flock.HmRow.loadPublic tags c f = .ok pop) :
    ∃ j, headerOf f = some j ∧ pop.header = _root_.Flock.norm j ∧ pop.population = none ∧
      ∃ m, pop.header = .obj m := by
  unfold _root_.Flock.HmRow.loadPublic at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hpj := ‹(pure _ : _root_.Flock.V Json) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hpj
    subst hpj
    refine ⟨_, ?_, rfl, rfl, FlockComplete.JsonRT.obj_of_getObjVal
      (FlockComplete.JsonRT.str_ok (k := "format") (by assumption))⟩
    have hps := ‹(pure _ : _root_.Flock.V String) = Except.ok _›
    simp only [pure, Except.pure, Except.ok.injEq] at hps
    subst hps
    simp [headerOf, ‹_root_.Flock.findByte f 10 = some _›, ‹String.fromUTF8? _ = some _›,
      ‹_root_.Flock.JsonParse.parse _ = .ok _›, Except.toOption]

/-- **A drawn sample keeps the header's `frame_v3`** and sets the population to the loaded file's. -/
theorem drawn_frame {c : _root_.Flock.Circuit} {pub pub' : _root_.Flock.Public} {d : Json}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pub.header = .obj m)
    (h : _root_.Flock.HmRow.drawn c pub d = .ok pub') :
    pub'.header.getObjVal? "frame_v3" = pub.header.getObjVal? "frame_v3" ∧
      pub'.population = some (pub.n, pub.digests, pub.outputs) := by
  unfold _root_.Flock.HmRow.drawn at h
  dsimp only at h
  simp only [FlockSoundness.Refine.throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine ⟨?_, rfl⟩
    dsimp only
    rw [hp]
    simp only [Json.setObjVal!, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?,
      Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert, Std.TreeMap.Raw.getElem?_insert hm.insert.insert,
      Std.TreeMap.Raw.getElem?_insert hm.insert, Std.TreeMap.Raw.getElem?_insert hm]
    simp

/-- **The raw header is well formed**: the parser builds well-formed trees (`JsonRT.parse_wf`). -/
theorem wf_of_headerOf {f : ByteArray} {hdr : Json} (h : headerOf f = some hdr) : Claims.Json.Wf hdr := by
  unfold headerOf at h
  cases hnl : _root_.Flock.findByte f 10 with
  | none => simp [hnl] at h
  | some nl =>
    cases hs : String.fromUTF8? (f.extract 0 nl) with
    | none => simp [hnl, hs] at h
    | some s =>
      cases hp : _root_.Flock.JsonParse.parse s with
      | error e => simp [hnl, hs, hp, Except.toOption] at h
      | ok j =>
        simp [hnl, hs, hp, Except.toOption] at h
        subst h
        exact FlockComplete.JsonRT.parse_wf s _ hp

/-- **The statement's public file keeps the loaded file's `frame_v3` and input roots**: it is the loaded file, or the
sample drawn from it, which keeps the header's `frame_v3`, the tables' counts and starts, and sets the population to
the loaded file's. -/
theorem drawn_ties {c : _root_.Flock.Circuit} {pop pub : _root_.Flock.Public} {draw : Option Json}
    {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hp : pop.header = .obj m) (hpn : pop.population = none)
    (hnone : draw = none → pub = pop)
    (hsome : ∀ d, draw = some d → _root_.Flock.HmOut.drawn c pop d = .ok pub) :
    pub.header.getObjVal? "frame_v3" = pop.header.getObjVal? "frame_v3" ∧
      ∀ q dom, FlockSoundness.Discharge.FailClosed.inRoot c pub q dom =
        FlockSoundness.Discharge.FailClosed.inRoot c pop q dom := by
  cases draw with
  | none => rw [hnone rfl]; exact ⟨rfl, fun _ _ => rfl⟩
  | some d =>
    have hd : _root_.Flock.HmRow.drawn { c with outPorts := #[] } pop d = .ok pub := hsome d rfl
    obtain ⟨hfv, hpop⟩ := drawn_frame hm hp hd
    obtain ⟨_, -, -, -, htab, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hd
    refine ⟨hfv, fun q dom => ?_⟩
    have hpo : FlockSoundness.Discharge.FailClosed.popOf pub = FlockSoundness.Discharge.FailClosed.popOf pop := by
      simp only [FlockSoundness.Discharge.FailClosed.popOf, hpop, hpn, Option.getD_some, Option.getD_none]
    have hir : FlockSoundness.Discharge.FailClosed.inRows c pub q =
        FlockSoundness.Discharge.FailClosed.inRows c pop q := by
      unfold FlockSoundness.Discharge.FailClosed.inRows
      rw [htab, hpo]
      cases pop.tables <;> rfl
    unfold FlockSoundness.Discharge.FailClosed.inRoot
    rw [hir, hpo]

/-- A field of a well-formed object is well formed. -/
theorem wf_of_getObjVal {a v : Json} {k : String} (ha : Claims.Json.Wf a) (h : a.getObjVal? k = .ok v) :
    Claims.Json.Wf v := by
  obtain ⟨m, rfl⟩ := FlockComplete.JsonRT.obj_of_getObjVal h
  cases ha with
  | obj hm hv => exact hv k v (FlockComplete.JsonRT.mem_of_getObjVal hm h).2

/-- **Equivalent well-formed objects have equivalent fields**, under the same keys. -/
theorem getObjVal_equiv {a b : Json} (ha : Claims.Json.Wf a) (hb : Claims.Json.Wf b)
    (he : FlockDeploy.JsonEquiv a b) {k : String} {v : Json} (hv : a.getObjVal? k = .ok v) :
    ∃ w, b.getObjVal? k = .ok w ∧ FlockDeploy.JsonEquiv v w ∧ Claims.Json.Wf w := by
  obtain ⟨m, rfl⟩ := FlockComplete.JsonRT.obj_of_getObjVal hv
  cases ha with
  | obj hm _ =>
    cases hb with
    | obj hn hbv =>
      have hmv := (FlockComplete.JsonRT.mem_of_getObjVal hm hv).2
      obtain ⟨hk, hval⟩ := (FlockDeploy.jsonEquiv_obj_iff hm hn).1 he
      have hs := hk k
      rw [hmv, Option.isSome_some] at hs
      obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hs.symm
      exact ⟨w, FlockComplete.JsonRT.getObjVal_of_get hw, hval k v w hmv hw, hbv k w hw⟩
    | null => simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at he
    | bool => simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at he
    | num => simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at he
    | str => simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at he
    | arr => simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at he

/-- A value equivalent to a string is that string. -/
theorem str_of_jsonEquiv {s : String} {w : Json} (h : FlockDeploy.JsonEquiv (.str s) w) : w = .str s := by
  cases w <;> simp [FlockDeploy.JsonEquiv, FlockDeploy.jsonForm] at h
  subst h
  rfl

/-- `Spot.getD` reads a present field as `getObjVal?` does. -/
theorem getD_of_getObjVal {j v : Json} {k : String} (h : j.getObjVal? k = .ok v) : Spot.getD j k = v := by
  simp only [Spot.getD, Spot.field?, h]
  rfl

/-- What `Public.fv3Bytes` reads: `frame_v3`, its field `k`, the port's string there, and its hex decoding. -/
theorem fv3Bytes_ok {pub : _root_.Flock.Public} {k port : String} {B : ByteArray}
    (h : pub.fv3Bytes k port = .ok B) :
    ∃ fv t s, pub.header.getObjVal? "frame_v3" = .ok fv ∧ fv.getObjVal? k = .ok t ∧
      t.getObjVal? port = .ok (.str s) ∧ _root_.Flock.unhex s = .ok B := by
  unfold _root_.Flock.Public.fv3Bytes _root_.Flock.Public.frameV3 at h
  obtain ⟨fv, hfv, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨t, ht, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨s, hs, h⟩ := FlockSoundness.Refine.ebind_ok h
  exact ⟨fv, t, s, FlockComplete.JsonRT.get_ok hfv, FlockComplete.JsonRT.get_ok ht,
    FlockComplete.JsonRT.str_ok hs, h⟩

/-- `Public.fv3Bytes` reads only the header's `frame_v3`. -/
theorem fv3Bytes_congr {p p' : _root_.Flock.Public}
    (h : p.header.getObjVal? "frame_v3" = p'.header.getObjVal? "frame_v3") (k port : String) :
    p.fv3Bytes k port = p'.fv3Bytes k port := by
  unfold _root_.Flock.Public.fv3Bytes _root_.Flock.Public.frameV3 _root_.Flock.J.get
  rw [h]

/-- **A string under `frame_v3.k.port` of the normalized header is there in the raw header**, as an entry of
`frame_v3.k` and as its value at `port`. -/
theorem raw_entry {hdr : Json} (hw : Claims.Json.Wf hdr) {k port s : String} {fv t : Json}
    (hfv : (_root_.Flock.norm hdr).getObjVal? "frame_v3" = .ok fv) (ht : fv.getObjVal? k = .ok t)
    (hs : t.getObjVal? port = .ok (.str s)) :
    (port, Json.str s) ∈ Spot.entries (Spot.getD (Spot.getD hdr "frame_v3") k) ∧
      Spot.getD (Spot.getD (Spot.getD hdr "frame_v3") k) port = .str s := by
  have hn := FlockDeploy.wf_norm hdr
  obtain ⟨fv', hfv', he₁, hw₁⟩ := getObjVal_equiv hn hw (FlockDeploy.jsonEquiv_norm hdr hw) hfv
  have hwf := wf_of_getObjVal hn hfv
  obtain ⟨t', ht', he₂, hw₂⟩ := getObjVal_equiv hwf hw₁ he₁ ht
  obtain ⟨u, hu, he₃, -⟩ := getObjVal_equiv (wf_of_getObjVal hwf ht) hw₂ he₂ hs
  obtain rfl := str_of_jsonEquiv he₃
  rw [getD_of_getObjVal hfv', getD_of_getObjVal ht', getD_of_getObjVal hu]
  refine ⟨?_, rfl⟩
  obtain ⟨n, rfl⟩ := FlockComplete.JsonRT.obj_of_getObjVal hu
  cases hw₂ with
  | obj hn' _ =>
    have hg := (FlockComplete.JsonRT.mem_of_getObjVal hn' hu).2
    show (port, Json.str s) ∈ n.toArray
    rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray,
      Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hn', ← Std.TreeMap.Raw.get?_eq_getElem?]
    exact hg

/-- The name `PortRoots.idOf` reads at a port in range is the port's. -/
theorem portName {c : _root_.Flock.Circuit} {q : ℕ} (hq : q < c.ports.size) :
    (c.ports[q]?.map (·.name)).getD "" = c.ports[q]!.name := by
  simp [hq]

/-- **Without a population, `FailClosed.inRoot` at the port's domain is `PortRoots.pubRoot`.** -/
theorem inRoot_idOf {c : _root_.Flock.Circuit} {pop : _root_.Flock.Public} (hpn : pop.population = none) {q : ℕ}
    (hq : q < c.ports.size) :
    FlockSoundness.Discharge.FailClosed.inRoot c pop q (_root_.Flock.PortRoots.idOf c pop q) =
      _root_.Flock.PortRoots.pubRoot c pop q := by
  have hs : FlockSoundness.Discharge.FailClosed.rowSchema c.ports[q]! = _root_.Flock.PortRoots.schemaOf c q := by
    simp [FlockSoundness.Discharge.FailClosed.rowSchema, _root_.Flock.PortRoots.schemaOf,
      _root_.Flock.Registered.portSchema, hq]
  have hp : FlockSoundness.Discharge.FailClosed.popOf pop = (pop.n, pop.digests, pop.outputs) := by
    simp [FlockSoundness.Discharge.FailClosed.popOf, hpn]
  unfold FlockSoundness.Discharge.FailClosed.inRoot _root_.Flock.PortRoots.pubRoot _root_.Flock.PortRoots.pubRootWith
  rw [hs, hp]
  cases ht : pop.tables with
  | none =>
    simp only [FlockSoundness.Discharge.FailClosed.inRows, _root_.Flock.PortRoots.countOf,
      _root_.Flock.PortRoots.digOf, _root_.Flock.Public.digest, ht, hp]
    rfl
  | some t =>
    simp only [FlockSoundness.Discharge.FailClosed.inRows, _root_.Flock.PortRoots.countOf,
      _root_.Flock.PortRoots.digOf, ht, Array.getElem!_eq_getD]
    rfl

/-- **The raw header names input port `q`'s registered root and domain**, at a loaded public file `pop` and a
statement's public file `pub` that is `pop` or the sample drawn from it, where `pub`'s `frame_v3` holds `q`'s domain and
its root over `pub`'s rows (`PublicTies.inputs`). -/
theorem roots_of_load {tags : _root_.Flock.Tags} {c : _root_.Flock.Circuit} {pf : ByteArray}
    {pop pub : _root_.Flock.Public} {draw : Option Json} {hdr : Json} {q : ℕ} (hq : q < c.ports.size)
    (hpop : _root_.Flock.HmRow.loadPublic tags { c with outPorts := #[] } pf = .ok pop)
    (hH : headerOf pf = some hdr) (hnone : draw = none → pub = pop)
    (hsome : ∀ d, draw = some d → _root_.Flock.HmOut.drawn c pop d = .ok pub)
    (hin : ∃ dom, pub.fv3Bytes "domain_ids" c.ports[q]!.name = .ok dom ∧
      pub.fv3Bytes "roots" c.ports[q]!.name = .ok (FlockSoundness.Discharge.FailClosed.inRoot c pub q dom)) :
    ∃ rs ds, (c.ports[q]!.name, Json.str rs) ∈ Spot.entries (Spot.getD (Spot.getD hdr "frame_v3") "roots") ∧
      _root_.Flock.unhex rs = .ok (_root_.Flock.PortRoots.pubRoot c pop q) ∧
      Spot.getD (Spot.getD (Spot.getD hdr "frame_v3") "domain_ids") c.ports[q]!.name = .str ds ∧
      _root_.Flock.unhex ds = .ok (_root_.Flock.PortRoots.idOf c pop q) := by
  obtain ⟨h₀, hh₀, hhdr, hpn, m, hm⟩ := loadPublic_raw hpop
  rw [hH, Option.some.injEq] at hh₀
  subst hh₀
  have hmw : m.WF := by
    have hw := FlockDeploy.wf_norm hdr
    rw [← hhdr, hm] at hw
    cases hw with
    | obj hmw _ => exact hmw
  have hwf := wf_of_headerOf hH
  obtain ⟨hfv, hir⟩ := drawn_ties hmw hm hpn hnone hsome
  obtain ⟨dom, hdom, hroot⟩ := hin
  have hid : _root_.Flock.PortRoots.idOf c pop q = dom := by
    unfold _root_.Flock.PortRoots.idOf
    rw [portName hq, ← fv3Bytes_congr hfv, hdom]
    rfl
  rw [hir, ← hid, inRoot_idOf hpn hq] at hroot
  obtain ⟨fv₁, t₁, s₁, hfv₁, ht₁, hs₁, hu₁⟩ := fv3Bytes_ok hroot
  obtain ⟨fv₂, t₂, s₂, hfv₂, ht₂, hs₂, hu₂⟩ := fv3Bytes_ok hdom
  rw [hfv, hhdr] at hfv₁ hfv₂
  refine ⟨s₁, s₂, (raw_entry hwf hfv₁ ht₁ hs₁).1, hu₁, (raw_entry hwf hfv₂ ht₂ hs₂).2, ?_⟩
  rw [hid]
  exact hu₂

end HeaderRoots

/-- **The public file's raw header names each input port's registered root and domain**: at a session statement in
Flock's scope with hidden outputs and one table, `frame_v3.roots` holds an entry for port `q` whose hex string decodes
to the port's published root over the loaded population (`PortRoots.pubRoot`), and `frame_v3.domain_ids` holds a hex
string at `q` that decodes to its domain (`PortRoots.idOf`). -/
theorem header_roots {xS : Msg} {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts))
    (hsc : ScopedHidden xS) (hc1 : ins.read.I.count ≤ 1) {j q : ℕ} (hj : j < sts.size)
    (hq : q < sts[j].c.ports.size) {pop : _root_.Flock.Public}
    (hpop : _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop)
    {hdr : Json} (hH : headerOf ins.read.I.publicFile = some hdr) :
    ∃ rs ds,
      (sts[j].c.ports[q]!.name, Json.str rs) ∈ Spot.entries (Spot.getD (Spot.getD hdr "frame_v3") "roots") ∧
      _root_.Flock.unhex rs = .ok (_root_.Flock.PortRoots.pubRoot sts[j].c pop q) ∧
      Spot.getD (Spot.getD (Spot.getD hdr "frame_v3") "domain_ids") sts[j].c.ports[q]!.name = .str ds ∧
      _root_.Flock.unhex ds = .ok (_root_.Flock.PortRoots.idOf sts[j].c pop q) := by
  obtain ⟨hc, hh⟩ := hsc ins spec sts hset
  obtain ⟨hRu, -⟩ := (_root_.Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : ins.read.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  obtain ⟨-, hsetup⟩ := FlockDeploy.setupOf_some hset
  have hset' := FlockDeploy.Ins.setup_read hsetup
  unfold _root_.Flock.Readable.Read.setup at hset'
  rw [hpi] at hset'
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset'
  obtain ⟨d, hd, hdraw⟩ := hall j hj
  rw [hdraw hc1] at hd
  unfold FlockSoundness.Discharge.HiddenExec.setupOne at hd
  rw [_root_.Flock.Readable.tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  have hspec := FlockSoundness.Discharge.Hidden.setupHidden_spec' hd
  obtain ⟨pop', hpopT, hnone, hsome⟩ := hspec.pub
  have hpop' := (_root_.Flock.Readable.loadPublicH_tagsAt _ _ _ _ _).symm.trans hpopT
  obtain rfl : pop' = pop := Except.ok.inj (hpop'.symm.trans hpop)
  have ties := FlockSoundness.Discharge.FailClosed.ties_of_check (hidden := true) (.inr ⟨rfl, hspec.checkPublic⟩)
  exact HeaderRoots.roots_of_load hq (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop) hH hnone hsome
    (ties.inputs q hq)

end SpotFlock
