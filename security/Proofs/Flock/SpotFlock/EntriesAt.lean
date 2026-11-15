module

public import Proofs.Flock.SpotFlock.FrameIsAt
public import Proofs.Flock.SpotFlock.HeaderRoots
public import Proofs.Flock.SpotFlock.Jeq
public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Json.Stmt
public import Proofs.Flock.Soundness.Discharge.Exec.Setup
public import Proofs.Flock.Soundness.Refine.Walk
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The entries the slots name, from round 2's checks

`entriesAt_of`: `EntriesAt` holds where the ports a session reads are no deeper than the code's own (`DepthsAt`),
which a code at least 64 levels deep has for the registered reads (`depthsAt_of`), leaving the published trees'
(`RootedDepthsAt`, which `Depths` proves).

* **The registered half** (`registered_entry`): in an accepted run, the entry round 2 reads each of a session's
  registered reads off (`readKeyAt`, under `readRoots`) has the read's root, leaf count and domain id, of at most
  `2^64` leaves. The verifier's registered reads are its `ports` file's entries (`portsOf_entries`); round 2's
  statement check reads that file (`stmtJsonOf`) and passes each entry (`SecondStage.readOk`), whose `sameAt` holds it
  to the entry it reads off, and to `MAX_LEAVES` (`readOk_entry`), the one `readKey` names in the record
  `readsInterior` picks (`recKeyAt_port`, `recKeyAt_reg`).
* **The roots half** (`interior_entry`): template `t`'s port of the interior record at a port no registered read is
  named like has the port's published root and domain id. Round 2's `rootsOk` holds the record's entry to the
  header's (`rootsOk_entry`), and the header's is the published tree's (`header_roots`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.NetworkCertifier.Bytes (bs)
open Lean (Json)
open FlockSoundness.Refine

/-! ## A record's entries by key -/

/-- **The entry key `k` names in record `rec`**, at the record's naming (`recEntries`). -/
def recKeyAt (rec : Json) (k : SlotKey) : Option Entry :=
  (recEntries rec).find? fun e => decide (e.name = keyOf (Spot.SecondStage.byTemplate rec) k)

theorem keyAt_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) (k : SlotKey) :
    keyAt (irootsOf m₁ ρ) k = recKeyAt rec k := by
  rw [keyAt, entryNamed, entriesOf_irootsOf hm, namedOf_irootsOf hm, recKeyAt]

theorem keyAt_decode {x : Msg} {st : Roots} (hx : decode x = some st) (k : SlotKey) :
    keyAt st.roots k = recKeyAt st.record k := by
  rw [keyAt, entryNamed, entriesOf_decode hx, namedOf_decode hx, recKeyAt]

theorem utf8_inj {a b : String} (h : a.toUTF8.data.toList = b.toUTF8.data.toList) : a = b := by
  have h1 : a.toUTF8 = b.toUTF8 := (bytes_bs a.toUTF8).symm.trans ((congrArg bytes h).trans (bytes_bs b.toUTF8))
  simp only [String.toUTF8_eq_toByteArray] at h1
  exact String.toByteArray_inj.1 h1

theorem recEntries_eq (rec : Json) : recEntries rec =
    (Spot.Record.byPort rec).toList.map (fun kj => entryOf (0, kj)) ++ (regList rec).map fun kj => entryOf (1, kj) := by
  simp only [recEntries, tagged, List.map_append, List.map_map, Function.comp_def]

theorem find?_entryOf (k : UInt8) (l : List (String × Json)) (v : String) :
    (l.map fun kj => entryOf (k, kj)).find? (fun e => decide (e.name = k :: v.toUTF8.data.toList)) =
      (l.find? (·.1 == v)).map fun kj => entryOf (k, kj) := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.map_cons, List.find?_cons]
    by_cases ha : a.1 = v
    · have h1 : decide ((entryOf (k, a)).name = k :: v.toUTF8.data.toList) = true := by simp [entryOf, ha]
      have h2 : (a.1 == v) = true := by simp [ha]
      rw [h1, h2]; rfl
    · have h1 : decide ((entryOf (k, a)).name = k :: v.toUTF8.data.toList) = false := by
        rw [decide_eq_false_iff_not]
        intro he
        simp only [entryOf, List.cons.injEq, eq_self, true_and] at he
        exact ha (utf8_inj he)
      have h2 : (a.1 == v) = false := by simpa using ha
      rw [h1, h2]; exact ih

theorem find?_entryOf_ne {k k' : UInt8} (hk : k ≠ k') (l : List (String × Json)) (m : Msg) :
    (l.map fun kj => entryOf (k, kj)).find? (fun e => decide (e.name = k' :: m)) = none := by
  rw [List.find?_eq_none]
  intro e he
  obtain ⟨kj, -, rfl⟩ := List.mem_map.1 he
  simp [entryOf, hk]

theorem find?_regs (l : List Json) (v : String) :
    (l.filterMap fun e => match Spot.getD e "value" with | .str w => some (w, e) | _ => none).find? (·.1 == v) =
      (l.find? fun e => Spot.jsonStrIs (Spot.getD e "value") v).map fun x => (v, x) := by
  induction l with
  | nil => rfl
  | cons e l ih =>
    simp only [List.filterMap_cons, List.find?_cons]
    cases hw : Spot.getD e "value" with
    | str w =>
      simp only [Spot.jsonStrIs, List.find?_cons]
      by_cases hwv : w = v
      · subst hwv
        simp
      · have h2 : (w == v) = false := by simpa using hwv
        simp only [h2]
        exact ih
    | _ => exact ih

theorem regList_find (rec : Json) (v : String) :
    (regList rec).find? (·.1 == v) = (Spot.SecondStage.registeredEntry rec v).map fun x => (v, x) := by
  unfold regList Spot.SecondStage.registeredEntry
  rw [← Array.find?_toList]
  cases Spot.getD rec "registered" with
  | arr es => exact find?_regs es.toList v
  | _ => rfl

theorem recKeyAt_port (rec : Json) (v : String) :
    recKeyAt rec (.port v) = ((Spot.Record.byPort rec).find? (·.1 == v)).map fun kj => entryOf (0, kj) := by
  show (recEntries rec).find? (fun e => decide (e.name = 0 :: v.toUTF8.data.toList)) = _
  rw [recEntries_eq, List.find?_append, find?_entryOf_ne (k := 1) (k' := 0) (by decide) (regList rec),
    Option.or_none, find?_entryOf, Array.find?_toList]

theorem recKeyAt_interior (rec : Json) (t p : String) :
    recKeyAt rec (.interior t p) = ((Spot.Record.byPort rec).find?
      (·.1 == Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) t p)).map fun kj => entryOf (0, kj) :=
  recKeyAt_port rec (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) t p)

theorem recKeyAt_reg (rec : Json) (v : String) :
    recKeyAt rec (.reg v) = (Spot.SecondStage.registeredEntry rec v).map fun x => entryOf (1, v, x) := by
  show (recEntries rec).find? (fun e => decide (e.name = 1 :: v.toUTF8.data.toList)) = _
  rw [recEntries_eq, List.find?_append, find?_entryOf_ne (k := 0) (k' := 1) (by decide), Option.none_or,
    find?_entryOf, regList_find, Option.map_map]; rfl

/-! ## What `readOk` holds an entry to -/

/-- **An entry `sameAt` holds a registered read to** has the read's root, leaf count and domain id, and the read has at
most `MAX_LEAVES` leaves. -/
theorem entryOf_sameAt {k : UInt8} {nm : String} {e x : Json} {rs ds : String} {r d : ByteArray} {n : ℕ}
    (hs : Spot.SecondStage.sameAt e x = true) (hr : Spot.getD e "root" = .str rs) (hru : _root_.Flock.unhex rs = .ok r)
    (hd : Spot.getD e "domain" = .str ds) (hdu : _root_.Flock.unhex ds = .ok d)
    (hl : Spot.getD e "leaves" = .num ⟨Int.ofNat n, 0⟩) :
    bytes (entryOf (k, nm, x)).root = r ∧ (entryOf (k, nm, x)).leaves = n ∧ bytes (entryOf (k, nm, x)).domain = d ∧
      n ≤ Spot.SecondStage.MAX_LEAVES := by
  unfold Spot.SecondStage.sameAt at hs
  simp only [Bool.and_eq_true] at hs
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := hs
  unfold Spot.SecondStage.leavesOk at h4
  rw [hr] at h1
  rw [hd] at h2
  rw [hl] at h3 h4
  refine ⟨?_, ?_, ?_, by simpa [Spot.nat?, Spot.int?] using h4⟩
  · show bytes (hexMsg (Spot.getD x "root")) = r
    rw [← hexMsg_of_jeq h1]
    exact hexMsg_str hru
  · show (Spot.nat? (Spot.getD x "leaves")).getD 0 = n
    rw [nat?_of_jeq h3]; rfl
  · show bytes (hexMsg (Spot.getD x "domain")) = d
    rw [← hexMsg_of_jeq h2]
    exact hexMsg_str hdu

/-- **A read `readOk` passes has its fields at the entry `readKey` names** in the record `readsInterior` picks: at a
port a link names, the value of its name the interior registration registers, else the boundary port it names; the
interior record's port the read names; or the registered value either record registers, each of which `sameAt` holds
to the read's root, domain and leaves, of at most `MAX_LEAVES`. -/
theorem readOk_entry {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {port v rs ds : String} {e : Json}
    {r d : ByteArray} {n : ℕ}
    (hok : Spot.SecondStage.readOk own (Spot.SecondStage.byTemplate own.interior) p port e = true)
    (hv : Spot.getD e "value" = .str v) (hr : Spot.getD e "root" = .str rs) (hru : _root_.Flock.unhex rs = .ok r)
    (hd : Spot.getD e "domain" = .str ds) (hdu : _root_.Flock.unhex ds = .ok d)
    (hl : Spot.getD e "leaves" = .num ⟨Int.ofNat n, 0⟩) :
    ∃ y, recKeyAt (if readsInterior own p.template port v then own.interior else own.boundary)
        (readKey own p.template port v) = some y ∧ bytes y.root = r ∧ y.leaves = n ∧ bytes y.domain = d ∧
      n ≤ Spot.SecondStage.MAX_LEAVES := by
  have hs : ∀ {k : UInt8} {nm : String} {x : Json}, Spot.SecondStage.sameAt e x = true →
      bytes (entryOf (k, nm, x)).root = r ∧ (entryOf (k, nm, x)).leaves = n ∧
        bytes (entryOf (k, nm, x)).domain = d ∧ n ≤ Spot.SecondStage.MAX_LEAVES :=
    fun hsx => entryOf_sameAt hsx hr hru hd hdu hl
  unfold Spot.SecondStage.readOk at hok
  rw [hv] at hok
  cases hlk : Spot.SecondStage.linkOf own p.template port with
  | some l =>
    rw [hlk] at hok
    simp only [Bool.and_eq_true, beq_iff_eq] at hok
    obtain ⟨hvl, hm⟩ := hok
    cases hri : Spot.SecondStage.registeredEntry own.interior v with
    | some x =>
      cases hro : Spot.SecondStage.rowsOf own p.template port with
      | none => simp [hri, hro] at hm
      | some ro =>
        simp only [hri, hro, Bool.and_eq_true] at hm
        have hRI : readsInterior own p.template port v = true := by simp [readsInterior, hlk, hri]
        have hRK : readKey own p.template port v = .reg v := by simp [readKey, hlk, hri]
        rw [hRI, hRK]
        refine ⟨entryOf (1, v, x), ?_, hs hm.1⟩
        show recKeyAt own.interior (.reg v) = _
        rw [recKeyAt_reg, hri]; rfl
    | none =>
    cases hf : (Spot.Record.byPort own.boundary).find? (·.1 == v) with
    | none => simp [hri, hf] at hm
    | some kb =>
      obtain ⟨k, b⟩ := kb
      cases hpop : Spot.nat? (Spot.getD own.boundary "population") with
      | none => simp [hri, hf, hpop] at hm
      | some m =>
        cases hro : Spot.SecondStage.rowsOf own p.template port with
        | none => simp [hri, hf, hpop, hro] at hm
        | some ro =>
        simp only [hri, hf, hpop, hro, Bool.and_eq_true] at hm
        have hRI : readsInterior own p.template port v = false := by simp [readsInterior, hlk, hri]
        have hRK : readKey own p.template port v = .port v := by simp [readKey, hlk, hri, ← hvl]
        rw [hRI, hRK]
        refine ⟨entryOf (0, k, b), ?_, hs hm.1⟩
        show recKeyAt own.boundary (.port v) = _
        rw [recKeyAt_port, hf]; rfl
  | none =>
    rw [hlk] at hok
    cases hw : Spot.SecondStage.wireOf own p.template port with
    | some w =>
      rw [hw] at hok
      cases hfi : (Spot.Record.byPort own.interior).find? (·.1 == v) with
      | none => simp [hfi] at hok
      | some kx =>
        obtain ⟨k, x⟩ := kx
        cases hpw : Spot.SecondStage.writerOf own w with
        | none => simp [hfi, hpw] at hok
        | some pw =>
          cases hro : Spot.SecondStage.rowsOf own p.template port with
          | none => simp [hfi, hpw, hro] at hok
          | some ro =>
          simp only [hfi, hpw, hro, Bool.and_eq_true] at hok
          obtain ⟨-, ⟨hsx, -⟩, -⟩ := hok
          have hRI : readsInterior own p.template port v = true := by simp [readsInterior, hlk, hfi]
          have hRK : readKey own p.template port v = .port v := by simp [readKey, hlk, hfi]
          rw [hRI, hRK]
          refine ⟨entryOf (0, k, x), ?_, hs hsx⟩
          show recKeyAt own.interior (.port v) = _
          rw [recKeyAt_port, hfi]; rfl
    | none =>
    rw [hw] at hok
    cases hfi : (Spot.Record.byPort own.interior).find? (·.1 == v) with
    | some kx =>
      obtain ⟨k, x⟩ := kx
      simp only [hfi, Bool.and_eq_true, beq_iff_eq] at hok
      obtain ⟨⟨-, hsx⟩, -⟩ := hok
      have hRI : readsInterior own p.template port v = true := by simp [readsInterior, hlk, hfi]
      have hRK : readKey own p.template port v = .port v := by simp [readKey, hlk, hfi]
      rw [hRI, hRK]
      refine ⟨entryOf (0, k, x), ?_, hs hsx⟩
      show recKeyAt own.interior (.port v) = _
      rw [recKeyAt_port, hfi]; rfl
    | none =>
      cases hfb : (Spot.Record.byPort own.boundary).find? (·.1 == v) with
      | some kb => simp [hfi, hfb] at hok
      | none =>
        have hRK : readKey own p.template port v = .reg v := by simp [readKey, hlk, hfi]
        cases hrb : Spot.SecondStage.registeredEntry own.boundary v with
        | some x =>
          simp only [hfi, hfb, hrb] at hok
          have hsx : Spot.SecondStage.sameAt e x = true := hok
          have hRI : readsInterior own p.template port v = false := by simp [readsInterior, hlk, hfi, hfb, hrb]
          rw [hRI, hRK]
          refine ⟨entryOf (1, v, x), ?_, hs hsx⟩
          show recKeyAt own.boundary (.reg v) = _
          rw [recKeyAt_reg, hrb]; rfl
        | none =>
          cases hri : Spot.SecondStage.registeredEntry own.interior v with
          | none => simp only [hfi, hfb, hrb, hri] at hok; cases hok
          | some x =>
            simp only [hfi, hfb, hrb, hri] at hok
            have hsx : Spot.SecondStage.sameAt e x = true := hok
            have hRI : readsInterior own p.template port v = true := by simp [readsInterior, hlk, hfi, hfb, hrb]
            rw [hRI, hRK]
            refine ⟨entryOf (1, v, x), ?_, hs hsx⟩
            show recKeyAt own.interior (.reg v) = _
            rw [recKeyAt_reg, hri]; rfl

/-! ## The verifier's registered reads, as its file states them -/

theorem getD_of_ok {j v : Json} {k : String} (h : j.getObjVal? k = .ok v) : Spot.getD j k = v := by
  simp only [Spot.getD, Spot.field?, h, Except.toOption, Option.getD_some]

theorem getD_mkObj {l : List (String × Json)} (hd : FlockComplete.JsonRT.Distinct l) {k : String} {v : Json}
    (hm : (k, v) ∈ l) : Spot.getD (Json.mkObj l) k = v :=
  getD_of_ok (FlockComplete.JsonRT.getObjVal?_mkObj hd hm)

theorem mapM_ok_mem {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) :
    (∀ y ∈ ys, ∃ x ∈ xs, f x = .ok y) ∧ ∀ x ∈ xs, ∃ y ∈ ys, f x = .ok y := by
  obtain ⟨hsz, hall⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h
  refine ⟨fun y hy => ?_, fun x hx => ?_⟩
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hy
    exact ⟨xs[i], Array.getElem_mem (by omega), hall i (by omega) hi⟩
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact ⟨ys[i], Array.getElem_mem (by omega), hall i hi (by omega)⟩

/-- Each element `f` maps to a value of `ys`, where `ys` is `xs.mapM f`, satisfies what every value `f` returns
there does. -/
theorem mapM_ret {α β : Type} {f : α → _root_.Flock.V β} {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys)
    (P : α → β → Prop) (hP : ∀ x, Ret (f x) (P x)) :
    (∀ y ∈ ys, ∃ x ∈ xs, P x y) ∧ ∀ x ∈ xs, ∃ y ∈ ys, P x y := by
  obtain ⟨h1, h2⟩ := mapM_ok_mem h
  refine ⟨fun y hy => ?_, fun x hx => ?_⟩
  · obtain ⟨x, hx, hf⟩ := h1 y hy
    exact ⟨x, hx, hP x y hf⟩
  · obtain ⟨y, hy, hf⟩ := h2 x hx
    exact ⟨y, hy, hP x y hf⟩

/-- **What a `ports` file says of a port it names `nm`** at its entry `e`: the port's name, and the hex strings of its
root and domain id, its leaf count and its value. -/
abbrev PortFields (nm : String) (e : Json) (reg : _root_.Flock.Registered.Port) : Prop :=
  reg.name = nm ∧ (∃ rs, e.getObjVal? "root" = .ok (.str rs) ∧ _root_.Flock.unhex rs = .ok reg.root) ∧
    (∃ ds, e.getObjVal? "domain" = .ok (.str ds) ∧ _root_.Flock.unhex ds = .ok reg.domain) ∧
    e.getObjVal? "leaves" = .ok (.num ⟨Int.ofNat reg.leaves, 0⟩) ∧ e.getObjVal? "value" = .ok (.str reg.value)

/-- **A `ports` file's ports are its entries**: each port is an entry of its `ports` object under the port's name, with
the port's fields (`PortFields`), and each entry's name is a port's. -/
theorem portsOf_entries {j : Json} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.Registered.portsOf j = .ok own) :
    ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
      (∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ PortFields reg.name e reg) ∧
      ∀ kv ∈ kvs.toArray, ∃ reg ∈ own, reg.name = kv.1 := by
  revert own
  show Ret (_root_.Flock.Registered.portsOf j) fun own => ∃ kvs, j.getObjVal? "ports" = .ok (.obj kvs) ∧
    (∀ reg ∈ own, ∃ e, (reg.name, e) ∈ kvs.toArray ∧ PortFields reg.name e reg) ∧
    ∀ kv ∈ kvs.toArray, ∃ reg ∈ own, reg.name = kv.1
  rw [_root_.Flock.Registered.portsOf.eq_def]
  ret_walk [Array.mapM]
  all_goals
    obtain ⟨hall, hall'⟩ := mapM_ret (by assumption)
        (fun (kv : String × Json) (reg : _root_.Flock.Registered.Port) => PortFields kv.1 kv.2 reg) fun kv => by
      obtain ⟨nm, e⟩ := kv
      dsimp (config := { zeta := false }) only
      ret_walk [_root_.Flock.J.str, _root_.Flock.J.nat, _root_.Flock.unhex]
      all_goals exact ⟨rfl, ⟨_, FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "root" _ = .ok _›, ‹_›⟩,
        ⟨_, FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "domain" _ = .ok _›, ‹_›⟩,
        FlockComplete.JsonRT.nat_ok ‹_root_.Flock.J.nat e "leaves" _ = .ok _›,
        FlockComplete.JsonRT.str_ok ‹_root_.Flock.J.str e "value" _ = .ok _›⟩
    refine ⟨_, by assumption, fun reg hreg => ?_, fun kv hkv => ?_⟩
    · obtain ⟨⟨nm, e⟩, hkv, hP⟩ := hall reg hreg
      obtain rfl : reg.name = nm := hP.1
      exact ⟨e, hkv, hP⟩
    · obtain ⟨reg, hreg, hP⟩ := hall' kv hkv
      exact ⟨reg, hreg, hP.1⟩

theorem portsOf_of_ofJson {j : Json} {own : Array _root_.Flock.Registered.Port}
    (h : _root_.Flock.Registered.ofJson j = .ok own) : _root_.Flock.Registered.portsOf j = .ok own := by
  obtain ⟨ps, hps, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases hd : _root_.Flock.Registered.dupName? ps.toList with
  | some nm => rw [hd] at h; cases h
  | none => rw [hd] at h; cases h; exact hps

theorem ins_reads_public {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    w.registered.mapM (fun b => do _root_.Flock.Registered.ofJson (← FlockDeploy.jsonOf "registered reads" b)) =
      .ok s.registered ∧ s.read.I.publicFile = w.publicFile := by
  unfold FlockDeploy.Raw.ins at h
  dsimp only at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨‹_›, rfl⟩

theorem map_some_ok {α : Type} {x : Except String α} {a : α} (h : (some <$> x) = Except.ok (some a)) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

/-- **The verifier's registered reads are its `registered reads` file's ports**, where it has some. -/
theorem registered_of_ins {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (hw : w.ins = .ok s)
    {own : Array _root_.Flock.Registered.Port} (ho : s.registered = some own) :
    ∃ b j, w.registered = some b ∧ FlockDeploy.jsonOf "registered reads" b = .ok j ∧
      _root_.Flock.Registered.ofJson j = .ok own := by
  have hm := (ins_reads_public hw).1
  rw [ho] at hm
  cases hr : w.registered with
  | none => rw [hr] at hm; cases hm
  | some b =>
    rw [hr, Option.mapM_some] at hm
    obtain ⟨j, hj, hoj⟩ := FlockSoundness.Refine.ebind_ok (map_some_ok hm)
    exact ⟨b, j, rfl, hj, hoj⟩

theorem ofJson?_fields {j : Json} {S : Spot.SecondStage.Stmt} (hS : Spot.SecondStage.Stmt.ofJson? j = some S) :
    Spot.getD j "registered" = S.registered ∧ Spot.getD j "template" = .str S.template ∧
      Spot.getD j "header" = S.header := by
  unfold Spot.SecondStage.Stmt.ofJson? at hS
  split at hS
  · cases hS
  · split at hS <;> cases hS <;> exact ⟨by assumption, by assumption, by assumption⟩

theorem ins_of_setupOf {xS : Msg} {w : FlockDeploy.Raw} (hw : FlockDeploy.decode xS = some w) {ins : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts)) : w.ins = .ok ins := by
  obtain ⟨hi, -⟩ := FlockDeploy.setupOf_some hset
  unfold FlockDeploy.insOf at hi
  obtain ⟨w', hw', hi⟩ := Option.bind_eq_some_iff.1 hi
  obtain rfl : w = w' := Option.some.inj (hw.symm.trans hw')
  by_cases hz : w.zk ≠ true
  · rw [ite_eq_left hz] at hi
    cases hi
  · rw [ite_eq_right hz] at hi
    change w.ins.toOption = some ins at hi
    cases hwi : w.ins with
    | error e => rw [hwi] at hi; cases hi
    | ok s => rw [hwi] at hi; cases hi; rfl

theorem map_some_ne_none {α : Type} {x : Except String α} (h : (some <$> x) = Except.ok none) : False := by
  cases x with
  | error e => cases h
  | ok b => cases h

/-- **The verifier has no registered reads only where its statement has no `registered reads` file.** -/
theorem registered_none {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (hw : w.ins = .ok s) (hn : s.registered = none) :
    w.registered = none := by
  have hm := (ins_reads_public hw).1
  rw [hn] at hm
  cases hr : w.registered with
  | none => rfl
  | some b =>
    rw [hr, Option.mapM_some] at hm
    exact (map_some_ne_none hm).elim

/-! ## The published roots -/

/-- **A port `rootsOk` passes the header's root of** is template `t`'s port of the record at an entry with the
header's root and domain id, for a port the header roots and `reads` doesn't read. -/
theorem rootsOk_entry {named : Bool} {rec : Json} {t : String} {hdr : Json} {reads : Array String}
    {p rs ds : String} {r d : ByteArray} (hok : Spot.SecondStage.rootsOk named rec t hdr reads = true)
    (hm : (p, Json.str rs) ∈ Spot.entries (Spot.getD (Spot.getD hdr "frame_v3") "roots"))
    (hp : reads.contains p = false) (hru : _root_.Flock.unhex rs = .ok r)
    (hd : Spot.getD (Spot.getD (Spot.getD hdr "frame_v3") "domain_ids") p = .str ds)
    (hdu : _root_.Flock.unhex ds = .ok d) :
    ∃ y, ((Spot.Record.byPort rec).find? (·.1 == Spot.SecondStage.portOf named t p)).map
        (fun kj => entryOf (0, kj)) = some y ∧ bytes y.root = r ∧ bytes y.domain = d := by
  unfold Spot.SecondStage.rootsOk at hok
  simp only [Bool.and_eq_true] at hok
  obtain ⟨⟨-, hall⟩, -⟩ := hok
  have h1 := Array.all_eq_true_iff_forall_mem.1 hall (p, Json.str rs)
    (Array.mem_filter.2 ⟨hm, by simp only [hp, Bool.not_false]⟩)
  cases hf : (Spot.Record.byPort rec).find? (·.1 == Spot.SecondStage.portOf named t p) with
  | none => simp [hf] at h1
  | some ke =>
    obtain ⟨k, e⟩ := ke
    simp only [hf, Bool.and_eq_true] at h1
    obtain ⟨hr1, hd1⟩ := h1
    rw [hd] at hd1
    refine ⟨entryOf (0, k, e), rfl, ?_, ?_⟩
    · show bytes (hexMsg (Spot.getD e "root")) = r
      rw [← hexMsg_of_jeq hr1]
      exact hexMsg_str hru
    · show bytes (hexMsg (Spot.getD e "domain")) = d
      rw [← hexMsg_of_jeq hd1]
      exact hexMsg_str hdu

variable (h : Held)

/-- **Round 2's statement for template `t`** names the template, reads the session's public file's header, and reads
its registered reads file (null where it has none). -/
theorem stmtJsonOf_fields {t : ℕ} {w : FlockDeploy.Raw} {S : Spot.SecondStage.Stmt} (hS : stmtJsonOf h t w = some S) :
    S.template = h.names.getD t "" ∧ headerOf w.publicFile = some S.header ∧
      (∀ b, w.registered = some b → (FlockDeploy.jsonOf "registered reads" b).toOption = some S.registered) ∧
      (w.registered = none → S.registered = .null) := by
  unfold stmtJsonOf at hS
  obtain ⟨hdr, hH, hS⟩ := Option.bind_eq_some_iff.1 hS
  obtain ⟨d, -, hS⟩ := Option.bind_eq_some_iff.1 hS
  obtain ⟨dr, -, hS⟩ := Option.bind_eq_some_iff.1 hS
  have key : ∀ r, Spot.SecondStage.Stmt.ofJson? (Json.mkObj [("draw", dr), ("header", hdr), ("registered", r),
      ("template", Json.str (h.names.getD t ""))]) = some S →
      r = S.registered ∧ S.template = h.names.getD t "" ∧ hdr = S.header := by
    intro r hr
    obtain ⟨h1, h2, h3⟩ := ofJson?_fields hr
    have hd : FlockComplete.JsonRT.Distinct
        [("draw", dr), ("header", hdr), ("registered", r), ("template", Json.str (h.names.getD t ""))] := by simp
    rw [getD_mkObj hd (k := "registered") (v := r) (by simp)] at h1
    rw [getD_mkObj hd (k := "template") (v := Json.str (h.names.getD t "")) (by simp)] at h2
    rw [getD_mkObj hd (k := "header") (v := hdr) (by simp)] at h3
    exact ⟨h1, (Json.str.inj h2).symm, h3⟩
  dsimp only at hS
  split at hS
  · rename_i b hb
    obtain ⟨r, hr, hS⟩ := Option.bind_eq_some_iff.1 hS
    obtain ⟨h1, h2, h3⟩ := key r hS
    refine ⟨h2, h3 ▸ hH, fun b' hb' => ?_, fun hn => ?_⟩
    · rw [hb] at hb'
      cases hb'
      rw [hr, h1]
    · rw [hb] at hn
      cases hn
  · rename_i hb
    obtain ⟨h1, h2, h3⟩ := key .null hS
    refine ⟨h2, h3 ▸ hH, fun b' hb' => ?_, fun _ => h1.symm⟩
    rw [hb] at hb'
    cases hb'

theorem readsOk_true {t : ℕ} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hr : readsOk h t x m₁ xS c₀ c₁ = true) :
    ∃ st rec draw w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      FlockDeploy.decode xS = some w ∧ stmtJsonOf h t w = some S ∧ codesOk h (ownOf h st rec draw c₀) t S = true := by
  unfold readsOk at hr
  cases hst : decode x with
  | none => rw [hst] at hr; cases hr
  | some st =>
    cases hrec : jsonMsg m₁ with
    | none => rw [hst, hrec] at hr; cases hr
    | some rec =>
      cases hdraw : secondDraw h c₀ c₁ with
      | none => rw [hst, hrec, hdraw] at hr; cases hr
      | some draw =>
        cases hw : FlockDeploy.decode xS with
        | none => rw [hst, hrec, hdraw, hw] at hr; cases hr
        | some w =>
          rw [hst, hrec, hdraw, hw] at hr
          simp only [Bool.and_eq_true] at hr
          obtain ⟨⟨-, hc⟩, -⟩ := hr
          cases hS : stmtJsonOf h t w with
          | none => rw [hS] at hc; cases hc
          | some S => rw [hS] at hc; exact ⟨st, rec, draw, w, S, rfl, rfl, rfl, rfl, hS, hc⟩

theorem keyAt_readRoots {x m₁ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {st : Roots} {rec draw : Json}
    (hst : decode x = some st) (hrec : jsonMsg m₁ = some rec) (hdraw : secondDraw h c₀ c₁ = some draw)
    (ρ t : ℕ) (port v : String) :
    keyAt (readRoots h x m₁ c₀ c₁ ρ t port v) (readKeyAt h x m₁ c₀ c₁ t port v) =
      recKeyAt (if readsInterior (ownOf h st rec draw c₀) (h.names.getD t "") port v then rec else st.record)
        (readKey (ownOf h st rec draw c₀) (h.names.getD t "") port v) := by
  unfold readRoots readKeyAt
  rw [hst, hrec, hdraw]
  dsimp only
  cases readsInterior (ownOf h st rec draw c₀) (h.names.getD t "") port v
  · exact keyAt_decode hst _
  · exact keyAt_irootsOf hrec ρ _

/-- **`EntriesAt`'s registered half**: in an accepted run, the entry round 2 reads each of a checked unit's session's
registered reads off, under the roots it reads it under, has the read's root, leaf count and domain id, and the read
has at most `MAX_LEAVES` leaves. -/
theorem registered_entry {x m₁ m₂ : Msg} {s} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u ρ : ℕ} {ins : FlockDeploy.Ins}
    {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hset : FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts))
    {own : Array _root_.Flock.Registered.Port} {reg : _root_.Flock.Registered.Port}
    (hown : ins.registered = some own) (hreg : reg ∈ own) :
    ∃ e, keyAt (readRoots h x m₁ c₀ c₁ ρ (templateOf h.ms u) reg.name reg.value)
        (readKeyAt h x m₁ c₀ c₁ (templateOf h.ms u) reg.name reg.value) = some e ∧
      bytes e.root = reg.root ∧ e.leaves = reg.leaves ∧ bytes e.domain = reg.domain ∧
      reg.leaves ≤ Spot.SecondStage.MAX_LEAVES := by
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins (ins_of_setupOf hw hset) hown
  obtain ⟨-, -, hSr, -⟩ := stmtJsonOf_fields h hS
  have hSj : S.registered = jR := by
    have h4 := hSr bR hbR
    rw [hjR] at h4
    exact (Option.some.inj h4).symm
  obtain ⟨kvs, hports, hent, -⟩ := portsOf_entries (portsOf_of_ofJson hoj)
  obtain ⟨ej, hm, -, ⟨rs, hr, hru⟩, ⟨ds, hd, hdu⟩, hl, hv⟩ := hent reg hreg
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hents : Spot.entries (Spot.getD S.registered "ports") = kvs.toArray := by
    simp only [hSj, getD_of_ok hports, Spot.entries]
  rw [hents] at hreads
  have hrd : Spot.SecondStage.readOk (ownOf h st rec draw c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec draw c₀).sample.size (templateOf h.ms u)) reg.name ej = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hm
  obtain ⟨y, hy, hyr, hyl, hyd, hyb⟩ :=
    readOk_entry hrd (getD_of_ok hv) (getD_of_ok hr) hru (getD_of_ok hd) hdu (getD_of_ok hl)
  refine ⟨y, ?_, hyr, hyl, hyd, hyb⟩
  rw [keyAt_readRoots h hst hrec hdraw]
  exact hy

/-- **`EntriesAt`'s roots half**: in an accepted run, template `t`'s port of the interior record at a port of a
checked unit's one-table session that no registered read is named like has the root and domain id of the port's
published tree over the population the session's public file loads, which the header names (`header_roots`): round
2's `rootsOk` holds the record's entry to the header's. -/
theorem interior_entry {x m₁ m₂ : Msg} {s} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u ρ : ℕ}
    {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hset : FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts))
    {j : ℕ} (hj : j < sts.size) {q : ℕ} {pop : _root_.Flock.Public} (hc1 : ins.read.I.count ≤ 1)
    (hq : q < sts[j].c.ports.size)
    (hnr : ∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name)
    (hpop : _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop) :
    ∃ e, keyAt (irootsOf m₁ ρ) (.interior (h.names.getD (templateOf h.ms u) "") sts[j].c.ports[q]!.name) = some e ∧
      bytes e.root = (pubPort sts[j].c pop q).root ∧ bytes e.domain = (pubPort sts[j].c pop q).domain := by
  have hat := stmtOkAt_unit h hx hok hu
  have hsc := sessionScoped h x m₁ (unitStmt h m₂ u) c₀ c₁ _ hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  have hwi := ins_of_setupOf hw hset
  obtain ⟨hSt, hSH, hSr, hSn⟩ := stmtJsonOf_fields h hS
  have hH : headerOf ins.read.I.publicFile = some S.header := by
    rw [(ins_reads_public hwi).2]
    exact hSH
  obtain ⟨rs, ds, hmem, hru, hdm, hdu⟩ := header_roots hset hsc hc1 hj hq hpop hH
  obtain ⟨-, -, -, -, hroots, -, -, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  have hnot : ((Spot.entries (Spot.getD S.registered "ports")).map (·.1)).contains
      sts[j].c.ports[q]!.name = false := by
    cases hreg : ins.registered with
    | none =>
      rw [hSn (registered_none hwi hreg)]
      show ((#[] : Array (String × Json)).map (·.1)).contains _ = false
      simp
    | some own =>
      obtain ⟨bR, jR, hbR, hjR, hoj⟩ := registered_of_ins hwi hreg
      have h4 := hSr bR hbR
      rw [hjR] at h4
      obtain ⟨kvs, hports, -, hnames⟩ := portsOf_entries (portsOf_of_ofJson hoj)
      rw [← Option.some.inj h4, getD_of_ok hports, Bool.eq_false_iff]
      intro hcn
      obtain ⟨kv, hkv, hkn⟩ := Array.mem_map.1 (Array.contains_iff_mem.1 hcn)
      obtain ⟨reg, hreg', hrn⟩ := hnames kv hkv
      exact hnr own hreg reg hreg' (hrn.trans hkn)
  rw [hSt] at hroots
  obtain ⟨y, hy, hyr, hyd⟩ := rootsOk_entry hroots hmem hnot hru hdm hdu
  refine ⟨y, ?_, hyr, hyd⟩
  rw [keyAt_irootsOf hrec ρ, recKeyAt_interior]
  exact hy

/-- **The ports a checked unit's session reads are no deeper than `c`'s own**: its registered reads, and the published
trees of a one-table session's ports that no registered read is named like, over any population its public file
loads. -/
def DepthsAt (c : Code) : Prop :=
  ∀ x s (m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) wF u ins spec sts,
    parse h x = some s → stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) →
    (∀ own reg, ins.registered = some own → reg ∈ own → _root_.Flock.log2ceil reg.leaves ≤ depth c.own) ∧
    (∀ j (hj : j < sts.size) q pop, ins.read.I.count ≤ 1 → q < sts[j].c.ports.size →
      (∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name) →
      _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop →
      _root_.Flock.log2ceil (pubPort sts[j].c pop q).leaves ≤ depth c.own)

/-- **The published trees a checked unit's one-table session reads are no deeper than `c`'s own**: each port's that
no registered read is named like, over any population its public file loads. -/
def RootedDepthsAt (c : Code) : Prop :=
  ∀ x s (m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) wF u ins spec sts,
    parse h x = some s → stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) →
    ∀ j (hj : j < sts.size) q pop, ins.read.I.count ≤ 1 → q < sts[j].c.ports.size →
      (∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name) →
      _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop →
      _root_.Flock.log2ceil (pubPort sts[j].c pop q).leaves ≤ depth c.own

/-- A tree of at most `MAX_LEAVES` leaves is at most 64 levels deep. -/
theorem log2ceil_le_64 {n : ℕ} (hn : n ≤ Spot.SecondStage.MAX_LEAVES) : _root_.Flock.log2ceil n ≤ 64 := by
  unfold _root_.Flock.log2ceil
  split
  · omega
  · rename_i h1
    have h2 : n - 1 < 2 ^ 64 := Nat.lt_of_lt_of_le (Nat.sub_lt (by omega) Nat.one_pos) hn
    have h3 : Nat.log2 (n - 1) < 64 := (Nat.log2_lt (by omega)).2 h2
    omega

/-- **`DepthsAt` at a code at least 64 levels deep**: round 2 holds each registered read to at most `MAX_LEAVES`
leaves (`registered_entry`), and the published trees' depths are `RootedDepthsAt`. -/
theorem depthsAt_of {c : Code} (h64 : 64 ≤ depth c.own) (hP : RootedDepthsAt h c) : DepthsAt h c := by
  intro x s m₁ m₂ c₀ c₁ wF u ins spec sts hx hok hu hck hset
  refine ⟨fun own reg hown hreg => ?_, hP x s m₁ m₂ c₀ c₁ wF u ins spec sts hx hok hu hck hset⟩
  obtain ⟨-, -, -, -, -, hb⟩ := registered_entry h (ρ := 0) hx hok hu hset hown hreg
  exact (log2ceil_le_64 hb).trans h64

/-- **`EntriesAt` from the ports' depths**: its registered half is `registered_entry` and its roots half
`interior_entry`. -/
theorem entriesAt_of {c : Code} (hD : DepthsAt h c) : EntriesAt h c := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset
  obtain ⟨hDr, hDp⟩ := hD x s m₁ m₂ c₀ c₁ wF u ins spec sts hx hok hu hck hset
  refine ⟨fun own reg hown hreg => ?_, fun j hj q pop hc1 hq hnr hpop => ?_⟩
  · obtain ⟨e, he, hr, hl, hdm, -⟩ := registered_entry h (ρ := ρ) hx hok hu hset hown hreg
    exact ⟨e, he, hr, hl, hdm, hDr own reg hown hreg⟩
  · obtain ⟨e, he, hr, hdm⟩ := interior_entry h (ρ := ρ) hx hok hu hset hj hc1 hq hnr hpop
    exact ⟨e, he, hr, hdm, hDp j hj q pop hc1 hq hnr hpop⟩

end SpotFlock
