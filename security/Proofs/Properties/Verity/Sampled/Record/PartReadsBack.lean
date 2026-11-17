module

public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Flock.Recursive.ZkServerWire
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
public import Proofs.Properties.Verity.Sampled.Record.PartSame

/-! A printed partition object reads back alike, and so do printed registered reads. -/

@[expose] public section

namespace Flock.PartCong

open Lean (Json)

theorem printed_readsBack (what : String) {o : Json} (h : Claims.Json.Good o) :
    ∃ P, (some (RecordText.printT o).toUTF8).mapM (FlockDeploy.jsonOf what) = .ok P ∧ PartSame P (some o) := by
  obtain ⟨j', hp, hs⟩ := FlockComplete.JsonRT.parse_print o h
  rw [FlockComplete.printJ_eq] at hp
  refine ⟨some j', ?_, .inr ⟨j', o, rfl, rfl, hs.symm⟩⟩
  rw [Option.mapM_some]
  unfold FlockDeploy.jsonOf
  rw [FlockSoundness.Discharge.Recursion.ZkServerWire.fromUTF8?_toUTF8]
  simp only [hp]
  rfl

theorem partition_readsBack {o : Json} (h : Claims.Json.Good o) :
    ∃ P, (some (RecordText.printT o).toUTF8).mapM (FlockDeploy.jsonOf "partition") = .ok P ∧
      PartSame P (some o) :=
  printed_readsBack "partition" h

theorem registered_readsBack {r : Option Json} (h : ∀ j, r = some j → Claims.Json.Good j) :
    ∃ R, (r.map fun j => (RecordText.printT j).toUTF8).mapM (FlockDeploy.jsonOf "registered reads") = .ok R ∧
      PartSame R r := by
  cases r with
  | none => exact ⟨none, rfl, .inl rfl⟩
  | some j => exact printed_readsBack "registered reads" (h j rfl)

private theorem list_ok_of_mapM {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → ∀ x ∈ l, ∃ y, f x = .ok y
  | [], _, _, x, hx => by simp at hx
  | a :: l, r, h, x, hx => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | error e => rw [hb] at h; cases h
    | ok b =>
      cases hl : l.mapM f with
      | error e => rw [hb] at h; simp only [bind, Except.bind] at h; rw [hl] at h; cases h
      | ok bs =>
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨b, hb⟩
        · exact list_ok_of_mapM hl x hx

private theorem array_ok_of_mapM {α β ε : Type} {f : α → Except ε β} {a : Array α} {r : Array β}
    (h : a.mapM f = .ok r) : ∀ x ∈ a, ∃ y, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l => exact fun x hx => list_ok_of_mapM hl x (Array.mem_toList_iff.2 hx)

theorem good_of_canon : ∀ (j : Json), Claims.Json.Wf j → ∀ s, Flock.canon j = .ok s → Claims.Json.Good j
  | _, .null, _, _ => .null
  | _, .bool b, _, _ => .bool b
  | .num ⟨m, e⟩, .num _, _, h => by
    rw [Flock.canon] at h
    split at h
    · rename_i he
      obtain rfl : e = 0 := by simpa using he
      exact .num m
    · cases h
  | _, .str t, _, _ => .str t
  | .arr a, .arr hw, _, h => by
    rw [Flock.canon] at h
    cases hp : a.attach.mapM (fun x => Flock.canon x.1) with
    | error _ => rw [hp] at h; cases h
    | ok _ =>
      refine .arr fun x hx => ?_
      obtain ⟨y, hy⟩ := array_ok_of_mapM hp ⟨x, hx⟩ (Array.mem_attach _ _)
      exact good_of_canon x (hw x hx) y hy
  | .obj m, .obj hm hw, _, h => by
    rw [Flock.canon] at h
    cases hp : m.toArray.attach.mapM
        (fun p => do pure (p.1.1, ← Flock.canon p.1.2) : _ → Flock.V (String × String)) with
    | error _ => rw [hp] at h; cases h
    | ok _ =>
      refine .obj hm fun k v hkv => ?_
      have hmem : (k, v) ∈ m.toArray :=
        (Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hm).2
          (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact hkv)
      obtain ⟨y, hy⟩ := array_ok_of_mapM hp ⟨(k, v), hmem⟩ (Array.mem_attach _ _)
      cases hc : Flock.canon v with
      | error _ => rw [hc] at hy; cases hy
      | ok t => exact good_of_canon v (hw k v hkv) t hc

theorem partition_readsBack_of_wf {o : Json} (hw : Claims.Json.Wf o) {s : String} (hc : Flock.canon o = .ok s) :
    ∃ P, (some (RecordText.printT o).toUTF8).mapM (FlockDeploy.jsonOf "partition") = .ok P ∧
      PartSame P (some o) :=
  partition_readsBack (good_of_canon o hw s hc)

theorem raw_partition_readsBack {w : FlockDeploy.Raw} {o : Json}
    (hp : w.partition = some (RecordText.printT o).toUTF8) (hw : Claims.Json.Wf o) {s : String}
    (hc : Flock.canon o = .ok s) :
    ∃ P, w.partition.mapM (FlockDeploy.jsonOf "partition") = .ok P ∧ PartSame P (some o) := by
  rw [hp]
  exact partition_readsBack_of_wf hw hc

private theorem bind_ok' {α β : Type} {x : V α} {k : α → V β} {r : β} (h : x >>= k = .ok r) :
    ∃ a, x = .ok a ∧ k a = .ok r := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

private theorem then_ok {α β γ : Type} {p : Prop} [Decidable p] (hp : p) {g : α → β} {x : V α} {k : β → V γ}
    {y : V γ} {r : γ} (h : (if p then g <$> x >>= k else y) = .ok r) : ∃ a, x = .ok a := by
  split at h
  · cases x with
    | error e => cases h
    | ok a => exact ⟨a, rfl⟩
  · contradiction

private theorem ite_bind_ok {α γ : Type} {p : Prop} [Decidable p] {x y : V α} {k : α → V γ} {r : γ}
    (h : (if p then x >>= k else y >>= k) = .ok r) : ∃ a, k a = .ok r := by
  split at h <;> exact let ⟨a, _, h⟩ := bind_ok' h; ⟨a, h⟩

section

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {draw : Option Json}
  {o : Json} {program own : Option ByteArray}

theorem canon_of_setupH {st : Stmt} (hb : tags.bindings = true)
    (h : Stmt.setupH tags cf pf coins tables draw (some o) program = .ok st) : ∃ s, Flock.canon o = .ok s := by
  rw [setupH_view] at h
  unfold setupHV at h
  obtain ⟨c, h⟩ := ite_bind_ok h
  obtain ⟨f, hf⟩ := then_ok hb h
  exact canon_of_chkPart (c := c) (o := o) hf

theorem canon_of_setupHidden {st : Stmt} (hb : tags.bindings = true)
    (h : Stmt.setupHidden tags cf pf coins tables draw (some o) program own = .ok st) :
    ∃ s, Flock.canon o = .ok s := by
  rw [setupHidden_view] at h
  unfold setupHiddenV at h
  obtain ⟨c, -, h⟩ := bind_ok' h
  obtain ⟨pi, -, h⟩ := bind_ok' h
  obtain ⟨f, hf⟩ := then_ok hb h
  exact canon_of_chkPart (c := c) (o := o) hf

variable {count : Nat} {work : Option (Nat × Array (String × Nat × Nat))} {stratified : Option Nat}
  {closure : Option (Std.HashMap Nat (Array Nat))} {spec : SessionSpec} {sts : Array Stmt}

theorem canon_of_setupTables (hh : tags.hiddenOutputs = true) (hb : tags.bindings = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw (some o) program work stratified closure own =
      .ok (spec, sts)) : ∃ s, Flock.canon o = .ok s := by
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh h with ⟨-, st, hst, -⟩ | ⟨hc, hsz, hall, -⟩
  · exact canon_of_setupHidden hb hst
  · obtain ⟨d, hd⟩ := hall 0 (by omega)
    exact canon_of_setupHidden (tags := { tags with identity := tags.tableIdentity count 0 }) hb hd

theorem partition_readsBack_of_setup (hw : Claims.Json.Wf o) (hh : tags.hiddenOutputs = true)
    (hb : tags.bindings = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw (some o) program work stratified closure own =
      .ok (spec, sts)) :
    ∃ P, (some (RecordText.printT o).toUTF8).mapM (FlockDeploy.jsonOf "partition") = .ok P ∧
      PartSame P (some o) :=
  let ⟨_, hc⟩ := canon_of_setupTables hh hb h
  partition_readsBack_of_wf hw hc

end

end Flock.PartCong
