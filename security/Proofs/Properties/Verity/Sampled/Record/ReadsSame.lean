module

public import Flock.Registered
public import Proofs.Properties.Verity.Sampled.Record.PartSame
import all Lean.Data.Json.Basic

/-!
# Alike registered reads are the same ports

A session statement carries the verifier's registered reads as text. Parsed back, the value is alike the one the
verifier holds (`Claims.Json.Same`) and equal to it only up to the shape of the tree an object keeps its keys in. The
verifier reads its ports off the value with `Flock.Registered.ofJson`, and that reading is the same on alike values:

* `get_same`, `has_same`, `str_same`, `nat_same`, `keysIn_same`: the readers of one key, and the test of an object's
  keys;
* `portsOf_same`, `ofJson_same`: the file's ports, and the ports once each name names one;
* `PartSame.ofJson`: so two readings of a statement's registered reads that are one value or two alike values
  (`PartSame`) are the same ports.

`Flock.Registered.portsOf` matches on the file's `ports` and on each port's `reads`, which are only alike on the two
sides, so its text is restated with each read of the file an argument (`portsV`, with `portsTail` for the ports and
`portV` for one port). The restated texts are private to this file, so no statement another file reads names one, and
together they are the verifier's function by `rfl` (`ports_view`), so a change to the verifier breaks this file's build
and nothing else.
-/

@[expose] public section

namespace Flock.ReadsSame

open Lean (Json)
open Claims.Json (Same)
open Flock.PartCong (AlikeE alikeE_get getStr_same getNat_same keysArr_same PartSame)

/-! ## The readers of one key -/

/-- `J.get` on two alike values: one refusal, or two alike values. -/
theorem get_same {j j' : Json} (h : Same j j') (k ctx : String) :
    (∃ e, J.get j k ctx = .error e ∧ J.get j' k ctx = .error e) ∨
      ∃ a b, J.get j k ctx = .ok a ∧ J.get j' k ctx = .ok b ∧ Same a b := by
  unfold J.get
  rcases alikeE_get h k with ⟨e, e', e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩
  · rw [e₁, e₂]; exact Or.inl ⟨_, rfl, rfl⟩
  · rw [e₁, e₂]; exact Or.inr ⟨a, b, rfl, rfl, hs⟩

theorem has_same {j j' : Json} (h : Same j j') (k : String) : J.has j k = J.has j' k := by
  unfold J.has
  rcases alikeE_get h k with ⟨e, e', e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩ <;> rw [e₁, e₂]

theorem str_same {j j' : Json} (h : Same j j') (k ctx : String) : J.str j k ctx = J.str j' k ctx := by
  unfold J.str
  rcases get_same h k ctx with ⟨e, e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩
  · rw [e₁, e₂]
  · rw [e₁, e₂]
    show J.asStr a _ = J.asStr b _
    unfold J.asStr
    rw [getStr_same hs]

theorem nat_same {j j' : Json} (h : Same j j') (k ctx : String) : J.nat j k ctx = J.nat j' k ctx := by
  unfold J.nat
  rcases get_same h k ctx with ⟨e, e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩
  · rw [e₁, e₂]
  · rw [e₁, e₂]
    show J.asNat a _ = J.asNat b _
    unfold J.asNat
    rw [getNat_same hs]

theorem keysIn_same : ∀ {a b : Json}, Same a b → ∀ keys : List String,
    Registered.keysIn a keys = Registered.keysIn b keys
  | _, _, .null, _ => rfl
  | _, _, .bool _, _ => rfl
  | _, _, .num _, _ => rfl
  | _, _, .str _, _ => rfl
  | _, _, .arr _ _, _ => rfl
  | .obj m, .obj n, .obj hm hn hk _, keys => by
    have e : ∀ t : Std.TreeMap.Raw String Json,
        Registered.keysIn (.obj t) keys = (t.toArray.map (·.1)).all fun k => keys.contains k := fun t => by
      show (t.toArray.all fun (k, _) => keys.contains k) = _
      rw [Array.all_map]
      rfl
    rw [e m, e n, keysArr_same hm hn hk]

/-! ## A port's reads -/

/-- One read position, as `portsOf` reads it. -/
private def readPos (ctx : String) (r : Json) : V (Option Nat) :=
  match r with
  | .null => pure none
  | r => some <$> J.asNat r s!"{ctx}: a read position"

private theorem readPos_same (ctx : String) {r r' : Json} (h : Same r r') : readPos ctx r = readPos ctx r' := by
  cases h <;> rfl

private theorem mapM_same {β : Type} {f : Json → V β} (hf : ∀ {r r' : Json}, Same r r' → f r = f r')
    {a b : Array Json} (hs : a.size = b.size)
    (he : ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i]) : a.mapM f = b.mapM f := by
  have e : a.map f = b.map f := by
    apply Array.ext
    · rw [Array.size_map, Array.size_map]; exact hs
    · intro i h₁ h₂
      rw [Array.getElem_map, Array.getElem_map]
      exact hf (he i _ _)
  have m : ∀ x : Array Json, x.mapM f = (x.map f).mapM id := fun x =>
    (Array.mapM_map (f := f) (g := id) (xs := x)).symm
  rw [m a, m b, e]

/-- A port's reads, handed to what `portsOf` does with them next. -/
private def readsK {β : Type} (ctx : String) (e : Json) (k : Array (Option Nat) → V β) : V β :=
  J.arr e "reads" ctx >>= fun a => a.mapM (readPos ctx) >>= k

private theorem readsK_same {β : Type} (ctx : String) {e e' : Json} (h : Same e e')
    (k : Array (Option Nat) → V β) : readsK ctx e k = readsK ctx e' k := by
  unfold readsK J.arr
  rcases get_same h "reads" ctx with ⟨x, e₁, e₂⟩ | ⟨a, b, e₁, e₂, hs⟩
  · rw [e₁, e₂]
  · rw [e₁, e₂]
    cases hs with
    | arr hs he =>
      show Array.mapM (readPos ctx) _ >>= k = Array.mapM (readPos ctx) _ >>= k
      rw [mapM_same (readPos_same ctx) hs he]
    | null => rfl
    | bool => rfl
    | num => rfl
    | str => rfl
    | obj => rfl

/-! ## One port -/

/-- `portsOf`'s reading of one port, with each read of the port's entry an argument. -/
private def portV (v1 : Bool) (name : String) (kin : Bool) (has : String → Bool) (nat : String → V Nat)
    (str : String → V String) (rds : (Array (Option Nat) → V Registered.Port) → V Registered.Port) :
    V Registered.Port := do
  let ctx := "registered reads"
  if v1 && !kin then
    throw s!"{ctx}: port {name}: a {Registered.FORMAT} port holds keys of {Registered.PORT_KEYS} and no other"
  if !v1 && has Registered.READS_FROM_KEY then
    throw s!"{ctx}: port {name}: {Registered.READS_FROM_KEY} is a {Registered.FORMAT} key, \
      and a {Registered.FORMAT_V0} file's reads start at unit 0"
  let readsFrom ← if has Registered.READS_FROM_KEY then nat Registered.READS_FROM_KEY else pure 0
  let root ← unhex (← str "root")
  let domain ← unhex (← str "domain")
  let leaves ← nat "leaves"
  if root.size != 64 || domain.size != 64 || leaves == 0 then
    throw s!"{ctx}: port {name}: a 64-byte root and domain, and at least one leaf"
  let wordBits ← if has "word_bits" then nat "word_bits" else pure 16
  let words ← if has "words" then some <$> nat "words" else pure none
  if !(wordBits == 16 || wordBits == 32) || words == some 0 || (words.isNone && wordBits != 16) then
    throw s!"{ctx}: port {name}: leaves of 16 or 32 bits, at least one to a row, and a row's words when they are 32"
  let schema ← if has Registered.SCHEMA_KEY then str Registered.SCHEMA_KEY else pure HmRow.ROW_SCHEMA
  if !Registered.SCHEMAS.contains schema then
    throw s!"{ctx}: port {name}: {Registered.SCHEMA_KEY} is one of {Registered.SCHEMAS}, not {schema}"
  if schema != HmRow.ROW_SCHEMA && words.isNone then
    throw s!"{ctx}: port {name}: a {schema} row is a bit row of its words' bits, and the port names no words"
  rds fun reads => do
    return { name, value := ← str "value", root, leaves, words, wordBits, domain, readsRel := reads, schema,
             readsFrom }

/-- One port's reading, of its name and its entry. -/
private def portE (v1 : Bool) (p : String × Json) : V Registered.Port :=
  portV v1 p.1 (Registered.keysIn p.2 Registered.PORT_KEYS) (J.has p.2) (fun k => J.nat p.2 k "registered reads")
    (fun k => J.str p.2 k "registered reads") (readsK "registered reads" p.2)

private theorem portE_same (v1 : Bool) (name : String) {e e' : Json} (h : Same e e') :
    portE v1 (name, e) = portE v1 (name, e') := by
  have h1 : J.has e = J.has e' := funext fun k => has_same h k
  have h2 : (fun k => J.nat e k "registered reads") = fun k => J.nat e' k "registered reads" :=
    funext fun k => nat_same h k _
  have h3 : (fun k => J.str e k "registered reads") = fun k => J.str e' k "registered reads" :=
    funext fun k => str_same h k _
  have h4 : readsK (β := Registered.Port) "registered reads" e = readsK "registered reads" e' :=
    funext fun k => readsK_same _ h k
  show portV v1 name (Registered.keysIn e Registered.PORT_KEYS) (J.has e) (fun k => J.nat e k "registered reads")
      (fun k => J.str e k "registered reads") (readsK "registered reads" e) =
    portV v1 name (Registered.keysIn e' Registered.PORT_KEYS) (J.has e') (fun k => J.nat e' k "registered reads")
      (fun k => J.str e' k "registered reads") (readsK "registered reads" e')
  rw [keysIn_same h, h1, h2, h3, h4]

/-! ## The file's ports -/

/-- `portsOf`'s reading of the file's ports, with the file's read of them an argument. -/
private def portsTail (v1 : Bool) (ports : Except String Json) : V (Array Registered.Port) := do
  let ctx := "registered reads"
  let .ok (.obj kvs) := ports | throw s!"{ctx}: ports is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: no port"
  kvs.toArray.mapM (portE v1)

/-- `portsOf`'s text, with each read of the file an argument. -/
private def portsV (fmt : V String) (kin : Bool) (tail : Bool → V (Array Registered.Port)) :
    V (Array Registered.Port) := do
  let ctx := "registered reads"
  let format ← fmt
  if format != Registered.FORMAT && format != Registered.FORMAT_V0 then
    throw s!"{ctx}: not a {Registered.FORMAT} file (nor a {Registered.FORMAT_V0} one)"
  let v1 := format == Registered.FORMAT
  if v1 && !kin then
    throw s!"{ctx}: a {Registered.FORMAT} file holds the keys {Registered.FILE_KEYS} and no other"
  tail v1

private theorem ports_view (j : Json) :
    Registered.portsOf j = portsV (J.str j "format" "registered reads") (Registered.keysIn j Registered.FILE_KEYS)
      (fun v1 => portsTail v1 (j.getObjVal? "ports")) := rfl

private theorem getElem_of_eq {α : Type} {a b : Array α} (h : a = b) (i : Nat) (h₁ : i < a.size)
    (h₂ : i < b.size) : a[i] = b[i] := by
  subst h
  rfl

/-- A reader of an object's entries that reads alike values the same reads two alike objects' entries the same: the
two trees list one run of keys, with alike values. -/
private theorem toArray_same {m n : Std.TreeMap.Raw String Json} (hm : m.WF) (hn : n.WF)
    (hk : ∀ k, (m.get? k).isSome = (n.get? k).isSome)
    (hv : ∀ k a b, m.get? k = some a → n.get? k = some b → Same a b) {β : Type} (f : String × Json → V β)
    (hf : ∀ (k : String) {e e' : Json}, Same e e' → f (k, e) = f (k, e')) :
    m.toArray.size = n.toArray.size ∧ m.toArray.mapM f = n.toArray.mapM f := by
  have hkeys := keysArr_same hm hn hk
  have hsz : m.toArray.size = n.toArray.size := by
    have := congrArg Array.size hkeys
    rwa [Array.size_map, Array.size_map] at this
  refine ⟨hsz, ?_⟩
  have e : m.toArray.map f = n.toArray.map f := by
    apply Array.ext
    · rw [Array.size_map, Array.size_map]; exact hsz
    · intro i h₁ h₂
      rw [Array.size_map] at h₁ h₂
      rw [Array.getElem_map, Array.getElem_map]
      have hk1 : (m.toArray[i]).1 = (n.toArray[i]).1 := by
        have := getElem_of_eq hkeys i (by rw [Array.size_map]; exact h₁) (by rw [Array.size_map]; exact h₂)
        rwa [Array.getElem_map, Array.getElem_map] at this
      have ga : m.get? (m.toArray[i]).1 = some (m.toArray[i]).2 := by
        rw [Std.TreeMap.Raw.get?_eq_getElem?]
        exact (Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hm).1 (Array.getElem_mem h₁)
      have gb : n.get? (m.toArray[i]).1 = some (n.toArray[i]).2 := by
        rw [hk1, Std.TreeMap.Raw.get?_eq_getElem?]
        exact (Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hn).1 (Array.getElem_mem h₂)
      have := hf (m.toArray[i]).1 (hv _ _ _ ga gb)
      rw [show n.toArray[i] = ((m.toArray[i]).1, (n.toArray[i]).2) from by rw [hk1]]
      exact this
  have mm : ∀ x : Array (String × Json), x.mapM f = (x.map f).mapM id := fun x =>
    (Array.mapM_map (f := f) (g := id) (xs := x)).symm
  rw [mm m.toArray, mm n.toArray, e]

private theorem portsTail_same {a b : Except String Json} (h : AlikeE a b) (v1 : Bool) :
    portsTail v1 a = portsTail v1 b := by
  rcases h with ⟨e, e', rfl, rfl⟩ | ⟨x, y, rfl, rfl, hs⟩
  · rfl
  · cases hs with
    | null => rfl
    | bool => rfl
    | num => rfl
    | str => rfl
    | arr => rfl
    | obj hm hn hk hv =>
      rename_i m n
      obtain ⟨hsz, hmap⟩ := toArray_same hm hn hk hv (portE v1) (fun k _ _ hs => portE_same v1 k hs)
      have he : m.toArray.isEmpty = n.toArray.isEmpty := by
        unfold Array.isEmpty
        rw [hsz]
      unfold portsTail
      dsimp only
      rw [he, hmap]

/-- **Alike registered reads have the same ports**, in the same order, or the same refusal. -/
theorem portsOf_same {j j' : Json} (h : Same j j') : Registered.portsOf j = Registered.portsOf j' := by
  rw [ports_view, ports_view, str_same h, keysIn_same h]
  exact congrArg _ (funext fun v1 => portsTail_same (alikeE_get h "ports") v1)

/-- **Alike registered reads are the same registered reads to the verifier**: `Registered.ofJson` returns the same
ports on both, or refuses both alike. -/
theorem ofJson_same {j j' : Json} (h : Same j j') : Registered.ofJson j = Registered.ofJson j' := by
  unfold Registered.ofJson
  rw [portsOf_same h]

/-- **Two readings of a statement's registered reads that are one value or two alike values are the same ports.** -/
theorem _root_.Flock.PartCong.PartSame.ofJson {x y : Option Json} (h : PartSame x y) :
    x.mapM Registered.ofJson = y.mapM Registered.ofJson := by
  rcases h with rfl | ⟨a, b, rfl, rfl, hs⟩
  · rfl
  · rw [Option.mapM_some, Option.mapM_some, ofJson_same hs]

end Flock.ReadsSame
