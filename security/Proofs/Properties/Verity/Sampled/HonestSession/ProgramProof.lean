module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Program
public import Proofs.Properties.Verity.Sampled
public import Proofs.Flock.Bridge.DrawnLaw
public import Proofs.Flock.SessionLoop.Concrete.Inputs
public import Proofs.Flock.SessionLoop.Concrete.Fwd
public import Proofs.Pouw.Period.Sort
public import Proofs.Flock.Recursive.FirewallJson
public import Std.Data.String.ToNat
import Circuits.Program
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Legacy
import all Init.Data.String.Slice
import all Init.Data.String.Search
import all Init.Data.ToString.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control

/-!
# `ProgramFacts`' fields that hold without the descriptor's parse

* `partitionDigest_ok`: the partition object's digest is `partitionDigestB` by that constant's definition: both are
  `Sha512.hash` of the one text `canon partitionObj`, which exists since the object holds only integers
  (`FlockDeploy.Concrete.canon_eq`). No hash is evaluated.
* `lawAt_subset`: the law is `subset`, so `Spot.Draw.deriveLaw` returns `none` without reading the program, and a verifier
  holding the session's query, law and no work table derives the subset of one (`flockLaw_law`). The sort of the law's
  keys is read by `qsort_of_sorted` and `qsort_eq_of_perm`, never run.
* `design_eq_of_parse`, `lawAt_of_parse`: `ProgramFacts.design_eq` and `lawAt_of` at any descriptor that parses.

`program_parse` and `program_sha` read `Flock.Program.decode`'s walk over `programFile` (`ProgramParse`), and `popMatches`
`Spot.Slots.layout`'s walk over the parsed program (`ProgramPop`).
What that walk reads, proved here:

* `plain_programJ`, `plain_programBare`: the descriptor is plain, so its printed text parses back to `norm programJ`
  (`FirewallJson.parse_printJson`), by `plainK`, `plain` with each string's characters read as a list.
* `foldl_insert_get`: a lookup in the definitions' `HashMap`, as `decode` builds it, is a search of their list; the
  kernel never hashes a string.
* `domainError_none`: a value `dKf` accepts is in the canonical JSON domain; `dKf` is `domainError`'s acceptance by
  structural recursion on a fuel, which the kernel evaluates, where `domainError`'s well-founded recursion is stuck.
* `mapM_except`, `stBind`, `stPure`, `stGet`, `stSet`, `stModify`, `iteApp`, `diteApp`, `fst_mk`, `snd_mk`,
  `forIn_st_nil`, `forIn_range_st`: the steps of `Array.mapM` and of `getDef`'s state monad, its loops unrolled by
  `forInSt`, as rules of `fwd_walk`; `hs_contains_*`, `hm_get_*`: its `HashSet`'s and `HashMap`'s lookups, for `String`
  keys, whose instances a rule's left side does not name.
* `splitOn_root`, `splitOn_unit`, `all_list`, `toNat_one`, `repr_one`, `inputId_not16`: the string functions a
  composite's id check and the input check read, at the descriptor's ids; the kernel evaluates none of them, `simp`
  does with their equations.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.ProgramProof

open Flock
open Lean (Json)

theorem intJ_partitionObj : FlockDeploy.Concrete.IntJ partitionObj = true := by
  unfold partitionObj Spot.Partition.build
  split <;> decide +kernel

/-- **`ProgramFacts.partitionDigest_ok`**. -/
theorem partitionDigest_ok : HmRow.partitionDigest partitionObj = .ok partitionDigestB := by
  unfold HmRow.partitionDigest
  rw [FlockDeploy.Concrete.canon_eq intJ_partitionObj]
  rfl

theorem keys_law : Spot.keys? law = some (Spot.sortStrings #["k", "law"]) := by kernel_rfl'

theorem sortStrings_kl : Spot.sortStrings #["k", "law"] = #["k", "law"] := by
  apply Array.toList_inj.1
  exact Properties.Pouw.Period.SecurityProofs.Sort.qsort_of_sorted (as := #["k", "law"]) (by decide +kernel)

theorem sortStrings_lawk : Spot.sortStrings #["law", "k"] = #["k", "law"] := by
  have h := Properties.Pouw.Period.SecurityProofs.Sort.qsort_eq_of_perm (as := #["law", "k"]) (bs := #["k", "law"])
    (List.Perm.swap "k" "law" [])
  apply Array.toList_inj.1
  exact h.trans (congrArg Array.toList sortStrings_kl)

theorem flockLaw_law : Spot.Draw.flockLaw law nUnits = .ok (.subset 1) := by
  unfold Spot.Draw.flockLaw
  rw [keys_law, sortStrings_kl, sortStrings_lawk]
  kernel_rfl'

theorem deriveLaw_law (p : Spot.Program) : Spot.Draw.deriveLaw p query law none = .ok none := by
  kernel_rfl'

theorem derive_law (p : Spot.Program) : Spot.Draw.derive p query law none = .ok law := by
  unfold Spot.Draw.derive
  rw [deriveLaw_law]
  rfl

/-- A verifier holding the session's query, law and no work table derives the subset of one, whatever its program. -/
theorem lawAt_subset (v : Spot.Registration.Verifier) (hq : v.query = query) (hl : v.law = law)
    (ht : v.workTable = none) : lawAt v nUnits = .ok (.subset 1) := by
  unfold lawAt
  rw [hq, hl, ht, derive_law, deriveLaw_law]
  simp only [Fwd.bind_ok]
  rw [flockLaw_law]
  rfl

theorem lawObj_none (l : Json) : Properties.Spot.SecurityProofs.lawObj l none = l := rfl

theorem drawnLaw_of_parse {p : Spot.Program} (h : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) :
    Flock.Bridge.drawnLaw desc query law none nUnits = .ok (.subset 1) := by
  unfold Flock.Bridge.drawnLaw
  simp only [h, Fwd.bind_ok, deriveLaw_law, lawObj_none, flockLaw_law]
  rfl

/-- **`ProgramFacts.design_eq`** at a descriptor that parses. -/
theorem design_eq_of_parse {p : Spot.Program} (h : Spot.Program.parse ⟨desc.toArray⟩ = .ok p) :
    Flock.Bridge.designAt desc query law none nUnits = .subset 1 := by
  unfold Flock.Bridge.designAt
  rw [drawnLaw_of_parse h]

/-- **`ProgramFacts.lawAt_of`** at a descriptor that parses. -/
theorem lawAt_of_parse {p : Spot.Program} (h : Spot.Program.parse ⟨desc.toArray⟩ = .ok p)
    (v : Spot.Registration.Verifier) (_ : v.program = program) (hq : v.query = query) (hl : v.law = law)
    (ht : v.workTable = none) : lawAt v nUnits = .ok (Flock.Bridge.designAt desc query law none nUnits) := by
  rw [lawAt_subset v hq hl ht, design_eq_of_parse h]

/-! ## The descriptor is plain -/

def okChar (c : Char) : Bool := c != '"' && c != '\\' && 0x20 ≤ c.val

mutual
/-- `Firewall.plain`, with each string's characters read as a list (`plainK_eq`). -/
def plainK : Json → Bool
  | .null => true
  | .bool _ => true
  | .num ⟨m, e⟩ => e == 0 && decide (0 ≤ m)
  | .str s => s.toList.all okChar
  | .arr ⟨a⟩ => plainKList a
  | .obj ⟨⟨t⟩⟩ => plainKObj t && Firewall.ascending t
def plainKList : List Json → Bool
  | [] => true
  | j :: js => plainK j && plainKList js
def plainKObj : Std.DTreeMap.Internal.Impl String (fun _ => Json) → Bool
  | .leaf => true
  | .inner _ k v l r => plainKObj l && k.toList.all okChar && plainK v && plainKObj r
end

theorem plainStr_eq (s : String) : Firewall.plainStr s = s.toList.all okChar := by
  rw [Firewall.plainStr, String.all_bool_eq]; rfl

mutual
theorem plainK_eq : ∀ j, plainK j = Firewall.plain j
  | .null => rfl
  | .bool _ => rfl
  | .num ⟨_, _⟩ => rfl
  | .str s => by simp only [plainK, Firewall.plain, plainStr_eq]
  | .arr ⟨a⟩ => by simp only [plainK, Firewall.plain]; exact plainKList_eq a
  | .obj ⟨⟨t⟩⟩ => by simp only [plainK, Firewall.plain]; rw [plainKObj_eq t]
theorem plainKList_eq : ∀ a, plainKList a = Firewall.plainList a
  | [] => rfl
  | j :: js => by simp only [plainKList, Firewall.plainList]; rw [plainK_eq j, plainKList_eq js]
theorem plainKObj_eq : ∀ t, plainKObj t = Firewall.plainObj t
  | .leaf => rfl
  | .inner _ k v l r => by
    simp only [plainKObj, Firewall.plainObj]; rw [plainKObj_eq l, plainK_eq v, plainKObj_eq r, plainStr_eq]
end

theorem plain_programJ : Firewall.plain programJ = true := by
  rw [← plainK_eq]; decide +kernel

theorem plain_programBare : Firewall.plain programBare = true := by
  rw [← plainK_eq]; decide +kernel

/-! ## Rules of `decode`'s walk -/

theorem foldl_insert_get (arr : Array (String × Json)) (a : String) :
    (arr.foldl (fun (m : Std.HashMap String Json) (p : String × Json) => m.insert p.1 p.2) {})[a]? =
      (arr.toList.reverse.find? (·.1 == a)).map (·.2) := by
  rw [← Array.foldl_toList]
  suffices h : ∀ (m : Std.HashMap String Json),
      (arr.toList.foldl (fun (m : Std.HashMap String Json) (p : String × Json) => m.insert p.1 p.2) m)[a]? =
        ((arr.toList.reverse.find? (·.1 == a)).map (·.2)).or m[a]? by
    rw [h]; simp
  generalize arr.toList = l
  induction l with
  | nil => intro m; simp
  | cons x l ih =>
    intro m
    rw [List.foldl_cons, ih, Std.HashMap.getElem?_insert]
    cases hf : List.find? (fun p : String × Json => p.1 == a) l.reverse <;>
      by_cases h : (x.1 == a) = true <;> simp [List.find?_append, hf, h]

def asciiK (s : String) : Bool := s.toList.all fun c : Char => c.toNat < 0x7f

/-- `domainError`'s acceptance, by structural recursion on a fuel the kernel evaluates. -/
def dKf : Nat → Json → Bool
  | 0, _ => false
  | _ + 1, .null => true
  | _ + 1, .bool _ => true
  | _ + 1, .num m => m.exponent == 0 && decide (m.mantissa.natAbs ≤ Flock.Program.MAX_SAFE_INT)
  | _ + 1, .str s => asciiK s
  | n + 1, .arr a => a.all (dKf n)
  | n + 1, .obj kvs => kvs.toArray.all fun x => asciiK x.1 && dKf n x.2

theorem dKf_sound : ∀ (n : Nat) (j : Json) (p : String), dKf n j = true → Flock.Program.domainError j p = none
  | 0, _, _, h => by simp [dKf] at h
  | n + 1, j, p, h => by
    unfold Flock.Program.domainError
    cases j with
    | null => rfl
    | bool _ => rfl
    | num m =>
      simp only [dKf, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
      simp only [h.1, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
      rw [if_neg (Nat.not_lt.2 h.2)]
    | str s =>
      simp only [dKf, asciiK] at h
      simp [h]
    | arr a =>
      simp only [dKf, Array.all_eq_true_iff_forall_mem] at h
      rw [Array.findSome?_eq_none_iff]
      intro i _
      exact dKf_sound n _ _ (h _ (Array.getElem_mem _))
    | obj kvs =>
      simp only [dKf, Array.all_eq_true_iff_forall_mem, Bool.and_eq_true] at h
      rw [Array.findSome?_eq_none_iff]
      intro x _
      have hx := h x.1 x.2
      simp only [asciiK] at hx
      rw [if_pos hx.1]
      exact dKf_sound n _ _ hx.2

theorem domainError_none (j : Json) (p : String) (h : dKf 24 j = true) : Flock.Program.domainError j p = none :=
  dKf_sound 24 j p h

theorem mapM_except {α β ε : Type} (f : α → Except ε β) (as : Array α) :
    as.mapM f = (as.toList.mapM f >>= fun l => pure l.toArray) := by
  rw [Array.mapM_eq_mapM_toList, map_eq_pure_bind]

theorem stBind {σ ε α β : Type} (x : StateT σ (Except ε) α) (f : α → StateT σ (Except ε) β) (s : σ) :
    (x >>= f) s = (x s >>= fun p => f p.1 p.2) := rfl

theorem stGet {σ ε : Type} (s : σ) : (get : StateT σ (Except ε) σ) s = .ok (s, s) := rfl

theorem stSet {σ ε : Type} (s' s : σ) : (set s' : StateT σ (Except ε) PUnit) s = .ok (⟨⟩, s') := rfl

theorem stModify {σ ε : Type} (f : σ → σ) (s : σ) : (modify f : StateT σ (Except ε) PUnit) s = .ok (⟨⟩, f s) := rfl

theorem iteApp {σ β : Type} (c : Prop) [Decidable c] (a b : σ → β) (s : σ) :
    (if c then a else b) s = if c then a s else b s := by split <;> rfl

theorem diteApp {σ β : Type} (c : Prop) [Decidable c] (a : c → σ → β) (b : ¬c → σ → β) (s : σ) :
    (if h : c then a h else b h) s = if h : c then a h s else b h s := by split <;> rfl

theorem stPure {σ ε α : Type} (a : α) (s : σ) : (pure a : StateT σ (Except ε) α) s = .ok (a, s) := rfl

theorem fst_mk {α β : Type} (a : α) (b : β) : (a, b).1 = a := rfl

theorem snd_mk {α β : Type} (a : α) (b : β) : (a, b).2 = b := rfl

theorem forIn_st_nil {σ ε α β : Type} (xs : Array α) (b : β) (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ)
    (h : xs = #[]) : (forIn xs b f : StateT σ (Except ε) β) s = .ok (b, s) := by
  subst h; rfl

/-- A loop of the state monad over a list, by structural recursion. -/
def forInSt {σ ε α β : Type} (f : α → β → StateT σ (Except ε) (ForInStep β)) : List α → β → σ → Except ε (β × σ)
  | [], b, s => .ok (b, s)
  | a :: as, b, s => f a b s >>= fun p => match p.1 with
    | .done b => .ok (b, p.2)
    | .yield b => forInSt f as b p.2

theorem forIn_list_st {σ ε α β : Type} (l : List α) (b : β) (f : α → β → StateT σ (Except ε) (ForInStep β)) (s : σ) :
    (forIn l b f : StateT σ (Except ε) β) s = forInSt f l b s := by
  induction l generalizing b s with
  | nil => rfl
  | cons a as ih =>
    rw [List.forIn_cons]
    show (f a b s >>= fun p => _) = _
    simp only [forInSt]
    congr 1
    funext p
    rcases p with ⟨r, s'⟩
    cases r with
    | done b' => rfl
    | yield b' => exact ih b' s'

theorem forIn_range_st {σ ε β : Type} (r : Std.Legacy.Range) (b : β) (f : Nat → β → StateT σ (Except ε) (ForInStep β))
    (s : σ) : (forIn r b f : StateT σ (Except ε) β) s = forInSt f (List.range' r.start r.size r.step) b s := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_list_st]

theorem hs_contains_insert (m : Std.HashSet String) (k a : String) :
    (m.insert k).contains a = (k == a || m.contains a) := by simp [Std.HashSet.contains_insert]

theorem hs_contains_empty (a : String) : (∅ : Std.HashSet String).contains a = false := by simp

theorem hs_contains_erase (m : Std.HashSet String) (k a : String) :
    (m.erase k).contains a = (!(k == a) && m.contains a) := by simp [Std.HashSet.contains_erase]

theorem hm_get_insert (m : Std.HashMap String Nat) (k a : String) (v : Nat) :
    (m.insert k v)[a]? = if k == a then some v else m[a]? := by simp [Std.HashMap.getElem?_insert]

theorem hm_get_empty (a : String) : (∅ : Std.HashMap String Nat)[a]? = none := by simp

/-! ## The strings a composite's id check and the input check read -/

theorem splitOn_root (s : String) (h : s = "Root[HsProg_v0]_v1") : s.splitOn "_v" = ["Root[HsProg", "0]", "1"] := by
  subst h; simp +decide [String.splitOn, String.splitOnAux]

theorem splitOn_unit (s : String) (h : s = "Unit_v1") : s.splitOn "_v" = ["Unit", "1"] := by
  subst h; simp +decide [String.splitOn, String.splitOnAux]

theorem all_list (s : String) (p : Char → Bool) : s.all p = s.toList.all p := String.all_bool_eq

theorem isNat_one : "1".toSlice.isNat = true :=
  String.Slice.isNat_of_isDigit (s := "1".toSlice) (by simp +decide) (by simp +decide)

theorem toNat_one (s : String) (h : s = "1") : s.toNat! = 1 := by
  subst h
  show "1".toSlice.toNat! = 1
  unfold String.Slice.toNat!
  rw [if_pos isNat_one, String.Slice.foldl_eq_foldl_toList]
  simp +decide

theorem repr_one : toString (1 : Nat) = "1" := by
  rw [← String.toList_inj]
  simp [Nat.toDigits_of_lt_base]
  all_goals decide +kernel

/-- `Program.widthAfter?` (private), restated: `inputId_K` reads `inputId?` through it by unfolding alone. -/
def widthAfterK (pre id : String) : Option Nat :=
  if !(id.startsWith pre && id.endsWith "_v1") then none else
  let w := ((id.drop pre.length).dropEnd 3).toString
  if w.length > 0 && w.all Char.isDigit && w.front != '0' then w.toNat? else none

theorem inputId_K (id : String) : Flock.Program.inputId? id = match widthAfterK "Input" id with
    | some w => some (Flock.Program.RUN, w)
    | none => (widthAfterK "RegisteredInput" id).map (Flock.Program.REGISTRATION, ·) := rfl

theorem inputId_not16 : Flock.Program.inputId? "Not16_v1" = none := by
  simp +decide [inputId_K, widthAfterK]

end Verity.Sampled.Instance.HonestSession.ProgramProof
