module

public import Proofs.Properties.NetworkPeriod
import Batteries.Tactic.OpenPrivate
import Proofs.Core.CanonUtf8
import Proofs.Flock.VBridge.Family.Names
import Proofs.Flock.SessionLoop.Concrete.Inputs
import Proofs.Flock.SessionLoop.Concrete.Sizes
import Proofs.Flock.Soundness.Refine.Walk
import Proofs.Properties.Hash.Identity
import Proofs.Properties.Pouw.LiveSalt.Canon
import Proofs.Properties.Spot.HistoryTable
import Std.Data.String.ToNat
import all Base.Canon
import all CoreProtocol.Ratio
import all CoreProtocol.Profile
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
import all Init.Data.String.Legacy
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Search
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.ToString.Basic
import all Init.Data.Array.Attach
import all Init.WF

/-!
# A scratch probe: what the honest rows period's witness needs read back (never landed)

Two builds, each on the code as written, with digests and strings as terms:

* the profile reads back: `Core.Ratio.ofString` of `Core.Ratio.toString`, and `Core.Integrity.ofJson` of
  `Core.Integrity.toJson` at one level sampled `subset:k` with δ = 1/n (`ofString_toString`, `integrity_of`);
* the link-window's rows have a domain: `rowsDomain` is defined, and 64 bytes, for a 64-byte schedule digest and a
  32-byte session (`rowsDomain_ok`), its manifest's canonical text written out with the hex strings as terms
  (`canon_manifest`) and bounded (`text_fits`).

A third: no tactic decides `BEq Json`'s `==`, even on one literal string.
-/

set_option autoImplicit false

open private natOf from CoreProtocol.Ratio
open private exactly strs from CoreProtocol.Profile
open private hex4 from Base.Canon

namespace ReadBackProbe

open Lean (Json toJson)
open Flock (V canonStrUtf8 canonUtf8)
open FlockDeploy.Concrete (IntJ canon_eq)

/-! ## `BEq Json` -/

/-- `BEq Json`'s `==` is a `partial def`: no tactic decides it, even at one literal string. -/
theorem json_beq_undecided : True := by
  fail_if_success (have : (Json.str "a" == Json.str "a") = true := by decide)
  fail_if_success (have : (Json.str "a" == Json.str "a") = true := by rfl)
  fail_if_success (have : (Json.str "a" == Json.str "a") = true := by simp)
  fail_if_success (have : (Json.str "a" == Json.str "a") = true := by decide +kernel)
  trivial

/-! ## The ratio reads back -/

/-- The legacy `String.splitOn "/"` is `List.splitOn '/'` on the characters. -/
theorem splitOn_slash (s : String) : s.splitOn "/" = (s.toList.splitOn '/').map String.ofList :=
  FlockVBridge.Names.splitOn_sep "/" '/' rfl rfl rfl rfl (by decide) s

theorem toNat!_repr (n : Nat) : (Nat.repr n).toNat! = n := by
  have h1 := Nat.isNat_repr n
  have h2 := Nat.toNat?_repr n
  unfold String.toNat! String.Slice.toNat!
  unfold String.toNat? String.Slice.toNat? at h2
  unfold String.isNat at h1
  rw [if_pos h1] at h2 ⊢
  exact Option.some.inj h2

theorem all_digit_repr (n : Nat) : (Nat.repr n).all Char.isDigit = true := by
  rw [String.all_bool_eq, List.all_eq_true]
  intro c hc
  rw [Nat.toList_repr] at hc
  exact Nat.isDigit_of_mem_toDigits (by omega) (by omega) hc

theorem head_toDigits : ∀ (k n : Nat), n < k → 10 ≤ n → (Nat.toDigits 10 n).head? ≠ some '0'
  | 0, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, n, hk, hn => by
    rw [Nat.toDigits_eq_if (by omega), if_neg (by omega)]
    by_cases h : n / 10 < 10
    · rw [Nat.toDigits_of_lt_base h]
      have h1 : 1 ≤ n / 10 := by omega
      generalize n / 10 = m at h h1
      have : m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6 ∨ m = 7 ∨ m = 8 ∨ m = 9 := by omega
      rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
        (simp only [List.cons_append, List.head?_cons]; decide)
    · have ih := head_toDigits k (n / 10) (by omega) (by omega)
      have hne : Nat.toDigits 10 (n / 10) ≠ [] := Nat.toDigits_ne_nil
      revert ih hne
      cases Nat.toDigits 10 (n / 10) with
      | nil => intro _ h; exact absurd rfl h
      | cons a l => intro ih _; simpa using ih

theorem canon_repr (n : Nat) : ((Nat.repr n).length = 1 || (Nat.repr n).front != '0') = true := by
  by_cases h : n < 10
  · have : (Nat.repr n).length = 1 := by
      rw [← String.length_toList, Nat.toList_repr, Nat.toDigits_of_lt_base h]; rfl
    simp [this]
  · have hh := head_toDigits (n + 1) n (by omega) (by omega)
    rw [← Nat.toList_repr, ← String.front?_eq] at hh
    have : (Nat.repr n).front ≠ '0' := by
      rw [String.front_eq]
      intro he
      apply hh
      revert he
      cases (Nat.repr n).front? with
      | none => intro he; exact absurd he (by decide)
      | some c => intro he; simpa using he
    rw [Bool.or_eq_true]
    exact Or.inr (bne_iff_ne.2 this)

theorem natOf_repr (n : Nat) (ctx : String) : natOf (Nat.repr n) ctx = .ok n := by
  unfold natOf
  have h0 : (Nat.repr n).isEmpty = false := String.isEmpty_eq_false_iff.2 Nat.repr_ne_empty
  rw [h0, all_digit_repr, canon_repr, toNat!_repr]
  rfl

theorem slash_notin_repr (n : Nat) : '/' ∉ (Nat.repr n).toList := by
  intro h
  rw [Nat.toList_repr] at h
  have := Nat.isDigit_of_mem_toDigits (b := 10) (by omega) (by omega) h
  exact absurd this (by decide)

theorem toString_den_one (n : Nat) : Core.Ratio.toString ⟨n, 1⟩ = Nat.repr n := by
  simp [Core.Ratio.toString, toString]

theorem toString_den (n d : Nat) (h : d ≠ 1) : Core.Ratio.toString ⟨n, d⟩ = Nat.repr n ++ "/" ++ Nat.repr d := by
  simp [Core.Ratio.toString, h, toString]

/-- **`Core.Ratio.ofString` reads `Core.Ratio.toString` back**, on a whole number and on a ratio in lowest terms with a
denominator of at least two: every ratio `toString` is meant to write. -/
theorem ofString_toString (r : Core.Ratio) (ctx : String) (h : r.den = 1 ∨ (2 ≤ r.den ∧ Nat.gcd r.num r.den = 1)) :
    Core.Ratio.ofString r.toString ctx = .ok r := by
  obtain ⟨n, d⟩ := r
  simp only at h
  rcases h with rfl | ⟨h2, hg⟩
  · rw [toString_den_one]
    unfold Core.Ratio.ofString
    rw [splitOn_slash, List.splitOn_eq_singleton (slash_notin_repr n)]
    simp only [List.map_cons, List.map_nil, String.ofList_toList]
    rw [natOf_repr]
    rfl
  · rw [toString_den n d (by omega)]
    unfold Core.Ratio.ofString
    rw [splitOn_slash]
    have hs : (Nat.repr n ++ "/" ++ Nat.repr d).toList.splitOn '/' = [(Nat.repr n).toList, (Nat.repr d).toList] := by
      have : (Nat.repr n ++ "/" ++ Nat.repr d).toList = (Nat.repr n).toList ++ '/' :: (Nat.repr d).toList := by
        simp [String.toList_append]
      rw [this, List.splitOn_append_cons_self_of_not_mem (slash_notin_repr n),
        List.splitOn_eq_singleton (slash_notin_repr d)]
    rw [hs]
    simp only [List.map_cons, List.map_nil, String.ofList_toList]
    rw [natOf_repr, natOf_repr]
    have hd : ¬ d ≤ 1 := by omega
    simp [bind, Except.bind, hd, hg, pure, Except.pure]


#print axioms ofString_toString

/-! ## The profile reads back -/

def lvl (size k : Nat) : Core.Level := ⟨"unit", none, size, some (.subset k)⟩

def honest (p : String) (size k n : Nat) : Core.Integrity :=
  { programs := #[p], prescribed := #[], linked := true, covered := #[], levels := #[lvl size k],
    delta := ⟨1, n⟩, drawn := #[("unit", k)] }

theorem exactly_self {α : Type} (toJson : α → Json) (a : α) (ctx : String) (h : IntJ (toJson a) = true) :
    exactly toJson (toJson a) ctx a = .ok a := by
  unfold exactly
  rw [canon_eq h]
  simp [bind, Except.bind, pure, Except.pure]

theorem design_step (k : Nat) (ctx : String) :
    Core.SamplingDesign.ofJson (Core.SamplingDesign.subset k).toJson ctx =
      exactly Core.SamplingDesign.toJson (Core.SamplingDesign.subset k).toJson ctx (.subset k) := by kernel_rfl

theorem design_of (k : Nat) (ctx : String) :
    Core.SamplingDesign.ofJson (Core.SamplingDesign.subset k).toJson ctx = .ok (.subset k) := by
  rw [design_step, exactly_self _ _ _ (by kernel_rfl)]

theorem level_step (size k : Nat) (ctx : String) :
    Core.Level.ofJson (lvl size k).toJson ctx =
      ((some <$> Core.SamplingDesign.ofJson (Core.SamplingDesign.subset k).toJson (s!"{ctx}.{"draw"}")) >>= fun d =>
        exactly Core.Level.toJson (lvl size k).toJson ctx ⟨"unit", none, size, d⟩) := by kernel_rfl

theorem level_of (size k : Nat) (ctx : String) : Core.Level.ofJson (lvl size k).toJson ctx = .ok (lvl size k) := by
  rw [level_step, design_of]
  exact exactly_self _ _ _ (by kernel_rfl)

theorem levels_of (p : String) (size k n : Nat) (ctx : String) :
    Flock.J.arr (honest p size k n).toJson "levels" ctx = .ok #[(lvl size k).toJson] := by kernel_rfl

theorem drawn_of (p : String) (size k n : Nat) (ctx : String) :
    Flock.J.get (honest p size k n).toJson "drawn" ctx = .ok (Json.mkObj [("unit", Json.num k)]) := by kernel_rfl

theorem has_unit (size k : Nat) : Flock.J.has (Json.mkObj [("unit", Json.num k)]) (lvl size k).name = true := by kernel_rfl

theorem nat_unit (size k : Nat) (ctx : String) :
    Flock.J.nat (Json.mkObj [("unit", Json.num k)]) (lvl size k).name ctx = .ok k := by kernel_rfl

theorem programs_of (p : String) (size k n : Nat) (ctx : String) :
    strs (honest p size k n).toJson "programs" ctx = .ok #[p] := by kernel_rfl

theorem prescribed_of (p : String) (size k n : Nat) (ctx : String) :
    strs (honest p size k n).toJson "prescribed" ctx = .ok #[] := by kernel_rfl

theorem covered_of (p : String) (size k n : Nat) (ctx : String) :
    strs (honest p size k n).toJson "covered" ctx = .ok #[] := by kernel_rfl

theorem linked_of (p : String) (size k n : Nat) (ctx : String) :
    Flock.J.bool (honest p size k n).toJson "linked" ctx = .ok true := by kernel_rfl

theorem delta_of (p : String) (size k n : Nat) (ctx : String) :
    Flock.J.str (honest p size k n).toJson "delta" ctx = .ok (Core.Ratio.toString ⟨1, n⟩) := by kernel_rfl

theorem intJ_honest (p : String) (size k n : Nat) : IntJ (honest p size k n).toJson = true := by kernel_rfl

theorem ratio_of (n : Nat) (hn : 2 ≤ n) (ctx : String) :
    Core.Ratio.ofString (Core.Ratio.toString ⟨1, n⟩) ctx = .ok ⟨1, n⟩ :=
  ofString_toString ⟨1, n⟩ ctx (Or.inr ⟨hn, Nat.gcd_one_left n⟩)

theorem integrity_of (p : String) (size k n : Nat) (hn : 2 ≤ n) (ctx : String) :
    Core.Integrity.ofJson (honest p size k n).toJson ctx = .ok (honest p size k n) := by
  unfold Core.Integrity.ofJson
  rw [levels_of]
  simp only [bind, Except.bind, List.push_toArray, List.nil_append, List.forIn_toArray, List.forIn_cons, List.forIn_nil, level_of, drawn_of, has_unit,
    nat_unit, programs_of, prescribed_of, covered_of, linked_of, delta_of, ratio_of n hn, pure, Except.pure, if_true]
  first
    | exact exactly_self Core.Integrity.toJson (honest p size k n) ctx (intJ_honest p size k n)
    | trace_state


#print axioms integrity_of

/-! ## The rows' domain -/

def manifest (sdh sessh link dv : String) (w cb al win : Nat) : Json := Json.mkObj [
    ("parameters", .str sdh), ("session", .str sessh), ("link", .str link),
    ("direction", .str dv), ("window", toJson w), ("cell_bytes", toJson cb),
    ("slots", toJson al), ("rows", toJson win), ("row_schema", .str "hm96-sha512/row/v2")]

theorem flat (sdh sessh link dv : String) (w cb al win : Nat) :
    Verity.CanonUtf8.jsonFlat (manifest sdh sessh link dv w cb al win) = true := by kernel_rfl

def text (sdh sessh link dv : String) (w cb al win : Nat) : String :=
  "{" ++ ",".intercalate ([("cell_bytes", toString (Int.ofNat cb)), ("direction", canonStrUtf8 dv),
    ("link", canonStrUtf8 link), ("parameters", canonStrUtf8 sdh), ("row_schema", canonStrUtf8 "hm96-sha512/row/v2"),
    ("rows", toString (Int.ofNat win)), ("session", canonStrUtf8 sessh), ("slots", toString (Int.ofNat al)),
    ("window", toString (Int.ofNat w))].map fun kv => canonStrUtf8 kv.1 ++ ":" ++ kv.2) ++ "}"

theorem canonFlat_manifest (sdh sessh link dv : String) (w cb al win : Nat) :
    Verity.CanonUtf8.canonFlat (manifest sdh sessh link dv w cb al win) = .ok (text sdh sessh link dv w cb al win) := by
  kernel_rfl

theorem canon_manifest (sdh sessh link dv : String) (w cb al win : Nat) :
    canonUtf8 (manifest sdh sessh link dv w cb al win) = .ok (text sdh sessh link dv w cb al win) := by
  rw [Verity.CanonUtf8.canonUtf8_flat _ (flat ..)]
  exact canonFlat_manifest ..

theorem intJ_manifest (sdh sessh link dv : String) (w cb al win : Nat) :
    IntJ (manifest sdh sessh link dv w cb al win) = true := by kernel_rfl

theorem key_sizes :
    (canonStrUtf8 "cell_bytes").utf8ByteSize = 12 ∧ (canonStrUtf8 "direction").utf8ByteSize = 11 ∧
    (canonStrUtf8 "link").utf8ByteSize = 6 ∧ (canonStrUtf8 "parameters").utf8ByteSize = 12 ∧
    (canonStrUtf8 "row_schema").utf8ByteSize = 12 ∧ (canonStrUtf8 "rows").utf8ByteSize = 6 ∧
    (canonStrUtf8 "session").utf8ByteSize = 9 ∧ (canonStrUtf8 "slots").utf8ByteSize = 7 ∧
    (canonStrUtf8 "window").utf8ByteSize = 8 ∧ (canonStrUtf8 "hm96-sha512/row/v2").utf8ByteSize = 20 := by
  decide +kernel

theorem size_text (sdh sessh link dv : String) (w cb al win : Nat) (B : Nat)
    (h1 : (canonStrUtf8 sdh).utf8ByteSize ≤ B) (h2 : (canonStrUtf8 sessh).utf8ByteSize ≤ B)
    (h3 : (canonStrUtf8 link).utf8ByteSize ≤ B) (h4 : (canonStrUtf8 dv).utf8ByteSize ≤ B)
    (n1 : (toString (Int.ofNat w)).utf8ByteSize ≤ B) (n2 : (toString (Int.ofNat cb)).utf8ByteSize ≤ B)
    (n3 : (toString (Int.ofNat al)).utf8ByteSize ≤ B) (n4 : (toString (Int.ofNat win)).utf8ByteSize ≤ B) :
    (text sdh sessh link dv w cb al win).utf8ByteSize ≤ 8 * B + 200 := by
  obtain ⟨k1, k2, k3, k4, k5, k6, k7, k8, k9, k10⟩ := key_sizes
  unfold text
  simp only [List.map, String.intercalate, String.intercalate.go, String.utf8ByteSize_append]
  first
    | (simp only [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10]
       have e1 : ("{" : String).utf8ByteSize = 1 := by decide
       have e2 : ("," : String).utf8ByteSize = 1 := by decide
       have e3 : (":" : String).utf8ByteSize = 1 := by decide
       have e4 : ("}" : String).utf8ByteSize = 1 := by decide
       simp only [e1, e2, e3, e4]
       omega)
    | trace_state

theorem esc_len (c : Char) : (Properties.Pouw.LiveSalt.Canon.esc c).length ≤ 6 := by
  unfold Properties.Pouw.LiveSalt.Canon.esc
  split
  all_goals first
    | (simp; done)
    | ((repeat' split) <;> simp [hex4])

theorem flatMap_len (l : List Char) : (l.flatMap Properties.Pouw.LiveSalt.Canon.esc).length ≤ 6 * l.length := by
  induction l with
  | nil => simp
  | cons c l ih =>
    rw [List.flatMap_cons, List.length_append, List.length_cons]
    have := esc_len c
    omega

theorem utf8_le (s : String) : s.utf8ByteSize ≤ 4 * s.length := by
  have := FlockDeploy.Concrete.utf8_ofList_le s.toList
  rwa [String.ofList_toList, String.length_toList] at this

theorem canonStr_size (s : String) : (canonStrUtf8 s).utf8ByteSize ≤ 24 * s.length + 8 := by
  have h := utf8_le (canonStrUtf8 s)
  rw [← String.length_toList, Properties.Pouw.LiveSalt.Canon.canonStr_toList] at h
  have := flatMap_len s.toList
  simp only [List.length_cons, List.length_append, List.length_nil, String.length_toList] at h this
  omega

theorem nat_size (n : Nat) (h : n < 2 ^ 64) : (toString (Int.ofNat n)).utf8ByteSize ≤ 80 := by
  have e : toString (Int.ofNat n) = Nat.repr n := rfl
  rw [e]
  have h1 := utf8_le (Nat.repr n)
  have h2 : (Nat.repr n).length ≤ 20 := (Nat.length_repr_le_iff (by omega)).2 (by omega)
  omega

/-- The rows manifest's canonical text fits the tagged hash's u64 length field. -/
theorem text_fits (sdh sessh link dv : String) (w cb al win : Nat) (h1 : sdh.length ≤ 128) (h2 : sessh.length ≤ 64)
    (h3 : link.length < 2 ^ 32) (h4 : dv.length ≤ 7) (n1 : w < 2 ^ 64) (n2 : cb < 2 ^ 64) (n3 : al < 2 ^ 64)
    (n4 : win < 2 ^ 64) : (text sdh sessh link dv w cb al win).utf8ByteSize < 2 ^ 64 := by
  have := size_text sdh sessh link dv w cb al win (24 * 2 ^ 32 + 8)
    (by have := canonStr_size sdh; omega) (by have := canonStr_size sessh; omega)
    (by have := canonStr_size link; omega) (by have := canonStr_size dv; omega)
    (by have := nat_size w n1; omega) (by have := nat_size cb n2; omega) (by have := nat_size al n3; omega)
    (by have := nat_size win n4; omega)
  omega


theorem hex_len (b : ByteArray) : (Flock.hexOf b).length = 2 * b.size := FlockDeploy.Concrete.hexOf_len b

theorem value_len (d : NetworkCheck.Direction) : d.value.length ≤ 7 := by cases d <;> decide

/-- **The link-window's rows have a domain**: the manifest has a canonical text, and it fits the tagged hash. -/
theorem rowsDomain_ok (s : NetworkCheck.Period.Schedule) (sd session : ByteArray) (link : String)
    (d : NetworkCheck.Direction) (w : Nat) (hsd : sd.size = 64) (hse : session.size = 32) (hl : link.length < 2 ^ 32)
    (hw : w < 2 ^ 64) (hB : s.cellBytes < 2 ^ 64) (hS : s.allocation < 2 ^ 64) (hT : s.window < 2 ^ 64) :
    ∃ dom, NetworkCheck.Period.rowsDomain s sd session link d w = .ok dom ∧ dom.size = 64 := by
  have hc : canonUtf8 (Json.mkObj [
      ("parameters", .str (Flock.hexOf sd)), ("session", .str (Flock.hexOf session)), ("link", .str link),
      ("direction", .str d.value), ("window", toJson w), ("cell_bytes", toJson s.cellBytes),
      ("slots", toJson s.allocation), ("rows", toJson s.window), ("row_schema", .str NetworkCheck.Period.rowSchema)]) =
      .ok (text (Flock.hexOf sd) (Flock.hexOf session) link d.value w s.cellBytes s.allocation s.window) :=
    canon_manifest (Flock.hexOf sd) (Flock.hexOf session) link d.value w s.cellBytes s.allocation s.window
  have hfit := text_fits (Flock.hexOf sd) (Flock.hexOf session) link d.value w s.cellBytes s.allocation s.window
    (by rw [hex_len]; omega) (by rw [hex_len]; omega) hl (value_len d) hw hB hS hT
  obtain ⟨x, -, hx⟩ := Properties.Identity.taggedSha512_of (tag := NetworkCheck.Period.domainTag)
    (payload := (text (Flock.hexOf sd) (Flock.hexOf session) link d.value w s.cellBytes s.allocation s.window).toUTF8)
    (by decide +kernel) (by decide +kernel) (by rw [FlockDeploy.Concrete.toUTF8_size]; exact hfit)
  refine ⟨(Flock.FV3.Sha512.Domain.make (Flock.Sha512.hash x) (-1) 0 s.window).id, ?_, ?_⟩
  · unfold NetworkCheck.Period.rowsDomain Identity.identityDigestSha512
    simp only [bind, Except.bind, hc, hx, pure, Except.pure]
  · first
      | exact FlockDeploy.Concrete.hash_size _
      | exact Properties.Spot.SecurityProofs.HistoryRoot.make_id_size _ _ _ _
      | exact Properties.Spot.SecurityProofs.make_id_size _ _ _ _

#print axioms rowsDomain_ok

end ReadBackProbe
