module

public import Flock.RecVStar
public import Proofs.Flock.Claims.Scope.Check
public import Aesop.BuiltinRules
public import Mathlib.Data.Nat.Notation
public import Mathlib.Tactic.ToAdditive
public import Mathlib.Tactic.ToDual
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.ExecLookup
import Proofs.Flock.Soundness.Refine.Setup

/-! The recursion's inner statement and Flock's statement check read one leaf scheme off the circuit file, so a check
that passes an inner circuit file ran in the plain-leaves form (`check_of_innerOf`). -/

@[expose] public section

namespace FlockSoundness.Discharge.InnerLeaf

open Flock FlockLevel3
open Lean (Json)

def metaLeaf (bytes : ByteArray) : Option String := do
  let text ← String.fromUTF8? bytes
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  let mj ← (Flock.JsonParse.parse (((lines.getD 1 "").trimAsciiEnd.toString).drop 5).toString).toOption
  let ls ← (mj.getObjVal? "leaf_scheme").toOption
  (ls.getObjVal? "id" >>= Json.getStr?).toOption

theorem get_ok {j v : Json} {k ctx : String} (h : J.get j k ctx = .ok v) : j.getObjVal? k = .ok v := by
  unfold J.get at h
  split at h
  · rename_i v' hv
    cases h
    exact hv
  · cases h

theorem str_ok {j : Json} {k ctx s : String} (h : J.str j k ctx = .ok s) : (j.getObjVal? k >>= Json.getStr?) = .ok s := by
  unfold J.str at h
  obtain ⟨v, hv, h⟩ := except_bind_ok h
  rw [get_ok hv]
  unfold J.asStr at h
  split at h
  · rename_i s' hs
    cases h
    exact hs
  · cases h

theorem metaLeaf_of {bytes : ByteArray} {text : String} {mj ls : Json} {ctx₁ ctx₂ s : String}
    (ht : String.fromUTF8? bytes = some text)
    (hj : Flock.JsonParse.parse
      (((if ((text.splitOn "\n").toArray).back? == some "" then ((text.splitOn "\n").toArray).pop
          else (text.splitOn "\n").toArray).getD 1 "").trimAsciiEnd.toString.drop 5).toString = .ok mj)
    (hl : J.get mj "leaf_scheme" ctx₁ = .ok ls) (hs : J.str ls "id" ctx₂ = .ok s) : metaLeaf bytes = some s := by
  simp only [metaLeaf, ht, hj, get_ok hl, str_ok hs, Option.bind_eq_bind, Option.bind_some, Except.toOption]

set_option maxHeartbeats 4000000 in
theorem parse_metaLeaf {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} {c : Circuit} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    metaLeaf bytes = some tags.merkleLeaf := by
  unfold HmRow.parse at h
  simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  rename_i text ht
  (have hh := except_bind_ok h; clear h; obtain ⟨text', hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := except_bind_ok h; cases hte
  rename_i mj0 hj
  (have hh := except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨f1, hf1, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨f2, hf2, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp6] at h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jp7] at h
      obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := except_bind_ok h
      exact ⟨cs, h⟩
  simp (config := { zeta := false }) only [jp5] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s1, hwhile, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h
  simp (config := { zeta := false }) only [jpT] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨s2, hlook, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ports, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨upv, -, h⟩ := hh)
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨ranges, hranges, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesIn, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leavesOut, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummy, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyDigests, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨dummyOutputs, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outPorts, -, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpR at h
  obtain ⟨outRowPorts, h⟩ : ∃ r, jpR r = .ok c := by
    split at h
    · exact ⟨_, h⟩
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨r, -, h⟩ := except_bind_ok h
      exact ⟨r, ite_throw_ok h⟩
  simp (config := { zeta := false }) only [jpR] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jpP] at h
  (have hh := except_bind_ok h; clear h; obtain ⟨shaNet, hsha, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨hmNet, hhm, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨outNet, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpO at h)
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpO] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpC] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := except_bind_ok h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived rowNet jpD at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only at h
  (have hh := except_bind_ok h; clear h; obtain ⟨ls, hls, h⟩ := hh)
  (have hh := except_bind_ok h; clear h; obtain ⟨leafId, hid, h⟩ := hh)
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h
  obtain ⟨hleaf, -⟩ := ite_throw_ok_not h
  have he : leafId = tags.merkleLeaf := by simpa [bne_iff_ne] using hleaf
  subst he
  exact metaLeaf_of ht hj hls hid

/-- The inner tags hide the statement's outputs and are untyped, so `innerOf` parses the circuit with `HmOut.parse`, which
is `HmRow.parse` there. -/
theorem innerTags_hiddenOutputs : RecVStar.innerTags.hiddenOutputs = true := rfl

theorem innerTags_typed : RecVStar.innerTags.typed = false := rfl

theorem innerOf_metaLeaf {cf pf : ByteArray} {partition : Option Json} {program : Option ByteArray} {holo : Bool}
    {dims : RecVStar.Inner} (h : RecVStar.innerOf cf pf partition program holo = .ok dims) :
    metaLeaf cf = some "flock-leaf/sha512-unsalted" := by
  unfold RecVStar.innerOf at h
  obtain ⟨⟨spec, sts⟩, hs, -⟩ := except_bind_ok h
  rcases HiddenExec.setupTables_okH innerTags_hiddenOutputs hs with ⟨-, st, hst, -, -⟩ | ⟨hc, -⟩
  · exact parse_metaLeaf (Hidden.parse_untyped innerTags_typed ((Hidden.setupHidden_spec' hst).parseF (by decide))).1
  · omega

theorem check_metaLeaf_tags {t : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {pi reg : Bool} (h : ProvedScope.check t zk cf pf tables count pi reg = .ok ()) :
    metaLeaf cf = some t.merkleLeaf ∧ Flock.Readable.LeafOk t zk := by
  obtain ⟨⟨-, htyped, -, hleaf, -⟩, c, -, hc, -⟩ := (Flock.Readable.check_ok_iff t zk cf pf tables count pi reg).1 h
  refine ⟨?_, hleaf⟩
  unfold ProvedScope.circuitOf at hc
  split at hc
  · obtain ⟨pre, c₀, hp, -⟩ := Hidden.parseCommit_untyped htyped hc
    exact parse_metaLeaf hp
  · exact parse_metaLeaf hc

theorem check_metaLeaf {t : Tags} {zk : Bool} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {pi reg : Bool} (h : ProvedScope.check t zk cf pf tables count pi reg = .ok ()) :
    metaLeaf cf = some "hm96-sha512/v1" ∨
      (zk = false ∧ t.hiddenOutputs = true ∧ metaLeaf cf = some "flock-leaf/sha512-unsalted") := by
  obtain ⟨hm, hl | ⟨hz, hh, hl⟩⟩ := check_metaLeaf_tags h
  · exact .inl (by rw [← hl]; exact hm)
  · exact .inr ⟨hz, hh, by rw [← hl]; exact hm⟩

theorem check_of_innerOf {cf pf pf' : ByteArray} {partition : Option Json} {program : Option ByteArray}
    {holo : Bool} {dims : RecVStar.Inner} (h : RecVStar.innerOf cf pf partition program holo = .ok dims) {t : Tags}
    {zk : Bool} {tables : List (String × ByteArray)} {count : ℕ} {pi reg : Bool}
    (hc : ProvedScope.check t zk cf pf' tables count pi reg = .ok ()) :
    zk = false ∧ t.hiddenOutputs = true ∧ t.merkleLeaf = "flock-leaf/sha512-unsalted" := by
  obtain ⟨hm, hl⟩ := check_metaLeaf_tags hc
  have ht : t.merkleLeaf = "flock-leaf/sha512-unsalted" :=
    Option.some.inj (hm.symm.trans (innerOf_metaLeaf h))
  rcases hl with hl | ⟨hz, hh, -⟩
  · exact absurd (ht.symm.trans hl) (by decide)
  · exact ⟨hz, hh, ht⟩

end FlockSoundness.Discharge.InnerLeaf
