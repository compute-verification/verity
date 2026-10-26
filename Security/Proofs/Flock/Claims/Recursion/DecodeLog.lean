import Proofs.Flock.Claims.Recursion.LogWalk
import Flock.With.VerifyE

/-!
# The record decoder's SHA-512 calls

`decode_log`: a run of `Record.decodeWith` with the logging oracle appends, in order, the digests' inputs of S5 (each
table's `root_B`), S6 (`root_F`), S7 (each table's public words, where they are not empty or have a digest), S12 (each
round's `H(ε)` size check and its retained bytes), S13 (each rep stream's `root_B`) and S14 (the exchange `lbOf`), and
nothing else. With them, what S6, S10, S11 and S13 check of the session it returns.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

local instance lawfulBEqByteArrayD : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- A round's digests' inputs: S12's size check of `H(ε)`, and its retained bytes. -/
def roundLog (spec : SessionSpec) (rd : Round) : List ByteArray :=
  dgl spec ByteArray.empty ++ rd.bytes.toList.flatMap (dgl spec)

/-- A stream's digests' inputs: its rounds'. -/
def streamLog (spec : SessionSpec) (s : StreamRec) : List ByteArray := s.rounds.toList.flatMap (roundLog spec)

/-- S14's exchange: `root_F`, the tables' `root_B` in lexicographic table order, the link points and `y`. -/
def lbOf (spec : SessionSpec) (sess : Session) : ByteArray :=
  sess.y.foldl F128.pushBytes (sess.points.foldl (fun lb p => p.foldl F128.pushBytes lb)
    (((spec.tables.zip sess.rootB).qsort (fun x1 x2 => decide (x1.1 < x2.1))).foldl (fun lb x => lb ++ x.2)
      sess.rootF))

theorem size_zero_of_not {p : ByteArray} {b : Bool} (h : ¬(!(p.size == 0 && b)) = true) : p.size = 0 := by
  by_contra hne
  exact h (by simp [hne])

open Lean (Json) in
/-- `decode_log` at `Record.decodeWithK keep`: `keep` wraps only comparisons, so every digest is still computed. -/
theorem decodeK_log (keep : Bool → Bool) {h : ByteArray → ByteArray} {spec : SessionSpec} {text : String}
    {l₀ : Array ByteArray} :
    RetL (Record.decodeWithK keep (Hashes.logged h) spec text) l₀ fun sess l' =>
      ∃ ds : List (List ByteArray), l'.toList = l₀.toList ++ (sess.rootB.toList.flatMap (dgl spec) ++
        dgl spec sess.rootF ++ ds.flatten ++ sess.streams.toList.flatMap (streamLog spec) ++
        spec.repStreams.toList.flatMap (fun x => dgl spec sess.rootB[x.1]!) ++ dgl spec (lbOf spec sess)) ∧
      List.Forall₂ (fun p d => d = dgl spec p ∨ d = [] ∧ p.size = 0) sess.publics.toList ds ∧
      sess.rootB.size = spec.tables.size ∧ sess.rootF = spec.sigma ∧
      sess.y.size = spec.points * spec.tables.size ∧ (∀ c ∈ sess.y, c.isZero = true) ∧
      sess.streams.size = spec.repStreams.size ∧
      ∀ x ∈ spec.repStreams.toList,
        ∃ s, sess.streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size := by
  unfold Record.decodeWithK Record.decodeJWithK
  retl_walk [Record.check]
  all_goals
    subst_vars
    obtain ⟨y5, e5, f5, g5⟩ := mapM_log (R := fun _ rb d => True ∧ d = dgl spec rb) (fun t l => by
      retl_walk []
      exact ⟨_, by simp [dgL_toList], trivial, rfl⟩) h_rootB
    obtain ⟨y7, e7, f7, g7⟩ := mapM_log (R := fun _ p d => d = dgl spec p ∨ d = [] ∧ p.size = 0) (fun t l => by
      retl_walk []
      all_goals first
        | (refine ⟨_, ?_, .inl rfl⟩; simp [dgL_toList]; done)
        | exact ⟨[], by simp, .inr ⟨rfl, size_zero_of_not hc⟩⟩) h_publics
    obtain ⟨y12, e12, f12, g12⟩ := mapM_log (R := fun _ s d => True ∧ d = streamLog spec s) (fun s l => by
      retl_walk []
      obtain ⟨yr, er, fr, gr⟩ := mapIdxM_log (R := fun _ rd d => True ∧ d = roundLog spec rd) (fun k r l => by
        retl_walk []
        all_goals exact ⟨_, by simp [dgL_toList, roundLog], trivial, rfl⟩) h_rounds
      refine ⟨_, ?_, trivial, rfl⟩
      rw [gr, flatten_snd (R := fun _ _ => True) (F := roundLog spec) fr, ← er]; rfl) h_streams
    have h13 := ‹StV.go (forIn spec.repStreams _ _) _ = _›
    obtain ⟨d13, f13, g13⟩ := forIn_log (R := fun x d => d = dgl spec rootB[x.1]! ∧
        ∃ s, streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size) (fun x _ l => by
      obtain ⟨i, r, name⟩ := x
      dsimp only
      retl_walk [Record.check]
      have hr := check_ok_iff.1 ‹Record.check (_ && decide (_ > 0)) _ = _›
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hr
      exact ⟨rfl, _, by simp [dgL_toList], rfl, _, by assumption, hr.2⟩) h13
    have h14a := ‹StV.go (forIn ((spec.tables.zip rootB).qsort _) _ _) _ = _›
    obtain ⟨rfl, rfl⟩ := go_forIn_pure_inv (f := fun (x : String × ByteArray) b => b ++ x.2) h14a
    have h14b := ‹StV.go (forIn pts _ _) _ = _›
    simp only [Array.forIn_pure_yield_eq_foldl, pure_bind] at h14b
    obtain ⟨rfl, rfl⟩ := go_pure_inv h14b
    have h14c := ‹StV.go (forIn y _ _) _ = _›
    obtain ⟨rfl, rfl⟩ := go_forIn_pure_inv (f := fun c b => F128.pushBytes b c) h14c
    obtain ⟨yp, -, fp, gp⟩ := mapM_log (R := fun _ _ d => True ∧ d = []) (fun p l => by
      retl_walk []
      exact ⟨[], by simp, trivial, rfl⟩) h_proofSha
    have h10 := check_ok_iff.1 ‹Record.check (y.size == _ && _) _ = _›
    have h11 := check_ok_iff.1 ‹Record.check (streams.size == _ && _) _ = _›
    simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true_iff_forall_mem] at h10 h11
    refine ⟨y7.map Prod.snd, ?_, ?_, ?_, beq_iff_eq.1 (check_ok_iff.1 ‹Record.check (_ == spec.sigma) _ = _›),
      h10.1, h10.2, h11.1, fun x hx => (forall₂_exists_right f13 x hx).elim fun _ h => h.2⟩
    · rw [gp, flatten_snd (R := fun _ _ => True) (F := fun _ => []) fp]
      simp only [Array.toList_append, dgL_toList, g13, g12, g7, g5, flatMap_nil', List.append_nil,
        flatten_snd (R := fun _ _ => True) (F := dgl spec) f5, flatten_snd (R := fun _ _ => True)
          (F := streamLog spec) f12, ← e5, ← e12, flatten_eq_flatMap (fun _ _ h => h.1) f13, List.append_assoc]
      rfl
    · rw [e7]
      exact forall₂_fst_snd f7
    · show rootB.size = _
      rw [← Array.length_toList, e5, List.length_map, ← f5.length_eq, Array.length_toList]

open Lean (Json) in
theorem decode_log {h : ByteArray → ByteArray} {spec : SessionSpec} {text : String} {l₀ : Array ByteArray} :
    RetL (Record.decodeWith (Hashes.logged h) spec text) l₀ fun sess l' =>
      ∃ ds : List (List ByteArray), l'.toList = l₀.toList ++ (sess.rootB.toList.flatMap (dgl spec) ++
        dgl spec sess.rootF ++ ds.flatten ++ sess.streams.toList.flatMap (streamLog spec) ++
        spec.repStreams.toList.flatMap (fun x => dgl spec sess.rootB[x.1]!) ++ dgl spec (lbOf spec sess)) ∧
      List.Forall₂ (fun p d => d = dgl spec p ∨ d = [] ∧ p.size = 0) sess.publics.toList ds ∧
      sess.rootB.size = spec.tables.size ∧ sess.rootF = spec.sigma ∧
      sess.y.size = spec.points * spec.tables.size ∧ (∀ c ∈ sess.y, c.isZero = true) ∧
      sess.streams.size = spec.repStreams.size ∧
      ∀ x ∈ spec.repStreams.toList,
        ∃ s, sess.streams.find? (·.name == x.2.2) = some s ∧ 0 < s.rounds.size :=
  decodeK_log id

end FlockDeploy.KS
