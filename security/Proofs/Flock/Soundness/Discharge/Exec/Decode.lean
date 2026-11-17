module

public import Proofs.Flock.Soundness.Refine.LiveSim
public import Proofs.Flock.Soundness.Refine.Regions
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Refine.Seam

/-! Two pieces of `DecodesAll`: a run on two lists is a run on one tape (`Plays.toT`), and the accepted proof's cap is a function of `root_B` alone (`capOfRoot`). -/

@[expose] public section

namespace FlockSoundness.Game

variable {W F : Type} [Inhabited W] {ι : W → F}

/-- A coin step as the coin array `c` gives it to a run: the vector of its words, or at a node of type `F`, its one
word. -/
def CoinOf (ι : W → F) (x : Msg) (c : Array W) : Prop :=
  x = ⟨Fin c.size → F, fun i => ι c[i.val]!⟩ ∨ (c.size = 1 ∧ x = ⟨F, ι c[0]!⟩)

theorem Plays.toT {α : Type} {g : Game α} {ms : List Msg} {cs : List (Array W)} {a : α} (h : Plays ι g ms cs a) :
    ∃ ts : List Step, PlaysT g ts a ∧ ts.filterMap Sum.getLeft? = ms ∧
      List.Forall₂ (CoinOf ι) (ts.filterMap Sum.getRight?) cs := by
  induction h with
  | ret a => exact ⟨[], .ret a, rfl, .nil⟩
  | send m _ ih =>
      obtain ⟨ts, hp, h1, h2⟩ := ih
      refine ⟨.inl ⟨_, m⟩ :: ts, .send m hp, ?_, ?_⟩
      · rw [List.filterMap_cons, Sum.getLeft?_inl, h1]
      · rw [List.filterMap_cons, Sum.getRight?_inl]
        exact h2
  | coinVec c hc _ _ ih =>
      obtain ⟨ts, hp, h1, h2⟩ := ih
      subst hc
      refine ⟨.inr ⟨_, fun i => ι c[i.val]!⟩ :: ts, .coin _ hp, ?_, ?_⟩
      · rw [List.filterMap_cons, Sum.getLeft?_inr, h1]
      · rw [List.filterMap_cons, Sum.getRight?_inr]
        exact List.Forall₂.cons (Or.inl rfl) h2
  | coinOne c hc _ ih =>
      obtain ⟨ts, hp, h1, h2⟩ := ih
      refine ⟨.inr ⟨F, ι c[0]!⟩ :: ts, .coin _ hp, ?_, ?_⟩
      · rw [List.filterMap_cons, Sum.getLeft?_inr, h1]
      · rw [List.filterMap_cons, Sum.getRight?_inr]
        exact List.Forall₂.cons (Or.inr ⟨hc, rfl⟩) h2

end FlockSoundness.Game

namespace FlockSoundness.Discharge.Exec

open Flock Refine Model

theorem flat_length (l : List ByteArray) (hl : ∀ x ∈ l, x.size = 64) :
    (l.map fun b => b.data.toList).flatten.length = 64 * l.length := by
  induction l with
  | nil => simp
  | cons x l ih =>
      simp only [List.map_cons, List.flatten_cons, List.length_append, Array.length_toList, ByteArray.size_data,
        List.length_cons]
      rw [hl x List.mem_cons_self, ih fun y hy => hl y (List.mem_cons_of_mem _ hy)]
      ring

theorem entries_eq_of_foldl {a b : Array ByteArray} (ha : ∀ x ∈ a.toList, x.size = 64)
    (hb : ∀ x ∈ b.toList, x.size = 64)
    (h : a.foldl (· ++ ·) ByteArray.empty = b.foldl (· ++ ·) ByteArray.empty) : a = b := by
  have e₀ := congrArg (fun x : ByteArray => x.data.toList) h
  simp only [← Array.foldl_toList, flat_data] at e₀
  have e : (a.toList.map fun b => b.data.toList).flatten = (b.toList.map fun b => b.data.toList).flatten := by
    simpa using e₀
  have hlen : a.toList.length = b.toList.length := by
    have := congrArg List.length e
    rw [flat_length _ ha, flat_length _ hb] at this
    omega
  exact Array.ext' (entries_eq_of_flat _ _ ha hb hlen e)

open Classical in
/-- **The cap read off `root_B`**: the `capDig` of the array of 64-byte entries whose concatenation is `root_B`, when
there is one. -/
noncomputable def capOfRoot (rootB : ByteArray) : Cap Dig :=
  if h : ∃ c : Array ByteArray, (∀ x ∈ c.toList, x.size = 64) ∧ c.foldl (· ++ ·) ByteArray.empty = rootB then
    capDig (Classical.choose h)
  else capDig #[]

theorem capOfRoot_eq {c : Array ByteArray} {rootB : ByteArray} (hc : ∀ x ∈ c.toList, x.size = 64)
    (h : c.foldl (· ++ ·) ByteArray.empty = rootB) : capOfRoot rootB = capDig c := by
  have hex : ∃ c : Array ByteArray, (∀ x ∈ c.toList, x.size = 64) ∧ c.foldl (· ++ ·) ByteArray.empty = rootB :=
    ⟨c, hc, h⟩
  unfold capOfRoot
  rw [dite_eq_left hex]
  obtain ⟨hc', h'⟩ := Classical.choose_spec hex
  rw [entries_eq_of_foldl hc' hc (h'.trans h.symm)]

theorem verify_cap_hm96 (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (extra : RunRecord → List (Model.Claim FlockLevel3.GF128 S.m))
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode spec record = .ok sess → ExtraRealizes st S sess.points (extra sess))
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ (sess : RunRecord) (sch : Flock.Schedule) (p₀ : Proof),
      Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[0]! = .ok p₀ ∧
      capDig p₀.cap = capOfRoot sess.rootB[0]! := by
  obtain ⟨sess, sch, c, hsess, hf, h0, -⟩ := verify_one_ok h
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  have hwf := decode_roundsWF hsess
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hschm : sch.m = S.m := hsm.trans hmS
  obtain ⟨-, p₀, -, hp₀, hc₀, hroot, -, hw₀, -⟩ := rep_refines (D := Dig) spec st S hm h15 sch hmS hkS hfold
    (extra sess) sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 0 _ _ h0
  rw [hms] at hw₀
  exact ⟨sess, sch, p₀, hsess, hf, hp₀, (capOfRoot_eq hw₀ (hc₀.symm.trans (bytes_eq_of_beq hroot))).symm⟩

theorem verify_cap_ofCircuit_hm96 (st : Flock.Stmt) (hwf : StmtWF st) (hr : RegionsWF st) (hm : 13 ≤ st.m)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (record : String) (proofs : Array ByteArray)
    (h : Flock.verify st.spec #[Setup.ofCircuit st] record proofs = .ok ()) :
    ∃ (sess : RunRecord) (sch : Flock.Schedule) (p₀ : Proof),
      Record.decode st.spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[0]! = .ok p₀ ∧
      capDig p₀.cap = capOfRoot sess.rootB[0]! :=
  verify_cap_hm96 st.spec (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) hm rfl rfl (ofCircuit_fold st hwf _)
    hms (fun sess => Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points))
    record proofs
    (fun sess hsess => ofCircuit_extra st hwf hr sess.points (decode_points hsess).1 (decode_points hsess).2) h

end FlockSoundness.Discharge.Exec
