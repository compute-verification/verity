module

public import Proofs.Properties.Verity.Sampled.MemberForm.Evidence
public import Proofs.Properties.Verity.Sampled.Window.Accepts

/-!
# A session's proof files, decoded from the evidence its closing round names

A session of the per-member form closes its coin tree with one more round, opened after the SHA-512 of the session's
evidence (`Spot.OneStageZk.sessionEvidence`): its record's bytes, then each rep file it names, the file's name and
its bytes. The session's verdict reads its proof files off the blobs by the same names
(`Spot.OneStageZk.filesOf`). This module decodes the proof files from the evidence's bytes alone, so that the bytes
the closing round fixes fix the files the verdict read:

* **The file under a name** (`fileNamed`): the first of the parts, name then file by twos, under the name.
* **The decoder** (`statementFiles`): the evidence's parts (`Spot.OneStageZk.evidenceParts`) after the record, and
  of them the file under each rep stream's file name, at the one table `tb`, in the rep streams' order (`repsAt`,
  which a session of the one table `tb` has, `repStreams_one`).
* **The decoder reads what the verdict read** (`statementFiles_evidence`): where a session's evidence is read off
  the blobs and its proof files are, at a session of the one table `tb`, `statementFiles tb` of the evidence's bytes
  is the proof files.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-! ## Lists -/

/-- Where `mapM` returns, the results are those of the function at each element, in order. -/
theorem filterMap_of_mapM {β γ : Type} {g : β → Except String γ} {G : β → Option γ} :
    ∀ (l : List β) (ps : List γ), l.mapM g = .ok ps → (∀ x ∈ l, ∀ p, g x = .ok p → G x = some p) →
      l.filterMap G = ps
  | [], ps, h, _ => by
    obtain rfl : [] = ps := Except.ok.inj h
    rfl
  | x :: l, ps, h, hG => by
    rw [List.mapM_cons] at h
    cases hx : g x with
    | error e => rw [hx] at h; cases h
    | ok p =>
      rw [hx] at h
      cases hl : l.mapM g with
      | error e => rw [hl] at h; cases h
      | ok ps' =>
        rw [hl] at h
        obtain rfl : p :: ps' = ps := Except.ok.inj h
        rw [List.filterMap_cons, hG x List.mem_cons_self p hx,
          filterMap_of_mapM l ps' hl fun y hy => hG y (List.mem_cons_of_mem _ hy)]

/-- Where an array's `mapM` returns, its list's does, to the result's list. -/
theorem mapM_toList {β γ : Type} {g : β → Except String γ} {a : Array β} {out : Array γ}
    (h : a.mapM g = .ok out) : a.toList.mapM g = .ok out.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM g with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    obtain rfl : l.toArray = out := Except.ok.inj h
    rfl

/-- The bytes of a byte array's list are the byte array. -/
theorem bytes_eta (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := by
  cases b
  simp

/-- Parts under the refusal `Spot.OneStageZk.sessionEvidence` makes of ones a word can't hold are under `2^64`
many, each under `2^64` bytes. -/
theorem fits_of {parts : Array ByteArray}
    (hc : ¬ (decide (_root_.Spot.OneStageZk.WORD ≤ parts.size) ||
      parts.any (fun part => decide (_root_.Spot.OneStageZk.WORD ≤ part.size))) = true) :
    parts.size < 2 ^ 64 ∧ ∀ p ∈ parts, p.size < 2 ^ 64 := by
  rw [Bool.or_eq_true, not_or] at hc
  refine ⟨Nat.lt_of_not_le fun h => hc.1 (decide_eq_true h), fun p hp => Nat.lt_of_not_le fun h => hc.2 ?_⟩
  exact Array.any_eq_true'.2 ⟨p, hp, decide_eq_true h⟩

/-- The parts after the record are the rep files' pairs, in order. -/
theorem drop_parts (record : ByteArray) (reps : Array (Array ByteArray)) :
    (#[record] ++ reps.flatten).toList.drop 1 = reps.toList.flatMap (·.toList) := by
  rw [Array.toList_append, Array.toList_flatten]
  rfl

/-! ## The file under a name -/

/-- **The file the parts hold under the name `n`**, the parts a name then a file by twos: the first under `n`'s
UTF-8 bytes. -/
def fileNamed (n : String) : List ByteArray → Option ByteArray
  | f :: b :: rest => if f = n.toUTF8 then some b else fileNamed n rest
  | _ => none

/-- **The file under a name is the blob the session names under it**: where the pairs `rs` are each of the
session's rep files `reps`, its name's bytes and the blob under its digest, the file `rs` holds under `n` is the
blob under the digest of the first of `reps` named `n`. -/
theorem fileNamed_pairs {bs : _root_.Core.Blobs} {g : String × String → Except String (Array ByteArray)}
    (hg : ∀ f d r, g (f, d) = .ok r → ∃ b w, bs.get d w = .ok b ∧ r = #[f.toUTF8, b]) :
    ∀ (reps : List (String × String)) (rs : List (Array ByteArray)), reps.mapM g = .ok rs →
      ∀ n f d w p, reps.find? (·.1 == n) = some (f, d) → bs.get d w = .ok p →
        fileNamed n (rs.flatMap (·.toList)) = some p
  | [], _, _, n, f, d, w, p, hf, _ => by cases hf
  | (f0, d0) :: reps, rs, h, n, f, d, w, p, hf, hp => by
    rw [List.mapM_cons] at h
    obtain ⟨r, hr, h⟩ := ebind_ok h
    obtain ⟨rs', hrs, h⟩ := ebind_ok h
    obtain rfl : r :: rs' = rs := Except.ok.inj h
    obtain ⟨b0, w0, hb0, rfl⟩ := hg f0 d0 r hr
    rw [List.flatMap_cons]
    show fileNamed n (f0.toUTF8 :: b0 :: rs'.flatMap (·.toList)) = some p
    rw [fileNamed]
    rw [List.find?_cons] at hf
    by_cases hn : f0 = n
    · subst hn
      simp only [beq_self_eq_true] at hf
      have hfd := Prod.mk.inj (Option.some.inj hf)
      rw [← hfd.2] at hp
      rw [ite_eq_left rfl, Except.ok.inj ((Record.get_what hb0).symm.trans hp)]
    · have hne : (f0 == n) = false := by simpa using hn
      simp only [hne] at hf
      have hne' : ¬ f0.toUTF8 = n.toUTF8 := fun e => by
        rw [String.toUTF8_eq_toByteArray, String.toUTF8_eq_toByteArray] at e
        exact hn (String.toByteArray_inj.1 e)
      rw [ite_eq_right hne']
      exact fileNamed_pairs hg reps rs' hrs n f d w p hf hp

/-! ## The decoder -/

/-- A rep stream's file's name, as `Spot.OneStageZk.filesOf` reads it. -/
def repFile (w : String) : String := s!"{w.replace "/" "."}.bin"

/-- **The rep streams of a session of the one table `tb`** (`Flock.SessionSpec.repStreams`, `repStreams_one`). -/
def repsAt (tb : String) : Array (ℕ × ℕ × String) :=
  (Array.range (#[tb] : Array String).size).flatMap fun i =>
    #[0, 1].map fun r => (i, r, s!"{(#[tb] : Array String)[i]!}/rep{r}")

/-- A session of the one table `tb` has `repsAt tb` for its rep streams. -/
theorem repStreams_one {spec : Flock.SessionSpec} {tb : String} (h : spec.tables = #[tb]) :
    spec.repStreams = repsAt tb := by
  unfold Flock.SessionSpec.repStreams repsAt
  rw [h]

/-- **A session's proof files, decoded from its evidence `w`** at the one table `tb`: of the evidence's parts after
the record, a name then a file by twos, the file under each rep stream's file name, in the rep streams' order.
Empty where `w` is no parts' bytes. -/
def statementFiles (tb : String) (w : Claims.Msg) : Array ByteArray :=
  match _root_.Spot.OneStageZk.evidenceParts ⟨w.toArray⟩ with
  | some ps => ((repsAt tb).toList.filterMap fun x => fileNamed (repFile x.2.2) (ps.toList.drop 1)).toArray
  | none => #[]

/-- **The decoder reads the proof files the verdict read**: where a session's evidence is read off the blobs
(`Spot.OneStageZk.sessionEvidence`) and its proof files are (`Spot.OneStageZk.filesOf`), at a session of the one
table `tb`, `statementFiles tb` of the evidence's bytes is the proof files. -/
theorem statementFiles_evidence {bs : _root_.Core.Blobs} {s : _root_.Spot.OneStageZk.RunFiles}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {rest : Json × Json} {proofs : Array ByteArray}
    {evidence : ByteArray} {tb : String}
    (hf : _root_.Spot.OneStageZk.filesOf bs s (.ok (spec, tabs, rest)) = .ok proofs)
    (he : _root_.Spot.OneStageZk.sessionEvidence bs s = .ok evidence) (hspec : spec.tables = #[tb]) :
    statementFiles tb evidence.data.toList = proofs := by
  unfold _root_.Spot.OneStageZk.sessionEvidence at he
  obtain ⟨record, -, he⟩ := ebind_ok he
  obtain ⟨reps, hreps, he⟩ := ebind_ok he
  dsimp only at he
  have hc : ¬ (decide (_root_.Spot.OneStageZk.WORD ≤ (#[record] ++ reps.flatten).size) ||
      (#[record] ++ reps.flatten).any (fun part => decide (_root_.Spot.OneStageZk.WORD ≤ part.size))) = true := by
    intro hc
    rw [ite_eq_left hc] at he
    cases he
  rw [ite_eq_right hc] at he
  have hev : _root_.Spot.OneStageZk.evidenceBytes (#[record] ++ reps.flatten) = evidence := Except.ok.inj he
  obtain ⟨hn, hp⟩ := fits_of hc
  have hparts := evidenceParts_evidenceBytes _ hn hp
  rw [hev] at hparts
  have hpairs := fileNamed_pairs (bs := bs)
    (fun f d r h => by
      obtain ⟨b, hb, h⟩ := ebind_ok h
      exact ⟨b, _, hb, (Except.ok.inj h).symm⟩)
    s.reps.toList reps.toList (mapM_toList hreps)
  unfold _root_.Spot.OneStageZk.filesOf at hf
  change spec.repStreams.mapM _ = _ at hf
  rw [repStreams_one hspec] at hf
  unfold statementFiles
  rw [bytes_eta, hparts]
  dsimp only
  rw [drop_parts]
  refine (congrArg List.toArray (filterMap_of_mapM _ _ (mapM_toList hf) fun x _ p hx => ?_)).trans
    (Array.toArray_toList (xs := proofs))
  obtain ⟨i, r, w⟩ := x
  dsimp only at hx
  split at hx
  · rename_i f d hfind
    exact hpairs _ f d _ p (Array.find?_toList.trans hfind) hx
  · cases hx

end Verity.Sampled.MemberForm
