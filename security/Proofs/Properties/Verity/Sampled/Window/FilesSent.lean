module

public import Proofs.Properties.Verity.Sampled.Instance

/-! The members' files v2 reads on the blobs sent. -/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

open Classical in
/-- **The first of the files `B` whose SHA-512 is `h`**, as v2 looks a file up by its digest (`Verity.namedFile`
is it, empty where none is). -/
noncomputable def blobWith (h : Option Hash.Digest) (B : Array ByteArray) : Option ByteArray :=
  B.find? fun f => decide (h = some (Verity.sha512 f.data.toList))

theorem blobWith_digest {h : Option Hash.Digest} {B : Array ByteArray} {f : ByteArray} (hf : blobWith h B = some f) :
    h = some (Verity.sha512 f.data.toList) := by
  unfold blobWith at hf
  have hp := Array.find?_some hf
  simpa using hp

/-- **The members' digests at the parameters `θ`**: each member's circuit file's and public file's, as v2's
parameters name them (`digestOf`); none where they name no members. -/
def memberDigests (θ : ByteArray) : List (Option Hash.Digest) :=
  match (paramsOf θ).bind (·.members) with
  | some ms => ms.toList.flatMap fun vm => [digestOf vm.own.circuit, digestOf vm.own.pub]
  | none => []

def FilesSent (V : Vocab d α) (o : Exchange) : Prop :=
  ∀ h ∈ memberDigests o.params, ∀ a b, blobWith h o.sent.blobs = some a →
    blobWith h (V.registered o.params) = some b → a = b

open Classical in
noncomputable def filesCollision (V : Vocab d α) (o : Exchange) : Option (Claims.Msg × Claims.Msg) :=
  (memberDigests o.params).findSome? fun h =>
    match blobWith h o.sent.blobs, blobWith h (V.registered o.params) with
    | some a, some b => if a = b then none else some (a.data.toList, b.data.toList)
    | _, _ => none

theorem filesCollision_valid {V : Vocab d α} {o : Exchange} {u v : Claims.Msg} (h : filesCollision V o = some (u, v)) :
    Claims.Collision Verity.sha512Msg u v := by
  obtain ⟨x, -, hx⟩ := List.exists_of_findSome?_eq_some h
  split at hx
  · rename_i a b ha hb
    split at hx
    · cases hx
    · rename_i hab
      cases hx
      exact ⟨fun e => hab (ByteArray.ext (Array.ext' e)),
        sha512Msg_of_sha512 (Option.some.inj ((blobWith_digest ha).symm.trans (blobWith_digest hb)))⟩
  · cases hx

noncomputable def executionPhaseReadFinder (V : Vocab d α) (T : PinnedTables) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t := (filesCollision V (Verity.Game.Tape.out _ σ t)).or ((Binding.readFinder V T dev θ F c σ).out t)
  cost t := (Binding.readFinder V T dev θ F c σ).cost t

theorem executionPhaseReadFinder_out (V : Vocab d α) (T : PinnedTables) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ) :
    (executionPhaseReadFinder V T dev θ F c σ).out t =
      (filesCollision V (Verity.Game.Tape.out _ σ t)).or ((Binding.readFinder V T dev θ F c σ).out t) :=
  rfl

theorem executionPhaseReadFinder_of (V : Vocab d α) (T : PinnedTables) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ)
    (h : ∃ u v, (Binding.readFinder V T dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) :
    ∃ u v, (executionPhaseReadFinder V T dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  rw [executionPhaseReadFinder_out]
  cases hf : filesCollision V (Verity.Game.Tape.out _ σ t) with
  | some uv =>
    obtain ⟨u, v⟩ := uv
    exact ⟨u, v, rfl, filesCollision_valid hf⟩
  | none => exact h

theorem finder_of_files (V : Vocab d α) (T : PinnedTables) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ) (h : ¬ FilesSent V (Verity.Game.Tape.out _ σ t)) :
    ∃ u v, (executionPhaseReadFinder V T dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  rw [executionPhaseReadFinder_out]
  cases hf : filesCollision V (Verity.Game.Tape.out _ σ t) with
  | some uv =>
    obtain ⟨u, v⟩ := uv
    exact ⟨u, v, rfl, filesCollision_valid hf⟩
  | none =>
    refine (h fun x hx a b ha hb => ?_).elim
    by_contra hab
    have hn := List.findSome?_eq_none_iff.1 hf x hx
    simp [ha, hb, hab] at hn

theorem logged_files_le (V : Vocab d α) (T : PinnedTables) (k : Verity.Registration.Composite)
    (dev : Exchange → Properties.Challenger.History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hb : Binding.Binds V k Binding.readsV2 dev θ)
    (hs : Binding.SignsPlay V dev θ F c) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧
        ¬ (Binding.Logged Binding.readsV2 o ∧ TablesSent T o.sent ∧ FilesSent V o)) (V.game θ F c) σ ≤
      ENNReal.ofReal ((executionPhaseReadFinder V T dev θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal ((Binding.forger V dev θ F c σ).Pr Verity.lmsVerify) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) ∧
          ¬ (Binding.Logged Binding.readsV2 (Verity.Game.Tape.out _ σ t) ∧
            TablesSent T (Verity.Game.Tape.out _ σ t).sent ∧ FilesSent V (Verity.Game.Tape.out _ σ t)))
      ≤ Verity.Game.prCoin (fun t =>
          (∃ u v, (executionPhaseReadFinder V T dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
            (Binding.forger V dev θ F c σ).Forges Verity.lmsVerify t) := by
        refine Verity.Game.prCoin_mono fun t ⟨hv, hn⟩ => ?_
        by_cases hf : FilesSent V (Verity.Game.Tape.out _ σ t)
        swap
        · exact .inl (finder_of_files V T dev θ F c σ t hf)
        by_cases hl : Binding.Logged Binding.readsV2 (Verity.Game.Tape.out _ σ t)
        · exact .inl (executionPhaseReadFinder_of V T dev θ F c σ t
            (Binding.finder_of_tables V T dev θ F c σ t fun ht => hn ⟨hl, ht, hf⟩))
        have hθ := (Binding.play_reaches (Verity.Game.Tape.reaches_out _ σ t)).1
        rcases hb _ hθ hv hl with ⟨x, hx, hk, hbr⟩ | ⟨u, v, huv, hne, hh⟩
        · exact (Binding.finders_of_break V T dev θ F c σ t hs hx hk hbr).imp_left
            (executionPhaseReadFinder_of V T dev θ F c σ t)
        · exact .inl (executionPhaseReadFinder_of V T dev θ F c σ t (Binding.finder_of_open V T dev θ F c σ t huv ⟨hne, hh⟩))
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

theorem verdict_le_files (V : Vocab d α) (T : PinnedTables) (k : Verity.Registration.Composite)
    (dev : Exchange → Properties.Challenger.History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c))
    (h : ∀ t : Verity.Game.Tape (V.game θ F c) σ, V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) →
      TablesSent T (Verity.Game.Tape.out _ σ t).sent → FilesSent V (Verity.Game.Tape.out _ σ t) → False) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((executionPhaseReadFinder V T dev θ F c σ).Pr Verity.sha512Msg) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t))
      ≤ Verity.Game.prCoin (fun t =>
          ∃ u v, (executionPhaseReadFinder V T dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) := by
        refine Verity.Game.prCoin_mono fun t hv => ?_
        by_cases hf : FilesSent V (Verity.Game.Tape.out _ σ t)
        swap
        · exact finder_of_files V T dev θ F c σ t hf
        by_cases ht : TablesSent T (Verity.Game.Tape.out _ σ t).sent
        · exact (h t hv ht hf).elim
        exact executionPhaseReadFinder_of V T dev θ F c σ t (Binding.finder_of_tables V T dev θ F c σ t ht)
    _ = _ := Verity.Game.prCoin_eq_ofReal_pr _

end Verity.Sampled
