module

public import Proofs.Properties.Verity.Sampled.Instance

/-!
# The members' files v2 reads on the blobs sent

At parameters naming one-stage v2's members, v2 reads each member's circuit file and public file at the first blob
sent with the member's digest (`Core.Blobs.get`): the committed files', where they hold one, else the evidence's
(`filesOn_play`). The commit phase's check holds the committed files to two digests (`Vocab.Registered`), so a
member's file may come from the evidence. This module states that v2 reads each member's registered file, and has a
read finder output its failure, a SHA-512 collision:

* **The members' digests** (`memberDigests`): each member's circuit file's and public file's, at the parameters.
* **The files sent** (`blobWith`, `FilesSent`): at each member's digest, the first blob sent with it is the first of
  the registered files with it.
* **The collision** (`filesCollision`, `filesCollision_valid`): the first member's digest at which the two differ,
  of one SHA-512.
* **The window's read finder** (`windowReadFinder`): that collision, else the binding's read finder's output
  (`Binding.readFinder`), at the binding's read finder's calls. It outputs a collision wherever the binding's read
  finder does (`windowReadFinder_of`) and wherever `FilesSent` fails (`finder_of_files`).
* **(a) with the members' files** (`logged_files_le`): `Binding.logged_tables_le` with `FilesSent` beside
  `TablesSent`, at the window's read finder.
-/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

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

/-- **The blobs sent hold each member's registered file**, at the play's parameters: at each member's digest
(`memberDigests`), the first blob sent with it is the first of the registered files with it. -/
def FilesSent (V : Vocab d α) (o : Exchange) : Prop :=
  ∀ h ∈ memberDigests o.params, ∀ a b, blobWith h o.sent.blobs = some a →
    blobWith h (V.registered o.params) = some b → a = b

open Classical in
/-- **A member's file v2 reads on the blobs sent at other bytes than its registered file**, of one SHA-512 with it:
at the first member's digest whose first blob sent and first registered file differ, the two. -/
noncomputable def filesCollision (V : Vocab d α) (o : Exchange) : Option (Claims.Msg × Claims.Msg) :=
  (memberDigests o.params).findSome? fun h =>
    match blobWith h o.sent.blobs, blobWith h (V.registered o.params) with
    | some a, some b => if a = b then none else some (a.data.toList, b.data.toList)
    | _, _ => none

/-- What `filesCollision` outputs is a collision of `sha512Msg`. -/
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

/-- **The window's read finder at a strategy**, at the device's history at each play `dev`: on a uniform tape of
the play, a member's file v2 reads on the blobs sent at other bytes than its registered file (`filesCollision`),
else the binding's read finder's output (`Binding.readFinder`), at the binding's read finder's calls: the collision
looks the blobs sent up by the SHA-512 the read computes of each, and the registered files by the digests the
parameters register. -/
noncomputable def windowReadFinder (V : Vocab d α) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t := (filesCollision V (Verity.Game.Tape.out _ σ t)).or ((Binding.readFinder V dev θ F c σ).out t)
  cost t := (Binding.readFinder V dev θ F c σ).cost t

theorem windowReadFinder_out (V : Vocab d α) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ) :
    (windowReadFinder V dev θ F c σ).out t =
      (filesCollision V (Verity.Game.Tape.out _ σ t)).or ((Binding.readFinder V dev θ F c σ).out t) :=
  rfl

/-- **Where the binding's read finder outputs a collision, the window's does.** -/
theorem windowReadFinder_of (V : Vocab d α) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ)
    (h : ∃ u v, (Binding.readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) :
    ∃ u v, (windowReadFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  rw [windowReadFinder_out]
  cases hf : filesCollision V (Verity.Game.Tape.out _ σ t) with
  | some uv =>
    obtain ⟨u, v⟩ := uv
    exact ⟨u, v, rfl, filesCollision_valid hf⟩
  | none => exact h

/-- **A run whose blobs sent hold a member's file at other bytes than its registered file is one where the window's
read finder succeeds** (`FilesSent`, `filesCollision`): the two files' SHA-512s are the member's digest. -/
theorem finder_of_files (V : Vocab d α) (dev : Exchange → Properties.Challenger.History) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (t : Verity.Game.Tape (V.game θ F c) σ) (h : ¬ FilesSent V (Verity.Game.Tape.out _ σ t)) :
    ∃ u v, (windowReadFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  rw [windowReadFinder_out]
  cases hf : filesCollision V (Verity.Game.Tape.out _ σ t) with
  | some uv =>
    obtain ⟨u, v⟩ := uv
    exact ⟨u, v, rfl, filesCollision_valid hf⟩
  | none =>
    refine (h fun x hx a b ha hb => ?_).elim
    by_contra hab
    have hn := List.findSome?_eq_none_iff.1 hf x hx
    simp [ha, hb, hab] at hn

/-- **(a) with v2's tables and the members' files: an accepted play whose log the check read isn't the device's,
whose blobs sent hold a pinned table at other bytes than its registered file (`TablesSent`), or whose blobs sent hold
a member's file at other bytes than its registered file (`FilesSent`), is at most as likely as the window's read
finder's collision plus the forger's forgery**, under `Binding.logged_le`'s premises: the window's read finder
outputs a collision wherever the binding's does (`windowReadFinder_of`) and wherever `FilesSent` fails
(`finder_of_files`). -/
theorem logged_files_le (V : Vocab d α) (k : Verity.Registration.Composite)
    (dev : Exchange → Properties.Challenger.History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hb : Binding.Binds V k Binding.readsV2 dev θ)
    (hs : Binding.SignsPlay V dev θ F c) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧
        ¬ (Binding.Logged Binding.readsV2 o ∧ TablesSent o.sent ∧ FilesSent V o)) (V.game θ F c) σ ≤
      ENNReal.ofReal ((windowReadFinder V dev θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal ((Binding.forger V dev θ F c σ).Pr Verity.lmsVerify) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) ∧
          ¬ (Binding.Logged Binding.readsV2 (Verity.Game.Tape.out _ σ t) ∧
            TablesSent (Verity.Game.Tape.out _ σ t).sent ∧ FilesSent V (Verity.Game.Tape.out _ σ t)))
      ≤ Verity.Game.prCoin (fun t =>
          (∃ u v, (windowReadFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
            (Binding.forger V dev θ F c σ).Forges Verity.lmsVerify t) := by
        refine Verity.Game.prCoin_mono fun t ⟨hv, hn⟩ => ?_
        by_cases hf : FilesSent V (Verity.Game.Tape.out _ σ t)
        swap
        · exact .inl (finder_of_files V dev θ F c σ t hf)
        by_cases hl : Binding.Logged Binding.readsV2 (Verity.Game.Tape.out _ σ t)
        · exact .inl (windowReadFinder_of V dev θ F c σ t
            (Binding.finder_of_tables V dev θ F c σ t fun ht => hn ⟨hl, ht, hf⟩))
        have hθ := (Binding.play_reaches (Verity.Game.Tape.reaches_out _ σ t)).1
        rcases hb _ hθ hv hl with ⟨x, hx, hk, hbr⟩ | ⟨u, v, huv, hne, hh⟩
        · exact (Binding.finders_of_break V dev θ F c σ t hs hx hk hbr).imp_left
            (windowReadFinder_of V dev θ F c σ t)
        · exact .inl (windowReadFinder_of V dev θ F c σ t (Binding.finder_of_open V dev θ F c σ t huv ⟨hne, hh⟩))
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

/-- **Every accepted play outside the window's read finder's collisions has its tables and members' files**: the
sampled verdict's acceptance at a play whose blobs sent hold a pinned table or a member's file at other bytes than
its registered file is at most as likely as the window's read finder's collision, at plays where those that hold
them aren't accepted (`finder_of_tables`, `finder_of_files`). -/
theorem verdict_le_files (V : Vocab d α) (k : Verity.Registration.Composite)
    (dev : Exchange → Properties.Challenger.History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c))
    (h : ∀ t : Verity.Game.Tape (V.game θ F c) σ, V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) →
      TablesSent (Verity.Game.Tape.out _ σ t).sent → FilesSent V (Verity.Game.Tape.out _ σ t) → False) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((windowReadFinder V dev θ F c σ).Pr Verity.sha512Msg) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t))
      ≤ Verity.Game.prCoin (fun t =>
          ∃ u v, (windowReadFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) := by
        refine Verity.Game.prCoin_mono fun t hv => ?_
        by_cases hf : FilesSent V (Verity.Game.Tape.out _ σ t)
        swap
        · exact finder_of_files V dev θ F c σ t hf
        by_cases ht : TablesSent (Verity.Game.Tape.out _ σ t).sent
        · exact (h t hv ht hf).elim
        exact windowReadFinder_of V dev θ F c σ t (Binding.finder_of_tables V dev θ F c σ t ht)
    _ = _ := Verity.Game.prCoin_eq_ofReal_pr _

end Verity.Sampled
