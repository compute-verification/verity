module

public import Proofs.Properties.Verity.SampledTwo.CutPrints
public import Proofs.Properties.Verity.Sampled.MemberForm
public import Proofs.Properties.Verity.SampledTwo.Steps

/-!
# The two-stage form holds the interior's records to its session's trees

`CutPrints.binds_of_prints` takes, of each accepted play, that the interior's records are held to the coin trees its
session's lines hold (`InteriorHeld`). The form's check reads them so: where it accepts, it ran the interior audit's
per-member sessions (`accepted_steps`, `inputAt_ok`: `OneStageZk.membersRun`), which reads the challenger's trees
from the interior's coin evidence (`treesOf`) and holds every statement's session's record to them
(`membersTreesHeld`): `membersTreeReason` accepts each as `memberTreeReason` does (`MemberForm.membersTreeReason_none`),
a logged tree whose last round closes the session and which, less that round, passes `sessionTreeReason`
(`MemberForm.memberTreeReason_none`). The trees `treesOf` reads are those `coinTreesOf` reads from the lines `linesOf`
reads (`Binding.trees_of`).

* `members_trees`: an accepted `membersRun` read its trees and held its statements' sessions to them.
* `held_size`: the statements' members (`membersHeld`) are one a statement, so every statement is in `sts.zip held`.
* `named_mem`: `membersTreesHeld`'s two loops name every statement's session whose record is on the blobs.
* `interiorHeld_of_accepts`: `Accepts s → InteriorHeld s`, and so `binds_of_accepts`: `BindsOfPrints` with no premise
  beyond the printing.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger

/-! ## Lists -/

theorem mapM_str_inv {f : Json → Flock.V String} (hf : ∀ j x, f j = .ok x → j = .str x) :
    ∀ (js : List Json) (l : List String), js.mapM f = .ok l → js = l.map Json.str
  | [], l, h => by
    rw [List.mapM_nil] at h
    obtain rfl : [] = l := Except.ok.inj h
    rfl
  | j :: js, l, h => by
    rw [List.mapM_cons] at h
    obtain ⟨x, hx, h⟩ := Verity.Sampled.Trees.bind_ok h
    obtain ⟨xs, hxs, h⟩ := Verity.Sampled.Trees.bind_ok h
    obtain rfl : x :: xs = l := Except.ok.inj h
    simp [hf j x hx, mapM_str_inv hf js xs hxs]

/-- The lines `linesOf` reads are the session's log's, as JSON strings. -/
theorem linesOf_arr {e : Spot.OneStageZk.Evidence} {l : List String} (h : Verity.Sampled.linesOf e = .ok l) :
    ∃ ls : Array Json, Spot.getD (Spot.getD e.base.coins "session") "log" = .arr ls ∧ ls.toList = l.map Json.str := by
  unfold Verity.Sampled.linesOf at h
  split at h
  · rename_i ls hls
    exact ⟨ls, hls, mapM_str_inv (fun j x hj => by cases j <;> cases hj <;> rfl) _ _ h⟩
  · first
      | cases h
      | simp [throw, throwThe, MonadExceptOf.throw] at h

theorem mem_zip_left {α β : Type} {a : α} :
    ∀ (as : List α) (bs : List β), bs.length = as.length → a ∈ as → ∃ b, (a, b) ∈ as.zip bs
  | [], _, _, ha => by cases ha
  | _ :: _, [], hl, _ => by simp at hl
  | x :: as, b :: bs, hl, ha => by
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨b, by simp⟩
    · obtain ⟨b', hb'⟩ := mem_zip_left as bs (by simpa using hl) ha
      exact ⟨b', by simp [hb']⟩

/-! ## The statements' members -/

theorem heldStep_size {ms held h' : Array Spot.OneStageZk.Member} {st : Spot.OneStageZk.StatementFiles}
    (h : Spot.OneStageZk.heldStep ms held st = .ok h') : h'.size = held.size + 1 := by
  unfold Spot.OneStageZk.heldStep at h
  repeat' split at h
  all_goals first
    | (cases h; simp)
    | cases h
    | simp [throw, throwThe, MonadExceptOf.throw] at h

theorem foldlM_held_size {ms : Array Spot.OneStageZk.Member} :
    ∀ (l : List Spot.OneStageZk.StatementFiles) (acc out : Array Spot.OneStageZk.Member),
      l.foldlM (Spot.OneStageZk.heldStep ms) acc = .ok out → out.size = acc.size + l.length
  | [], acc, out, h => by
    rw [List.foldlM_nil] at h
    obtain rfl : acc = out := Except.ok.inj h
    simp
  | st :: l, acc, out, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨a, ha, h⟩ := Verity.Sampled.Trees.bind_ok h
    rw [foldlM_held_size l a out h, heldStep_size ha]
    simp only [List.length_cons]
    omega

/-- **The statements' members are one a statement.** -/
theorem held_size {ms held : Array Spot.OneStageZk.Member} {sts : Array Spot.OneStageZk.StatementFiles}
    (h : Spot.OneStageZk.membersHeld ms sts = .ok held) : held.size = sts.size := by
  unfold Spot.OneStageZk.membersHeld at h
  rw [← Array.foldlM_toList] at h
  simpa using foldlM_held_size sts.toList #[] held h

/-! ## The per-member sessions held to the trees -/

/-- **An accepted `membersRun` read the challenger's trees and held its statements' sessions to them.** -/
theorem members_trees {p : Spot.OneStageZk.Params} {ms : Array Spot.OneStageZk.Member}
    {c : Spot.OneStage.Commitment} {e : Spot.OneStageZk.Evidence} {sts : Array Spot.OneStageZk.StatementFiles}
    {bs : _root_.Core.Blobs} {r : Spot.OneStageZk.Run} (h : Spot.OneStageZk.membersRun p ms c e sts bs = .ok r) :
    ∃ held ts L, Spot.OneStageZk.membersHeld ms sts = .ok held ∧ Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
      Spot.OneStageZk.membersTreesHeld ts L (sts.zip held) bs = .ok () := by
  unfold Spot.OneStageZk.membersRun at h
  obtain ⟨_, h, -⟩ := Verity.Sampled.Trees.bind_ok h
  unfold Spot.OneStageZk.membersStructure at h
  dsimp only at h
  obtain ⟨held, hheld, h⟩ := Verity.Sampled.Trees.bind_ok h
  have hrest : ∃ ts L, Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧
      Spot.OneStageZk.membersTreesHeld ts L (sts.zip held) bs = .ok () := by
    first
      | (obtain ⟨_, -, h⟩ := Verity.Sampled.Trees.bind_ok h
         obtain ⟨ts, hts, h⟩ := Verity.Sampled.Trees.bind_ok h
         obtain ⟨_, -, h⟩ := Verity.Sampled.Trees.bind_ok h
         obtain ⟨L, -, h⟩ := Verity.Sampled.Trees.bind_ok h
         obtain ⟨_, hM, -⟩ := Verity.Sampled.Trees.bind_ok h
         exact ⟨ts, L, hts, hM⟩)
      | (split at h
         · first
             | (obtain ⟨_, h', -⟩ := Verity.Sampled.Trees.bind_ok h; cases h')
             | cases h
         · obtain ⟨ts, hts, h⟩ := Verity.Sampled.Trees.bind_ok h
           obtain ⟨_, -, h⟩ := Verity.Sampled.Trees.bind_ok h
           obtain ⟨L, -, h⟩ := Verity.Sampled.Trees.bind_ok h
           obtain ⟨_, hM, -⟩ := Verity.Sampled.Trees.bind_ok h
           exact ⟨ts, L, hts, hM⟩)
  obtain ⟨ts, L, hts, hM⟩ := hrest
  exact ⟨held, ts, L, hheld, hts, hM⟩

/-- **`membersTreesHeld` names every statement's session whose record is on the blobs**: its two loops push, of each
session of each statement with its member, the record `Flock.JsonParse.parse` reads, beside its tables and its
evidence's digest. -/
theorem named_mem {ts : Array LoggedTree} {L : List String}
    {sts : Array (Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member)} {bs : _root_.Core.Blobs}
    (h : Spot.OneStageZk.membersTreesHeld ts L sts bs = .ok ()) :
    ∃ named : Array (String × Array (String × Option ℕ) × Json × String),
      membersTreeReason ts L named = none ∧
      ∀ x ∈ sts.toList, ∀ s ∈ x.1.sessions.toList, ∀ w b str, (bs.get s.record w).toOption = some b →
        String.fromUTF8? b = some str → ∃ tabs ev, (s.name, tabs, (Flock.JsonParse.parse str).toOption.getD .null, ev) ∈ named := by
  unfold Spot.OneStageZk.membersTreesHeld at h
  dsimp only at h
  obtain ⟨named, hn, hrest⟩ := Verity.Sampled.Trees.bind_ok h
  rw [← Array.forIn_toList] at hn
  have hI := Verity.Sampled.Trees.forIn_inv
    (I := fun pre (named : Array (String × Array (String × Option ℕ) × Json × String)) =>
      ∀ x ∈ pre, ∀ s ∈ x.1.sessions.toList, ∀ w b str, (bs.get s.record w).toOption = some b →
        String.fromUTF8? b = some str → ∃ tabs ev, (s.name, tabs, (Flock.JsonParse.parse str).toOption.getD .null, ev) ∈ named)
    ?_ sts.toList [] #[] named (by simp) hn
  · refine ⟨named, ?_, by simpa using hI⟩
    split at hrest
    · cases hrest
    · rename_i hne
      cases hw : membersTreeReason ts L named with
      | none => rfl
      | some w => exact (hne w hw).elim
  · rintro pre ⟨st, m⟩ named' res hI' hres
    dsimp only at hres
    obtain ⟨named'', hin, hres⟩ := Verity.Sampled.Trees.bind_ok hres
    obtain rfl : ForInStep.yield named'' = res := Except.ok.inj hres
    rw [← Array.forIn_toList] at hin
    have hJ := Verity.Sampled.Trees.forIn_inv
      (I := fun pre' (n : Array (String × Array (String × Option ℕ) × Json × String)) =>
        (∀ y ∈ named', y ∈ n) ∧ ∀ s ∈ pre', ∀ w b str, (bs.get s.record w).toOption = some b →
          String.fromUTF8? b = some str → ∃ tabs ev, (s.name, tabs, (Flock.JsonParse.parse str).toOption.getD .null, ev) ∈ n)
      ?_ st.sessions.toList [] named' named'' ⟨fun _ hy => hy, by simp⟩ hin
    · refine ⟨named'', rfl, fun x hx s hs w b str hb hstr => ?_⟩
      rcases List.mem_append.1 hx with hx | hx
      · obtain ⟨tabs, ev, hm⟩ := hI' x hx s hs w b str hb hstr
        exact ⟨tabs, ev, hJ.1 _ hm⟩
      · rw [List.mem_singleton] at hx
        subst hx
        exact hJ.2 s (by simpa using hs) w b str hb hstr
    · intro pre' s n res' hJ' hres'
      obtain ⟨b', hb', hres'⟩ := Verity.Sampled.Trees.bind_ok hres'
      split at hres'
      case h_2 => first | cases hres' | simp [throw, throwThe, MonadExceptOf.throw] at hres'
      rename_i record hrec
      obtain ⟨ev, -, hres'⟩ := Verity.Sampled.Trees.bind_ok hres'
      obtain rfl := Except.ok.inj hres'
      refine ⟨_, rfl, fun y hy => Array.mem_push.2 (.inl (hJ'.1 y hy)), fun s' hs' w'' b'' str' hb'' hs'' => ?_⟩
      rcases List.mem_append.1 hs' with hs' | hs'
      · obtain ⟨tabs, ev', hm⟩ := hJ'.2 s' hs' w'' b'' str' hb'' hs''
        exact ⟨tabs, ev', Array.mem_push.2 (.inl hm)⟩
      · rw [List.mem_singleton] at hs'
        subst hs'
        obtain rfl : b'' = b' := Option.some.inj (hb''.symm.trans (Verity.Sampled.Trees.get_ok w'' hb'))
        obtain rfl : str' = record := Option.some.inj (hs''.symm.trans hrec)
        exact ⟨_, _, Array.mem_push.2 (.inr rfl)⟩

/-! ## The interior held -/

/-- **The form's check holds the interior's records to its session's trees** (`InteriorHeld`): where it accepts, each
record `secondRecords` reads is a statement's session's, which `membersTreesHeld` named; `membersTreeReason` accepted
it as `memberTreeReason` does, at a tree less its closing round; and the trees `treesOf` read are those `coinTreesOf`
reads from the session's lines. -/
theorem interiorHeld_of_accepts {s : Verity.Sent} (ha : Accepts s) : InteriorHeld s := by
  intro p c e h1 recs hrecs lines hlines ts hts rec hrec
  obtain ⟨p', c', e', pub, prof, h1', hacc⟩ := ha
  rw [h1] at h1'
  have hpce := Except.ok.inj h1'
  simp only [Prod.mk.injEq] at hpce
  obtain ⟨rfl, rfl, rfl⟩ := hpce
  obtain ⟨-, c₂, ξ₂, r, i, _run, hin, -, -, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, hrun, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain ⟨held, ts₀, L, hheld, hts₀, hM⟩ := members_trees hrun
  obtain ⟨ls, hls, hl⟩ := linesOf_arr hlines
  obtain rfl : ts₀ = ts := Except.ok.inj ((Verity.Sampled.Binding.trees_of hts₀ hls hl).symm.trans hts)
  obtain ⟨named, hnamed, hall⟩ := named_mem hM
  obtain ⟨hmem, -, -⟩ := Verity.Sampled.MemberForm.membersTreeReason_none hnamed
  unfold secondRecords at hrecs
  obtain ⟨rf, hrf, hro⟩ := Verity.Sampled.Binding.mapM_mem hrecs rec hrec
  obtain ⟨st, hst, hrf⟩ := List.mem_flatMap.1 hrf
  obtain ⟨m, hm⟩ := mem_zip_left e.statements.toList held.toList (by simp [held_size hheld]) hst
  unfold Verity.Sampled.recordOn at hro
  obtain ⟨b, hb, hro⟩ := Verity.Sampled.Binding.obind_some hro
  obtain ⟨str, hstr, hro⟩ := Verity.Sampled.Binding.obind_some hro
  have hrec' : (Flock.JsonParse.parse str).toOption.getD .null = rec := Option.some.inj hro
  have hzip : (st, m) ∈ (e.statements.zip held).toList := by
    first
      | (rw [Array.toList_zip]; exact hm)
      | simpa using hm
  obtain ⟨tabs, ev, hx⟩ := hall (st, m) hzip rf hrf _ b str hb hstr
  obtain ⟨t, ht, -, o, -, -, hS⟩ := Verity.Sampled.MemberForm.memberTreeReason_none (hmem _ hx)
  have hS' : sessionTreeReason #[t.unclosed] tabs rec = none := by
    rw [← hrec']
    exact hS
  exact ⟨t, ht, tabs, hS'⟩

/-- **`Binds` where the device printed each play's log in its two sessions**: `binds_of_prints` with the interior's
records held by the form's own check (`interiorHeld_of_accepts`). -/
theorem binds_of_accepts (dev : Exchange → History) (θ : ByteArray) : BindsOfPrints dev θ :=
  binds_of_prints dev θ fun _ _ ha => interiorHeld_of_accepts ha

end Verity.SampledTwo
