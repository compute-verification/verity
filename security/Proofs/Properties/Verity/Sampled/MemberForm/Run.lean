module

public import Proofs.Properties.Verity.Sampled.MemberForm.Accept

/-! The per-member form's run is its statements' sessions' verdicts, in order. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

structure HeldAfter (ms held : Array _root_.Spot.OneStageZk.Member) (st : _root_.Spot.OneStageZk.StatementFiles)
    (m : _root_.Spot.OneStageZk.Member) : Prop where
  named : st.member = some m.name
  member : ms.find? (·.name == m.name) = some m
  fresh : held.any (·.name == m.name) = false
  ordered : _root_.Spot.OneStageZk.follows held m = true
  one : st.sessions.size = 1

theorem heldStep_ok {ms held held' : Array _root_.Spot.OneStageZk.Member}
    {st : _root_.Spot.OneStageZk.StatementFiles} (h : _root_.Spot.OneStageZk.heldStep ms held st = .ok held') :
    ∃ m, HeldAfter ms held st m ∧ held' = held.push m := by
  unfold _root_.Spot.OneStageZk.heldStep at h
  cases hmem : st.member with
  | none => rw [hmem] at h; cases h
  | some name =>
    rw [hmem] at h
    dsimp only at h
    cases hfind : ms.find? (·.name == name) with
    | none => rw [hfind] at h; cases h
    | some m =>
      rw [hfind] at h
      dsimp only at h
      have hb : (m.name == name) = true :=
        Array.find?_some (p := fun x : _root_.Spot.OneStageZk.Member => x.name == name) hfind
      have hname : m.name = name := eq_of_beq hb
      subst hname
      cases hany : held.any (·.name == m.name) with
      | true => rw [hany] at h; cases h
      | false =>
        rw [hany] at h
        dsimp only at h
        cases hfol : _root_.Spot.OneStageZk.follows held m with
        | false => rw [hfol] at h; cases h
        | true =>
          rw [hfol] at h
          dsimp only at h
          cases hone : (st.sessions.size != 1) with
          | true => rw [hone] at h; cases h
          | false =>
            rw [hone] at h
            refine ⟨m, ⟨hmem, hfind, hany, hfol, ?_⟩, (Except.ok.inj h).symm⟩
            cases hd : decide (st.sessions.size = 1) with
            | true => exact of_decide_eq_true hd
            | false =>
              have hne : (st.sessions.size != 1) = true := bne_iff_ne.2 (of_decide_eq_false hd)
              rw [hone] at hne
              cases hne

def HeldFrom (ms : Array _root_.Spot.OneStageZk.Member) : Array _root_.Spot.OneStageZk.Member →
    List _root_.Spot.OneStageZk.StatementFiles → Array _root_.Spot.OneStageZk.Member → Prop
  | held, [], held' => held' = held
  | held, st :: sts, held' => ∃ m, HeldAfter ms held st m ∧ HeldFrom ms (held.push m) sts held'

theorem foldlM_heldStep {ms : Array _root_.Spot.OneStageZk.Member} :
    ∀ (sts : List _root_.Spot.OneStageZk.StatementFiles) {held held' : Array _root_.Spot.OneStageZk.Member},
      sts.foldlM (_root_.Spot.OneStageZk.heldStep ms) held = .ok held' → HeldFrom ms held sts held'
  | [], _, _, h => (Except.ok.inj h).symm
  | st :: sts, held, held', h => by
    rw [List.foldlM_cons] at h
    obtain ⟨held1, h1, h⟩ := ebind_ok h
    obtain ⟨m, hm, rfl⟩ := heldStep_ok h1
    exact ⟨m, hm, foldlM_heldStep sts h⟩

theorem membersHeld_ok {ms : Array _root_.Spot.OneStageZk.Member}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {held : Array _root_.Spot.OneStageZk.Member}
    (h : _root_.Spot.OneStageZk.membersHeld ms sts = .ok held) : HeldFrom ms #[] sts.toList held := by
  unfold _root_.Spot.OneStageZk.membersHeld at h
  rw [← Array.foldlM_toList] at h
  exact foldlM_heldStep sts.toList h

theorem heldFrom_new {ms : Array _root_.Spot.OneStageZk.Member} :
    ∀ (sts : List _root_.Spot.OneStageZk.StatementFiles) {held held' : Array _root_.Spot.OneStageZk.Member},
      HeldFrom ms held sts held' →
      ∃ new, held'.toList = held.toList ++ new ∧ sts.length = new.length ∧
        ∀ z ∈ sts.zip new, z.1.member = some z.2.name ∧ ms.find? (·.name == z.2.name) = some z.2 ∧
          z.1.sessions.size = 1
  | [], held, held', h => by
    obtain rfl : held' = held := h
    exact ⟨[], (List.append_nil _).symm, rfl, fun _ hz => (nomatch hz)⟩
  | st :: sts, held, held', h => by
    obtain ⟨m, hm, hrest⟩ := h
    obtain ⟨new, hnew, hlen, hall⟩ := heldFrom_new sts hrest
    refine ⟨m :: new, ?_, congrArg (· + 1) hlen, fun z hz => ?_⟩
    · rw [hnew, Array.toList_push, List.append_assoc]
      rfl
    · rw [List.zip_cons_cons, List.mem_cons] at hz
      rcases hz with rfl | hz
      · exact ⟨hm.named, hm.member, hm.one⟩
      · exact hall z hz

theorem membersHeld_zip {ms : Array _root_.Spot.OneStageZk.Member}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {held : Array _root_.Spot.OneStageZk.Member}
    (h : _root_.Spot.OneStageZk.membersHeld ms sts = .ok held) :
    sts.size = held.size ∧ ∀ z ∈ (sts.zip held).toList,
      z.1.member = some z.2.name ∧ ms.find? (·.name == z.2.name) = some z.2 ∧ z.1.sessions.size = 1 := by
  obtain ⟨new, hnew, hlen, hall⟩ := heldFrom_new _ (membersHeld_ok h)
  have hnew' : held.toList = new := hnew
  subst hnew'
  refine ⟨hlen, fun z hz => hall z ?_⟩
  rw [Array.toList_zip] at hz
  exact hz

def InOrder (a b : _root_.Spot.OneStageZk.Member) : Prop := a.name ≠ b.name ∧ a.first ≤ b.first

theorem heldAfter_inOrder {ms held : Array _root_.Spot.OneStageZk.Member}
    {st : _root_.Spot.OneStageZk.StatementFiles} {m : _root_.Spot.OneStageZk.Member}
    (hm : HeldAfter ms held st m) (hp : held.toList.Pairwise InOrder) :
    (held.push m).toList.Pairwise InOrder := by
  rw [Array.toList_push, List.pairwise_append]
  refine ⟨hp, List.pairwise_singleton _ _, fun a ha b hb => ?_⟩
  obtain rfl : b = m := List.mem_singleton.1 hb
  constructor
  · have hfresh := hm.fresh
    rw [← Array.any_toList, List.any_eq_false] at hfresh
    exact fun e => hfresh a ha (beq_iff_eq.2 e)
  · have hfol := hm.ordered
    unfold _root_.Spot.OneStageZk.follows at hfol
    cases hb : held.back? with
    | none =>
      rw [Array.back?_eq_none_iff] at hb
      subst hb
      cases ha
    | some prev =>
      rw [hb] at hfol
      dsimp only at hfol
      have hle : prev.first ≤ b.first := Nat.le_of_not_lt fun hlt => by
        rw [decide_eq_true hlt] at hfol
        cases hfol
      obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hb
      rw [Array.toList_push] at ha hp
      rcases List.mem_append.1 ha with ha | ha
      · exact Nat.le_trans ((List.pairwise_append.1 hp).2.2 a ha prev (List.mem_singleton.2 rfl)).2 hle
      · obtain rfl : a = prev := List.mem_singleton.1 ha
        exact hle

theorem heldFrom_inOrder {ms : Array _root_.Spot.OneStageZk.Member} :
    ∀ (sts : List _root_.Spot.OneStageZk.StatementFiles) {held held' : Array _root_.Spot.OneStageZk.Member},
      HeldFrom ms held sts held' → held.toList.Pairwise InOrder → held'.toList.Pairwise InOrder
  | [], held, held', h, hp => by
    obtain rfl : held' = held := h
    exact hp
  | st :: sts, held, held', h, hp => by
    obtain ⟨m, hm, hrest⟩ := h
    exact heldFrom_inOrder sts hrest (heldAfter_inOrder hm hp)

theorem membersHeld_inOrder {ms : Array _root_.Spot.OneStageZk.Member}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {held : Array _root_.Spot.OneStageZk.Member}
    (h : _root_.Spot.OneStageZk.membersHeld ms sts = .ok held) : held.toList.Pairwise InOrder :=
  heldFrom_inOrder _ (membersHeld_ok h) List.Pairwise.nil

theorem runFrom_ok {st : Flock.V _root_.Spot.OneStageZk.StepAt}
    {sessions : Array _root_.Spot.OneStageZk.RunFiles} {bs : _root_.Core.Blobs}
    {r r' : _root_.Spot.OneStageZk.Run} (h : _root_.Spot.OneStageZk.runFrom st sessions bs r = .ok r')
    (hall : r'.all = true) :
    r.all = true ∧ ∃ a, st = .ok a ∧ Verdicts bs r (sessions.toList.map fun s => (stepAt a, s)) r' := by
  unfold _root_.Spot.OneStageZk.runFrom at h
  obtain ⟨a, ha, h⟩ := ebind_ok h
  have hl : (sessions.toList.map fun s => (stepAt a, s)).foldlM (fun r x => x.1.step bs r x.2) r = .ok r' := by
    rw [List.foldlM_map, Array.foldlM_toList]
    exact h
  obtain ⟨hr, hv⟩ := (steps_iff _).1 ⟨hl, hall⟩
  exact ⟨hr, a, ha, hv⟩

inductive Sessions (st : _root_.Spot.OneStageZk.Member → Flock.V _root_.Spot.OneStageZk.StepAt) :
    List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member) →
      List (StepAt × _root_.Spot.OneStageZk.RunFiles) → Prop
  | nil : Sessions st [] []
  | cons {z : _root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member}
      {zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)}
      {a : _root_.Spot.OneStageZk.StepAt} {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} :
      st z.2 = .ok a → Sessions st zs l →
      Sessions st (z :: zs) ((z.1.sessions.toList.map fun s => (stepAt a, s)) ++ l)

theorem members_ok {st : _root_.Spot.OneStageZk.Member → Flock.V _root_.Spot.OneStageZk.StepAt}
    {bs : _root_.Core.Blobs} :
    ∀ (zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member))
      {r r' : _root_.Spot.OneStageZk.Run},
      zs.foldlM (_root_.Spot.OneStageZk.stepMember st bs) r = .ok r' → r'.all = true →
        r.all = true ∧ ∃ l, Sessions st zs l ∧ Verdicts bs r l r'
  | [], r, r', h, hall => by
    obtain rfl : r = r' := Except.ok.inj h
    exact ⟨hall, [], .nil, .nil r⟩
  | z :: zs, r, r', h, hall => by
    rw [List.foldlM_cons] at h
    obtain ⟨r1, h1, h⟩ := ebind_ok h
    obtain ⟨hall1, l, hs, hv⟩ := members_ok zs h hall
    obtain ⟨hr, a, ha, hv1⟩ :=
      runFrom_ok (_root_.Spot.OneStageZk.mapError_ok (x := _root_.Spot.OneStageZk.runFrom (st z.2) z.1.sessions bs r)
        h1) hall1
    exact ⟨hr, _, .cons ha hs, (verdicts_append _).2 ⟨r1, hv1, hv⟩⟩

def Stated (st : _root_.Spot.OneStageZk.Member → Flock.V _root_.Spot.OneStageZk.StepAt)
    (z : _root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)
    (x : StepAt × _root_.Spot.OneStageZk.RunFiles) : Prop :=
  ∃ a, st z.2 = .ok a ∧ x.1 = stepAt a ∧ z.1.sessions = #[x.2]

theorem stated_of_sessions {st : _root_.Spot.OneStageZk.Member → Flock.V _root_.Spot.OneStageZk.StepAt}
    {zs : List (_root_.Spot.OneStageZk.StatementFiles × _root_.Spot.OneStageZk.Member)}
    {l : List (StepAt × _root_.Spot.OneStageZk.RunFiles)} (h : Sessions st zs l) :
    (∀ z ∈ zs, z.1.sessions.size = 1) → List.Forall₂ (Stated st) zs l := by
  induction h with
  | nil => exact fun _ => .nil
  | @cons z zs a l ha _ ih =>
    intro hone
    obtain ⟨s, hs⟩ : ∃ s, z.1.sessions = #[s] := ⟨_, eq_single (hone z List.mem_cons_self)⟩
    rw [hs]
    exact .cons ⟨a, ha, rfl, hs⟩ (ih fun z' hz' => hone z' (List.mem_cons_of_mem _ hz'))

theorem runMembers_ok {slice : _root_.Spot.OneStageZk.Member → Json} {ms : Array _root_.Spot.OneStageZk.Member}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {want : Array ℕ} {bs : _root_.Core.Blobs}
    {r : _root_.Spot.OneStageZk.Run} (h : _root_.Spot.OneStageZk.runMembers slice ms sts want bs = .ok r)
    (hall : r.all = true) :
    ∃ held files l, _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      _root_.Spot.OneStageZk.membersFiles slice bs ms.toList = .ok files ∧
      List.Forall₂ (Stated (_root_.Spot.OneStageZk.statementAt files)) (sts.zip held).toList l ∧
      Verdicts bs ⟨#[], #[], #[], true⟩ l r ∧ _root_.Spot.OneStageZk.provedHeld r.proved want = .ok () := by
  unfold _root_.Spot.OneStageZk.runMembers at h
  obtain ⟨held, hheld, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
  obtain ⟨files, hfiles, h⟩ := ebind_ok h
  obtain ⟨r1, hfold, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hprov, h⟩ := ebind_ok h
  obtain rfl : r1 = r := Except.ok.inj h
  rw [hall] at hprov
  rw [← Array.foldlM_toList] at hfold
  obtain ⟨-, l, hs, hv⟩ := members_ok _ hfold hall
  exact ⟨held, files, l, hheld, hfiles,
    stated_of_sessions hs fun z hz => ((membersHeld_zip hheld).2 z hz).2.2, hv, hprov⟩

theorem membersRun_ok {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    {c : _root_.Spot.OneStage.Commitment} {e : _root_.Spot.OneStageZk.Evidence}
    {sts : Array _root_.Spot.OneStageZk.StatementFiles} {bs : _root_.Core.Blobs}
    {r : _root_.Spot.OneStageZk.Run} (h : _root_.Spot.OneStageZk.membersRun p ms c e sts bs = .ok r)
    (hall : r.all = true) :
    _root_.Spot.OneStageZk.membersStructure p ms c e sts bs = .ok () ∧
      ∃ roots, _root_.Spot.OneStageZk.membersRootsOf p ms c.record = .ok roots ∧
        ∃ held files l, _root_.Spot.OneStageZk.membersHeld ms sts = .ok held ∧
          _root_.Spot.OneStageZk.membersFiles (fun m => m.slice c.record roots) bs ms.toList = .ok files ∧
          List.Forall₂ (Stated (_root_.Spot.OneStageZk.statementAt files)) (sts.zip held).toList l ∧
          Verdicts bs ⟨#[], #[], #[], true⟩ l r ∧
          _root_.Spot.OneStageZk.provedHeld r.proved e.base.provedUnits = .ok () := by
  unfold _root_.Spot.OneStageZk.membersRun at h
  obtain ⟨⟨⟩, hst, h⟩ := ebind_ok h
  obtain ⟨roots, hroots, h⟩ := ebind_ok h
  obtain ⟨r0, hrun, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain rfl : r0 = r := Except.ok.inj h
  exact ⟨hst, roots, hroots, runMembers_ok hrun hall⟩

theorem membersFiles_lookup {slice : _root_.Spot.OneStageZk.Member → Json} {bs : _root_.Core.Blobs} :
    ∀ {ms : List _root_.Spot.OneStageZk.Member} {files : List (String × _root_.Spot.OneStageZk.StepAt)},
      _root_.Spot.OneStageZk.membersFiles slice bs ms = .ok files →
      ∀ {name : String} {m : _root_.Spot.OneStageZk.Member} {a : _root_.Spot.OneStageZk.StepAt},
        ms.find? (·.name == name) = some m → files.lookup name = some a → m.files (slice m) bs = .ok a
  | [], _, _, _, _, _, hfind, _ => nomatch hfind
  | m0 :: ms, files, h, name, m, a, hfind, hlook => by
    rw [_root_.Spot.OneStageZk.membersFiles] at h
    obtain ⟨a0, ha0, h⟩ := ebind_ok h
    obtain ⟨as, hrest, h⟩ := ebind_ok h
    obtain rfl : (m0.name, a0) :: as = files := Except.ok.inj h
    rw [List.find?_cons] at hfind
    rw [List.lookup_cons] at hlook
    by_cases hn : m0.name = name
    · have h1 : (m0.name == name) = true := beq_iff_eq.2 hn
      have h2 : (name == m0.name) = true := beq_iff_eq.2 hn.symm
      rw [h1] at hfind
      rw [h2] at hlook
      obtain rfl : m0 = m := Option.some.inj hfind
      obtain rfl : a0 = a := Option.some.inj hlook
      exact ha0
    · have h1 : (m0.name == name) = false := beq_eq_false_iff_ne.2 hn
      have h2 : (name == m0.name) = false := beq_eq_false_iff_ne.2 fun e => hn e.symm
      rw [h1] at hfind
      rw [h2] at hlook
      exact membersFiles_lookup hrest hfind hlook

theorem statementAt_files {slice : _root_.Spot.OneStageZk.Member → Json} {bs : _root_.Core.Blobs}
    {ms : Array _root_.Spot.OneStageZk.Member} {files : List (String × _root_.Spot.OneStageZk.StepAt)}
    (h : _root_.Spot.OneStageZk.membersFiles slice bs ms.toList = .ok files) {m : _root_.Spot.OneStageZk.Member}
    (hm : ms.find? (·.name == m.name) = some m) {a : _root_.Spot.OneStageZk.StepAt}
    (ha : _root_.Spot.OneStageZk.statementAt files m = .ok a) : m.files (slice m) bs = .ok a := by
  unfold _root_.Spot.OneStageZk.statementAt at ha
  cases hl : files.lookup m.name with
  | none => rw [hl] at ha; cases ha
  | some a' =>
    rw [hl] at ha
    obtain rfl : a' = a := Except.ok.inj ha
    exact membersFiles_lookup h (by rw [Array.find?_toList]; exact hm) hl

end Verity.Sampled.MemberForm
