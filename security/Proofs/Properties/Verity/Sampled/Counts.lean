module

public import Proofs.Properties.Verity.Sampled.Trees

/-! A `--zk` session's record counts the coins its tree's log shows opened. -/

@[expose] public section

namespace Verity.Sampled.Trees

open Lean (Json)
open Spot Spot.Challenger

def recordCount (record : Json) (stream : String) (k : Nat) : Option Nat :=
  if stream == TREE_SESSION && k == 0 then (Verity.Sampled.recordRound record stream k).map (·.size / 16)
  else
    match getD record "streams" with
    | .arr ss =>
      (ss.find? fun x => match getD x "stream" with | .str n => n == stream | _ => false).bind
        fun x => match getD x "rounds" with
          | .arr rs => rs[k]?.bind fun r => nat? (getD r "n")
          | _ => none
    | _ => none

def RoundCounted (t : LoggedTree) (name : String) (x : Json × Nat) : Prop :=
  ∃ cs n, recordCoins? (getD x.1 "coins") = some cs ∧ nat? (getD x.1 "n") = some n ∧
    ∃ o, t.opening name x.2 = some o ∧ o.n = n ∧ o.coins = sha512Hex cs

def StreamCounted (t : LoggedTree) (s : Json) : Prop :=
  ∀ name rounds, getD s "stream" = .str name → getD s "rounds" = .arr rounds →
    ∀ x ∈ rounds.zipIdx, RoundCounted t name x

theorem ret_last {α γ σ : Type} {P : α → Prop}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    {g : Option (Option γ) × σ → Id (Option γ)}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ P x))
    (hg : ∀ w u, g (some (some w), u) ≠ none) {xs : Array α} {u : σ}
    (h : (forIn (m := Id) xs (none, u) f >>= g) = none) : (∀ x ∈ xs, P x) ∧ ∃ u', g (none, u') = none := by
  rcases forIn_held_arr hf xs u with ⟨w, hw⟩ | ⟨hn, hall⟩
  · have e : forIn (m := Id) xs (none, u) f = (some (some w), Prod.snd (forIn (m := Id) xs (none, u) f)) :=
      Prod.ext hw rfl
    rw [e] at h
    exact absurd h (hg w _)
  · have e : forIn (m := Id) xs (none, u) f = (none, Prod.snd (forIn (m := Id) xs (none, u) f)) := Prod.ext hn rfl
    rw [e] at h
    exact ⟨hall, Prod.snd (forIn (m := Id) xs (none, u) f), h⟩

theorem session_counted_opened {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧
      (∀ (stream : String) (k : Nat) (cs : ByteArray), (stream == TREE_SESSION && k == 0) = true →
        Verity.Sampled.recordRound record stream k = some cs →
        ∃ o, t.opening stream k = some o ∧ o.n = cs.size / 16 ∧ o.coins = sha512Hex cs) ∧
      (∃ streams, getD record "streams" = .arr streams ∧ ∀ s ∈ streams, StreamCounted t s) ∧
      (t.opened.all fun o => o.round < Spot.Challenger.recordRounds record o.stream) = true := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold Verity.Sampled.recordTree
    rw [strIs hp]
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i ps hlink
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i rootF points streams htri
  rcases hlo : t.opening TREE_SESSION 0 with _ | o
  · simp only [hlo, Bool.false_eq_true, ↓reduceIte] at h
    all_goals (split at h <;> (try split at h) <;> cases h)
  simp only [hlo] at h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  rename_i hlinkOk
  simp only [Prod.mk.injEq] at htri hlink
  obtain ⟨-, hpts, hst⟩ := htri
  obtain ⟨-, -, hps⟩ := hlink
  have hloop := ret_last (P := StreamCounted t) ?_ (fun w _ hw => by cases hw) h
  · obtain ⟨hall, u', hg⟩ := hloop
    have hop : (t.opened.all fun o => o.round < Spot.Challenger.recordRounds record o.stream) = true := by
      try dsimp only at hg
      split at hg
      · assumption
      · cases hg
    refine ⟨fun stream k cs hsess hr => ?_, ⟨streams, hst, hall⟩, hop⟩
    unfold Verity.Sampled.recordRound at hr
    rw [ite_eq_left hsess] at hr
    simp only [Bool.and_eq_true, beq_iff_eq] at hsess
    obtain ⟨rfl, rfl⟩ := hsess
    rw [hps] at hr
    dsimp only at hr
    cases hr.symm.trans hpts
    simp only [Bool.and_eq_true, beq_iff_eq] at hlinkOk
    exact ⟨o, hlo, hlinkOk.1.1, hlinkOk.2⟩
  · intro s u res hres
    split at hres
    case h_2 => exact .inl ⟨_, _, hres.symm⟩
    rename_i name rounds hsr
    refine Or.imp id ?_ (step_held (P := RoundCounted t name) ?_ ?_ ?_ hres)
    · rintro ⟨u', hy, hrs⟩
      refine ⟨u', hy, fun name' rounds' hn hr' => ?_⟩
      simp only [Prod.mk.injEq] at hsr
      cases hn.symm.trans hsr.1
      cases hr'.symm.trans hsr.2
      exact hrs
    · intro x u res' hres'
      split at hres'
      case h_2 => exact .inl ⟨_, _, hres'.symm⟩
      rename_i coins m hcn
      rcases ho : t.opening name x.2 with _ | o
      · simp only [ho, Bool.false_eq_true, ↓reduceIte] at hres'
        all_goals (split at hres' <;> exact .inl ⟨_, _, hres'.symm⟩)
      simp only [ho] at hres'
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      rename_i hok
      refine .inr ⟨_, hres'.symm, ?_⟩
      simp only [Prod.mk.injEq] at hcn
      simp only [Bool.and_eq_true, beq_iff_eq] at hok
      exact ⟨coins, m, hcn.1, hcn.2, o, ho, hok.1.1, hok.2⟩
    · exact fun _ _ => ⟨(), rfl⟩
    · exact fun _ => ⟨(), rfl⟩

theorem session_counted {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧
      (∀ (stream : String) (k : Nat) (cs : ByteArray), (stream == TREE_SESSION && k == 0) = true →
        Verity.Sampled.recordRound record stream k = some cs →
        ∃ o, t.opening stream k = some o ∧ o.n = cs.size / 16 ∧ o.coins = sha512Hex cs) ∧
      ∃ streams, getD record "streams" = .arr streams ∧ ∀ s ∈ streams, StreamCounted t s := by
  obtain ⟨t, ht, hroot, hlink, hs, -⟩ := session_counted_opened h
  exact ⟨t, ht, hroot, hlink, hs⟩

theorem streams_counted {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧
      ∃ streams, getD record "streams" = .arr streams ∧ ∀ s ∈ streams, StreamCounted t s := by
  obtain ⟨t, ht, hroot, -, hs⟩ := session_counted h
  exact ⟨t, ht, hroot, hs⟩

theorem round_counted {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json}
    {stream : String} {k n : Nat} {cs : ByteArray} (h : sessionTreeReason trees tables record = none)
    (hr : Verity.Sampled.recordRound record stream k = some cs) (hn : recordCount record stream k = some n) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧
      ∃ o, t.opening stream k = some o ∧ o.n = n ∧ o.coins = sha512Hex cs := by
  obtain ⟨t, ht, hroot, hlink, streams, hst, hall⟩ := session_counted h
  refine ⟨t, ht, hroot, ?_⟩
  by_cases hsess : (stream == TREE_SESSION && k == 0) = true
  · unfold recordCount at hn
    rw [ite_eq_left hsess, hr] at hn
    have hcount : cs.size / 16 = n := by simpa using hn
    obtain ⟨o, ho, hon, hoc⟩ := hlink stream k cs hsess hr
    exact ⟨o, ho, hon.trans hcount, hoc⟩
  · unfold recordCount at hn
    rw [ite_eq_right hsess] at hn
    rw [hst] at hn
    dsimp only at hn
    unfold Verity.Sampled.recordRound at hr
    rw [ite_eq_right hsess] at hr
    rw [hst] at hr
    dsimp only at hr
    obtain ⟨x, hx, hr⟩ := Option.bind_eq_some_iff.1 hr
    obtain ⟨x', hx', hn⟩ := Option.bind_eq_some_iff.1 hn
    have hname := Array.find?_some hx
    split at hname
    case h_2 => cases hname
    rename_i nm hxn
    cases beq_iff_eq.1 hname
    have hname' := Array.find?_some hx'
    split at hname'
    case h_2 => cases hname'
    rename_i nm' hxn'
    cases beq_iff_eq.1 hname'
    split at hr
    case h_2 => cases hr
    rename_i rs hxr
    obtain ⟨r, hrk, hcs⟩ := Option.bind_eq_some_iff.1 hr
    split at hn
    case h_2 => cases hn
    rename_i rs' hxr'
    obtain ⟨r', hrk', hcn⟩ := Option.bind_eq_some_iff.1 hn
    obtain ⟨cs₁, _, hcs₁, -, o₁, ho₁, -, hc₁⟩ :=
      hall x (Array.mem_of_find?_eq_some hx) _ rs hxn hxr (r, k) (Array.mem_zipIdx_iff_getElem?.2 hrk)
    obtain ⟨_, n₂, -, hn₂, o₂, ho₂, hon₂, -⟩ :=
      hall x' (Array.mem_of_find?_eq_some hx') _ rs' hxn' hxr' (r', k) (Array.mem_zipIdx_iff_getElem?.2 hrk')
    have e1 : cs₁ = cs := Option.some.inj (hcs₁.symm.trans hcs)
    have e2 : n₂ = n := Option.some.inj (hn₂.symm.trans hcn)
    have e3 : o₂ = o₁ := Option.some.inj (ho₂.symm.trans ho₁)
    refine ⟨o₁, ho₁, ?_, ?_⟩
    · rw [← e3, ← e2]
      exact hon₂
    · rw [← e1]
      exact hc₁

end Verity.Sampled.Trees
