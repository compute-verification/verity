import Proofs.Properties.Verity.SampledDrive
import Proofs.Properties.Verity.PlayDeviceLog
import Proofs.Properties.Verity.PlayDevice
import Proofs.Properties.Verity.AcceptRounds

/-!
# The play's releases as `Drive`'s blocks

`Canon.runs_at` proves `RunsAt` at the readers `Canon.readersAt`: off the break, where sampled proofs' verdict accepts,
the play's tagged releases are the draw's block, the link points', then one block per round of session 0.

The readers tag a release by its request, where the play's device serves it (`tagOfServed`): the registration's
challenger label (`drawLabel`) is the draw, session 0's link points' family the points, and a family of one of the
statement's table's two rep streams a round at the count its request carries (`repStreamOf`, `coinCountOf`). v2's
acceptance runs the round coins at the draw's use, then session 0's (`OneStageZk.Session0`), whose link points and
rounds are released after requests it rebuilds from the record (`OneStageZk.session0_uses`, `OneStageZk.reqsOf`). The
device releases each family's coins after its uses' roots, the draw's family before every later one, and session 0's
families one run per use (`Vocab.rounds_at_play`), so the served releases of the draw's and session 0's families are
the draw's run, the link points', then some of session 0's rounds' runs (`Challenger.runs_session0`), and those are the
tagged ones. A round's release reads its count from its request's bytes, which are the record's request unless the
two are a SHA-512 collision (`pairsAt`, counted by `colAt`).

What the readers need of the sizes is `SizesAt`: the draw opens one segment per release of the draw's coin, session 0
has at most `A.N` rounds, each at fewer than `2 ^ 64` coins, and its link points take one release per chunk of their
coin. Each release is 512 bits (`Vocab.rounds`).
-/

open Lean (Json)

namespace SampledProofs.Challenger

theorem mapM_except_getElem? {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = .ok ys → ∀ {i : Nat} {x : α}, l[i]? = some x →
      ∃ y, ys[i]? = some y ∧ f x = .ok y
  | [], _, _, _, _, hx => by cases hx
  | a :: l, ys, h, i, x, hx => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => rw [ha] at h; cases h
    | ok y =>
      rw [ha] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok zs =>
        rw [hl] at h
        cases h
        cases i with
        | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
          subst hx
          exact ⟨y, rfl, ha⟩
        | succ i =>
          obtain ⟨y', hy', hf⟩ := mapM_except_getElem? hl (i := i) (by simpa using hx)
          exact ⟨y', by simpa using hy', hf⟩

theorem zip_size {us : List Use} {ls : Array Json} {ξs : List ByteArray} {F : Use × Json → Except String ByteArray}
    (hsz : (ls.size == us.length) = true) (hm : (us.zip ls.toList).mapM F = .ok ξs)
    (hF : ∀ x r, F x = .ok r → r.size = x.1.coins * COIN_BYTES) {i : Nat} {u : Use} (hu : us[i]? = some u) :
    ∃ ξ, ξs[i]? = some ξ ∧ ξ.size = u.coins * COIN_BYTES := by
  have hi : i < us.length := (List.getElem?_eq_some_iff.1 hu).1
  have hl : i < ls.size := by
    have : ls.size = us.length := beq_iff_eq.1 hsz
    omega
  have hz : (us.zip ls.toList)[i]? = some (u, ls[i]) := by
    rw [List.getElem?_zip_eq_some]
    exact ⟨hu, by simp [hl]⟩
  obtain ⟨y, hy, hf⟩ := mapM_except_getElem? (f := F) hm hz
  exact ⟨y, hy, hF (u, ls[i]) y hf⟩

/-- **Each use's coins are its count of coins**: where `roundCoins` returns, its `i`th bytes are `us[i]`'s count of
coins. -/
theorem roundCoins_size {key : ChallengerKey} {w : String} {us : List Use} {order : List (String × String)}
    {groups : List (List String)} {coins : Json} {ξs : List ByteArray}
    (h : roundCoins key w us order groups coins = .ok ξs) {i : Nat} {u : Use} (hu : us[i]? = some u) :
    ∃ ξ, ξs[i]? = some ξ ∧ ξ.size = u.coins * COIN_BYTES := by
  unfold roundCoins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done) | skip
  all_goals
    cases h
    refine zip_size ‹_› ‹List.mapM _ _ = Except.ok _› (fun x r hf => ?_) hu
    split at hf
    · cases hf
    · split at hf
      · cases hf; exact beq_iff_eq.1 ‹_›
      · split at hf <;> first | (cases hf; done) | (cases (‹(throw _ : Except String _) = Except.ok _›))

/-- Lists that agree label by label have one length. -/
theorem length_of_filters {β : Type} (f : β → String) {xs ys : List β}
    (h : ∀ ℓ, xs.filter (f · == ℓ) = ys.filter (f · == ℓ)) : xs.length = ys.length := by
  have hp : (xs.map f).Perm (ys.map f) := by
    rw [List.perm_iff_count]
    intro a
    rw [List.count_eq_countP, List.count_eq_countP, List.countP_map, List.countP_map, List.countP_eq_length_filter,
      List.countP_eq_length_filter]
    have := congrArg List.length (h a)
    simpa [Function.comp_def] using this
  simpa using hp.length_eq

/-- **A family of a group that its runs never received has no roots**: where the log's families of `g` are the runs
`vs` of `g`'s uses with a coin (`Session.Runs`), a family of `g` the log never received has no use with a coin. -/
theorem roots_nil_of_runs {us vs : List Use} {g R : List String} {ℓ : String} (hg : g.contains ℓ = true)
    (hvs : ∀ ℓ, vs.filter (·.label == ℓ) = us.filter fun u => u.label == ℓ && g.contains ℓ && u.coins != 0)
    (hrun : R.filter g.contains = vs.flatMap fun u => List.replicate u.coins u.label) (hR : ℓ ∉ R) :
    Use.roots us ℓ = [] := by
  unfold Use.roots
  rw [List.flatMap_eq_nil_iff]
  intro u hu
  obtain ⟨huU, hul⟩ := List.mem_filter.1 hu
  have hl : u.label = ℓ := by simpa using hul
  rw [List.replicate_eq_nil_iff]
  refine Classical.byContradiction fun hc => ?_
  have hm : u ∈ vs := by
    have : u ∈ us.filter fun u => u.label == ℓ && g.contains ℓ && u.coins != 0 :=
      List.mem_filter.2 ⟨huU, by rw [hl, hg]; simpa using hc⟩
    rw [← hvs ℓ] at this
    exact (List.mem_filter.1 this).1
  have : ℓ ∈ R.filter g.contains := by
    rw [hrun]
    exact List.mem_flatMap.2 ⟨u, hm, by rw [← hl]; exact List.mem_replicate.2 ⟨hc, rfl⟩⟩
  exact hR (List.mem_filter.1 this).1

/-- **Session 0's runs**: where the draw's use `d` (family `label`) comes first, then session 0's link points `pts`
(family `P`) and rounds `rds0`, then later uses of other families; every root of the draw's family is received before
any of a later use's family, and the link points' before any other family of session 0's; session 0's families are
received as their uses' runs (`Session.Runs`), and each family's roots are its uses' (`Use.roots`): the releases of the
draw's and session 0's families, each its family and root, are the draw's run, the link points', then some of session
0's rounds' runs, each with a coin; and no later use is of the draw's family. -/
theorem runs_session0 {Z : Type} (lab : Z → String) (rt : Z → ByteArray) (S : List Z)
    (label P : String) (d pts : Use) (rds0 later : List Use)
    (hd : d.label = label) (hdc : d.coins ≠ 0) (hpl : pts.label = P) (hpc : pts.coins ≠ 0)
    (hrP : ∀ u ∈ rds0, u.label ≠ P)
    (hlat : ∀ u ∈ later, ∀ v ∈ pts :: rds0, u.label ≠ v.label)
    (hbefD : ∀ u ∈ pts :: (rds0 ++ later), ∀ i j : Nat, (S.map lab)[i]? = some label → (S.map lab)[j]? = some u.label →
      i < j)
    (hbefP : ∀ ℓ ∈ ((pts :: rds0).map (·.label)).eraseDups, ℓ ≠ P →
      ∀ i j : Nat, (S.map lab)[i]? = some P → (S.map lab)[j]? = some ℓ → i < j)
    (hruns : ∃ vs : List Use, (∀ ℓ, vs.filter (·.label == ℓ) = (d :: pts :: (rds0 ++ later)).filter fun u =>
        u.label == ℓ && ((pts :: rds0).map (·.label)).eraseDups.contains ℓ && u.coins != 0) ∧
      (S.map lab).filter ((pts :: rds0).map (·.label)).eraseDups.contains =
        vs.flatMap fun u => List.replicate u.coins u.label)
    (hroots : ∀ ℓ ∈ label :: ((pts :: rds0).map (·.label)).eraseDups,
      (S.filter (lab · == ℓ)).map rt = Use.roots (d :: pts :: (rds0 ++ later)) ℓ) :
    ∃ rUs : List Use, (S.filter fun z => (label :: ((pts :: rds0).map (·.label)).eraseDups).contains (lab z)).map
        (fun z => (lab z, rt z)) = (d :: pts :: rUs).flatMap (fun u => List.replicate u.coins (u.label, u.after)) ∧
      (∀ u ∈ rUs, u ∈ rds0 ∧ u.coins ≠ 0) ∧ rUs.length ≤ rds0.length ∧
      ∀ u ∈ pts :: (rds0 ++ later), u.label ≠ label := by
  generalize hF : ((pts :: rds0).map (·.label)).eraseDups = fams at hbefP hruns hroots ⊢
  have hfams : ∀ ℓ, ℓ ∈ fams ↔ ∃ v ∈ pts :: rds0, v.label = ℓ := by
    intro ℓ
    rw [← hF, List.mem_eraseDups, List.mem_map]
  obtain ⟨vs, hvs, hrun⟩ := hruns
  have hdroots : Use.roots (d :: pts :: (rds0 ++ later)) label ≠ [] := by
    unfold Use.roots
    obtain ⟨k, hk⟩ := Nat.exists_eq_succ_of_ne_zero hdc
    simp [List.filter_cons, hd, hk, List.replicate_succ]
  have hocc : ∃ i : Nat, (S.map lab)[i]? = some label := by
    have h := hroots label List.mem_cons_self
    rw [← h] at hdroots
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hdroots
    obtain ⟨z, hz, -⟩ := List.mem_map.1 hx
    obtain ⟨hzS, hzl⟩ := List.mem_filter.1 hz
    obtain ⟨i, hi, hiz⟩ := List.getElem_of_mem hzS
    refine ⟨i, ?_⟩
    rw [List.getElem?_map, List.getElem?_eq_getElem hi, hiz]
    simpa using hzl
  have hne : ∀ u ∈ pts :: (rds0 ++ later), u.label ≠ label := by
    intro u hu he
    obtain ⟨i, hi⟩ := hocc
    exact Nat.lt_irrefl i (hbefD u hu i i hi (by rw [he]; exact hi))
  have hlf : label ∉ fams := by
    intro hm
    obtain ⟨v, hv, hvl⟩ := (hfams label).1 hm
    refine hne v ?_ hvl
    rcases List.mem_cons.1 hv with rfl | hv
    · exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ (List.mem_append_left _ hv)
  have hsplit : S.filter (fun z => (label :: fams).contains (lab z)) =
      S.filter (fun z => lab z == label) ++ S.filter (fun z => fams.contains (lab z)) := by
    rw [← SampledProofs.Challenger.filter_or_of_before]
    · apply List.filter_congr
      intro z _
      by_cases h : lab z = label <;> simp [h]
    · intro z hz
      have : lab z = label := by simpa using hz
      rw [this]
      simpa using hlf
    · intro i j x y hx hy hpx hqy
      have hxl : lab x = label := by simpa using hpx
      obtain ⟨v, hv, hvl⟩ := (hfams (lab y)).1 (by simpa using hqy)
      refine hbefD v ?_ i j (by rw [List.getElem?_map, hx]; simp [hxl]) (by rw [List.getElem?_map, hy]; simp [hvl])
      rcases List.mem_cons.1 hv with rfl | hv
      · exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ (List.mem_append_left _ hv)
  have hdrun : (S.filter fun z => lab z == label).map (fun z => (lab z, rt z)) =
      List.replicate d.coins (d.label, d.after) := by
    have hr := hroots label List.mem_cons_self
    have hu : (d :: pts :: (rds0 ++ later)).filter (·.label == label) = [d] := by
      rw [List.filter_cons_of_pos (by simp [hd]), List.filter_eq_nil_iff.2]
      intro u hu hul
      exact hne u hu (by simpa using hul)
    rw [Use.roots, hu] at hr
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil] at hr
    have : (S.filter fun z => lab z == label).map (fun z => (lab z, rt z)) =
        ((S.filter fun z => lab z == label).map rt).map (label, ·) := by
      rw [List.map_map]
      apply List.map_congr_left
      intro z hz
      have : lab z = label := by simpa using (List.mem_filter.1 hz).2
      simp [this]
    rw [this, hr, List.map_replicate, hd]
  have hsrun : (S.filter fun z => fams.contains (lab z)).map (fun z => (lab z, rt z)) =
      vs.flatMap fun u => List.replicate u.coins (u.label, u.after) :=
    SampledProofs.Challenger.runs_blocks lab rt fams _ vs S hvs hrun
      (fun ℓ hℓ => hroots ℓ (List.mem_cons_of_mem _ hℓ))
  have hvsc : ∀ u ∈ vs, u.coins ≠ 0 := by
    intro u hu
    have : u ∈ vs.filter (·.label == u.label) := List.mem_filter.2 ⟨hu, by simp⟩
    rw [hvs] at this
    have := (List.mem_filter.1 this).2
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at this
    exact this.2
  have hPf : P ∈ fams := (hfams P).2 ⟨pts, List.mem_cons_self, hpl⟩
  have hPu : (d :: pts :: (rds0 ++ later)).filter (fun u => u.label == P && fams.contains P && u.coins != 0) =
      [pts] := by
    have hdP : d.label ≠ P := by
      rw [hd]; intro h; exact hlf (h ▸ hPf)
    rw [List.filter_cons_of_neg (by simp [hdP]), List.filter_cons_of_pos (by simp [hpl, hPf, hpc]),
      List.filter_eq_nil_iff.2]
    intro u hu hul
    simp only [Bool.and_eq_true, beq_iff_eq] at hul
    rcases List.mem_append.1 hu with hu | hu
    · exact hrP u hu hul.1.1
    · exact hlat u hu pts List.mem_cons_self (hul.1.1.trans hpl.symm)
  have hvsP : vs.filter (·.label == P) = [pts] := by rw [hvs, hPu]
  have hbef : SampledProofs.Challenger.Bef (vs.flatMap fun u => List.replicate u.coins u.label) (· = P) (· ≠ P) := by
    rw [← hrun]
    have h0 : SampledProofs.Challenger.Bef (S.map lab) (· = P) (fun y => y ≠ P ∧ fams.contains y = true) := by
      intro i j x y hx hy hpx hqy
      subst hpx
      exact hbefP y (by simpa using hqy.2) hqy.1 i j hx hy
    have h1 := SampledProofs.Challenger.bef_filter fams.contains _ h0
    intro i j x y hx hy hpx hqy
    refine h1 i j x y hx hy hpx ⟨hqy, ?_⟩
    exact (List.mem_filter.1 (List.mem_of_getElem? hy)).2
  obtain ⟨p, rest, rfl, hp⟩ := SampledProofs.Challenger.runs_head Use.label Use.coins P vs hvsc
    ⟨pts, by
      have : pts ∈ vs.filter (·.label == P) := by rw [hvsP]; exact List.mem_singleton_self _
      exact (List.mem_filter.1 this).1, hpl⟩ hbef
  have hpp : p = pts := by
    have : p ∈ (p :: rest).filter (·.label == P) := List.mem_filter.2 ⟨List.mem_cons_self, by simp [hp]⟩
    rw [hvsP] at this
    exact List.mem_singleton.1 this
  subst hpp
  have hrestP : ∀ u ∈ rest, u.label ≠ P := by
    intro u hu hul
    rw [List.filter_cons_of_pos (by simp [hp])] at hvsP
    have : u ∈ rest.filter (·.label == P) := List.mem_filter.2 ⟨hu, by simp [hul]⟩
    rw [(List.cons.inj hvsP).2] at this
    cases this
  have hmem : ∀ u ∈ rest, u ∈ rds0 ∧ u.coins ≠ 0 := by
    intro u hu
    refine ⟨?_, hvsc u (List.mem_cons_of_mem _ hu)⟩
    have : u ∈ (p :: rest).filter (·.label == u.label) := List.mem_filter.2 ⟨List.mem_cons_of_mem _ hu, by simp⟩
    rw [hvs] at this
    obtain ⟨hu', hc⟩ := List.mem_filter.1 this
    simp only [Bool.and_eq_true, beq_self_eq_true, true_and] at hc
    obtain ⟨v, hv, hvl⟩ := (hfams u.label).1 (by simpa using hc.1)
    rcases List.mem_cons.1 hu' with rfl | hu'
    · exact absurd (hd ▸ ((hfams _).2 ⟨v, hv, hvl⟩)) hlf
    rcases List.mem_cons.1 hu' with rfl | hu'
    · exact absurd hpl (hrestP _ hu)
    rcases List.mem_append.1 hu' with hu' | hu'
    · exact hu'
    · exact absurd hvl.symm (hlat u hu' v hv)
  refine ⟨rest, ?_, hmem, ?_, hne⟩
  · rw [hsplit, List.map_append, hdrun, hsrun]
    simp [List.flatMap_cons]
  · have hlen := length_of_filters Use.label (xs := p :: rest)
      (ys := (d :: p :: (rds0 ++ later)).filter fun u => fams.contains u.label && u.coins != 0) fun ℓ => by
        rw [hvs, List.filter_filter]
        apply List.filter_congr
        intro u _
        by_cases h : (u.label == ℓ) = true
        · rw [beq_iff_eq.1 h]; simp
        · simp [h]
    have hdo : (d :: p :: (rds0 ++ later)).filter (fun u => fams.contains u.label && u.coins != 0) =
        (p :: rds0).filter fun u => fams.contains u.label && u.coins != 0 := by
      rw [List.filter_cons_of_neg (by simp [hd, hlf]), List.filter_cons, List.filter_cons, List.filter_append,
        (List.filter_eq_nil_iff (l := later)).2, List.append_nil]
      intro u hu hc
      simp only [Bool.and_eq_true] at hc
      obtain ⟨v, hv, hvl⟩ := (hfams u.label).1 (by simpa using hc.1)
      exact hlat u hu v hv hvl.symm
    rw [hdo] at hlen
    have := List.length_filter_le (fun u => fams.contains u.label && u.coins != 0) (p :: rds0)
    simp only [List.length_cons] at hlen this
    omega

/-- A list whose first entries are `xs` is `xs` then the rest. -/
theorem prefix_of_getElem? {α : Type} : ∀ (xs us : List α), (∀ (j : Nat) x, xs[j]? = some x → us[j]? = some x) →
    us = xs ++ us.drop xs.length
  | [], us, _ => by simp
  | x :: xs, [], h => by have := h 0 x rfl; simp at this
  | x :: xs, u :: us, h => by
    have h0 := h 0 x rfl
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h0
    subst h0
    have := prefix_of_getElem? xs us (fun j y hy => by simpa using h (j + 1) y (by simpa using hy))
    simp only [List.cons_append, List.length_cons, List.drop_succ_cons, List.cons.injEq, true_and]
    exact this

theorem map_getD {A B C : Type} {f : A → Option C} {g : B → C} (c : C) :
    ∀ {xs : List A} {ys : List B}, xs.map f = ys.map (fun y => some (g y)) → xs.map (fun x => (f x).getD c) = ys.map g
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    rw [List.map_cons, List.map_cons, h.1, map_getD c h.2]
    rfl

theorem some_of_map {A B C : Type} {f : A → Option C} {g : B → C} {xs : List A} {ys : List B}
    (h : xs.map f = ys.map (fun y => some (g y))) {x : A} (hx : x ∈ xs) : ∃ c, f x = some c := by
  have : f x ∈ ys.map (fun y => some (g y)) := by rw [← h]; exact List.mem_map_of_mem hx
  obtain ⟨y, -, hy⟩ := List.mem_map.1 this
  exact ⟨g y, hy.symm⟩

theorem filter_map_eq {A B C : Type} (f : A → String × C) (g : B → String × C) :
    ∀ {xs : List A} {ys : List B}, xs.map f = ys.map g → ∀ ℓ,
      (xs.filter (fun a => (f a).1 == ℓ)).map (fun a => (f a).2) = (ys.filter (fun b => (g b).1 == ℓ)).map (fun b => (g b).2)
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: xs, b :: ys, h, ℓ => by
    simp only [List.map_cons, List.cons.injEq] at h
    have ih := filter_map_eq f g h.2 ℓ
    rw [List.filter_cons, List.filter_cons, h.1]
    split <;> simp [ih, h.1]

end SampledProofs.Challenger

namespace SampledProofs.OneStageZk

theorem hash64 (m : ByteArray) : (Flock.Sha512.hash m).size = 64 := by
  unfold Flock.Sha512.hash
  exact Array.size_ofFn

theorem TABLES_eq : TABLES = #["circuit"] := by decide

theorem STREAMS_eq : STREAMS = #["circuit/rep0", "circuit/rep1"] := by
  rw [STREAMS, TABLES_eq]
  rfl

/-- The requests a stream's rounds are released after, rebuilt from the record: each round's at its chain, from `a`,
and the count of the coins it lists. -/
def roundReqs (name : String) : List Json → ByteArray → List ByteArray
  | [], _ => []
  | r :: rs, a =>
    let a' := match getD r "msg_sha512" with
      | .str sha => match Flock.unhex sha with
        | .ok d => Flock.Sha512.hash (a ++ d)
        | .error _ => .empty
      | _ => .empty
    let n := match getD r "coins" with
      | .arr xs => xs.size
      | _ => 0
    roundRequest name a' n :: roundReqs name rs a'

/-- **The round requests a record's uses are released after**, rebuilt from the record: each stream's rounds' requests
(`roundRequest`), at their chain from the empty one and their coin counts. -/
def reqsOf (record : String) : List ByteArray :=
  match Flock.JsonParse.parse record with
  | .ok j =>
    match getD j "streams" with
    | .arr ss => ss.toList.flatMap fun s =>
      match getD s "stream", getD s "rounds" with
      | .str name, .arr rs => roundReqs name rs.toList .empty
      | _, _ => []
    | _ => []
  | .error _ => []

/-- A round's use, as its stream's steps make it: of the stream's family, released after the SHA-512 of the round's
request at a 64-byte chain value and its coin count, one coin per four F128s. -/
def RoundUse (slot : Nat) (name : String) (reqs : List ByteArray) (x : Challenger.Use × Array String) : Prop :=
  ∃ a : ByteArray, a.size = 64 ∧
    x.1 = ⟨Hello.slotStream slot name, Flock.Sha512.hash (roundRequest name a x.2.size), (x.2.size + 3) / 4⟩ ∧
    roundRequest name a x.2.size ∈ reqs

theorem roundSteps_mem {slot : Nat} {name : String} :
    ∀ {rs : List Json} {s st : List (Challenger.Use × Array String) × ByteArray},
      Steps (RoundStep slot name) rs s st →
        ∃ new, st.1 = s.1 ++ new ∧ ∀ x ∈ new, RoundUse slot name (roundReqs name rs s.2) x
  | _, _, _, .nil _ => ⟨[], by simp, by simp⟩
  | r :: rs, s, st, .cons (s₁ := s₁) hp t => by
    obtain ⟨sha, d, cs, w, hsha, hd, hcs, rfl⟩ := hp
    obtain ⟨xs, hxs, hfa⟩ := hexes_ok hcs
    have hsz : cs.size = xs.size := by
      have := hfa.length_eq
      simp only [Array.length_toList] at this
      exact this.symm
    have hhead : roundReqs name (r :: rs) s.2 =
        roundRequest name (Flock.Sha512.hash (s.2 ++ d)) cs.size :: roundReqs name rs (Flock.Sha512.hash (s.2 ++ d)) := by
      simp only [roundReqs, hsha, hd, hxs, hsz]
    obtain ⟨new, hst, hnew⟩ := roundSteps_mem t
    refine ⟨_ :: new, by rw [hst, List.append_assoc]; rfl, fun x hx => ?_⟩
    rw [hhead]
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨_, hash64 _, rfl, List.mem_cons_self⟩
    · obtain ⟨a, ha, hx1, hm⟩ := hnew x hx
      exact ⟨a, ha, hx1, List.mem_cons_of_mem _ hm⟩

theorem steps_mem {α σ : Type} {P : α → σ → σ → Prop} :
    ∀ {l : List α} {s s' : σ}, Steps P l s s' → ∀ x ∈ l, ∃ a b, P x a b
  | _, _, _, .nil _, _, hx => by cases hx
  | _, _, _, .cons hp t, x, hx => by
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨_, _, hp⟩
    · exact steps_mem t x hx

theorem streamSteps_mem {slot : Nat} :
    ∀ {ss : List Json} {out us : List (Challenger.Use × Array String)}, Steps (StreamStep slot) ss out us →
      ∃ new, us = out ++ new ∧ ∀ x ∈ new, ∃ s ∈ ss, ∃ name rs, getD s "stream" = .str name ∧
        getD s "rounds" = .arr rs ∧ RoundUse slot name (roundReqs name rs.toList .empty) x
  | _, _, _, .nil _ => ⟨[], by simp, by simp⟩
  | s :: ss, out, us, .cons hp t => by
    obtain ⟨name, rs, st, hname, hrs, hst, rfl⟩ := hp
    obtain ⟨n1, h1, q1⟩ := roundSteps_mem hst
    obtain ⟨n2, h2, q2⟩ := streamSteps_mem t
    refine ⟨n1 ++ n2, by rw [h2, h1, List.append_assoc], fun x hx => ?_⟩
    rcases List.mem_append.1 hx with hx | hx
    · exact ⟨s, List.mem_cons_self, name, rs, hname, hrs, q1 x hx⟩
    · obtain ⟨s', hs', rest⟩ := q2 x hx
      exact ⟨s', List.mem_cons_of_mem _ hs', rest⟩

/-- **A record `roundsOk` passes names only rep streams of the statement's table.** -/
theorem roundsOk_streams {record : String} (h : roundsOk record = .ok ()) :
    ∃ j ss, Flock.JsonParse.parse record = .ok j ∧ getD j "streams" = .arr ss ∧
      ∀ s ∈ ss.toList, ∀ name, getD s "stream" = .str name → STREAMS.contains name = true := by
  unfold roundsOk at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases (‹(throw _ : Except String _) = Except.ok _›); done) | skip
  rename_i _ j hj _ _ ss hss _ _ hfor
  refine ⟨j, ss, ?_, hss, ?_⟩
  · cases hp : Flock.JsonParse.parse record with
    | error e => rw [hp] at hj; cases hj
    | ok j' => rw [hp] at hj; cases hj; rfl
  rw [← Array.forIn_toList] at hfor
  have hs := steps_of_forIn (fun s _ _ => ∀ name, getD s "stream" = .str name → STREAMS.contains name = true)
    (fun x o r hr => by
      repeat' split at hr
      all_goals first
        | (cases hr; done)
        | (simp [throw, throwThe, MonadExceptOf.throw] at *; done)
        | skip
      all_goals (cases hr; exact ⟨_, rfl, fun name' hn' => by simp_all⟩)) _ _ _ hfor
  intro s hm name hn
  obtain ⟨_, _, hP⟩ := steps_mem hs s hm
  exact hP name hn

/-- **Session 0's uses**: its link points', then its rounds', each of one of the statement's table's two rep streams
(`RoundUse`), released after a request `reqsOf` rebuilds from the record. -/
theorem session0_uses {record : String} {us0 : List (Challenger.Use × Array String)}
    (h : sessionUses 0 record = .ok us0) :
    ∃ p rest, us0 = p :: rest ∧ p.1.label = Hello.slotStream 0 POINTS ∧
      ∀ x ∈ rest, ∃ name, (name = "circuit/rep0" ∨ name = "circuit/rep1") ∧ RoundUse 0 name (reqsOf record) x := by
  obtain ⟨hu, hok⟩ := usesOf_of_sessionUses h
  obtain ⟨j, ps, pts, a, ss, hj, -, -, -, hss, hs⟩ := usesOf_steps hu
  obtain ⟨j', ss', hj', hss', hst⟩ := roundsOk_streams hok
  rw [hj] at hj'
  cases hj'
  rw [hss] at hss'
  cases hss'
  obtain ⟨new, rfl, hnew⟩ := streamSteps_mem hs
  refine ⟨_, new, rfl, rfl, fun x hx => ?_⟩
  obtain ⟨s, hsm, name, rs, hname, hrs, hru⟩ := hnew x hx
  have hc := hst s hsm name hname
  rw [STREAMS_eq] at hc
  refine ⟨name, by simpa using hc, ?_⟩
  obtain ⟨a', ha', hx1, hm⟩ := hru
  refine ⟨a', ha', hx1, ?_⟩
  unfold reqsOf
  rw [hj]
  simp only [hss]
  exact List.mem_flatMap.2 ⟨s, hsm, by simp only [hname, hrs]; exact hm⟩

/-- A session's step keeps the draw before every later use: its uses' families follow the draw in `order`. -/
theorem order_step {label : String} {uses : Array Challenger.Use} {order : Array (String × String)}
    (h0 : uses.toList ≠ []) (h : ∀ u ∈ uses.toList.tail, (label, u.label) ∈ order.toList)
    (us : List (Challenger.Use × Array String)) (extra : Array (String × String)) :
    (uses ++ us.toArray.map (·.1)).toList ≠ [] ∧
      ∀ u ∈ (uses ++ us.toArray.map (·.1)).toList.tail,
        (label, u.label) ∈ (order ++ ((us.map (·.1.label)).eraseDups.map (label, ·)).toArray ++ extra).toList := by
  refine ⟨by simp [h0], fun u hu => ?_⟩
  rw [Array.toList_append, List.tail_append_of_ne_nil h0] at hu
  simp only [Array.toList_append, List.mem_append]
  rcases List.mem_append.1 hu with hu | hu
  · exact Or.inl (Or.inl (h u hu))
  · refine Or.inl (Or.inr ?_)
    simp only [Array.toList_map, List.mem_map] at hu
    obtain ⟨x, hx, rfl⟩ := hu
    simp only [List.toList_toArray, List.mem_map]
    exact ⟨x.1.label, List.mem_eraseDups.2 (List.mem_map_of_mem hx), rfl⟩

/-- **`roundsIn` reads the draw before every later use**: each use after the draw's is of a family `order` puts
after the draw's. -/
theorem roundsIn_order {label : String} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {uses : Array Challenger.Use} {wants : Array (Option (String × Array String))} {order : Array (String × String)}
    {groups : Array (List String)} {coins : Json}
    (h : roundsIn label c e bs = .ok (uses, wants, order, groups, coins)) :
    ∀ u ∈ uses.toList.tail, (label, u.label) ∈ order.toList := by
  unfold roundsIn at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · split at h
    · cases h
    · rename_i st hf
      cases h
      have hI := forIn_except_inv (fun st : Array Challenger.Use × Array (Option (String × Array String)) × Array Json ×
          Array (String × String) × Array (List String) =>
          st.1.toList ≠ [] ∧ ∀ u ∈ st.1.toList.tail, (label, u.label) ∈ st.2.2.2.1.toList)
        (fun x s r hs hr => by
          try simp only at hr
          repeat' split at hr
          all_goals first
            | (cases hr; done)
            | (cases hr; exact hs)
            | (cases hr; simp only [ForInStep.value]; exact order_step hs.1 hs.2 _ _))
        _ _ _ ⟨by simp, by simp⟩ hf
      exact hI.2
  · cases h

end SampledProofs.OneStageZk

namespace Verity

theorem get!_toList (x : ByteArray) (k : ℕ) : x.get! k = x.data.toList.getD k 0 := by
  cases x with
  | mk d =>
    show d[k]! = d.toList.getD k 0
    rw [List.getD_eq_getElem?_getD, Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?]
    simp
    rfl

theorem le64_nat (n : ℕ) (hn : n < 2 ^ 64) : (Flock.le64 n.toUInt64).data.toList.map UInt8.toNat =
    [n % 256, n / 2 ^ 8 % 256, n / 2 ^ 16 % 256, n / 2 ^ 24 % 256, n / 2 ^ 32 % 256, n / 2 ^ 40 % 256,
      n / 2 ^ 48 % 256, n / 2 ^ 56 % 256] := by
  simp [Flock.le64, List.range', ByteArray.push, ByteArray.emptyWithCapacity, UInt64.toNat_shiftRight,
    Nat.shiftRight_eq_div_pow, Nat.mod_eq_of_lt hn]
  rfl

theorem toNat_get! (x : ByteArray) (k : ℕ) : (x.get! k).toNat = (x.data.toList.map UInt8.toNat).getD k 0 := by
  rw [get!_toList, List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases x.data.toList[k]? <;> rfl

/-- **A round request carries its count**: `coinCountOf` reads back the count `roundRequest` writes, for a stream name
of fewer than `2 ^ 16` bytes, a 64-byte chain and a count below `2 ^ 64`. -/
set_option maxRecDepth 100000 in
theorem coinCountOf_roundRequest (s : String) (a : ByteArray) (n : ℕ) (hs : s.toUTF8.size < 65536)
    (ha : a.size = 64) (hn : n < 2 ^ 64) :
    coinCountOf (_root_.SampledProofs.OneStageZk.roundRequest s a n) = some n := by
  have hs' : s.utf8ByteSize < 65536 := by simpa using hs
  have hx : ((_root_.SampledProofs.OneStageZk.roundRequest s a n).data.toList.map UInt8.toNat) =
      ([s.toUTF8.size / 256, s.toUTF8.size % 256] ++ s.toUTF8.data.toList.map UInt8.toNat ++
        a.data.toList.map UInt8.toNat) ++ [n % 256, n / 2 ^ 8 % 256, n / 2 ^ 16 % 256, n / 2 ^ 24 % 256,
          n / 2 ^ 32 % 256, n / 2 ^ 40 % 256, n / 2 ^ 48 % 256, n / 2 ^ 56 % 256] := by
    rw [← le64_nat n hn]
    simp only [_root_.SampledProofs.OneStageZk.roundRequest, ByteArray.data_append, Array.toList_append,
      List.map_append]
    simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.append_assoc]
    congr 2
    all_goals (simp [UInt8.toNat_ofNat]; try omega)
  have hP : ([s.toUTF8.size / 256, s.toUTF8.size % 256] ++ s.toUTF8.data.toList.map UInt8.toNat ++
      a.data.toList.map UInt8.toNat).length = 2 + s.toUTF8.size + 64 := by
    simp only [List.length_append, List.length_map, Array.length_toList, ByteArray.size_data, List.length_cons,
      List.length_nil, ha]
    try omega
  have hsz : (_root_.SampledProofs.OneStageZk.roundRequest s a n).size = 2 + s.toUTF8.size + 72 := by
    have := congrArg List.length hx
    simp only [List.length_map, Array.length_toList, ByteArray.size_data, List.length_append, List.length_cons,
      List.length_nil] at this hP
    omega
  unfold coinCountOf
  simp only [toNat_get!, hx, hsz]
  have he : ∀ i, 2 + s.toUTF8.size + 72 - 8 + i = ([s.toUTF8.size / 256, s.toUTF8.size % 256] ++
      s.toUTF8.data.toList.map UInt8.toNat ++ a.data.toList.map UInt8.toNat).length + i := by
    intro i; rw [hP]; omega
  simp only [he]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left]
  simp only [List.append_assoc, List.cons_append, List.nil_append, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.getD_some]
  rw [if_neg (by omega), if_neg (by omega), show List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl]
  simp only [List.foldl_cons, List.foldl_nil, List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some,
    Nat.reducePow, Nat.reduceMul, Nat.reduceAdd, Nat.zero_add, Option.some.injEq]
  omega

end Verity

namespace Verity.Canon

open Accounting Core
open FlockSoundness.Discharge

local notation "𝔽" => FlockLevel3.GF128

variable {ι α : Type} [Fintype α] {d : Device ι}

/-! ## The play's requests -/

/-- The play's log at its releases `hs`. -/
def logOf (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) : List Release := hs.map (Game.recOf releaseOf)

/-- The play's releases the device `D` serves, each with the requests before it. -/
def servedReqs (D : PlayDevice) (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) :
    List (Σ m : ℕ, Game.LiveDrive.Req ByteArray × (Fin m → Bool)) :=
  (Game.LiveDrive.annot [] hs).filter fun z => servesNext D z.2.1.1 z.2.1.2

/-- A request's family and root, the empty ones where it has none. -/
def reqKey (b : ByteArray) : String × ByteArray := (PlayDevice.requestOf b).getD ("", .empty)

/-- A release's family and root, as its request names them. -/
def zKey (z : Σ m : ℕ, Game.LiveDrive.Req ByteArray × (Fin m → Bool)) : String × ByteArray := reqKey z.2.1.2

def zLab (z : Σ m : ℕ, Game.LiveDrive.Req ByteArray × (Fin m → Bool)) : String := (zKey z).1

def zRoot (z : Σ m : ℕ, Game.LiveDrive.Req ByteArray × (Fin m → Bool)) : ByteArray := (zKey z).2

/-- A served entry's family and root. -/
def eKey (e : String × ℕ × ByteArray × Release) : String × ByteArray := (e.1, e.2.2.1)

theorem served_keys (D : PlayDevice) (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) :
    (servedReqs D hs).map zKey = (D.served (logOf hs)).map eKey :=
  _root_.SampledProofs.Challenger.map_getD (f := fun z => PlayDevice.requestOf z.2.1.2) ("", .empty)
    (served_requests D hs)

theorem served_labels (D : PlayDevice) (hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))) :
    (servedReqs D hs).map zLab = (D.served (logOf hs)).map (·.1) := by
  have h := congrArg (List.map Prod.fst) (served_keys D hs)
  simp only [List.map_map] at h
  exact h

theorem tagged_served' (D : PlayDevice) (draw : String) (streamOf : String → Option (ℕ × Fin 2)) (m : ℕ)
    (hs : List (Σ m' : ℕ, ByteArray × (Fin m' → Bool))) :
    (Game.LiveDrive.annot [] hs).filter (Game.LiveDrive.Tagged m (tagOfServed D draw streamOf)) =
      (servedReqs D hs).filter fun z => decide (z.1 = m) && (tagOfPlay draw streamOf z.2.1.2).isSome :=
  tagged_served D draw streamOf m hs

theorem annot_mem {M : Type} : ∀ (h : List (ℕ × M)) (hs : List (Σ m : ℕ, M × (Fin m → Bool)))
    (z : Σ m : ℕ, Game.LiveDrive.Req M × (Fin m → Bool)), z ∈ Game.LiveDrive.annot h hs →
    ∃ w ∈ hs, z.1 = w.1 ∧ z.2.1.2 = w.2.1
  | _, [], _, hz => by simp [Game.LiveDrive.annot] at hz
  | h, ⟨m, b, x⟩ :: hs, z, hz => by
    simp only [Game.LiveDrive.annot, List.mem_cons] at hz
    rcases hz with rfl | hz
    · exact ⟨_, List.mem_cons_self, rfl, rfl⟩
    · obtain ⟨w, hw, h1, h2⟩ := annot_mem _ hs z hz
      exact ⟨w, List.mem_cons_of_mem _ hw, h1, h2⟩

theorem parts_of_req {b : ByteArray} {k : String × ByteArray} (h : PlayDevice.requestOf b = some k) :
    PlayDevice.requestParts b = some (k.1, bytesOfRequest b) ∧ Flock.Sha512.hash (bytesOfRequest b) = k.2 := by
  unfold PlayDevice.requestOf at h
  unfold bytesOfRequest
  cases hp : PlayDevice.requestParts b with
  | none => rw [hp] at h; cases h
  | some lx =>
    rw [hp] at h
    obtain ⟨l, x⟩ := lx
    cases h
    exact ⟨rfl, rfl⟩

/-- **A served release's request**: its family and bytes, whose SHA-512 is its root. -/
theorem served_parts {D : PlayDevice} {hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))}
    {z : Σ m : ℕ, Game.LiveDrive.Req ByteArray × (Fin m → Bool)} (hz : z ∈ servedReqs D hs) :
    PlayDevice.requestParts z.2.1.2 = some (zLab z, bytesOfRequest z.2.1.2) ∧
      Flock.Sha512.hash (bytesOfRequest z.2.1.2) = zRoot z := by
  obtain ⟨k, hk⟩ : ∃ k, PlayDevice.requestOf z.2.1.2 = some k :=
    _root_.SampledProofs.Challenger.some_of_map (served_requests D hs) hz
  have hkey : reqKey z.2.1.2 = k := by
    unfold reqKey
    rw [hk] <;> rfl
  have h := parts_of_req hk
  rw [← hkey] at h
  exact h

theorem served_family {D : PlayDevice} {rs : List Release} {x : String × ℕ × ByteArray × Release}
    (hx : x ∈ D.served rs) : x.1 ∈ D.families.map (·.1) := by
  obtain ⟨k, hk, hkx⟩ := List.getElem_of_mem hx
  have h : (D.families.find? (·.1 == x.1)).isSome = true :=
    (PlayDevice.served_cond D rs k x (by rw [List.getElem?_eq_getElem hk, hkx])).1
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.1 h
  have hfe : (f.1 == x.1) = true := List.find?_some hf
  exact List.mem_map.2 ⟨f, List.mem_of_find?_eq_some hf, beq_iff_eq.1 hfe⟩

/-- **A family released something only if it was served**: where the device's session releases bytes after a root,
it has a commitment and served an entry of that family. -/
theorem released_ne {D : PlayDevice} {rs : List Release} {l : String} {a : ByteArray}
    (h : ((D.session rs).released l a).size ≠ 0) :
    ∃ cm, (D.session rs).commitment = some cm ∧ ∃ x ∈ D.served rs, x.1 = l := by
  cases hc : (D.session rs).commitment with
  | none =>
    exfalso
    apply h
    unfold Properties.Challenger.Session.released
    rw [hc]
    rfl
  | some cm =>
    refine ⟨cm, rfl, ?_⟩
    obtain ⟨roots, hr, hf⟩ := PlayDevice.releasedRootsOf_session hc
    refine Classical.byContradiction fun hno => ?_
    have hnil : (D.served rs).filter (·.1 == l) = [] := by
      rw [List.filter_eq_nil_iff]
      intro x hx hxl
      exact hno ⟨x, hx, beq_iff_eq.1 hxl⟩
    have hl := hf l
    rw [hnil, List.map_nil] at hl
    apply h
    unfold Properties.Challenger.Session.released
    rw [hc]
    simp [hr, hl]

theorem splitBits_rest512 (K : ℕ) : (Game.splitBits 512 (by decide) K).rest.length = (K - 1) / 512 := by
  induction K using Nat.strong_induction_on with
  | _ K ih =>
    rw [Game.splitBits]
    split
    · dsimp only
      simp only [List.length_nil]
      omega
    · dsimp only
      simp only [List.length_cons]
      rw [ih (K - 512) (by omega)]
      omega

theorem splitRounds_len {k : ℕ} (hk : 1 ≤ k) : (splitRounds k).rest.length + 1 = (k + 3) / 4 := by
  show (Game.splitBits 512 (by decide) (k * 128)).rest.length + 1 = _
  rw [splitBits_rest512]
  omega

theorem contains_iff {l : List String} {a : String} : l.contains a = true ↔ a ∈ l := by
  simp

/-! ## How the readers tag a release -/

/-- Session 0's rep streams of the statement's table: `slot/0/circuit/rep0` and `slot/0/circuit/rep1`. -/
def repStreamOf (l : String) : Option (ℕ × Fin 2) :=
  if l = _root_.SampledProofs.Hello.slotStream 0 "circuit/rep0" then some (0, 0)
  else if l = _root_.SampledProofs.Hello.slotStream 0 "circuit/rep1" then some (0, 1) else none

/-- A rep stream's rep. -/
def streamRep (name : String) : Fin 2 := if name = "circuit/rep0" then 0 else 1

theorem repStream_slot {name : String} (h : name = "circuit/rep0" ∨ name = "circuit/rep1") :
    repStreamOf (_root_.SampledProofs.Hello.slotStream 0 name) = some (0, streamRep name) := by
  rcases h with rfl | rfl <;> decide

theorem repStreamOf_isSome {l : String} (h : (repStreamOf l).isSome = true) :
    ∃ name, (name = "circuit/rep0" ∨ name = "circuit/rep1") ∧ l = _root_.SampledProofs.Hello.slotStream 0 name := by
  unfold repStreamOf at h
  by_cases h0 : l = _root_.SampledProofs.Hello.slotStream 0 "circuit/rep0"
  · exact ⟨_, Or.inl rfl, h0⟩
  · by_cases h1 : l = _root_.SampledProofs.Hello.slotStream 0 "circuit/rep1"
    · exact ⟨_, Or.inr rfl, h1⟩
    · rw [if_neg h0, if_neg h1] at h
      cases h

theorem rep_ne_points {name : String} (h : name = "circuit/rep0" ∨ name = "circuit/rep1") :
    _root_.SampledProofs.Hello.slotStream 0 name ≠
      _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS := by
  rcases h with rfl | rfl <;> decide

theorem rep_utf8 {name : String} (h : name = "circuit/rep0" ∨ name = "circuit/rep1") :
    name.toUTF8.size < 65536 := by
  rcases h with rfl | rfl <;> decide

theorem tagOfPlay_parts {draw : String} {streamOf : String → Option (ℕ × Fin 2)} {b : ByteArray} {l : String}
    {x : ByteArray} (h : PlayDevice.requestParts b = some (l, x)) :
    tagOfPlay draw streamOf b = if l = draw then some .draw
      else if l = _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS then some .points
      else (roundOfRequest streamOf l x).map fun (t, r, n) => Game.LiveDrive.Tag.round t r n := by
  unfold tagOfPlay tagOfRequest
  rw [h] <;> rfl

theorem stream_of_round {streamOf : String → Option (ℕ × Fin 2)} {l : String} {x : ByteArray} {β : Type}
    {f : ℕ × Fin 2 × ℕ → β} (h : ((roundOfRequest streamOf l x).map f).isSome = true) :
    (streamOf l).isSome = true := by
  unfold roundOfRequest at h
  cases hs : streamOf l with
  | none => rw [hs] at h; simp at h
  | some _ => rfl

theorem round_of_slot {name : String} {a : ByteArray} {k : ℕ} (hn : name = "circuit/rep0" ∨ name = "circuit/rep1")
    (ha : a.size = 64) (hk : k < 2 ^ 64) :
    roundOfRequest repStreamOf (_root_.SampledProofs.Hello.slotStream 0 name)
      (_root_.SampledProofs.OneStageZk.roundRequest name a k) = some (0, streamRep name, k) := by
  simp [roundOfRequest, repStream_slot hn, coinCountOf_roundRequest name a k (rep_utf8 hn) ha hk]

/-! ## The registration's challenger label -/

/-- The challenger label the registered components' v2 entry names, as `registeredKey` reads its key. -/
def drawLabel (θ : ByteArray) : String :=
  ((canonicalJson θ >>= Core.Composition.decodeRegisteredParams).toOption.bind fun r =>
    (r.2.find? (·.id == _root_.SampledProofs.OneStageZk.PROTOCOL)).bind fun nm =>
      (Properties.Challenger.paramsOfNamed nm).toOption.bind fun p => p.base.challenger.map (·.2)).getD ""

/-- **Every accepting case names the registration's challenger label**, as `key_of_case` its key. -/
theorem label_of_case (V : Vocab d α) {θ : ByteArray} {o : Outcome} (hθ : o.params = θ) (hv : V.Verdict .sampled o)
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {e : _root_.SampledProofs.OneStageZk.Evidence} (hcase : SampledCase o.sent p c e)
    {key : _root_.SampledProofs.Challenger.ChallengerKey} {label : String}
    (hc : p.base.challenger = some (key, label)) : drawLabel θ = label := by
  obtain ⟨-, hra, -⟩ := hv
  obtain ⟨n, ps, b, dg, k, ev, hD, nm, cs, hx, hp, -, -⟩ := hcase
  obtain ⟨n', ps', b', d', k', ev', hD', _, _, hreg⟩ := hra
  have hps : ps' = ps := congrArg Prod.snd (Except.ok.inj (hD'.1.symm.trans hD.1))
  obtain ⟨hfit, -⟩ := hreg.2.2.2.2.2.2
  rw [hps] at hfit
  have hn : nm ∈ ps := (List.of_mem_zip (List.of_mem_zip hx).2).1
  have hid : nm.id = _root_.SampledProofs.OneStageZk.PROTOCOL := OnsiteIds.id_of_triples hfit.1 _ hx
  have hfind := OnsiteIds.find_of_unique hn hid (by rw [hfit.1]; exact OnsiteIds.onsite_count _)
  have hd : (canonicalJson θ >>= Core.Composition.decodeRegisteredParams) = .ok (n, ps) := by
    rw [← hθ]
    exact hD.1
  have hpn : Properties.Challenger.paramsOfNamed nm = .ok p := hp
  unfold drawLabel
  rw [hd]
  simp [Except.toOption, hfind, hpn, hc]

/-! ## The round requests sent, and the collision they would be -/

/-- The round requests session 0's record rebuilds, in one of the components' cases (`OneStageZk.reqsOf`). -/
def reqsCase (cs : Core.Composition.Case) (bs : Blobs) : List ByteArray :=
  match _root_.SampledProofs.OneStageZk.evidenceOf cs.evidence with
  | .ok e =>
    match e.sessions.toList with
    | s0 :: _ =>
      match bs.get s0.record s!"session {s0.name}'s record" with
      | .ok b =>
        match String.fromUTF8? b with
        | some record => _root_.SampledProofs.OneStageZk.reqsOf record
        | none => []
      | .error _ => []
    | [] => []
  | .error _ => []

/-- **The round requests in what was sent**: each of the components' cases' (`reqsCase`). -/
def reqsSent (s : Sent) : List ByteArray :=
  match canonicalJson s.params >>= Core.Composition.decodeRegisteredParams,
      Core.Composition.decodeRegisteredEvidence s.evidence with
  | .ok (_, ps), .ok (_, ev) =>
    (Properties.Composition.triples s.components ps ev).flatMap fun x => reqsCase x.2.2 (Blobs.ofArray s.blobs)
  | _, _ => []

theorem mem_reqsSent {s : Sent} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {e : _root_.SampledProofs.OneStageZk.Evidence}
    (hcase : SampledCase s p c e) {s0 : _root_.SampledProofs.OneStageZk.Session}
    {rest : List _root_.SampledProofs.OneStageZk.Session} (hl : e.sessions.toList = s0 :: rest) {b : ByteArray}
    {record : String} (hb : (Blobs.ofArray s.blobs).get s0.record s!"session {s0.name}'s record" = .ok b)
    (hrec : String.fromUTF8? b = some record) {r : ByteArray}
    (hr : r ∈ _root_.SampledProofs.OneStageZk.reqsOf record) : r ∈ reqsSent s := by
  obtain ⟨n, ps, b', dg, k, ev, hD, nm, cs, hx, -, -, he⟩ := hcase
  have hrc : r ∈ reqsCase cs (Blobs.ofArray s.blobs) := by
    simpa only [reqsCase, he, hl, hb, hrec] using hr
  simp only [reqsSent, hD.1, hD.2.2]
  exact List.mem_flatMap.2 ⟨_, hx, hrc⟩

/-- **The pairs a collision among the play's requests would be**: each release's request bytes with each round
request in what was sent. -/
def pairsAt (_θ : ByteArray) (o : Outcome) : List (ByteArray × ByteArray) :=
  (o.log.map fun r => bytesOfRequest r.answers).flatMap fun x => (reqsSent o.sent).map (x, ·)

/-- **Off the counted collision, a release's request bytes are the round request of the same SHA-512.** -/
theorem bytes_eq_req (V : Vocab d α) {θ F c : ByteArray} {hs : List (Σ m : ℕ, ByteArray × (Fin m → Bool))}
    {e : ByteArray × Array ByteArray}
    (hb : ¬ (ColEvent (V.colAt recsAt pairsAt θ) (outAt θ F c hs e) ∨ V.Forges .sampled θ (outAt θ F c hs e)))
    {w : Σ m : ℕ, ByteArray × (Fin m → Bool)} (hw : w ∈ hs) {req : ByteArray}
    (hreq : req ∈ reqsSent (outAt θ F c hs e).sent)
    (hh : Flock.Sha512.hash (bytesOfRequest w.2.1) = Flock.Sha512.hash req) : bytesOfRequest w.2.1 = req := by
  refine Classical.byContradiction fun hne => hb (Or.inl (V.pair_counted ?_ hne hh))
  show (bytesOfRequest w.2.1, req) ∈ pairsAt θ (outAt θ F c hs e)
  unfold pairsAt
  refine List.mem_flatMap.2 ⟨bytesOfRequest w.2.1, ?_, List.mem_map_of_mem hreq⟩
  exact List.mem_map.2 ⟨Game.recOf releaseOf w, List.mem_map_of_mem hw, rfl⟩

/-! ## The readers, and the releases as their blocks -/

section

variable {n : ℕ} {D : Flock.Desc} {hU : FlockSoundness.Built D.unroll.unit} {kd K₀ : ℕ} {desc : Claims.Msg}

/-- **The sizes the readers need**, off the break, wherever sampled proofs' verdict accepts: the draw's segment opens
one unit per release of the draw's coin, and session 0 has at most `A.N` rounds after its link points, each round at
fewer than `2 ^ 64` coins, and its link points one release per chunk of their coin. -/
def SizesAt (V : Vocab d α) (Brk : Outcome → Prop) (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) : Prop :=
  ∀ o, V.Verdict .sampled o → ¬ Brk o → ∀ p c' e pub prof, SampledCase o.sent p c' e →
    _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
    (∀ opened : Array Json,
      _root_.SampledProofs.getD (_root_.SampledProofs.getD e.base.coins "segment") "openings" = .arr opened →
        opened.size = (splitDraw A.Lb).rest.length + 1) ∧
    ∀ (s0 : _root_.SampledProofs.OneStageZk.Session) rest, e.sessions.toList = s0 :: rest →
      ∀ b record us0, (Core.Blobs.ofArray o.sent.blobs).get s0.record s!"session {s0.name}'s record" = .ok b →
        String.fromUTF8? b = some record → _root_.SampledProofs.OneStageZk.sessionUses 0 record = .ok us0 →
        us0.length ≤ A.N + 1 ∧ (∀ x ∈ us0, x.2.size < 2 ^ 64) ∧
          ∀ x ∈ us0.head?,
            x.1.coins = (splitPoints (Exec.mPtsOf D.unroll.g (kd / Composed.nTab A.I))).rest.length + 1

/-- **The readers at the play's device**: releases of 512 bits, tagged by their requests where the device serves them
(`tagOfServed`, at the challenger label `l`), the draw's, link points' and rounds' coins drawn a release at a time,
and a request's bytes and their SHA-512. -/
def readersAt (V : Vocab d α) (θ : ByteArray) (A : Flock.Guarantees.LiveAt n D hU kd K₀ desc) (R₀ : A.Reg)
    (l : String) : Readers A where
  R₀ := R₀
  m := 512
  tagOf := tagOfServed (V.device θ) l repStreamOf
  spΩ := splitDraw A.Lb
  spP := splitPoints _
  spR := splitRounds
  hOf z := Flock.Sha512.hash (bytesOfRequest z.2)
  bytesOf z := bytesOfRequest z.2

variable {A : Flock.Guarantees.LiveAt n D hU kd K₀ desc}

-- one proof that reads every served release's family, root and tag
set_option maxHeartbeats 1000000 in
/-- **The releases split into blocks at `readersAt`**, off the break, wherever sampled proofs' verdict accepts, at a
registration whose releases are 512 bits each and the sizes `SizesAt` reads: the challenger device runs v2's uses
(`RoundsByDevice`), so the device's log serves the draw's family, the link points' and session 0's rounds' as their
uses' runs; each release of a served request is tagged by it, a round's request off the collision being the one v2
rebuilds from the record. -/
theorem runs_at (V : Vocab d α) (hR : Properties.Challenger.RoundsByDevice)
    (hf : ∀ θ o, V.forger .sampled θ o = V.forgerAt recsAt θ o) {θ F c : ByteArray}
    (rounds_eq : ∀ k ∈ V.rounds θ, k = 512) (R₀ : A.Reg)
    (hSz : SizesAt V (fun o => ColEvent (V.colAt recsAt pairsAt θ) o ∨ V.Forges .sampled θ o) A) :
    RunsAt V θ F c (fun o => ColEvent (V.colAt recsAt pairsAt θ) o ∨ V.Forges .sampled θ o)
      (readersAt V θ A R₀ (drawLabel θ)) := by
  intro hs e hsz hv hb
  have hb' : ¬ (ColEvent (V.colAt recsAt pairsAt θ) (outAt θ F c hs e) ∨
      V.Forges .sampled θ (outAt θ F c hs e)) := hb
  obtain ⟨p, c', ev, pub, prof, hcase, hacc⟩ := componentAccepts_sampled hv.2.2
  obtain ⟨key, label, w, opened, uses, wants, order, groups, coins, ξs, hc, -, hop, hrc, hin, -, hS0⟩ :=
    _root_.SampledProofs.OneStageZk.roundsWhy_spec (_root_.SampledProofs.OneStageZk.roundsWhy_of_checkBlobs hacc)
  have hkey := key_of_case V (o := outAt θ F c hs e) rfl hv hcase hc
  have hpub : key.pub = (registeredKey θ).2 := by
    rw [hkey] <;> rfl
  rcases V.rounds_at_play hR (outAt θ F c hs e) hpub hrc with ⟨-, hrel, hroots, hbef, hruns⟩ | ⟨i, u, hu, hbrk⟩
  swap
  · exact (no_break V hf (o := outAt θ F c hs e) rfl hv hb' hcase hc hin ⟨i, u, by simpa using hu, hbrk⟩).elim
  have hlog : (outAt θ F c hs e).log = logOf hs := rfl
  rw [hlog] at hrel hroots hbef hruns
  have hlab := label_of_case V (o := outAt θ F c hs e) rfl hv hcase hc
  subst hlab
  obtain ⟨s0', rest, hl⟩ : ∃ s0 rest, ev.sessions.toList = s0 :: rest := by
    obtain ⟨s0, hs0⟩ := Properties.SampledProofs.OneStageZk.session_of_accept hacc
    rcases hl : ev.sessions.toList with _ | ⟨s, rest⟩
    · have h := Array.mem_toList_iff.2 hs0
      rw [hl] at h
      cases h
    · exact ⟨s, rest, rfl⟩
  obtain ⟨b0, record, us0, hb0, hrec, hus, -, hu0, -, hj, -, hordP, hgrp, hlater⟩ := hS0 s0' rest hl
  obtain ⟨hdn, hS⟩ := hSz _ hv hb p c' ev pub prof hcase hacc
  have hdn := hdn opened hop
  obtain ⟨hN, h64, hpn⟩ := hS s0' rest hl b0 record us0 hb0 hrec hus
  obtain ⟨pt, rest0, rfl, hptl, hrnd⟩ := _root_.SampledProofs.OneStageZk.session0_uses hus
  obtain ⟨du, hdl, hdcn, hu0⟩ : ∃ du : _root_.SampledProofs.Challenger.Use,
      du.label = drawLabel θ ∧ du.coins = opened.size ∧ uses[0]? = some du := ⟨_, rfl, rfl, hu0⟩
  have hpfx : ∀ (j : ℕ) x, (du :: (pt :: rest0).map Prod.fst)[j]? = some x → uses[j]? = some x := by
    intro j x hx
    cases j with
    | zero =>
      have hx' : du = x := by simpa using hx
      subst hx'
      exact hu0
    | succ j =>
      cases hy : (pt :: rest0)[j]? with
      | none =>
        rw [List.getElem?_cons_succ, List.getElem?_map, hy] at hx
        simp at hx
      | some y =>
        rw [List.getElem?_cons_succ, List.getElem?_map, hy] at hx
        have hx' : y.1 = x := by simpa using hx
        subst hx'
        exact (hj j y hy).1
  have hpre := _root_.SampledProofs.Challenger.prefix_of_getElem? _ _ hpfx
  have hlen : (du :: (pt :: rest0).map Prod.fst).length = (pt :: rest0).length + 1 := by simp
  rw [hlen] at hpre
  generalize hL : uses.drop ((pt :: rest0).length + 1) = L at hpre hlater
  have huses : uses = du :: pt.1 :: (rest0.map Prod.fst ++ L) := by
    rw [hpre]
    simp only [List.map_cons, List.cons_append]
  subst huses
  -- the device's session: its draw's family released, its served requests' families and roots
  have hne0 : (((V.device θ).session (logOf hs)).released du.label du.after).size ≠ 0 := by
    obtain ⟨ξ, hξ, hsz0⟩ := _root_.SampledProofs.Challenger.roundCoins_size hrc hu0
    have h1 : some ξ = some (((V.device θ).session (logOf hs)).released du.label du.after) :=
      hξ.symm.trans (hrel 0 du hu0)
    rw [← Option.some.inj h1, hsz0, hdcn, hdn]
    exact Nat.mul_ne_zero (Nat.succ_ne_zero _) (by decide)
  obtain ⟨cm, hcm, x0, hx0, hx0l⟩ := released_ne hne0
  have hlabfam : drawLabel θ ∈ (V.device θ).families.map (·.1) := by
    have h := served_family hx0
    rw [hx0l, hdl] at h
    exact h
  have hkeys := served_keys (V.device θ) hs
  have hlabs := served_labels (V.device θ) hs
  have hrecv : _root_.SampledProofs.Challenger.receivedOrder ((V.device θ).session (logOf hs)).log =
      (servedReqs (V.device θ) hs).map zLab := by
    rw [PlayDevice.receivedOrder_session]
    exact hlabs.symm
  have hbef' : ∀ ab ∈ order, ∀ i j : ℕ, ((servedReqs (V.device θ) hs).map zLab)[i]? = some ab.1 →
      ((servedReqs (V.device θ) hs).map zLab)[j]? = some ab.2 → i < j := by
    intro ab hab i j hi hj'
    exact hbef ab hab i j (by rw [hrecv]; exact hi) (by rw [hrecv]; exact hj')
  have hz512 : ∀ z ∈ servedReqs (V.device θ) hs, z.1 = 512 := by
    intro z hz
    obtain ⟨w', hw', h1, -⟩ := annot_mem [] hs z (List.mem_filter.1 hz).1
    rw [h1]
    exact rounds_eq _ (by rw [← hsz]; exact List.mem_map_of_mem hw')
  have hNoCol : ∀ z ∈ servedReqs (V.device θ) hs, ∀ req ∈ _root_.SampledProofs.OneStageZk.reqsOf record,
      Flock.Sha512.hash (bytesOfRequest z.2.1.2) = Flock.Sha512.hash req → bytesOfRequest z.2.1.2 = req := by
    intro z hz req hreq hh
    obtain ⟨w', hw', -, h2⟩ := annot_mem [] hs z (List.mem_filter.1 hz).1
    rw [h2] at hh ⊢
    exact bytes_eq_req V hb' hw' (mem_reqsSent hcase hl hb0 hrec hreq) hh
  have hzU : ∀ z ∈ servedReqs (V.device θ) hs, ∃ v ∈ du :: pt.1 :: (rest0.map Prod.fst ++ L),
      v.label = zLab z ∧ v.after = zRoot z ∧ v.coins ≠ 0 := by
    intro z hz
    have hm : zKey z ∈ ((V.device θ).served (logOf hs)).map eKey := by
      rw [← hkeys]
      exact List.mem_map_of_mem hz
    obtain ⟨x, hx, hxz⟩ := List.mem_map.1 hm
    have hr := hroots x.1 (served_family hx)
    rw [PlayDevice.rootsOf_session hcm] at hr
    have hxr : x.2.2.1 ∈ _root_.SampledProofs.Challenger.Use.roots (du :: pt.1 :: (rest0.map Prod.fst ++ L)) x.1 := by
      rw [← hr]
      exact List.mem_map_of_mem (List.mem_filter.2 ⟨hx, by simp⟩)
    unfold _root_.SampledProofs.Challenger.Use.roots at hxr
    obtain ⟨v, hv', hvr⟩ := List.mem_flatMap.1 hxr
    obtain ⟨hvU, hvl⟩ := List.mem_filter.1 hv'
    obtain ⟨hvc, hva⟩ := List.mem_replicate.1 hvr
    have hk1 : x.1 = zLab z := congrArg Prod.fst hxz
    have hk2 : x.2.2.1 = zRoot z := congrArg Prod.snd hxz
    refine ⟨v, hvU, ?_, ?_, hvc⟩
    · rw [← hk1]
      simpa using hvl
    · rw [← hk2]
      exact hva.symm
  -- session 0's uses: the link points', then rounds of the table's rep streams; later sessions' are of slots past 0
  have hRU : ∀ u ∈ rest0.map Prod.fst, ∃ (name : String) (a : ByteArray) (k : ℕ),
      (name = "circuit/rep0" ∨ name = "circuit/rep1") ∧ a.size = 64 ∧ k < 2 ^ 64 ∧
      u = (⟨_root_.SampledProofs.Hello.slotStream 0 name,
        Flock.Sha512.hash (_root_.SampledProofs.OneStageZk.roundRequest name a k), (k + 3) / 4⟩ :
          _root_.SampledProofs.Challenger.Use) ∧
      _root_.SampledProofs.OneStageZk.roundRequest name a k ∈ _root_.SampledProofs.OneStageZk.reqsOf record := by
    intro u hu
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hu
    obtain ⟨name, hn, a, ha, hx1, hm⟩ := hrnd x hx
    exact ⟨name, a, x.2.size, hn, ha, h64 x (List.mem_cons_of_mem _ hx), hx1, hm⟩
  have hrP : ∀ u ∈ rest0.map Prod.fst,
      u.label ≠ _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS := by
    intro u hu
    obtain ⟨name, a, k, hn, -, -, rfl, -⟩ := hRU u hu
    exact rep_ne_points hn
  have hlat : ∀ u ∈ L, ∀ v ∈ pt.1 :: rest0.map Prod.fst, u.label ≠ v.label := by
    intro u hu v hv h
    obtain ⟨i, nm, hi, hul⟩ := hlater u hu
    rcases List.mem_cons.1 hv with rfl | hv
    · rw [hul, hptl] at h
      exact _root_.SampledProofs.Hello.slotStream_ne (by omega) h
    · obtain ⟨name, a, k, hn, -, -, rfl, -⟩ := hRU v hv
      rw [hul] at h
      exact _root_.SampledProofs.Hello.slotStream_ne (by omega) h
  have hF0 : (pt :: rest0).map (·.1.label) = (pt.1 :: rest0.map Prod.fst).map (·.label) := by
    simp [Function.comp_def]
  have hbefD : ∀ u ∈ pt.1 :: (rest0.map Prod.fst ++ L), ∀ i j : ℕ,
      ((servedReqs (V.device θ) hs).map zLab)[i]? = some (drawLabel θ) →
      ((servedReqs (V.device θ) hs).map zLab)[j]? = some u.label → i < j := by
    intro u hu i j hi hj'
    have hu' : u ∈ (List.toArray (du :: pt.1 :: (rest0.map Prod.fst ++ L))).toList.tail := by
      rw [List.toList_toArray]
      exact hu
    have hord := _root_.SampledProofs.OneStageZk.roundsIn_order hin u hu'
    rw [List.toList_toArray] at hord
    exact hbef' _ hord i j hi hj'
  have hbefP : ∀ ℓ ∈ ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups,
      ℓ ≠ _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS → ∀ i j : ℕ,
      ((servedReqs (V.device θ) hs).map zLab)[i]? =
        some (_root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS) →
      ((servedReqs (V.device θ) hs).map zLab)[j]? = some ℓ → i < j := by
    intro ℓ hℓ hne i j hi hj'
    have hℓ' : ℓ ∈ ((pt :: rest0).map (·.1.label)).eraseDups := by
      rw [hF0]
      exact hℓ
    exact hbef' _ (hordP ℓ hℓ' hne) i j hi hj'
  have hruns' : ∃ vs : List _root_.SampledProofs.Challenger.Use,
      (∀ ℓ, vs.filter (·.label == ℓ) = (du :: pt.1 :: (rest0.map Prod.fst ++ L)).filter fun u =>
        u.label == ℓ && ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups.contains ℓ && u.coins != 0) ∧
      ((servedReqs (V.device θ) hs).map zLab).filter ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups.contains =
        vs.flatMap fun u => List.replicate u.coins u.label := by
    obtain ⟨vs, hvs, hrun⟩ := hruns _ hgrp
    refine ⟨vs, fun ℓ => ?_, ?_⟩
    · rw [hvs ℓ, hF0]
    · rw [← hrecv, ← hF0]
      exact hrun
  have hroots' : ∀ ℓ ∈ drawLabel θ :: ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups,
      ((servedReqs (V.device θ) hs).filter (zLab · == ℓ)).map zRoot =
        _root_.SampledProofs.Challenger.Use.roots (du :: pt.1 :: (rest0.map Prod.fst ++ L)) ℓ := by
    intro ℓ hℓ
    have hfm := _root_.SampledProofs.Challenger.filter_map_eq zKey eKey hkeys ℓ
    by_cases hfam : ℓ ∈ (V.device θ).families.map (·.1)
    · have hr := hroots ℓ hfam
      rw [PlayDevice.rootsOf_session hcm] at hr
      rw [← hr]
      exact hfm
    · have hnil : ((V.device θ).served (logOf hs)).filter (fun b => (eKey b).1 == ℓ) = [] := by
        rw [List.filter_eq_nil_iff]
        intro x hx hxl
        exact hfam (by rw [← (beq_iff_eq.1 hxl : x.1 = ℓ)]; exact served_family hx)
      have hfm' : ((servedReqs (V.device θ) hs).filter (zLab · == ℓ)).map zRoot = [] := by
        have h := hfm
        rw [hnil, List.map_nil] at h
        exact h
      rw [hfm']
      have hℓf : ℓ ∈ ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups := by
        refine (List.mem_cons.1 hℓ).resolve_left fun h => hfam ?_
        rw [h]
        exact hlabfam
      obtain ⟨vs, hvs, hrun⟩ := hruns'
      refine (_root_.SampledProofs.Challenger.roots_nil_of_runs (contains_iff.2 hℓf) hvs hrun fun hm => ?_).symm
      rw [hlabs] at hm
      obtain ⟨x, hx, hxl⟩ := List.mem_map.1 hm
      exact hfam (by rw [← hxl]; exact served_family hx)
  have hdc : du.coins ≠ 0 := by
    rw [hdcn, hdn]
    exact Nat.succ_ne_zero _
  have hpn' := hpn pt (by simp)
  have hpc : pt.1.coins ≠ 0 := by
    rw [hpn']
    exact Nat.succ_ne_zero _
  obtain ⟨rUs, hT, hrUs, hlen, hneL⟩ := _root_.SampledProofs.Challenger.runs_session0 zLab zRoot
    (servedReqs (V.device θ) hs) (drawLabel θ)
    (_root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS) du pt.1 (rest0.map Prod.fst) L
    hdl hdc hptl hpc hrP hlat hbefD hbefP hruns' hroots'
  have hPl : _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS ≠ drawLabel θ := by
    rw [← hptl]
    exact hneL _ List.mem_cons_self
  have hPmem : _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS ∈
      ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups :=
    List.mem_eraseDups.2 (List.mem_map.2 ⟨pt.1, List.mem_cons_self, hptl⟩)
  generalize hS' : (servedReqs (V.device θ) hs).filter (fun z =>
    (drawLabel θ :: ((pt.1 :: rest0.map Prod.fst).map (·.label)).eraseDups).contains (zLab z)) = S' at hT
  have hS'sub : ∀ z ∈ S', z ∈ servedReqs (V.device θ) hs := by
    intro z hz
    rw [← hS'] at hz
    exact (List.mem_filter.1 hz).1
  -- the tags: a round's release, off the collision, is tagged by the round request v2 rebuilds
  have hTagR : ∀ z ∈ servedReqs (V.device θ) hs, ∀ (name : String) (a : ByteArray) (k : ℕ),
      (name = "circuit/rep0" ∨ name = "circuit/rep1") → a.size = 64 → k < 2 ^ 64 →
      _root_.SampledProofs.OneStageZk.roundRequest name a k ∈ _root_.SampledProofs.OneStageZk.reqsOf record →
      zLab z = _root_.SampledProofs.Hello.slotStream 0 name →
      zRoot z = Flock.Sha512.hash (_root_.SampledProofs.OneStageZk.roundRequest name a k) →
      _root_.SampledProofs.Hello.slotStream 0 name ≠ drawLabel θ →
      tagOfPlay (drawLabel θ) repStreamOf z.2.1.2 = some (.round 0 (streamRep name) k) := by
    intro z hz name a k hn ha hk hm hzl hzr hne
    obtain ⟨hpart, hh⟩ := served_parts hz
    have hbytes : bytesOfRequest z.2.1.2 = _root_.SampledProofs.OneStageZk.roundRequest name a k :=
      hNoCol z hz _ hm (by rw [hh, hzr])
    rw [tagOfPlay_parts hpart, hzl, if_neg hne, if_neg (rep_ne_points hn), hbytes, round_of_slot hn ha hk] <;> rfl
  have hfilter : (Game.LiveDrive.annot [] hs).filter
      (Game.LiveDrive.Tagged 512 (tagOfServed (V.device θ) (drawLabel θ) repStreamOf)) = S' := by
    rw [tagged_served', ← hS']
    apply List.filter_congr
    intro z hz
    dsimp only
    rw [decide_eq_true (hz512 z hz), Bool.true_and]
    obtain ⟨hpart, -⟩ := served_parts hz
    apply Bool.eq_iff_iff.2
    rw [contains_iff]
    constructor
    · intro hsome
      obtain ⟨v, hv, hvl, -, -⟩ := hzU z hz
      rcases List.mem_cons.1 hv with rfl | hv
      · exact List.mem_cons.2 (Or.inl (hvl.symm.trans hdl))
      rcases List.mem_cons.1 hv with rfl | hv
      · refine List.mem_cons_of_mem _ ?_
        rw [← hvl]
        exact List.mem_eraseDups.2 (List.mem_map.2 ⟨_, List.mem_cons_self, rfl⟩)
      rcases List.mem_append.1 hv with hv | hv
      · refine List.mem_cons_of_mem _ ?_
        rw [← hvl]
        exact List.mem_eraseDups.2 (List.mem_map.2 ⟨v, List.mem_cons_of_mem _ hv, rfl⟩)
      · exfalso
        obtain ⟨i, nm, hi, hvl'⟩ := hlater v hv
        have h1 : zLab z ≠ drawLabel θ := fun h =>
          hneL v (List.mem_cons_of_mem _ (List.mem_append_right _ hv)) (hvl.trans h)
        have h2 : zLab z ≠ _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS := fun h =>
          hlat v hv pt.1 List.mem_cons_self (hvl.trans (h.trans hptl.symm))
        rw [tagOfPlay_parts hpart, if_neg h1, if_neg h2] at hsome
        obtain ⟨name, -, hzl⟩ := repStreamOf_isSome (stream_of_round hsome)
        exact _root_.SampledProofs.Hello.slotStream_ne (by omega) (hvl'.symm.trans (hvl.trans hzl))
    · intro hm
      obtain ⟨v, hv, hvl, hvr, -⟩ := hzU z hz
      rcases List.mem_cons.1 hv with rfl | hv
      · rw [tagOfPlay_parts hpart, if_pos (hvl.symm.trans hdl)] <;> rfl
      rcases List.mem_cons.1 hv with rfl | hv
      · have h2 : zLab z = _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS :=
          hvl.symm.trans hptl
        rw [tagOfPlay_parts hpart, if_neg (by rw [h2]; exact hPl), if_pos h2] <;> rfl
      rcases List.mem_append.1 hv with hv | hv
      · obtain ⟨name, a, k, hn, ha, hk, rfl, hmr⟩ := hRU v hv
        rw [hTagR z hz name a k hn ha hk hmr hvl.symm hvr.symm
          (hneL _ (List.mem_cons_of_mem _ (List.mem_append_left _ hv)))] <;> rfl
      · exfalso
        rcases List.mem_cons.1 hm with h | h
        · exact hneL v (List.mem_cons_of_mem _ (List.mem_append_right _ hv)) (hvl.trans h)
        · obtain ⟨u, hu, hul⟩ := List.mem_map.1 (List.mem_eraseDups.1 h)
          exact hlat v hv u hu (hvl.trans hul.symm)
  have hTagK : ∀ z ∈ S', ∃ t, ∀ z' ∈ servedReqs (V.device θ) hs, zKey z' = zKey z →
      tagOfPlay (drawLabel θ) repStreamOf z'.2.1.2 = some t := by
    intro z hz
    have hmem : (zLab z, zRoot z) ∈ (du :: pt.1 :: rUs).flatMap
        (fun u => List.replicate u.coins (u.label, u.after)) := by
      rw [← hT]
      exact List.mem_map_of_mem hz
    obtain ⟨v, hv, hvr⟩ := List.mem_flatMap.1 hmem
    have hva := (List.mem_replicate.1 hvr).2
    have hvl : zLab z = v.label := congrArg Prod.fst hva
    have hvf : zRoot z = v.after := congrArg Prod.snd hva
    rcases List.mem_cons.1 hv with rfl | hv
    · refine ⟨.draw, fun z' hz' hk => ?_⟩
      obtain ⟨hpart, -⟩ := served_parts hz'
      have h1 : zLab z' = drawLabel θ := ((show zLab z' = zLab z from congrArg Prod.fst hk).trans hvl).trans hdl
      rw [tagOfPlay_parts hpart, h1, if_pos rfl]
    rcases List.mem_cons.1 hv with rfl | hv
    · refine ⟨.points, fun z' hz' hk => ?_⟩
      obtain ⟨hpart, -⟩ := served_parts hz'
      have h1 : zLab z' = _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS :=
        ((show zLab z' = zLab z from congrArg Prod.fst hk).trans hvl).trans hptl
      rw [tagOfPlay_parts hpart, h1, if_neg hPl, if_pos rfl]
    · obtain ⟨hvR, -⟩ := hrUs v hv
      obtain ⟨name, a, k, hn, ha, hk, rfl, hm⟩ := hRU v hvR
      exact ⟨.round 0 (streamRep name) k, fun z' hz' hkz => hTagR z' hz' name a k hn ha hk hm
        ((show zLab z' = zLab z from congrArg Prod.fst hkz).trans hvl)
        ((show zRoot z' = zRoot z from congrArg Prod.snd hkz).trans hvf)
        (hneL _ (List.mem_cons_of_mem _ (List.mem_append_left _ hvR)))⟩
  obtain ⟨tk, htkdef⟩ : ∃ tk : String × ByteArray → Game.LiveDrive.Tag,
      ∀ k (h : ∃ z ∈ servedReqs (V.device θ) hs, zKey z = k),
        tk k = (tagOfPlay (drawLabel θ) repStreamOf (Classical.choose h).2.1.2).getD .draw :=
    ⟨fun k => @dite _ (∃ z ∈ servedReqs (V.device θ) hs, zKey z = k) (Classical.propDecidable _)
      (fun h => (tagOfPlay (drawLabel θ) repStreamOf (Classical.choose h).2.1.2).getD .draw) fun _ => .draw,
      fun k h => dif_pos h⟩
  have htk : ∀ z ∈ servedReqs (V.device θ) hs, ∀ t,
      (∀ z' ∈ servedReqs (V.device θ) hs, zKey z' = zKey z → tagOfPlay (drawLabel θ) repStreamOf z'.2.1.2 = some t) →
      tk (zKey z) = t := by
    intro z hz t ht
    have hex : ∃ z' ∈ servedReqs (V.device θ) hs, zKey z' = zKey z := ⟨z, hz, rfl⟩
    rw [htkdef _ hex, ht _ (Classical.choose_spec hex).1 (Classical.choose_spec hex).2] <;> rfl
  have htkv : ∀ v ∈ du :: pt.1 :: rUs, v.coins ≠ 0 → ∀ t,
      (∀ z' ∈ servedReqs (V.device θ) hs, zKey z' = (v.label, v.after) →
        tagOfPlay (drawLabel θ) repStreamOf z'.2.1.2 = some t) → tk (v.label, v.after) = t := by
    intro v hv hc t ht
    have hmem : (v.label, v.after) ∈ S'.map (fun z => (zLab z, zRoot z)) := by
      rw [hT]
      exact List.mem_flatMap.2 ⟨v, hv, List.mem_replicate.2 ⟨hc, rfl⟩⟩
    obtain ⟨z, hz, hzk⟩ := List.mem_map.1 hmem
    have hzk' : zKey z = (v.label, v.after) := hzk
    rw [← hzk']
    exact htk z (hS'sub z hz) t fun z' hz' hk => ht z' hz' (hk.trans hzk')
  have hd : tk (du.label, du.after) = .draw :=
    htkv du List.mem_cons_self hdc .draw fun z' hz' hk => by
      obtain ⟨hpart, -⟩ := served_parts hz'
      have h1 : zLab z' = drawLabel θ := (show zLab z' = du.label from congrArg Prod.fst hk).trans hdl
      rw [tagOfPlay_parts hpart, h1]
      simp
  have hp : tk (pt.1.label, pt.1.after) = .points :=
    htkv pt.1 (List.mem_cons_of_mem _ List.mem_cons_self) hpc .points fun z' hz' hk => by
      obtain ⟨hpart, -⟩ := served_parts hz'
      have h1 : zLab z' = _root_.SampledProofs.Hello.slotStream 0 _root_.SampledProofs.OneStageZk.POINTS :=
        (show zLab z' = pt.1.label from congrArg Prod.fst hk).trans hptl
      rw [tagOfPlay_parts hpart, h1, if_neg hPl]
      simp
  -- the blocks
  obtain ⟨T', hTT'⟩ := Game.LiveDrive.rels_of_sizes 512 S' fun z hz => hz512 z (hS'sub z hz)
  have hTT : T'.map (fun q => reqKey q.1.2) =
      (du :: pt.1 :: rUs).flatMap fun u => List.replicate u.coins (u.label, u.after) := by
    rw [← hT, hTT', List.map_map] <;> rfl
  have htag : ∀ q ∈ T', tagOfServed (V.device θ) (drawLabel θ) repStreamOf q.1 = some (tk (reqKey q.1.2)) := by
    intro q hq
    have hz : Game.LiveDrive.rel 512 q ∈ S' := by
      rw [hTT']
      exact List.mem_map_of_mem hq
    have hzS := hS'sub _ hz
    have hsv : servesNext (V.device θ) q.1.1 q.1.2 = true := (List.mem_filter.1 hzS).2
    obtain ⟨t, ht⟩ := hTagK _ hz
    have h1 : tagOfServed (V.device θ) (drawLabel θ) repStreamOf q.1 = tagOfPlay (drawLabel θ) repStreamOf q.1.2 := by
      show (if servesNext (V.device θ) q.1.1 q.1.2 then tagOfPlay (drawLabel θ) repStreamOf q.1.2 else none) = _
      exact if_pos hsv
    have ht1 : tagOfPlay (drawLabel θ) repStreamOf q.1.2 = some t := ht _ hzS rfl
    have ht2 : tk (reqKey q.1.2) = t := htk _ hzS t ht
    rw [h1, ht1, ht2]
  have hr : ∀ u ∈ rUs, ∀ l : List (Game.LiveDrive.Req ByteArray × (Fin 512 → Bool)), l.length = u.coins →
      (∀ q ∈ l, tagOfServed (V.device θ) (drawLabel θ) repStreamOf q.1 = some (tk (u.label, u.after))) →
      ∃ r : Game.LiveDrive.RoundBlock (Game.LiveDrive.Req ByteArray) 512, r.k.rels = l ∧
        (r.k.Fits 512 (tagOfServed (V.device θ) (drawLabel θ) repStreamOf) (.round r.tab r.rep r.n)
          (splitRounds r.n) ∧ r.tab = 0) := by
    intro u hu l hl htg
    obtain ⟨hu0', hc⟩ := hrUs u hu
    obtain ⟨name, a, k, hn, ha, hk, rfl, hm⟩ := hRU u hu0'
    have htku := htkv _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ hu)) hc (.round 0 (streamRep name) k)
      fun z' hz' hkz => hTagR z' hz' name a k hn ha hk hm (congrArg Prod.fst hkz) (congrArg Prod.snd hkz)
        (hneL _ (List.mem_cons_of_mem _ (List.mem_append_left _ hu0')))
    have htg' : ∀ q ∈ l, tagOfServed (V.device θ) (drawLabel θ) repStreamOf q.1 =
        some (.round 0 (streamRep name) k) := fun q hq => (htg q hq).trans (congrArg some htku)
    have hk1 : 1 ≤ k := by
      have h : (k + 3) / 4 ≠ 0 := hc
      omega
    obtain ⟨kb, hkb, hkbF⟩ := Game.LiveDrive.block_of_list 512 (tagOfServed (V.device θ) (drawLabel θ) repStreamOf)
      (splitRounds k) (.round 0 (streamRep name) k) l (hl.trans (splitRounds_len hk1).symm) htg'
    exact ⟨⟨0, streamRep name, k, kb⟩, hkb, hkbF, rfl⟩
  obtain ⟨kd', kp, rds, hTeq, hkdF, hkpF, hrlen, hRF⟩ := Game.LiveDrive.blocks_of_runs 512
    (tagOfServed (V.device θ) (drawLabel θ) repStreamOf) (splitDraw A.Lb)
    (splitPoints (Exec.mPtsOf D.unroll.g (kd / Composed.nTab A.I))) (fun q : Game.LiveDrive.Req ByteArray => reqKey q.2)
    (fun u : _root_.SampledProofs.Challenger.Use => (u.label, u.after))
    (fun u : _root_.SampledProofs.Challenger.Use => u.coins) tk T' du pt.1 rUs hTT htag hd (hdcn.trans hdn) hp hpn'
    (fun r => r.k.Fits 512 (tagOfServed (V.device θ) (drawLabel θ) repStreamOf) (.round r.tab r.rep r.n)
      (splitRounds r.n) ∧ r.tab = 0) hr
  refine ⟨kd', kp, rds, ?_⟩
  unfold Blocks
  refine ⟨hfilter.trans (hTT'.trans (congrArg (List.map (Game.LiveDrive.rel 512)) hTeq)), hkdF, hkpF, ?_, ?_⟩
  · rw [hrlen]
    simp only [List.length_map, List.length_cons] at hlen hN
    omega
  · intro r hr'
    obtain ⟨hfit, h0⟩ := hRF r hr'
    unfold Game.LiveDrive.RoundBlock.Fits
    exact ⟨by
      show r.tab < List.length _
      rw [h0, ZkHidden.planZHJ_length]
      exact Composed.nTab_pos _, hfit⟩

end

/-- **`SampledDeploy.accept_le` at the readers `readersAt`**: `accept_le_of_canon` with the releases' split into blocks
proved (`runs_at`), where the challenger device runs v2's uses, the forger is the one on the records sent, every release
is 512 bits, `SizesAt` holds at each session, and the inputs are the statement's (`inputsAt_of_statement`); the pairs
a collision reads are the play's requests with the round requests sent (`pairsAt`). -/
theorem accept_le_of_runs {V : Vocab d α} (kd : ByteArray → ℕ) (desc : ByteArray → Claims.Msg)
    (A : ∀ (θ F c : ByteArray) (_σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed), V.parsed F = some r →
      Flock.Guarantees.LiveAt r.n r.D r.hU (kd θ) (⌊(V.certOf θ).ε * r.n⌋₊ + 1) (desc θ))
    (hR : Properties.Challenger.RoundsByDevice) (hf : ∀ θ o, V.forger .sampled θ o = V.forgerAt recsAt θ o)
    (rounds_eq : ∀ θ, ∀ k ∈ V.rounds θ, k = 512) (R₀ : ∀ θ F c σ r hr, (A θ F c σ r hr).Reg)
    (hSz : ∀ θ F c σ r hr,
      SizesAt V (fun o => ColEvent (V.colAt recsAt pairsAt θ) o ∨ V.Forges .sampled θ o) (A θ F c σ r hr))
    (hSt : ∀ θ F c σ r hr, ∀ o, V.Verdict .sampled o →
      ¬ (ColEvent (V.colAt recsAt pairsAt θ) o ∨ V.Forges .sampled θ o) → ∀ p c' e pub prof, SampledCase o.sent p c' e →
      _root_.SampledProofs.OneStageZk.checkBlobs p c' e (Core.Blobs.ofArray o.sent.blobs) = .accept pub prof →
      Properties.SampledProofs.OneStageZk.StatementInputs p c' o.sent.blobs (A θ F c σ r hr).I)
    (hDraw : ∀ θ F c σ r hr,
      DrawsAt V θ F c (fun o => ColEvent (V.colAt recsAt pairsAt θ) o ∨ V.Forges .sampled θ o)
        (readersAt V θ (A θ F c σ r hr) (R₀ θ F c σ r hr) (drawLabel θ)))
    (hLink : ∀ θ F c σ r hr,
      LinksAt V θ F c (fun o => ColEvent (V.colAt recsAt pairsAt θ) o ∨ V.Forges .sampled θ o)
        (readersAt V θ (A θ F c σ r hr) (R₀ θ F c σ r hr) (drawLabel θ))) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (r : Flock.Bridge.Parsed)
      (hr : V.parsed F = some r), V.Registered θ F →
      Game.prob (fun o => V.Verdict .sampled o ∧ ¬ ColEvent (V.colAt recsAt pairsAt θ) o ∧ ¬ V.Forges .sampled θ o)
          (V.game θ F c) σ ≤
        FlockSoundness.Game.prob (fun o => ZkExec.LiveAcceptsZK (A θ F c σ r hr).I o) (A θ F c σ r hr).GH
          (drive V θ F c σ _ (readersAt V θ (A θ F c σ r hr) (R₀ θ F c σ r hr) (drawLabel θ))
            (inputsAt_of_statement (hSt θ F c σ r hr))
            (runs_at (θ := θ) (F := F) (c := c) V hR hf (rounds_eq θ) (R₀ θ F c σ r hr) (hSz θ F c σ r hr))
            (hDraw θ F c σ r hr) (hLink θ F c σ r hr)).τ :=
  accept_le_of_canon kd desc A pairsAt (fun θ F c σ r hr => readersAt V θ (A θ F c σ r hr) (R₀ θ F c σ r hr) (drawLabel θ))
    (fun θ F c σ r hr => inputsAt_of_statement (hSt θ F c σ r hr))
    (fun θ F c σ r hr => runs_at (θ := θ) (F := F) (c := c) V hR hf (rounds_eq θ) (R₀ θ F c σ r hr) (hSz θ F c σ r hr))
    hDraw hLink

end Verity.Canon
