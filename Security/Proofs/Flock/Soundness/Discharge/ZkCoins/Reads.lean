import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs

/-!
# The session on committed coins opens each round at most once

`ReadsIn S g`: every committed round `g` opens is in `S`, and no path opens one twice. It gives `Distinct`
(`distinct_of_readsIn`); `bind` and `interleave` of interactions on disjoint rounds read the union, `batch` of
interactions on pairwise disjoint rounds the union of all, and a stream (`tagRep` on an injective `ix`) reads its rounds
in order. So `sessionZKC` is `Distinct` (`distinct_sessionZKC`): the link points read the session stream's round 0,
and table `j`'s rep `r` reads only the rounds of its own stream.
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game Model ZkSession
open scoped ENNReal

namespace TGame

variable {I Blk α β : Type}

/-- **Every committed round `g` opens is in `S`, and no path opens one twice.** -/
def ReadsIn : Set I → TGame I Blk α → Prop
  | _, .ret _ => True
  | S, .send _ k => ∀ m, ReadsIn S (k m)
  | S, .coin i k => i ∈ S ∧ ∀ b, ReadsIn (S \ {i}) (k b)
  | S, @TGame.fresh _ _ _ _ _ _ k => ∀ c, ReadsIn S (k c)

theorem ReadsIn.mono : ∀ {g : TGame I Blk α} {S S' : Set I}, ReadsIn S g → S ⊆ S' → ReadsIn S' g
  | .ret _, _, _, _, _ => trivial
  | .send _ k, _, _, h, hS => fun m => ReadsIn.mono (g := k m) (h m) hS
  | .coin _ k, _, _, h, hS => ⟨hS h.1, fun b => ReadsIn.mono (g := k b) (h.2 b) (Set.sdiff_subset_sdiff_left hS)⟩
  | @TGame.fresh _ _ _ _ _ _ k, _, _, h, hS => fun c => ReadsIn.mono (g := k c) (h c) hS

theorem noRead_of_readsIn {i : I} : ∀ {g : TGame I Blk α} {S : Set I}, ReadsIn S g → i ∉ S → NoRead i g
  | .ret _, _, _, _ => trivial
  | .send _ k, _, h, hi => fun m => noRead_of_readsIn (g := k m) (h m) hi
  | .coin _ k, _, h, hi =>
    ⟨fun hji => hi (hji ▸ h.1), fun b => noRead_of_readsIn (g := k b) (h.2 b) fun hm => hi hm.1⟩
  | @TGame.fresh _ _ _ _ _ _ k, _, h, hi => fun c => noRead_of_readsIn (g := k c) (h c) hi

theorem distinct_of_readsIn : ∀ {g : TGame I Blk α} {S : Set I}, ReadsIn S g → Distinct g
  | .ret _, _, _ => trivial
  | .send _ k, _, h => fun m => distinct_of_readsIn (g := k m) (h m)
  | .coin _ k, _, h => fun b =>
    ⟨noRead_of_readsIn (g := k b) (h.2 b) fun hm => hm.2 rfl, distinct_of_readsIn (g := k b) (h.2 b)⟩
  | @TGame.fresh _ _ _ _ _ _ k, _, h => fun c => distinct_of_readsIn (g := k c) (h c)

theorem diff_union_sub {S S' : Set I} {i : I} (hi : i ∉ S') : (S \ {i}) ∪ S' ⊆ (S ∪ S') \ {i} := by
  rintro x (⟨hx, hne⟩ | hx)
  · exact ⟨Or.inl hx, hne⟩
  · refine ⟨Or.inr hx, fun h => hi ?_⟩
    rw [Set.mem_singleton_iff] at h
    rwa [h] at hx

theorem union_diff_sub {T S : Set I} {i : I} (hi : i ∉ T) : T ∪ (S \ {i}) ⊆ (T ∪ S) \ {i} := by
  rintro x (hx | ⟨hx, hne⟩)
  · refine ⟨Or.inl hx, fun h => hi ?_⟩
    rw [Set.mem_singleton_iff] at h
    rwa [h] at hx
  · exact ⟨Or.inr hx, hne⟩

theorem readsIn_bind {S' : Set I} {f : α → TGame I Blk β} (hf : ∀ a, ReadsIn S' (f a)) :
    ∀ {g : TGame I Blk α} {S : Set I}, ReadsIn S g → Disjoint S S' → ReadsIn (S ∪ S') (g.bind f)
  | .ret a, _, _, _ => (hf a).mono Set.subset_union_right
  | .send _ k, _, h, hd => fun m => readsIn_bind hf (g := k m) (h m) hd
  | .coin _ k, _, h, hd =>
    ⟨Or.inl h.1, fun b => (readsIn_bind hf (g := k b) (h.2 b) (hd.mono_left Set.sdiff_subset)).mono
      (diff_union_sub (Set.disjoint_left.1 hd h.1))⟩
  | @TGame.fresh _ _ _ _ _ _ k, _, h, hd => fun c => readsIn_bind hf (g := k c) (h c) hd

theorem readsIn_map {S : Set I} {g : TGame I Blk α} (h : ReadsIn S g) (f : α → β) : ReadsIn S (g.map f) := by
  have := readsIn_bind (S' := ∅) (f := fun a => TGame.ret (f a)) (fun _ => trivial) h (Set.disjoint_empty _)
  rwa [Set.union_empty] at this

theorem readsIn_interleaveWith {γ : Type} {T : Set I} {adv : TGame I Blk β → TGame I Blk (γ × β)}
    {fin : β → TGame I Blk (γ × β)} (hfin : ∀ b, ReadsIn T (fin b)) :
    ∀ {g : TGame I Blk β} {S : Set I}, ReadsIn S g → Disjoint T S →
      (∀ (S' : Set I) (g' : TGame I Blk β), S' ⊆ S → ReadsIn S' g' → ReadsIn (T ∪ S') (adv g')) →
      ReadsIn (T ∪ S) (TGame.interleaveWith adv fin g)
  | .ret b, _, _, _, _ => (hfin b).mono Set.subset_union_left
  | .send M k, S, h, hd, hadv => by
    simp only [TGame.interleaveWith, ReadsIn]
    intro first
    cases first
    · simp only [Bool.false_eq_true, ite_false, ReadsIn]
      exact fun m => readsIn_interleaveWith hfin (g := k m) (h m) hd hadv
    · simp only [ite_true]
      exact hadv S _ subset_rfl h
  | .coin i k, S, h, hd, hadv => by
    simp only [TGame.interleaveWith, ReadsIn]
    intro first
    cases first
    · simp only [Bool.false_eq_true, ite_false, ReadsIn]
      exact ⟨Or.inr h.1, fun b => (readsIn_interleaveWith hfin (g := k b) (h.2 b) (hd.mono_right Set.sdiff_subset)
        (fun S' g' hS' => hadv S' g' (hS'.trans Set.sdiff_subset))).mono
        (union_diff_sub (Set.disjoint_right.1 hd h.1))⟩
    · simp only [ite_true]
      exact hadv S _ subset_rfl h
  | @TGame.fresh _ _ _ C _ _ k, S, h, hd, hadv => by
    simp only [TGame.interleaveWith, ReadsIn]
    intro first
    cases first
    · simp only [Bool.false_eq_true, ite_false, ReadsIn]
      exact fun c => readsIn_interleaveWith hfin (g := k c) (h c) hd hadv
    · simp only [ite_true]
      exact hadv S _ subset_rfl h

theorem readsIn_interleave :
    ∀ {g₁ : TGame I Blk α} {S₁ : Set I} {g₂ : TGame I Blk β} {S₂ : Set I},
      ReadsIn S₁ g₁ → ReadsIn S₂ g₂ → Disjoint S₁ S₂ → ReadsIn (S₁ ∪ S₂) (g₁.interleave g₂)
  | .ret a, _, _, _, _, h₂, _ => by
    simp only [TGame.interleave]
    exact (readsIn_map h₂ _).mono Set.subset_union_right
  | .send M k, S₁, _, _, h₁, h₂, hd => by
    simp only [TGame.interleave]
    exact readsIn_interleaveWith (fun b => readsIn_map h₁ _) h₂ hd
      fun S' g' hS' h' => fun m => readsIn_interleave (g₁ := k m) (h₁ m) h' (hd.mono_right hS')
  | .coin i k, S₁, _, _, h₁, h₂, hd => by
    simp only [TGame.interleave]
    exact readsIn_interleaveWith (fun b => readsIn_map h₁ _) h₂ hd
      fun S' g' hS' h' => ⟨Or.inl h₁.1, fun c => (readsIn_interleave (g₁ := k c) (h₁.2 c) h'
        ((hd.mono_right hS').mono_left Set.sdiff_subset)).mono
        (diff_union_sub (Set.disjoint_left.1 (hd.mono_right hS') h₁.1))⟩
  | @TGame.fresh _ _ _ C _ _ k, S₁, _, _, h₁, h₂, hd => by
    simp only [TGame.interleave]
    exact readsIn_interleaveWith (fun b => readsIn_map h₁ _) h₂ hd
      fun S' g' hS' h' => fun c => readsIn_interleave (g₁ := k c) (h₁ c) h' (hd.mono_right hS')

/-- A batch whose `k`-th interaction reads `L (n + k)`, the `L k` pairwise disjoint, reads `⋃ k ≥ n, L k`. -/
theorem readsIn_batch (L : ℕ → Set I) (hL : ∀ a b, a ≠ b → Disjoint (L a) (L b)) :
    ∀ (gs : List (TGame I Blk α)) (n : ℕ), (∀ k (hk : k < gs.length), ReadsIn (L (n + k)) (gs.get ⟨k, hk⟩)) →
      ReadsIn {x | ∃ k, n ≤ k ∧ x ∈ L k} (TGame.batch gs)
  | [], _, _ => trivial
  | g :: gs, n, h => by
    have h₁ : ReadsIn (L n) g := h 0 (Nat.zero_lt_succ _)
    have h₂ := readsIn_batch L hL gs (n + 1) fun k hk => by
      have := h (k + 1) (Nat.succ_lt_succ hk)
      rw [show n + 1 + k = n + (k + 1) by omega]
      exact this
    have hd : Disjoint (L n) {x | ∃ k, n + 1 ≤ k ∧ x ∈ L k} := by
      rw [Set.disjoint_left]
      rintro x hx ⟨k, hk, hxk⟩
      exact Set.disjoint_left.1 (hL n k (by omega)) hx hxk
    simp only [TGame.batch]
    refine (readsIn_map (readsIn_interleave h₁ h₂ hd) _).mono ?_
    rintro x (hx | ⟨k, hk, hxk⟩)
    · exact ⟨n, le_rfl, hx⟩
    · exact ⟨k, by omega, hxk⟩

end TGame

open TGame

variable {I Blk α : Type} [Fintype Blk]

/-- **A stream reads its rounds in order**: from its `t`-th draw on, `tagRep` reads only `ix t'` with `t ≤ t'`, each
at most once. -/
theorem readsIn_tagRep (dc : Decoders Blk) {R : ℕ} (ix : Fin R → I) (hix : Function.Injective ix) :
    ∀ (t : ℕ) (g : Game α), ReadsIn {x | ∃ t' : Fin R, t ≤ t'.val ∧ x = ix t'} (tagRep dc ix t g)
  | _, .ret _ => trivial
  | t, .send M k => fun m => readsIn_tagRep dc ix hix t (k m)
  | t, @Game.coin _ C _ _ k => by
    unfold tagRep
    split_ifs with h
    · split
      · rename_i d _
        refine ⟨⟨⟨t, h⟩, le_rfl, rfl⟩, fun b => (readsIn_tagRep dc ix hix (t + 1) (k (d b))).mono ?_⟩
        rintro x ⟨t', ht', rfl⟩
        refine ⟨⟨t', by omega, rfl⟩, fun hx => ?_⟩
        have h' : t'.val = t := congrArg Fin.val (hix (Set.mem_singleton_iff.1 hx))
        omega
      · exact fun c => readsIn_tagRep dc ix hix t (k c)
    · exact fun c => readsIn_tagRep dc ix hix t (k c)

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

/-- Table `j`'s rounds: round `t` of either of its reps' streams. -/
def tabRounds (J R : ℕ) (k : ℕ) : Set (SessIdx J R) := {x | ∃ q : Fin J × Fin 2 × Fin R, q.1.val = k ∧ x = some q}

omit [Fintype K] in
theorem readsIn_afterC (dc : Decoders Blk) (R : ℕ) {J : ℕ} (j : Fin J) (A : Arith F K) (T : TabZK F mPts)
    (C₀ : Oracle F) (pts : Fin 2 → Fin mPts → F) : ReadsIn (tabRounds J R j) (afterC dc R j A T C₀ pts) := by
  have hinj : ∀ r : Fin 2, Function.Injective fun t : Fin R => (some (j, r, t) : SessIdx J R) :=
    fun r t t' h => by simpa using h
  refine (readsIn_map (readsIn_interleave (readsIn_tagRep dc _ (hinj 0) 0 _) (readsIn_tagRep dc _ (hinj 1) 0 _)
    ?_) _).mono ?_
  · rw [Set.disjoint_left]
    rintro x ⟨t, _, rfl⟩ ⟨t', _, h⟩
    simp at h
  · rintro x (⟨t, _, rfl⟩ | ⟨t, _, rfl⟩)
    · exact ⟨(j, 0, t), rfl, rfl⟩
    · exact ⟨(j, 1, t), rfl, rfl⟩

omit [Fintype K] in
/-- **The session on committed coins opens each round at most once.** -/
theorem distinct_sessionZKC (dc : Decoders Blk) (R : ℕ) (A : Arith F K) (ts : List (TabZK F mPts)) :
    (sessionZKC dc R A ts).Distinct := by
  have hL : ∀ a b, a ≠ b → Disjoint (tabRounds ts.length R a) (tabRounds ts.length R b) := by
    intro a b hab
    rw [Set.disjoint_left]
    rintro x ⟨q, hq, rfl⟩ ⟨q', hq', h⟩
    cases h
    exact hab (hq.symm.trans hq')
  have hb : ∀ (Cs : Fin ts.length → Oracle F) (pts : Fin 2 → Fin mPts → F),
      ReadsIn {x | ∃ k, 0 ≤ k ∧ x ∈ tabRounds ts.length R k}
        (TGame.batch (List.ofFn fun i => afterC dc R i A (ts.get i) (Cs i) pts)) := by
    intro Cs pts
    have key : ∀ i : Fin ts.length, ReadsIn (tabRounds ts.length R i) (afterC dc R i A (ts.get i) (Cs i) pts) :=
      fun i => readsIn_afterC dc R i A _ _ pts
    refine readsIn_batch _ hL _ 0 fun k hk => ?_
    simpa using key ⟨k, by simpa using hk⟩
  refine distinct_of_readsIn (S := Set.univ) ?_
  unfold sessionZKC
  simp only [ReadsIn]
  intro Cs
  unfold coinAt
  split
  · refine ⟨trivial, fun b => (hb Cs _).mono ?_⟩
    rintro x ⟨k, _, q, _, rfl⟩
    exact ⟨trivial, fun h => by simp at h⟩
  · exact fun c => (hb Cs c).mono (Set.subset_univ _)

theorem card_sessIdx (J R : ℕ) : Fintype.card (SessIdx J R) = 1 + 2 * J * R := by
  simp only [SessIdx, Fintype.card_option, Fintype.card_prod, Fintype.card_fin]
  ring

end FlockSoundness.Discharge.ZkCoins
