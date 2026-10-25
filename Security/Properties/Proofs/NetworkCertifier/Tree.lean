import Properties.Proofs.NetworkCertifier.Bytes
import Properties.Proofs.NetworkCertifier.Strings

/-!
# The frame tree binds its frames

Where SHA-512 has no collision on the strings a tree hashes (`FramesString`), its root determines its frames
(`framesRoot_inj`): no level's node is its level's empty subtree, so a level determines the one below (`pairs_inj`), and
the top of a tree of at most `2^D` leaves is one node or none.
-/

open NetworkCheck hiding Params
open Flock
open NetworkCheck.Check (Grid)

namespace Properties.NetworkCertifier.Tree

open Properties.NetworkCertifier Properties.NetworkCertifier.Bytes

/-- SHA-512 has no collision on `X`: a hypothesis of these lemmas only. The claims either discharge it on the run's
strings (`cf_run`) or conclude that two distinct strings of the run have one digest. -/
abbrev CF (X : ByteArray → Prop) : Prop := ∀ x y, X x → X y → Sha512.hash x = Sha512.hash y → x = y

variable {X : ByteArray → Prop}

theorem node_inj (hX : CF X) {a b a' b' : ByteArray} (ha : a.size = 64) (ha' : a'.size = 64)
    (hx : X (nodeInput a b)) (hx' : X (nodeInput a' b')) (h : node a b = node a' b') : a = a' ∧ b = b' :=
  nodeInput_inj ha ha' (hX _ _ hx hx' h)

theorem empty_succ (l : Nat) : empty (l + 1) = node (empty l) (empty l) := rfl

/-- Every element of level `lv` is a digest, and none is the level's empty subtree. -/
def Good (lv : Nat) (L : List ByteArray) : Prop := ∀ a ∈ L, a.size = 64 ∧ a ≠ empty lv

theorem pairs_inj (hX : CF X) (lv : Nat) : ∀ {L L' : List ByteArray}, Good lv L → Good lv L' →
    (∀ x ∈ pairInputs lv L, X x) → (∀ x ∈ pairInputs lv L', X x) → pairs lv L = pairs lv L' → L = L'
  | [], [], _, _, _, _, _ => rfl
  | [], [_], _, _, _, _, h => by simp [pairs] at h
  | [], _ :: _ :: _, _, _, _, _, h => by simp [pairs] at h
  | [_], [], _, _, _, _, h => by simp [pairs] at h
  | _ :: _ :: _, [], _, _, _, _, h => by simp [pairs] at h
  | [a], [a'], hg, hg', hp, hp', h => by
    simp only [pairs, List.cons.injEq, and_true] at h
    have e := node_inj hX (hg a (by simp)).1 (hg' a' (by simp)).1 (hp _ (by simp [pairInputs]))
      (hp' _ (by simp [pairInputs])) h
    rw [e.1]
  | [a], a' :: b' :: r', hg, hg', hp, hp', h => by
    simp only [pairs, List.cons.injEq] at h
    have e := node_inj hX (hg a (by simp)).1 (hg' a' (by simp)).1 (hp _ (by simp [pairInputs]))
      (hp' _ (by simp [pairInputs])) h.1
    exact absurd e.2.symm (hg' b' (by simp)).2
  | a :: b :: r, [a'], hg, hg', hp, hp', h => by
    simp only [pairs, List.cons.injEq] at h
    have e := node_inj hX (hg a (by simp)).1 (hg' a' (by simp)).1 (hp _ (by simp [pairInputs]))
      (hp' _ (by simp [pairInputs])) h.1
    exact absurd e.2 (hg b (by simp)).2
  | a :: b :: r, a' :: b' :: r', hg, hg', hp, hp', h => by
    simp only [pairs, List.cons.injEq] at h
    obtain ⟨e1, e2⟩ := node_inj hX (hg a (by simp)).1 (hg' a' (by simp)).1 (hp _ (by simp [pairInputs]))
      (hp' _ (by simp [pairInputs])) h.1
    have e3 := pairs_inj hX lv (fun x hx => hg x (by simp [hx])) (fun x hx => hg' x (by simp [hx]))
      (fun x hx => hp x (by simp [pairInputs, hx])) (fun x hx => hp' x (by simp [pairInputs, hx])) h.2
    rw [e1, e2, e3]

theorem node_good (hX : CF X) {lv : Nat} {a b : ByteArray} (ha : a.size = 64) (hne : a ≠ empty lv)
    (hx : X (nodeInput a b)) (he : X (nodeInput (empty lv) (empty lv))) :
    (node a b).size = 64 ∧ node a b ≠ empty (lv + 1) := by
  refine ⟨hash_size _, fun h => hne ?_⟩
  rw [empty_succ] at h
  exact (node_inj hX ha (empty_size lv) hx he h).1

theorem pairs_good (hX : CF X) (lv : Nat) (he : X (nodeInput (empty lv) (empty lv))) :
    ∀ {L : List ByteArray}, Good lv L → (∀ x ∈ pairInputs lv L, X x) → Good (lv + 1) (pairs lv L)
  | [], _, _ => by simp [Good, pairs]
  | [a], hg, hp => by
    intro x hx
    simp only [pairs, List.mem_singleton] at hx
    subst hx
    exact node_good hX (hg a (by simp)).1 (hg a (by simp)).2 (hp _ (by simp [pairInputs])) he
  | a :: b :: r, hg, hp => by
    intro x hx
    simp only [pairs, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact node_good hX (hg a (by simp)).1 (hg a (by simp)).2 (hp _ (by simp [pairInputs])) he
    · exact pairs_good hX lv he (fun x hx => hg x (by simp [hx])) (fun x hx => hp x (by simp [pairInputs, hx])) x hx

theorem pairs_length (lv : Nat) : ∀ L : List ByteArray, (pairs lv L).length = (L.length + 1) / 2
  | [] => rfl
  | [_] => by simp [pairs]
  | _ :: _ :: r => by
    simp only [pairs, List.length_cons, pairs_length lv r]
    omega

theorem pairs_ne_nil (lv : Nat) : ∀ {L : List ByteArray}, L ≠ [] → pairs lv L ≠ []
  | [], h => absurd rfl h
  | [_], _ => by simp [pairs]
  | _ :: _ :: _, _ => by simp [pairs]

theorem levelAt_ne_nil : ∀ (k : Nat) {L : List ByteArray}, L ≠ [] → levelAt k L ≠ []
  | 0, _, h => h
  | k + 1, _, h => pairs_ne_nil k (levelAt_ne_nil k h)

theorem levelAt_nil : ∀ k : Nat, levelAt k [] = []
  | 0 => rfl
  | k + 1 => by simp only [levelAt, levelAt_nil k]; rfl

theorem levelAt_length_le : ∀ (k : Nat) (L : List ByteArray) (m : Nat), L.length ≤ 2 ^ k * m →
    (levelAt k L).length ≤ m
  | 0, L, m, h => by simpa [levelAt] using h
  | k + 1, L, m, h => by
    have ih := levelAt_length_le k L (2 * m) (by rw [Nat.pow_succ] at h; rw [← Nat.mul_assoc]; exact h)
    simp only [levelAt, pairs_length]
    omega

theorem top_levelAt : ∀ (k j : Nat) (L : List ByteArray), top j k (levelAt j L) = levelAt (j + k) L
  | 0, j, L => rfl
  | k + 1, j, L => by
    simp only [top]
    rw [show pairs j (levelAt j L) = levelAt (j + 1) L from rfl, top_levelAt k (j + 1) L,
      show j + 1 + k = j + (k + 1) by omega]

/-- The tree's levels below `D` hash only strings of `X`, and its empty subtrees are those of `X`. -/
structure Hashes (X : ByteArray → Prop) (D : Nat) (L : List ByteArray) : Prop where
  pairs : ∀ lv < D, ∀ x ∈ pairInputs lv (levelAt lv L), X x
  empties : ∀ l < D, X (nodeInput (empty l) (empty l))

theorem levelAt_good (hX : CF X) {D : Nat} {L : List ByteArray} (h0 : Good 0 L) (hh : Hashes X D L) :
    ∀ k ≤ D, Good k (levelAt k L)
  | 0, _ => h0
  | k + 1, hk => pairs_good hX k (hh.empties k (by omega)) (levelAt_good hX h0 hh k (by omega))
      (hh.pairs k (by omega))

theorem levelAt_inj (hX : CF X) {D : Nat} {L L' : List ByteArray} (h0 : Good 0 L) (h0' : Good 0 L')
    (hh : Hashes X D L) (hh' : Hashes X D L') : ∀ k ≤ D, levelAt k L = levelAt k L' → L = L'
  | 0, _, h => h
  | k + 1, hk, h =>
    levelAt_inj hX h0 h0' hh hh' k (by omega)
      (pairs_inj hX k (levelAt_good hX h0 hh k (by omega)) (levelAt_good hX h0' hh' k (by omega))
        (hh.pairs k (by omega)) (hh'.pairs k (by omega)) h)

/-- The top of trees of at most `2^D` leaves, as `framesRoot` reads it, determines the leaves. -/
theorem root_inj (hX : CF X) {D : Nat} {L L' : List ByteArray} (h0 : Good 0 L) (h0' : Good 0 L')
    (hh : Hashes X D L) (hh' : Hashes X D L') (hl : L.length ≤ 2 ^ D) (hl' : L'.length ≤ 2 ^ D)
    (h : (top 0 D L).headD (empty D) = (top 0 D L').headD (empty D)) : L = L' := by
  have e : ∀ L : List ByteArray, top 0 D L = levelAt D L := fun L => by
    have := top_levelAt D 0 L
    rwa [Nat.zero_add] at this
  rw [e, e] at h
  have g := levelAt_good hX h0 hh D (Nat.le_refl _)
  have g' := levelAt_good hX h0' hh' D (Nat.le_refl _)
  have n := levelAt_length_le D L 1 (by simpa using hl)
  have n' := levelAt_length_le D L' 1 (by simpa using hl')
  have nil : ∀ {L : List ByteArray}, levelAt D L = [] → L = [] := fun {L} h => by
    by_contra hne
    exact levelAt_ne_nil D hne h
  match hA : levelAt D L, hB : levelAt D L', n, n', g, g' with
  | [], [], _, _, _, _ => rw [nil hA, nil hB]
  | [x], [], _, _, g, _ => exact absurd (by rw [hA, hB] at h; exact h) (g x (by simp)).2
  | [], [x'], _, _, _, g' => exact absurd (by rw [hA, hB] at h; exact h.symm) (g' x' (by simp)).2
  | [x], [x'], _, _, _, _ =>
    exact levelAt_inj hX h0 h0' hh hh' D (Nat.le_refl _) (by rw [hA, hB] at h ⊢; simp only [List.headD] at h; rw [h])
  | _ :: _ :: _, _, n, _, _, _ => simp at n
  | _, _ :: _ :: _, _, n', _, _ => simp at n'

/-! ## From the frames to the root -/

theorem mapM_mem {α β ε : Type} {f : α → Except ε β} : ∀ {l : List α} {r : List β}, l.mapM f = .ok r →
    ∀ b ∈ r, ∃ a ∈ l, f a = .ok b
  | [], r, h, b, hb => by
    rw [List.mapM_nil] at h
    cases Except.ok.inj h
    simp at hb
  | a :: l, r, h, b, hb => by
    rw [List.mapM_cons] at h
    cases hf : f a with
    | error _ => rw [hf] at h; exact (error_ne_ok h).elim
    | ok b₀ =>
      cases hl : l.mapM f with
      | error _ => rw [hf, hl] at h; exact (error_ne_ok h).elim
      | ok r₀ =>
        rw [hf, hl] at h
        cases Except.ok.inj h
        rcases List.mem_cons.mp hb with rfl | hb
        · exact ⟨a, by simp, hf⟩
        · obtain ⟨a', ha', e⟩ := mapM_mem hl b hb
          exact ⟨a', by simp [ha'], e⟩

theorem mapM_length {α β ε : Type} {f : α → Except ε β} : ∀ {l : List α} {r : List β}, l.mapM f = .ok r →
    r.length = l.length
  | [], r, h => by
    rw [List.mapM_nil] at h
    cases Except.ok.inj h
    rfl
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    cases hf : f a with
    | error _ => rw [hf] at h; exact (error_ne_ok h).elim
    | ok b₀ =>
      cases hl : l.mapM f with
      | error _ => rw [hf, hl] at h; exact (error_ne_ok h).elim
      | ok r₀ =>
        rw [hf, hl] at h
        cases Except.ok.inj h
        simp [mapM_length hl]

theorem mapM_inj {α β ε : Type} {f : α → Except ε β} {P : α → Prop}
    (hf : ∀ a a' b, P a → P a' → f a = .ok b → f a' = .ok b → a = a') :
    ∀ {l l' : List α} {r : List β}, (∀ a ∈ l, P a) → (∀ a ∈ l', P a) → l.mapM f = .ok r → l'.mapM f = .ok r → l = l'
  | [], l', r, _, _, h, h' => by
    have := mapM_length h'
    rw [List.mapM_nil] at h
    cases Except.ok.inj h
    exact (List.eq_nil_of_length_eq_zero this.symm).symm
  | a :: l, l', r, hp, hp', h, h' => by
    rw [List.mapM_cons] at h
    cases hf₁ : f a with
    | error _ => rw [hf₁] at h; exact (error_ne_ok h).elim
    | ok b₀ =>
      cases hl : l.mapM f with
      | error _ => rw [hf₁, hl] at h; exact (error_ne_ok h).elim
      | ok r₀ =>
        rw [hf₁, hl] at h
        cases Except.ok.inj h
        match l', hp', h' with
        | [], _, h' => rw [List.mapM_nil] at h'; cases Except.ok.inj h'
        | a' :: l', hp', h' =>
          rw [List.mapM_cons] at h'
          cases hf₂ : f a' with
          | error _ => rw [hf₂] at h'; exact (error_ne_ok h').elim
          | ok b₁ =>
            cases hl' : l'.mapM f with
            | error _ => rw [hf₂, hl'] at h'; exact (error_ne_ok h').elim
            | ok r₁ =>
              rw [hf₂, hl'] at h'
              have e := Except.ok.inj h'
              simp only [List.cons.injEq] at e
              obtain ⟨e1, e2⟩ := e
              subst e1 e2
              rw [hf a a' _ (hp a (by simp)) (hp' a' (by simp)) hf₁ hf₂,
                mapM_inj hf (fun x hx => hp x (by simp [hx])) (fun x hx => hp' x (by simp [hx])) hl hl']

theorem frameLeaf_ok {f b : ByteArray} {B : Nat} (h : frameLeaf f B = .ok b) :
    ∃ x, leafInput f B = .ok x ∧ b = Sha512.hash x := by
  unfold frameLeaf at h
  cases hl : leafInput f B with
  | error _ => rw [hl] at h; exact (error_ne_ok h).elim
  | ok x => rw [hl] at h; exact ⟨x, rfl, (Except.ok.inj h).symm⟩

theorem leaves_ok {fr ls : Array ByteArray} {g : Grid} (h : leaves fr g = .ok ls) :
    fr.size ≤ g.buckets * g.allocation ∧ fr.toList.mapM (frameLeaf · g.frameBytes) = .ok ls.toList := by
  unfold leaves at h
  by_cases hc : fr.size > g.buckets * g.allocation
  · rw [ite_eq_left hc] at h
    exact (error_ne_ok h).elim
  rw [ite_eq_right hc] at h
  refine ⟨by omega, ?_⟩
  have h' : Array.mapM (frameLeaf · g.frameBytes) fr = .ok ls := h
  rw [Array.mapM_eq_mapM_toList] at h'
  cases hm : fr.toList.mapM (frameLeaf · g.frameBytes) with
  | error _ => rw [hm] at h'; exact (error_ne_ok h').elim
  | ok l => rw [hm] at h'; rw [← Except.ok.inj h']

theorem framesRoot_ok {fr : Array ByteArray} {g : Grid} {rt : ByteArray} (h : framesRoot fr g = .ok rt) :
    ∃ ls, leaves fr g = .ok ls ∧ rt = (top 0 (depth g) ls.toList).headD (empty (depth g)) := by
  unfold framesRoot at h
  cases hl : leaves fr g with
  | error _ => rw [hl] at h; exact (error_ne_ok h).elim
  | ok ls => rw [hl] at h; exact ⟨ls, rfl, (Except.ok.inj h).symm⟩

theorem le_pow_depth (g : Grid) : g.buckets * g.allocation ≤ 2 ^ depth g := by
  unfold depth bitLength
  split
  · rename_i h
    simp only [Nat.pow_zero]
    omega
  · rename_i h
    have := @Nat.lt_log2_self (g.buckets * g.allocation - 1)
    omega

/-- **The root binds the frames**: two frame sequences with one root, where SHA-512 has no collision on the strings
both trees hash, are one. -/
theorem framesRoot_inj (hX : CF X) {g : Grid} {fr fr' : Array ByteArray} {rt : ByteArray}
    (h : framesRoot fr g = .ok rt) (h' : framesRoot fr' g = .ok rt)
    (hs : ∀ x, FramesString g fr x → X x) (hs' : ∀ x, FramesString g fr' x → X x) : fr = fr' := by
  obtain ⟨ls, hl, rfl⟩ := framesRoot_ok h
  obtain ⟨ls', hl', hr⟩ := framesRoot_ok h'
  obtain ⟨hsz, hm⟩ := leaves_ok hl
  obtain ⟨hsz', hm'⟩ := leaves_ok hl'
  have good : ∀ {fr ls : Array ByteArray}, (∀ x, FramesString g fr x → X x) →
      fr.toList.mapM (frameLeaf · g.frameBytes) = .ok ls.toList → Good 0 ls.toList := by
    intro fr ls hs hm b hb
    obtain ⟨f, hf, e⟩ := mapM_mem hm b hb
    obtain ⟨x, hx, rfl⟩ := frameLeaf_ok e
    refine ⟨hash_size _, fun he => leafInput_ne_emptyInput hx (hX _ _ ?_ ?_ he)⟩
    · exact hs _ (Or.inl ⟨f, hf, hx⟩)
    · exact hs _ (Or.inr (Or.inl rfl))
  have hashes : ∀ {fr ls : Array ByteArray}, (∀ x, FramesString g fr x → X x) → leaves fr g = .ok ls →
      Hashes X (depth g) ls.toList := fun {fr ls} hs hl =>
    ⟨fun lv hlv x hx => hs x (Or.inr (Or.inr (Or.inr ⟨ls, hl, lv, hlv, hx⟩))),
     fun l hl => hs _ (Or.inr (Or.inr (Or.inl ⟨l, hl, rfl⟩)))⟩
  have len : ∀ {fr ls : Array ByteArray}, fr.size ≤ g.buckets * g.allocation →
      fr.toList.mapM (frameLeaf · g.frameBytes) = .ok ls.toList → ls.toList.length ≤ 2 ^ depth g := by
    intro fr ls hsz hm
    rw [mapM_length hm, Array.length_toList]
    exact Nat.le_trans hsz (le_pow_depth g)
  have e := root_inj hX (good hs hm) (good hs' hm') (hashes hs hl) (hashes hs' hl') (len hsz hm) (len hsz' hm') hr
  rw [e] at hm
  apply Array.toList_inj.mp
  refine mapM_inj (P := fun f => ∀ x, leafInput f g.frameBytes = .ok x → X x) ?_ ?_ ?_ hm hm'
  · intro a a' b hp hp' ha ha'
    obtain ⟨x, hx, rfl⟩ := frameLeaf_ok ha
    obtain ⟨x', hx', e⟩ := frameLeaf_ok ha'
    have := hX _ _ (hp x hx) (hp' x' hx') e
    subst this
    exact leafInput_inj hx hx'
  · exact fun a ha x hx => hs _ (Or.inl ⟨a, ha, hx⟩)
  · exact fun a ha x hx => hs' _ (Or.inl ⟨a, ha, hx⟩)

end Properties.NetworkCertifier.Tree
