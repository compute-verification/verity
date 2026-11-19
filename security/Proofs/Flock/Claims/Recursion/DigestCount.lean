module

public import Mathlib.Data.Finset.Card
public import Flock.With.Log
public import Circuits.Program

/-!
# The composed program digest's SHA-512 calls, counted

`Program.definitionDigestsWith` hashes each entry of a descriptor's `definitions` at most once
(`definitionDigests_calls`), so `Program.programDigestWith` makes at most one call per entry and one more. `digestGo`'s
memo bounds it: a key is hashed only while it is absent from the memo and is inserted right after, and a call on a key
still on the stack of calls in progress never returns, since its loop reaches the next key on that stack
(`digestGo_calls`).
-/

@[expose] public section

namespace RecDeploy.DigestCount

open Flock
open Lean (Json)

local notation "obs " x:max m:max k:max => Id.run (StateT.run (ExceptT.run (StateT.run x m)) k)

set_option quotPrecheck false in
/-- A run of `x` from memo `m` and count `k`: on success its value, memo and count satisfy `R`, on a refusal its count
satisfies `E`. -/
local notation "Spec " x:max m:max k:max R:max E:max =>
  ∀ ρ κ, obs x m k = (ρ, κ) → (∀ a μ', ρ = Except.ok (a, μ') → R a μ' κ) ∧ (∀ ε, ρ = Except.error ε → E κ)

variable {σ α β : Type}

theorem obs_bind (x : StateT σ CountV α) (f : α → StateT σ CountV β) (m : σ) (k : Nat) :
    obs (x >>= f) m k = match obs x m k with
      | (.ok (a, m'), k') => obs (f a) m' k'
      | (.error e, k') => (.error e, k') := by
  simp only [StateT.run_bind, ExceptT.run_bind, Id.run]
  rcases hr : (StateT.run (ExceptT.run (StateT.run x m)) k) with ⟨r, k'⟩
  rcases r with e | ⟨a, m'⟩ <;> simp [hr] <;> rfl

theorem spec_ok {x : StateT σ CountV α} {m : σ} {k : Nat} {R : α → σ → Nat → Prop} {E : Nat → Prop} {a : α} {m1 : σ}
    {k1 : Nat} (hp : obs x m k = (Except.ok (a, m1), k1)) (h : R a m1 k1) : Spec x m k R E := by
  intro r k' hr
  rw [hp, Prod.mk.injEq] at hr
  obtain ⟨rfl, rfl⟩ := hr
  exact ⟨fun _ _ he => (by cases he; exact h), fun _ he => (by cases he)⟩

theorem spec_err {x : StateT σ CountV α} {m : σ} {k : Nat} {R : α → σ → Nat → Prop} {E : Nat → Prop} {e : String}
    {k1 : Nat} (hp : obs x m k = (Except.error e, k1)) (h : E k1) : Spec x m k R E := by
  intro r k' hr
  rw [hp, Prod.mk.injEq] at hr
  obtain ⟨rfl, rfl⟩ := hr
  exact ⟨fun _ _ he => (by cases he), fun _ _ => h⟩

theorem spec_bind {x : StateT σ CountV α} {f : α → StateT σ CountV β} {m : σ} {k : Nat} {R : β → σ → Nat → Prop}
    {E : Nat → Prop} (hx : Spec x m k (fun a m1 k1 => Spec (f a) m1 k1 R E) E) : Spec (x >>= f) m k R E := by
  intro r k' hr
  rw [obs_bind] at hr
  rcases hx1 : obs x m k with ⟨r1, k1⟩
  rw [hx1] at hr
  obtain ⟨hok, herr⟩ := hx r1 k1 hx1
  rcases r1 with e | ⟨a, m1⟩
  · rw [Prod.mk.injEq] at hr
    obtain ⟨rfl, rfl⟩ := hr
    exact ⟨fun _ _ he => (by cases he), fun _ _ => herr e rfl⟩
  · exact hok a m1 rfl r k' hr

theorem spec_mono {x : StateT σ CountV α} {m : σ} {k : Nat} {R R' : α → σ → Nat → Prop} {E E' : Nat → Prop}
    (hx : Spec x m k R E) (hR : ∀ a m1 k1, R a m1 k1 → R' a m1 k1) (hE : ∀ k1, E k1 → E' k1) :
    Spec x m k R' E' := by
  intro r k' hr
  obtain ⟨hok, herr⟩ := hx r k' hr
  exact ⟨fun a m1 he => hR _ _ _ (hok a m1 he), fun e he => hE _ (herr e he)⟩

theorem spec_pure {a : α} {m : σ} {k : Nat} {R : α → σ → Nat → Prop} {E : Nat → Prop} (h : R a m k) :
    Spec (pure a : StateT σ CountV α) m k R E :=
  spec_ok rfl h

theorem spec_get {m : σ} {k : Nat} {R : σ → σ → Nat → Prop} {E : Nat → Prop} (h : R m m k) :
    Spec (get : StateT σ CountV σ) m k R E :=
  spec_ok rfl h

theorem spec_modify {g : σ → σ} {m : σ} {k : Nat} {R : PUnit → σ → Nat → Prop} {E : Nat → Prop}
    (h : R PUnit.unit (g m) k) : Spec (modify g : StateT σ CountV PUnit) m k R E :=
  spec_ok rfl h

theorem spec_throw {e : String} {m : σ} {k : Nat} {R : α → σ → Nat → Prop} {E : Nat → Prop} (h : E k) :
    Spec (throw e : StateT σ CountV α) m k R E :=
  spec_err rfl h

theorem spec_lift {v : V α} {m : σ} {k : Nat} {R : α → σ → Nat → Prop} {E : Nat → Prop} (hok : ∀ a, v = .ok a → R a m k)
    (herr : E k) : Spec (liftM v : StateT σ CountV α) m k R E := by
  rcases v with e | a
  · exact spec_err rfl herr
  · exact spec_ok rfl (hok a rfl)

theorem spec_hash (h : ByteArray → ByteArray) {b : ByteArray} {m : σ} {k : Nat} {R : ByteArray → σ → Nat → Prop}
    {E : Nat → Prop} (hR : R (h b) m (k + 1)) : Spec (liftM ((Hashes.counted h).sha512 b) : StateT σ CountV ByteArray) m k R E :=
  spec_ok rfl hR

theorem spec_discard {x : StateT σ CountV α} {m : σ} {k : Nat} {R : PUnit → σ → Nat → Prop} {E : Nat → Prop}
    (hx : Spec x m k (fun _ m1 k1 => R PUnit.unit m1 k1) E) : Spec (discard x) m k R E := by
  have e : (discard x : StateT σ CountV PUnit) = x >>= fun _ => pure PUnit.unit := rfl
  rw [e]
  exact spec_bind (spec_mono hx (fun _ _ _ h => spec_pure h) (fun _ h => h))

/-- A loop whose every step keeps `Inv`, adds at most what `μ` drops, and is not `Bad`: on success no element is `Bad`. -/
theorem spec_forIn (l : List β) (g : β → PUnit → StateT σ CountV (ForInStep PUnit)) (Inv : σ → Prop) (μ : σ → Nat)
    (Bad : β → Prop)
    (hg : ∀ a ∈ l, ∀ m k, Inv m → Spec (g a PUnit.unit) m k
      (fun r m' k' => r = ForInStep.yield PUnit.unit ∧ ¬Bad a ∧ Inv m' ∧ k' + μ m' ≤ k + μ m) (fun k' => k' ≤ k + μ m)) :
    ∀ m k, Inv m → Spec (forIn l PUnit.unit g) m k
      (fun _ m' k' => (∀ a ∈ l, ¬Bad a) ∧ Inv m' ∧ k' + μ m' ≤ k + μ m) (fun k' => k' ≤ k + μ m) := by
  induction l with
  | nil =>
    intro m k hm
    rw [List.forIn_nil]
    exact spec_pure ⟨by simp, hm, le_refl _⟩
  | cons a l ih =>
    intro m k hm
    rw [List.forIn_cons]
    apply spec_bind
    intro r k1 hr
    obtain ⟨hok, herr⟩ := hg a List.mem_cons_self m k hm r k1 hr
    refine ⟨fun s m1 hs => ?_, herr⟩
    obtain ⟨rfl, hnb, hm1, hk1⟩ := hok s m1 hs
    intro r2 k2 hr2
    obtain ⟨hok2, herr2⟩ := ih (fun b hb => hg b (List.mem_cons_of_mem _ hb)) m1 k1 hm1 r2 k2 hr2
    refine ⟨fun _ m2 h2 => ?_, fun e h2 => ?_⟩
    · obtain ⟨hall, hm2, hk2⟩ := hok2 _ m2 h2
      refine ⟨fun b hb => ?_, hm2, by omega⟩
      rcases List.mem_cons.mp hb with rfl | hb
      · exact hnb
      · exact hall b hb
    · have := herr2 e h2
      omega

section

variable (h : ByteArray → ByteArray) (ddefs : Std.HashMap String Json) (callees types : Array Json)
  (bm : Std.HashMap String String)

set_option hygiene false in
local notation "free " m:max =>
  ((Std.HashMap.keys ddefs).toFinset.filter (fun x => Std.HashMap.contains m x = false)).card

theorem free_mono {m m' : Std.HashMap String String} (hmono : ∀ x, m.contains x = true → m'.contains x = true) :
    free m' ≤ free m := by
  apply Finset.card_le_card
  intro x hx
  rw [Finset.mem_filter] at hx ⊢
  refine ⟨hx.1, ?_⟩
  cases hc : m.contains x
  · rfl
  · have := hx.2
    rw [hmono x hc] at this
    exact absurd this (by simp)

theorem free_insert {m : Std.HashMap String String} {fid v : String} (hd : ddefs.contains fid = true)
    (hm : m.contains fid = false) : free (m.insert fid v) + 1 ≤ free m := by
  have hmem : fid ∈ (ddefs.keys.toFinset.filter (fun x => m.contains x = false)) := by
    rw [Finset.mem_filter, List.mem_toFinset, Std.HashMap.mem_keys]
    exact ⟨Std.HashMap.contains_iff_mem.mp hd, hm⟩
  have hsub : (ddefs.keys.toFinset.filter (fun x => (m.insert fid v).contains x = false)) ⊆
      (ddefs.keys.toFinset.filter (fun x => m.contains x = false)).erase fid := by
    intro x hx
    rw [Finset.mem_filter, Std.HashMap.contains_insert, Bool.or_eq_false_iff, beq_eq_false_iff_ne] at hx
    rw [Finset.mem_erase, Finset.mem_filter]
    exact ⟨fun e => hx.2.1 e.symm, hx.1, hx.2.2⟩
  have := Finset.card_le_card hsub
  rw [Finset.card_erase_of_mem hmem] at this
  have hpos := Finset.card_pos.mpr ⟨fid, hmem⟩
  omega

/-- **`digestGo`'s calls**: from a memo `m` that holds no key of the stack `S` of calls in progress, each of which has a
ref on the stack or the called key `fid`, a run adds at most as many calls as the memo gains entries of `ddefs`; on
success `fid` is not on the stack, and the memo grows and still holds no key of it. -/
theorem digestGo_calls : ∀ (n : Nat) (fid : String) (S : List String) (m : Std.HashMap String String) (k : Nat),
    (∀ s ∈ S, m.contains s = false) →
    (∀ s ∈ S, ∃ b raw e, ddefs[s]? = some raw ∧ Program.canonEntry callees types raw = .ok e ∧
      b ∈ Program.entryRefs callees e ++ (bm[s]?).toArray ∧ (b ∈ S ∨ b = fid)) →
    Spec (Program.digestGo (Hashes.counted h).sha512 ddefs callees types bm n fid) m k
      (fun _ m' k' => fid ∉ S ∧ (∀ s ∈ S, m'.contains s = false) ∧
        (∀ x, m.contains x = true → m'.contains x = true) ∧ k' + free m' ≤ k + free m)
      (fun k' => k' ≤ k + free m) := by
  intro n
  induction n with
  | zero =>
    intro fid S m k _ _
    rw [Program.digestGo.eq_1]
    exact spec_throw (by omega)
  | succ n ih =>
    intro fid S m k hS hP
    rw [Program.digestGo.eq_2]
    apply spec_bind
    apply spec_get
    try dsimp only
    rcases hmf : m[fid]? with _ | x
    · have hmc : m.contains fid = false := by
        rw [Std.HashMap.contains_eq_isSome_getElem?, hmf]
        rfl
      try dsimp only
      rcases hd : ddefs[fid]? with _ | raw
      · dsimp only
        exact spec_throw (by omega)
      · have hdc : ddefs.contains fid = true := by
          rw [Std.HashMap.contains_eq_isSome_getElem?, hd]
          rfl
        try dsimp only
        apply spec_bind
        apply spec_lift _ (by omega)
        intro e hc
        apply spec_bind
        rw [← Array.forIn_toList]
        refine spec_mono (spec_forIn _ _
          (fun m1 => (∀ s ∈ fid :: S, m1.contains s = false) ∧ (∀ x, m.contains x = true → m1.contains x = true))
          (fun m1 => free m1) (fun b => b ∈ fid :: S) ?_ m k ⟨?_, fun _ hx => hx⟩) ?_ (fun _ hk => hk)
        · intro a ha m1 k1 ⟨hS1, hmono1⟩
          apply spec_bind
          apply spec_discard
          refine spec_mono (ih a (fid :: S) m1 k1 hS1 ?_) ?_ (fun _ hk => hk)
          · intro s hs
            rcases List.mem_cons.mp hs with rfl | hs
            · exact ⟨a, raw, e, hd, hc, Array.mem_def.mpr ha, Or.inr rfl⟩
            · obtain ⟨b, raw', e', h1, h2, h3, h4⟩ := hP s hs
              exact ⟨b, raw', e', h1, h2, h3,
                Or.inl (h4.elim (List.mem_cons_of_mem _) (fun hb => hb ▸ List.mem_cons_self))⟩
          · intro _ m' k' ⟨hna, hS', hmono', hk'⟩
            exact spec_pure ⟨rfl, hna, ⟨hS', fun x hx => hmono' x (hmono1 x hx)⟩, hk'⟩
        · intro s hs
          rcases List.mem_cons.mp hs with rfl | hs
          · exact hmc
          · exact hS s hs
        · intro _ m1 k1 ⟨hall, ⟨hS1, hmono1⟩, hk1⟩
          try dsimp only
          apply spec_bind
          apply spec_get
          try dsimp only
          apply spec_bind
          apply spec_lift _ (by omega)
          intro c _
          apply spec_bind
          apply spec_lift _ (by omega)
          intro t _
          apply spec_bind
          apply spec_hash
          try dsimp only
          apply spec_bind
          apply spec_modify
          try dsimp only
          apply spec_pure
          have hfS : fid ∉ S := by
            intro hf
            obtain ⟨b, raw', e', h1, h2, h3, h4⟩ := hP fid hf
            rw [hd, Option.some.injEq] at h1
            subst h1
            rw [hc, Except.ok.injEq] at h2
            subst h2
            exact hall b (Array.mem_def.mp h3)
              (h4.elim (List.mem_cons_of_mem _) (fun hb => hb ▸ List.mem_cons_self))
          refine ⟨hfS, fun s hs => ?_, fun x hx => ?_, ?_⟩
          · have hne : fid ≠ s := fun he => hfS (he ▸ hs)
            rw [Std.HashMap.contains_insert, hS1 s (List.mem_cons_of_mem _ hs), Bool.or_false,
              beq_eq_false_iff_ne]
            exact hne
          · rw [Std.HashMap.contains_insert, hmono1 x hx, Bool.or_true]
          · have := free_insert ddefs (v := hexOf (h t.toUTF8)) hdc (hS1 fid List.mem_cons_self)
            omega
    · dsimp only
      apply spec_pure
      refine ⟨fun hf => ?_, hS, fun _ hx => hx, le_refl _⟩
      have := hS fid hf
      rw [Std.HashMap.contains_eq_isSome_getElem?, hmf] at this
      exact absurd this (by simp)

end

end RecDeploy.DigestCount
