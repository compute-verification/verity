import Proofs.Flock.Soundness.Refine.Run
import Flock.Verify
import Proofs.Flock.Verifier.Transcript
import Mathlib.Data.List.Range

/-!
# Runs of the executable's transcript programs

The executable verifier's phases are programs in `TM = StateT Transcript (Except String)` (PROTOCOL.md §6). A run that
succeeds, `x t = .ok (a, t')`, is taken apart one step at a time: each lemma below peels one step off the front of a
`bind` and says what that step did.

* Labels, absorbs and proof-of-work nonces change only the pending bytes: `t` and `t'` have the same rounds and the same
  round counter (`Keeps`).
* A squeeze of `n ≠ 0` coins consumes the next round, which issued `n` coins, and returns that round's coins.
* A check that passes changes nothing.

`Adv t t' cs` records what a stretch of a run consumed: the same rounds, and the coins of the rounds from `t.k` to
`t'.k` are `cs`, in order (`consumed`). It composes (`Adv.trans`), so a phase's coins are its steps' coins, concatenated.
-/

namespace FlockSoundness.Refine

open Flock Transcript

/-- The executable's `==` on field elements is equality. -/
instance instLawfulBEqF128_flockSoundness : LawfulBEq F128 where
  eq_of_beq {a b} h := by
    cases a with | mk alo ahi => cases b with | mk blo bhi =>
    have h' : ((alo == blo) && (ahi == bhi)) = true := h
    rw [Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h'
    rw [h'.1, h'.2]
  rfl {a} := by
    cases a with | mk alo ahi =>
    show ((alo == alo) && (ahi == ahi)) = true
    simp

instance instLawfulBEqF256_flockSoundness : LawfulBEq F256 where
  eq_of_beq {a b} h := by
    cases a with | mk a0 a1 => cases b with | mk b0 b1 =>
    have h' : ((a0 == b0) && (a1 == b1)) = true := h
    rw [Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h'
    rw [h'.1, h'.2]
  rfl {a} := by
    cases a with | mk a0 a1 =>
    show ((a0 == a0) && (a1 == a1)) = true
    simp

/-- `t'` follows `t` on the same rounds, and the rounds consumed in between have the coins `cs`. -/
def Adv (t t' : Transcript) (cs : List (Array F128)) : Prop :=
  t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧ consumed t.rounds t.k t'.k = cs

/-- Every read of a well-formed record's rounds has the round's coin count; a read past the end is the empty round. -/
theorem RoundsWF.get {rs : Array Round} (h : RoundsWF rs) (i : ℕ) : (rs[i]!).coins.size = (rs[i]!).n := by
  by_cases hi : i < rs.size
  · exact h i hi
  · rw [getElem!_neg rs i hi]; rfl

theorem consumed_self (rs : Array Round) (k : ℕ) : consumed rs k k = [] := by simp [consumed]

theorem consumed_append (rs : Array Round) {k k' k'' : ℕ} (h₁ : k ≤ k') (h₂ : k' ≤ k'') :
    consumed rs k k'' = consumed rs k k' ++ consumed rs k' k'' := by
  obtain ⟨a, rfl⟩ := Nat.exists_eq_add_of_le h₁
  obtain ⟨b, rfl⟩ := Nat.exists_eq_add_of_le h₂
  have e1 : k + a + b - k = a + b := by omega
  have e2 : k + a - k = a := by omega
  have e3 : k + a + b - (k + a) = b := by omega
  unfold consumed
  rw [e1, e2, e3, ← List.range'_append_1, List.map_append]

theorem consumed_succ (rs : Array Round) (k : ℕ) : consumed rs k (k + 1) = [(rs[k]!).coins] := by
  simp [consumed]

theorem Adv.refl (t : Transcript) : Adv t t [] := ⟨rfl, le_rfl, consumed_self _ _⟩

theorem Adv.of_keeps {t t' : Transcript} (hk : t'.k = t.k) (hr : t'.rounds = t.rounds) : Adv t t' [] :=
  ⟨hr, hk.ge, by rw [hk, consumed_self]⟩

theorem Adv.trans {t₁ t₂ t₃ : Transcript} {cs₁ cs₂ : List (Array F128)} (h₁ : Adv t₁ t₂ cs₁) (h₂ : Adv t₂ t₃ cs₂) :
    Adv t₁ t₃ (cs₁ ++ cs₂) := by
  obtain ⟨r₁, k₁, c₁⟩ := h₁
  obtain ⟨r₂, k₂, c₂⟩ := h₂
  refine ⟨r₂.trans r₁, k₁.trans k₂, ?_⟩
  rw [consumed_append _ k₁ k₂, c₁, ← r₁, c₂]

theorem Adv.step {t t' : Transcript} (hr : t'.rounds = t.rounds) (hk : t'.k = t.k + 1) :
    Adv t t' [(t.rounds[t.k]!).coins] :=
  ⟨hr, by omega, by rw [hk, consumed_succ]⟩

/-! ## Peeling one step -/

variable {α β : Type}

theorem bind_ok {x : TM α} {f : α → TM β} {t t' : Transcript} {b : β} (h : (x >>= f) t = .ok (b, t')) :
    ∃ a t₁, x t = .ok (a, t₁) ∧ f a t₁ = .ok (b, t') := by
  change (StateT.bind x f) t = _ at h
  unfold StateT.bind at h
  change Except.bind (x t) _ = _ at h
  cases hx : x t with
  | error e => rw [hx] at h; cases h
  | ok v => rw [hx] at h; exact ⟨v.1, v.2, rfl, h⟩

theorem pure_ok {a : α} {t t' : Transcript} {a' : α} (h : (pure a : TM α) t = .ok (a', t')) : a' = a ∧ t' = t := by
  cases h; exact ⟨rfl, rfl⟩

theorem modify_bind {g : Transcript → Transcript} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (modify g >>= f) t = .ok (b, t')) : f () (g t) = .ok (b, t') := by
  obtain ⟨_, t₁, h₁, h₂⟩ := bind_ok h
  cases h₁
  exact h₂

/-- A step that keeps the rounds and the round counter. -/
def Keeps (t t' : Transcript) : Prop := t'.k = t.k ∧ t'.rounds = t.rounds

theorem Keeps.adv {t t' : Transcript} (h : Keeps t t') : Adv t t' [] := Adv.of_keeps h.1 h.2

theorem label_bind {l : String} {f : Unit → TM β} {t t' : Transcript} {b : β} (h : (label l >>= f) t = .ok (b, t')) :
    ∃ t₁, Keeps t t₁ ∧ f () t₁ = .ok (b, t') := by
  refine ⟨_, ?_, modify_bind h⟩; exact ⟨rfl, rfl⟩

theorem absorbF_bind {v : F128} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (absorbF v >>= f) t = .ok (b, t')) : ∃ t₁, Keeps t t₁ ∧ f () t₁ = .ok (b, t') := by
  refine ⟨_, ?_, modify_bind h⟩; exact ⟨rfl, rfl⟩

theorem absorbSlice_bind {vs : Array F128} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (absorbSlice vs >>= f) t = .ok (b, t')) : ∃ t₁, Keeps t t₁ ∧ f () t₁ = .ok (b, t') := by
  refine ⟨_, ?_, modify_bind h⟩; exact ⟨rfl, rfl⟩

theorem absorbK_bind {v : F256} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (absorbK v >>= f) t = .ok (b, t')) : ∃ t₁, Keeps t t₁ ∧ f () t₁ = .ok (b, t') := by
  refine ⟨_, ?_, modify_bind h⟩; exact ⟨rfl, rfl⟩

theorem absorbBytes_bind {bs : ByteArray} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (absorbBytes bs >>= f) t = .ok (b, t')) : ∃ t₁, Keeps t t₁ ∧ f () t₁ = .ok (b, t') := by
  refine ⟨_, ?_, modify_bind h⟩; exact ⟨rfl, rfl⟩

theorem pow_bind {nonce : UInt64} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (Transcript.pow nonce >>= f) t = .ok (b, t')) : ∃ t₁, Keeps t t₁ ∧ f () t₁ = .ok (b, t') :=
  absorbBytes_bind h

theorem check_bind {c : Bool} {why : String} {f : Unit → TM β} {t t' : Transcript} {b : β}
    (h : (check c why >>= f) t = .ok (b, t')) : c = true ∧ f () t = .ok (b, t') := by
  obtain ⟨_, t₁, h₁, h₂⟩ := bind_ok h
  unfold check at h₁
  cases c with
  | false => cases h₁
  | true => cases h₁; exact ⟨rfl, h₂⟩

/-- A squeeze of `n ≠ 0` coins consumes round `t.k`, which issued `n` coins, and returns its coins. -/
theorem squeeze_ok {kind : UInt8} {n : ℕ} (hn : n ≠ 0) {t t₁ : Transcript} {cs : Array F128}
    (h : squeeze kind n t = .ok (cs, t₁)) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧ cs = (t.rounds[t.k]!).coins ∧ t₁.rounds = t.rounds ∧
      t₁.k = t.k + 1 := by
  obtain ⟨r, hr, hrn, _, hc, ht⟩ := squeeze_answers kind n t t₁ cs hn h
  have hlt : t.k < t.rounds.size := by
    by_contra hge; rw [Array.getElem?_eq_none (by omega)] at hr; cases hr
  have hget : t.rounds[t.k]! = r := by
    rw [getElem!_pos t.rounds t.k hlt]; exact Option.some.inj ((Array.getElem?_eq_getElem hlt).symm.trans hr)
  subst ht
  exact ⟨hlt, hget ▸ hrn, hget ▸ hc, rfl, rfl⟩

/-- `sampleVec n` then `f`: round `t.k` issued `n` coins, and `f` continues with them. -/
theorem sampleVec_bind {n : ℕ} (hn : n ≠ 0) {f : Array F128 → TM β} {t t' : Transcript} {b : β}
    (h : (sampleVec n >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = n ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f (t.rounds[t.k]!).coins t₁ = .ok (b, t') := by
  obtain ⟨cs, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hcs, hr, hk⟩ := squeeze_ok hn h₁
  exact ⟨hlt, hn', t₁, hr, hk, hcs ▸ h₂⟩

theorem sample_ok {t t₁ : Transcript} {c : F128} (h : sample t = .ok (c, t₁)) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 1 ∧ c = (t.rounds[t.k]!).coins.getD 0 0 ∧ t₁.rounds = t.rounds ∧
      t₁.k = t.k + 1 := by
  obtain ⟨cs, t₂, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hcs, hr, hk⟩ := squeeze_ok Nat.one_ne_zero h₁
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  exact ⟨hlt, hn', hcs ▸ rfl, hr, hk⟩

/-- `sample` then `f`: round `t.k` issued one coin, and `f` continues with it. -/
theorem sample_bind {f : F128 → TM β} {t t' : Transcript} {b : β} (h : (sample >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 1 ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧ f ((t.rounds[t.k]!).coins.getD 0 0) t₁ = .ok (b, t') := by
  obtain ⟨c, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hc, hr, hk⟩ := sample_ok h₁
  exact ⟨hlt, hn', t₁, hr, hk, hc ▸ h₂⟩

theorem sampleK_ok {t t₁ : Transcript} {c : F256} (h : sampleK t = .ok (c, t₁)) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 2 ∧
      c = ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩ ∧ t₁.rounds = t.rounds ∧
      t₁.k = t.k + 1 := by
  obtain ⟨cs, t₂, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hcs, hr, hk⟩ := squeeze_ok (by decide) h₁
  obtain ⟨rfl, rfl⟩ := pure_ok h₂
  exact ⟨hlt, hn', hcs ▸ rfl, hr, hk⟩

/-- `sampleK` then `f`: round `t.k` issued two coins, and `f` continues with them as one `F256`. -/
theorem sampleK_bind {f : F256 → TM β} {t t' : Transcript} {b : β} (h : (sampleK >>= f) t = .ok (b, t')) :
    t.k < t.rounds.size ∧ (t.rounds[t.k]!).n = 2 ∧
      ∃ t₁, t₁.rounds = t.rounds ∧ t₁.k = t.k + 1 ∧
        f ⟨(t.rounds[t.k]!).coins.getD 0 0, (t.rounds[t.k]!).coins.getD 1 0⟩ t₁ = .ok (b, t') := by
  obtain ⟨c, t₁, h₁, h₂⟩ := bind_ok h
  obtain ⟨hlt, hn', hc, hr, hk⟩ := sampleK_ok h₁
  exact ⟨hlt, hn', t₁, hr, hk, hc ▸ h₂⟩

/-! ## Loops over a range, every step yielding -/

/-- A loop `for i in [a:a + n]` whose every step yields: run the body on `i = a, …, a + n − 1`. -/
theorem forIn_range'_cons {σ : Type} (a n : ℕ) (s : σ) (g : ℕ → σ → TM (ForInStep σ)) :
    (forIn (List.range' a (n + 1)) s g : TM σ) =
      g a s >>= fun r => match r with
        | ForInStep.done b => pure b
        | ForInStep.yield b => forIn (List.range' (a + 1) n) b g := by
  rw [List.range'_succ, List.forIn_cons]
  rfl

theorem forIn_range_eq {σ : Type} (n : ℕ) (s : σ) (g : ℕ → σ → TM (ForInStep σ)) :
    (forIn [:n] s g : TM σ) = forIn (List.range' 0 n) s g := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp [Std.Legacy.Range.size]

/-- **A loop consuming one round per step.** If every step of the body consumes the next round, which issued `N`
coins, and yields `step i s c` for that round's coins `c`, then a loop over `a, …, a + n − 1` from round
`base + a` consumes rounds `base + a, …, base + a + n − 1` and returns the fold of `step` over their coins. -/
theorem forIn_rounds {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ) (N : ℕ)
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') →
      (t.rounds[t.k]!).n = N ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧
        b = .yield (step i s (t.rounds[t.k]!).coins)) :
    ∀ (n a base : ℕ) (s : σ) (t t' : Transcript) (out : σ), t.k = base + a →
      forIn (List.range' a n) s body t = .ok (out, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ (∀ i ∈ List.range' a n, (t.rounds[base + i]!).n = N) ∧
      out = (List.range' a n).foldl (fun s i => step i s (t.rounds[base + i]!).coins) s := by
  intro n
  induction n with
  | zero =>
      intro a base s t t' out _ h
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      simp
  | succ n ih =>
      intro a base s t t' out hk h
      rw [forIn_range'_cons] at h
      obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨hN, hr₁, hk₁, rfl⟩ := hbody a s t b t₁ h₁
      obtain ⟨hr, hk', hNs, hout⟩ := ih (a + 1) base _ t₁ t' out (by omega) h₂
      refine ⟨hr.trans hr₁, by omega, ?_, ?_⟩
      · intro i hi
        rw [List.range'_succ, List.mem_cons] at hi
        rcases hi with rfl | hi
        · rw [← hk]; exact hN
        · rw [← hr₁]; exact hNs i hi
      · rw [hout, hr₁, List.range'_succ, List.foldl_cons, hk]

end FlockSoundness.Refine
