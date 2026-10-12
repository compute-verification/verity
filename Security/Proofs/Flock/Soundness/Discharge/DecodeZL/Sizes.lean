import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements

/-!
# The lengths of a decoded proof

`decodeProof_sizes`: a proof `decodeProof` returns has the lengths its `vecExact` checks fix (`ProofSizes`), which make
the message-carrying frames of its rounds well formed.

The reader program is stepped through in the logic of `Post x Q`: every value the reader `x` returns from some position
(`Yields x a`) satisfies `Q`. `post_bind` steps over a bind; `post_bind_vec`, `post_bind_vecBytes`, `post_bind_switches`
and `post_bind_caps` step over a `vecExact`, a vector of digests, the ring switches and the recursive caps' loop, with
their lengths.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Reader
open FlockSoundness.Refine (ebind_ok)

/-- **`a` is a value the reader `x` returns** from some position. -/
def Yields {α : Type} (x : Reader.M α) (a : α) : Prop := ∃ r r', x.run r = .ok (a, r')

/-- **Every value `x` returns satisfies `Q`.** -/
def Post {α : Type} (x : Reader.M α) (Q : α → Prop) : Prop := ∀ a, Yields x a → Q a

theorem yields_bind {α β : Type} {x : Reader.M α} {f : α → Reader.M β} {b : β} (h : Yields (x >>= f) b) :
    ∃ a, Yields x a ∧ Yields (f a) b := by
  obtain ⟨r, r', h⟩ := h
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, r₁⟩, h1, h2⟩ := ebind_ok h
  exact ⟨a, ⟨r, r₁, h1⟩, ⟨r₁, r', h2⟩⟩

theorem post_bind {α β : Type} {x : Reader.M α} {f : α → Reader.M β} {Q : β → Prop}
    (h : ∀ a, Yields x a → Post (f a) Q) : Post (x >>= f) Q := fun b hb => by
  obtain ⟨a, ha, hb⟩ := yields_bind hb
  exact h a ha b hb

theorem yields_pure {α : Type} {a b : α} (h : Yields (pure a : Reader.M α) b) : b = a := by
  obtain ⟨r, r', h⟩ := h
  cases h
  rfl

theorem post_pure {α : Type} {a : α} {Q : α → Prop} (h : Q a) : Post (pure a : Reader.M α) Q := fun b hb => by
  rw [yields_pure hb]
  exact h

theorem yields_throw {α : Type} {e : String} {a : α} (h : Yields (throw e : Reader.M α) a) : False := by
  obtain ⟨r, r', h⟩ := h
  cases h

theorem post_ite {α : Type} {c : Prop} [Decidable c] {t e : Reader.M α} {Q : α → Prop} (ht : c → Post t Q)
    (he : ¬c → Post e Q) : Post (ite c t e) Q := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact ht hc
  · rw [ite_eq_right hc]; exact he hc

theorem yields_forIn_push {α : Type} (g : ℕ → Reader.M α) :
    ∀ (l : List ℕ) (acc out : Array α),
      Yields (forIn l acc fun i (s : Array α) => g i >>= fun x => pure (ForInStep.yield (s.push x))) out →
      ∃ xs : List α, out.toList = acc.toList ++ xs ∧ List.Forall₂ (fun i x => Yields (g i) x) l xs
  | [], acc, out, h => ⟨[], by rw [List.forIn_nil] at h; rw [yields_pure h, List.append_nil], .nil⟩
  | i :: l, acc, out, h => by
      rw [List.forIn_cons] at h
      obtain ⟨st, hst, h⟩ := yields_bind h
      obtain ⟨x, hx, hst⟩ := yields_bind hst
      rw [yields_pure hst] at h
      obtain ⟨xs, hout, hxs⟩ := yields_forIn_push g l _ out h
      refine ⟨x :: xs, ?_, .cons hx hxs⟩
      rw [hout, Array.toList_push, List.append_assoc, List.singleton_append]

theorem forall₂_range' {α : Type} {P : ℕ → α → Prop} :
    ∀ (s n : ℕ) {xs : List α}, List.Forall₂ P (List.range' s n) xs →
      xs.length = n ∧ ∀ (k : ℕ) (hk : k < xs.length), P (s + k) xs[k]
  | _, 0, _, h => by rw [List.range'_zero] at h; cases h; exact ⟨rfl, fun _ hk => absurd hk (Nat.not_lt_zero _)⟩
  | s, n + 1, _, h => by
      rw [List.range'_succ] at h
      cases h with
      | cons hp h =>
        obtain ⟨hl, hk⟩ := forall₂_range' (s + 1) n h
        refine ⟨by rw [List.length_cons, hl], fun k hk' => ?_⟩
        rcases k with _ | k
        · simpa using hp
        · simpa [show s + (k + 1) = s + 1 + k by omega] using hk k (by simpa using hk')

theorem yields_forIn_range {α : Type} (g : ℕ → Reader.M α) (R : ℕ) (acc out : Array α)
    (h : Yields (forIn [:R] acc fun i (s : Array α) => g i >>= fun x => pure (ForInStep.yield (s.push x))) out) :
    out.size = acc.size + R ∧ ∀ k : ℕ, k < R → ∃ hk' : acc.size + k < out.size, Yields (g k) out[acc.size + k] := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  obtain ⟨xs, hout, hxs⟩ := yields_forIn_push g _ acc out h
  have hsz : (List.range' ([:R] : Std.Legacy.Range).start ([:R] : Std.Legacy.Range).size
      ([:R] : Std.Legacy.Range).step) = List.range' 0 R := by simp [Std.Legacy.Range.size]
  rw [hsz] at hxs
  obtain ⟨hl, hk⟩ := forall₂_range' 0 R hxs
  have hs : out.size = acc.size + R := by
    rw [← Array.length_toList, hout, List.length_append, Array.length_toList, hl]
  refine ⟨hs, fun k hkR => ⟨by omega, ?_⟩⟩
  have := hk k (by omega)
  rw [Nat.zero_add] at this
  have e : out[acc.size + k] = xs[k] := by
    rw [← Array.getElem_toList]
    simp only [hout]
    rw [List.getElem_append_right (by simp)]
    simp
  rw [e]
  exact this

theorem yields_vecExact {α : Type} {want : ℕ} {elem : Reader.M α} {w : String} {a : Array α}
    (h : Yields (vecExact want elem w) a) : a.size = want ∧ ∀ e ∈ a, Yields elem e := by
  unfold vecExact at h
  obtain ⟨n, -, h⟩ := yields_bind h
  dsimp only at h
  by_cases hc : (n.toNat != want) = true
  · rw [ite_eq_left hc] at h
    obtain ⟨_, h, -⟩ := yields_bind h
    exact (yields_throw h).elim
  · rw [ite_eq_right hc] at h
    obtain ⟨out, hout, h⟩ := yields_bind h
    rw [yields_pure h]
    obtain ⟨hs, hk⟩ := yields_forIn_range (fun _ => elem) want _ out hout
    have h0 : (Array.mkEmpty want : Array α).size = 0 := rfl
    refine ⟨by rw [hs, h0, Nat.zero_add], fun e he => ?_⟩
    obtain ⟨k, hk', rfl⟩ := Array.mem_iff_getElem.1 he
    obtain ⟨_, hy⟩ := hk k (by omega)
    simpa [h0] using hy

theorem yields_bytes {n : ℕ} {b : ByteArray} (h : Yields (Reader.bytes n) b) : b.size = n := by
  obtain ⟨r, r', h⟩ := h
  unfold Reader.bytes Reader.need at h
  simp at h
  by_cases hc : r.data.size < r.pos + n
  · rw [ite_eq_left hc] at h
    cases h
  · rw [ite_eq_right hc] at h
    simp at h
    obtain ⟨rfl, -⟩ := h
    simp [ByteArray.size_extract]
    omega

theorem yields_vecBytes {want n : ℕ} {w : String} {a : Array ByteArray}
    (h : Yields (vecExact want (Reader.bytes n) w) a) : a.size = want ∧ ∀ e ∈ a, e.size = n :=
  ⟨(yields_vecExact h).1, fun e he => yields_bytes ((yields_vecExact h).2 e he)⟩

/-- **The recursive caps' loop**: `R` caps, cap `j` of `want j` digests of `n` bytes. -/
theorem yields_capsLoop {R n : ℕ} {want : ℕ → ℕ} {w : ℕ → String} {out : Array (Array ByteArray)}
    (h : Yields (forIn [:R] (#[] : Array (Array ByteArray)) fun i (s : Array (Array ByteArray)) =>
      vecExact (want i) (Reader.bytes n) (w i) >>= fun x => pure (ForInStep.yield (s.push x))) out) :
    out.size = R ∧ ∀ (j : ℕ) (hj : j < out.size), out[j].size = want j ∧ ∀ e ∈ out[j], e.size = n := by
  obtain ⟨hs, hk⟩ := yields_forIn_range (fun i => vecExact (want i) (Reader.bytes n) (w i)) R #[] out h
  replace hs : out.size = R := by simpa using hs
  refine ⟨hs, fun j hj => ?_⟩
  obtain ⟨_, hy⟩ := hk j (by omega)
  simpa using yields_vecBytes hy

theorem post_bind_throw {α β : Type} {e : String} {f : α → Reader.M β} {Q : β → Prop} :
    Post ((throw e : Reader.M α) >>= f) Q :=
  post_bind fun _ h => (yields_throw h).elim

theorem post_bind_vec {α β : Type} {want : ℕ} {elem : Reader.M α} {w : String} {f : Array α → Reader.M β}
    {Q : β → Prop} (h : ∀ a, a.size = want → Post (f a) Q) : Post (vecExact want elem w >>= f) Q :=
  post_bind fun a ha => h a (yields_vecExact ha).1

theorem post_bind_vecBytes {β : Type} {want n : ℕ} {w : String} {f : Array ByteArray → Reader.M β} {Q : β → Prop}
    (h : ∀ a, a.size = want → (∀ e ∈ a, e.size = n) → Post (f a) Q) :
    Post (vecExact want (Reader.bytes n) w >>= f) Q :=
  post_bind fun a ha => h a (yields_vecBytes ha).1 (yields_vecBytes ha).2

theorem post_bind_caps {β : Type} {R n : ℕ} {want : ℕ → ℕ} {w : ℕ → String} {f : Array (Array ByteArray) → Reader.M β}
    {Q : β → Prop}
    (h : ∀ a, a.size = R → (∀ (j : ℕ) (hj : j < a.size), a[j].size = want j ∧ ∀ e ∈ a[j], e.size = n) →
      Post (f a) Q) :
    Post ((forIn [:R] (#[] : Array (Array ByteArray)) fun i (s : Array (Array ByteArray)) =>
      vecExact (want i) (Reader.bytes n) (w i) >>= fun x => pure (ForInStep.yield (s.push x))) >>= f) Q :=
  post_bind fun a ha => h a (yields_capsLoop ha).1 (yields_capsLoop ha).2

/-- A ring switch's `s_hat_v` (PROTOCOL.md §8): 128 values, then a zero grinding nonce. -/
theorem post_switch :
    Post (do
        let v ← vecExact 128 f128 "s_hat_v"
        let nonce ← u64
        if nonce != 0 then throw "ring switch grinding_nonce ≠ 0"
        pure v : Reader.M (Array F128)) (fun v => v.size = 128) :=
  post_bind_vec fun _ hv => post_bind fun _ _ =>
    post_ite (fun _ => post_bind_throw) (fun _ => post_pure hv)

theorem post_bind_switches {β : Type} {want : ℕ} {w : String} {f : Array (Array F128) → Reader.M β} {Q : β → Prop}
    (h : ∀ a, a.size = want → (∀ v ∈ a, v.size = 128) → Post (f a) Q) :
    Post (vecExact want (do
        let v ← vecExact 128 f128 "s_hat_v"
        let nonce ← u64
        if nonce != 0 then throw "ring switch grinding_nonce ≠ 0"
        pure v) w >>= f) Q :=
  post_bind fun a ha => h a (yields_vecExact ha).1 fun v hv => post_switch v ((yields_vecExact ha).2 v hv)

/-- **A decoded proof has the lengths its shape fixes**: stepping through `decodeProof`'s reader program, each
`vecExact` it returns has its length, the caps hold `hashLen`-byte digests and the ring switches 128 values. -/
theorem decodeProof_sizes : DecodeSizes := by
  intro sh bytes p h
  unfold decodeProof at h
  obtain ⟨⟨p', r'⟩, hrun, hp⟩ := ebind_ok h
  have hp' : p' = p := Except.ok.inj hp
  rw [← hp']
  refine (?_ : Post _ (ProofSizes sh)) p' ⟨_, _, hrun⟩
  dsimp only
  refine post_bind_vecBytes fun cap hcap hcapE => ?_
  refine post_bind fun m _ => post_bind fun lir _ => post_bind fun lbs _ => post_bind fun profile _ => ?_
  refine post_bind fun tag _ => ?_
  split
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun numLanes _ => post_bind fun mh _ => ?_
    refine post_bind_vec fun r1ab hr1ab => post_bind_vec fun r1c hr1c => post_bind_vec fun rounds hrounds => ?_
    refine post_bind fun fa _ => post_bind fun fb _ => post_bind fun fc _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun lr hlr => post_bind_vec fun zp hzp => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_switches fun rs hrs hrsE => post_bind fun _ _ => ?_
    refine post_bind_vecBytes fun icap _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun n _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind_caps fun rcaps hrc hrcE => ?_
    refine post_bind fun np _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun rproofs _ => post_bind_vec fun yr hyr => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun msgs hmsgs => post_bind_vec fun qn hqn => post_bind_vec fun ood hood => ?_
    refine post_bind fun _ _ => post_bind_vec fun cbn hcbn => post_bind_vec fun con hcon => ?_
    exact post_pure ⟨hr1ab, hr1c, hrounds, hlr, hzp, hrs, hrsE, hcap, hcapE, hrc, hrcE, hyr, hmsgs, hqn, hood, hcbn,
      hcon⟩

end FlockSoundness.Discharge.DecodeZL
