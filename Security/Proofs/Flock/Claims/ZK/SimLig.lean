import Proofs.Flock.Claims.ZK.Split

/-!
# A rep's Ligerito opening from the cut (`SimLig`)

`SimLig` (`Split`) compares `repMix 1`, whose Ligerito opening reads level 0's masked messages, `y₁'`, `ȳ` and levels
≥ 1's salts from the cut (`ligeritoCut`), with `repMix 2`, the session's (`ligeritoPZk`).

* **The salt roundtrip** (`blobSalt_saltBlob`): level `l ≥ 1`'s salt at position `p`, read back from the salts' string,
  is the salt fitted to 192 bytes; a 192-byte salt comes back whole (`fit192_of_size`).
-/

namespace Claims.ZKMal.Masked

namespace LigPf

/-! ## The salt roundtrip -/

theorem fit192_size (b : ByteArray) : (fit192 b).size = 192 := by
  unfold fit192
  rw [ByteArray.size_append, ByteArray.size_extract]
  show _ + (Array.replicate (192 - b.size) (0 : UInt8)).size = 192
  rw [Array.size_replicate]
  omega

theorem fit192_of_size {b : ByteArray} (h : b.size = 192) : fit192 b = b := by
  unfold fit192
  rw [h, Nat.sub_self, ← h, ByteArray.extract_zero_size]
  exact ByteArray.append_empty

/-- Level `l`'s salts, fitted, one after another. -/
def levelBlob (s : ℕ → ℕ → ByteArray) (l N : ℕ) : ByteArray :=
  (List.range N).foldl (fun acc p => acc ++ fit192 (s l p)) ByteArray.empty

theorem foldl_app_eq {α : Type} (g : α → ByteArray) :
    ∀ (xs : List α) (acc : ByteArray),
      xs.foldl (fun acc a => acc ++ g a) acc = acc ++ xs.foldl (fun acc a => acc ++ g a) ByteArray.empty
  | [], acc => by simp [ByteArray.append_empty]
  | a :: xs, acc => by
    simp only [List.foldl_cons]
    rw [foldl_app_eq g xs (acc ++ g a), foldl_app_eq g xs (ByteArray.empty ++ g a), ByteArray.empty_append,
      ByteArray.append_assoc]

theorem levelBlob_succ (s : ℕ → ℕ → ByteArray) (l N : ℕ) :
    levelBlob s l (N + 1) = levelBlob s l N ++ fit192 (s l N) := by
  unfold levelBlob
  rw [List.range_succ, List.foldl_append]
  rfl

theorem levelBlob_size (s : ℕ → ℕ → ByteArray) (l : ℕ) : ∀ N, (levelBlob s l N).size = 192 * N
  | 0 => rfl
  | N + 1 => by
    rw [levelBlob_succ, ByteArray.size_append, levelBlob_size s l N, fit192_size]
    ring

theorem extract_append_left_of_le {a b : ByteArray} {i j : ℕ} (hj : j ≤ a.size) :
    (a ++ b).extract i j = a.extract i j := by
  rw [ByteArray.extract_append, show j - a.size = 0 by omega,
    show b.extract (i - a.size) 0 = ByteArray.empty from ByteArray.extract_eq_empty_iff.2 (by omega),
    ByteArray.append_empty]

theorem levelBlob_extract (s : ℕ → ℕ → ByteArray) (l : ℕ) :
    ∀ N p, p < N → (levelBlob s l N).extract (192 * p) (192 * (p + 1)) = fit192 (s l p)
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | N + 1, p, h => by
    rw [levelBlob_succ]
    by_cases hp : p = N
    · subst hp
      refine ByteArray.extract_append_eq_right (by rw [levelBlob_size]) ?_
      rw [levelBlob_size, fit192_size]
      ring
    · rw [extract_append_left_of_le (by rw [levelBlob_size]; nlinarith [show p + 1 ≤ N by omega])]
      exact levelBlob_extract s l N p (by omega)

/-- The salts' string through level `k`. -/
def blobUpTo (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) (k : ℕ) : ByteArray :=
  (List.range' 1 k).foldl (fun acc l =>
    (List.range (2 ^ (ls.getD l default).logLen)).foldl (fun acc p => acc ++ fit192 (s l p)) acc) ByteArray.empty

theorem blobUpTo_succ (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) (k : ℕ) :
    blobUpTo ls s (k + 1) = blobUpTo ls s k ++ levelBlob s (1 + k) (2 ^ (ls.getD (1 + k) default).logLen) := by
  unfold blobUpTo
  rw [List.range'_1_concat, List.foldl_append]
  simp only [List.foldl_cons, List.foldl_nil]
  exact foldl_app_eq (fun p => fit192 (s (1 + k) p)) _ _

theorem blobUpTo_size (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) :
    ∀ k, k + 1 ≤ ls.length → (blobUpTo ls s k).size = 192 * hiOff ls (k + 1)
  | 0, _ => by simp [blobUpTo, hiOff, ByteArray.size_empty]
  | k + 1, h => by
    rw [blobUpTo_succ, ByteArray.size_append, blobUpTo_size ls s k (by omega), levelBlob_size,
      hiOff_succ ls (show 1 ≤ k + 1 by omega) (by omega), show 1 + k = k + 1 by omega]
    ring

theorem blobUpTo_prefix (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) (k : ℕ) :
    ∀ d, ∃ rest, blobUpTo ls s (k + d) = blobUpTo ls s k ++ rest
  | 0 => ⟨ByteArray.empty, by rw [Nat.add_zero, ByteArray.append_empty]⟩
  | d + 1 => by
    obtain ⟨rest, hX⟩ := blobUpTo_prefix ls s k d
    exact ⟨rest ++ levelBlob s (1 + (k + d)) (2 ^ (ls.getD (1 + (k + d)) default).logLen), by
      rw [← Nat.add_assoc, blobUpTo_succ, hX, ByteArray.append_assoc]⟩

theorem saltBlob_eq (ls : List FlockSoundness.Accounting.Level) (s : ℕ → ℕ → ByteArray) :
    saltBlob ls s = blobUpTo ls s (ls.length - 1) := rfl

/-- **The salt roundtrip**: level `l ≥ 1`'s salt at position `p` of its tree, read back from the salts' string, is the
salt fitted to 192 bytes. -/
theorem blobSalt_saltBlob {ls : List FlockSoundness.Accounting.Level} {s : ℕ → ℕ → ByteArray} {l p : ℕ}
    (h : HiAt ls l p) : blobSalt ls (saltBlob ls s) l p = fit192 (s l p) := by
  obtain ⟨h1, hl, hp⟩ := h
  unfold blobSalt
  rw [ite_eq_left_of_eq_true _ _ (eq_true (show HiAt ls l p from ⟨h1, hl, hp⟩)), saltBlob_eq]
  obtain ⟨rest, hX⟩ := blobUpTo_prefix ls s l (ls.length - 1 - l)
  rw [show ls.length - 1 = l + (ls.length - 1 - l) by omega, hX]
  have hl' : l = (l - 1) + 1 := by omega
  have hsz : (blobUpTo ls s (l - 1)).size = 192 * hiOff ls l := by
    rw [blobUpTo_size ls s (l - 1) (by omega), ← hl']
  have hU : blobUpTo ls s l = blobUpTo ls s (l - 1) ++ levelBlob s l (2 ^ (ls.getD l default).logLen) := by
    conv_lhs => rw [hl']
    rw [blobUpTo_succ, show 1 + (l - 1) = l by omega]
  have hUsz : (blobUpTo ls s l).size = 192 * hiOff ls l + 192 * 2 ^ (ls.getD l default).logLen := by
    rw [hU, ByteArray.size_append, hsz, levelBlob_size]
  rw [extract_append_left_of_le (by rw [hUsz]; nlinarith), hU,
    show 192 * (hiOff ls l + p) = 192 * hiOff ls l + 192 * p by ring,
    show 192 * (hiOff ls l + p + 1) = 192 * hiOff ls l + 192 * (p + 1) by ring,
    ByteArray.extract_append_size_add' hsz.symm]
  exact levelBlob_extract s l _ p hp

/-- A 192-byte salt comes back whole. -/
theorem blobSalt_saltBlob_of_size {ls : List FlockSoundness.Accounting.Level} {s : ℕ → ℕ → ByteArray} {l p : ℕ}
    (h : HiAt ls l p) (hs : (s l p).size = 192) : blobSalt ls (saltBlob ls s) l p = s l p := by
  rw [blobSalt_saltBlob h, fit192_of_size hs]

end LigPf

end Claims.ZKMal.Masked

namespace FlockComplete

namespace LigLock

open Flock Transcript Replay

/-! ## Lockstep

The two openings are compared in lockstep, as `SimZerocheck`'s `GoAlike` compares the zerocheck: both threw the same
error after the same bodies, or both returned at the same transcript after the same bodies, with related values. -/

/-- **Two runs in lockstep.** -/
abbrev Alike {α β : Type} (R : α → β → Array Claims.Msg → Prop)
    (x : Except String (α × Transcript) × Array Claims.Msg) (y : Except String (β × Transcript) × Array Claims.Msg) :
    Prop :=
  (∃ e L, x = (.error e, L) ∧ y = (.error e, L)) ∨ ∃ a b t L, x = (.ok (a, t), L) ∧ y = (.ok (b, t), L) ∧ R a b L

theorem bind_go_err {α β : Type} {x : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l l' : Array Claims.Msg}
    {e : String} (h : (x.run t).go l = (.error e, l')) : ((x >>= f).run t).go l = (.error e, l') := by
  rw [StateT.run_bind, go_bind, h]

theorem Alike.refl {α : Type} (x : Except String (α × Transcript) × Array Claims.Msg) :
    Alike (fun a b _ => a = b) x x := by
  rcases x with ⟨r, L⟩
  rcases r with e | ⟨a, t⟩
  · exact .inl ⟨e, L, rfl, rfl⟩
  · exact .inr ⟨a, a, t, L, rfl, rfl, rfl⟩

theorem Alike.eq {α : Type} {x y : Except String (α × Transcript) × Array Claims.Msg}
    (h : Alike (fun a b _ => a = b) x y) : x = y := by
  rcases h with ⟨e, L, rfl, rfl⟩ | ⟨a, b, t, L, rfl, rfl, hab⟩
  · rfl
  · obtain rfl : a = b := hab
    rfl

theorem Alike.mono {α β : Type} {R S : α → β → Array Claims.Msg → Prop}
    {x : Except String (α × Transcript) × Array Claims.Msg} {y : Except String (β × Transcript) × Array Claims.Msg}
    (h : Alike R x y) (hRS : ∀ a b L, R a b L → S a b L) : Alike S x y := by
  rcases h with ⟨e, L, hx, hy⟩ | ⟨a, b, t, L, hx, hy, hR⟩
  · exact .inl ⟨e, L, hx, hy⟩
  · exact .inr ⟨a, b, t, L, hx, hy, hRS a b L hR⟩

/-- **A bind in lockstep**: related programs, and continuations related from every common return. -/
theorem alike_bind {α β γ δ : Type} {R : α → β → Array Claims.Msg → Prop} {S : γ → δ → Array Claims.Msg → Prop}
    {x : TMW PS α} {y : TMW PS β} {f : α → TMW PS γ} {g : β → TMW PS δ} {t : Transcript} {l : Array Claims.Msg}
    (hxy : Alike R ((x.run t).go l) ((y.run t).go l))
    (hfg : ∀ a b t' l', (x.run t).go l = (.ok (a, t'), l') → (y.run t).go l = (.ok (b, t'), l') → R a b l' →
      Alike S (((f a).run t').go l') (((g b).run t').go l')) :
    Alike S (((x >>= f).run t).go l) (((y >>= g).run t).go l) := by
  rcases hxy with ⟨e, L, hx, hy⟩ | ⟨a, b, t', L, hx, hy, hR⟩
  · exact .inl ⟨e, L, bind_go_err hx, bind_go_err hy⟩
  · rw [bind_go hx, bind_go hy]
    exact hfg a b t' L hx hy hR

/-- **A shared step**: the same program, then continuations related from each of its returns. -/
theorem alike_bind_eq {α γ δ : Type} {S : γ → δ → Array Claims.Msg → Prop} {x : TMW PS α}
    {f : α → TMW PS γ} {g : α → TMW PS δ} {t : Transcript} {l : Array Claims.Msg}
    (hfg : ∀ a t' l', (x.run t).go l = (.ok (a, t'), l') → Alike S (((f a).run t').go l') (((g a).run t').go l')) :
    Alike S (((x >>= f).run t).go l) (((x >>= g).run t).go l) :=
  alike_bind (Alike.refl _) fun a b t' l' hx _ (hab : a = b) => by subst hab; exact hfg a t' l' hx

/-! ## `LP`'s steps in lockstep -/

/-- **An `LP` bind in lockstep.** -/
theorem lbind {α β γ δ : Type} {R : α × LigState → β × LigState → Array Claims.Msg → Prop}
    {S : γ × LigState → δ × LigState → Array Claims.Msg → Prop} {x : LP PS α} {y : LP PS β} {f : α → LP PS γ}
    {g : β → LP PS δ} {s s' : LigState} {t : Transcript} {l : Array Claims.Msg}
    (hxy : Alike R (((x.run s).run t).go l) (((y.run s').run t).go l))
    (hfg : ∀ a s₁ b s₁' t' l', ((x.run s).run t).go l = (.ok ((a, s₁), t'), l') →
      ((y.run s').run t).go l = (.ok ((b, s₁'), t'), l') → R (a, s₁) (b, s₁') l' →
      Alike S ((((f a).run s₁).run t').go l') ((((g b).run s₁').run t').go l')) :
    Alike S ((((x >>= f).run s).run t).go l) ((((y >>= g).run s').run t).go l) := by
  rw [StateT.run_bind, StateT.run_bind]
  exact alike_bind (f := fun p => (f p.1).run p.2) (g := fun p => (g p.1).run p.2) hxy
    fun ⟨a, s₁⟩ ⟨b, s₁'⟩ t' l' hx hy hR => hfg a s₁ b s₁' t' l' hx hy hR

theorem lpure {α β : Type} {R : α × LigState → β × LigState → Array Claims.Msg → Prop} {a : α} {b : β}
    {s s' : LigState} {t : Transcript} {l : Array Claims.Msg} (h : R (a, s) (b, s') l) :
    Alike R ((((pure a : LP PS α).run s).run t).go l) ((((pure b : LP PS β).run s').run t).go l) :=
  .inr ⟨(a, s), (b, s'), t, l, rfl, rfl, h⟩

theorem liftT_bind_run {α γ : Type} (m : TMW PS α) (f : α → LP PS γ) (s : LigState) :
    (liftT m >>= f).run s = m >>= fun a => (f a).run s := by
  simp only [StateT.run_bind, liftT, StateT.run_lift, bind_assoc, pure_bind]

/-- **A shared transcript step** from any two states. -/
theorem lliftT {α γ δ : Type} {S : γ × LigState → δ × LigState → Array Claims.Msg → Prop} {m : TMW PS α}
    {f : α → LP PS γ} {g : α → LP PS δ} {s s' : LigState} {t : Transcript} {l : Array Claims.Msg}
    (h : ∀ a t' l', (m.run t).go l = (.ok (a, t'), l') →
      Alike S ((((f a).run s).run t').go l') ((((g a).run s').run t').go l')) :
    Alike S ((((liftT m >>= f).run s).run t).go l) ((((liftT m >>= g).run s').run t).go l) := by
  rw [liftT_bind_run, liftT_bind_run]
  exact alike_bind_eq h

theorem modify_bind_run {γ : Type} (F : LigState → LigState) (f : PUnit → LP PS γ) (s : LigState) :
    (modify F >>= f).run s = (f ()).run (F s) := by
  simp only [StateT.run_bind, StateT.run_modify, pure_bind]

theorem get_bind_run {γ : Type} (f : LigState → LP PS γ) (s : LigState) :
    (get >>= f).run s = (f s).run s := by
  simp only [StateT.run_bind, StateT.run_get, pure_bind]

theorem set_bind_run {γ : Type} (s₁ : LigState) (f : PUnit → LP PS γ) (s : LigState) :
    (set s₁ >>= f).run s = (f ()).run s₁ := by
  simp only [StateT.run_bind, StateT.run_set, pure_bind]

/-- **A shared `LP` step** from one state. -/
theorem lbind_eq {α γ δ : Type} {S : γ × LigState → δ × LigState → Array Claims.Msg → Prop} {x : LP PS α}
    {f : α → LP PS γ} {g : α → LP PS δ} {s : LigState} {t : Transcript} {l : Array Claims.Msg}
    (h : ∀ a s₁ t' l', ((x.run s).run t).go l = (.ok ((a, s₁), t'), l') →
      Alike S ((((f a).run s₁).run t').go l') ((((g a).run s₁).run t').go l')) :
    Alike S ((((x >>= f).run s).run t).go l) ((((x >>= g).run s).run t).go l) :=
  lbind (Alike.refl _) fun a s₁ b s₁' t' l' hx _ hR => by
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hR
    exact h a s₁ t' l' hx

/-- **A loop in lockstep**: if step `i`, from values and states related by `I i` with `L + i` bodies sent, yields ones
related by `I (i + 1)` after one more body, the loops over `[0:n]` end related by `I n` after `L + n` bodies. -/
theorem lforIn {σ τ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (body' : ℕ → τ → LP PS (ForInStep τ))
    (I : ℕ → σ × LigState → τ × LigState → Prop) (L n : ℕ)
    (hb : ∀ i < n, ∀ (u : σ) (s : LigState) (u' : τ) (s' : LigState) (t : Transcript) (l : Array Claims.Msg),
      l.size = L + i → I i (u, s) (u', s') →
      Alike (fun r r' l₁ => ∃ v v', r.1 = .yield v ∧ r'.1 = .yield v' ∧ I (i + 1) (v, r.2) (v', r'.2) ∧
          l₁.size = L + i + 1)
        ((((body i u).run s).run t).go l) ((((body' i u').run s').run t).go l))
    {u : σ} {u' : τ} {s s' : LigState} (t : Transcript) (l : Array Claims.Msg) (hl : l.size = L)
    (h₀ : I 0 (u, s) (u', s')) :
    Alike (fun r r' l₁ => I n r r' ∧ l₁.size = L + n)
      ((((forIn [:n] u body).run s).run t).go l) ((((forIn [:n] u' body').run s').run t).go l) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (s : LigState) (u' : τ) (s' : LigState) (t : Transcript) (l : Array Claims.Msg),
      a + k = n → l.size = L + a → I a (u, s) (u', s') →
      Alike (fun r r' l₁ => I n r r' ∧ l₁.size = L + n)
        ((((forIn (List.range' a k) u body).run s).run t).go l)
        ((((forIn (List.range' a k) u' body').run s').run t).go l) from
    key n 0 u s u' s' t l (by omega) (by omega) h₀
  intro k
  induction k with
  | zero =>
    intro a u s u' s' t l ha hl hI
    have e : a = n := by omega
    exact .inr ⟨(u, s), (u', s'), t, l, rfl, rfl, e ▸ hI, by omega⟩
  | succ k ih =>
    intro a u s u' s' t l ha hl hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine lbind (hb a (by omega) u s u' s' t l hl hI) fun r s₁ r' s₁' t₁ l₁ _ _ hr => ?_
    obtain ⟨v, v', hv, hv', hI₁, hl₁⟩ := hr
    dsimp only at hv hv'
    subst hv hv'
    exact ih (a + 1) v s₁ v' s₁' t₁ l₁ (by omega) (by omega) hI₁

/-! ## `asksCut`'s squeezes -/

theorem asksCut_round_go (coin : ℕ → ℕ → F128) (n : ℕ) (c : ByteArray) (l : Array Claims.Msg) :
    ((asksCut coin n).round c).go l =
      ((if n ≤ l.size then throw "past the openings" else pure (coin l.size) : PS (ℕ → F128)).go
        (l.push (Claims.toMsg c))) := rfl

theorem squeezeP_cut_ok (coin : ℕ → ℕ → F128) {n : ℕ} (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript)
    {l : Array Claims.Msg} (hl : l.size < n) :
    ((squeezeP (asksCut coin n) kind k).run t).go l =
      (.ok ((Array.range k).map (coin l.size), afterSq t kind k (coin l.size)),
        l.push (Claims.toMsg (framed t kind k))) := by
  simp only [squeezeP, StateT.run, hk, ite_false]
  rw [go_bind, asksCut_round_go, ite_eq_right (show ¬ n ≤ l.size by omega)]
  rfl

theorem squeezeP_cut_err (coin : ℕ → ℕ → F128) {n : ℕ} (kind : UInt8) {k : ℕ} (hk : k ≠ 0) (t : Transcript)
    {l : Array Claims.Msg} (hl : n ≤ l.size) :
    ((squeezeP (asksCut coin n) kind k).run t).go l =
      (.error "past the openings", l.push (Claims.toMsg (framed t kind k))) := by
  simp only [squeezeP, StateT.run, hk, ite_false]
  rw [go_bind, asksCut_round_go, ite_eq_left hl]
  rfl

/-- **A squeeze that returned** was before round `n`, returned the coins at `l.size` and sent one body. -/
theorem squeezeP_cut_inv (coin : ℕ → ℕ → F128) {n : ℕ} (kind : UInt8) {k : ℕ} (hk : k ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} {a : Array F128}
    (h : ((squeezeP (asksCut coin n) kind k).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = (Array.range k).map (coin l.size) ∧ l'.size = l.size + 1 := by
  by_cases hn : n ≤ l.size
  · rw [squeezeP_cut_err coin kind hk t hn] at h
    simp at h
  · rw [squeezeP_cut_ok coin kind hk t (by omega)] at h
    simp only [Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨by omega, rfl, by simp⟩

theorem sampleVecP_cut_inv (coin : ℕ → ℕ → F128) {n k : ℕ} (hk : k ≠ 0) {t t' : Transcript}
    {l l' : Array Claims.Msg} {a : Array F128} (h : ((sampleVecP (asksCut coin n) k).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = (Array.range k).map (coin l.size) ∧ l'.size = l.size + 1 :=
  squeezeP_cut_inv coin 2 hk h

theorem sampleP_cut_inv (coin : ℕ → ℕ → F128) {n : ℕ} {t t' : Transcript} {l l' : Array Claims.Msg} {a : F128}
    (h : ((sampleP (asksCut coin n)).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = coin l.size 0 ∧ l'.size = l.size + 1 := by
  unfold sampleP at h
  by_cases hn : n ≤ l.size
  · rw [bind_go_err (squeezeP_cut_err coin 1 one_ne_zero t hn)] at h
    simp at h
  · rw [bind_go (squeezeP_cut_ok coin 1 one_ne_zero t (by omega))] at h
    simp only [pure_go, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨by omega, by simp, by simp⟩

theorem sampleKP_cut_inv (coin : ℕ → ℕ → F128) {n : ℕ} {t t' : Transcript} {l l' : Array Claims.Msg} {a : F256}
    (h : ((sampleKP (asksCut coin n)).run t).go l = (.ok (a, t'), l')) :
    l.size < n ∧ a = ⟨coin l.size 0, coin l.size 1⟩ ∧ l'.size = l.size + 1 := by
  unfold sampleKP at h
  by_cases hn : n ≤ l.size
  · rw [bind_go_err (squeezeP_cut_err coin 2 two_ne_zero t hn)] at h
    simp at h
  · rw [bind_go (squeezeP_cut_ok coin 2 two_ne_zero t (by omega))] at h
    simp only [pure_go, Prod.mk.injEq, Except.ok.injEq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨by omega, by simp, by simp⟩

/-- A transcript action sends nothing. -/
theorem liftTM_inv {α : Type} {x : TM α} {t t' : Transcript} {l l' : Array Claims.Msg} {a : α}
    (h : ((liftTM x : TMW PS α).run t).go l = (.ok (a, t'), l')) : l = l' :=
  congrArg Prod.snd ((liftTM_go x t l).symm.trans h)

end LigLock

end FlockComplete
