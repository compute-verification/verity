import Proofs.Flock.Claims.ZK.Split

/-!
# A rep's Ligerito opening from the cut (`SimLig`)

`SimLig` (`Split`) compares `repMix 1`, whose Ligerito opening reads level 0's masked messages, `y₁'`, `ȳ` and levels
≥ 1's salts from the cut (`ligeritoCut`), with `repMix 2`, the session's (`ligeritoPZk`).

* **The salt roundtrip** (`blobSalt_saltBlob`): level `l ≥ 1`'s salt at position `p`, read back from the salts' string,
  is the salt fitted to 192 bytes; a 192-byte salt comes back whole (`fit192_of_size`).
* **Lockstep** (`ligeritoCut_alike`): where the source holds what the session's opening computes at the coins from
  round `p₀` on (`LigHyp`), the two openings send the same and return the same erased proof; `simLig_of` from that at
  the deployed rep (`LigSrcE`).
* **The PIOP answered by `asksCut`** (`cutS`, `piopPZk_cut_ok`, `piopPZk_exact`): where it returns, it is the run at
  `asksS` with replies reading as the coins, so it sent `piopZk` bodies and returned the closed form's switched vectors.
* **At the deployed rep's coins** (`ligSrcE_of`, `ligHypE_of`): `LigSrcE` from the source's values at round
  `repSlot … 0` and the basis at the coins (`LigHypE`), and that from the schedule's facts and six value statements
  (`LigOodE`, `LigMsgE`, `LigSumsE`, `LigY1E`, `LigYbarE`, `LigSaltE`; `simLig_of_fields`).
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

theorem sendMsg_bind_run {γ : Type} (m : F256 × F256) (f : Unit → LP PS γ) (s : LigState) :
    (sendMsg m >>= f).run s =
      (liftT (liftTM (absorbK m.1)) >>= fun _ => liftT (liftTM (absorbK m.2)) >>= f).run
        { s with msgs := s.msgs.push m } := by
  simp only [sendMsg, bind_assoc, modify_bind_run]

theorem sendMsg_run (m : F256 × F256) (s : LigState) :
    (sendMsg m : LP PS Unit).run s =
      (liftT (liftTM (absorbK m.1)) >>= fun _ => liftT (liftTM (absorbK m.2))).run
        { s with msgs := s.msgs.push m } := by
  simp only [sendMsg, modify_bind_run]

theorem powP_bind_run {γ : Type} (f : Unit → LP PS γ) (s : LigState) :
    (powP (fun _ _ => 0) 2 >>= f).run s = (liftT (liftTM (pow 0)) >>= f).run { s with cbn := s.cbn.push 0 } := by
  simp only [powP, bind_assoc, get_bind_run, set_bind_run]

theorem throw_go {α : Type} (e : String) (s : LigState) (t : Transcript) (l : Array Claims.Msg) :
    ((((throw e : LP PS α)).run s).run t).go l = (.error e, l) := rfl

/-- **A shared `LP` step** from one state. -/
theorem lbind_eq {α γ δ : Type} {S : γ × LigState → δ × LigState → Array Claims.Msg → Prop} {x : LP PS α}
    {f : α → LP PS γ} {g : α → LP PS δ} {s : LigState} {t : Transcript} {l : Array Claims.Msg}
    (h : ∀ a s₁ t' l', ((x.run s).run t).go l = (.ok ((a, s₁), t'), l') →
      Alike S ((((f a).run s₁).run t').go l') ((((g a).run s₁).run t').go l')) :
    Alike S ((((x >>= f).run s).run t).go l) ((((x >>= g).run s).run t).go l) :=
  lbind (Alike.refl _) fun a s₁ b s₁' t' l' hx _ hR => by
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hR
    exact h a s₁ t' l' hx

/-- **A loop in lockstep**: if step `i`, from values, states and bodies related by `I i`, yields ones related by
`I (i + 1)`, the loops over `[0:n]` end related by `I n`. -/
theorem lforIn {σ τ : Type} (body : ℕ → σ → LP PS (ForInStep σ)) (body' : ℕ → τ → LP PS (ForInStep τ))
    (I : ℕ → σ × LigState → τ × LigState → Array Claims.Msg → Prop) (n : ℕ)
    (hb : ∀ i < n, ∀ (u : σ) (s : LigState) (u' : τ) (s' : LigState) (t : Transcript) (l : Array Claims.Msg),
      I i (u, s) (u', s') l →
      Alike (fun r r' l₁ => ∃ v v', r.1 = .yield v ∧ r'.1 = .yield v' ∧ I (i + 1) (v, r.2) (v', r'.2) l₁)
        ((((body i u).run s).run t).go l) ((((body' i u').run s').run t).go l))
    {u : σ} {u' : τ} {s s' : LigState} (t : Transcript) (l : Array Claims.Msg) (h₀ : I 0 (u, s) (u', s') l) :
    Alike (fun r r' l₁ => I n r r' l₁)
      ((((forIn [:n] u body).run s).run t).go l) ((((forIn [:n] u' body').run s').run t).go l) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (s : LigState) (u' : τ) (s' : LigState) (t : Transcript) (l : Array Claims.Msg),
      a + k = n → I a (u, s) (u', s') l →
      Alike (fun r r' l₁ => I n r r' l₁)
        ((((forIn (List.range' a k) u body).run s).run t).go l)
        ((((forIn (List.range' a k) u' body').run s').run t).go l) from
    key n 0 u s u' s' t l (by omega) h₀
  intro k
  induction k with
  | zero =>
    intro a u s u' s' t l ha hI
    have e : a = n := by omega
    exact .inr ⟨(u, s), (u', s'), t, l, rfl, rfl, e ▸ hI⟩
  | succ k ih =>
    intro a u s u' s' t l ha hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine lbind (hb a (by omega) u s u' s' t l hI) fun r s₁ r' s₁' t₁ l₁ _ _ hr => ?_
    obtain ⟨v, v', hv, hv', hI₁⟩ := hr
    dsimp only at hv hv'
    subst hv hv'
    exact ih (a + 1) v s₁ v' s₁' t₁ l₁ (by omega) hI₁

/-- **A loop of equal steps**: bodies alike to equality from every value and state run the loops alike to equality. -/
theorem lforIn_eq {σ : Type} (body body' : ℕ → σ → LP PS (ForInStep σ)) (n : ℕ)
    (hb : ∀ i < n, ∀ (u : σ) (s : LigState) (t : Transcript) (l : Array Claims.Msg),
      Alike (fun a b _ => a = b) ((((body i u).run s).run t).go l) ((((body' i u).run s).run t).go l))
    (u : σ) (s : LigState) (t : Transcript) (l : Array Claims.Msg) :
    Alike (fun a b _ => a = b)
      ((((forIn [:n] u body).run s).run t).go l) ((((forIn [:n] u body').run s).run t).go l) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (s : LigState) (t : Transcript) (l : Array Claims.Msg), a + k = n →
      Alike (fun a b _ => a = b)
        ((((forIn (List.range' a k) u body).run s).run t).go l)
        ((((forIn (List.range' a k) u body').run s).run t).go l) from
    key n 0 u s t l (by omega)
  intro k
  induction k with
  | zero => intro a u s t l _; exact Alike.refl _
  | succ k ih =>
    intro a u s t l ha
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine lbind (hb a (by omega) u s t l) fun r s₁ r' s₁' t₁ l₁ _ _ hr => ?_
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj hr
    rcases r with v | v
    · exact Alike.refl _
    · exact ih (a + 1) v s₁ t₁ l₁ (by omega)

/-- Equal programs run alike to equality. -/
theorem Alike.of_eq {α : Type} {x y : LP PS α} (h : x = y) (s : LigState) (t : Transcript) (l : Array Claims.Msg) :
    Alike (fun a b _ => a = b) (((x.run s).run t).go l) (((y.run s).run t).go l) := h ▸ Alike.refl _

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

theorem alike_bind_assoc {α β γ δ ε : Type} {R : α → β → Array Claims.Msg → Prop}
    {S : γ → ε → Array Claims.Msg → Prop} {x : TMW PS α} {y : TMW PS β} {f : α → TMW PS γ} {g : β → TMW PS δ}
    {C : δ → TMW PS ε} {t : Transcript} {l : Array Claims.Msg}
    (hxy : Alike R ((x.run t).go l) ((y.run t).go l))
    (hfg : ∀ a b t' l', (x.run t).go l = (.ok (a, t'), l') → (y.run t).go l = (.ok (b, t'), l') → R a b l' →
      Alike S (((f a).run t').go l') (((g b >>= C).run t').go l')) :
    Alike S (((x >>= f).run t).go l) ((((y >>= g) >>= C).run t).go l) := by
  rw [bind_assoc]
  exact alike_bind hxy hfg

theorem alike_bind_eq_assoc {α γ δ ε : Type} {S : γ → ε → Array Claims.Msg → Prop} {x : TMW PS α}
    {f : α → TMW PS γ} {g : α → TMW PS δ} {C : δ → TMW PS ε} {t : Transcript} {l : Array Claims.Msg}
    (hfg : ∀ a t' l', (x.run t).go l = (.ok (a, t'), l') → Alike S (((f a).run t').go l') (((g a >>= C).run t').go l')) :
    Alike S (((x >>= f).run t).go l) ((((x >>= g) >>= C).run t).go l) :=
  alike_bind_assoc (Alike.refl _) fun a b t' l' hx _ (hab : a = b) => by subst hab; exact hfg a t' l' hx

/-! ## Commitments' salts and erased openings -/

/-- A commitment reads its salt function only on its positions, cut to the scheme's width. -/
theorem commitP_congr (hs : Hashes PS) (ms : MerkleScheme) (lv : Level) (g : Array F128) {s s' : ℕ → ByteArray}
    (h : ∀ q < 2 ^ lv.d, fitBytes ms.saltBytes (s q) = fitBytes ms.saltBytes (s' q)) :
    commitP hs ms lv g s = commitP hs ms lv g s' := by
  have hsalts : (Array.range (2 ^ lv.d)).map (fun q => fitBytes ms.saltBytes (s q)) =
      (Array.range (2 ^ lv.d)).map (fun q => fitBytes ms.saltBytes (s' q)) := by
    apply Array.ext (by simp)
    intro i h1 _
    simp only [Array.getElem_map, Array.getElem_range]
    exact h i (by simpa using h1)
  simp only [commitP, hsalts]

theorem getElem!_map_const {α β : Type} [Inhabited β] (xs : Array α) (b : β) (hb : b = default) (p : ℕ) :
    (xs.map fun _ => b)[p]! = b := by
  by_cases hp : p < xs.size
  · rw [getElem!_pos (xs.map fun _ => b) p (by simpa using hp)]; simp
  · rw [getElem!_neg (xs.map fun _ => b) p (by simpa using hp), hb]

/-- **An erased commitment opens to the erased opening.** -/
theorem eraseLP_openP (ms : MerkleScheme) (cm : Commit) (lv : Level) (pos : Array ℕ) :
    eraseLP (openP ms cm lv pos) = openP ms (eraseC cm) lv pos := by
  simp only [eraseLP, openP, eraseC]
  congr 1
  · rw [Array.map_map]
    exact congrArg (fun f => Array.map f pos) (funext fun p => (getElem!_map_const cm.rows _ rfl p).symm)
  · split
    · simp
    · rw [Array.map_map]
      exact congrArg (fun f => Array.map f pos) (funext fun p => (getElem!_map_const cm.salts _ rfl p).symm)

theorem queryP_eraseC (io : Asks PS) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    queryP io nonce ms lv (eraseC cm) = (fun p => (eraseLP p.1, p.2)) <$> queryP io nonce ms lv cm := by
  simp only [queryP, map_bind, map_pure, eraseLP_openP]

/-! ## Level 0's opening at the coins from round `p₀` -/

section Vals

variable (coin : ℕ → ℕ → F128) (p0 : ℕ)

/-- Level 0's out-of-domain point, the reply to round `p₀`. -/
def zS (n : ℕ) : Array F128 := (Array.range n).map (coin p0)

/-- Lane round `j`'s challenge, the reply to round `p₀ + 2 + j`. -/
def chS (j : ℕ) : F256 := ⟨coin (p0 + 2 + j) 0, coin (p0 + 2 + j) 1⟩

/-- The table after the first `j` lane challenges. -/
def tabS (t : Array F128) : ℕ → Array F256
  | 0 => embedF t
  | j + 1 => foldK (chS coin p0 j) (tabS t j)

/-- The weight after the first `j` lane challenges: the basis plus `b₀·eq(z, ·)`, `b₀` the reply to round `p₀ + 1`. -/
def wS (b : Array F128) (n : ℕ) : ℕ → Array F256
  | 0 => addScaled (embedF b) (embedF (eqTable (zS coin p0 n))) (coin (p0 + 1) 0)
  | j + 1 => foldK (chS coin p0 j) (wS b n j)

/-- The pair's coins, the replies to rounds `p₀ + 2 + k₀` and `p₀ + 3 + k₀`. -/
def rhoS (k0 : ℕ) : F256 × F256 :=
  (⟨coin (p0 + 2 + k0) 0, coin (p0 + 2 + k0) 1⟩, ⟨coin (p0 + 3 + k0) 0, coin (p0 + 3 + k0) 1⟩)

end Vals

/-- Lane round `j`'s message as the source holds it, masked. -/
def mmOf (src : RepSrc) (lay : Zk.PadLayout) (j : ℕ) : F256 × F256 :=
  (⟨src.masked (lay.lgMsg j 0), src.masked (lay.lgMsg j 1)⟩, ⟨src.masked (lay.lgMsg j 2), src.masked (lay.lgMsg j 3)⟩)

/-- **What level 0's opening reads from the source**, as the session computes it at the coins from round `p₀`: the
out-of-domain value, each lane message, the pair's sums; where the pair's coins are answered and independent, the
restart `y₁'`, `ȳ`, and the salts of levels ≥ 1 on the positions their commitments read. -/
structure LigHyp (coin : ℕ → ℕ → F128) (n p0 : ℕ) (ms : MerkleScheme) (sch : Schedule) (t b : Array F128)
    (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (src : RepSrc) : Prop where
  k0 : 1 ≤ sch.k0
  n0 : sch.n ≠ 0
  levels : 2 ≤ sch.levels.size
  ood : src.masked (lay.lgOod 0) = mleF t (zS coin p0 sch.n) + h[lay.lgOod 0]!
  msg : ∀ j < sch.k0, mmOf src lay j = maskMsg lay h j (roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j))
  e0 : (⟨src.masked (lay.lgE 0 0), src.masked (lay.lgE 0 1)⟩ : F256) =
    dotK (wS coin p0 b sch.n sch.k0) (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep)))) +
      ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩
  e1 : (⟨src.masked (lay.lgE 1 0), src.masked (lay.lgE 1 1)⟩ : F256) =
    dotK (wS coin p0 b sch.n sch.k0) (embedF ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + 1)))) +
      ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩
  y1 : p0 + 3 + sch.k0 < n → Zk.rhoIndependent (rhoS coin p0 sch.k0) = true →
    src.y1 = (Array.range (tabS coin p0 t sch.k0).size).map fun c =>
      (tabS coin p0 t sch.k0)[c]! +
        (rhoS coin p0 sch.k0).1.smul ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep)))[c]! +
        (rhoS coin p0 sch.k0).2.smul ((Array.range (tabS coin p0 t sch.k0).size).map (extra (2 * rep + 1)))[c]!
  ybar : p0 + 3 + sch.k0 < n → Zk.rhoIndependent (rhoS coin p0 sch.k0) = true →
    src.ybar = (Array.range tPad).map fun i =>
      (List.range (2 ^ (sch.levels.getD 0 default).lanes)).foldl
        (fun acc l => acc + ((eqTableK ((Array.range sch.k0).map (chS coin p0)))[l]!).smul (lanePad l i))
        ((rhoS coin p0 sch.k0).1.smul (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rep) i) +
          (rhoS coin p0 sch.k0).2.smul (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rep + 1) i))
  salt : p0 + 3 + sch.k0 < n → Zk.rhoIndependent (rhoS coin p0 sch.k0) = true →
    ∀ i, 1 ≤ i → i < sch.levels.size → ∀ q < 2 ^ (sch.levels.getD i default).d,
      fitBytes ms.saltBytes (src.hiSalt i q) = fitBytes ms.saltBytes (salt i q)

/-- **Level 0's opening from the source, in lockstep with the session's**: answered by `asksCut` from `l`, where the
source holds what the session computes at the coins from round `l.size` on (`LigHyp`), the source's opening of `cm0`
erased sends what the session's sends and returns the session's proof but level 0's opened rows and salts. -/
theorem ligeritoCut_alike (coin : ℕ → ℕ → F128) (n : ℕ) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (src : RepSrc) (tr : Transcript)
    (l : Array Claims.Msg)
    (H : LigHyp coin n l.size ms sch t b lay h rep tPad extra lanePad salt src) :
    ((ligeritoCut (asksCut coin n) hs ms sch T (eraseC cm0) b lay src).run tr).go l =
      ((do
        let (lg, zl, rec) ← ligeritoPZk (asksCut coin n) hs ms sch T cm0 t b lay h rep tPad extra lanePad salt
          fun _ _ => 0
        pure (eraseL lg, zl, rec) : TMW PS (LigeritoProof × Zk.Level0Msgs × Zk.L0Rec)).run tr).go l := by
  apply Alike.eq
  unfold ligeritoCut ligeritoPZk
  refine alike_bind_eq_assoc fun _ t₁ l₁ e₁ => ?_
  obtain rfl := liftTM_inv e₁
  refine alike_bind_eq_assoc fun _ t₂ l₂ e₂ => ?_
  obtain rfl := liftTM_inv e₂
  refine alike_bind_eq_assoc fun _ t₃ l₃ e₃ => ?_
  obtain rfl := liftTM_inv e₃
  refine alike_bind_eq_assoc fun _ t₄ l₄ e₄ => ?_
  obtain rfl := liftTM_inv e₄
  refine alike_bind_assoc (R := fun a b _ =>
    a.1 = (b.1.1, b.1.2.1, b.1.2.2.1, eraseLP b.1.2.2.2.1, b.1.2.2.2.2) ∧ a.2 = b.2) ?prog ?tail
  case tail =>
    rintro ⟨⟨caps, rproofs, fin, op0, zl, initCh, rho, b0, y⟩, st⟩
      ⟨⟨caps', rproofs', fin', op0', zl', initCh', rho', b0', y'⟩, st'⟩ t' l' _ _ ⟨h1, h2⟩
    simp only [Prod.mk.injEq] at h1
    obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ := h1
    subst h2
    exact .inr ⟨_, _, _, _, rfl, rfl, rfl⟩
  case prog =>
    refine lliftT fun z t₅ l₅ e₅ => ?_
    obtain ⟨hn₅, rfl, hl₅⟩ := sampleVecP_cut_inv coin H.n0 e₅
    dsimp only
    rw [H.ood, modify_bind_run, modify_bind_run]
    refine lliftT fun _ t₆ l₆ e₆ => ?_
    obtain rfl := liftTM_inv e₆
    rw [powP_bind_run, powP_bind_run]
    refine lliftT fun _ t₇ l₇ e₇ => ?_
    obtain rfl := liftTM_inv e₇
    refine lliftT fun b0 t₈ l₈ e₈ => ?_
    obtain ⟨hn₈, rfl, hl₈⟩ := sampleP_cut_inv coin e₈
    rw [modify_bind_run, modify_bind_run, get_bind_run]
    dsimp only
    have hm0 := H.msg 0 H.k0
    simp only [mmOf] at hm0
    rw [hl₅, hm0, sendMsg_bind_run, sendMsg_bind_run]
    refine lliftT fun _ t₉ l₉ e₉ => ?_
    obtain rfl := liftTM_inv e₉
    refine lliftT fun _ t₁₀ l₁₀ e₁₀ => ?_
    obtain rfl := liftTM_inv e₁₀
    refine lbind (lforIn _ _ (fun j (p : (Array F256 × (F256 × F256) × Zk.Level0Msgs) × LigState)
        (p' : (Array F256 × (F256 × F256) × Zk.Level0Msgs) × LigState) (l' : Array Claims.Msg) =>
        p.1 = p'.1 ∧
        (j < sch.k0 → l'.size = l.size + 2 + j ∧ p'.2 = { p.2 with tab := tabS coin l.size t j } ∧
          p.2.w = wS coin l.size b sch.n j ∧ p.1.1 = (Array.range j).map (chS coin l.size)) ∧
        (sch.k0 ≤ j → p.2 = p'.2 ∧ l.size + 3 + sch.k0 < n ∧
          Zk.rhoIndependent (rhoS coin l.size sch.k0) = true))
      sch.k0 ?step _ _ ?init) fun r s₁ r' s₁' t₁ l₁ _ _ hI => ?rest
    case init =>
      refine ⟨rfl, fun _ => ⟨by omega, rfl, rfl, by simp⟩, fun h => absurd h (by have := H.k0; omega)⟩
    case step =>
      intro j hj u s u' s' tj lj ⟨hu, hlt, _⟩
      obtain ⟨hlj, rfl, hw, hch⟩ := hlt hj
      subst hu
      rcases u with ⟨initCh, rho, zl⟩
      dsimp only at hch ⊢
      refine lliftT fun r tr lr er => ?_
      obtain ⟨hnr, rfl, hlr⟩ := sampleKP_cut_inv coin er
      rw [modify_bind_run, modify_bind_run, hlj, hw]
      by_cases hk : j + 1 = sch.k0
      · have hb : (j + 1 == sch.k0) = true := by simpa using hk
        rw [ite_eq_left_of_eq_true _ _ (eq_true hb), ite_eq_left_of_eq_true _ _ (eq_true hb),
          get_bind_run, get_bind_run]
        subst hch
        have he0 := H.e0
        have he1 := H.e1
        rw [← hk] at he0 he1
        rw [he0, he1]
        refine lliftT fun _ t₁ l₁ e₁ => ?_
        obtain rfl := liftTM_inv e₁
        refine lliftT fun _ t₂ l₂ e₂ => ?_
        obtain rfl := liftTM_inv e₂
        refine lliftT fun rh0 t₃ l₃ e₃ => ?_
        obtain ⟨hn₃, rfl, hl₃⟩ := sampleKP_cut_inv coin e₃
        refine lliftT fun rh1 t₄ l₄ e₄ => ?_
        obtain ⟨hn₄, rfl, hl₄⟩ := sampleKP_cut_inv coin e₄
        rw [show lr.size = l.size + 2 + (j + 1) by omega, show l₃.size = l.size + 3 + (j + 1) by omega]
        by_cases hρ : Zk.rhoIndependent (rhoS coin l.size (j + 1)) = true
        · have hρu : Zk.rhoIndependent (⟨coin (l.size + 2 + (j + 1)) 0, coin (l.size + 2 + (j + 1)) 1⟩,
              ⟨coin (l.size + 3 + (j + 1)) 0, coin (l.size + 3 + (j + 1)) 1⟩) = true := hρ
          have hb2 : ¬ (!Zk.rhoIndependent (⟨coin (l.size + 2 + (j + 1)) 0, coin (l.size + 2 + (j + 1)) 1⟩,
              ⟨coin (l.size + 3 + (j + 1)) 0, coin (l.size + 3 + (j + 1)) 1⟩)) = true := by simp [hρu]
          rw [ite_eq_right_of_eq_false _ _ (eq_false hb2), ite_eq_right_of_eq_false _ _ (eq_false hb2)]
          have hnl : l.size + 3 + sch.k0 < n := by omega
          have hρk : Zk.rhoIndependent (rhoS coin l.size sch.k0) = true := by rw [← hk]; exact hρ
          have hy1 := H.y1 hnl hρk
          have hyb := H.ybar hnl hρk
          rw [← hk] at hy1 hyb
          rw [hy1, hyb, show ((Array.range j).map (chS coin l.size)).push
              ⟨coin (l.size + 2 + j) 0, coin (l.size + 2 + j) 1⟩ = (Array.range (j + 1)).map (chS coin l.size) by
            simp [Array.range_succ, chS]]
          refine lliftT fun _ t₅ l₅' e₅ => ?_
          obtain rfl := liftTM_inv e₅
          refine lliftT fun _ t₆ l₆ e₆ => ?_
          obtain rfl := liftTM_inv e₆
          refine lliftT fun _ t₇ l₇ e₇ => ?_
          obtain rfl := liftTM_inv e₇
          rw [modify_bind_run, modify_bind_run, get_bind_run, get_bind_run, sendMsg_bind_run, sendMsg_bind_run]
          refine lliftT fun _ t₈ l₈' e₈ => ?_
          obtain rfl := liftTM_inv e₈
          refine lliftT fun _ t₉ l₉ e₉ => ?_
          obtain rfl := liftTM_inv e₉
          exact .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, rfl, fun h => absurd h (by omega),
            fun _ => ⟨rfl, hnl, hρk⟩⟩
        · have hρu : ¬ Zk.rhoIndependent (⟨coin (l.size + 2 + (j + 1)) 0, coin (l.size + 2 + (j + 1)) 1⟩,
              ⟨coin (l.size + 3 + (j + 1)) 0, coin (l.size + 3 + (j + 1)) 1⟩) = true := hρ
          have hb2 : (!Zk.rhoIndependent (⟨coin (l.size + 2 + (j + 1)) 0, coin (l.size + 2 + (j + 1)) 1⟩,
              ⟨coin (l.size + 3 + (j + 1)) 0, coin (l.size + 3 + (j + 1)) 1⟩)) = true := by simpa using hρu
          rw [ite_eq_left_of_eq_true _ _ (eq_true hb2), ite_eq_left_of_eq_true _ _ (eq_true hb2)]
          exact .inl ⟨_, _, rfl, rfl⟩
      · have hb : ¬ (j + 1 == sch.k0) = true := by simpa using hk
        rw [ite_eq_right_of_eq_false _ _ (eq_false hb), ite_eq_right_of_eq_false _ _ (eq_false hb), get_bind_run]
        have hm := H.msg (j + 1) (by omega)
        simp only [mmOf] at hm
        rw [hm, sendMsg_bind_run, sendMsg_bind_run]
        refine lliftT fun _ t₁ l₁ e₁ => ?_
        obtain rfl := liftTM_inv e₁
        refine lliftT fun _ t₂ l₂ e₂ => ?_
        obtain rfl := liftTM_inv e₂
        refine .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, rfl, fun _ => ⟨by omega, rfl, rfl, ?_⟩,
          fun h => absurd h (by omega)⟩
        rw [hch]
        simp [Array.range_succ, chS]
    case rest =>
      obtain ⟨hu, -, hge⟩ := hI
      obtain ⟨rfl, hnl, hρk⟩ := hge le_rfl
      subst hu
      rcases r with ⟨initCh, rho, zl⟩
      dsimp only
      rw [get_bind_run, get_bind_run,
        commitP_congr hs ms _ _ (H.salt hnl hρk 1 le_rfl (by have := H.levels; omega))]
      refine lliftT fun cm1 t₂ l₂ _ => ?_
      refine lliftT fun _ t₃ l₃ _ => ?_
      refine lbind_eq fun _ s₄ t₄ l₄ _ => ?_
      refine lbind_eq fun _ s₅ t₅ l₅ _ => ?_
      rw [queryP_eraseC, bind_map_left]
      refine lbind_eq fun a s₆ t₆ l₆ _ => ?_
      rcases a with ⟨op0, v0⟩
      dsimp only
      refine lbind_eq fun _ s₇ t₇ l₇ _ => ?_
      refine lbind_eq fun _ s₈ t₈ l₈ _ => ?_
      refine lliftT fun bq t₉ l₉ _ => ?_
      rw [modify_bind_run, modify_bind_run]
      refine lbind (lforIn_eq _ _ _ (fun i hi u s t l => ?_) _ _ _ _) fun r s₁₀ r' s₁₀' t₁₀ l₁₀ _ _ hr => ?_
      · refine Alike.of_eq ?_ s t l
        by_cases hlast : i + 1 = sch.levels.size - 1
        · have hl : (!(i + 1 == sch.levels.size - 1)) = false := by simp [hlast]
          simp only [hl, Bool.false_eq_true, ↓reduceIte]
        · have hc : ∀ g, commitP hs ms (sch.levels.getD (i + 2) default) g (src.hiSalt (i + 2)) =
              commitP hs ms (sch.levels.getD (i + 2) default) g (salt (i + 2)) := fun g =>
            commitP_congr hs ms _ g (H.salt hnl hρk (i + 2) (by omega) (by omega))
          simp only [hc]
      · obtain ⟨rfl, rfl⟩ := Prod.mk.inj hr
        exact lpure ⟨rfl, rfl⟩

theorem go_bind_congr_l {α β : Type} {x y : TMW PS α} {f : α → TMW PS β} {t : Transcript} {l : Array Claims.Msg}
    (h : (x.run t).go l = (y.run t).go l) : ((x >>= f).run t).go l = ((y >>= f).run t).go l := by
  rcases hx : (x.run t).go l with ⟨r, L⟩
  rcases r with e | ⟨a, t'⟩
  · rw [bind_go_err hx, bind_go_err (h ▸ hx)]
  · rw [bind_go hx, bind_go (h ▸ hx)]

/-- **Hybrids 1 and 2 differ in level 0's opening only**: both run the session's PIOP, and where it returns, an opening
from the source that sends what the session's sends and returns its erased proof runs the same rep. -/
theorem repMix_one_two (io : Asks PS) (hs : Hashes PS) (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (src : RepSrc) (t₀ : Transcript) (l : Array Claims.Msg)
    (h : ∀ zp lp ss T sws t' l', ((piopPZk io st (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr (flat cm0.cap)
        ic.root extras prior words).run t₀).go l = (.ok ((zp, lp, ss, T, sws), t'), l') →
      ((ligeritoCut io hs st.merkle sch T (eraseC cm0) (basisVals sws (st.m - 7))
          (Zk.PadLayout.new st.m st.kLog sch.k0) src).run t').go l' =
        ((do
          let (lg, zl, rec) ← ligeritoPZk io hs st.merkle sch T cm0 t (basisVals sws (st.m - 7))
            (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h rep tPad extra lanePad salt nonce
          pure (eraseL lg, zl, rec) : TMW PS (LigeritoProof × Zk.Level0Msgs × Zk.L0Rec)).run t').go l') :
    ((repMix 1 io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src).run t₀).go l =
      ((repMix 2 io hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words src).run
        t₀).go l := by
  apply Alike.eq
  unfold repMix
  simp only [show ¬ (1 : ℕ) = 0 by decide, show ¬ (2 : ℕ) = 0 by decide, show (1 : ℕ) ≤ 1 by decide,
    show ¬ (2 : ℕ) ≤ 1 by decide, show (1 : ℕ) ≤ 2 by decide, show (2 : ℕ) ≤ 2 by decide, ↓reduceIte]
  refine alike_bind_eq fun a t' l' ha => ?_
  rcases a with ⟨zp, lp, ss, T, sws⟩
  dsimp only
  rw [go_bind_congr_l (h zp lp ss T sws t' l' ha)]
  simp only [bind_assoc]
  refine alike_bind_eq fun b t'' l'' _ => ?_
  rcases b with ⟨lg, zl, rec⟩
  exact Alike.refl _

/-! ## The session's PIOP answered by `asksCut` -/

section Cut

open Replay With

/-- **A run answered by `asksCut` that returns is the run answered by `asksS`.** -/
def cutS : With.Sim PS PS where
  R x y := ∀ l a l', x.go l = (.ok a, l') → y.go l = (.ok a, l')
  pure _ _ _ _ h := h
  bind {α β x y f g} hxy hfg l b l'' h := by
    rw [go_bind] at h ⊢
    rcases hx : x.go l with ⟨r, l'⟩
    rw [hx] at h
    cases r with
    | error e => exact absurd h (by simp)
    | ok a =>
      rw [hxy l a l' hx]
      exact hfg a l' b l'' h
  throw e l a l' h := by simp [StV.go] at h
  lift _ _ _ _ h := h

section CutS

variable (coin : ℕ → ℕ → F128) (n : ℕ) (resp : ℕ → Claims.Msg) (hr : ∀ k < n, coinOf (resp k) = coin k)
include hr

theorem cutS_round (c : ByteArray) : cutS.R ((asksCut coin n).round c) ((asksS resp).round c) := by
  intro l a l' h
  rw [asksCut_round_go] at h
  by_cases hn : n ≤ l.size
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hn)] at h
    simp [StV.go] at h
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hn)] at h
    change (Except.ok (coinOf (resp l.size)), l.push (Claims.toMsg c)) = _
    rw [hr l.size (by omega)]
    exact h

theorem squeezeP_cutS (kind : UInt8) (k : ℕ) :
    (cutS.st Transcript).R (squeezeP (asksCut coin n) kind k) (squeezeP (asksS resp) kind k) := by
  intro t
  unfold squeezeP
  sim_walk [cutS_round coin n resp hr]

theorem sampleP_cutS : (cutS.st Transcript).R (sampleP (asksCut coin n)) (sampleP (asksS resp)) := by
  unfold sampleP
  sim_walk [squeezeP_cutS coin n resp hr]

theorem zerocheckPZk_cutS (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) :
    (cutS.st Transcript).R (zerocheckPZk (asksCut coin n) m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_cutS coin n resp hr, sampleP_cutS coin n resp hr]

theorem lincheckPZk_cutS (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (zc : ZcOut) :
    (cutS.st Transcript).R (lincheckPZk (asksCut coin n) c m lay h tr zc) (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_cutS coin n resp hr]

theorem ringSwitchP_cutS (tr : Trace) (claims : Array Claim) :
    (cutS.st Transcript).R (ringSwitchP (asksCut coin n) tr claims) (ringSwitchP (asksS resp) tr claims) := by
  unfold ringSwitchP sampleVecP
  sim_walk [squeezeP_cutS coin n resp hr]

theorem piopPZk_cutS (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) :
    (cutS.st Transcript).R (piopPZk (asksCut coin n) st lay h tr cap root extras prior words)
      (piopPZk (asksS resp) st lay h tr cap root extras prior words) := by
  unfold piopPZk rankCheckP
  sim_walk [zerocheckPZk_cutS coin n resp hr, lincheckPZk_cutS coin n resp hr, ringSwitchP_cutS coin n resp hr]

/-- **The session's PIOP from `asksCut`, where it returns, is the session's from `asksS`** at replies reading as the
coins before round `n`. -/
theorem piopPZk_cut_ok (st : Setup) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (cap root : ByteArray)
    (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ) {t : Transcript} {l l' : Array Claims.Msg}
    {a : (ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched) × Transcript}
    (e : ((piopPZk (asksCut coin n) st lay h tr cap root extras prior words).run t).go l = (.ok a, l')) :
    ((piopPZk (asksS resp) st lay h tr cap root extras prior words).run t).go l = (.ok a, l') :=
  piopPZk_cutS coin n resp hr st lay h tr cap root extras prior words t l a l' e

end CutS

/-- **The session's PIOP at `asksS`, where it returns**, sent `piopZk st extras` bodies, and its ring switch's vectors
are the closed form's at the replies (`ringSwitchP_go`, after `zerocheckPZk_go` and `lincheckPZk_go`). -/
theorem piopPZk_exact (resp : ℕ → Claims.Msg) (st : Setup) (hm : 14 ≤ st.m) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (cap root : ByteArray) (extras : Array Claim) (prior : Array (List F128)) (words : List ℕ)
    {t t' : Transcript} {l l' : Array Claims.Msg}
    {a : ZerocheckProof × LincheckProof × Array (Array F128) × F128 × Array Switched}
    (e : ((piopPZk (asksS resp) st lay h tr cap root extras prior words).run t).go l = (.ok (a, t'), l')) :
    l'.size = l.size + piopZk st extras ∧
      a.2.2.2.2 = Rs.sws (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
          (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
        (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6)) (2 + extras.size)) := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckPZk_go resp hm st.digest cap root lay h tr t l
  unfold piopPZk at e
  rw [bind_go hz] at e
  dsimp only at e
  have e' := e
  rw [StateT.run_bind] at e'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok e'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, hl₂, hl⟩ := lincheckPZk_go resp st.fold st.m lay h tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at e
  dsimp only at e
  rw [StateT.run_bind] at e
  obtain ⟨⟨u, t₃⟩, l₃', hk, e⟩ := JsonRT.go_bind_ok e
  obtain ⟨ht₃, hl₃⟩ := rankCheckP_go hk
  subst t₃ l₃'
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras) (by simp [abcClaims]) _ l₂
  rw [bind_go hrs] at e
  simp only [StateT.run_pure, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at e
  obtain ⟨⟨rfl, -⟩, rfl⟩ := e
  have hsz : (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) (Lc.coinsAt resp l₁.size (st.fold.kLog - 6))) ++
      extras).size = 2 + extras.size := by simp [abcClaims, Nat.add_comm]
  refine ⟨?_, ?_⟩
  · rw [hl₃, hsz, hl₂, hl₁]
    unfold piopZk
    omega
  · rw [hsz, hl₂, hl₁]

end Cut

end LigLock

end FlockComplete

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.ZK FlockSoundness.Discharge CoinLayer

/-- **What the deployed source carries for level 0's opening**: wherever the deployed rep's PIOP returns, the cut's
source holds what the session's opening computes at the coins from the PIOP's last send on (`LigHyp`). -/
def LigSrcE : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
      (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras →
    ∀ l : Array Msg, l.size = repBase x (2 * j.val + r.val) →
    ∀ zp lp ss T sws t' l',
      ((FlockComplete.piopPZk (ioE c n) (FlockDeploy.tabsOf (stsOf x))[j.val]!
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
          (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
              (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)).h
          (trsE x w (UE.tapeZk τ))[j.val]! (FlockComplete.flat cm0.cap) ic.root extras (priorE c j.val r.val)
          (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!).run (_root_.Flock.Transcript.init #[] dom)).go l =
        (.ok ((zp, lp, ss, T, sws), t'), l') →
      FlockComplete.LigLock.LigHyp (fun k i => slotAt c k i) n l'.size
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.merkle (zkTab x j).sch
        (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (FlockDeploy.tabsOf (stsOf x))[j.val]!.m)
        (FlockComplete.basisVals sws ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 7))
        (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
          (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
        (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val
          (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
            (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)).h
        r.val (zkTab x j).tPad ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val)
        ((UE.tapeZk τ).salt j.val r.val) ((srcE x c w τ j.val).rep r.val)

/-- **`SimLig` from what the cut carries for level 0's opening**: hybrids 1 and 2 run the session's PIOP, and where it
returns, the source's opening sends what the session's sends and returns its erased proof (`ligeritoCut_alike`, at the
coins from the PIOP's last send on), every nonce `0` (`exeTape`'s). -/
theorem simLig_of (h : LigSrcE) : SimLig := by
  intro x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl
  unfold mixE
  refine FlockComplete.LigLock.repMix_one_two _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    fun zp lp ss T sws t' l' hp => ?_
  exact FlockComplete.LigLock.ligeritoCut_alike _ n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    (h x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl zp lp ss T sws t' l' hp)

/-- **The replies the opened coins read as**: round `k`'s coin's bytes, the zero coin's past the pre-final rounds. -/
noncomputable def respE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) : Msg :=
  (if h : k < refK x then c ⟨k, h⟩ else default).bytes

theorem coinOf_respE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) :
    FlockComplete.coinOf (respE c k) = fun i => slotAt c k i := by
  funext i
  unfold respE slotAt
  rw [FlockComplete.JsonRT.coinOf_bytes]
  split_ifs
  · rfl
  · unfold FlockDeploy.Coin.at
    split <;> rfl

/-- **Table `j`'s rep `r` switched vectors at the opened coins**: the session's ring switch's, in closed form at the
replies `respE c` from the rep's first round (`piopPZk_exact`). -/
noncomputable def swsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim) : Array _root_.Flock.Switched :=
  FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog
      (FlockComplete.Replay.Zc.out (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockComplete.Replay.Zc.coinsAt (respE c) (repBase x (2 * j.val + r.val))) (trsE x w (UE.tapeZk τ))[j.val]!)
      (FlockComplete.Replay.Lc.out (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (trsE x w (UE.tapeZk τ))[j.val]!
        (FlockComplete.Replay.Zc.out (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
          (FlockComplete.Replay.Zc.coinsAt (respE c) (repBase x (2 * j.val + r.val))) (trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Lc.coinsAt (respE c)
          (repBase x (2 * j.val + r.val) + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6))
          ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6))) ++ extras)
    (FlockComplete.Replay.Rs.coinsAt (respE c)
      (repBase x (2 * j.val + r.val) + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6) + 3 +
        ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6)) (2 + extras.size))

/-! ## Level 0's opening at the deployed rep's coins -/

/-- **A statement of each deployed rep**: at a live statement and a witness, a tape, opened coins `c` not refused by
round `n`, the session's setup, table `j`, rep `r` and the table's extra claims at the link's points. -/
def LigEach (P : (x : Msg) → Claims.Flock.Wit → UE.Tape → (Fin (refK x) → FlockDeploy.Coin) → ℕ →
    _root_.Flock.SessionSpec → Fin (jOf x) → Fin 2 → Array _root_.Flock.Claim → Prop) : Prop :=
  ∀ x, Live x → ∀ w, RZk x w → ∀ (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ), ¬ Refused x c n →
    ∀ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) →
    ∀ (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras → P x w τ c n spec j r extras

/-- Table `j`'s setup. -/
abbrev ligSt (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Setup := (FlockDeploy.tabsOf (stsOf x))[j.val]!

/-- Table `j`'s pad layout. -/
noncomputable abbrev ligLay (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Zk.PadLayout :=
  _root_.Flock.Zk.PadLayout.new (ligSt x j).m (ligSt x j).kLog (zkTab x j).sch.k0

/-- Table `j`'s rep `r` pads. -/
noncomputable abbrev ligPad (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) : Array _root_.Flock.F128 :=
  (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (ligLay x j)).h

/-- Table `j`'s packed table, at `w`'s trace. -/
noncomputable abbrev ligTab (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x)) : Array _root_.Flock.F128 :=
  FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (ligSt x j).m

/-- Table `j`'s rep `r` basis at the opened coins: its switched vectors' (`swsE`). -/
noncomputable abbrev ligBasis (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x))
    (r : Fin 2) (extras : Array _root_.Flock.Claim) : Array _root_.Flock.F128 :=
  FlockComplete.basisVals (swsE x c w τ j r extras) ((ligSt x j).m - 7)

/-- Rep `2j + r`'s round after its PIOP: Ligerito's first. -/
abbrev ligP0 (x : Msg) (j : Fin (jOf x)) (r : Fin 2) : ℕ := repSlot x (2 * j.val + r.val) fun _ => 0

/-- The opened coins by round and element. -/
abbrev ligCoin {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) : ℕ → ℕ → _root_.Flock.F128 := fun k i => slotAt c k i

/-- Table `j`'s rep `r` source. -/
noncomputable abbrev ligSrc (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x))
    (r : Fin 2) : FlockComplete.RepSrc :=
  (srcE x c w τ j.val).rep r.val

/-- **What the deployed source carries for level 0's opening, at the coins**: rep `r`'s source holds what the
session's opening computes at the coins from round `ligP0` on, the round after the rep's PIOP, with the basis of the
PIOP's switched vectors at the coins (`ligBasis`). -/
def LigHypE : Prop := LigEach fun x w τ c n _ j r extras =>
  FlockComplete.LigLock.LigHyp (ligCoin c) n (ligP0 x j r) (ligSt x j).merkle (zkTab x j).sch (ligTab x w τ j)
    (ligBasis x c w τ j r extras) (ligLay x j) (ligPad x τ j r) r.val (zkTab x j).tPad ((UE.tapeZk τ).extra j.val)
    ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val) (ligSrc x c w τ j r)

section Fields

open FlockComplete FlockComplete.LigLock

/-- **Level 0's out-of-domain value at the coins**: the source's, masked, is the packed table's multilinear extension
at the reply to round `p₀`, plus its pad. -/
def LigOodE : Prop := LigEach fun x w τ c _ _ j r _ =>
  (ligSrc x c w τ j r).masked ((ligLay x j).lgOod 0) =
    mleF (ligTab x w τ j) (zS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.n) + (ligPad x τ j r)[(ligLay x j).lgOod 0]!

/-- **Level 0's lane messages at the coins**: the source's, masked, are the round polynomials of the table and the
weight after each lane challenge. -/
def LigMsgE : Prop := LigEach fun x w τ c _ _ j r extras =>
  ∀ i < (zkTab x j).sch.k0, mmOf (ligSrc x c w τ j r) (ligLay x j) i =
    maskMsg (ligLay x j) (ligPad x τ j r) i
      (roundPoly (tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i)
        (wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i))

/-- **Level 0's pair's sums at the coins**: the source's, masked, are the folded weight's products with the rep's
extra lanes. -/
def LigSumsE : Prop := LigEach fun x w τ c _ _ j r extras =>
  let sch := (zkTab x j).sch
  let src := ligSrc x c w τ j r
  let lay := ligLay x j
  let h := ligPad x τ j r
  let wk := wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) sch.n sch.k0
  let N := (tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) sch.k0).size
  (⟨src.masked (lay.lgE 0 0), src.masked (lay.lgE 0 1)⟩ : _root_.Flock.F256) =
      dotK wk (embedF ((Array.range N).map ((UE.tapeZk τ).extra j.val (2 * r.val)))) + ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩ ∧
    (⟨src.masked (lay.lgE 1 0), src.masked (lay.lgE 1 1)⟩ : _root_.Flock.F256) =
      dotK wk (embedF ((Array.range N).map ((UE.tapeZk τ).extra j.val (2 * r.val + 1)))) +
        ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩

/-- **The restart's `y₁'` at the coins**: where the pair's coins are answered and independent, the source's is the
folded table plus the pair's coins times the rep's extra lanes. -/
def LigY1E : Prop := LigEach fun x w τ c n _ j r _ =>
  let sch := (zkTab x j).sch
  let tk := tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) sch.k0
  let ρ := rhoS (ligCoin c) (ligP0 x j r) sch.k0
  ligP0 x j r + 3 + sch.k0 < n → _root_.Flock.Zk.rhoIndependent ρ = true →
    (ligSrc x c w τ j r).y1 = (Array.range tk.size).map fun i =>
      tk[i]! + ρ.1.smul ((Array.range tk.size).map ((UE.tapeZk τ).extra j.val (2 * r.val)))[i]! +
        ρ.2.smul ((Array.range tk.size).map ((UE.tapeZk τ).extra j.val (2 * r.val + 1)))[i]!

/-- **The restart's `ȳ` at the coins**: where the pair's coins are answered and independent, the source's is the lane
pads under the lane challenges' `eq` table plus the pair's coins times the rep's extra lane pads. -/
def LigYbarE : Prop := LigEach fun x w τ c n _ j r _ =>
  let sch := (zkTab x j).sch
  let ρ := rhoS (ligCoin c) (ligP0 x j r) sch.k0
  let lp := (UE.tapeZk τ).lanePad j.val
  let L := 2 ^ (sch.levels.getD 0 default).lanes
  ligP0 x j r + 3 + sch.k0 < n → _root_.Flock.Zk.rhoIndependent ρ = true →
    (ligSrc x c w τ j r).ybar = (Array.range (zkTab x j).tPad).map fun i =>
      (List.range L).foldl
        (fun acc l => acc + ((_root_.Flock.eqTableK ((Array.range sch.k0).map (chS (ligCoin c) (ligP0 x j r))))[l]!).smul (lp l i))
        (ρ.1.smul (lp (L + 2 * r.val) i) + ρ.2.smul (lp (L + 2 * r.val + 1) i))

/-- **Levels ≥ 1's salts at the coins**: where the pair's coins are answered and independent, the source's, fitted,
are the tape's on the positions each level's commitment reads. -/
def LigSaltE : Prop := LigEach fun x w τ c n _ j r _ =>
  let sch := (zkTab x j).sch
  ligP0 x j r + 3 + sch.k0 < n → _root_.Flock.Zk.rhoIndependent (rhoS (ligCoin c) (ligP0 x j r) sch.k0) = true →
    ∀ i, 1 ≤ i → i < sch.levels.size → ∀ q < 2 ^ (sch.levels.getD i default).d,
      fitBytes (ligSt x j).merkle.saltBytes ((ligSrc x c w τ j r).hiSalt i q) =
        fitBytes (ligSt x j).merkle.saltBytes ((UE.tapeZk τ).salt j.val r.val i q)

end Fields

/-- **`LigHypE` from its fields**: the schedule's facts (`zk_schedule_seams`: `k₀ ≥ 1`, `n = m − 7` with `m ≥ 25`, at
least two levels) and the values at the coins. -/
theorem ligHypE_of (hood : LigOodE) (hmsg : LigMsgE) (hsum : LigSumsE) (hy1 : LigY1E) (hybar : LigYbarE)
    (hsalt : LigSaltE) : LigHypE := by
  intro x hx w hw τ c n hn s spec hs j r extras he
  obtain ⟨h25, -, hsm, hk0, -, hdim, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨hn7, h2, -⟩ := hdim
  exact
    { k0 := hk0
      n0 := by omega
      levels := h2
      ood := hood x hx w hw τ c n hn s spec hs j r extras he
      msg := hmsg x hx w hw τ c n hn s spec hs j r extras he
      e0 := (hsum x hx w hw τ c n hn s spec hs j r extras he).1
      e1 := (hsum x hx w hw τ c n hn s spec hs j r extras he).2
      y1 := hy1 x hx w hw τ c n hn s spec hs j r extras he
      ybar := hybar x hx w hw τ c n hn s spec hs j r extras he
      salt := hsalt x hx w hw τ c n hn s spec hs j r extras he }

/-- Table `j`'s statement is rep `2j + r`'s. -/
theorem stsOf_half (x : Msg) (j : Fin (jOf x)) (r : Fin 2) : (stsOf x)[(2 * j.val + r.val) / 2]! = stAt x j := by
  rw [show (2 * j.val + r.val) / 2 = j.val by omega]
  exact getElem!_pos (stsOf x) j.val (lt_of_jOf j)

/-- The session's setup of table `j` is its statement's. -/
theorem tabsOf_stAt (x : Msg) (j : Fin (jOf x)) :
    (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
  rw [getElem!_pos _ _ (by simpa [FlockDeploy.tabsOf] using lt_of_jOf j)]
  simp [FlockDeploy.tabsOf, stAt]

/-- **`LigSrcE` from the source at the coins** (`LigHypE`): the deployed rep's PIOP, answered by `asksCut`, returns as
it does at `asksS (respE c)` (`piopPZk_cut_ok`), so it sent `piopZk` bodies from `repBase`, landing at `repSlot … 0`,
and returned `swsE` (`piopPZk_exact`). -/
theorem ligSrcE_of (h : LigHypE) : LigSrcE := by
  intro x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl zp lp ss T sws t' l' hp
  have hr : ∀ k < n, FlockComplete.coinOf (respE c k) = (fun k i => slotAt c k i) k := fun k _ => coinOf_respE c k
  have hp' := FlockComplete.LigLock.piopPZk_cut_ok _ n (respE c) hr _ _ _ _ _ _ _ _ _ hp
  have hst := tabsOf_stAt x j
  obtain ⟨h25, -, hsm, -, -, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hm : 14 ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m := by
    rw [hst]
    show 14 ≤ (stAt x j).m
    omega
  obtain ⟨hsize, hsws⟩ := FlockComplete.LigLock.piopPZk_exact _ _ hm _ _ _ _ _ _ _ _ hp'
  have hx' : extras.size = (stAt x j).regions.size * spec.points := by
    rw [hst] at he
    rw [FlockComplete.JsonRT.extraClaims_size he]
    simp [ptsE]
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hslot : l'.size = repSlot x (2 * j.val + r.val) fun _ => 0 := by
    rw [hsize, hl, repBase_eq hs, repSlot_eq hs hz, stsOf_half, hst]
    unfold FlockComplete.piopZk FlockComplete.piopRoundsZk
    rw [hx']
    show _ + (3 + ((stAt x j).m - 6) + (3 + ((stAt x j).c.kLog - 6)) + (3 + (stAt x j).regions.size * spec.points)) =
      _ + ((stAt x j).m + (stAt x j).c.kLog - 6 + (2 + spec.points * (stAt x j).regions.size + 1)) + 0
    rw [Nat.mul_comm spec.points]
    omega
  rw [hl] at hsws
  rw [hslot, show sws = swsE x c w τ j r extras from hsws]
  exact h x hx w hw τ c n hn s spec hs j r extras he

/-- **`SimLig` from level 0's opening at the deployed rep's coins**: the out-of-domain value, the lane messages, the
pair's sums, and where the pair's coins are answered and independent the restart's `y₁'`, `ȳ` and levels ≥ 1's salts. -/
theorem simLig_of_fields (hood : LigOodE) (hmsg : LigMsgE) (hsum : LigSumsE) (hy1 : LigY1E) (hybar : LigYbarE)
    (hsalt : LigSaltE) : SimLig :=
  simLig_of (ligSrcE_of (ligHypE_of hood hmsg hsum hy1 hybar hsalt))

end Masked

end Claims.ZKMal
