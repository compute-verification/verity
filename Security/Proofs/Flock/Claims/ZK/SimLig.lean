import Proofs.Flock.Claims.ZK.Split
import Proofs.Flock.Claims.ZK.SentE
import Proofs.Flock.Claims.ZK.PiopSrcSv
import Proofs.Flock.Claims.ZK.HiSalts

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
* **The six values** (`ligOodE`, `ligYbarE`, `ligY1E`, `ligSaltE`, `ligMsgE`, `ligSumsE`): the cut's source holds the
  cut's view's values, which at the coins are the executable's: the packed table's extension, the lane pads and folds,
  the tape's salts (HiSalts' `hiSalt_exe`), and the lane rounds' tables, whose basis is the PIOP's switched vectors'
  (`openB_basis`); so `simLig`.
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
noncomputable def ligRespE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) : Msg :=
  (if h : k < refK x then c ⟨k, h⟩ else default).bytes

theorem coinOf_ligRespE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (k : ℕ) :
    FlockComplete.coinOf (ligRespE c k) = fun i => slotAt c k i := by
  funext i
  unfold ligRespE slotAt
  rw [FlockComplete.JsonRT.coinOf_bytes]
  split_ifs
  · rfl
  · unfold FlockDeploy.Coin.at
    split <;> rfl

/-- **Table `j`'s rep `r` switched vectors at the opened coins**: the session's ring switch's, in closed form at the
replies `ligRespE c` from the rep's first round (`piopPZk_exact`). -/
noncomputable def swsE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim) : Array _root_.Flock.Switched :=
  FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog
      (FlockComplete.Replay.Zc.out (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val))) (trsE x w (UE.tapeZk τ))[j.val]!)
      (FlockComplete.Replay.Lc.out (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (trsE x w (UE.tapeZk τ))[j.val]!
        (FlockComplete.Replay.Zc.out (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
          (FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val))) (trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Lc.coinsAt (ligRespE c)
          (repBase x (2 * j.val + r.val) + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6))
          ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6))) ++ extras)
    (FlockComplete.Replay.Rs.coinsAt (ligRespE c)
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
it does at `asksS (ligRespE c)` (`piopPZk_cut_ok`), so it sent `piopZk` bodies from `repBase`, landing at `repSlot … 0`,
and returned `swsE` (`piopPZk_exact`). -/
theorem ligSrcE_of (h : LigHypE) : LigSrcE := by
  intro x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl zp lp ss T sws t' l' hp
  have hr : ∀ k < n, FlockComplete.coinOf (ligRespE c k) = (fun k i => slotAt c k i) k := fun k _ => coinOf_ligRespE c k
  have hp' := FlockComplete.LigLock.piopPZk_cut_ok _ n (ligRespE c) hr _ _ _ _ _ _ _ _ _ hp
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

/-! ### The out-of-domain sample

The deployed source's masked value at `lgOod 0` is the cut's clear value there plus the tape's pad
(`srcE_masked_pad_L`). The clear value is `Σ_y eqAt(z, y)·word(y)` at the drawn point `z` (`Composed.honestAt_lgOod`),
which is the executable's `mleF` of the packed trace at `zS` (`ligOodE`). -/

section Ood

/-- Table `j`'s pad layout at the executable's schedule. -/
private noncomputable abbrev layL (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Zk.PadLayout :=
  _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0

/-- Table `j`'s witness as the cut's view reads it: `w`'s share with the tape's mask words (`Composed.zWith`). -/
noncomputable abbrev zwL (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x)) :
    FlockSoundness.Model.Witness (tabC PE x j).S.m :=
  Composed.zWith (stAt x j) (wtOf PE x w j) ((UE.tape x τ).1 j).1

/-- **The deployed source's masked value at a sent index** `k < n_E` is the clear value the cut's view masks
(`Composed.honestAt`) plus the tape's pad `k` of table `j`'s rep `r`. -/
private theorem srcE_masked_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (tabC PE x j).z.nE) :
    ((srcE x c w τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
        (((UE.tape x τ).1 j).2.1.2.1 r) k) + (UE.tapeZk τ).h j.val r.val k := by
  obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = τ := ⟨τ, rfl⟩
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hz := fits_zkTab PE (zkTab x j)
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := hz.nE
  have tH : ∀ q : Fin (2 ^ 64 + 4), FlockLevel3.GF128.of ((UE.tapeZk t).h j.val r.val q.val) =
      (t ⟨j.val, hc⟩).1.2.2.1 r q := fun q => by
    show FlockLevel3.GF128.of (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.1 r) q.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]
    exact fAt_lt _ _ q.isLt
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w t) j ⟨r.val, hr⟩ else default).masked k = _
  rw [dite_eq_left r.isLt]
  show (if h : k < ((tabC PE x j).T (czExe PE x c j)).tab.nE then
      FlockLevel3.GF128.val ((cutE x c w t j).masked r ⟨k, h⟩) else 0) = _
  rw [dite_eq_left (show k < ((tabC PE x j).T (czExe PE x c j)).tab.nE from hk)]
  show FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w t j)
      (((UE.tape x t).1 j).2.1.2.1 r) k + (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨k, hk⟩))) = _
  rw [FlockLevel3.val_add]
  congr 1
  exact (congrArg FlockLevel3.GF128.val (tH ⟨k, by omega⟩)).symm

/-- **The deployed source's masked value below `h_ab`** is the clear value plus the session's pad (`padsOf`). -/
private theorem srcE_masked_pad_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (layL x j).hab) :
    ((srcE x c w τ j.val).rep r.val).masked k =
      FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
        (((UE.tape x τ).1 j).2.1.2.1 r) k) + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layL x j)).h[k]! := by
  have hE : (tabC PE x j).z.nE = (layL x j).hab := by
    show (layL x j).len - 4 = (layL x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen : (layL x j).len = (layL x j).tw + 1 := rfl
  have htw : (layL x j).tw = (layL x j).hab + 3 := rfl
  rw [srcE_masked_L x c w τ j r (k := k) (by omega),
    FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layL x j) k (by omega) (by omega) (by omega)]



/-- An equation in the field read as one between its values. -/
private theorem val_eq_of {a : FlockLevel3.GF128} {b : _root_.Flock.F128} (h : a = FlockLevel3.GF128.of b) :
    FlockLevel3.GF128.val a = b := by
  rw [h]
  rfl

/-- `eqAt` at points of equal dimension and coordinates. -/
private theorem eqAt_congr_fin {a b : ℕ} (h : a = b) (p : Fin a → FlockLevel3.GF128) (q : Fin b → FlockLevel3.GF128)
    (hpq : ∀ i (ha : i < a) (hb : i < b), p ⟨i, ha⟩ = q ⟨i, hb⟩) (y : ℕ) :
    FlockSoundness.Model.eqAt p y = FlockLevel3.eqAt q y := by
  subst h
  have : p = q := funext fun i => hpq i.val i.isLt i.isLt
  subst this
  rfl

/-- **The deployed witness's bits are its trace's**: table `j`'s share with the tape's mask words (`zwL`) at address `a`
is the bit the session's trace holds there. -/
theorem zwL_trace {x : Msg} (w : Claims.Flock.Wit) (τ : UE.Tape) (j : Fin (jOf x)) :
    ∀ a : Fin (2 ^ (zkTab x j).table.S.m),
      zwL x w τ j a = _root_.Flock.Readable.toZ ((trsE x w (UE.tapeZk τ))[j.val]!).z[a.val]! := by
  obtain ⟨t, rfl⟩ : ∃ t' : Fin Claims.Flock.zkTableCap → ExeCell, t' = τ := ⟨τ, rfl⟩
  have hj := lt_of_jOf j
  have htr : (trsE x w (UE.tapeZk t))[j.val]! = FlockComplete.traceOf (stAt x j)
      (_root_.Flock.Readable.tableWit (stsOf x).size (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask)
        j.val).hid.toArray 0 :=
    FlockComplete.ZkTables.traces_getElem _ _ hj
  have hzw : _root_.Flock.Readable.zOf 1 (_root_.Flock.Readable.tableWit (stsOf x).size
      (FlockComplete.ZkTables.maskWOf x w (UE.tapeZk t).mask) j.val) 0 (stAt x j).m = zwL x w t j :=
    (_root_.Flock.Readable.zOf_tableWit hj _ _).trans (zOf_ofLayout (TapeLayout.exe PE) exeTape x t w j)
  intro a
  rw [← hzw, htr, FlockComplete.Replay.traceOf_z _ _ a.isLt]
  simp [_root_.Flock.Readable.zOf]

/-- **Level 0's out-of-domain point's `eq` at the coins** is the executable's `eq` table at the reply to round `p₀`. -/
private theorem eqAt_oodPoint {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) {y : ℕ}
    (hy : y < 2 ^ ((zkTab x j).table.S.m - 7)) :
    FlockSoundness.Model.eqAt (Composed.oodPoint (zkTab x j).table (stAt x j) (czExe PE x c j) r 0) y =
      FlockLevel3.GF128.of (_root_.Flock.eqTable (FlockComplete.LigLock.zS (fun k i => slotAt c k i)
        (repSlot x (2 * j.val + r.val) fun _ => 0) (zkTab x j).sch.n))[y]! := by
  obtain ⟨h25, -, hsm, -, -, hdim, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨hn7, -⟩ := hdim
  have hm : (zkTab x j).table.S.m = (stAt x j).m := rfl
  have h0 : 0 < (Composed.layOf (zkTab x j).table (stAt x j)).nOod0 := by
    show 0 < (layL x j).nOod0
    simp [layL, _root_.Flock.Zk.PadLayout.new]
  have hzs : (FlockComplete.LigLock.zS (fun k i => slotAt c k i) (repSlot x (2 * j.val + r.val) fun _ => 0)
      (zkTab x j).sch.n).size = (zkTab x j).table.S.m - 7 := by
    simp [FlockComplete.LigLock.zS]
    omega
  rw [FlockLevel3.eqTable_eqAt _ _ (by rw [hzs]; exact hy)]
  refine eqAt_congr_fin hzs.symm _ _ (fun i ha hb => ?_) y
  unfold Composed.oodPoint
  rw [dite_eq_left ⟨h0, by omega⟩]
  have hi : i < (zkTab x j).sch.n := by omega
  unfold FlockComplete.LigLock.zS
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, Array.getElem_range]
  rfl

theorem ligOodE : LigOodE := by
  intro x hx w hw τ c n hn s spec hs j r extras he
  have hst := tabsOf_stAt x j
  have hlay : ligLay x j = layL x j := by
    unfold ligLay layL ligSt
    rw [hst]
    rfl
  obtain ⟨h25, h35, hsm, hk0, -, hdim, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨hn7, -⟩ := hdim
  have hm : (zkTab x j).table.S.m = (stAt x j).m := rfl
  unfold ligSrc ligPad ligTab ligCoin ligP0
  rw [hlay]
  have hk : (layL x j).lgOod 0 < (layL x j).hab := by
    simp only [layL, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]
    omega
  rw [srcE_masked_pad_L x c w τ j r hk]
  congr 1
  have hlo : Composed.layOf (zkTab x j).table (stAt x j) = layL x j := rfl
  have h0 : 0 < (Composed.layOf (zkTab x j).table (stAt x j)).nOod0 := by
    rw [hlo]
    simp [layL, _root_.Flock.Zk.PadLayout.new]
  have e := Composed.honestAt_lgOod (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r) 0 h0
  rw [hlo] at e
  rw [e]
  have hM7 : 7 ≤ (stAt x j).m := by omega
  have hmL : (ligSt x j).m = (stAt x j).m := by
    unfold ligSt
    rw [hst]
    rfl
  have hsz : (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (ligSt x j).m).size =
      2 ^ ((zkTab x j).table.S.m - 7) := by
    rw [FlockComplete.JsonRT.packed_size, hmL, hm]
  apply val_eq_of
  simp only [FlockComplete.mleF]
  rw [FlockComplete.Replay.Zc.of_sumF, hsz]
  refine Finset.sum_congr rfl fun y hy => ?_
  rw [Finset.mem_range] at hy
  have hwd : Composed.wordOf (zwL x w τ j) y =
      FlockSoundness.Refine.fOf (FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (ligSt x j).m)[y]! := by
    rw [hmL]
    exact FlockComplete.wordOf_packed _ _ (zwL_trace w τ j) hy hM7
  rw [eqAt_oodPoint c j r hy]
  refine (congrArg (_ * ·) hwd).trans ?_
  exact (mul_comm _ _).trans (FlockSoundness.Refine.fOf_mul _ _).symm

end Ood

/-! ### The restart's `ȳ`

The deployed source's `ȳ` is the cut's view's, the table's `ȳ` of the tape's paddings (`srcE_ybar_L`, unguarded by
the cut). Its lane sum is the executable's fold under `eqTableK` of the lane challenges (`eqTableK_eqList`, the lane
coins at `p₀ + 2 + l`), and its pair's terms are `rhoS`'s times the extra lanes' paddings (`rhoAt` at `p₀ + 2 + k₀`)
(`ligYbarE`). -/

section Ybar

/-- **The deployed source's `ȳ`** is the cut's view's: the table's `ȳ` of the tape's lane paddings and extra lanes'. -/
private theorem srcE_ybar_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) :
    ((srcE x c w τ j.val).rep r.val).ybar = Array.ofFn fun i =>
      kVal (((tabC PE x j).T (czExe PE x c j)).tab.ybar ((UE.tape x τ).1 j).2.1.1 ((UE.tape x τ).1 j).2.1.2.2 r i) := by
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩ else default).ybar = _
  rw [dite_eq_left r.isLt]
  rfl

/-- **The table's `ȳ`**: the lane paddings under `eq(r̄)` plus the pair's coins times the extra lanes' paddings. -/
private theorem tab_ybar_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2)
    (μ0 : Fin (tabC PE x j).z.nL → Fin (tabC PE x j).z.tpad → FlockLevel3.GF128)
    (μR : Fin 2 → Fin 2 → Fin (tabC PE x j).z.tpad → FlockLevel3.GF128) (i : Fin (tabC PE x j).z.tpad) :
    ((tabC PE x j).T (czExe PE x c j)).tab.ybar μ0 μR r i =
      ∑ l, Composed.eqLane ((czExe PE x c j).rest.lane r) l.val * FlockLevel3.embed (μ0 l i) +
        (FlockLevel3.GF256.of (rhoAt c j.val r.val).1 * FlockLevel3.embed (μR r 0 i) +
          FlockLevel3.GF256.of (rhoAt c j.val r.val).2 * FlockLevel3.embed (μR r 1 i)) := rfl

private theorem kVal_eq_of {a : FlockLevel3.GF256} {b : _root_.Flock.F256} (h : a = FlockSoundness.Refine.gOf b) :
    kVal a = b := by
  rw [h]
  rfl

private theorem gOf_foldl_smul (E : Array _root_.Flock.F256) (lp : ℕ → _root_.Flock.F128) :
    ∀ (L : List ℕ) (a : _root_.Flock.F256),
      FlockSoundness.Refine.gOf (L.foldl (fun acc l => acc + (E[l]!).smul (lp l)) a) =
        FlockSoundness.Refine.gOf a + (L.map fun l =>
          FlockSoundness.Refine.gOf E[l]! * FlockLevel3.embed (FlockSoundness.Refine.fOf (lp l))).sum
  | [], a => by simp
  | l :: L, a => by
    rw [List.foldl_cons, gOf_foldl_smul E lp L, FlockSoundness.Refine.gOf_add, FlockSoundness.Refine.gOf_smul,
      List.map_cons, List.sum_cons, add_assoc]

theorem sum_fin_range {M : Type} [AddCommMonoid M] {a b : ℕ} (h : a = b) (f : Fin a → M) (g : ℕ → M)
    (hfg : ∀ l (hl : l < a), f ⟨l, hl⟩ = g l) : ∑ l, f l = ∑ l ∈ Finset.range b, g l := by
  subst h
  rw [← Fin.sum_univ_eq_sum_range]
  exact Finset.sum_congr rfl fun l _ => hfg l l.isLt

theorem prod_fin_range {M : Type} [CommMonoid M] {a b : ℕ} (h : a = b) (f : Fin a → M) (g : ℕ → M)
    (hfg : ∀ l (hl : l < a), f ⟨l, hl⟩ = g l) : ∏ l, f l = ∏ l ∈ Finset.range b, g l := by
  subst h
  rw [← Fin.prod_univ_eq_prod_range]
  exact Finset.prod_congr rfl fun l _ => hfg l l.isLt

/-- Table `j`'s rep `r`'s rounds past its PIOP, from `p₀`. -/
theorem repSlot_p0 {x : Msg} {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (f : _root_.Flock.Schedule → ℕ) :
    repSlot x (2 * j.val + r.val) f = ligP0 x j r + f (zkTab x j).sch := by
  have hz' : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  unfold ligP0
  rw [repSlot_eq hs hz' f, repSlot_eq hs hz' fun _ => 0, Nat.add_zero]

theorem ligYbarE : LigYbarE := by
  intro x hx w hw τ c n hn s spec hs j r extras he
  dsimp only
  intro _ _
  obtain ⟨hnL, -, htp, -, -, -⟩ := tabC_z PE x j
  obtain ⟨-, -, hl0, -, -, -⟩ := zkLevel0 (zkTab x j).hz
  unfold ligSrc
  rw [srcE_ybar_L]
  apply Array.ext
  · simp only [Array.size_ofFn, Array.size_map, Array.size_range]
    exact htp
  · intro i h1 h2
    simp only [Array.size_map, Array.size_range] at h2
    have hi : i < (tabC PE x j).z.tpad := by rw [htp]; exact h2
    rw [Array.getElem_ofFn, Array.getElem_map, Array.getElem_range]
    apply kVal_eq_of
    refine (tab_ybar_L x c j r _ _ _).trans ?_
    rw [gOf_foldl_smul, FlockSoundness.Refine.gOf_add, FlockSoundness.Refine.gOf_smul,
      FlockSoundness.Refine.gOf_smul, FlockLevel3.list_sum_range]
    refine (add_comm _ _).trans (congrArg₂ (· + ·) ?_ ?_)
    · have hρ : rhoAt c j.val r.val = FlockComplete.LigLock.rhoS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.k0 := by
        unfold rhoAt rhoSlot
        rw [repSlot_p0 hs j r]
        unfold FlockComplete.LigLock.rhoS ligCoin
        rw [show ligP0 x j r + ((zkTab x j).sch.k0 + 2) = ligP0 x j r + 2 + (zkTab x j).sch.k0 by omega,
          show ligP0 x j r + 2 + (zkTab x j).sch.k0 + 1 = ligP0 x j r + 3 + (zkTab x j).sch.k0 by omega]
      have e0 := (agrees0_exe PE).extPad x τ j r 0 ⟨i, hi⟩
      have e1 := (agrees0_exe PE).extPad x τ j r 1 ⟨i, hi⟩
      simp only [Fin.val_zero, Fin.val_one, Nat.add_zero, lv0] at e0 e1
      rw [← Nat.add_assoc] at e1
      rw [hρ]
      exact congrArg₂ (· + ·) (congrArg₂ (· * ·) rfl (congrArg FlockLevel3.embed e0))
        (congrArg₂ (· * ·) rfl (congrArg FlockLevel3.embed e1))
    · refine sum_fin_range (by rw [hnL]; rfl) _ _ fun l hl => ?_
      have hp := (agrees0_exe PE).pad x τ j ⟨l, hl⟩ ⟨i, hi⟩
      refine congrArg₂ (· * ·) ?_ (congrArg FlockLevel3.embed hp)
      unfold lv0 at hnL
      have hl' : l < 2 ^ (zkTab x j).sch.k0 := by rw [← hl0, ← hnL]; exact hl
      rw [FlockSoundness.Refine.eqTableK_eqList _ l (by simpa using hl'), FlockSoundness.Refine.eqList_eq]
      unfold Composed.eqLane
      refine prod_fin_range (by simp; rfl) _ _ fun q hq => ?_
      have hq' : q < (zkTab x j).sch.k0 := hq
      rw [List.getD_eq_getElem _ _ (by simpa using hq')]
      simp only [List.getElem_map, Array.getElem_toList, Array.getElem_map, Array.getElem_range]
      have e : (czExe PE x c j).rest.lane r ⟨q, hq⟩ = FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r) q := by
        show kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + q) = _
        rw [repSlot_p0 hs j r]
        simp only [kAt, FlockComplete.LigLock.chS, ligCoin, Nat.add_assoc]
      rw [e]

end Ybar

/-! ### The restart's `y₁'`

Where the pair's coins are answered (`hlt`) the restart is sent (`famPasses_of`, guard slot `ρ`'s plus two), so the
deployed source's `y₁'` is the cut's view's, the table's `y₁'` of the witness lanes and the extra lanes (`srcE_y1_L`).
Its lane sum is the executable's iterated fold of the packed table (`tabS_foldKs`, `foldLoL_eq_sum`), whose entries are
the witness's lane words (`wordOf_packed`), and its pair's terms are `rhoS`'s times the extra lanes (`ligY1E`). -/

section Y1

/-- **The deployed source's `y₁'`** where the rep's restart is sent: the table's `y₁'` of the table witness, the tape's
mask slot and extra lanes. -/
private theorem srcE_y1_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2)
    (hS : famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
      j r .restart) :
    ((srcE x c w τ j.val).rep r.val).y1 = Array.ofFn fun i =>
      kVal (((tabC PE x j).T (czExe PE x c j)).tab.y1' (wtOf PE x w j) ((UE.tape x τ).1 j).1
        ((UE.tape x τ).1 j).2.1.2.1 r i) := by
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩ else default).y1 = _
  rw [dite_eq_left r.isLt]
  have e : (cutE x c w τ j).y1 r = some (((tabC PE x j).T (czExe PE x c j)).tab.y1' (wtOf PE x w j)
      ((UE.tape x τ).1 j).1 ((UE.tape x τ).1 j).2.1.2.1 r) := ite_eq_left hS
  unfold repSrcCut
  dsimp only
  rw [e]

/-- **The table's `y₁'`**: the witness lanes under `eq(r̄)` plus the pair's coins times the extra lanes. -/
theorem tab_y1_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (i : Fin (tabC PE x j).z.L) :
    ((tabC PE x j).T (czExe PE x c j)).tab.y1' (wtOf PE x w j) ((UE.tape x τ).1 j).1 ((UE.tape x τ).1 j).2.1.2.1 r i =
      ∑ l, Composed.eqLane ((czExe PE x c j).rest.lane r) l.val *
          FlockLevel3.embed (((tabC PE x j).T (czExe PE x c j)).lanes (wtOf PE x w j) ((UE.tape x τ).1 j).1 l i) +
        (FlockLevel3.GF256.of (rhoAt c j.val r.val).1 * FlockLevel3.embed (((UE.tape x τ).1 j).2.1.2.1 r 0 i) +
          FlockLevel3.GF256.of (rhoAt c j.val r.val).2 * FlockLevel3.embed (((UE.tape x τ).1 j).2.1.2.1 r 1 i)) := rfl

/-- **`tabS` is the executable's iterated fold**: the table after `k` lane challenges, lowest variable first. -/
theorem tabS_foldKs (coin : ℕ → ℕ → _root_.Flock.F128) (p0 : ℕ) (t : Array _root_.Flock.F128) :
    ∀ k, FlockComplete.LigLock.tabS coin p0 t k =
      FlockComplete.Replay.Lg.foldKs ((List.range k).map (FlockComplete.LigLock.chS coin p0)) (FlockComplete.embedF t)
  | 0 => rfl
  | k + 1 => by
    rw [FlockComplete.LigLock.tabS, tabS_foldKs coin p0 t k, List.range_succ, List.map_append]
    unfold FlockComplete.Replay.Lg.foldKs
    rw [List.foldl_append]
    rfl

theorem ligY1E : LigY1E := by
  intro x hx w hw τ c n hn s spec hs j r extras he
  dsimp only
  intro hlt _
  have hS := famPasses_of (pointReads_czExe PE) (coinReads_czExe PE) hx hn hs j r .restart (by
    show rhoSlot x j.val r.val + 2 ≤ n
    unfold rhoSlot
    rw [repSlot_p0 hs j r]
    omega)
  obtain ⟨hnL, hL, -, -, -, hm⟩ := tabC_z PE x j
  obtain ⟨-, -, hl0, -, -, -⟩ := zkLevel0 (zkTab x j).hz
  unfold lv0 at hnL hL hm
  rw [hl0] at hnL hm
  have hmL : (ligSt x j).m = (tabC PE x j).S.m := by
    unfold ligSt
    rw [tabsOf_stAt]
    rfl
  have hesz : (FlockComplete.embedF (ligTab x w τ j)).size = 2 ^ ((tabC PE x j).S.m - 7) := by
    simp [FlockComplete.embedF, ligTab, FlockComplete.JsonRT.packed_size, hmL]
  have hpow : 2 ^ ((tabC PE x j).S.m - 7) =
      2 ^ (zkTab x j).sch.k0 * 2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by
    rw [← pow_add]
    congr 1
    omega
  have hdiv : 2 ^ ((tabC PE x j).S.m - 7) / 2 ^ (zkTab x j).sch.k0 =
      2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by
    rw [hpow, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  have hlen : (((List.range (zkTab x j).sch.k0).map (FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r))).map
      FlockSoundness.Refine.gOf).length = (zkTab x j).sch.k0 := by simp
  have htsz : (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0).size =
      2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by
    rw [tabS_foldKs, FlockComplete.Replay.Lg.foldKs_size, hesz]
    simpa using hdiv
  unfold ligSrc
  rw [srcE_y1_L x c w τ j r hS]
  apply Array.ext
  · simp only [Array.size_ofFn, Array.size_map, Array.size_range, htsz]
    exact hL
  · intro i h1 h2
    simp only [Array.size_map, Array.size_range, htsz] at h2
    have hi : i < (tabC PE x j).z.L := by rw [hL]; exact h2
    rw [Array.getElem_ofFn, Array.getElem_map, Array.getElem_range]
    apply kVal_eq_of
    refine (tab_y1_L x c w τ j r ⟨i, hi⟩).trans ?_
    rw [FlockSoundness.Refine.gOf_add, FlockSoundness.Refine.gOf_add, FlockSoundness.Refine.gOf_smul,
      FlockSoundness.Refine.gOf_smul, add_assoc]
    have hit : i < (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j)
        (zkTab x j).sch.k0).size := by rw [htsz]; exact h2
    refine congrArg₂ (· + ·) ?_ ?_
    · show _ = FlockComplete.Replay.Lg.gv _ i
      rw [tabS_foldKs, FlockComplete.Replay.Lg.gv_foldKs _ _ i (by
          rw [hesz, List.length_map, List.length_range, hdiv]; exact h2),
        FlockSoundness.foldLoL_eq_sum, hlen]
      refine sum_fin_range hnL _ _ fun l hl => ?_
      refine congrArg₂ (· * ·) ?_ ?_
      · rw [FlockSoundness.Refine.eqList_eq]
        unfold Composed.eqLane
        refine prod_fin_range (by rw [hlen]; rfl) _ _ fun q hq => ?_
        have hq' : q < (zkTab x j).sch.k0 := hq
        rw [List.getD_eq_getElem _ _ (by simpa using hq')]
        simp only [List.getElem_map, List.getElem_range]
        have e : (czExe PE x c j).rest.lane r ⟨q, hq⟩ = FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r) q := by
          show kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + q) = _
          rw [repSlot_p0 hs j r]
          simp only [kAt, FlockComplete.LigLock.chS, ligCoin, Nat.add_assoc]
        rw [e]
      · have hl' : l < 2 ^ (zkTab x j).sch.k0 := by rw [← hnL]; exact hl
        have hy : l + 2 ^ (zkTab x j).sch.k0 * i < 2 ^ ((tabC PE x j).S.m - 7) := by
          have hc : 2 ^ (zkTab x j).sch.k0 * (i + 1) ≤
              2 ^ (zkTab x j).sch.k0 * 2 ^ ((zkTab x j).sch.levels.getD 0 default).cols :=
            Nat.mul_le_mul_left _ h2
          rw [hpow]
          rw [Nat.mul_succ] at hc
          omega
        show FlockLevel3.embed (Composed.wordOf (zwL x w τ j) (l + (tabC PE x j).z.nL * i)) = _
        rw [FlockComplete.Replay.Lg.gv_embedF _ (by rw [FlockComplete.JsonRT.packed_size, hmL]; exact hy)]
        refine congrArg FlockLevel3.embed ?_
        have hM7 : 7 ≤ (zkTab x j).table.S.m := by
          have e : (zkTab x j).table.S.m = (tabC PE x j).S.m := rfl
          omega
        unfold ligTab
        rw [hnL, hmL]
        exact FlockComplete.wordOf_packed _ _ (zwL_trace w τ j) hy hM7
    · have hρ : rhoAt c j.val r.val = FlockComplete.LigLock.rhoS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.k0 := by
        unfold rhoAt rhoSlot
        rw [repSlot_p0 hs j r]
        unfold FlockComplete.LigLock.rhoS ligCoin
        rw [show ligP0 x j r + ((zkTab x j).sch.k0 + 2) = ligP0 x j r + 2 + (zkTab x j).sch.k0 by omega,
          show ligP0 x j r + 2 + (zkTab x j).sch.k0 + 1 = ligP0 x j r + 3 + (zkTab x j).sch.k0 by omega]
      have e0 := (agrees0_exe PE).ext x τ j r 0 ⟨i, hi⟩
      have e1 := (agrees0_exe PE).ext x τ j r 1 ⟨i, hi⟩
      simp only [Fin.val_zero, Fin.val_one, Nat.add_zero] at e0 e1
      rw [hρ, FlockSoundness.Refine.getElem!_range_map _ hit, FlockSoundness.Refine.getElem!_range_map _ hit]
      exact congrArg₂ (· + ·) (congrArg₂ (· * ·) rfl (congrArg FlockLevel3.embed e0))
        (congrArg₂ (· * ·) rfl (congrArg FlockLevel3.embed e1))

end Y1

/-! ### Levels ≥ 1's salts

Where the restart is sent its levels ≥ 1 value is sent too, `PE`'s salts as one string (`saltBlob` of `hiSalt` at the
tape's levels ≥ 1 salts, `srcE_hiSalt_L`), so the source's salt at a level `≥ 1` and position of the schedule is
`hiSalt`'s fitted to 192 bytes (`blobSalt_saltBlob`), the tape's by HiSalts' `hiSalt_exe` (`HiSaltE`), and the deployed
scheme's salts are 192 bytes (`ligSaltE`). -/

section Salt

/-- **Levels ≥ 1's salts at the executable's layout**: rep `r`'s levels ≥ 1 salt that `hiSalt` reads at a level `l ≥ 1`
and position `p` of the schedule is the masked prover's tape's. -/
def HiSaltE : Prop :=
  ∀ (x : Msg) (t : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (l p : ℕ), HiAt (zkTab x j).table.sch.levels l p →
    hiSalt (zkTab x j).table.sch.levels (((UE.tape x t).1 j).2.2.2.2.2.2.2.2.2 r) l p = (exeTape t).salt j.val r.val l p

/-- HiSalts' `hiSalt_exe` at `PE`. -/
theorem hiSaltE : HiSaltE := fun x t j r _ _ h => hiSalt_exe PE x t j r h

private theorem fitBytes_fit192 (b : ByteArray) :
    FlockComplete.fitBytes 192 (fit192 b) = FlockComplete.fitBytes 192 b := by
  rw [fitBytes_of_size (LigPf.fit192_size b)]
  unfold fit192 FlockComplete.fitBytes
  rw [show 192 - min 192 b.size = 192 - b.size by omega]

/-- **The deployed source's levels ≥ 1 salts** where the rep's restart is sent: read back from `PE`'s string of the
tape's levels ≥ 1 salts. -/
private theorem srcE_hiSalt_L (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2)
    (hS : famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab) (fun j => ((tabC PE x j).T (czExe PE x c j)).rep)
      j r .restart) :
    ((srcE x c w τ j.val).rep r.val).hiSalt = blobSalt (zkTab x j).table.sch.levels
      (saltBlob (zkTab x j).table.sch.levels
        (hiSalt (zkTab x j).table.sch.levels (((UE.tape x τ).1 j).2.2.2.2.2.2.2.2.2 r))) := by
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩ else default).hiSalt = _
  rw [dite_eq_left r.isLt]
  have e : (cutE x c w τ j).lig r = some (((tabC PE x j).T (czExe PE x c j)).tab.lig r
      (((tabC PE x j).T (czExe PE x c j)).tab.y1' (wtOf PE x w j) ((UE.tape x τ).1 j).1 ((UE.tape x τ).1 j).2.1.2.1 r)
      (((UE.tape x τ).1 j).2.2.2.2.2.2.2.2.2 r)) := ite_eq_left hS
  unfold repSrcCut
  dsimp only
  rw [e]
  rfl

theorem ligSaltE (hhi : HiSaltE) : LigSaltE := by
  intro x hx w hw τ c n hn s spec hs j r extras he
  dsimp only
  intro hlt _ i hi1 hi2 q hq
  have hS := famPasses_of (pointReads_czExe PE) (coinReads_czExe PE) hx hn hs j r .restart (by
    show rhoSlot x j.val r.val + 2 ≤ n
    unfold rhoSlot
    rw [repSlot_p0 hs j r]
    omega)
  obtain ⟨-, -, -, -, -, -, hseam, -, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hat : HiAt (zkTab x j).table.sch.levels i q := by
    have e : (zkTab x j).table.sch.levels.getD i default =
        FlockSoundness.Refine.levelOf ((zkTab x j).sch.levels.getD i default) := by
      show ((zkTab x j).sch.levels.toList.map FlockSoundness.Refine.levelOf).getD i default = _
      simp [List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?, hi2]
    refine ⟨hi1, ?_, ?_⟩
    · show i < ((zkTab x j).sch.levels.toList.map FlockSoundness.Refine.levelOf).length
      simpa using hi2
    · rw [e, ← (hseam i hi2).2.2.2.2]
      exact hq
  have hms : (ligSt x j).merkle.saltBytes = 192 := by
    obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
    have hj : j.val < (stsOf x).size := lt_of_jOf j
    have hm : (ligSt x j).merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
      unfold ligSt
      rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
      simp only [FlockDeploy.tabsOf, Array.getElem_map]
      exact (hall j.val hj).2.2
    rw [hm]
    rfl
  unfold ligSrc
  rw [srcE_hiSalt_L x c w τ j r hS, LigPf.blobSalt_saltBlob hat, hhi x τ j r i q hat, hms]
  exact fitBytes_fit192 _

end Salt

/-! ### The opening's basis

At the coins, the model's batched basis (`Composed.openB`) is the executable's basis of the PIOP's switched vectors
(`ligBasis`, `fOf_basisVals`): term by term, the ring switch's `γ` and `rpp` coins are the coins at the rep's slots,
and the claims' points past their first coordinate are the model's (`abTail`, `cTail`, and the table's extra claims'
region points, `extras_pt`) (`openB_basis`). -/

section Basis

/-- **The deployed table's extra claims**: claim `i` has `m − 6` coordinates, and past its first it is region `i / 2`'s
point at link point `i % 2` of the coins (`regionPtOf`). -/
private theorem extras_pt {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin) {s spec}
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) {extras : Array _root_.Flock.Claim}
    (he : (stAt x j).extraClaims (ptsE c spec) = .ok extras) (i : ℕ) (hi : i < 2 * (stAt x j).regions.size) :
    (extras[i]!).x.size = (stAt x j).m - 6 ∧
      ((extras[i]!).x.extract 1).toList = Composed.regionPtOf (stAt x j)
        ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link ⟨i, hi⟩ := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm14 : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  set ρ := ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) with hρdef
  obtain ⟨hpts, hmp⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hmP : (stAt x j).mPts = spec.mPts := hmp j.val hj
  have hmx : mPtsOf x = spec.mPts := by unfold mPtsOf; rw [hs]
  obtain ⟨ess, hess, heq⟩ := FlockSoundness.Refine.extraClaims_ok (stAt x j) (ptsE c spec) extras he
  have hlink : (ptsE c spec).toList = [Array.ofFn (ρ.link 0), Array.ofFn (ρ.link 1)] := by
    have hp : ∀ p : Fin 2, (Array.range spec.mPts).map (fun i => slotAt c 0 (p.val * spec.mPts + i)) =
        Array.ofFn (ρ.link p) := by
      intro p
      apply Array.ext
      · simp [hmP]
      · intro i h1 h2
        rw [Array.getElem_map, Array.getElem_range, Array.getElem_ofFn]
        show slotAt c 0 (p.val * spec.mPts + i) = slotAt c 0 (p.val * mPtsOf x + i)
        rw [hmx]
    apply List.ext_getElem
    · simp [ptsE, hpts]
    · intro i h1 h2
      have hi2 : i < 2 := by simpa [ptsE, hpts] using h1
      rw [Array.getElem_toList]
      simp only [ptsE, Array.getElem_map, Array.getElem_range]
      interval_cases i
      · exact hp 0
      · exact hp 1
  rw [hlink] at hess
  have hxe := extras_x (stAt x j) _ _ _ ess hess i (by rw [Array.length_toList]; exact hi)
  have hget : extras[i]! = ess.flatten[i]! := by
    rw [← heq, getElem!_def, getElem!_def, Array.getElem?_toList]
  rw [← hget] at hxe
  have hreg : (stAt x j).regions.toList[i / 2]! = (stAt x j).regions[i / 2]! := by
    rw [getElem!_def, getElem!_def, Array.getElem?_toList]
  have hif : (if i % 2 = 0 then Array.ofFn (ρ.link 0) else Array.ofFn (ρ.link 1)) =
      Array.ofFn (ρ.link (if i % 2 = 0 then 0 else 1)) := by
    split <;> rfl
  rw [hreg, hif] at hxe
  refine ⟨?_, ?_⟩
  · rw [hxe, Array.size_extract, FlockSoundness.Refine.regionPoint_size]; omega
  · rw [hxe, extract_six_one]
    rfl

/-- Past its first coordinate, entry `i` of a point is entry `i + 1`. -/
private theorem extract_one_get (a : Array _root_.Flock.F128) {i : ℕ} (hi : i + 1 < a.size) :
    (a.extract 1 a.size)[i]! = a[i + 1]! := by
  rw [getElem!_pos _ i (by rw [Array.size_extract]; omega), Array.getElem_extract,
    ← getElem!_pos a (1 + i) (by omega), Nat.add_comm]

/-- `eqAt` at points of equal dimension and coordinates. -/
private theorem eqAt_congr_fin' {a b : ℕ} (h : a = b) (p : Fin a → FlockLevel3.GF128) (q : Fin b → FlockLevel3.GF128)
    (hpq : ∀ i (ha : i < a) (hb : i < b), p ⟨i, ha⟩ = q ⟨i, hb⟩) (y : ℕ) :
    FlockSoundness.Model.eqAt p y = FlockSoundness.Model.eqAt q y := by
  subst h
  have : p = q := funext fun i => hpq i.val i.isLt i.isLt
  subst this
  rfl

/-- **The opening's basis is the deployed basis** at any extra claims two per region, at the region points of the link
(`hpt`, as `extras_pt` reads the statement's). -/
theorem openB_basisOf {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim)
    (hesz : extras.size = (stAt x j).regions.size * 2)
    (hpt : ∀ (i : ℕ) (hi : i < 2 * (stAt x j).regions.size), (extras[i]!).x.size = (stAt x j).m - 6 ∧
      ((extras[i]!).x.extract 1).toList = Composed.regionPtOf (stAt x j)
        ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link ⟨i, hi⟩)
    (y : ℕ) (hy : y < 2 ^ ((stAt x j).m - 7)) :
    Composed.openB (zkTab x j).table (stAt x j) (czExe PE x c j) r y =
      FlockSoundness.Refine.fOf (ligBasis x c w τ j r extras)[y]! := by
  have hst := tabsOf_stAt x j
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
  set zc := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr with hzc
  set lc := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr zc (lcE c j r) with hlc
  set A := _root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog zc lc with hA
  have hA2 : A.size = 2 := rfl
  set L := repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).c.kLog - 6) with hL
  set cs := FlockComplete.Replay.Rs.coinsAt (ligRespE c) L (2 + extras.size) with hcs
  have hzc' : FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val)) = zcE c j r := by
    unfold FlockComplete.Replay.Zc.coinsAt
    simp only [coinOf_ligRespE]
    rfl
  have hlc' : FlockComplete.Replay.Lc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6))
      ((stAt x j).c.kLog - 6) = lcE c j r := by
    unfold FlockComplete.Replay.Lc.coinsAt
    simp only [coinOf_ligRespE]
    rfl
  have hsws : swsE x c w τ j r extras = FlockComplete.Replay.Rs.sws (A ++ extras) cs := by
    unfold swsE
    rw [hst]
    show FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
      (FlockComplete.Replay.Zc.out (stAt x j).m
        (FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val))) tr)
      (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
        (FlockComplete.Replay.Zc.out (stAt x j).m
          (FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val))) tr)
        (FlockComplete.Replay.Lc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6))
          ((stAt x j).c.kLog - 6))) ++ extras) cs = _
    rw [hzc', hlc']
  have hcl : (A ++ extras).size = 2 + extras.size := by rw [Array.size_append, hA2]
  have hsize : (swsE x c w τ j r extras).size = 2 + 2 * (stAt x j).regions.size := by
    rw [hsws]
    simp [FlockComplete.Replay.Rs.sws, hcl, hesz, Nat.mul_comm]
  have hsw : ∀ k < (swsE x c w τ j r extras).size, (swsE x c w τ j r extras)[k]! =
      { gamma := (FlockComplete.Replay.Rs.gamma cs (A ++ extras).size)[k]!, rpp := cs.rpp k,
        z := (A ++ extras)[k]!.x.extract 1 (A ++ extras)[k]!.x.size } := by
    intro k hk
    rw [hsws] at hk ⊢
    have hk' : k < (A ++ extras).size := by simpa [FlockComplete.Replay.Rs.sws] using hk
    rw [FlockComplete.Replay.Rs.sws, FlockSoundness.Refine.getElem!_range_map _ hk', FlockComplete.Replay.Rs.perClaim,
      FlockComplete.Replay.pushes_get _ _ k hk']
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm14 : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hA0 : (A[0]!).x.size = (stAt x j).m - 6 := by
    show ((FlockComplete.Replay.pushes (lcE c j r).τ ((stAt x j).circuitFold.kLog - 6)).reverse ++
      (FlockComplete.Replay.pushes (zcE c j r).ρ ((stAt x j).m - 6)).extract ((stAt x j).c.kLog - 6)
        ((stAt x j).m - 6)).size = (stAt x j).m - 6
    rw [Array.size_append, Array.size_reverse, FlockComplete.Replay.pushes_size, Array.size_extract,
      FlockComplete.Replay.pushes_size, Nat.min_self]
    have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
    omega
  have hA1 : (A[1]!).x.size = (stAt x j).m - 6 := FlockComplete.Replay.Zc.r_size (stAt x j).m (by omega) (zcE c j r)
  have hx6 : ∀ k < (A ++ extras).size, ((A ++ extras)[k]!).x.size = (stAt x j).m - 6 := by
    intro k hk
    rcases Nat.lt_or_ge k 2 with hk2 | hk2
    · rw [FlockComplete.Replay.get_left (show k < A.size by rw [hA2]; exact hk2)]
      interval_cases k
      · exact hA0
      · exact hA1
    · rw [FlockComplete.Replay.get_right (show A.size ≤ k by rw [hA2]; omega), hA2]
      exact (hpt (k - 2) (by rw [hcl, hesz] at hk; omega)).1
  have hmL : (ligSt x j).m = (stAt x j).m := by
    unfold ligSt
    rw [hst]
    rfl
  have hrpp : ∀ k < (swsE x c w τ j r extras).size, ((swsE x c w τ j r extras)[k]!).rpp.size = 7 := by
    intro k hk
    rw [hsw k hk]
    simp [FlockComplete.Replay.Rs.Challenges.rpp]
  have hz : ∀ k < (swsE x c w τ j r extras).size, ((swsE x c w τ j r extras)[k]!).z.size = (ligSt x j).m - 7 := by
    intro k hk
    have hk' : k < (A ++ extras).size := by rw [hcl, hesz]; rw [hsize] at hk; omega
    rw [hsw k hk]
    simp only [Array.size_extract, Nat.min_self, hx6 k hk', hmL]
    omega
  unfold ligBasis
  rw [FlockComplete.Replay.Lg.fOf_basisVals _ _ hrpp hz y (by rw [hmL]; exact hy)]
  unfold Composed.openB
  symm
  refine Fintype.sum_equiv (finCongr hsize) _ _ fun k => ?_
  have hk : k.val < (A ++ extras).size := by
    have := lt_of_lt_of_eq k.isLt hsize
    rw [hcl, hesz]
    omega
  rw [hsw k.val k.isLt]
  refine congrArg₂ (· * ·) ?_ (congrArg₂ (FlockSoundness.Model.phi FlockSoundness.execArith) ?_ ?_)
  · have hz' : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
        .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
      rw [stsOf_half]
      exact (zkTab x j).hz
    have hslot : L + (2 + extras.size) = repSlot x (2 * j.val + r.val) (fun _ => 0) - 1 := by
      rw [repSlot_eq hs hz', stsOf_half, hL, repBase_eq hs]
      unfold FlockComplete.piopRoundsZk
      rw [hesz, hpts]
      omega
    show FlockSoundness.Refine.fOf (FlockComplete.Replay.Rs.gamma cs (A ++ extras).size)[k.val]! =
      FlockSoundness.Refine.fOf (slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 0) - 1) k.val)
    unfold FlockComplete.Replay.Rs.gamma
    rw [FlockSoundness.Refine.getElem!_range_map _ hk, hcs]
    show FlockSoundness.Refine.fOf (FlockComplete.coinOf (ligRespE c (L + (2 + extras.size))) k.val) = _
    rw [coinOf_ligRespE, hslot]
  · funext i
    show FlockSoundness.Refine.fOf (cs.rpp k.val)[i.val]! = FlockSoundness.Refine.fOf (slotAt c
      (repBase x (2 * j.val + r.val) + (6 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) + k.val)) i.val)
    unfold FlockComplete.Replay.Rs.Challenges.rpp
    rw [FlockSoundness.Refine.getElem!_range_map _ i.isLt, hcs]
    show FlockSoundness.Refine.fOf (FlockComplete.coinOf (ligRespE c (L + k.val)) i.val) = _
    rw [coinOf_ligRespE, hL]
    rw [show repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).c.kLog - 6) + k.val =
      repBase x (2 * j.val + r.val) + (6 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) + k.val) by omega]
  · refine eqAt_congr_fin' (by rw [hmL]; rfl) _ _ (fun i ha hb => ?_) y
    show FlockSoundness.Refine.fOf ((A ++ extras)[k.val]!.x.extract 1 (A ++ extras)[k.val]!.x.size)[i]! =
      Composed.claimX (zkTab x j).table (stAt x j) (czExe PE x c j) r k.val ⟨i, hb⟩
    have hb' : i + 1 < (stAt x j).m - 6 := by
      have : (zkTab x j).table.S.m = (stAt x j).m := rfl
      omega
    rw [extract_one_get _ (by rw [hx6 _ hk]; exact hb')]
    rcases Nat.lt_or_ge k.val 2 with hk2 | hk2
    · rw [FlockComplete.Replay.get_left (show k.val < A.size by rw [hA2]; exact hk2)]
      unfold Composed.claimX
      by_cases h0 : k.val = 0
      · rw [h0, ite_eq_left rfl]
        show _ = Composed.abX (zkTab x j).table (stAt x j) (czExe PE x c j) r (i + 1)
        rw [← Composed.abTail (zkTab x j).table (stAt x j) (czExe PE x c j) r rfl rfl (i + 1) hb']
        have e : (A[0]!).x = (Composed.Lincheck.lcAbClaim (zkTab x j).table.S.m (zkTab x j).table.S.kLog
            (Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r)
            (Composed.tsF (zkTab x j).table (stAt x j) (czExe PE x c j) r)
            ((czExe PE x c j).rest.rPrime r)).x := by
          rw [rhoF_out c j r tr, tsF_lc c j r]
          rfl
        rw [e]
      · have h1 : k.val = 1 := by omega
        rw [h1, ite_eq_right (by decide), ite_eq_left rfl]
        show _ = Composed.zcReq (zkTab x j).table (stAt x j) (czExe PE x c j) r (i + 1)
        rw [← Composed.cTail (zkTab x j).table (stAt x j) (czExe PE x c j) r rfl (i + 1) hb', rF_czExe c hs j r]
        rfl
    · rw [FlockComplete.Replay.get_right (show A.size ≤ k.val by rw [hA2]; omega), hA2]
      have hkr : k.val - 2 < 2 * (stAt x j).regions.size := by
        have := lt_of_lt_of_eq k.isLt hsize
        omega
      obtain ⟨hsz, hpt⟩ := hpt (k.val - 2) hkr
      unfold Composed.claimX
      rw [ite_eq_right (by omega), ite_eq_right (by omega), dite_eq_left hkr, ← hpt]
      refine congrArg FlockSoundness.Refine.fOf ?_
      show _ = (extras[k.val - 2]!.x.extract 1).toList.getD i 0
      rw [List.getD_eq_getElem _ _ (by simp only [Array.length_toList, Array.size_extract]; omega),
        Array.getElem_toList, Array.getElem_extract, getElem!_pos _ (i + 1) (by omega)]
      congr 1
      omega

theorem openB_basis {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim)
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (y : ℕ) (hy : y < 2 ^ ((stAt x j).m - 7)) :
    Composed.openB (zkTab x j).table (stAt x j) (czExe PE x c j) r y =
      FlockSoundness.Refine.fOf (ligBasis x c w τ j r extras)[y]! := by
  have hst := tabsOf_stAt x j
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hpsz : (ptsE c spec).size = 2 := by unfold ptsE; simp [hpts]
  have he' : (stAt x j).extraClaims (ptsE c spec) = .ok extras := by
    rw [hst] at he
    exact he
  refine openB_basisOf hx w τ c hs j r extras ?_ (fun i hi => extras_pt hx c hs j he' i hi) y hy
  rw [FlockComplete.JsonRT.extraClaims_size he', hpsz]

end Basis

/-! ### Level 0's lane rounds

At the coins, the model's lane tables (`Composed.laneF`, `Composed.laneB`) are the executable's arrays (`tabS`, `wS`)
on their entries (`Lg.gv`): both fold by `foldLo` at the lane challenges (`fold_gv`, `laneCh_chS`), from the packed
table (`laneF_tabS`) and from the basis plus the out-of-domain term (`laneB_wS`: `openB_basis`, `eqAt_oodPoint`). So
each lane message is the round polynomial of the two arrays (`laneMsg_roundPoly`, `ligMsgE`), and the pair's sums are
the folded weight's products with the rep's extra lanes (`ligSumsE`). -/

section Lane

/-- **Folds that agree on their entries**: tables folded by `foldLo` at `t i` and arrays folded by `foldK` at `ch i`,
equal on the first `2^M` entries and at each of the first `k ≤ M` challenges, agree on the first `2^(M − i)` entries
after `i ≤ k` folds. -/
private theorem fold_gv {M k : ℕ} (hk : k ≤ M) (u : ℕ → ℕ → FlockLevel3.GF256) (a : ℕ → Array _root_.Flock.F256)
    (t : ℕ → FlockLevel3.GF256) (ch : ℕ → _root_.Flock.F256)
    (hu : ∀ i, u (i + 1) = FlockSoundness.foldLo (t i) (u i)) (ha : ∀ i, a (i + 1) = FlockComplete.foldK (ch i) (a i))
    (ht : ∀ i < k, t i = FlockSoundness.Refine.gOf (ch i)) (hsz : (a 0).size = 2 ^ M)
    (h0 : ∀ y < 2 ^ M, u 0 y = FlockComplete.Replay.Lg.gv (a 0) y) :
    ∀ i ≤ k, (a i).size = 2 ^ (M - i) ∧ ∀ y < 2 ^ (M - i), u i y = FlockComplete.Replay.Lg.gv (a i) y := by
  intro i
  induction i with
  | zero =>
    intro _
    rw [Nat.sub_zero]
    exact ⟨hsz, h0⟩
  | succ i ih =>
    intro hi
    obtain ⟨hs, hu'⟩ := ih (by omega)
    have hp : 2 ^ (M - i) = 2 * 2 ^ (M - (i + 1)) := by
      rw [← pow_succ']
      congr 1
      omega
    refine ⟨by rw [ha, FlockComplete.Replay.Lg.foldK_size, hs, hp]; omega, fun y hy => ?_⟩
    rw [hu, ha, FlockComplete.Replay.Lg.gv_foldK _ _ (by rw [hs, hp]; omega), ← ht i (by omega)]
    unfold FlockSoundness.foldLo
    rw [hu' (2 * y) (by omega), hu' (2 * y + 1) (by omega)]

/-- **The model's lane challenges are the executable's** (`chS`, the replies to rounds `p₀ + 2 + i`). -/
private theorem laneCh_chS {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {s spec}
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) :
    ∀ i < (zkTab x j).sch.k0, Composed.laneCh (zkTab x j).table (stAt x j) (czExe PE x c j) r i =
      FlockSoundness.Refine.gOf (FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r) i) := by
  intro i hi
  unfold Composed.laneCh Composed.seqOf
  split_ifs with h
  · show FlockLevel3.GF256.of (kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + i)) = _
    rw [repSlot_p0 hs j r]
    simp only [kAt, FlockComplete.LigLock.chS, ligCoin, Nat.add_assoc]
  · exact absurd hi h

/-- **The model's lane tables `f` are the executable's** (`tabS`, from the packed table). -/
private theorem laneF_tabS {x : Msg} (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin)
    {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) :
    ∀ i ≤ (zkTab x j).sch.k0,
      (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i).size = 2 ^ ((stAt x j).m - 7 - i) ∧
      ∀ y < 2 ^ ((stAt x j).m - 7 - i),
        Composed.laneF (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j) i y =
          FlockComplete.Replay.Lg.gv (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i) y := by
  have hst := tabsOf_stAt x j
  have hmL : (ligSt x j).m = (stAt x j).m := by
    unfold ligSt
    rw [hst]
    rfl
  obtain ⟨-, -, -, -, -, hk7⟩ := zkLevel0 (zkTab x j).hz
  have hM7 : 7 ≤ (zkTab x j).table.S.m := by
    have e : (zkTab x j).table.S.m = (stAt x j).m := rfl
    omega
  have hsz : (ligTab x w τ j).size = 2 ^ ((stAt x j).m - 7) := by
    unfold ligTab
    rw [FlockComplete.JsonRT.packed_size, hmL]
  refine fold_gv (by omega) _ _ _ _ (fun _ => rfl) (fun _ => rfl) (laneCh_chS c hs j r)
    ((FlockComplete.Replay.Lg.embedF_size _).trans hsz) fun y hy => ?_
  show FlockLevel3.embed (Composed.wordOf (zwL x w τ j) y) =
    FlockComplete.Replay.Lg.gv (FlockComplete.embedF (ligTab x w τ j)) y
  rw [FlockComplete.Replay.Lg.gv_embedF _ (by rw [hsz]; exact hy)]
  refine congrArg FlockLevel3.embed ?_
  unfold ligTab
  rw [hmL]
  exact FlockComplete.wordOf_packed _ _ (zwL_trace w τ j) hy hM7

/-- **The model's lane bases `B` are the executable's weights** (`wS`, from the basis plus the out-of-domain term),
wherever the opening's basis is the deployed one (`hob`). -/
theorem laneB_wSOf {x : Msg} (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim)
    (hob : ∀ y < 2 ^ ((stAt x j).m - 7), Composed.openB (zkTab x j).table (stAt x j) (czExe PE x c j) r y =
      FlockSoundness.Refine.fOf (ligBasis x c w τ j r extras)[y]!) :
    ∀ i ≤ (zkTab x j).sch.k0,
      (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i).size =
        2 ^ ((stAt x j).m - 7 - i) ∧
      ∀ y < 2 ^ ((stAt x j).m - 7 - i), Composed.laneB (zkTab x j).table (stAt x j) (czExe PE x c j) r i y =
        FlockComplete.Replay.Lg.gv
          (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i) y := by
  have hst := tabsOf_stAt x j
  have hmL : (ligSt x j).m = (stAt x j).m := by
    unfold ligSt
    rw [hst]
    rfl
  obtain ⟨-, -, -, -, -, hk7⟩ := zkLevel0 (zkTab x j).hz
  obtain ⟨-, -, hsm, -, -, hdim, -⟩ := FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨hn7, -⟩ := hdim
  have hm : (zkTab x j).table.S.m = (stAt x j).m := rfl
  have hbsz : (ligBasis x c w τ j r extras).size = 2 ^ ((stAt x j).m - 7) := by
    unfold ligBasis
    simp only [FlockComplete.basisVals, Array.size_map, Array.size_range]
    rw [hmL]
  have hzs : (FlockComplete.LigLock.zS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.n).size = (stAt x j).m - 7 := by
    simp only [FlockComplete.LigLock.zS, Array.size_map, Array.size_range]
    omega
  have hesz : (_root_.Flock.eqTable (FlockComplete.LigLock.zS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.n)).size =
      2 ^ ((stAt x j).m - 7) := by
    rw [FlockLevel3.eqTable_size, hzs]
  refine fold_gv (by omega) _ _ _ _ (fun _ => rfl) (fun _ => rfl) (laneCh_chS c hs j r)
    ((FlockComplete.Replay.Lg.addScaled_size _ _ _).trans ((FlockComplete.Replay.Lg.embedF_size _).trans hbsz))
    fun y hy => ?_
  show Composed.laneB0 (zkTab x j).table (stAt x j) (czExe PE x c j) r y =
    FlockComplete.Replay.Lg.gv (FlockComplete.addScaled (FlockComplete.embedF (ligBasis x c w τ j r extras))
      (FlockComplete.embedF (_root_.Flock.eqTable (FlockComplete.LigLock.zS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.n)))
      (ligCoin c (ligP0 x j r + 1) 0)) y
  rw [FlockComplete.Replay.Lg.gv_addScaled _ _ _ (by rw [FlockComplete.Replay.Lg.embedF_size, hbsz]; exact hy),
    FlockComplete.Replay.Lg.gv_embedF _ (by rw [hbsz]; exact hy),
    FlockComplete.Replay.Lg.gv_embedF _ (by rw [hesz]; exact hy), ← hob y hy]
  unfold Composed.laneB0
  refine congrArg₂ (· + ·) rfl ?_
  have hn1 : (Composed.layOf (zkTab x j).table (stAt x j)).nOod0 = 1 := rfl
  rw [Fintype.sum_eq_single ⟨0, by rw [hn1]; omega⟩ fun i hi =>
    absurd (Fin.ext (show i.val = 0 by have := i.isLt; omega)) hi]
  refine (mul_comm _ _).trans (congrArg₂ (· * ·) (congrArg FlockLevel3.embed ?_) (congrArg FlockLevel3.embed ?_))
  · exact eqAt_oodPoint c j r hy
  · show FlockSoundness.Refine.fOf (slotAt c (repSlot x (2 * j.val + r.val) fun _ => 1) 0) = _
    rw [repSlot_p0 hs j r]

/-- **Two limbs read as one element**: where the model's element of limbs `a`, `b` is `R`'s, `R` is the pair of the
limbs' values. -/
private theorem mk_val_eq {a b : FlockLevel3.GF128} {R : _root_.Flock.F256}
    (h : FlockSoundness.execArith.ofLimbs a b = FlockSoundness.Refine.gOf R) :
    (⟨FlockLevel3.GF128.val a, FlockLevel3.GF128.val b⟩ : _root_.Flock.F256) = R := by
  have e : FlockLevel3.GF256.of ⟨FlockLevel3.GF128.val a, FlockLevel3.GF128.val b⟩ = FlockSoundness.Refine.gOf R := by
    rw [FlockLevel3.ofLimbs, ← h, FlockSoundness.Discharge.ZkExec.ofLimbs_eq]
    rfl
  exact congrArg kVal e

private theorem laneB_wS {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim)
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras) :
    ∀ i ≤ (zkTab x j).sch.k0,
      (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i).size =
        2 ^ ((stAt x j).m - 7 - i) ∧
      ∀ y < 2 ^ ((stAt x j).m - 7 - i), Composed.laneB (zkTab x j).table (stAt x j) (czExe PE x c j) r i y =
        FlockComplete.Replay.Lg.gv
          (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i) y :=
  laneB_wSOf w τ c hs j r extras fun y hy => openB_basis hx w τ c hs j r extras he y hy

/-- **Lane round `i`'s message at the coins**: the model's (`Composed.laneMsg`) is the round polynomial of the
executable's table and weight after the first `i` lane challenges. -/
private theorem laneMsg_roundPoly {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) (extras : Array _root_.Flock.Claim)
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras) (i : ℕ)
    (hi : i < (zkTab x j).sch.k0) :
    (Composed.laneMsg (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j) i).1 =
        FlockSoundness.Refine.gOf (FlockComplete.roundPoly
          (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i)
          (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i)).1 ∧
      (Composed.laneMsg (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j) i).2 =
        FlockSoundness.Refine.gOf (FlockComplete.roundPoly
          (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i)
          (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i)).2 := by
  obtain ⟨-, -, -, -, -, hk7⟩ := zkLevel0 (zkTab x j).hz
  have hm : (zkTab x j).table.S.m = (stAt x j).m := rfl
  obtain ⟨htsz, hF⟩ := laneF_tabS w τ c hs j r i hi.le
  obtain ⟨-, hB⟩ := laneB_wS hx w τ c hs j r extras he i hi.le
  obtain ⟨g1, g2⟩ := FlockComplete.Replay.Lg.gOf_roundPoly
    (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i)
    (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n i)
  have hp : 2 ^ ((stAt x j).m - 7 - i) = 2 * 2 ^ ((zkTab x j).table.S.m - 7 - i - 1) := by
    rw [← pow_succ']
    congr 1
    omega
  have hh : (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) i).size / 2 =
      2 ^ ((zkTab x j).table.S.m - 7 - i - 1) := by
    rw [htsz, hp]
    omega
  rw [hh] at g1 g2
  refine ⟨?_, ?_⟩
  · rw [g1]
    unfold Composed.laneMsg
    dsimp only
    refine Finset.sum_congr rfl fun y hy => ?_
    rw [Finset.mem_range] at hy
    rw [hF (2 * y) (by omega), hB (2 * y) (by omega)]
  · rw [g2]
    unfold Composed.laneMsg
    dsimp only
    refine Finset.sum_congr rfl fun y hy => ?_
    rw [Finset.mem_range] at hy
    rw [hF (2 * y) (by omega), hF (2 * y + 1) (by omega), hB (2 * y) (by omega), hB (2 * y + 1) (by omega),
      add_comm (FlockComplete.Replay.Lg.gv _ (2 * y + 1)), add_comm (FlockComplete.Replay.Lg.gv _ (2 * y + 1))]

theorem ligMsgE : LigMsgE := by
  intro x hx w hw τ c n hn s spec hs j r extras he i hi
  have hst := tabsOf_stAt x j
  have hlay : ligLay x j = layL x j := by
    unfold ligLay layL ligSt
    rw [hst]
    rfl
  have hk : ∀ t < 4, (layL x j).lgMsg i t < (layL x j).hab := by
    intro t ht
    simp only [layL, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.lgMsg, _root_.Flock.Zk.PadLayout.hab]
    omega
  have hlo : Composed.layOf (zkTab x j).table (stAt x j) = layL x j := rfl
  have e1 := Composed.honestAt_lgMsg_fst (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r) i hi
  have e2 := Composed.honestAt_lgMsg_snd (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
    (((UE.tape x τ).1 j).2.1.2.1 r) i hi
  rw [hlo] at e1 e2
  obtain ⟨q1, q2⟩ := laneMsg_roundPoly hx w τ c hs j r extras he i hi
  unfold ligSrc ligPad
  rw [hlay]
  unfold FlockComplete.LigLock.mmOf FlockComplete.maskMsg
  rw [srcE_masked_pad_L x c w τ j r (hk 0 (by omega)), srcE_masked_pad_L x c w τ j r (hk 1 (by omega)),
    srcE_masked_pad_L x c w τ j r (hk 2 (by omega)), srcE_masked_pad_L x c w τ j r (hk 3 (by omega))]
  exact Prod.ext (congrArg (· + (⟨_, _⟩ : _root_.Flock.F256)) (mk_val_eq (e1.trans q1)))
    (congrArg (· + (⟨_, _⟩ : _root_.Flock.F256)) (mk_val_eq (e2.trans q2)))

theorem ligSumsE : LigSumsE := by
  intro x hx w hw τ c n hn s spec hs j r extras he
  dsimp only
  have hst := tabsOf_stAt x j
  have hlay : ligLay x j = layL x j := by
    unfold ligLay layL ligSt
    rw [hst]
    rfl
  obtain ⟨htsz, -⟩ := laneF_tabS w τ c hs j r _ le_rfl
  obtain ⟨hwsz, hB⟩ := laneB_wS hx w τ c hs j r extras he _ le_rfl
  obtain ⟨-, hL, -, -, -, -⟩ := tabC_z PE x j
  obtain ⟨-, -, -, -, -, hk7⟩ := zkLevel0 (zkTab x j).hz
  have hLe : (tabC PE x j).z.L = 2 ^ ((stAt x j).m - 7 - (zkTab x j).sch.k0) := by
    rw [hL]
    congr 1
    unfold lv0
    omega
  have hk : ∀ i < 2, ∀ t < 2, (layL x j).lgE i t < (layL x j).hab := by
    intro i hi t ht
    simp only [layL, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.lgE, _root_.Flock.Zk.PadLayout.hab]
    omega
  have key : ∀ (i : Fin 2) (e : ℕ), e = 2 * r.val + i.val →
      (⟨FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
          (((UE.tape x τ).1 j).2.1.2.1 r) ((layL x j).lgE i.val 0)),
        FlockLevel3.GF128.val (Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j)
          (((UE.tape x τ).1 j).2.1.2.1 r) ((layL x j).lgE i.val 1))⟩ : _root_.Flock.F256) =
        FlockComplete.dotK
          (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r extras) (zkTab x j).sch.n
            (zkTab x j).sch.k0)
          (FlockComplete.embedF ((Array.range
            (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0).size).map
              ((UE.tapeZk τ).extra j.val e))) := by
    intro i e he'
    subst he'
    apply mk_val_eq
    refine (Composed.honestAt_lgE (zkTab x j).table (stAt x j) (czExe PE x c j) r (zwL x w τ j) _ i).trans ?_
    rw [FlockComplete.Replay.gOf_dotK, hwsz]
    refine sum_fin_range hLe _ _ fun col hcol => ?_
    have hcol' : col < 2 ^ ((stAt x j).m - 7 - (zkTab x j).sch.k0) := hLe ▸ hcol
    refine (mul_comm _ _).trans (congrArg₂ (· * ·) (hB col hcol') ?_)
    rw [FlockComplete.Replay.Lg.gv_embedF _ (by simp only [Array.size_map, Array.size_range]; rw [htsz]; exact hcol')]
    refine congrArg FlockLevel3.embed (((agrees0_exe PE).ext x τ j r i ⟨col, hcol⟩).trans ?_)
    rw [getElem!_pos _ col (by simp only [Array.size_map, Array.size_range]; rw [htsz]; exact hcol'),
      Array.getElem_map, Array.getElem_range]
    rfl
  have k0' := key 0 (2 * r.val) rfl
  have k1' := key 1 (2 * r.val + 1) rfl
  unfold ligSrc ligPad
  rw [hlay]
  rw [srcE_masked_pad_L x c w τ j r (hk 0 (by omega) 0 (by omega)),
    srcE_masked_pad_L x c w τ j r (hk 0 (by omega) 1 (by omega)),
    srcE_masked_pad_L x c w τ j r (hk 1 (by omega) 0 (by omega)),
    srcE_masked_pad_L x c w τ j r (hk 1 (by omega) 1 (by omega))]
  exact ⟨congrArg (· + (⟨_, _⟩ : _root_.Flock.F256)) k0', congrArg (· + (⟨_, _⟩ : _root_.Flock.F256)) k1'⟩

end Lane

/-- **Hybrid step 1** (`SimLig`): the six value statements, `ligSaltE` at HiSalts' salts (`hiSaltE`). -/
theorem simLig : SimLig :=
  simLig_of_fields ligOodE ligMsgE ligSumsE ligY1E ligYbarE (ligSaltE hiSaltE)

end Masked

end Claims.ZKMal
