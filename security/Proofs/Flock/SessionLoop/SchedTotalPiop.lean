module

public import Proofs.Flock.Claims.ZK.FrameRho
public import Proofs.Flock.Claims.Complete.Reps
public import Proofs.Flock.Soundness.Refine.Arith
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic

/-!
# The opening's PIOP takes the algebraic phase's coins

On the opening's PIOP channel (`Drive.stepAsks coins false`) the masked zerocheck and lincheck take the algebraic
phase's coins in order, and stop only where a coin is missing or holds the wrong count, or where the statement's fold
refuses the zerocheck's point. This file shows they don't stop where the coins hold the rounds' counts and the fold
returns, and that the fold returns at the opening's point wherever it returned at the skeleton's: it refuses on sizes
alone.

* `Fits coins j ns`: the coins from `j` on hold the counts `ns`. `PT coins ns Q p`: from a channel state whose next
  coins fit `ns`, `p` returns, in `Q`, having taken `ns.length` of them (`pt_*`).
* **`repPiop_pt`**: the PIOP takes `zcCounts m ++ lcCounts kLog` coins and leaves `abcClaims`' two claims and the extra
  ones, where the fold returns at every zerocheck point of `m - 6` coins.
* **`fold_ok_of`**: `Stmt.fold` returns at any point of the sizes of one where it returned.
* **`rounds_fold`**: the skeleton's run went through the fold, at a zerocheck point of `m - 6` coins.
-/

@[expose] public section

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (squeezeP sampleP sampleVecP zerocheckFw lincheckFw repPiop)
open _root_.Flock.Firewall.Skeleton (RecM recAsks)

/-! ## Taking the algebraic phase's coins -/

/-- **The coins from `j` fit the counts `ns`**: the `i`-th is there and holds `ns[i]` coins. -/
def Fits (coins : Got) (j : ℕ) (ns : List ℕ) : Prop :=
  ∀ i (h : i < ns.length), ∃ cs, coins[j + i]? = some cs ∧ cs.size = ns[i]

/-- **A PIOP program that takes its coins**: from a channel state whose next coins fit `ns`, `p` returns, with a value
in `Q`, having taken `ns.length` coins. -/
def PT {α : Type} (coins : Got) (ns : List ℕ) (Q : α → Prop) (p : TMW OpenM α) : Prop :=
  ∀ t (got : Got), Fits coins got.size ns →
    ∃ r, ((p.run t).run got) = .done r ∧ Q r.1.1 ∧ r.2.size = got.size + ns.length

variable {coins : Got}

theorem pt_pure {α : Type} {Q : α → Prop} {a : α} (h : Q a) : PT coins [] Q (pure a : TMW OpenM α) :=
  fun t got _ => ⟨((a, t), got), rfl, h, rfl⟩

theorem pt_bind {α β : Type} {ns ms : List ℕ} {Q : α → Prop} {R : β → Prop} {p : TMW OpenM α}
    {f : α → TMW OpenM β} (hp : PT coins ns Q p) (hf : ∀ a, Q a → PT coins ms R (f a)) :
    PT coins (ns ++ ms) R (p >>= f) := by
  intro t got hF
  obtain ⟨⟨⟨a, t₁⟩, g₁⟩, e₁, hq, hs₁⟩ := hp t got fun i hi => by
    have := hF i (by rw [List.length_append]; omega)
    simpa only [List.getElem_append_left hi] using this
  obtain ⟨r, e₂, hr, hs₂⟩ := hf a hq t₁ g₁ fun i hi => by
    have := hF (ns.length + i) (by rw [List.length_append]; omega)
    simpa only [List.getElem_append_right (Nat.le_add_right ns.length i), Nat.add_sub_cancel_left,
      ← Nat.add_assoc, ← hs₁] using this
  refine ⟨r, ?_, hr, by rw [hs₂, hs₁, List.length_append, Nat.add_assoc]⟩
  rw [StateT.run_bind, StateT.run_bind, e₁]
  exact e₂

theorem pt_eq {α : Type} {ns ns' : List ℕ} {Q : α → Prop} {p : TMW OpenM α} (h : PT coins ns Q p) (e : ns = ns') :
    PT coins ns' Q p :=
  e ▸ h

/-- A transcript step: every one is a `modify`. -/
theorem pt_modify (g : Flock.Transcript → Flock.Transcript) :
    PT coins [] (fun _ => True) (liftTM (modify g : Flock.Transcript.TM Unit) : TMW OpenM Unit) :=
  fun t got _ => ⟨(((), g t), got), rfl, trivial, rfl⟩

/-- **A PIOP round** takes its coin where it fits the count its squeeze header holds. -/
theorem round_false (got : Got) (c : ByteArray) {cs : Array F128} (hc : coins[got.size]? = some cs)
    (hn : cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat) :
    ((stepAsks coins false).round c).run got = .done ((fun i => cs[i]!), got.push cs) := by
  simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind, hc]
  refine (step_pure_bind _ _).trans ?_
  dsimp only
  simp only [hn, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  rfl

theorem pt_squeezeP (kind : UInt8) (n : ℕ) (hn : n < 2 ^ 64) :
    PT coins (sq n) (fun v => v.size = n) (squeezeP (stepAsks coins false) kind n) := by
  intro t got hF
  by_cases h0 : n = 0
  · subst h0
    exact ⟨_, rfl, rfl, by simp [sq]⟩
  · obtain ⟨cs, hc, hs⟩ := hF 0 (by simp [sq, h0])
    rw [Nat.add_zero] at hc
    have hs' : cs.size =
        (_root_.Flock.u64OfLE (Flock.Transcript.framed t kind n) ((Flock.Transcript.framed t kind n).size - 8)).toNat := by
      rw [framed_count t kind hn]
      simpa [sq, h0] using hs
    rw [show (squeezeP (stepAsks coins false) kind n).run t = if n = 0 then _ else _ from rfl, ite_eq_right h0,
      StateT.run_bind, round_false got _ hc hs']
    exact ⟨_, rfl, by simp, by simp [sq, h0]⟩

theorem pt_sampleP : PT coins [1] (fun _ => True) (sampleP (stepAsks coins false)) :=
  pt_eq (pt_bind (pt_squeezeP 1 1 (by decide)) fun _ _ => pt_pure trivial) (by simp [sq])

theorem pt_sampleVecP (n : ℕ) (hn : n < 2 ^ 64) :
    PT coins (sq n) (fun v => v.size = n) (sampleVecP (stepAsks coins false) n) :=
  pt_squeezeP 2 n hn

/-- **A loop over `[0:n]`**, step `i` from `J i` yielding into `J (i + 1)` and taking `ms`. -/
theorem pt_forIn {σ : Type} (n : ℕ) (ms : List ℕ) (J : ℕ → σ → Prop) (body : ℕ → σ → TMW OpenM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, J i u → PT coins ms (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u)) (u : σ)
    (hu : J 0 u) : PT coins (List.replicate n ms).flatten (J n) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → J a u →
      PT coins (List.replicate k ms).flatten (J n) (forIn (List.range' a k) u body) from
    key n 0 u (by omega) hu
  intro k
  induction k with
  | zero =>
    intro a u ha hJ
    have e : a = n := by omega
    subst e
    exact pt_pure hJ
  | succ k ih =>
    intro a u ha hJ
    rw [List.range'_succ, List.forIn_cons, List.replicate_succ, List.flatten_cons]
    refine pt_bind (hb a (by omega) u hJ) fun r hr => ?_
    obtain ⟨v, rfl, hv⟩ := hr
    exact ih (a + 1) v (by omega) hv

theorem pt_forIn_bind {σ β : Type} {n : ℕ} {ms ms' : List ℕ} (J : ℕ → σ → Prop) {R : β → Prop}
    {body : ℕ → σ → TMW OpenM (ForInStep σ)} {u : σ} {f : σ → TMW OpenM β}
    (hb : ∀ i < n, ∀ u, J i u → PT coins ms (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u))
    (hf : ∀ r, J n r → PT coins ms' R (f r)) (h0 : J 0 u) :
    PT coins ((List.replicate n ms).flatten ++ ms') R (forIn [:n] u body >>= f) :=
  pt_bind (pt_forIn n ms J body hb u h0) hf

theorem zerocheckFw_pt (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hm : m < 2 ^ 64) :
    PT coins (zcCounts m) (fun r => r.2.rho.size = m - 6)
      (zerocheckFw (stepAsks coins false) m digest cap root lay mk) := by
  unfold zerocheckFw
  apply pt_eq
  · apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_sampleVecP 6 (by decide)); intro _ _
    apply pt_bind (pt_sampleVecP (m - 13) (by omega)); intro _ _
    dsimp only
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind (pt_modify _); intro _ _
    apply pt_bind pt_sampleP; intro _ _
    apply pt_forIn_bind (fun i (s : Array (F128 × F128) × Array F128) => s.2.size = i)
    · intro i _ u hu
      try dsimp only
      apply pt_bind (pt_modify _); intro _ _
      apply pt_bind (pt_modify _); intro _ _
      apply pt_bind pt_sampleP; intro _ _
      exact pt_pure ⟨_, rfl, by simp [hu]⟩
    · intro r hr
      try dsimp only
      apply pt_bind (pt_modify _); intro _ _
      apply pt_bind (pt_modify _); intro _ _
      exact pt_pure hr
    · rfl
  · simp [zcCounts, sq]

theorem lincheckFw_pt (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut)
    (hfold : ∀ α, ∃ v, c.fold α (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok v) :
    PT coins (lcCounts c.kLog) (fun _ => True) (lincheckFw (stepAsks coins false) c lay mk zc) := by
  unfold lincheckFw
  apply pt_eq
  · apply pt_bind (pt_modify _); intro _ _
    apply pt_bind pt_sampleP; intro _ _
    apply pt_bind pt_sampleP; intro _ _
    dsimp only
    split
    · apply pt_bind (pt_pure (Q := fun _ => True) trivial); intro _ _
      try dsimp only
      apply pt_forIn_bind (fun _ _ => True)
      · intro i _ u _
        try dsimp only
        apply pt_bind (pt_modify _); intro _ _
        apply pt_bind (pt_modify _); intro _ _
        apply pt_bind pt_sampleP; intro _ _
        exact pt_pure ⟨_, rfl, trivial⟩
      · intro r _
        try dsimp only
        apply pt_bind (pt_modify _); intro _ _
        apply pt_bind pt_sampleP; intro _ _
        exact pt_pure trivial
      · trivial
    · exfalso
      exact FlockComplete.JsonRT.fold_ne_error hfold ‹_›
  · simp [lcCounts]

/-- **The PIOP takes its coins** and leaves `abcClaims`' two claims and the extra ones, where the statement's fold
returns at every zerocheck point of `m - 6` coins. -/
theorem repPiop_pt (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hm : st.m < 2 ^ 64)
    (hfold : ∀ zc : _root_.Flock.ZcOut, zc.rho.size = st.m - 6 → ∀ α, ∃ v,
      st.fold.fold α (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (st.fold.kLog - 6)) = .ok v) :
    PT coins (zcCounts st.m ++ lcCounts st.fold.kLog) (fun r => r.2.2.size = extras.size + 2)
      (repPiop (stepAsks coins false) st lay cap root extras mk) := by
  unfold repPiop
  apply pt_eq
  · apply pt_bind (zerocheckFw_pt st.m st.digest cap root lay mk hm); intro a ha
    obtain ⟨zp, zc⟩ := a
    dsimp only
    apply pt_bind (lincheckFw_pt st.fold lay mk zc (hfold zc ha)); intro b _
    obtain ⟨lp, lc⟩ := b
    dsimp only
    refine pt_pure ?_
    show (_ ++ extras).size = extras.size + 2
    rw [Array.size_append, Nat.add_comm]
    rfl
  · simp

/-! ## The fold refuses on sizes alone -/

theorem slotBase_ok_of {st : _root_.Flock.Stmt} {α α' : F128} {lw lw' ρ ρ' : Array F128}
    {key : _root_.Flock.Kind × ℕ} {b : Array F128} (h : st.slotBase α lw ρ key = .ok b) :
    ∃ b', st.slotBase α' lw' ρ' key = .ok b' := by
  obtain ⟨k, sl⟩ := key
  cases k with
  | comp =>
    by_cases hc : (sl != st.leaf.comp.spec.kLog) = true
    · simp only [_root_.Flock.Stmt.slotBase] at h
      rw [ite_eq_left hc] at h
      simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    · simp only [_root_.Flock.Stmt.slotBase]
      rw [ite_eq_right hc]
      exact ⟨_, rfl⟩
  | net i =>
    by_cases hc : ((st.c.nets.getD i default).2.unitLog != sl) = true
    · simp only [_root_.Flock.Stmt.slotBase] at h
      rw [ite_eq_left hc] at h
      simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    · simp only [_root_.Flock.Stmt.slotBase]
      rw [ite_eq_right hc]
      exact ⟨_, rfl⟩
  | mask => exact ⟨_, rfl⟩

theorem find_isSome {key : _root_.Flock.Kind × ℕ} {bases bases' : Array ((_root_.Flock.Kind × ℕ) × Array F128)}
    (hb : bases.map Prod.fst = bases'.map Prod.fst) :
    (bases.find? fun kb => decide (kb.1 = key)).isSome = (bases'.find? fun kb => decide (kb.1 = key)).isSome := by
  have e : ∀ bs : Array ((_root_.Flock.Kind × ℕ) × Array F128), (bs.find? fun kb => decide (kb.1 = key)).isSome =
      ((bs.map Prod.fst).find? fun k => decide (k = key)).isSome := fun bs => by
    simp only [Array.find?_map, Option.isSome_map, Function.comp_def]
  rw [e, e, hb]

theorem foldRanges_ok_of {st : _root_.Flock.Stmt} {α α' : F128} {lw lw' ρ ρ' : Array F128} :
    ∀ (rs : List _root_.Flock.Range) (bases bases' : Array ((_root_.Flock.Kind × ℕ) × Array F128))
      (out out' res : Array F128), bases.map Prod.fst = bases'.map Prod.fst →
      st.foldRanges α lw ρ rs bases out = .ok res → ∃ res', st.foldRanges α' lw' ρ' rs bases' out' = .ok res'
  | [], _, _, _, out', _, _, _ => ⟨out', rfl⟩
  | r :: rs, bases, bases', out, out', res, hb, h => by
    have hf := find_isSome (key := (r.kind, r.slotLog)) hb
    simp only [_root_.Flock.Stmt.foldRanges] at h ⊢
    cases e : bases.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
    | some kb =>
      cases e' : bases'.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
      | none => rw [e, e'] at hf; cases hf
      | some kb' =>
        rw [e] at h
        obtain ⟨_, b⟩ := kb
        obtain ⟨_, b'⟩ := kb'
        simp only [bind, Except.bind, pure, Except.pure] at h ⊢
        exact foldRanges_ok_of rs bases bases' _ _ _ hb h
    | none =>
      cases e' : bases'.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
      | some _ => rw [e, e'] at hf; cases hf
      | none =>
        rw [e] at h
        cases hsb : st.slotBase α lw ρ (r.kind, r.slotLog) with
        | error _ => rw [hsb] at h; simp [bind, Except.bind] at h
        | ok b =>
          rw [hsb] at h
          obtain ⟨b', hb'⟩ := slotBase_ok_of (α' := α') (lw' := lw') (ρ' := ρ') hsb
          rw [hb']
          simp only [bind, Except.bind, pure, Except.pure] at h ⊢
          exact foldRanges_ok_of rs _ _ _ _ _ (by simp [hb]) h

/-- **The fold's sizes**: it returned only at a point of the block's size. -/
theorem fold_size {st : _root_.Flock.Stmt} {α : F128} {lw ρ v : Array F128} (h : st.fold α lw ρ = .ok v) :
    ρ.size + 6 = st.c.kLog ∧ lw.size = 64 := by
  by_cases hck : (ρ.size + 6 != st.c.kLog || lw.size != 64) = true
  · simp only [_root_.Flock.Stmt.fold] at h
    rw [ite_eq_left hck] at h
    simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · simpa using hck

/-- **The fold refuses on sizes alone**: it returns at any point of the sizes of one where it returned. -/
theorem fold_ok_of {st : _root_.Flock.Stmt} {α α' : F128} {lw lw' ρ ρ' v : Array F128} (h : st.fold α lw ρ = .ok v)
    (hl : lw'.size = lw.size) (hρ : ρ'.size = ρ.size) : ∃ v', st.fold α' lw' ρ' = .ok v' := by
  by_cases hck : (ρ.size + 6 != st.c.kLog || lw.size != 64) = true
  · simp only [_root_.Flock.Stmt.fold] at h
    rw [ite_eq_left hck] at h
    simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  · simp only [_root_.Flock.Stmt.fold] at h ⊢
    rw [ite_eq_right hck] at h
    rw [hl, hρ, ite_eq_right hck]
    cases hr : st.foldRanges α lw ρ st.c.ranges.toList #[] (Array.replicate (2 ^ st.c.kLog) 0) with
    | error _ => rw [hr] at h; simp [bind, Except.bind] at h
    | ok res =>
      obtain ⟨res', hres'⟩ := foldRanges_ok_of (α' := α') (lw' := lw') (ρ' := ρ') _ #[] #[] _
        (Array.replicate (2 ^ st.c.kLog) 0) res rfl hr
      simp only [bind, Except.bind, pure, Except.pure, hres']
      exact ⟨_, rfl⟩

/-! ## The skeleton's run went through the fold -/

/-- **A loop over `[0:n]`'s counts**, step `i` from `J i` yielding into `J (i + 1)`. -/
theorem tally_forIn_inv {σ : Type} (n : ℕ) (ms : List ℕ) (J : ℕ → σ → Prop) (body : ℕ → σ → TMW RecM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, J i u → Tally ms (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u)) (u : σ)
    (hu : J 0 u) : Tally (List.replicate n ms).flatten (J n) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → J a u →
      Tally (List.replicate k ms).flatten (J n) (forIn (List.range' a k) u body) from
    key n 0 u (by omega) hu
  intro k
  induction k with
  | zero =>
    intro a u ha hJ
    have e : a = n := by omega
    subst e
    exact tally_pure hJ
  | succ k ih =>
    intro a u ha hJ
    rw [List.range'_succ, List.forIn_cons, List.replicate_succ, List.flatten_cons]
    refine tally_bind (hb a (by omega) u hJ) fun r hr => ?_
    obtain ⟨v, rfl, hv⟩ := hr
    exact ih (a + 1) v (by omega) hv

theorem tally_forIn_inv_bind {σ β : Type} {n : ℕ} {ms ms' : List ℕ} (J : ℕ → σ → Prop) {R : β → Prop}
    {body : ℕ → σ → TMW RecM (ForInStep σ)} {u : σ} {f : σ → TMW RecM β}
    (hb : ∀ i < n, ∀ u, J i u → Tally ms (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u))
    (hf : ∀ r, J n r → Tally ms' R (f r)) (h0 : J 0 u) :
    Tally ((List.replicate n ms).flatten ++ ms') R (forIn [:n] u body >>= f) :=
  tally_bind (tally_forIn_inv n ms J body hb u h0) hf

theorem zerocheckFw_rho (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hm : m < 2 ^ 64) :
    Tally (zcCounts m) (fun r => r.2.rho.size = m - 6) (zerocheckFw recAsks m digest cap root lay mk) := by
  unfold zerocheckFw
  apply tally_eq
  · apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_sampleVecP 6 (by decide)); intro _ _
    apply tally_bind (tally_sampleVecP (m - 13) (by omega)); intro _ _
    dsimp only
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind tally_sampleP; intro _ _
    apply tally_forIn_inv_bind (fun i (s : Array (F128 × F128) × Array F128) => s.2.size = i)
    · intro i _ u hu
      try dsimp only
      apply tally_bind (tally_liftTM _); intro _ _
      apply tally_bind (tally_liftTM _); intro _ _
      apply tally_bind tally_sampleP; intro _ _
      exact tally_pure ⟨_, rfl, by simp [hu]⟩
    · intro r hr
      try dsimp only
      apply tally_bind (tally_liftTM _); intro _ _
      apply tally_bind (tally_liftTM _); intro _ _
      exact tally_pure hr
    · rfl
  · simp [zcCounts, sq]

theorem lincheckFw_fold (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut) :
    Tally (lcCounts c.kLog)
      (fun _ => ∃ α v, c.fold α (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (c.kLog - 6)) = .ok v)
      (lincheckFw recAsks c lay mk zc) := by
  unfold lincheckFw
  apply tally_eq
  · apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind tally_sampleP; intro a _
    apply tally_bind tally_sampleP; intro _ _
    dsimp only
    split
    · rename_i v hv
      apply tally_bind (tally_pure (Q := fun _ => True) trivial); intro _ _
      try dsimp only
      apply tally_forIn_bind
      · intro i _ u
        try dsimp only
        apply tally_bind (tally_liftTM _); intro _ _
        apply tally_bind (tally_liftTM _); intro _ _
        apply tally_bind tally_sampleP; intro _ _
        exact tally_pure ⟨_, rfl⟩
      · intro r
        try dsimp only
        apply tally_bind (tally_liftTM _); intro _ _
        apply tally_bind tally_sampleP; intro _ _
        exact tally_pure ⟨a, v, hv⟩
    · exact tally_throw_bind _ _
  · simp [lcCounts]

/-- **The fold returned** at a zerocheck point of `m - 6` coins. -/
def FoldAt (st : _root_.Flock.Setup) : Prop :=
  ∃ (zc : _root_.Flock.ZcOut) (α : F128) (v : Array F128), zc.rho.size = st.m - 6 ∧
    st.fold.fold α (_root_.Flock.skipWeights zc.z) (zc.rho.extract 0 (st.fold.kLog - 6)) = .ok v

theorem repPiop_fold (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hm : st.m < 2 ^ 64) :
    Tally (zcCounts st.m ++ lcCounts st.fold.kLog) (fun _ => FoldAt st)
      (repPiop recAsks st lay cap root extras mk) := by
  unfold repPiop
  apply tally_eq
  · apply tally_bind (zerocheckFw_rho st.m st.digest cap root lay mk hm); intro a ha
    obtain ⟨zp, zc⟩ := a
    dsimp only
    apply tally_bind (lincheckFw_fold st.fold lay mk zc); intro b hb
    obtain ⟨lp, lc⟩ := b
    dsimp only
    obtain ⟨α, v, hv⟩ := hb
    exact tally_pure ⟨zc, α, v, ha, hv⟩
  · simp

/-- **The skeleton's run went through the fold**, at a zerocheck point of `m - 6` coins. -/
theorem rounds_fold {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (hm : st.m < 2 ^ 64) (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) : FoldAt st := by
  have hT := tally_bind (repPiop_fold st (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) cap
    (ByteArray.mk (Array.replicate 64 0)) #[] (fun _ => 0) hm) fun _ hq =>
    tally_pure (Q := fun _ => FoldAt st) (a := ()) hq
  unfold _root_.Flock.Firewall.Skeleton.rounds at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i v hv
    exact (tally_run' (p := _) hT hv).1

/-- `--zk` needs `m ≥ 25`. -/
theorem schedule_m {m : ℕ} {r : _root_.Flock.Schedule × ℕ} (h : _root_.Flock.Zk.schedule m = .ok r) : 25 ≤ m := by
  by_contra hc
  unfold _root_.Flock.Zk.schedule _root_.Flock.Zk.queries0 at h
  rw [ite_eq_left (show m ≤ 24 by omega)] at h
  cases h

end Claims.ZKServer.Count
