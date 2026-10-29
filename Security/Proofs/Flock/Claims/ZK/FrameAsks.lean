import Flock.Firewall.Drive
import Flock.Firewall.SessionPlan
import Proofs.Flock.Claims.ZK.FrameCount

/-!
# The opening's coin asks

After the algebraic phase, the opening runs live (`Drive.openStep`): every squeeze past the coins it was given asks the
driver for its count, as the squeeze's framed header holds it (`Drive.stepAsks`). This file counts those asks, as
`FrameCount` counts the algebraic phase's rounds, so the frame's schedule can be checked against the plan
(`Session.repPlan`).

* `SP Q s ns`: the step program `s` asks for the counts `ns` in order, as far as it asks; `OT I ns Q p` and
  `OTL I ns Q p`: a transcript program over `OpenM`, and one over Ligerito's prover state, asks for `ns` from a channel
  state in `I` (`Past coins`: past the coins it was given), with their `pure`, `bind`, `throw`, transcript steps and
  loops.
* `ot_squeezeP_live`, `ot_sampleP`, `ot_sampleVecP`, `ot_sampleKP`: a live squeeze of `n` coins asks for `n`, none for
  `n = 0`.
* `ringSwitchP_ot`: the ring switch asks for `7` per claim, then the batching's count.
* `ligeritoPZk_ot`: with `1 ≤ k0`, Ligerito asks for `ligeritoCoins` without its zeros.
-/

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (squeezeP sampleP sampleVecP sampleKP)

/-- **A step program's asks**: it asks for coins at the counts `ns` in order, as far as it asks, and where it ends with
`a`, `Q a` of the counts left. -/
def SP {α : Type} (Q : α → List ℕ → Prop) : Step α → List ℕ → Prop
  | .done a, ns => Q a ns
  | .fail _, _ => True
  | .ask _ n k, ns => ∃ ns', ns = n :: ns' ∧ ∀ cs, SP Q (k cs) ns'

theorem sp_bind {α β : Type} {Q : α → List ℕ → Prop} {R : β → List ℕ → Prop} {f : α → Step β}
    (hf : ∀ a rest, Q a rest → SP R (f a) rest) : ∀ (s : Step α) (ns : List ℕ), SP Q s ns → SP R (s.bind f) ns
  | .done a, ns, h => hf a ns h
  | .fail _, _, _ => trivial
  | .ask _ _ k, _, ⟨ns', e, h⟩ => ⟨ns', e, fun cs => sp_bind hf (k cs) ns' (h cs)⟩

abbrev Got := Array (Array F128)

/-- **A transcript program's asks over `OpenM`**, at the channel's states `I`: from a state in `I`, it asks for the
counts `ns` and then ends, if it does, with a value in `Q` at a state in `I`, the counts after `ns` left. -/
def OT {α : Type} (I : Got → Prop) (ns : List ℕ) (Q : α → Prop) (p : TMW OpenM α) : Prop :=
  ∀ t got rest, I got → SP (fun r rest' => Q r.1.1 ∧ I r.2 ∧ rest' = rest) ((p.run t).run got) (ns ++ rest)

variable {I : Got → Prop}

theorem ot_pure {α : Type} {Q : α → Prop} {a : α} (h : Q a) : OT I [] Q (pure a : TMW OpenM α) :=
  fun _ _ _ hI => ⟨h, hI, rfl⟩

theorem ot_bind {α β : Type} {ns ms : List ℕ} {Q : α → Prop} {R : β → Prop} {p : TMW OpenM α}
    {f : α → TMW OpenM β} (hp : OT I ns Q p) (hf : ∀ a, Q a → OT I ms R (f a)) : OT I (ns ++ ms) R (p >>= f) := by
  intro t got rest hI
  rw [StateT.run_bind, StateT.run_bind, List.append_assoc]
  exact sp_bind (fun r rest' h => by
    obtain ⟨hq, hI', rfl⟩ := h
    exact hf r.1.1 hq r.1.2 r.2 rest hI') _ _ (hp t got (ms ++ rest) hI)

theorem ot_eq {α : Type} {ns ns' : List ℕ} {Q : α → Prop} {p : TMW OpenM α} (h : OT I ns Q p) (e : ns = ns') :
    OT I ns' Q p :=
  e ▸ h

theorem ot_throw {α : Type} {ns : List ℕ} {Q : α → Prop} (e : String) : OT I ns Q (throw e : TMW OpenM α) :=
  fun _ _ _ _ => trivial

theorem ot_throw_bind {α β : Type} {ns : List ℕ} {R : β → Prop} (e : String) (f : α → TMW OpenM β) :
    OT I ns R (throw e >>= f) :=
  ot_eq (ot_bind (Q := fun _ => False) (ms := []) (ot_throw e) fun _ h => h.elim) (List.append_nil ns)

theorem ot_liftTM {α : Type} (x : _root_.Flock.Transcript.TM α) : OT I [] (fun _ => True) (liftTM x : TMW OpenM α) := by
  intro t got rest hI
  rcases hx : x t with e | r
  · have : ((liftTM x : TMW OpenM α).run t).run got = .fail e := by
      show Step.bind (monadLift (x t)) _ = _
      rw [hx]; rfl
    rw [this]; trivial
  · have : ((liftTM x : TMW OpenM α).run t).run got = .done (r, got) := by
      show Step.bind (monadLift (x t)) _ = _
      rw [hx]; rfl
    rw [this]
    exact ⟨trivial, hI, rfl⟩

/-- **A live round's ask**: past the coins it was given, the opening's channel asks for the round's count. -/
theorem sp_round_live {Q : (ℕ → F128) × Got → List ℕ → Prop} (coins got : Got) (c : ByteArray) (n : ℕ) (ns : List ℕ)
    (h : coins.size ≤ got.size) (hcn : (_root_.Flock.u64OfLE c (c.size - 8)).toNat = n)
    (hQ : ∀ cs : Array F128, cs.size = n → Q ((fun i => cs[i]!), got.push cs) ns) :
    SP Q (((stepAsks coins true).round c).run got) (n :: ns) := by
  subst hcn
  have hn : coins[got.size]? = none := Array.getElem?_eq_none h
  simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind, hn,
    StateT.lift, SP]
  refine ⟨ns, rfl, fun cs => ?_⟩
  by_cases hc : cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat
  · simp only [hc, bne_self_eq_false, Bool.false_eq_true, if_false]
    exact hQ cs hc
  · simp only [hc, bne_iff_ne, ne_eq, not_false_eq_true, if_true, throw, throwThe, MonadExceptOf.throw]
    exact trivial

/-- The opening's states past the coins it was given, where every round asks. -/
def Past (coins : Got) (got : Got) : Prop := coins.size ≤ got.size

/-- **A live squeeze's ask**: a squeeze of `n` coins asks for `n`, none for `n = 0`. -/
theorem ot_squeezeP_live (coins : Got) (kind : UInt8) (n : ℕ) (hn : n < 2 ^ 64) :
    OT (Past coins) (sq n) (fun _ => True) (squeezeP (stepAsks coins true) kind n) := by
  intro t got rest hI
  by_cases h0 : n = 0
  · subst h0
    exact ⟨trivial, hI, rfl⟩
  · simp only [squeezeP, StateT.run, h0, if_false, bind, StateT.bind, sq, List.singleton_append]
    refine sp_bind (Q := fun r rest' => got.size + 1 = r.2.size ∧ rest' = rest) (fun r rest' h => ?_) _ _
      (sp_round_live coins got _ n rest hI (framed_count t kind hn) fun cs _ => ⟨by simp, rfl⟩)
    obtain ⟨f, got'⟩ := r
    obtain ⟨hs, rfl⟩ := h
    exact ⟨trivial, by unfold Past at hI ⊢; dsimp only at hs ⊢; omega, rfl⟩

theorem ot_sampleP (coins : Got) : OT (Past coins) [1] (fun _ => True) (sampleP (stepAsks coins true)) :=
  ot_eq (ot_bind (ot_squeezeP_live coins 1 1 (by decide)) fun _ _ => ot_pure trivial) (by simp [sq])

theorem ot_sampleVecP (coins : Got) (n : ℕ) (hn : n < 2 ^ 64) :
    OT (Past coins) (sq n) (fun _ => True) (sampleVecP (stepAsks coins true) n) :=
  ot_squeezeP_live coins 2 n hn

theorem ot_sampleKP (coins : Got) : OT (Past coins) [2] (fun _ => True) (sampleKP (stepAsks coins true)) :=
  ot_eq (ot_bind (ot_squeezeP_live coins 2 2 (by decide)) fun _ _ => ot_pure trivial) (by simp [sq])

/-- **A loop over `[0:n]`'s asks**: each step from any value yields, having asked for `ms`. -/
theorem ot_forIn {σ : Type} (n : ℕ) (ms : List ℕ) (body : ℕ → σ → TMW OpenM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, OT I ms (fun r => ∃ v, r = .yield v) (body i u)) (u : σ) :
    OT I (List.replicate n ms).flatten (fun _ => True) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n →
      OT I (List.replicate k ms).flatten (fun _ => True) (forIn (List.range' a k) u body) from
    key n 0 u (by omega)
  intro k
  induction k with
  | zero =>
    intro a u _
    exact ot_pure trivial
  | succ k ih =>
    intro a u ha
    rw [List.range'_succ, List.forIn_cons, List.replicate_succ, List.flatten_cons]
    refine ot_bind (hb a (by omega) u) fun r hr => ?_
    obtain ⟨v, rfl⟩ := hr
    exact ih (a + 1) v (by omega)

theorem ot_forIn_bind {σ β : Type} {n : ℕ} {ms ms' : List ℕ} {R : β → Prop} {body : ℕ → σ → TMW OpenM (ForInStep σ)}
    {u : σ} {f : σ → TMW OpenM β} (hb : ∀ i < n, ∀ u, OT I ms (fun r => ∃ v, r = .yield v) (body i u))
    (hf : ∀ r, OT I ms' R (f r)) : OT I ((List.replicate n ms).flatten ++ ms') R (forIn [:n] u body >>= f) :=
  ot_bind (ot_forIn n ms body hb u) fun r _ => hf r

/-- **The ring switch's asks**: `7` a claim, then the claims' count (none for no claims). -/
theorem ringSwitchP_ot (coins : Got) (sv : ℕ → Array F128) (claims : Array _root_.Flock.Claim)
    (hc : claims.size < 2 ^ 64) :
    OT (Past coins) ((List.replicate claims.size [7]).flatten ++ sq claims.size) (fun _ => True)
      (_root_.Flock.Firewall.Finish.ringSwitchP (stepAsks coins true) sv claims) := by
  unfold _root_.Flock.Firewall.Finish.ringSwitchP
  apply ot_eq
  · apply ot_bind (ot_liftTM _); intro _ _
    try dsimp only
    apply ot_forIn_bind
    · intro i _ u
      try dsimp only
      apply ot_bind (ot_liftTM _); intro _ _
      apply ot_bind (ot_liftTM _); intro _ _
      apply ot_bind (ot_eq (ot_sampleVecP coins 7 (by decide)) (show sq 7 = [7] by simp [sq])); intro _ _
      exact ot_pure ⟨_, rfl⟩
    · intro r
      try dsimp only
      apply ot_bind (ot_sampleVecP coins _ hc); intro _ _
      exact ot_pure trivial
  · simp

/-! ## Ligerito's prover state -/

open _root_.Flock.Firewall.Finish (LP LigState liftT sendMsg powP challengeP oodIntroP queryP)

/-- `OT` over the Ligerito prover's state: from any state. -/
def OTL {α : Type} (I : Got → Prop) (ns : List ℕ) (Q : α → Prop) (p : LP OpenM α) : Prop :=
  ∀ st, OT I ns (fun r => Q r.1) (p.run st)

theorem otl_pure {α : Type} {Q : α → Prop} {a : α} (h : Q a) : OTL I [] Q (pure a : LP OpenM α) := fun st => by
  rw [StateT.run_pure]
  exact ot_pure h

theorem otl_bind {α β : Type} {ns ms : List ℕ} {Q : α → Prop} {R : β → Prop} {p : LP OpenM α}
    {f : α → LP OpenM β} (hp : OTL I ns Q p) (hf : ∀ a, Q a → OTL I ms R (f a)) : OTL I (ns ++ ms) R (p >>= f) :=
  fun st => by
    rw [StateT.run_bind]
    exact ot_bind (hp st) fun q hq => hf q.1 hq q.2

theorem otl_eq {α : Type} {ns ns' : List ℕ} {Q : α → Prop} {p : LP OpenM α} (h : OTL I ns Q p) (e : ns = ns') :
    OTL I ns' Q p :=
  e ▸ h

theorem otl_liftT {α : Type} {ns : List ℕ} {Q : α → Prop} {m : TMW OpenM α} (h : OT I ns Q m) :
    OTL I ns Q (liftT m : LP OpenM α) := fun st => by
  show OT I ns _ (StateT.lift m st)
  exact ot_eq (ot_bind h fun a ha => ot_pure (Q := fun (r : α × LigState) => Q r.1) ha) (List.append_nil ns)

theorem otl_get : OTL I [] (fun _ => True) (get : LP OpenM LigState) := fun st => by
  rw [StateT.run_get]
  exact ot_pure trivial

theorem otl_set (s : LigState) : OTL I [] (fun _ => True) (set s : LP OpenM PUnit) := fun st => by
  rw [StateT.run_set]
  exact ot_pure trivial

theorem otl_modify (F : LigState → LigState) : OTL I [] (fun _ => True) (modify F : LP OpenM PUnit) := fun st => by
  rw [StateT.run_modify]
  exact ot_pure trivial

theorem otl_throw {α : Type} {ns : List ℕ} {Q : α → Prop} (e : String) : OTL I ns Q (throw e : LP OpenM α) :=
  fun _ _ _ _ _ => trivial

theorem otl_ite {α : Type} (c : Prop) [Decidable c] {ns ns' : List ℕ} {Q : α → Prop} {p q : LP OpenM α}
    (h1 : c → OTL I ns Q p) (h2 : ¬ c → OTL I ns' Q q) :
    OTL I (if c then ns else ns') Q (if c then p else q) := by
  by_cases hc : c
  · rw [if_pos hc, if_pos hc]; exact h1 hc
  · rw [if_neg hc, if_neg hc]; exact h2 hc

/-- **A loop over `[0:n]`'s asks**, step `i` from a value in `J i` yielding one in `J (i + 1)` having asked for
`ms i`. -/
theorem otl_forIn {σ : Type} (n : ℕ) (ms : ℕ → List ℕ) (J : ℕ → σ → Prop) (body : ℕ → σ → LP OpenM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, J i u → OTL I (ms i) (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u)) (u : σ)
    (h0 : J 0 u) : OTL I ((List.range n).flatMap ms) (J n) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.range_eq_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → J a u →
      OTL I ((List.range' a k).flatMap ms) (J n) (forIn (List.range' a k) u body) from
    key n 0 u (by omega) h0
  intro k
  induction k with
  | zero =>
    intro a u ha hJ
    have e : a = n := by omega
    exact otl_pure (e ▸ hJ)
  | succ k ih =>
    intro a u ha hJ
    rw [List.range'_succ, List.forIn_cons, List.flatMap_cons]
    refine otl_bind (hb a (by omega) u hJ) fun r hr => ?_
    obtain ⟨v, rfl, hv⟩ := hr
    exact ih (a + 1) v (by omega) hv

theorem otl_forIn_list {τ σ : Type} (l : List τ) (body : τ → σ → LP OpenM (ForInStep σ))
    (hb : ∀ x u, OTL I [] (fun r => ∃ v, r = .yield v) (body x u)) (u : σ) :
    OTL I [] (fun _ => True) (forIn l u body) := by
  induction l generalizing u with
  | nil => exact otl_pure trivial
  | cons x l ih =>
    rw [List.forIn_cons]
    refine otl_eq (otl_bind (hb x u) fun r hr => ?_) rfl
    obtain ⟨v, rfl⟩ := hr
    exact ih v

theorem otl_sendMsg (m : _root_.Flock.F256 × _root_.Flock.F256) : OTL I [] (fun _ => True) (sendMsg m : LP OpenM Unit) := by
  unfold sendMsg
  exact otl_bind (otl_modify _) fun _ _ => otl_bind (otl_liftT (ot_liftTM _)) fun _ _ => otl_liftT (ot_liftTM _)

theorem otl_powP (nonce : ℕ → ℕ → UInt64) (kind : ℕ) : OTL I [] (fun _ => True) (powP nonce kind : LP OpenM Unit) := by
  unfold powP
  exact otl_bind otl_get fun _ _ => otl_bind (otl_set _) fun _ _ => otl_liftT (ot_liftTM _)

theorem otl_challengeP (coins : Got) (split : Bool) :
    OTL (Past coins) [2] (fun _ => True) (challengeP (stepAsks coins true) split) := by
  unfold challengeP
  apply otl_eq
  · apply otl_bind (otl_liftT (ot_sampleKP coins)); intro _ _
    apply otl_bind (otl_modify _); intro _ _
    dsimp only
    apply otl_ite
    · intro _
      apply otl_bind (otl_modify _); intro _ _
      apply otl_bind otl_get; intro _ _
      apply otl_bind (otl_sendMsg _); intro _ _
      exact otl_pure trivial
    · intro _
      apply otl_bind otl_get; intro _ _
      apply otl_bind (otl_sendMsg _); intro _ _
      exact otl_pure trivial
  · simp

theorem otl_oodIntroP (coins : Got) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) (hd : dim < 2 ^ 64) :
    OTL (Past coins) (sq dim ++ [1]) (fun _ => True) (oodIntroP (stepAsks coins true) nonce g dim) := by
  unfold oodIntroP
  refine otl_eq (otl_bind (otl_liftT (ot_sampleVecP coins dim hd)) fun _ _ => otl_bind (otl_modify _) fun _ _ =>
    otl_bind (otl_liftT (ot_liftTM _)) fun _ _ => otl_bind (otl_sendMsg _) fun _ _ =>
    otl_bind (otl_powP nonce 2) fun _ _ => otl_bind (otl_liftT (ot_sampleP coins)) fun _ _ => otl_modify _) (by simp)

/-- A level's query counts: its queries', their `α`'s. -/
def qc (lv : _root_.Flock.Level) : List ℕ := sq lv.q ++ [Nat.log2 (lv.q - 1) + 1]

theorem otl_queryP (coins : Got) (nonce : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme) (lv : _root_.Flock.Level)
    (cm : _root_.Flock.Firewall.Finish.Commit) (hq : lv.q < 2 ^ 63) :
    OTL (Past coins) (qc lv) (fun _ => True) (queryP (stepAsks coins true) nonce ms lv cm) := by
  unfold queryP
  have hl : Nat.log2 (lv.q - 1) + 1 < 2 ^ 64 := by
    have := Nat.log2_le_self (lv.q - 1)
    omega
  refine otl_eq (otl_bind (otl_powP nonce 0) fun _ _ => otl_bind (otl_liftT (ot_sampleVecP coins lv.q (by omega)))
    fun _ _ => otl_bind (otl_powP nonce 1) fun _ _ => otl_bind (otl_liftT (ot_sampleVecP coins _ hl)) fun _ _ =>
    otl_pure trivial) ?_
  simp [qc, sq]

theorem ot_lift_pure {α : Type} (x : α) :
    OT I [] (fun r => r = x) (StateT.lift (pure x : OpenM α) : TMW OpenM α) :=
  fun _ _ _ hI => ⟨rfl, hI, rfl⟩

theorem otl_lift_pure {α : Type} (x : α) :
    OTL I [] (fun r => r = x) (liftT (StateT.lift (pure x : OpenM α)) : LP OpenM α) :=
  otl_liftT (ot_lift_pure x)

theorem otl_at {α : Type} {ns : List ℕ} {Q : α → Prop} {p : LP OpenM α} (h : OTL I ns Q p) (st : LigState) :
    OT I ns (fun r => Q r.1) (p.run st) :=
  h st

theorem ot_run_bind {α β : Type} {ns ms : List ℕ} {Q : α → Prop} {R : β → Prop} {p : LP OpenM α} {st : LigState}
    {f : α × LigState → TMW OpenM β} (hp : OTL I ns Q p) (hf : ∀ a, Q a.1 → OT I ms R (f a)) :
    OT I (ns ++ ms) R (p.run st >>= f) :=
  ot_bind (hp st) hf

theorem otl_ite_same {α : Type} (c : Prop) [Decidable c] {ns : List ℕ} {Q : α → Prop} {p q : LP OpenM α}
    (h1 : c → OTL I ns Q p) (h2 : ¬ c → OTL I ns Q q) : OTL I ns Q (if c then p else q) := by
  by_cases hc : c
  · rw [if_pos hc]; exact h1 hc
  · rw [if_neg hc]; exact h2 hc

theorem otl_throw_bind {α β : Type} {ns : List ℕ} {R : β → Prop} (e : String) (f : α → LP OpenM β) :
    OTL I ns R (throw e >>= f) :=
  otl_eq (otl_bind (Q := fun _ => False) (ms := []) (otl_throw e) fun _ h => h.elim) (List.append_nil ns)

theorem otl_forIn_bind {σ β : Type} {n : ℕ} (ms : ℕ → List ℕ) (J : ℕ → σ → Prop) {ms' : List ℕ} {R : β → Prop}
    {body : ℕ → σ → LP OpenM (ForInStep σ)} {u : σ} {f : σ → LP OpenM β} (h0 : J 0 u)
    (hb : ∀ i < n, ∀ u, J i u → OTL I (ms i) (fun r => ∃ v, r = .yield v ∧ J (i + 1) v) (body i u))
    (hf : ∀ r, J n r → OTL I ms' R (f r)) :
    OTL I ((List.range n).flatMap ms ++ ms') R (forIn [:n] u body >>= f) :=
  otl_bind (otl_forIn n ms J body hb u h0) hf

theorem otl_forIn_array {τ σ : Type} (l : Array τ) (body : τ → σ → LP OpenM (ForInStep σ))
    (hb : ∀ x u, OTL I [] (fun r => ∃ v, r = .yield v) (body x u)) (u : σ) :
    OTL I [] (fun _ => True) (forIn l u body) := by
  rw [← Array.forIn_toList]
  exact otl_forIn_list l.toList body hb u

theorem splitAt_le (s : ℕ) : ∀ i, _root_.Flock.Firewall.Session.splitAt s i ≤ s + 1
  | 0 => by simp [_root_.Flock.Firewall.Session.splitAt]
  | i + 1 => by
    have := splitAt_le s i
    simp only [_root_.Flock.Firewall.Session.splitAt]
    omega

theorem k0_flat (K : ℕ) (hK : 1 ≤ K) : ∀ k ≤ K,
    (List.range k).flatMap (fun j => [2] ++ if (j + 1 == K) = true then [2, 2] else []) =
      List.replicate k 2 ++ if k = K then [2, 2] else [] := by
  intro k hk
  induction k with
  | zero =>
    have h0 : ¬ 0 = K := by omega
    simp only [h0, ↓reduceIte]; rfl
  | succ k ih =>
    have h1 : ¬ k = K := by omega
    rw [List.range_succ, List.flatMap_append, ih (by omega), List.replicate_succ']
    simp only [h1, ↓reduceIte, List.flatMap_cons, List.flatMap_nil, List.append_nil, beq_iff_eq, List.append_assoc]

theorem qcoins_filter (sch : _root_.Flock.Schedule) (i : ℕ) :
    (_root_.Flock.Firewall.Session.queryCoins sch i).toList.filter (· != 0) =
      qc (sch.levels.getD i default) ++ [1] := by
  simp only [_root_.Flock.Firewall.Session.queryCoins, qc, sq]
  split <;> simp_all

theorem filter_cons_ne (a : ℕ) (l : List ℕ) : (a :: l).filter (· != 0) = sq a ++ l.filter (· != 0) := by
  unfold sq; by_cases h : a = 0 <;> simp [h]

theorem sq_one : sq 1 = [1] := rfl

theorem sq_two : sq 2 = [2] := rfl

theorem four_two : (List.range 4).flatMap (fun _ : ℕ => [2]) = List.replicate 4 2 := rfl

theorem two_bne : ((2 : ℕ) != 0) = true := rfl

/-- Ligerito's fold `i` after level 0, as `ligeritoCoins` lists it, its zeros dropped. -/
def ligStep (sch : _root_.Flock.Schedule) : ℕ → List ℕ := fun i =>
  (Array.replicate 4 2 ++ if i + 1 < sch.levels.size - 1 then
      #[_root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) (i + 1), 1,
        _root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) (i + 1), 1] ++
        _root_.Flock.Firewall.Session.queryCoins sch (i + 1)
    else _root_.Flock.Firewall.Session.queryCoins sch (sch.levels.size - 1)).toList.filter (· != 0)

theorem ligeritoPZk_ot (coins : Got) (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : F128)
    (cm0 : _root_.Flock.Firewall.Finish.Commit) (t b : Array F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hk : 1 ≤ sch.k0) (hn : sch.n < 2 ^ 62) (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 63) :
    OT (Past coins) ((_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0)) (fun _ => True)
      (_root_.Flock.Firewall.Finish.ligeritoPZk (stepAsks coins true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) ms sch T cm0 t b lay h rep tPad extra
        lanePad salt nonce) := by
  unfold _root_.Flock.Firewall.Finish.ligeritoPZk
  apply ot_eq
  · apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    apply ot_bind (ot_liftTM _); intro _ _
    dsimp only
    apply ot_run_bind (Q := fun _ => True)
    · 
      apply otl_bind (otl_liftT (ot_sampleVecP coins sch.n (by omega))); intro _ _
      apply otl_bind (otl_modify _); intro _ _
      apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
      apply otl_bind (otl_powP nonce 2); intro _ _
      apply otl_bind (otl_liftT (ot_sampleP coins)); intro _ _
      apply otl_bind (otl_modify _); intro _ _
      apply otl_bind otl_get; intro _ _
      apply otl_bind (otl_sendMsg _); intro _ _
      apply otl_forIn_bind (fun j => [2] ++ if (j + 1 == sch.k0) = true then [2, 2] else []) (fun _ _ => True) trivial
      · intro j _ u _
        apply otl_bind (otl_liftT (ot_sampleKP coins)); intro _ _
        apply otl_bind (otl_modify _); intro _ _
        apply otl_ite
        · intro _
          apply otl_bind otl_get; intro _ _
          apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
          apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
          apply otl_bind (otl_liftT (ot_sampleKP coins)); intro _ _
          apply otl_bind (otl_liftT (ot_sampleKP coins)); intro _ _
          apply otl_ite_same
          · intro _
            exact otl_throw_bind _ _
          · intro _
            apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
            apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
            apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
            apply otl_bind (otl_modify _); intro _ _
            apply otl_bind otl_get; intro _ _
            apply otl_bind (otl_sendMsg _); intro _ _
            exact otl_pure ⟨_, rfl, trivial⟩
        · intro _
          apply otl_bind otl_get; intro _ _
          apply otl_bind (otl_sendMsg _); intro _ _
          exact otl_pure ⟨_, rfl, trivial⟩
      · intro _ _
        apply otl_bind otl_get; intro _ _
        apply otl_bind (otl_lift_pure _); intro _ _
        apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
        apply otl_bind (otl_oodIntroP coins nonce _ _ (by omega)); intro _ _
        apply otl_bind (otl_oodIntroP coins nonce _ _ (by omega)); intro _ _
        apply otl_bind (otl_queryP coins nonce ms _ cm0 (hq 0)); intro _ _
        apply otl_bind (otl_sendMsg _); intro _ _
        apply otl_bind (otl_powP nonce 2); intro _ _
        apply otl_bind (otl_liftT (ot_sampleP coins)); intro _ _
        apply otl_bind (otl_modify _); intro _ _
        apply otl_forIn_bind (ligStep sch)
          (fun i u => i < sch.levels.size - 1 → u.1 = _root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) i)
          (fun _ => rfl)
        · intro i hi u hJ
          obtain ⟨sd, pc, cs, rps, fn⟩ := u
          have hsd : sd = _root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) i := hJ hi
          subst hsd
          apply otl_eq
          · apply otl_forIn_bind (fun _ => [2]) (fun _ _ => True) trivial
            · intro j _ v _
              apply otl_ite_same
              · intro _
                apply otl_bind (otl_liftT (ot_sampleKP coins)); intro _ _
                apply otl_bind (otl_modify _); intro _ _
                apply otl_bind otl_get; intro _ _
                apply otl_bind (otl_modify _); intro _ _
                apply otl_bind otl_get; intro _ _
                apply otl_bind (otl_sendMsg _); intro _ _
                exact otl_pure ⟨_, rfl, trivial⟩
              · intro _
                apply otl_bind (otl_challengeP coins false); intro _ _
                exact otl_pure ⟨_, rfl, trivial⟩
            · intro f _
              have hs := splitAt_le (sch.n - sch.k0 + 1) (i + 1)
              apply otl_ite
              · intro _
                simp only [_root_.Flock.Firewall.Session.splitAt] at hs
                apply otl_bind (otl_lift_pure _); intro _ _
                apply otl_bind (otl_liftT (ot_liftTM _)); intro _ _
                apply otl_bind (otl_oodIntroP coins nonce _
                  (_root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) i - 4 + 1) (by omega)); intro _ _
                apply otl_bind (otl_oodIntroP coins nonce _
                  (_root_.Flock.Firewall.Session.splitAt (sch.n - sch.k0 + 1) i - 4 + 1) (by omega)); intro _ _
                apply otl_bind (otl_queryP coins nonce ms _ _ (hq (i + 1))); intro _ _
                apply otl_bind (otl_sendMsg _); intro _ _
                apply otl_bind (otl_powP nonce 2); intro _ _
                apply otl_bind (otl_liftT (ot_sampleP coins)); intro _ _
                apply otl_bind (otl_modify _); intro _ _
                exact otl_pure ⟨_, rfl, fun _ => rfl⟩
              · intro hl
                apply otl_bind (otl_forIn_array _ _ (fun _ _ => otl_bind (otl_liftT (ot_liftTM _)) fun _ _ =>
                  otl_pure ⟨_, rfl⟩) _); intro _ _
                apply otl_bind (otl_queryP coins nonce ms _ _ (hq _)); intro _ _
                apply otl_bind (otl_powP nonce 2); intro _ _
                apply otl_bind (otl_liftT (ot_sampleP coins)); intro _ _
                exact otl_pure ⟨_, rfl, fun h => by simp at hl; omega⟩
          · simp only [ligStep, _root_.Flock.Firewall.Session.splitAt, Array.toList_append, List.filter_append,
              Array.toList_replicate, List.filter_replicate, two_bne, ↓reduceIte, four_two]
            by_cases hc : i + 1 < sch.levels.size - 1
            · have hne : i + 1 ≠ sch.levels.size - 1 := by omega
              simp only [hc, hne, Bool.not_eq_true', beq_eq_false_iff_ne, ne_eq, not_false_eq_true, ↓reduceIte,
                Array.toList_append, qcoins_filter, filter_cons_ne, sq_one, List.append_assoc, List.nil_append,
                List.append_nil, List.cons_append]
            · have heq : i + 1 = sch.levels.size - 1 := by omega
              simp only [heq, lt_self_iff_false, beq_self_eq_true, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
                qcoins_filter, List.nil_append, List.append_nil]
        · intro _ _
          exact otl_pure trivial
    · intro _ _
      exact ot_pure trivial
  · unfold ligStep
    simp only [_root_.Flock.Firewall.Session.ligeritoCoins, Array.toList_append, List.filter_append,
      Array.toList_flatMap, Array.toList_range, List.filter_flatMap, qcoins_filter, Array.toList_replicate,
      List.filter_replicate, two_bne, ↓reduceIte, filter_cons_ne, List.filter_nil, sq_one, sq_two,
      List.append_assoc, List.nil_append, List.append_nil]
    rw [k0_flat sch.k0 hk sch.k0 le_rfl, if_pos rfl]
    simp only [List.append_assoc, List.cons_append, List.nil_append]

end Claims.ZKServer.Count
