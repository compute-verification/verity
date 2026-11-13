module

public import Proofs.Properties.Verity.Sampled
public import Proofs.Properties.Verity.SampledCoins
public import Proofs.Properties.Spot
public import Proofs.Properties.Challenger
public import Init.Internal.Order.While

/-!
# The sample: a sampled verdict's units are the live audit's sample at the play's coins

The sample's piece of `CouplingTarget`. In a play of the period's game whose parameters name sampled proofs' one-stage
form (`ParamsAt`), where sampled proofs' case (`VStar.Recursive.Sampled.protocol`, whose one-stage form is v2,
`Spot.OneStageZk`) accepts in what the prover sent, and the device's log as the sampled check read it is the
play's (`Vocab.Logged`), the units v2 drew are the live audit's sample at any sample whose bytes start with the play's
coins of the sample's family released after the registration's digest.

* **v2's acceptance** (`case_drew`, `drew_of`): sampled proofs' case in what the prover sent decodes (`V2Case`), in
  the one-stage form the composite's parameters name, since the components' ids are distinct (`params_of_case`); v2
  read its coins `ξ` under the challenger its parameters name and accepted at them (`V2Accepts`), so it drew the
  evidence's units from
  `ξ` (`Drew`): under the sampling design and population its parameters and record name (`LawOf`), `Draw.drawUnits` on `ξ` ends
  with those units (`checkBlobsAt_checkAt`, `checkAt_drew`, `coinsReason_none`, `drawnFrom_ok`).
* **What the read log shows** (`DrawRead`): what the sample asks of the sampled check's read of the device's log
  (`Vocab.coinsRead .sampled`). Wherever the composite accepted what the prover sent (`Verity.SameAccepts`) and v2
  accepts in it at `ξ` under the challenger `dr`, the log it reads holds `ξ`'s bits at the start of the coins of `dr`'s
  family released after `dr`'s digest (`ShowsDraw`, `freshCoins`). Under `Logged` that log is the play's
  (`shows_of_logged`). The sample reads `coinsRead` through `DrawRead` alone, so it holds at any definition of
  `coinsRead` that meets it (`coinsRead` does, with no premise: `Proofs.Properties.Verity.SampledRead`'s
  `drawRead_coinsRead`), and a log shows the samples of any log whose coins of
  the sample's family after its digest start its own (`ShowsDraw.mono`).
* **The bytes** (`append_of_bitsOf_prefix`, `fresh_prefix_coinBytes`): bytes whose bits start another's bits are its
  prefix, and coins at the start of a sample's coins have their bits at the start of the sample's bytes (`coinBytes`).
* **A sample reads a prefix of its bytes** (`drawUnits_append`): a sample that ends on some bytes ends the same, having read
  the same bytes, on them with any bytes after. An accepted sample opens as many coins as it read, which vary with the
  coins, and a session's sample holds a fixed number. So at a session whose sampling design and population are v2's, a sample whose
  bytes start with `ξ` samples v2's units (`draw_of_drew`), at the session widened by a projection's coins too
  (`extend_draw`, `coupled_draw`). At a session whose bytes are `coinBytes` of the sample's coins, that is a sample
  whose first coins are the play's (`coupled_draw_coins`).
-/

@[expose] public section

namespace Verity.Sampled

open Properties.Challenger (bitsOf Draws)

/-! ## A sample reads a prefix of its bytes -/

/-- `y` returns what `x` returns, from every state `x` returns from. -/
def Agrees {β : Type} (x y : StateT ℕ (Except String) β) : Prop :=
  ∀ p r p', x.run p = .ok (r, p') → y.run p = .ok (r, p')

theorem agrees_refl {β : Type} (x : StateT ℕ (Except String) β) : Agrees x x := fun _ _ _ h => h

theorem agrees_bind {β γ : Type} {x y : StateT ℕ (Except String) β} {f g : β → StateT ℕ (Except String) γ}
    (hx : Agrees x y) (hf : ∀ a, Agrees (f a) (g a)) : Agrees (x >>= f) (y >>= g) := by
  intro p r p' h
  rw [StateT.run_bind] at h ⊢
  cases hx' : x.run p with
  | error e => rw [hx'] at h; cases h
  | ok a =>
    rw [hx'] at h
    rw [hx p a.1 a.2 hx']
    exact hf a.1 a.2 r p' h

theorem take_run (src : ByteArray) (need p : ℕ) :
    (_root_.Spot.Draw.take src need).run p =
      if src.size < p + need then Except.error "the randomness source ran out"
      else Except.ok (src.extract p (p + need), p + need) := by
  unfold _root_.Spot.Draw.take
  by_cases h : src.size < p + need <;> simp [h, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get,
    StateT.get, set, StateT.set, pure, StateT.pure, throw, throwThe, MonadExceptOf.throw, StateT.lift, Except.bind,
    Except.pure, _root_.Spot.Draw.RAN_OUT]

/-- `take` on bytes with more after them returns what it returns on the bytes. -/
theorem take_append (src t : ByteArray) (need : ℕ) :
    Agrees (_root_.Spot.Draw.take src need) (_root_.Spot.Draw.take (src ++ t) need) := by
  intro p r p' h
  rw [take_run] at h ⊢
  split at h
  · cases h
  · rename_i hs
    have hs' : ¬ (src ++ t).size < p + need := by rw [ByteArray.size_append]; omega
    rw [ite_eq_right_of_eq_false _ _ (eq_false hs'), ByteArray.extract_append,
      show p + need - src.size = 0 by omega,
      ByteArray.extract_eq_empty_iff.2 (Nat.le_trans (Nat.min_le_left _ _) (Nat.zero_le _)), ByteArray.append_empty]
    exact h

theorem take_step (src : ByteArray) (need p : ℕ) (r : ByteArray) (p' : ℕ)
    (h : (_root_.Spot.Draw.take src need).run p = .ok (r, p')) : p + need ≤ p' ∧ p' ≤ src.size := by
  rw [take_run] at h
  split at h
  · cases h
  · cases h
    omega

/-- A `repeat` loop agrees with another whose steps agree with its own, by a measure each of its own `yield`s
lowers. -/
theorem agrees_loop {β : Type} (f g : Unit → β → StateT ℕ (Except String) (ForInStep β)) (μ : β → ℕ → ℕ)
    (hfg : ∀ b, Agrees (f () b) (g () b))
    (hy : ∀ b s b' s', (f () b).run s = .ok (.yield b', s') → μ b' s' < μ b s) :
    ∀ N b s, μ b s < N → ∀ r s', (Lean.Loop.forIn Lean.Loop.mk b f).run s = .ok (r, s') →
      (Lean.Loop.forIn Lean.Loop.mk b g).run s = .ok (r, s') := by
  intro N
  induction N with
  | zero => intro b s h; omega
  | succ N ih =>
    intro b s hμ r s' h
    rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind] at h ⊢
    cases hb : (f () b).run s with
    | error e => rw [hb] at h; cases h
    | ok x =>
      obtain ⟨step, s₁⟩ := x
      rw [hb] at h
      rw [hfg b s step s₁ hb]
      cases step with
      | done b' => exact h
      | yield b' => exact ih b' s₁ (by have := hy b s b' s₁ hb; omega) r s' h

/-- A getter that, wherever it returns `need` bytes, moves the state up by at least `need`, to at most `S`. -/
def Steps (g : ℕ → StateT ℕ (Except String) ByteArray) (S : ℕ) : Prop :=
  ∀ need p r p', (g need).run p = .ok (r, p') → p + need ≤ p' ∧ p' ≤ S

theorem agrees_onWith (g₁ g₂ : ℕ → StateT ℕ (Except String) ByteArray) (S N : ℕ) (hN : 0 < N)
    (hg : ∀ need, Agrees (g₁ need) (g₂ need)) (hs : Steps g₁ S) (run : ByteArray → Option (Array ℕ)) :
    Agrees (Flock.Draw.onWith g₁ N run) (Flock.Draw.onWith g₂ N run) := by
  unfold Flock.Draw.onWith
  refine agrees_bind (hg N) fun a => agrees_bind ?_ fun _ => agrees_refl _
  intro p r p' h
  refine agrees_loop _ _ (fun _ s => S + 1 - s) ?_ ?_ _ _ _ (Nat.lt_succ_self _) r p' h
  · intro b
    dsimp only
    split
    · exact agrees_refl _
    · exact agrees_bind (hg N) fun _ => agrees_refl _
  · intro b s b' s' hb
    dsimp only at hb
    split at hb
    · rw [StateT.run_pure] at hb; cases hb
    · rw [StateT.run_bind] at hb
      cases hgr : (g₁ N).run s with
      | error e => rw [hgr] at hb; cases hb
      | ok y =>
        obtain ⟨d, s₁⟩ := y
        rw [hgr] at hb
        have := hs N s d s₁ hgr
        change (pure (ForInStep.yield (none, b.2 ++ d)) : StateT ℕ (Except String) _).run s₁ = _ at hb
        rw [StateT.run_pure] at hb
        cases hb
        omega

theorem agrees_forIn_list {α β : Type} (f g : α → β → StateT ℕ (Except String) (ForInStep β))
    (h : ∀ a b, Agrees (f a b) (g a b)) : ∀ (l : List α) (b : β), Agrees (forIn l b f) (forIn l b g)
  | [], b => by rw [List.forIn_nil, List.forIn_nil]; exact agrees_refl _
  | a :: l, b => by
    rw [List.forIn_cons, List.forIn_cons]
    refine agrees_bind (h a b) fun step => ?_
    cases step with
    | done b' => exact agrees_refl _
    | yield b' => exact agrees_forIn_list f g h l b'

theorem agrees_stratifiedWith (g₁ g₂ : ℕ → ℕ → StateT ℕ (Except String) ByteArray) (S : ℕ)
    (hg : ∀ i need, Agrees (g₁ i need) (g₂ i need)) (hs : ∀ i, Steps (g₁ i) S) (strata : Array Flock.Draw.Stratum) :
    Agrees (Flock.Draw.stratifiedWith g₁ strata) (Flock.Draw.stratifiedWith g₂ strata) := by
  unfold Flock.Draw.stratifiedWith
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  refine agrees_bind (agrees_forIn_list _ _ (fun i b => ?_) _ _) fun _ => agrees_refl _
  refine agrees_bind (agrees_onWith _ _ S _ (by unfold Flock.Draw.Stratum.need; omega) (hg i) (hs i) _) fun x => ?_
  split
  · exact agrees_refl _
  · exact agrees_refl _

theorem agrees_drawWith (g₁ g₂ : ℕ → ℕ → StateT ℕ (Except String) ByteArray) (S : ℕ)
    (hg : ∀ i need, Agrees (g₁ i need) (g₂ i need)) (hs : ∀ i, Steps (g₁ i) S) (law : Flock.Draw.Law) (n : ℕ) :
    Agrees (Flock.Draw.drawWith g₁ law n) (Flock.Draw.drawWith g₂ law n) := by
  cases law with
  | subset k =>
    unfold Flock.Draw.drawWith
    exact agrees_bind (agrees_onWith _ _ S _ (by omega) (hg 0) (hs 0) _) fun _ => agrees_refl _
  | bernoulli a b =>
    unfold Flock.Draw.drawWith
    exact agrees_bind (agrees_onWith _ _ S _ (by omega) (hg 0) (hs 0) _) fun _ => agrees_refl _
  | stratified k strata =>
    unfold Flock.Draw.drawWith
    exact agrees_bind (agrees_stratifiedWith _ _ S hg hs _) fun _ => agrees_refl _
  | work k strata =>
    unfold Flock.Draw.drawWith
    exact agrees_bind (agrees_stratifiedWith _ _ S hg hs _) fun _ => agrees_refl _

/-- **A sample reads a prefix of its bytes**: a sample that ends on `src` ends the same, having read the same bytes, on
`src` with any bytes after it. -/
theorem drawUnits_append (law : Flock.Draw.Law) (n : ℕ) (src t : ByteArray) (units : Array ℕ) (read : ℕ)
    (h : _root_.Spot.Draw.drawUnits law n src = .ok (units, read)) :
    _root_.Spot.Draw.drawUnits law n (src ++ t) = .ok (units, read) := by
  unfold _root_.Spot.Draw.drawUnits at h ⊢
  cases hw : (Flock.Draw.drawWith (fun _ need => _root_.Spot.Draw.take src need) law n).run 0 with
  | error e => rw [hw] at h; cases h
  | ok x =>
    rw [hw] at h
    rw [agrees_drawWith _ _ src.size (fun _ need => take_append src t need)
      (fun _ need p r p' => take_step src need p r p') law n 0 x.1 x.2 hw]
    exact h

/-! ## The bytes a sample reads -/

theorem prefix_of_flatMap_byteBits : ∀ {l l' : List UInt8}, l.flatMap byteBits <+: l'.flatMap byteBits → l <+: l'
  | [], _, _ => List.nil_prefix
  | x :: l, [], h => by
    obtain ⟨t, ht⟩ := h
    have := congrArg List.length ht
    simp [byteBits] at this
  | x :: l, y :: l', h => by
    rw [List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨t, ht⟩ := h
    rw [List.append_assoc] at ht
    obtain ⟨h1, h2⟩ := List.append_inj ht (by simp [byteBits])
    obtain ⟨u, hu⟩ := prefix_of_flatMap_byteBits (l := l) (l' := l') ⟨t, h2⟩
    exact ⟨u, by rw [byteBits_inj h1, List.cons_append, hu]⟩

/-- **Bytes whose bits start another's bits are its prefix.** -/
theorem append_of_bitsOf_prefix {ξ ζ : ByteArray} (h : bitsOf ξ <+: bitsOf ζ) : ∃ t, ζ = ξ ++ t := by
  rw [bitsOf_eq, bitsOf_eq] at h
  obtain ⟨u, hu⟩ := prefix_of_flatMap_byteBits h
  refine ⟨⟨u.toArray⟩, ByteArray.ext (Array.toList_inj.1 ?_)⟩
  rw [ByteArray.toList_data_append, ← hu]

/-- **Coins at the start of a sample's coins have their bits at the start of the sample's bytes** (`coinBytes`). -/
theorem fresh_prefix_coinBytes {k B : ℕ} {cs : List (List Bool)} {κ : Fin k → Fin (8 * B) → Bool}
    (h : cs <+: List.ofFn fun j => List.ofFn (κ j)) :
    cs.flatten <+: bitsOf (_root_.Properties.Spot.bytesOf (coinBytes k B κ 0)) := by
  have e : bitsOf (_root_.Properties.Spot.bytesOf (coinBytes k B κ 0)) =
      (List.ofFn fun j => List.ofFn (κ j)).flatten := bitsOf_streamOf_coinBytes κ
  rw [e]
  obtain ⟨t, ht⟩ := h
  rw [← ht, List.flatten_append]
  exact List.prefix_append _ _

/-! ## What v2's acceptance gives -/

/-- The units a sample object names, as v1's check reads them (`OneStage.coinsReason`, `OneStage.check`). -/
def drawnUnits (draw : Lean.Json) : Array ℕ :=
  match _root_.Spot.getD draw "units" with
  | .arr us => us.filterMap _root_.Spot.nat?
  | _ => #[]

/-- **The sampling design and population v2's sample runs under** at the parameters `p` and the commitment `c`
(`OneStage.drawnFrom`'s): the verifier's law as `Draw.derive` expands it, the record's population `n`, and the sampling design `L`
the sample reads its bytes under, the derived law where `Draw.deriveLaw` gives one and else the sampling design's own
(`Draw.flockLaw`). -/
def LawOf (p : _root_.Spot.OneStageZk.Params) (c : _root_.Spot.OneStage.Commitment)
    (L : Flock.Draw.Law) (n : ℕ) : Prop :=
  ∃ law derived fl,
    _root_.Spot.Draw.derive p.base.verifier.program p.base.verifier.query p.base.verifier.law
      p.base.verifier.workTable = .ok law ∧
    _root_.Spot.nat? (_root_.Spot.getD c.record "population") = some n ∧
    _root_.Spot.Draw.deriveLaw p.base.verifier.program p.base.verifier.query p.base.verifier.law
      p.base.verifier.workTable = .ok derived ∧
    _root_.Spot.Draw.flockLaw law n = .ok fl ∧ L = derived.getD fl

/-- The parameters and the commitment fix the sampling design and the population. -/
theorem lawOf_unique {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {L L' : Flock.Draw.Law} {n n' : ℕ} (h : LawOf p c L n) (h' : LawOf p c L' n') : L = L' ∧ n = n' := by
  obtain ⟨law, der, fl, h1, h2, h3, h4, rfl⟩ := h
  obtain ⟨law', der', fl', h1', h2', h3', h4', rfl⟩ := h'
  rw [h1] at h1'; cases h1'
  rw [h2] at h2'; cases h2'
  rw [h3] at h3'; cases h3'
  rw [h4] at h4'; cases h4'
  exact ⟨rfl, rfl⟩

theorem drawnFrom_ok {v : _root_.Spot.Registration.Verifier} {law : Lean.Json} {n : ℕ} {src : ByteArray}
    {units : Array ℕ} {read : ℕ} (h : _root_.Spot.OneStage.drawnFrom v law n src = .ok (units, read)) :
    ∃ derived fl, _root_.Spot.Draw.deriveLaw v.program v.query v.law v.workTable = .ok derived ∧
      _root_.Spot.Draw.flockLaw law n = .ok fl ∧
      _root_.Spot.Draw.drawUnits (derived.getD fl) n src = .ok (units, read) := by
  unfold _root_.Spot.OneStage.drawnFrom at h
  cases hd : _root_.Spot.Draw.deriveLaw v.program v.query v.law v.workTable with
  | error e => rw [hd] at h; cases h
  | ok derived =>
    cases hf : _root_.Spot.Draw.flockLaw law n with
    | error e => rw [hd, hf] at h; cases h
    | ok fl =>
      rw [hd, hf] at h
      change (_root_.Spot.OneStage.coveredDesign (derived.getD fl) >>= fun l =>
        _root_.Spot.Draw.drawUnits l n src) = _ at h
      refine ⟨derived, fl, rfl, rfl, ?_⟩
      revert h
      cases derived.getD fl
      case bernoulli a b =>
        intro h
        change Except.error _ = _ at h
        cases h
      all_goals intro h; exact h

theorem eq_of_bne_ite {a b : Array ℕ} {s : String} {rest : Option String}
    (h : (if (a != b) = true then some s else rest) = none) : a = b := by
  by_contra hab
  simp [hab] at h

theorem coinsReason_none {v : _root_.Spot.Registration.Verifier} {law rec_ draw : Lean.Json}
    {src : ByteArray} (h : _root_.Spot.OneStage.coinsReason v law rec_ draw src = none) :
    ∃ n read, _root_.Spot.nat? (_root_.Spot.getD rec_ "population") = some n ∧
      _root_.Spot.OneStage.drawnFrom v law n src = .ok (drawnUnits draw, read) := by
  unfold _root_.Spot.OneStage.coinsReason at h
  cases hn : _root_.Spot.nat? (_root_.Spot.getD rec_ "population") with
  | none => simp [hn] at h
  | some n =>
    cases hdf : _root_.Spot.OneStage.drawnFrom v law n src with
    | error why => simp [hn, hdf] at h
    | ok ur =>
      obtain ⟨units, read⟩ := ur
      simp only [hn, hdf] at h
      refine ⟨n, read, rfl, ?_⟩
      have hx : units = drawnUnits draw := by
        rw [eq_of_bne_ite h]
        unfold drawnUnits
        generalize _root_.Spot.getD draw "units" = j
        cases j <;> rfl
      rw [hdf, hx]

/-- **v1's check at the coins `ξ` accepts only where `ξ` samples the evidence's units**: under the verifier's law as
derived and the record's population, `OneStage.drawnFrom` on `ξ` gives them. -/
theorem checkAt_drew {p : _root_.Spot.OneStage.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStage.Evidence} {pub : _root_.Spot.OneStage.Public}
    {prof : Core.Integrity} (h : _root_.Spot.OneStage.checkAt p c (some ξ) e = .accept pub prof) :
    ∃ law n read,
      _root_.Spot.Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable =
        .ok law ∧
      _root_.Spot.nat? (_root_.Spot.getD c.record "population") = some n ∧
      _root_.Spot.OneStage.drawnFrom p.verifier law n ξ = .ok (drawnUnits e.draw, read) := by
  unfold _root_.Spot.OneStage.checkAt at h
  cases hw : _root_.Spot.OneStage.coinsWhy p c e ξ with
  | some why =>
    have hb : (some ξ).bind (_root_.Spot.OneStage.coinsWhy p c e) = some why := hw
    rw [hb] at h; simp at h
  | none =>
    have hb : (some ξ).bind (_root_.Spot.OneStage.coinsWhy p c e) = none := hw
    rw [hb] at h
    have hck : _root_.Spot.OneStage.check p c e = .accept pub prof := h
    obtain ⟨law, _, _, hreg, -⟩ := _root_.Properties.Spot.acceptMeans _ _ _ _ _ hck
    have hd := _root_.Properties.Spot.registered_law hreg
    have hr : _root_.Spot.OneStage.coinsReason p.verifier law c.record e.draw ξ = none := by
      have h' := hw
      unfold _root_.Spot.OneStage.coinsWhy at h'
      simp only [hd] at h'
      exact h'
    obtain ⟨n, read, hn, hdf⟩ := coinsReason_none hr
    exact ⟨law, n, read, hd, hn, hdf⟩

/-- **v2's check at the coins `ξ` accepts only where v1's does**, at the evidence's verdicts as the session ran them. -/
theorem checkBlobsAt_checkAt {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : Core.Blobs} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ (out : Array (String × Lean.Json)) (pub' : _root_.Spot.OneStage.Public) (prof' : Core.Integrity),
      _root_.Spot.OneStage.checkAt p.base c (some ξ) { e.base with verdicts := out } = .accept pub' prof' := by
  replace h := (_root_.Spot.OneStageZk.checkBlobsAt_accept h).1
  unfold _root_.Spot.OneStageZk.checkOneAt at h
  try simp only [] at h
  split at h
  · cases h
  · split at h
    all_goals exact ⟨_, _, _, by assumption⟩

/-! ## v2's case at sampled proofs' component -/

theorem find?_nodup {β : Type} {f : β → String} : ∀ {l : List β} {y : β}, (l.map f).Nodup → y ∈ l →
    l.find? (fun x => f x == f y) = some y
  | z :: l, y, hnd, hy => by
    rw [List.map_cons, List.nodup_cons] at hnd
    rcases List.mem_cons.1 hy with rfl | hy'
    · simp
    · have hne : f z ≠ f y := fun e => hnd.1 (e ▸ List.mem_map_of_mem hy')
      rw [List.find?_cons, show (f z == f y) = false by simpa using hne]
      exact find?_nodup hnd.2 hy'

universe u v w

theorem ids_of_zip {α : Type u} {β : Type v} {γ : Type w} {fa : α → String} {fb : β → String} {fc : γ → String}
    {cs : List α} :
    ∀ {ps : List β} {ev : List γ}, ps.map fb = cs.map fa → ev.map fc = cs.map fa →
      ∀ {x}, x ∈ cs.zip (ps.zip ev) → fb x.2.1 = fa x.1 ∧ fc x.2.2 = fa x.1 := by
  induction cs with
  | nil => intro ps ev _ _ x hx; simp at hx
  | cons c cs ih =>
    intro ps ev hp he x hx
    cases ps with
    | nil => simp at hp
    | cons p ps =>
      cases ev with
      | nil => simp at he
      | cons e ev =>
        simp only [List.map_cons, List.cons.injEq] at hp he
        rw [List.zip_cons_cons, List.zip_cons_cons, List.mem_cons] at hx
        rcases hx with rfl | hx
        · exact ⟨hp.1, he.1⟩
        · exact ih hp.2 he.2 hx

/-- Composite `k`'s components on the blobs sent have distinct ids. -/
theorem components_nodup (k : Verity.Registration.Composite) (s : Verity.Sent) :
    ((s.components k).map (·.id)).Nodup := by
  have e : (s.components k).map (·.id) = (Verity.componentsOf k).map (·.id) := by
    unfold Verity.Sent.components
    rw [List.map_map]
    rfl
  rw [e]
  cases k <;> decide

theorem eq_of_ids {β : Type} {f : β → String} {l : List β} {x y : β} (hnd : (l.map f).Nodup) (hx : x ∈ l)
    (hy : y ∈ l) (h : f x = f y) : x = y := by
  have h1 := find?_nodup hnd hx
  have h2 := find?_nodup hnd hy
  rw [h] at h1
  exact Option.some.inj (h1.symm.trans h2)

/-- **Sampled proofs' case in what the prover sent composite `k`**: the three byte strings decode, and one of the
components' cases, at its parameters `a`, its commitment `b` and its evidence `e`, is sampled proofs'
(`VStar.Recursive.Sampled.protocol`) on the blobs sent (`CaseAccepts`' case without its acceptance). -/
def V2Case (k : Verity.Registration.Composite) (s : Verity.Sent) (a b e : ByteArray) : Prop :=
  ∃ ps c ev, Verity.Decoded s ps c ev ∧ ∃ x ∈ Properties.Composition.triples (s.components k) ps ev,
    x.1 = VStar.Recursive.Sampled.protocol.onBlobs (Core.Blobs.ofArray s.blobs) ∧
      a = x.2.1.params.toUTF8 ∧ b = x.2.2.commitment ∧ e = x.2.2.evidence

/-- **v2 accepts at the coins it read** on the blobs `bs`: `a`, `b` and `e` decode, in sampled proofs' one-stage form,
to `p`, `c` and `ev`, v2 reads the coins `ξ` under the challenger its parameters name (`dr`,
`Properties.Challenger.oneStageZkSource`), and its check at them accepts. -/
def V2Accepts (bs : Core.Blobs) (a b e : ByteArray) (p : _root_.Spot.OneStageZk.Params)
    (c : _root_.Spot.OneStage.Commitment) (ev : _root_.Spot.OneStageZk.Evidence) (ξ : ByteArray)
    (dr : Draws) : Prop :=
  (Core.canonicalJson a >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p) ∧
    _root_.Spot.OneStage.commitmentOf b = .ok c ∧ VStar.Recursive.Sampled.evidenceOf e = .ok (.inl ev) ∧
    _root_.Spot.OneStageZk.coinsOf p c ev = .ok ξ ∧ Properties.Challenger.oneStageZkSource p c = some dr ∧
    ∃ pub prof, _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof

/-- **The case's parameters are the composite's one-stage parameters**: in a composition composite `k` accepted, the
parameters sampled proofs' case decodes to are the ones `ParamsAt` reads, since the components' ids are distinct. -/
theorem params_of_case {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e) {P : VStar.Recursive.Sampled.Params}
    (hp : (Core.canonicalJson a >>= VStar.Recursive.Sampled.paramsOfJson) = .ok P)
    {p : _root_.Spot.OneStageZk.Params} (h : ParamsAt s.params p) : P = .inl p := by
  obtain ⟨ps, c₀, evs, ⟨hps, -, hevs⟩, x, hx, hx1, rfl, rfl, rfl⟩ := hcase
  obtain ⟨ps', c₀', evs', ⟨hps', -, hevs'⟩, pub, prof, ⟨⟨⟨hfp, hfe, -, -⟩, -⟩, -⟩, -⟩ := hS
  rw [Except.ok.inj (hps'.symm.trans hps)] at hfp
  rw [Except.ok.inj (hevs'.symm.trans hevs)] at hfe
  obtain ⟨ps'', hps'', y, hfind, hyp⟩ := h
  have hy := List.mem_of_find?_eq_some hfind
  have hyid : y.id = VStar.Recursive.Sampled.PROTOCOL := by simpa using List.find?_some hfind
  rw [Except.ok.inj (hps''.symm.trans hps)] at hy
  have hid : x.1.id = VStar.Recursive.Sampled.PROTOCOL := (congrArg (·.id) hx1).trans rfl
  have hpid : x.2.1.id = x.1.id := (ids_of_zip hfp hfe hx).1
  have hxy : y = x.2.1 :=
    eq_of_ids (by rw [hfp]; exact components_nodup k s) hy (List.of_mem_zip (List.of_mem_zip hx).2).1
      (hyid.trans (hid.symm.trans hpid.symm))
  rw [hxy] at hyp
  exact (Except.ok.inj (hyp.symm.trans hp)).symm

/-- **v2's case accepts only at coins it read under its parameters' challenger**, in a composition composite `k`
accepted whose parameters name sampled proofs' one-stage form (`ParamsAt`). -/
theorem case_drew {k : Verity.Registration.Composite} {s : Verity.Sent} (hS : Verity.SameAccepts k s)
    {p₀ : _root_.Spot.OneStageZk.Params} (hθ : ParamsAt s.params p₀)
    (h : Verity.CaseAccepts k VStar.Recursive.Sampled.protocol s) :
    ∃ a b e p c ev ξ dr, V2Case k s a b e ∧ V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr := by
  obtain ⟨ps, c₀, ev₀, hd, x, hx, hQ, hacc⟩ := h
  have hcase : V2Case k s x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence :=
    ⟨ps, c₀, ev₀, hd, x, hx, hQ, rfl, rfl, rfl⟩
  rw [hQ] at hacc
  obtain ⟨P, c, E, hp, hc, hev, pub, prof, hck⟩ := hacc
  have hP : P = .inl p₀ := params_of_case hS hcase hp hθ
  subst hP
  change VStar.Recursive.Sampled.checkBlobs (.inl p₀) c E (Core.Blobs.ofArray s.blobs) = .accept pub prof at hck
  rcases E with ev | e'
  · change _root_.Spot.OneStageZk.checkBlobs p₀ c ev (Core.Blobs.ofArray s.blobs) = .accept pub prof at hck
    unfold _root_.Spot.OneStageZk.checkBlobs at hck
    cases hξ : _root_.Spot.OneStageZk.coinsOf p₀ c ev with
    | error r => rw [hξ] at hck; simp [Core.atCoins] at hck
    | ok ξ =>
      rw [hξ] at hck
      simp only [Core.atCoins] at hck
      obtain ⟨dr, hdr⟩ := Properties.Challenger.oneStageZk_source hξ
      exact ⟨_, _, _, p₀, c, ev, ξ, dr, hcase, hp, hc, hev, hξ, hdr, pub, prof, hck⟩
  · simp [VStar.Recursive.Sampled.checkBlobs] at hck

/-- **v2 drew `units` from the coins `ξ`** in what the prover sent composite `k`: its case there accepts at `ξ`, read
under the challenger `dr`, and under the sampling design `L` over `n` units its parameters and record name (`LawOf`),
`Draw.drawUnits` on `ξ` ends with the evidence's units. -/
def Drew (k : Verity.Registration.Composite) (s : Verity.Sent) (dr : Draws) (L : Flock.Draw.Law) (n : ℕ)
    (ξ : ByteArray) (units : Array ℕ) : Prop :=
  ∃ a b e p c ev, V2Case k s a b e ∧ V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr ∧ LawOf p c L n ∧
    units = drawnUnits ev.base.draw ∧ ∃ read, _root_.Spot.Draw.drawUnits L n ξ = .ok (units, read)

theorem drew_of {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws} (hcase : V2Case k s a b e)
    (hacc : V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) : ∃ L n units, Drew k s dr L n ξ units := by
  obtain ⟨_, _, _, _, _, pub, prof, hck⟩ := id hacc
  obtain ⟨out, pub', prof', hat⟩ := checkBlobsAt_checkAt hck
  obtain ⟨law, n, read, hd, hn, hdf⟩ := checkAt_drew hat
  obtain ⟨derived, fl, hder, hfl, hdu⟩ := drawnFrom_ok hdf
  exact ⟨derived.getD fl, n, drawnUnits ev.base.draw, a, b, e, p, c, ev, hcase, hacc,
    ⟨law, derived, fl, hd, hn, hder, hfl, rfl⟩, rfl, read, hdu⟩

/-! ## What the sampled check's read of the device's log shows of its sample -/

/-- **The coins of family `label` released after `after`** in the device's log `log`, in order. -/
def freshCoins (label : String) (after : ByteArray) (log : List Verity.Release) : List (List Bool) :=
  log.filterMap fun r => match r.frame with
    | .fresh l => if l = label ∧ r.answers = after then some r.coins else none
    | _ => none

/-- **The log `log` shows the sample at the coins `ξ` read under `dr`**: `ξ`'s bits start the coins of `dr`'s family
released after `dr`'s digest there. -/
def ShowsDraw (log : List Verity.Release) (dr : Draws) (ξ : ByteArray) : Prop :=
  bitsOf ξ <+: (freshCoins dr.label dr.after log).flatten

/-- A log shows every sample a log shows whose coins of the sample's family after its digest start the log's. -/
theorem ShowsDraw.mono {log log' : List Verity.Release} {dr : Draws} {ξ : ByteArray}
    (h : freshCoins dr.label dr.after log <+: freshCoins dr.label dr.after log') (hs : ShowsDraw log dr ξ) :
    ShowsDraw log' dr ξ := by
  obtain ⟨t, ht⟩ := h
  refine List.IsPrefix.trans hs ⟨t.flatten, ?_⟩
  rw [← ht, List.flatten_append]

/-- **What the sample asks of the sampled check's read of the device's log** (`read`, `Vocab.coinsRead .sampled`):
wherever composite `k` accepted what the prover sent as a `same` composition (`Verity.SameAccepts`) and v2's case in it
accepts at the coins `ξ` it read under the challenger `dr`, the read returns a log that shows the sample there
(`ShowsDraw`). -/
def DrawRead (k : Verity.Registration.Composite) (read : Verity.Sent → Option (List Verity.Release)) : Prop :=
  ∀ s a b e p c ev ξ dr, Verity.SameAccepts k s → V2Case k s a b e →
    V2Accepts (Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr → ∃ log, read s = some log ∧ ShowsDraw log dr ξ

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- At a deployment, the sampled check's read is v2's (`Deployed.coinsRead`), so it shows the sample where v2's does. -/
theorem drawRead_deployed {V : Vocab d α} {K : Constructions V} {S : ByteArray → Array ByteArray → Option Live}
    {k : Verity.Registration.Composite} (hD : Deployed V K S) (h : DrawRead k coinsRead) :
    DrawRead k (V.coinsRead .sampled) := by
  rw [show V.coinsRead .sampled = coinsRead from funext hD.coinsRead]
  exact h

/-- **The play's log shows the sample**: the sampled check's read shows it (`DrawRead`), and it is the play's log
(`Logged`). -/
theorem shows_of_logged {V : Vocab d α} {k : Verity.Registration.Composite} (hR : DrawRead k (V.coinsRead .sampled))
    {o : Verity.Exchange} {a b e : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray}
    {dr : Draws} (hS : Verity.SameAccepts k o.sent) (hcase : V2Case k o.sent a b e)
    (hacc : V2Accepts (Core.Blobs.ofArray o.sent.blobs) a b e p c ev ξ dr) (hl : V.Logged .sampled o) :
    ShowsDraw o.log dr ξ := by
  obtain ⟨log, hlog, hs⟩ := hR o.sent a b e p c ev ξ dr hS hcase hacc
  rw [← hl log (Option.mem_def.2 hlog)]
  exact hs

/-- **A sampled verdict that accepts is sampled proofs' case accepting** (`composite_readsV2`). -/
theorem verdict_case {V : Vocab d α} {k : Verity.Registration.Composite} {o : Verity.Exchange}
    (h : V.Verdict k .sampled o) : Verity.CaseAccepts k VStar.Recursive.Sampled.protocol o.sent := by
  obtain ⟨-, -, P, hP, hc⟩ := h
  have hk' : Verity.component k .sampled = some VStar.Recursive.Sampled.protocol := composite_readsV2 k
  rw [hk'] at hP
  cases hP
  exact hc

/-- **The sample in a play**: where the play's parameters name sampled proofs' one-stage form, the sampled verdict
accepts and the sampled check's read of the device's log is the play's, v2 drew `units` from coins `ξ` read under `dr`
whose bits start the play's coins of `dr`'s family released after `dr`'s digest. -/
theorem logged_draw {V : Vocab d α} {k : Verity.Registration.Composite} (hR : DrawRead k (V.coinsRead .sampled))
    {o : Verity.Exchange} {p₀ : _root_.Spot.OneStageZk.Params} (hθ : ParamsAt o.params p₀)
    (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧ ShowsDraw o.log dr ξ := by
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hθ (verdict_case hv)
  obtain ⟨L, n, units, hd⟩ := drew_of hcase hacc
  exact ⟨dr, L, n, ξ, units, hd, shows_of_logged hR hv.2.1 hcase hacc hl⟩

/-! ## The live audit's sample -/

/-- **The session's sample at a sample whose sample ends** with `units` is the units it ended with. -/
theorem draw_eq (A : Live) (ω : A.Ω) {units : Array ℕ} {read : ℕ}
    (h : _root_.Spot.Draw.drawUnits A.design A.n (_root_.Properties.Spot.bytesOf (A.src ω 0)) =
      .ok (units, read)) :
    A.L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  unfold Live.L
  dsimp only
  rw [h]

/-- The session widened by a projection's coins samples as the session does at the sample's first part. -/
theorem extend_draw (A : Live) (S : Widening) (ω : A.Ω × S.T) : (A.extend S).L.draw ω = A.L.draw ω.1 := rfl

/-- **v2's units are the session's sample** at a sample whose bytes start with the coins v2 drew them from, at a session
whose sampling design and population are the ones v2 drew under. -/
theorem draw_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law} {n : ℕ}
    {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) (A : Live) (ω : A.Ω) (hL : A.design = L)
    (hn : A.n = n) (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (A.src ω 0))) :
    A.L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨_, _, _, _, _, _, _, _, _, _, read, hdu⟩ := h
  obtain ⟨t, ht⟩ := append_of_bitsOf_prefix hξ
  apply draw_eq A ω (read := read)
  rw [ht, hL, hn]
  exact drawUnits_append L n ξ t units read hdu

/-- **The sample's piece of `CouplingTarget`**: in a play whose parameters name sampled proofs' one-stage form, where
the sampled verdict accepts and the sampled check's read of the device's log is the play's, v2 drew `units` under the
sampling design `L` over `n` units from coins read under `dr`; at a session with that sampling design and population, widened by a projection's
coins, every sample whose bytes start with the play's coins of `dr`'s family released after `dr`'s digest samples exactly
`units`. -/
theorem coupled_draw {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange} {p₀ : _root_.Spot.OneStageZk.Params}
    (hθ : ParamsAt o.params p₀) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T, A.design = L → A.n = n →
        (freshCoins dr.label dr.after o.log).flatten <+:
          bitsOf (_root_.Properties.Spot.bytesOf (A.src ω.1 0)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨dr, L, n, ξ, units, hd, hs⟩ := logged_draw hR hθ hv hl
  exact ⟨dr, L, n, ξ, units, hd, fun ω hL hn hω =>
    (extend_draw A _ ω).trans (draw_of_drew hd A ω.1 hL hn (List.IsPrefix.trans hs hω))⟩

/-- **The sample at a sample whose first coins are the play's**: at a session whose bytes are `coinBytes` of the sample's
coins `κ` (A3's source), every sample whose first coins are the play's coins of `dr`'s family released after `dr`'s
digest samples exactly `units`. -/
theorem coupled_draw_coins {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V) (A : Live)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hR : DrawRead k (V.coinsRead .sampled)) {o : Verity.Exchange} {p₀ : _root_.Spot.OneStageZk.Params}
    (hθ : ParamsAt o.params p₀) (hv : V.Verdict k .sampled o) (hl : V.Logged .sampled o) {m B : ℕ}
    (κ : A.Ω → Fin m → Fin (8 * B) → Bool)
    (hsrc : ∀ ω, _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m B (κ ω) 0)) :
    ∃ dr L n ξ units, Drew k o.sent dr L n ξ units ∧
      ∀ ω : A.Ω × (K.coins A θ F c σ).T, A.design = L → A.n = n →
        freshCoins dr.label dr.after o.log <+: (List.ofFn fun j => List.ofFn (κ ω.1 j)) →
        (K.live A θ F c σ).L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨dr, L, n, ξ, units, hd, h⟩ := coupled_draw K A θ F c σ hR hθ hv hl
  exact ⟨dr, L, n, ξ, units, hd, fun ω hL hn hp => h ω hL hn (by rw [hsrc]; exact fresh_prefix_coinBytes hp)⟩

end Verity.Sampled
