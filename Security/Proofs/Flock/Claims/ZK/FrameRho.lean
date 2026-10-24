import Proofs.Flock.Claims.ZK.FrameRankFault

/-!
# The frames refuse dependent `ρ₀, ρ₁` (`FrameUnrefused`'s `ρ` fault)

The firewall's own masked Ligerito (`Finish.ligeritoPZk`), which the rep's schedule runs live (`Drive.openStep`, through
`Loop.schedSoFar`), stops at `ρ₀, ρ₁` dependent over F128, right after `ρ₁`'s squeeze. So a schedule that runs to its
end on the stream's coins was answered independent `ρ₀, ρ₁`, at Ligerito's rounds `k₀ + 2` and `k₀ + 3`, past the ring
switch's:

* `SPW Q s hs`: the step program `s`, its answers so far `hs`, ends, if it does, with a value and answers in `Q`
  (`spw_bind`, `spw_stable`: a property of the answers that more answers keep holds wherever it ends; `spw_of_sp`: a
  program that asks for `ns` ends `ns` answers on; `spw_feed`: fed coins to its end, it ends on them).
* `OW A B p`, `OWL A B p`: a transcript program over `OpenM`, and one over Ligerito's prover state, from a channel state
  and answers in `A` ends in `B`; `ow_of_ot`, `owl_of_otl`: an `OT` program moves the answers by its asks.
* `ow_sampleKP`: a live `F256` squeeze's value is its answer's first two words.
* **`ligeritoPZk_rho`**, `repOpening_rho`, **`openStep_rho`**: where the opening ends, its answers at `ρ₀`'s and
  `ρ₁`'s asks are independent (`Rho`).
* `rhoOk_lt`, `sched_rho`, `openList_rho`, `repPlan_rho`: every `--zk` schedule asks `z`'s coins (`1 ≤ n`), and its
  plan two coins at `ρ₀`'s and `ρ₁`'s rounds, by the kernel's evaluation.
* **`Count.schedSoFar_rho`**: a rep's schedule so far, complete, ran its opening to its end, past independent answers.
* **`LeanRelease.rho_unrefused`**: the fault. At the rep's inner round 2 (`betaSlot`) its schedule so far is complete,
  so its coins at `rhoSlot` and the next, the opened ones (`wordsAt_slot`), are independent.
-/

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Drive (Step OpenM stepAsks)
open _root_.Flock.Firewall.Finish (squeezeP sampleP sampleVecP sampleKP LP LigState liftT)

/-- A step program's answers so far. -/
abbrev Hist := List (Array F128)

/-- **Where a step program ends**: from the answers `hs`, wherever it ends, with a value and answers in `Q`. -/
def SPW {α : Type} (Q : α → Hist → Prop) : Step α → Hist → Prop
  | .done a, hs => Q a hs
  | .fail _, _ => True
  | .ask _ _ k, hs => ∀ cs, SPW Q (k cs) (hs ++ [cs])

theorem spw_bind {α β : Type} {Q : α → Hist → Prop} {R : β → Hist → Prop} {f : α → Step β}
    (hf : ∀ a hs, Q a hs → SPW R (f a) hs) : ∀ (s : Step α) (hs : Hist), SPW Q s hs → SPW R (s.bind f) hs
  | .done a, hs, h => hf a hs h
  | .fail _, _, _ => trivial
  | .ask _ _ k, hs, h => fun cs => spw_bind hf (k cs) (hs ++ [cs]) (h cs)

theorem spw_mono {α : Type} {Q Q' : α → Hist → Prop} (h : ∀ a hs, Q a hs → Q' a hs) :
    ∀ (s : Step α) (hs : Hist), SPW Q s hs → SPW Q' s hs
  | .done a, hs, hq => h a hs hq
  | .fail _, _, _ => trivial
  | .ask _ _ k, hs, hq => fun cs => spw_mono h (k cs) (hs ++ [cs]) (hq cs)

/-- **A property of the answers that more answers keep** holds wherever a program ends. -/
theorem spw_stable {α : Type} (R : Hist → Prop) (hR : ∀ hs cs, R hs → R (hs ++ [cs])) :
    ∀ (s : Step α) (hs : Hist), R hs → SPW (fun _ hs' => R hs') s hs
  | .done _, _, h => h
  | .fail _, _, _ => trivial
  | .ask _ _ k, hs, h => fun cs => spw_stable R hR (k cs) (hs ++ [cs]) (hR hs cs h)

/-- **A program that asks for `ns` ends `ns` answers on**, the counts after them left. -/
theorem spw_of_sp {α : Type} {Q : α → List ℕ → Prop} : ∀ (s : Step α) (ns : List ℕ) (hs : Hist), SP Q s ns →
    SPW (fun a hs' => ∃ rest, Q a rest ∧ hs'.length + rest.length = hs.length + ns.length) s hs
  | .done a, ns, hs, h => ⟨ns, h, rfl⟩
  | .fail _, _, _, _ => trivial
  | .ask _ n k, ns, hs, ⟨ns', e, h⟩ => fun cs => by
    refine spw_mono ?_ _ _ (spw_of_sp (k cs) ns' (hs ++ [cs]) (h cs))
    rintro a hs' ⟨rest, hq, hl⟩
    refine ⟨rest, hq, ?_⟩
    subst e
    simp only [List.length_append, List.length_cons, List.length_nil] at hl ⊢
    omega

/-- A program that asks for nothing ends on the answers it started with. -/
theorem spw_of_na {α : Type} {Q : α → Prop} : ∀ (s : Step α) (hs : Hist), NA Q s →
    SPW (fun a hs' => Q a ∧ hs' = hs) s hs
  | .done _, _, h => ⟨h, rfl⟩
  | .fail _, _, _ => trivial
  | .ask .., _, h => h.elim

/-- **Fed coins to its end, a program ends on them**: on the answers it started with and the coins its rounds took. -/
theorem spw_feed {α : Type} {Q : α → Hist → Prop} : ∀ (s : Step α) (L : List (Array F128))
    (rs : Array (ByteArray × ℕ)) (hs : Hist) {a : α} {rs' : Array (ByteArray × ℕ)},
    SPW Q s hs → s.feed L rs = (.done a, rs') → rs.size ≤ rs'.size ∧ Q a (hs ++ L.take (rs'.size - rs.size))
  | .done a', L, rs, hs, a, rs', h, he => by
    have e : (Step.done a').feed L rs = (.done a', rs) := by cases L <;> rfl
    rw [e] at he
    cases he
    refine ⟨le_rfl, ?_⟩
    rw [Nat.sub_self, List.take_zero, List.append_nil]
    exact h
  | .fail e', L, rs, hs, a, rs', h, he => by
    have e : (Step.fail e' : Step α).feed L rs = (.fail e', rs) := by cases L <;> rfl
    rw [e] at he
    cases he
  | .ask c n k, [], rs, hs, a, rs', h, he => by
    have e : (Step.ask c n k).feed [] rs = (.ask c n k, rs) := rfl
    rw [e] at he
    cases he
  | .ask c n k, x :: L, rs, hs, a, rs', h, he => by
    have ih := spw_feed (k x) L (rs.push (c, n)) (hs ++ [x]) (h x) he
    rw [Array.size_push] at ih
    refine ⟨by omega, ?_⟩
    rw [show rs'.size - rs.size = rs'.size - (rs.size + 1) + 1 by omega, List.take_succ_cons]
    simpa only [List.append_assoc, List.singleton_append] using ih.2

/-! ## Transcript programs over `OpenM` -/

/-- **Where a transcript program over `OpenM` ends**: from a channel state and answers in `A`, in `B`. -/
def OW {α : Type} (A : Got → Hist → Prop) (B : α → Got → Hist → Prop) (p : TMW OpenM α) : Prop :=
  ∀ t got hs, A got hs → SPW (fun r hs' => B r.1.1 r.2 hs') ((p.run t).run got) hs

theorem ow_bind {α β : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} {C : β → Got → Hist → Prop}
    {p : TMW OpenM α} {f : α → TMW OpenM β} (hp : OW A B p) (hf : ∀ a, OW (B a) C (f a)) : OW A C (p >>= f) := by
  intro t got hs hA
  rw [StateT.run_bind, StateT.run_bind]
  exact spw_bind (fun r hs' h => hf r.1.1 r.1.2 r.2 hs' h) _ _ (hp t got hs hA)

theorem ow_mono {α : Type} {A A' : Got → Hist → Prop} {B B' : α → Got → Hist → Prop} {p : TMW OpenM α}
    (h : OW A B p) (hA : ∀ g hs, A' g hs → A g hs) (hB : ∀ a g hs, B a g hs → B' a g hs) : OW A' B' p :=
  fun t got hs h' => spw_mono (fun _ _ hb => hB _ _ _ hb) _ _ (h t got hs (hA _ _ h'))

theorem ow_pure {α : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} (a : α)
    (h : ∀ g hs, A g hs → B a g hs) : OW A B (pure a : TMW OpenM α) :=
  fun _ got hs hA => h got hs hA

theorem ow_throw {α : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} (e : String) :
    OW A B (throw e : TMW OpenM α) :=
  fun _ _ _ _ => trivial

/-- A property of the answers that more answers keep holds wherever any program ends. -/
theorem ow_any {α : Type} (R : Hist → Prop) (hR : ∀ hs cs, R hs → R (hs ++ [cs])) (p : TMW OpenM α) :
    OW (fun _ hs => R hs) (fun _ _ hs => R hs) p :=
  fun _ _ hs h => spw_stable R hR _ hs h

/-- **An `OT` program moves the answers by its asks.** -/
theorem ow_of_ot {α : Type} {I : Got → Prop} {ns : List ℕ} {Q : α → Prop} {p : TMW OpenM α} (h : OT I ns Q p)
    (m : ℕ) : OW (fun g hs => I g ∧ hs.length = m) (fun a g hs => Q a ∧ I g ∧ hs.length = m + ns.length) p := by
  intro t got hs ⟨hI, hl⟩
  refine spw_mono ?_ _ _ (spw_of_sp _ _ hs (h t got [] hI))
  rintro r hs' ⟨rest, ⟨hq, hI', rfl⟩, hlen⟩
  refine ⟨hq, hI', ?_⟩
  simp only [List.append_nil, List.length_nil, Nat.add_zero] at hlen
  omega

/-- `ow_of_ot`, then `f`. -/
theorem ow_step {α β : Type} {I : Got → Prop} {ns : List ℕ} {Q : α → Prop} {C : β → Got → Hist → Prop}
    {p : TMW OpenM α} {f : α → TMW OpenM β} (h : OT I ns Q p) {m m' : ℕ} (hm : m + ns.length = m')
    (hf : ∀ a, Q a → OW (fun g hs => I g ∧ hs.length = m') C (f a)) :
    OW (fun g hs => I g ∧ hs.length = m) C (p >>= f) :=
  ow_bind (ow_of_ot h m) fun a t got hs ⟨hq, hI, hl⟩ => hf a hq t got hs ⟨hI, hm ▸ hl⟩

/-- **A live round's answer**: past the coins it was given, the opening's channel takes the round's answer. -/
theorem spw_round_live {Q : (ℕ → F128) × Got → Hist → Prop} (coins got : Got) (c : ByteArray) (hs : Hist)
    (h : coins.size ≤ got.size) (hQ : ∀ cs : Array F128, Q ((fun i => cs[i]!), got.push cs) (hs ++ [cs])) :
    SPW Q (((stepAsks coins true).round c).run got) hs := by
  have hn : coins[got.size]? = none := Array.getElem?_eq_none h
  simp only [stepAsks, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Step.bind, hn,
    StateT.lift, SPW]
  intro cs
  by_cases hc : cs.size = (_root_.Flock.u64OfLE c (c.size - 8)).toNat
  · simp only [hc, bne_self_eq_false, Bool.false_eq_true, ite_false]
    exact hQ cs
  · simp only [hc, bne_iff_ne, ne_eq, not_false_eq_true, ite_true, throw, throwThe, MonadExceptOf.throw]
    exact trivial

/-- **A live squeeze's value is its answer's words.** -/
theorem ow_squeezeP_live (coins : Got) (kind : UInt8) {n : ℕ} (hn0 : n ≠ 0) (h0 : Hist) :
    OW (fun g hs => Past coins g ∧ hs = h0)
      (fun v g hs => Past coins g ∧ ∃ cs, hs = h0 ++ [cs] ∧ v = (Array.range n).map fun i => cs[i]!)
      (squeezeP (stepAsks coins true) kind n) := by
  rintro t got hs ⟨hI, rfl⟩
  simp only [squeezeP, StateT.run, hn0, ite_false, bind, StateT.bind]
  refine spw_bind (Q := fun r hs' => ∃ cs, r = ((fun i => cs[i]!), got.push cs) ∧ hs' = hs ++ [cs])
    (fun r hs' h => ?_) _ _ (spw_round_live coins got _ hs hI fun cs => ⟨cs, rfl, rfl⟩)
  obtain ⟨cs, rfl, rfl⟩ := h
  refine ⟨?_, cs, rfl, rfl⟩
  unfold Past at hI ⊢
  rw [Array.size_push]
  omega

/-- **A live `F256` squeeze's value is its answer's first two words.** -/
theorem ow_sampleKP (coins : Got) (h0 : Hist) :
    OW (fun g hs => Past coins g ∧ hs = h0)
      (fun v g hs => Past coins g ∧ ∃ cs, hs = h0 ++ [cs] ∧ v = (⟨cs[0]!, cs[1]!⟩ : _root_.Flock.F256))
      (sampleKP (stepAsks coins true)) := by
  unfold sampleKP
  refine ow_bind (ow_squeezeP_live coins 2 (by decide) h0) fun v => ?_
  refine ow_pure _ ?_
  rintro g hs ⟨hI, cs, rfl, rfl⟩
  exact ⟨hI, cs, rfl, by simp⟩

/-! ## Over Ligerito's prover state -/

/-- `OW` over the Ligerito prover's state: from any state. -/
def OWL {α : Type} (A : Got → Hist → Prop) (B : α → Got → Hist → Prop) (p : LP OpenM α) : Prop :=
  ∀ st, OW A (fun r => B r.1) (p.run st)

theorem owl_bind {α β : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} {C : β → Got → Hist → Prop}
    {p : LP OpenM α} {f : α → LP OpenM β} (hp : OWL A B p) (hf : ∀ a, OWL (B a) C (f a)) : OWL A C (p >>= f) :=
  fun st => by
    rw [StateT.run_bind]
    exact ow_bind (hp st) fun q => hf q.1 q.2

theorem owl_mono {α : Type} {A A' : Got → Hist → Prop} {B : α → Got → Hist → Prop} {p : LP OpenM α}
    (h : OWL A B p) (hA : ∀ g hs, A' g hs → A g hs) : OWL A' B p :=
  fun st => ow_mono (h st) hA fun _ _ _ hb => hb

theorem owl_pure {α : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} (a : α)
    (h : ∀ g hs, A g hs → B a g hs) : OWL A B (pure a : LP OpenM α) := fun st => by
  rw [StateT.run_pure]
  exact ow_pure (B := fun r => B r.1) (a, st) h

theorem owl_throw {α : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} (e : String) :
    OWL A B (throw e : LP OpenM α) :=
  fun _ _ _ _ _ => trivial

theorem owl_throw_bind {α β : Type} {A : Got → Hist → Prop} {C : β → Got → Hist → Prop} (e : String)
    (f : α → LP OpenM β) : OWL A C (throw e >>= f) :=
  owl_bind (B := fun _ _ _ => False) (owl_throw e) fun _ _ _ _ _ h => h.elim

theorem owl_ite_same {α : Type} (c : Prop) [Decidable c] {A : Got → Hist → Prop} {B : α → Got → Hist → Prop}
    {p q : LP OpenM α} (h1 : c → OWL A B p) (h2 : ¬ c → OWL A B q) : OWL A B (if c then p else q) := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact h1 hc
  · rw [ite_eq_right hc]; exact h2 hc

theorem owl_liftT {α : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} {m : TMW OpenM α} (h : OW A B m) :
    OWL A B (liftT m : LP OpenM α) := fun st => by
  show OW A _ (StateT.lift m st)
  exact ow_bind h fun a => ow_pure (B := fun (r : α × LigState) => B r.1) (a, st) fun _ _ hb => hb

theorem owl_any {α : Type} (R : Hist → Prop) (hR : ∀ hs cs, R hs → R (hs ++ [cs])) (p : LP OpenM α) :
    OWL (fun _ hs => R hs) (fun _ _ hs => R hs) p :=
  fun st => ow_any R hR (p.run st)

/-- **An `OTL` program moves the answers by its asks.** -/
theorem owl_of_otl {α : Type} {I : Got → Prop} {ns : List ℕ} {Q : α → Prop} {p : LP OpenM α} (h : OTL I ns Q p)
    (m : ℕ) : OWL (fun g hs => I g ∧ hs.length = m) (fun a g hs => Q a ∧ I g ∧ hs.length = m + ns.length) p :=
  fun st => ow_of_ot (h st) m

/-- `owl_of_otl`, then `f`. -/
theorem owl_step {α β : Type} {I : Got → Prop} {ns : List ℕ} {Q : α → Prop} {C : β → Got → Hist → Prop}
    {p : LP OpenM α} {f : α → LP OpenM β} (h : OTL I ns Q p) {m m' : ℕ} (hm : m + ns.length = m')
    (hf : ∀ a, Q a → OWL (fun g hs => I g ∧ hs.length = m') C (f a)) :
    OWL (fun g hs => I g ∧ hs.length = m) C (p >>= f) :=
  owl_bind (owl_of_otl h m) fun a st t got hs ⟨hq, hI, hl⟩ => hf a hq st t got hs ⟨hI, hm ▸ hl⟩

/-- **A loop over `[0:n]`**, step `i` from `J i` yielding into `J (i + 1)`. -/
theorem owl_forIn {σ : Type} (n : ℕ) (J : ℕ → σ → Got → Hist → Prop) (body : ℕ → σ → LP OpenM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, OWL (J i u) (fun r g hs => ∃ v, r = .yield v ∧ J (i + 1) v g hs) (body i u)) (u : σ) :
    OWL (J 0 u) (fun r => J n r) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n → OWL (J a u) (fun r => J n r) (forIn (List.range' a k) u body) from
    key n 0 u (by omega)
  intro k
  induction k with
  | zero =>
    intro a u ha
    have e : a = n := by omega
    subst e
    exact owl_pure u fun _ _ h => h
  | succ k ih =>
    intro a u ha
    rw [List.range'_succ, List.forIn_cons]
    refine owl_bind (hb a (by omega) u) fun r => ?_
    intro st t got hs ⟨v, hv, hJ⟩
    subst hv
    exact ih (a + 1) v (by omega) st t got hs hJ

/-! ## The opening -/

/-- **`ρ₀`'s and `ρ₁`'s answers independent**: the answers reach them, `ρ₀`'s at `P` and `ρ₁`'s next, and they are
independent over F128. -/
def Rho (P : ℕ) (hs : Hist) : Prop :=
  P + 2 ≤ hs.length ∧ _root_.Flock.Zk.rhoIndependent
    (⟨(hs.getD P #[])[0]!, (hs.getD P #[])[1]!⟩, ⟨(hs.getD (P + 1) #[])[0]!, (hs.getD (P + 1) #[])[1]!⟩) = true

theorem rho_stable (P : ℕ) (hs : Hist) (cs : Array F128) (h : Rho P hs) : Rho P (hs ++ [cs]) := by
  obtain ⟨hl, hr⟩ := h
  have e : ∀ i, i < hs.length → (hs ++ [cs]).getD i #[] = hs.getD i #[] := fun i hi => by
    simp only [List.getD_eq_getElem?_getD, List.getElem?_append_left hi]
  refine ⟨by rw [List.length_append]; omega, ?_⟩
  rw [e P (by omega), e (P + 1) (by omega)]
  exact hr


theorem owl_forIn_bind {σ β : Type} {n : ℕ} (J : ℕ → σ → Got → Hist → Prop) {A : Got → Hist → Prop}
    {C : β → Got → Hist → Prop} {body : ℕ → σ → LP OpenM (ForInStep σ)} {u : σ} {f : σ → LP OpenM β}
    (h0 : ∀ g hs, A g hs → J 0 u g hs)
    (hb : ∀ i < n, ∀ u, OWL (J i u) (fun r g hs => ∃ v, r = .yield v ∧ J (i + 1) v g hs) (body i u))
    (hf : ∀ r, OWL (J n r) C (f r)) : OWL A C (forIn [:n] u body >>= f) :=
  owl_bind (owl_mono (owl_forIn n J body hb u) h0) hf

/-- The answers so far, as a ghost. -/
theorem owl_ghost {α : Type} {I : Got → Prop} {R : Hist → Prop} {B : α → Got → Hist → Prop} {p : LP OpenM α}
    (h : ∀ h0, R h0 → OWL (fun g hs => I g ∧ hs = h0) B p) : OWL (fun g hs => I g ∧ R hs) B p :=
  fun st t got hs ⟨hI, hR⟩ => h hs hR st t got hs ⟨hI, rfl⟩

theorem owl_exists {α : Type} {I : Got → Prop} {Px : Array F128 → Prop} {h0 : Hist} {B : α → Got → Hist → Prop}
    {p : LP OpenM α} (h : ∀ cs, Px cs → OWL (fun g hs => I g ∧ hs = h0 ++ [cs]) B p) :
    OWL (fun g hs => I g ∧ ∃ cs, hs = h0 ++ [cs] ∧ Px cs) B p :=
  fun st t got hs ⟨hI, cs, he, hx⟩ => h cs hx st t got hs ⟨hI, he⟩

theorem ow_run_bind {α β : Type} {A : Got → Hist → Prop} {B : α → Got → Hist → Prop} {C : β → Got → Hist → Prop}
    {p : LP OpenM α} {st : LigState} {f : α × LigState → TMW OpenM β} (hp : OWL A B p)
    (hf : ∀ a, OW (B a.1) C (f a)) : OW A C (p.run st >>= f) :=
  ow_bind (hp st) hf

theorem spw_run' {α : Type} {p : TMW OpenM α} {t : Transcript} {s : Got} {R : Hist → Prop} {hs : Hist}
    (h : SPW (fun _ hs' => R hs') ((p.run t).run s) hs) : SPW (fun _ hs' => R hs') ((p.run' t).run' s) hs := by
  refine spw_bind (Q := fun _ hs' => R hs') (fun _ _ h => ?_) _ _
    (spw_bind (Q := fun _ hs' => R hs') (fun _ _ h => ?_) _ _ h) <;> exact h

theorem spw_getThe_bind {β : Type} {R : β × Transcript × Got → Hist → Prop} {f : Got → TMW OpenM β}
    {t : Transcript} {got : Got} {hs : Hist} (h : SPW (fun r => R (r.1.1, r.1.2, r.2)) (((f got).run t).run got) hs) :
    SPW (fun r => R (r.1.1, r.1.2, r.2)) (((getThe Got >>= f).run t).run got) hs := h

open _root_.Flock.Firewall.Finish (sendMsg powP) in
/-- **Where the masked Ligerito ends, its `ρ₀, ρ₁` answers are independent**: after `z`'s, `β₀`'s and the `k₀` lanes'
asks from the answers' position `m`, the prover stops at `ρ₀, ρ₁` dependent. -/
theorem ligeritoPZk_rho (coins : Got) (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T : F128)
    (cm0 : _root_.Flock.Firewall.Finish.Commit) (t b : Array F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array F128)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hk : 1 ≤ sch.k0) (hn : sch.n < 2 ^ 62) (m : ℕ) :
    OW (fun g hs => Past coins g ∧ hs.length = m) (fun _ _ hs => Rho (m + (sq sch.n).length + 1 + sch.k0) hs)
      (_root_.Flock.Firewall.Finish.ligeritoPZk (stepAsks coins true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) ms sch T cm0 t b lay h rep tPad extra
        lanePad salt nonce) := by
  unfold _root_.Flock.Firewall.Finish.ligeritoPZk
  apply ow_step (ot_liftTM _) (m' := m) rfl; intro _ _
  apply ow_step (ot_liftTM _) (m' := m) rfl; intro _ _
  apply ow_step (ot_liftTM _) (m' := m) rfl; intro _ _
  apply ow_step (ot_liftTM _) (m' := m) rfl; intro _ _
  dsimp only
  refine ow_run_bind (B := fun _ _ hs => Rho (m + (sq sch.n).length + 1 + sch.k0) hs) ?_
    fun _ => ow_any _ (rho_stable _) _
  apply owl_step (otl_liftT (ot_sampleVecP coins sch.n (by omega))) (m' := m + (sq sch.n).length) rfl; intro _ _
  apply owl_step (otl_modify _) (m' := m + (sq sch.n).length) rfl; intro _ _
  apply owl_step (otl_liftT (ot_liftTM _)) (m' := m + (sq sch.n).length) rfl; intro _ _
  apply owl_step (otl_powP nonce 2) (m' := m + (sq sch.n).length) rfl; intro _ _
  apply owl_step (otl_liftT (ot_sampleP coins)) (m' := m + (sq sch.n).length + 1) rfl; intro _ _
  apply owl_step (otl_modify _) (m' := m + (sq sch.n).length + 1) rfl; intro _ _
  apply owl_step otl_get (m' := m + (sq sch.n).length + 1) rfl; intro _ _
  apply owl_step (otl_sendMsg _) (m' := m + (sq sch.n).length + 1) rfl; intro _ _
  refine owl_forIn_bind (fun j _ g hs => (j < sch.k0 → Past coins g ∧ hs.length = m + (sq sch.n).length + 1 + j) ∧
      (sch.k0 ≤ j → Rho (m + (sq sch.n).length + 1 + sch.k0) hs))
    (fun g hs hA => ⟨fun _ => hA, fun h' => absurd h' (by omega)⟩) ?_
    (fun _ => owl_mono (owl_any _ (rho_stable _) _) fun _ _ hJ => hJ.2 le_rfl)
  intro i hi u
  refine owl_mono (A := fun g hs => Past coins g ∧ hs.length = m + (sq sch.n).length + 1 + i) ?_
    (fun _ _ hJ => hJ.1 hi)
  apply owl_step (otl_liftT (ot_sampleKP coins)) (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
  apply owl_step (otl_modify _) (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
  apply owl_ite_same
  · intro hl
    have hi1 : i + 1 = sch.k0 := by simpa using hl
    apply owl_step otl_get (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
    apply owl_step (otl_liftT (ot_liftTM _)) (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
    apply owl_step (otl_liftT (ot_liftTM _)) (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
    apply owl_ghost; intro h0 hl0
    apply owl_bind (owl_liftT (ow_sampleKP coins h0)); intro rh0
    apply owl_exists; intro cs0 hr0
    apply owl_bind (owl_liftT (ow_sampleKP coins (h0 ++ [cs0]))); intro rh1
    apply owl_exists; intro cs1 hr1
    apply owl_ite_same
    · intro _
      exact owl_throw_bind _ _
    · intro hd
      refine owl_mono (A := fun _ hs => Rho (m + (sq sch.n).length + 1 + sch.k0) hs) ?_ ?_
      · repeat (apply owl_bind (owl_any _ (rho_stable _) _); intro _)
        exact owl_pure _ fun _ _ hR => ⟨_, rfl, fun h' => absurd h' (by omega), fun _ => hR⟩
      · rintro _ _ ⟨-, rfl⟩
        subst hr0 hr1
        have e0 : (h0 ++ [cs0] ++ [cs1]).getD (m + (sq sch.n).length + 1 + sch.k0) #[] = cs0 := by
          simp [List.getD_eq_getElem?_getD, List.getElem?_append_right, hl0, ← hi1]
          rw [show m + (sq sch.n).length + 1 + (i + 1) - (m + (sq sch.n).length + 1 + i + 1) = 0 by omega]
          simp
        have e1 : (h0 ++ [cs0] ++ [cs1]).getD (m + (sq sch.n).length + 1 + sch.k0 + 1) #[] = cs1 := by
          simp [List.getD_eq_getElem?_getD, List.getElem?_append_right, hl0, ← hi1]
          rw [show m + (sq sch.n).length + 1 + (i + 1) - (m + (sq sch.n).length + 1 + i) = 1 by omega]
          simp
        refine ⟨by simp [hl0]; omega, ?_⟩
        rw [e0, e1]
        simpa using hd
  · intro hl
    have hi1 : i + 1 ≠ sch.k0 := by simpa using hl
    apply owl_step otl_get (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
    apply owl_step (otl_sendMsg _) (m' := m + (sq sch.n).length + 1 + i + 1) rfl; intro _ _
    exact owl_pure _ fun _ _ ⟨hI, hl'⟩ => ⟨_, rfl, fun _ => ⟨hI, by omega⟩, fun h' => absurd h' (by omega)⟩

/-- **Where a rep's opening ends past its PIOP, its `ρ₀, ρ₁` answers are independent**: past the ring switch's asks. -/
theorem repOpening_rho (coins : Got) (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule)
    (lay : _root_.Flock.Zk.PadLayout) (tPad rep : ℕ) (g : Array F128) (cm0 : _root_.Flock.Firewall.Finish.Commit)
    (claims : Array _root_.Flock.Claim) (h : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (hc : claims.size < 2 ^ 64) (hc0 : claims.size ≠ 0) (hk : 1 ≤ sch.k0) (hn : sch.n < 2 ^ 62) :
    OW (fun g' hs => Past coins g' ∧ hs.length = 0)
      (fun _ _ hs => Rho (claims.size + 1 + (sq sch.n).length + 1 + sch.k0) hs)
      (_root_.Flock.Firewall.Finish.repOpening (stepAsks coins true)
        (fun lv g s => pure (_root_.Flock.Firewall.Finish.commitFw lv g s)) st sch lay tPad rep g cm0 claims h extra
        lanePad salt) := by
  unfold _root_.Flock.Firewall.Finish.repOpening
  apply ow_step (ringSwitchP_ot coins _ claims hc) (m' := claims.size + 1)
    (by simp [sq, hc0]); intro r _
  obtain ⟨ss, T, sws⟩ := r
  dsimp only
  exact ow_bind (ligeritoPZk_rho coins _ sch _ _ _ _ _ _ _ _ _ _ _ _ hk hn _) fun _ => ow_any _ (rho_stable _) _

/-- **Where a rep's opening ends, its `ρ₀, ρ₁` answers are independent**: after the ring switch's `|xs| + 2` claims'
asks and their batching's, `z`'s, `β₀`'s and the `k₀` lanes'. -/
theorem openStep_rho (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (coins : Got) (hdom : dom.size < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hm : t.st.m < 2 ^ 64)
    (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62) :
    SPW (fun _ hs => Rho (xs.size + 2 + 1 + (sq t.sch.n).length + 1 + t.sch.k0) hs)
      (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi coins) [] := by
  obtain ⟨st, sch, tPad, g, extra, lanePad, cm0⟩ := t
  unfold _root_.Flock.Firewall.Drive.openStep
  dsimp only
  refine spw_run' ?_
  rw [StateT.run_bind, StateT.run_bind]
  refine spw_bind (fun r hs h => ?_) _ _
    (spw_of_na _ [] (piop_na coins st _ _ _ _ _ hdg hcn (by decide) hm _ (Sched.trAlike_init dom hdom) _))
  obtain ⟨hr, rfl⟩ := h
  have hc : r.1.1.2.2.size = xs.size + 2 := by
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map] at hr
    exact hr
  refine spw_getThe_bind (R := fun _ hs => Rho (xs.size + 2 + 1 + (sq sch.n).length + 1 + sch.k0) hs) ?_
  split
  · exact trivial
  · rename_i hne
    have hP : Past coins r.2 := by
      simp only [bne_iff_ne, ne_eq, Decidable.not_not] at hne
      unfold Past; omega
    have ho := repOpening_rho coins st sch (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0) tPad rep g cm0 r.1.1.2.2
      (Array.replicate (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0).len 0) extra lanePad
      (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty)) (by omega) (by omega) hk hn
      r.1.2 r.2 [] ⟨hP, rfl⟩
    rw [hc] at ho
    rw [StateT.run_bind, StateT.run_bind]
    exact spw_bind (fun _ _ h => h) _ _ ho

/-- Every `--zk` schedule asks `z`'s coins, and Ligerito's rounds `k₀ + 2` and `k₀ + 3` (`ρ₀`'s and `ρ₁`'s) two each. -/
def rhoOk (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (sch, _) => decide (1 ≤ sch.n) &&
      ((_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0))[sch.k0 + 2]? == some 2 &&
      ((_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0))[sch.k0 + 3]? == some 2
  | .error _ => true

/-- Every `--zk` schedule, `m < 36`, by the kernel's evaluation. -/
theorem rhoOk_lt : ∀ m, m < 36 → rhoOk m = true := by decide +kernel

theorem sched_rho {m tp : ℕ} {sch : _root_.Flock.Schedule} (h : _root_.Flock.Zk.schedule m = .ok (sch, tp)) :
    1 ≤ sch.n ∧ ∀ e < 2,
      ((_root_.Flock.Firewall.Session.ligeritoCoins sch).toList.filter (· != 0))[sch.k0 + 2 + e]? = some 2 := by
  have hok := rhoOk_lt m (sched_facts h).1
  unfold rhoOk at hok
  rw [h] at hok
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hok
  obtain ⟨⟨hn, h2⟩, h3⟩ := hok
  refine ⟨hn, fun e he => ?_⟩
  rcases (show e = 0 ∨ e = 1 by omega) with rfl | rfl
  · rw [Nat.add_zero]; exact h2
  · rw [show sch.k0 + 2 + 1 = sch.k0 + 3 by omega]; exact h3

theorem openList_length {m tp : ℕ} {sch : _root_.Flock.Schedule} (h : _root_.Flock.Zk.schedule m = .ok (sch, tp))
    {c : ℕ} (hc : c ≠ 0) : (openList c sch).length = c + 1 + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) := by
  have hlig := Loop.ligOk_all m
  unfold Loop.ligOk at hlig
  rw [h] at hlig
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hlig
  rw [Loop.size_filter_toList] at hlig
  unfold openList
  rw [List.length_append, List.length_append, hlig.2]
  simp [sq, hc]

/-- **The opening's `ρ₀, ρ₁` rounds ask two coins each**: past the ring switch's `c + 1` rounds, Ligerito's `k₀ + 2`
and `k₀ + 3`. -/
theorem openList_rho {m tp : ℕ} {sch : _root_.Flock.Schedule} (h : _root_.Flock.Zk.schedule m = .ok (sch, tp))
    {c : ℕ} (hc : c ≠ 0) {e : ℕ} (he : e < 2) : (openList c sch)[c + 1 + (sch.k0 + 2 + e)]? = some 2 := by
  have hlen : ((List.replicate c [7]).flatten ++ sq c).length = c + 1 := by simp [sq, hc]
  unfold openList
  rw [List.getElem?_append_right (by rw [hlen]; omega), hlen, Nat.add_sub_cancel_left]
  exact (sched_rho h).2 e he

/-- **A rep's plan asks two coins at its `ρ₀, ρ₁` rounds.** -/
theorem repPlan_rho {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) {e : ℕ} (he : e < 2) :
    (_root_.Flock.Firewall.Session.repPlan spec st).getD
      (((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)).length +
        ((2 + spec.points * st.regions.size) + 1 + (sch.k0 + 2 + e))) 0 = 2 := by
  have hlt : ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)).length +
      ((2 + spec.points * st.regions.size) + 1 + (sch.k0 + 2 + e)) <
      (((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)) ++
        openList (2 + spec.points * st.regions.size) sch).length := by
    rw [List.length_append, openList_length h (by omega)]
    omega
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, repPlan_list h, List.getElem?_append_left hlt,
    List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left, openList_rho h (by omega) he]
  rfl

/-- **Answers `Q` and `Q + 1` of the coins `cr` are independent `ρ₀, ρ₁`.** -/
def RhoA (Q : ℕ) (cr : Array (Array F128)) : Prop :=
  Q + 2 ≤ cr.size ∧ _root_.Flock.Zk.rhoIndependent
    (⟨(cr.getD Q #[])[0]!, (cr.getD Q #[])[1]!⟩, ⟨(cr.getD (Q + 1) #[])[0]!, (cr.getD (Q + 1) #[])[1]!⟩) = true

theorem rho_extract (cr : Array (Array F128)) (a P k : ℕ) (h : Rho P (((cr.extract a cr.size).toList).take k)) :
    RhoA (a + P) cr := by
  obtain ⟨hl, hr⟩ := h
  simp only [List.length_take, Array.length_toList, Array.size_extract] at hl
  have e : ∀ i, i < P + 2 → (((cr.extract a cr.size).toList).take k).getD i #[] = cr.getD (a + i) #[] := by
    intro i hi
    rw [List.getD_eq_getElem?_getD, List.getElem?_take_of_lt (by omega), Array.getElem?_toList,
      Array.getElem?_extract, Array.getD_eq_getD_getElem?, ite_eq_left (by omega)]
  have e1 := e (P + 1) (by omega)
  rw [← Nat.add_assoc] at e1
  refine ⟨by omega, ?_⟩
  rw [← e P (by omega), ← e1]
  exact hr

/-- **The schedule so far, complete, ends past independent `ρ₀, ρ₁`**: fed the coins of the algebraic phase's rounds
and of all the opening's, a rep's schedule so far reached the opening's end (`feed_full`), which it does only past
independent answers at `ρ₀`'s and `ρ₁`'s rounds (`openStep_rho`). -/
theorem schedSoFar_rho (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (cr : Array (Array F128)) {alg sched : Array (ByteArray × ℕ)}
    (ha : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (h : Loop.schedSoFar t rep dom xs hi cr = .ok sched) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63)
    (hle : alg.size + (openList (xs.size + 2) t.sch).length ≤ cr.size) :
    RhoA (alg.size + (xs.size + 2 + 1 + (sq t.sch.n).length + 1 + t.sch.k0)) cr := by
  unfold Loop.schedSoFar at h
  rw [ha] at h
  simp only [bind, Except.bind] at h
  split at h
  · omega
  · have hf := feed_full _ (cr.extract alg.size cr.size).toList #[] _
      (openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn hq)
      (by simp only [Array.length_toList, Array.size_extract]; omega)
    have ho := openStep_rho t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcn hm hx hk hn
    rcases hfe : (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
      (cr.extract alg.size cr.size).toList #[] with ⟨s', rs⟩
    rw [hfe] at h hf
    cases s' with
    | ask c n k => exact absurd rfl (hf.1 c n k rs)
    | done a =>
      have hr := (spw_feed _ _ _ _ ho hfe).2
      rw [List.nil_append] at hr
      exact rho_extract cr _ _ _ hr
    | fail e => cases h

end Claims.ZKServer.Count

namespace Claims.ZKServer.LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open Loop

/-- **The frames refuse dependent `ρ₀, ρ₁`**: at a `Live` statement the loop serves, on a live history of `refK x`
openings whose last coins the frame takes, table `j`'s rep `r`'s pair coins at the opened coins are independent. At
the rep's inner round 2 (`betaSlot`) the rep's schedule so far is complete (`schedSoFar_size`'s bound), so it ended
past independent answers at its `ρ₀, ρ₁` rounds (`schedSoFar_rho`), Ligerito's `k₀ + 2` and `k₀ + 3`, which ask two
coins each (`repPlan_rho`), and those answers are the opened coins (`wordsAt_slot`). -/
theorem rho_unrefused (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) :
    _root_.Flock.Zk.rhoIndependent (Masked.rhoAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r) = true := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hK := Masked.refK_eq hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j + r) / 2 = j := by omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [hjj]
    exact getElem!_pos _ j hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hrR : ∀ q, q / 2 = (2 * j + r) / 2 → FlockDeploy.repRounds true spec (Masked.stsOf x)[q / 2]! = N + 4 := by
    intro q hq'
    rw [hq']
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hrq : ∀ q, q / 2 = (2 * j + r) / 2 → q < spec.repStreams.size →
      (roundsOf spec (Masked.stsOf x)).getD q 0 = N + 4 := by
    intro q hq' hq
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq, Option.map_some,
      Option.getD_some]
    exact hrR q hq'
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hend : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j + r) / 2]! ≤ refK x := by
    rw [← FlockComplete.ZkTables.repStart_succ, hK, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (by omega)
  rw [hrR _ rfl] at hend
  have hs1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := by
    unfold FlockComplete.repStartZk; omega
  obtain ⟨s, hs⟩ : ∃ s, s = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + N + 2 := ⟨_, rfl⟩
  have hb : s < refK x := by omega
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    (by rw [hlast]; simp [hl]) hr (i := s + 2) (by omega) (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer _
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  rw [List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, N + 2) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) (N + 2) (by rw [hrsl]; exact hq2)
      (by rw [hrq _ rfl hq2]; omega)
    rw [Nat.zero_add] at e
    rw [← e, hs, repStartZk_eq spec _ hq2.le]
    congr 1
    omega
  rw [progOf_snoc x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := s - 1)
    (by rw [List.length_map, hts]; omega), hloc] at ha
  dsimp only at ha
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at ha
    exact absurd ha (answer_owed_none _ rfl)
  rw [hph] at ha
  dsimp only at ha
  obtain ⟨CR0, CR, hCR0, hCR, ha⟩ := repState_args ha
  obtain ⟨alg, sched, hal, hsc⟩ := repState_ok ha
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  obtain ⟨hm36, hk0, hn, hq⟩ := Count.sched_facts (htsch ▸ hz : _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad))
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨-, -, -, -, hdm, -⟩ := placeholder_ok hph
  have hls : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    obtain ⟨p0, rest, hpr⟩ : ∃ p0 rest, ops.take s = p0 :: rest := by
      cases h' : ops.take s with
      | nil => rw [h', List.length_nil] at hts; omega
      | cons p0 rest => exact ⟨p0, rest, rfl⟩
    rw [hpr]
    simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hpts : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size /
      (Masked.stsOf x)[(2 * j + r) / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  have hxn : xs.size = spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm]
  have hxb : xs.size + 2 < 2 ^ 64 := by
    rw [hxs, Loop.extras_size]
    have := Nat.mul_le_mul_left (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size
      ((linkPoints_size_le (Masked.stsOf x)[(2 * j + r) / 2]! _).trans (hls ▸ Nat.min_le_right _ _))
    omega
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : t.st.m < 2 ^ 64 := by rw [hts']; exact Nat.lt_trans hm36 (by decide)
  have hA : alg.size = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[(2 * j + r) / 2]!.m
      (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have e := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts', List.length_map, Array.length_toList] at e
    exact e
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) (htsch ▸ hz :
    _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad)))
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)
      t.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [hA, hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hx2 : xs.size + 2 = 2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by omega
  have hzt : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (t.sch, y.tPad) := htsch ▸ hz
  obtain ⟨hn1, -⟩ := Count.sched_rho hzt
  have hsq : (Count.sq t.sch.n).length = 1 := by simp [Count.sq, show t.sch.n ≠ 0 by omega]
  have hkk : y.sch.k0 = t.sch.k0 := by rw [htsch]
  have hll : y.sch.levels.size = t.sch.levels.size := by rw [htsch]
  have hOL := Count.openList_length hzt
    (c := (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)) (by omega)
  have hAP : alg.size + (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 =
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! := by
    rw [hOL] at hFO
    omega
  have hR := Count.schedSoFar_rho t _ dom xs hi _ hal hsc (hdm ▸ hdom') hdg (placeholder_cap hph) hmb hxb hk0 hn hq
    (by
      rw [hx2, hFO, hCR]
      simp only [Array.size_pop, Array.size_push, Array.size_map, Array.size_range]
      omega)
  rw [hsq] at hR
  obtain ⟨Q, hQ⟩ : ∃ Q, Q = alg.size + (xs.size + 2 + 1 + 1 + 1 + t.sch.k0) := ⟨_, rfl⟩
  rw [← hQ] at hR
  obtain ⟨-, hRi⟩ := hR
  have hW : ∀ i < N + 2, CR.pop.getD i #[] =
      wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2)
        (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + i) := by
    intro i hi'
    rw [hCR, Array.pop_push, crMap_getD _ _ hi']
    generalize hR0 : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) = R0
    rw [show s - 1 - (N + 2) + i + 1 = R0 + i by omega]
  have hplv : ∀ e < 2, loopNr x (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      (alg.size + ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 +
        (t.sch.k0 + 2 + e)))) = 2 := by
    intro e he
    rw [loopNr_rep hsx hq2 (by rw [hrq _ rfl hq2]; omega), hA]
    exact Count.repPlan_rho hzt he
  have hsz : ∀ e < 2, ∀ e' < 2, e' < (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2)
      (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + (alg.size +
        ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 + (t.sch.k0 + 2 + e))))).size := by
    intro e he e' he'
    rw [wordsAt_size x ops (by omega) (by omega), hplv e he]
    exact lt_of_lt_of_le he' (by decide)
  have hrs : Masked.rhoSlot x j r =
      FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + (alg.size +
        ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 + (t.sch.k0 + 2 + 0))) := by
    unfold Masked.rhoSlot
    rw [Masked.repSlot_eq hsx hz]
    omega
  have hrs1 : Masked.rhoSlot x j r + 1 =
      FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + (alg.size +
        ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 + (t.sch.k0 + 2 + 1))) := by
    omega
  unfold Masked.rhoAt
  rw [hrs1, hrs]
  rw [← wordsAt_slot T x ops hl (s := s) (by omega) (by omega) (hsz 0 (by decide) 0 (by decide)),
    ← wordsAt_slot T x ops hl (s := s) (by omega) (by omega) (hsz 0 (by decide) 1 (by decide)),
    ← wordsAt_slot T x ops hl (s := s) (by omega) (by omega) (hsz 1 (by decide) 0 (by decide)),
    ← wordsAt_slot T x ops hl (s := s) (by omega) (by omega) (hsz 1 (by decide) 1 (by decide)),
    ← hW (alg.size + ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 +
      (t.sch.k0 + 2 + 0))) (by omega),
    ← hW (alg.size + ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 +
      (t.sch.k0 + 2 + 1))) (by omega),
    show alg.size + ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 +
      (t.sch.k0 + 2 + 0)) = Q by omega,
    show alg.size + ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 +
      (t.sch.k0 + 2 + 1)) = Q + 1 by omega]
  exact hRi

end Claims.ZKServer.LeanRelease
