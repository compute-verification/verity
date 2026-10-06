import Proofs.Flock.Soundness.Refine.Live
import Proofs.Flock.Soundness.Refine.Plays
import Proofs.Flock.Soundness.Model.Compiled

/-!
# The table's simulation (the transfer, R11c)

A live prover for one table's session (`liveTable`) is simulated in the model's table (`modelTable`, which `tableC` is,
`tableC_eq_modelTable`). The simulated prover sends the cap `root_B` decodes to, shares the link points, and follows the
rounds: for a round on rep `r`'s stream, the model's rep `r` takes the messages decoded from the round's bytes, then
draws its coins with the live round's. At the end it takes the reps' final messages and the openings, decoded from the
proof files. Wherever the model's game doesn't fit what was decoded, no accepted live run goes on, and the model plays
any strategy.

* `PlaysT g ts a`: `g`, played with the typed steps `ts` (messages and coins, in order), ends with `a`.
* `Decodes`: on every accepted live outcome, each rep's model game runs on the tape decoded from its stream and its
  proof file (`repTape`), and the model accepts those runs. This is what the refinement and framing (R11b) provide.
* `live_le`: against any live prover, the live verdict's probability is at most the model's against the simulated
  prover.
-/

namespace FlockSoundness.Game

variable {α β A B : Type}

/-! ## Runs on one typed tape -/

/-- A step of a run: a message (`inl`) or a coin (`inr`), each as its type and value. -/
abbrev Step := Msg ⊕ Msg

/-- **A run on one tape.** `g`, played with the messages and coins of `ts` in their order, ends with `a`. A step is
taken only when its type is the node's. -/
inductive PlaysT : {α : Type} → Game α → List Step → α → Prop
  | ret {α : Type} (a : α) : PlaysT (.ret a) [] a
  | send {α M : Type} {k : M → Game α} (m : M) {ts : List Step} {a : α} :
      PlaysT (k m) ts a → PlaysT (.send M k) (.inl ⟨M, m⟩ :: ts) a
  | coin {α C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} (x : C) {ts : List Step} {a : α} :
      PlaysT (k x) ts a → PlaysT (@Game.coin α C iF iN k) (.inr ⟨C, x⟩ :: ts) a

theorem PlaysT.ret_inv {a b : α} {ts : List Step} (h : PlaysT (.ret a) ts b) : ts = [] ∧ b = a := by
  cases h; exact ⟨rfl, rfl⟩

theorem PlaysT.send_inv {M : Type} {k : M → Game α} {s : Step} {ts : List Step} {a : α}
    (h : PlaysT (.send M k) (s :: ts) a) : ∃ m : M, s = .inl ⟨M, m⟩ ∧ PlaysT (k m) ts a := by
  cases h with
  | send m h => exact ⟨m, rfl, h⟩

theorem PlaysT.coin_inv {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} {s : Step} {ts : List Step}
    {a : α} (h : PlaysT (@Game.coin α C iF iN k) (s :: ts) a) : ∃ x : C, s = .inr ⟨C, x⟩ ∧ PlaysT (k x) ts a := by
  cases h with
  | coin x h => exact ⟨x, rfl, h⟩

theorem PlaysT.nil_inv {g : Game α} {a : α} (h : PlaysT g [] a) : g = .ret a := by
  cases h; rfl

/-- A message step at a node of its type continues with that message. -/
theorem PlaysT.send_msg {M : Type} {k : M → Game α} {m : M} {ts : List Step} {a : α}
    (h : PlaysT (.send M k) (.inl ⟨M, m⟩ :: ts) a) : PlaysT (k m) ts a := by
  obtain ⟨m', hm, h⟩ := h.send_inv
  cases hm
  exact h

theorem PlaysT.coin_val {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} {x : C} {ts : List Step}
    {a : α} (h : PlaysT (@Game.coin α C iF iN k) (.inr ⟨C, x⟩ :: ts) a) : PlaysT (k x) ts a := by
  obtain ⟨x', hx, h⟩ := h.coin_inv
  cases hx
  exact h

/-! ## Every message type inhabited -/

/-- Every message node of `g` has a nonempty type, so a prover can always answer. -/
inductive AllInh : {α : Type} → Game α → Prop
  | ret {α : Type} (a : α) : AllInh (.ret a)
  | send {α M : Type} {k : M → Game α} : Nonempty M → (∀ m, AllInh (k m)) → AllInh (.send M k)
  | coin {α C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} :
      (∀ c, AllInh (k c)) → AllInh (@Game.coin α C iF iN k)

theorem AllInh.strategy : ∀ {g : Game α}, AllInh g → Nonempty (Strategy g)
  | .ret _, _ => ⟨PUnit.unit⟩
  | .send _ _, .send ⟨m⟩ h => let ⟨s⟩ := (h m).strategy; ⟨⟨m, s⟩⟩
  | @Game.coin _ _ _ _ _, .coin h => ⟨fun c => Classical.choice (h c).strategy⟩

theorem AllInh.send_k {M : Type} {k : M → Game α} (h : AllInh (.send M k)) (m : M) : AllInh (k m) := by
  cases h with
  | send _ h => exact h m

theorem AllInh.coin_k {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (h : AllInh (@Game.coin α C iF iN k)) (c : C) : AllInh (k c) := by
  cases h with
  | coin h => exact h c

theorem AllInh.bind {f : α → Game β} (hf : ∀ a, AllInh (f a)) : ∀ {g : Game α}, AllInh g → AllInh (g.bind f)
  | .ret a, _ => hf a
  | .send _ _, .send hM h => .send hM fun m => AllInh.bind hf (h m)
  | @Game.coin _ _ _ _ _, .coin h => .coin fun c => AllInh.bind hf (h c)

theorem AllInh.map (f : α → β) {g : Game α} (h : AllInh g) : AllInh (g.map f) := AllInh.bind (fun _ => .ret _) h

theorem AllInh.interleaveWith {adv : Game B → Game (A × B)} {fin : B → Game (A × B)}
    (hadv : ∀ g₂, AllInh g₂ → AllInh (adv g₂)) (hfin : ∀ b, AllInh (fin b)) :
    ∀ {g₂ : Game B}, AllInh g₂ → AllInh (interleaveWith adv fin g₂)
  | .ret b, _ => hfin b
  | .send _ _, h@(.send hM hk) => .send ⟨true⟩ fun first => by
      cases first
      · exact .send hM fun m => AllInh.interleaveWith hadv hfin (hk m)
      · exact hadv _ h
  | @Game.coin _ _ _ _ _, h@(.coin hk) => .send ⟨true⟩ fun first => by
      cases first
      · exact .coin fun c => AllInh.interleaveWith hadv hfin (hk c)
      · exact hadv _ h

theorem AllInh.interleave : ∀ {g₁ : Game A} {g₂ : Game B}, AllInh g₁ → AllInh g₂ → AllInh (interleave g₁ g₂)
  | .ret _, _, _, h₂ => h₂.map _
  | .send M k, _, h₁@(.send hM hk), h₂ =>
      AllInh.interleaveWith (adv := fun g₂ => .send M fun m => Game.interleave (k m) g₂)
        (fin := fun b => (Game.send M k).map fun a => (a, b))
        (fun _ h => .send hM fun m => AllInh.interleave (hk m) h) (fun _ => h₁.map _) h₂
  | @Game.coin _ C iF iN k, _, h₁@(.coin hk), h₂ =>
      AllInh.interleaveWith (adv := fun g₂ => @Game.coin _ C iF iN fun c => Game.interleave (k c) g₂)
        (fin := fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))
        (fun _ h => .coin fun c => AllInh.interleave (hk c) h) (fun _ => h₁.map _) h₂

/-! ## The model's moves in an interleave -/

namespace Sim

variable {pL : β → Prop} {pg : α → Prop}

/-- No live run from here is accepted: the model plays any strategy. -/
noncomputable def deadOf {L : Game β} {g : Game α} (h : ∀ b, Out L b → ¬ pL b) (hg : AllInh g) : Sim pL pg L g :=
  .dead (prob_eq_zero_of_out h) (Classical.choice hg.strategy)

variable {L : Game β} {K : A × B → Game α}

/-- The model sends `m` in the first of two concurrent interactions. -/
noncomputable def send₀ {M : Type} {k : M → Game A} (m : M) : (g₁ : Game B) →
    Sim pL pg L ((interleave (k m) g₁).bind K) → Sim pL pg L ((interleave (.send M k) g₁).bind K)
  | .ret b, d => .model m (show Sim pL pg L (((k m).map fun a => (a, b)).bind K) from Plays.interleave_ret (k m) b ▸ d)
  | .send _ _, d => .model true (.model m d)
  | @Game.coin _ _ _ _ _, d => .model true (.model m d)

/-- The model sends `m` in the second of two concurrent interactions. -/
noncomputable def send₁ {M : Type} {k : M → Game B} (m : M) : (g₀ : Game A) →
    Sim pL pg L ((interleave g₀ (k m)).bind K) → Sim pL pg L ((interleave g₀ (.send M k)).bind K)
  | .ret _, d => .model m d
  | .send _ _, d => .model false (.model m d)
  | @Game.coin _ _ _ _ _, d => .model false (.model m d)

/-- The first interaction's coin, matched to the live one by `e`. -/
noncomputable def coin₀ {C C' : Type} {iF : Fintype C} {iN : Nonempty C} {iF' : Fintype C'} {iN' : Nonempty C'}
    {kL : C' → Game β} {k : C → Game A} (e : C ≃ C') : (g₁ : Game B) →
    (∀ c, Sim pL pg (kL (e c)) ((interleave (k c) g₁).bind K)) →
    Sim pL pg (@Game.coin β C' iF' iN' kL) ((interleave (@Game.coin A C iF iN k) g₁).bind K)
  | .ret b, d => .coin e fun c =>
      show Sim pL pg (kL (e c)) (((k c).map fun a => (a, b)).bind K) from Plays.interleave_ret (k c) b ▸ d c
  | .send _ _, d => .model true (.coin e d)
  | @Game.coin _ _ _ _ _, d => .model true (.coin e d)

/-- The second interaction's coin, matched to the live one by `e`. -/
noncomputable def coin₁ {C C' : Type} {iF : Fintype C} {iN : Nonempty C} {iF' : Fintype C'} {iN' : Nonempty C'}
    {kL : C' → Game β} {k : C → Game B} (e : C ≃ C') : (g₀ : Game A) →
    (∀ c, Sim pL pg (kL (e c)) ((interleave g₀ (k c)).bind K)) →
    Sim pL pg (@Game.coin β C' iF' iN' kL) ((interleave g₀ (@Game.coin B C iF iN k)).bind K)
  | .ret _, d => .coin e d
  | .send _ _, d => .model false (.coin e d)
  | @Game.coin _ _ _ _ _, d => .model false (.coin e d)

end Sim

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game

local notation "𝔽" => FlockLevel3.GF128

section Table

variable {P OutT Cp Op : Type} {mPts : ℕ}

/-- **The model table's shape**, `tableC`'s: the cap, the link points, the two reps concurrently, then the openings. -/
def modelTable (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (out : Cp → P × P → Op → OutT) : Game OutT := do
  let cap ← recv Cp
  let pts ← draw (Fin 2 → Fin mPts → 𝔽)
  let pend ← interleave (G pts 0) (G pts 1)
  let o ← recv Op
  pure (out cap pend o)

/-- Rep `r`'s stream. -/
def pick (acc : List Round × List Round) (r : Fin 2) : List Round := if r = 0 then acc.1 else acc.2

/-- A round's coins as the model reads them: one field element when the decoder says so and the round has one coin,
else the vector of all of them. -/
def coinMsg (one : Bool) (rd : Round) : Msg :=
  if one ∧ rd.n = 1 then ⟨𝔽, fOf rd.coins[0]!⟩ else ⟨Fin rd.n → 𝔽, fun i => fOf rd.coins[i.val]!⟩

variable (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool) (fin : Fin 2 → Array ByteArray → List Msg)

/-- A stream's tape from its round `j` on: each round's messages, decoded from its bytes, then its coins. -/
def tapeFrom : ℕ → List Round → List Step
  | _, [] => []
  | j, rd :: rs => (dec j (rd.bytes.getD ByteArray.empty)).map .inl ++ .inr (coinMsg (one j) rd) :: tapeFrom (j + 1) rs

/-- Rep `r`'s whole tape in a live outcome: its rounds, then its final messages, decoded from the proof files. -/
def repTape (o : LiveOut mPts) (r : Fin 2) : List Step :=
  tapeFrom dec one 0 (pick (o.rep0, o.rep1) r) ++ (fin r o.proofs).map .inl

/-- Rep `r`'s tape after the rounds `acc` has. -/
def restTape (acc : List Round × List Round) (r : Fin 2) (o : LiveOut mPts) : List Step :=
  tapeFrom dec one (pick acc r).length ((pick (o.rep0, o.rep1) r).drop (pick acc r).length) ++ (fin r o.proofs).map .inl

/-- Rep `r`'s tape from round `j`'s coins on. -/
def midTape (r : Fin 2) (j : ℕ) (o : LiveOut mPts) : List Step :=
  .inr (coinMsg (one j) ((pick (o.rep0, o.rep1) r).getD j default)) ::
    (tapeFrom dec one (j + 1) ((pick (o.rep0, o.rep1) r).drop (j + 1)) ++ (fin r o.proofs).map .inl)

theorem pick_addRound (acc : List Round × List Round) (r r' : Fin 2) (rd : Round) :
    pick (addRound acc r rd) r' = if r' = r then pick acc r' ++ [rd] else pick acc r' := by
  unfold pick addRound
  rcases Fin.exists_fin_two.1 ⟨r, rfl⟩ with rfl | rfl <;> rcases Fin.exists_fin_two.1 ⟨r', rfl⟩ with rfl | rfl <;> simp

theorem restTape_eq_mid (acc : List Round × List Round) (r : Fin 2) (b : ByteArray) (rd : Round) (hb : rd.bytes = some b)
    (l : List Round) (o : LiveOut mPts) (ho : pick (o.rep0, o.rep1) r = pick acc r ++ rd :: l) :
    restTape dec one fin acc r o = (dec (pick acc r).length b).map .inl ++ midTape dec one fin r (pick acc r).length o := by
  simp only [restTape, midTape, ho, List.drop_left', List.getD_eq_getElem?_getD, List.getElem?_append_right le_rfl,
    Nat.sub_self, List.getElem?_cons_zero, Option.getD_some, tapeFrom, hb, Option.getD_some, List.append_assoc,
    List.cons_append]
  rw [show (pick acc r ++ rd :: l).drop ((pick acc r).length + 1) = l by rw [List.drop_append]; simp]

theorem midTape_eq_rest (acc : List Round × List Round) (r : Fin 2) (rd : Round) (l : List Round) (o : LiveOut mPts)
    (ho : pick (o.rep0, o.rep1) r = pick acc r ++ rd :: l) :
    midTape dec one fin r (pick acc r).length o = .inr (coinMsg (one (pick acc r).length) rd) ::
      restTape dec one fin (addRound acc r rd) r o := by
  simp only [midTape, restTape, pick_addRound, ite_true, List.length_append, List.length_singleton, ho,
    List.getD_eq_getElem?_getD, List.getElem?_append_right le_rfl, Nat.sub_self, List.getElem?_cons_zero,
    Option.getD_some]

theorem restTape_other (acc : List Round × List Round) (r r' : Fin 2) (h : r' ≠ r) (rd : Round) (o : LiveOut mPts) :
    restTape dec one fin (addRound acc r rd) r' o = restTape dec one fin acc r' o := by
  simp only [restTape, pick_addRound, h, ite_false]

theorem coinMsg_vec (dg : ByteArray → ByteArray) (b : ByteArray) (k : ℕ) (c : Fin k → 𝔽) (one : Bool)
    (h : ¬ (one = true ∧ k = 1)) : coinMsg one (roundOf dg b k c) = ⟨Fin k → 𝔽, c⟩ := by
  simp only [coinMsg, roundOf, h, ite_false]
  congr 1
  funext i
  rw [getElem!_pos _ i.val (by simp), Array.getElem_ofFn]
  rfl

theorem coinMsg_one (dg : ByteArray → ByteArray) (b : ByteArray) (c : Fin 1 → 𝔽) :
    coinMsg true (roundOf dg b 1 c) = ⟨𝔽, c 0⟩ := by
  simp only [coinMsg, roundOf, and_self, ite_true]
  rw [getElem!_pos _ 0 (by simp), Array.getElem_ofFn]
  rfl

end Table

section Sim

variable {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
  (pg : OutT → Prop) (opn : Array ByteArray → Op)

/-- The model's continuation after the reps: the openings, then the output. -/
def Kg (cap : Cp) (pend : P × P) : Game OutT := .send Op fun o => .ret (out cap pend o)

/-- The live session's continuation after the rounds: the proof files, then the outcome. -/
def KL (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (rs : List Round × List Round) : Game (LiveOut mPts) :=
  .send (Array ByteArray) fun proofs => .ret ⟨rootB, pts, rs.1, rs.2, proofs⟩

theorem allInh_ib (cap : Cp) {g₀ g₁ : Game P} (h₀ : AllInh g₀) (h₁ : AllInh g₁) :
    AllInh ((interleave g₀ g₁).bind (Kg out cap)) :=
  AllInh.bind (fun _ => .send inferInstance fun _ => .ret _) (AllInh.interleave h₀ h₁)

/-- **The simulation's invariant** at a live subgame `L` and the model reps' positions `g₀ g₁`: on every accepted live
outcome, the reps' remaining tapes `t₀ o` and `t₁ o` run the positions to outputs the model accepts. -/
def InvG (cap : Cp) (L : Game (LiveOut mPts)) (g₀ g₁ : Game P) (t₀ t₁ : LiveOut mPts → List Step) : Prop :=
  ∀ o, Out L o → pL o → ∃ pend : P × P, PlaysT g₀ (t₀ o) pend.1 ∧ PlaysT g₁ (t₁ o) pend.2 ∧ pg (out cap pend (opn o.proofs))

variable {out pL pg opn}

omit [Nonempty Op] in
theorem InvG.congr {cap : Cp} {L L' : Game (LiveOut mPts)} {g₀ g₁ : Game P} {t₀ t₁ t₀' t₁' : LiveOut mPts → List Step}
    (h : InvG out pL pg opn cap L g₀ g₁ t₀ t₁) (hL : ∀ o, Out L' o → Out L o)
    (h₀ : ∀ o, Out L' o → t₀ o = t₀' o) (h₁ : ∀ o, Out L' o → t₁ o = t₁' o) :
    InvG out pL pg opn cap L' g₀ g₁ t₀' t₁' := fun o ho hp => by
  obtain ⟨pend, a, b, c⟩ := h o (hL o ho) hp
  exact ⟨pend, h₀ o ho ▸ a, h₁ o ho ▸ b, c⟩

variable (out pL pg opn)

open Classical in
/-- Feed the messages `ms` into the model's first rep, then continue at its new position. -/
noncomputable def feed₀ (cap : Cp) (L : Game (LiveOut mPts)) :
    (ms : List Msg) → (g₀ g₁ : Game P) → (t₀ t₁ : LiveOut mPts → List Step) → AllInh g₀ → AllInh g₁ →
    InvG out pL pg opn cap L g₀ g₁ (fun o => ms.map .inl ++ t₀ o) t₁ →
    ((g₀' : Game P) → AllInh g₀' → InvG out pL pg opn cap L g₀' g₁ t₀ t₁ →
      Sim pL pg L ((interleave g₀' g₁).bind (Kg out cap))) →
    Sim pL pg L ((interleave g₀ g₁).bind (Kg out cap))
  | [], g₀, _, _, _, h₀, _, hI, k => k g₀ h₀ (hI.congr (fun _ h => h) (fun _ _ => by simp) (fun _ _ => rfl))
  | ⟨M', m'⟩ :: ms, .send M kk, g₁, t₀, t₁, h₀, h₁, hI, k =>
      if hM : M' = M then
        Sim.send₀ (cast hM m') g₁ (feed₀ cap L ms (kk (cast hM m')) g₁ t₀ t₁ (h₀.send_k _) h₁ (fun o ho hp => by
          obtain ⟨pend, a, b, c⟩ := hI o ho hp
          subst hM
          exact ⟨pend, PlaysT.send_msg (by simpa using a), b, c⟩) k)
      else Sim.deadOf (fun o ho hp => by
          obtain ⟨pend, a, -⟩ := hI o ho hp
          obtain ⟨m, hm, -⟩ := PlaysT.send_inv (by simpa using a)
          simp only [Sum.inl.injEq] at hm
          exact hM (congrArg Sigma.fst hm)) (allInh_ib out cap h₀ h₁)
  | _ :: _, .ret _, _, _, _, h₀, h₁, hI, _ => Sim.deadOf (fun o ho hp => by
          obtain ⟨pend, a, -⟩ := hI o ho hp
          have := (PlaysT.ret_inv a).1
          simp at this) (allInh_ib out cap h₀ h₁)
  | _ :: _, @Game.coin _ _ _ _ _, _, _, _, h₀, h₁, hI, _ => Sim.deadOf (fun o ho hp => by
          obtain ⟨pend, a, -⟩ := hI o ho hp
          obtain ⟨x, hx, -⟩ := PlaysT.coin_inv (by simpa using a)
          simp at hx) (allInh_ib out cap h₀ h₁)

open Classical in
/-- Feed the messages `ms` into the model's second rep, then continue at its new position. -/
noncomputable def feed₁ (cap : Cp) (L : Game (LiveOut mPts)) :
    (ms : List Msg) → (g₀ g₁ : Game P) → (t₀ t₁ : LiveOut mPts → List Step) → AllInh g₀ → AllInh g₁ →
    InvG out pL pg opn cap L g₀ g₁ t₀ (fun o => ms.map .inl ++ t₁ o) →
    ((g₁' : Game P) → AllInh g₁' → InvG out pL pg opn cap L g₀ g₁' t₀ t₁ →
      Sim pL pg L ((interleave g₀ g₁').bind (Kg out cap))) →
    Sim pL pg L ((interleave g₀ g₁).bind (Kg out cap))
  | [], _, g₁, _, _, _, h₁, hI, k => k g₁ h₁ (hI.congr (fun _ h => h) (fun _ _ => rfl) (fun _ _ => by simp))
  | ⟨M', m'⟩ :: ms, g₀, .send M kk, t₀, t₁, h₀, h₁, hI, k =>
      if hM : M' = M then
        Sim.send₁ (cast hM m') g₀ (feed₁ cap L ms g₀ (kk (cast hM m')) t₀ t₁ h₀ (h₁.send_k _) (fun o ho hp => by
          obtain ⟨pend, a, b, c⟩ := hI o ho hp
          subst hM
          exact ⟨pend, a, PlaysT.send_msg (by simpa using b), c⟩) k)
      else Sim.deadOf (fun o ho hp => by
          obtain ⟨pend, -, b, -⟩ := hI o ho hp
          obtain ⟨m, hm, -⟩ := PlaysT.send_inv (by simpa using b)
          simp only [Sum.inl.injEq] at hm
          exact hM (congrArg Sigma.fst hm)) (allInh_ib out cap h₀ h₁)
  | _ :: _, _, .ret _, _, _, h₀, h₁, hI, _ => Sim.deadOf (fun o ho hp => by
          obtain ⟨pend, -, b, -⟩ := hI o ho hp
          have := (PlaysT.ret_inv b).1
          simp at this) (allInh_ib out cap h₀ h₁)
  | _ :: _, _, @Game.coin _ _ _ _ _, _, _, h₀, h₁, hI, _ => Sim.deadOf (fun o ho hp => by
          obtain ⟨pend, -, b, -⟩ := hI o ho hp
          obtain ⟨x, hx, -⟩ := PlaysT.coin_inv (by simpa using b)
          simp at hx) (allInh_ib out cap h₀ h₁)

theorem out_live (dg : ByteArray → ByteArray) (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ)
    (acc : List Round × List Round) {o : LiveOut mPts} (h : Out ((liveRounds dg n acc).bind (KL rootB pts)) o) :
    o.rootB = rootB ∧ o.pts = pts ∧ ∃ l₀ l₁, o.rep0 = acc.1 ++ l₀ ∧ o.rep1 = acc.2 ++ l₁ := by
  obtain ⟨rs, hrs, ho⟩ := Out.of_bind h
  obtain ⟨l₀, l₁, rfl⟩ := liveRounds_out dg n acc hrs
  unfold KL at ho
  cases ho with
  | send proofs ho => cases ho; exact ⟨rfl, rfl, l₀, l₁, rfl, rfl⟩

theorem out_coin {k : ℕ} {kL : (Fin k → 𝔽) → Game (LiveOut mPts)} {o : LiveOut mPts}
    (h : Out (.coin (Fin k → 𝔽) kL) o) : ∃ c, Out (kL c) o := by
  cases h with
  | coin c h => exact ⟨c, h⟩

/-- A single coin as a one-coin round's. -/
def eOne {C : Type} {k : ℕ} (hC : C = 𝔽) (hk : k = 1) : C ≃ (Fin k → 𝔽) :=
  (Equiv.cast hC).trans ((Equiv.funUnique (Fin 1) 𝔽).symm.trans (Equiv.cast (by rw [hk])))

theorem coinMsg_e {C : Type} (dg : ByteArray → ByteArray) (b : ByteArray) (k : ℕ) (one : Bool) (x : C)
    (e : C ≃ (Fin k → 𝔽)) (he : (∃ (hs : one = true ∧ k = 1) (hC : C = 𝔽), e = eOne hC hs.2) ∨
      (¬ (one = true ∧ k = 1) ∧ ∃ hC : C = (Fin k → 𝔽), e = Equiv.cast hC)) :
    coinMsg one (roundOf dg b k (e x)) = ⟨C, x⟩ := by
  rcases he with ⟨⟨h1, rfl⟩, rfl, rfl⟩ | ⟨hs, rfl, rfl⟩
  · subst h1
    rw [coinMsg_one]
    rfl
  · rw [coinMsg_vec dg b _ _ one hs]
    rfl

variable (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool)
  (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp)

open Classical in
/-- The first rep's coin, drawn with the live round's: matched when the model's coin has the type the round decodes to;
past any other node no accepted run goes on. -/
noncomputable def coinAt₀ (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ) (acc : List Round × List Round)
    (b : ByteArray) (k : ℕ) (g₁ : Game P) (h₁ : AllInh g₁)
    (next : (c : Fin k → 𝔽) → (g₀' : Game P) → AllInh g₀' →
      InvG out pL pg opn (capOf rootB) ((liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts)) g₀' g₁
        (restTape dec one fin (addRound acc 0 (roundOf dg b k c)) 0)
        (restTape dec one fin (addRound acc 0 (roundOf dg b k c)) 1) →
      Sim pL pg ((liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts))
        ((interleave g₀' g₁).bind (Kg out (capOf rootB)))) :
    (g₀ : Game P) → AllInh g₀ →
    InvG out pL pg opn (capOf rootB) (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts))
      g₀ g₁ (midTape dec one fin 0 acc.1.length) (restTape dec one fin acc 1) →
    Sim pL pg (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts))
      ((interleave g₀ g₁).bind (Kg out (capOf rootB)))
  | @Game.coin _ C iF iN kk, h₀, hI =>
      have key : ∀ (e : C ≃ (Fin k → 𝔽)), ((∃ (hs : one acc.1.length = true ∧ k = 1) (hC : C = 𝔽), e = eOne hC hs.2) ∨
          (¬ (one acc.1.length = true ∧ k = 1) ∧ ∃ hC : C = (Fin k → 𝔽), e = Equiv.cast hC)) → ∀ x : C,
          InvG out pL pg opn (capOf rootB) ((liveRounds dg n (addRound acc 0 (roundOf dg b k (e x)))).bind (KL rootB pts))
            (kk x) g₁ (restTape dec one fin (addRound acc 0 (roundOf dg b k (e x))) 0)
            (restTape dec one fin (addRound acc 0 (roundOf dg b k (e x))) 1) := fun e he x o ho hp => by
        obtain ⟨pend, a, b', c'⟩ := hI o (.coin (e x) ho) hp
        obtain ⟨-, -, l₀, l₁, h0, -⟩ := out_live dg rootB pts n _ ho
        have hp0 : pick (o.rep0, o.rep1) 0 = pick acc 0 ++ roundOf dg b k (e x) :: l₀ := by
          simp [pick, h0, addRound]
        rw [show acc.1.length = (pick acc 0).length from rfl, midTape_eq_rest dec one fin acc 0 _ l₀ o hp0,
          show (pick acc 0).length = acc.1.length from rfl, coinMsg_e dg b k _ x e he] at a
        refine ⟨pend, a.coin_val, ?_, c'⟩
        rw [restTape_other dec one fin acc 0 1 (by decide)]
        exact b'
      if hs : one acc.1.length = true ∧ k = 1 then
        if hC : C = 𝔽 then
          Sim.coin₀ (eOne hC hs.2) g₁ fun x => next _ (kk x) (h₀.coin_k x) (key _ (.inl ⟨hs, hC, rfl⟩) x)
        else Sim.deadOf (fun o ho hp => by
          obtain ⟨c, hc⟩ := out_coin ho
          obtain ⟨pend, a, -⟩ := hI o ho hp
          obtain ⟨-, -, l₀, l₁, h0, -⟩ := out_live dg rootB pts n _ hc
          have hp0 : pick (o.rep0, o.rep1) 0 = pick acc 0 ++ roundOf dg b k c :: l₀ := by simp [pick, h0, addRound]
          rw [show acc.1.length = (pick acc 0).length from rfl, midTape_eq_rest dec one fin acc 0 _ l₀ o hp0] at a
          obtain ⟨x, hx, -⟩ := a.coin_inv
          obtain ⟨h1, rfl⟩ := hs
          rw [show (pick acc 0).length = acc.1.length from rfl, h1, coinMsg_one] at hx
          simp only [Sum.inr.injEq] at hx
          exact hC (congrArg Sigma.fst hx).symm) (allInh_ib out (capOf rootB) h₀ h₁)
      else
        if hC : C = (Fin k → 𝔽) then
          Sim.coin₀ (Equiv.cast hC) g₁ fun x => next _ (kk x) (h₀.coin_k x) (key _ (.inr ⟨hs, hC, rfl⟩) x)
        else Sim.deadOf (fun o ho hp => by
          obtain ⟨c, hc⟩ := out_coin ho
          obtain ⟨pend, a, -⟩ := hI o ho hp
          obtain ⟨-, -, l₀, l₁, h0, -⟩ := out_live dg rootB pts n _ hc
          have hp0 : pick (o.rep0, o.rep1) 0 = pick acc 0 ++ roundOf dg b k c :: l₀ := by simp [pick, h0, addRound]
          rw [show acc.1.length = (pick acc 0).length from rfl, midTape_eq_rest dec one fin acc 0 _ l₀ o hp0] at a
          obtain ⟨x, hx, -⟩ := a.coin_inv
          rw [show (pick acc 0).length = acc.1.length from rfl, coinMsg_vec dg b k c _ hs] at hx
          simp only [Sum.inr.injEq] at hx
          exact hC (congrArg Sigma.fst hx).symm) (allInh_ib out (capOf rootB) h₀ h₁)
  | .ret _, h₀, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, a, -⟩ := hI o ho hp
      have := (PlaysT.ret_inv a).1
      simp [midTape] at this) (allInh_ib out (capOf rootB) h₀ h₁)
  | .send _ _, h₀, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, a, -⟩ := hI o ho hp
      obtain ⟨m, hm, -⟩ := PlaysT.send_inv (by simpa [midTape] using a)
      simp at hm) (allInh_ib out (capOf rootB) h₀ h₁)

open Classical in
/-- The second rep's coin, as `coinAt₀`. -/
noncomputable def coinAt₁ (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ) (acc : List Round × List Round)
    (b : ByteArray) (k : ℕ) (g₀ : Game P) (h₀ : AllInh g₀)
    (next : (c : Fin k → 𝔽) → (g₁' : Game P) → AllInh g₁' →
      InvG out pL pg opn (capOf rootB) ((liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts)) g₀ g₁'
        (restTape dec one fin (addRound acc 1 (roundOf dg b k c)) 0)
        (restTape dec one fin (addRound acc 1 (roundOf dg b k c)) 1) →
      Sim pL pg ((liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts))
        ((interleave g₀ g₁').bind (Kg out (capOf rootB)))) :
    (g₁ : Game P) → AllInh g₁ →
    InvG out pL pg opn (capOf rootB) (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts))
      g₀ g₁ (restTape dec one fin acc 0) (midTape dec one fin 1 acc.2.length) →
    Sim pL pg (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts))
      ((interleave g₀ g₁).bind (Kg out (capOf rootB)))
  | @Game.coin _ C iF iN kk, h₁, hI =>
      have key : ∀ (e : C ≃ (Fin k → 𝔽)), ((∃ (hs : one acc.2.length = true ∧ k = 1) (hC : C = 𝔽), e = eOne hC hs.2) ∨
          (¬ (one acc.2.length = true ∧ k = 1) ∧ ∃ hC : C = (Fin k → 𝔽), e = Equiv.cast hC)) → ∀ x : C,
          InvG out pL pg opn (capOf rootB) ((liveRounds dg n (addRound acc 1 (roundOf dg b k (e x)))).bind (KL rootB pts))
            g₀ (kk x) (restTape dec one fin (addRound acc 1 (roundOf dg b k (e x))) 0)
            (restTape dec one fin (addRound acc 1 (roundOf dg b k (e x))) 1) := fun e he x o ho hp => by
        obtain ⟨pend, a, b', c'⟩ := hI o (.coin (e x) ho) hp
        obtain ⟨-, -, l₀, l₁, -, h1⟩ := out_live dg rootB pts n _ ho
        have hp1 : pick (o.rep0, o.rep1) 1 = pick acc 1 ++ roundOf dg b k (e x) :: l₁ := by
          simp [pick, h1, addRound]
        rw [show acc.2.length = (pick acc 1).length from rfl, midTape_eq_rest dec one fin acc 1 _ l₁ o hp1,
          show (pick acc 1).length = acc.2.length from rfl, coinMsg_e dg b k _ x e he] at b'
        refine ⟨pend, ?_, b'.coin_val, c'⟩
        rw [restTape_other dec one fin acc 1 0 (by decide)]
        exact a
      if hs : one acc.2.length = true ∧ k = 1 then
        if hC : C = 𝔽 then
          Sim.coin₁ (eOne hC hs.2) g₀ fun x => next _ (kk x) (h₁.coin_k x) (key _ (.inl ⟨hs, hC, rfl⟩) x)
        else Sim.deadOf (fun o ho hp => by
          obtain ⟨c, hc⟩ := out_coin ho
          obtain ⟨pend, -, a, -⟩ := hI o ho hp
          obtain ⟨-, -, l₀, l₁, -, h1⟩ := out_live dg rootB pts n _ hc
          have hp1 : pick (o.rep0, o.rep1) 1 = pick acc 1 ++ roundOf dg b k c :: l₁ := by simp [pick, h1, addRound]
          rw [show acc.2.length = (pick acc 1).length from rfl, midTape_eq_rest dec one fin acc 1 _ l₁ o hp1] at a
          obtain ⟨x, hx, -⟩ := a.coin_inv
          obtain ⟨h1', rfl⟩ := hs
          rw [show (pick acc 1).length = acc.2.length from rfl, h1', coinMsg_one] at hx
          simp only [Sum.inr.injEq] at hx
          exact hC (congrArg Sigma.fst hx).symm) (allInh_ib out (capOf rootB) h₀ h₁)
      else
        if hC : C = (Fin k → 𝔽) then
          Sim.coin₁ (Equiv.cast hC) g₀ fun x => next _ (kk x) (h₁.coin_k x) (key _ (.inr ⟨hs, hC, rfl⟩) x)
        else Sim.deadOf (fun o ho hp => by
          obtain ⟨c, hc⟩ := out_coin ho
          obtain ⟨pend, -, a, -⟩ := hI o ho hp
          obtain ⟨-, -, l₀, l₁, -, h1⟩ := out_live dg rootB pts n _ hc
          have hp1 : pick (o.rep0, o.rep1) 1 = pick acc 1 ++ roundOf dg b k c :: l₁ := by simp [pick, h1, addRound]
          rw [show acc.2.length = (pick acc 1).length from rfl, midTape_eq_rest dec one fin acc 1 _ l₁ o hp1] at a
          obtain ⟨x, hx, -⟩ := a.coin_inv
          rw [show (pick acc 1).length = acc.2.length from rfl, coinMsg_vec dg b k c _ hs] at hx
          simp only [Sum.inr.injEq] at hx
          exact hC (congrArg Sigma.fst hx).symm) (allInh_ib out (capOf rootB) h₀ h₁)
  | .ret _, h₁, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, -, a, -⟩ := hI o ho hp
      have := (PlaysT.ret_inv a).1
      simp [midTape] at this) (allInh_ib out (capOf rootB) h₀ h₁)
  | .send _ _, h₁, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, -, a, -⟩ := hI o ho hp
      obtain ⟨m, hm, -⟩ := PlaysT.send_inv (by simpa [midTape] using a)
      simp at hm) (allInh_ib out (capOf rootB) h₀ h₁)

/-- A round on the first rep's stream: the model's first rep takes the messages decoded from the round's bytes, then
its coin. -/
noncomputable def round₀ (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ) (acc : List Round × List Round)
    (b : ByteArray) (k : ℕ) (g₀ g₁ : Game P) (h₀ : AllInh g₀) (h₁ : AllInh g₁)
    (hI : InvG out pL pg opn (capOf rootB) (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts))
      g₀ g₁ (restTape dec one fin acc 0) (restTape dec one fin acc 1))
    (next : (c : Fin k → 𝔽) → (g₀' : Game P) → AllInh g₀' →
      InvG out pL pg opn (capOf rootB) ((liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts)) g₀' g₁
        (restTape dec one fin (addRound acc 0 (roundOf dg b k c)) 0)
        (restTape dec one fin (addRound acc 0 (roundOf dg b k c)) 1) →
      Sim pL pg ((liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts))
        ((interleave g₀' g₁).bind (Kg out (capOf rootB)))) :
    Sim pL pg (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 0 (roundOf dg b k c))).bind (KL rootB pts))
      ((interleave g₀ g₁).bind (Kg out (capOf rootB))) :=
  feed₀ out pL pg opn (capOf rootB) _ (dec acc.1.length b) g₀ g₁ (midTape dec one fin 0 acc.1.length)
    (restTape dec one fin acc 1) h₀ h₁
    (hI.congr (fun _ h => h) (fun o ho => by
        obtain ⟨c, hc⟩ := out_coin ho
        obtain ⟨-, -, l₀, l₁, h0, -⟩ := out_live dg rootB pts n _ hc
        have hp0 : pick (o.rep0, o.rep1) 0 = pick acc 0 ++ roundOf dg b k c :: l₀ := by simp [pick, h0, addRound]
        exact restTape_eq_mid dec one fin acc 0 b _ rfl l₀ o hp0) (fun _ _ => rfl))
    (fun g₀' h₀' hI' => coinAt₀ out pL pg opn dg dec one fin capOf rootB pts n acc b k g₁ h₁ next g₀' h₀' hI')

/-- A round on the second rep's stream, as `round₀`. -/
noncomputable def round₁ (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (n : ℕ) (acc : List Round × List Round)
    (b : ByteArray) (k : ℕ) (g₀ g₁ : Game P) (h₀ : AllInh g₀) (h₁ : AllInh g₁)
    (hI : InvG out pL pg opn (capOf rootB) (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts))
      g₀ g₁ (restTape dec one fin acc 0) (restTape dec one fin acc 1))
    (next : (c : Fin k → 𝔽) → (g₁' : Game P) → AllInh g₁' →
      InvG out pL pg opn (capOf rootB) ((liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts)) g₀ g₁'
        (restTape dec one fin (addRound acc 1 (roundOf dg b k c)) 0)
        (restTape dec one fin (addRound acc 1 (roundOf dg b k c)) 1) →
      Sim pL pg ((liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts))
        ((interleave g₀ g₁').bind (Kg out (capOf rootB)))) :
    Sim pL pg (.coin (Fin k → 𝔽) fun c => (liveRounds dg n (addRound acc 1 (roundOf dg b k c))).bind (KL rootB pts))
      ((interleave g₀ g₁).bind (Kg out (capOf rootB))) :=
  feed₁ out pL pg opn (capOf rootB) _ (dec acc.2.length b) g₀ g₁ (restTape dec one fin acc 0)
    (midTape dec one fin 1 acc.2.length) h₀ h₁
    (hI.congr (fun _ h => h) (fun _ _ => rfl) (fun o ho => by
        obtain ⟨c, hc⟩ := out_coin ho
        obtain ⟨-, -, l₀, l₁, -, h1⟩ := out_live dg rootB pts n _ hc
        have hp1 : pick (o.rep0, o.rep1) 1 = pick acc 1 ++ roundOf dg b k c :: l₁ := by simp [pick, h1, addRound]
        exact restTape_eq_mid dec one fin acc 1 b _ rfl l₁ o hp1))
    (fun g₁' h₁' hI' => coinAt₁ out pL pg opn dg dec one fin capOf rootB pts n acc b k g₀ h₀ next g₁' h₁' hI')

/-- Both reps done: the model sends the openings the proof files decode to, and accepts where the live verdict holds. -/
noncomputable def finish (cap : Cp) (o₀ : LiveOut mPts) : (g₀ g₁ : Game P) → AllInh g₀ → AllInh g₁ →
    InvG out pL pg opn cap (.ret o₀) g₀ g₁ (fun _ => []) (fun _ => []) →
    Sim pL pg (.ret o₀) ((interleave g₀ g₁).bind (Kg out cap))
  | .ret a₀, .ret a₁, _, _, hI => .model (opn o₀.proofs) (.ret fun hp => by
      obtain ⟨pend, a, b, c⟩ := hI o₀ (.ret o₀) hp
      obtain ⟨-, h0⟩ := PlaysT.ret_inv a
      obtain ⟨-, h1⟩ := PlaysT.ret_inv b
      have : pend = (a₀, a₁) := Prod.ext h0 h1
      subst this
      exact c)
  | .ret _, .send _ _, h₀, h₁, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, -, b, -⟩ := hI o ho hp; exact absurd (PlaysT.nil_inv b) (by simp)) (allInh_ib out cap h₀ h₁)
  | .ret _, @Game.coin _ _ _ _ _, h₀, h₁, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, -, b, -⟩ := hI o ho hp; exact absurd (PlaysT.nil_inv b) (by simp)) (allInh_ib out cap h₀ h₁)
  | .send _ _, _, h₀, h₁, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, a, -⟩ := hI o ho hp; exact absurd (PlaysT.nil_inv a) (by simp)) (allInh_ib out cap h₀ h₁)
  | @Game.coin _ _ _ _ _, _, h₀, h₁, hI => Sim.deadOf (fun o ho hp => by
      obtain ⟨pend, a, -⟩ := hI o ho hp; exact absurd (PlaysT.nil_inv a) (by simp)) (allInh_ib out cap h₀ h₁)

/-- The live rounds are over: the model's reps take their final messages, decoded from the proof files, then the
openings. -/
noncomputable def simEnd (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (acc : List Round × List Round)
    (g₀ g₁ : Game P) (h₀ : AllInh g₀) (h₁ : AllInh g₁)
    (hI : InvG out pL pg opn (capOf rootB) (KL rootB pts acc) g₀ g₁ (restTape dec one fin acc 0) (restTape dec one fin acc 1)) :
    Sim pL pg (KL rootB pts acc) ((interleave g₀ g₁).bind (Kg out (capOf rootB))) :=
  .live fun proofs =>
    feed₀ out pL pg opn (capOf rootB) (.ret ⟨rootB, pts, acc.1, acc.2, proofs⟩) (fin 0 proofs) g₀ g₁ (fun _ => [])
      (fun o => (fin 1 o.proofs).map .inl) h₀ h₁
      (hI.congr (fun _ ho => .send proofs ho) (fun o ho => by cases ho; simp [restTape, pick, tapeFrom])
        (fun o ho => by cases ho; simp [restTape, pick, tapeFrom]))
      (fun g₀' h₀' hI₁ => feed₁ out pL pg opn (capOf rootB) _ (fin 1 proofs) g₀' g₁ (fun _ => []) (fun _ => []) h₀' h₁
        (hI₁.congr (fun _ h => h) (fun _ _ => rfl) (fun o ho => by cases ho; simp))
        (fun g₁' h₁' hI₂ => finish out pL pg opn (capOf rootB) _ g₀' g₁' h₀' h₁' hI₂))

/-- **The rounds**: the model follows the live prover round by round, on the rep each round names. -/
noncomputable def simR (rootB : ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) : (n : ℕ) → (acc : List Round × List Round) →
    (g₀ g₁ : Game P) → AllInh g₀ → AllInh g₁ →
    InvG out pL pg opn (capOf rootB) ((liveRounds dg n acc).bind (KL rootB pts)) g₀ g₁ (restTape dec one fin acc 0)
      (restTape dec one fin acc 1) →
    Sim pL pg ((liveRounds dg n acc).bind (KL rootB pts)) ((interleave g₀ g₁).bind (Kg out (capOf rootB)))
  | 0, acc, g₀, g₁, h₀, h₁, hI => simEnd out pL pg opn dec one fin capOf rootB pts acc g₀ g₁ h₀ h₁ hI
  | n + 1, acc, g₀, g₁, h₀, h₁, hI => .live fun
    | none => simEnd out pL pg opn dec one fin capOf rootB pts acc g₀ g₁ h₀ h₁
        (hI.congr (fun _ h => .send none h) (fun _ _ => rfl) (fun _ _ => rfl))
    | some ⟨⟨0, h0⟩, b, k⟩ => round₀ out pL pg opn dg dec one fin capOf rootB pts n acc b k g₀ g₁ h₀ h₁
        (hI.congr (fun _ h => .send (some (⟨⟨0, h0⟩, b, k⟩ : Submit)) h) (fun _ _ => rfl) (fun _ _ => rfl))
        (fun _ g₀' h₀' hI' => simR rootB pts n _ g₀' g₁ h₀' h₁ hI')
    | some ⟨⟨1, h1⟩, b, k⟩ => round₁ out pL pg opn dg dec one fin capOf rootB pts n acc b k g₀ g₁ h₀ h₁
        (hI.congr (fun _ h => .send (some (⟨⟨1, h1⟩, b, k⟩ : Submit)) h) (fun _ _ => rfl) (fun _ _ => rfl))
        (fun _ g₁' h₁' hI' => simR rootB pts n _ g₀ g₁' h₀ h₁' hI')
    | some ⟨⟨_ + 2, h⟩, _, _⟩ => absurd h (by omega)

/-- **The decoder is right on accepted sessions.** When the live verdict holds, each rep's model game runs on the tape
decoded from its stream and its proof file, and the model's verdict on those runs, with the cap `root_B` decodes to and
the openings the proof files decode to, holds. -/
def Decodes (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) : Prop :=
  ∀ o, pL o → ∃ pend : P × P, PlaysT (G o.pts 0) (repTape dec one fin o 0) pend.1 ∧
    PlaysT (G o.pts 1) (repTape dec one fin o 1) pend.2 ∧ pg (out (capOf o.rootB) pend (opn o.proofs))

/-- **The table's simulation.** The model sends the cap `root_B` decodes to, shares the link points, follows the rounds,
and sends the openings the proof files decode to. -/
noncomputable def simTable (N : ℕ) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (hinh : ∀ pts r, AllInh (G pts r))
    (hdec : Decodes out pL pg opn dec one fin capOf G) : Sim pL pg (liveTable dg mPts N) (modelTable G out) :=
  .live fun rootB => .model (capOf rootB) (.coin (Equiv.refl _) fun pts =>
    simR out pL pg opn dg dec one fin capOf rootB pts N ([], []) (G pts 0) (G pts 1) (hinh pts 0) (hinh pts 1)
      (fun o ho hp => by
        obtain ⟨hr, hpts, -⟩ := out_live dg rootB pts N ([], []) ho
        obtain ⟨pend, a, b, c⟩ := hdec o hp
        rw [hpts] at a b
        rw [hr] at c
        exact ⟨pend, by simpa [restTape, repTape, pick] using a, by simpa [restTape, repTape, pick] using b, c⟩))

/-- **The transfer, for one table.** Against any live prover, the executable's verdict on the live session holds with
probability at most the model table's verdict against the simulated prover, whenever the decoder is right on accepted
sessions. -/
theorem live_le (N : ℕ) (G : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (hinh : ∀ pts r, AllInh (G pts r))
    (hdec : Decodes out pL pg opn dec one fin capOf G) (Pr : Strategy (liveTable dg mPts N)) :
    prob pL (liveTable dg mPts N) Pr ≤
      prob pg (modelTable G out) ((simTable out pL pg opn dg dec one fin capOf N G hinh hdec).strategy Pr) :=
  Sim.prob_le _ Pr

end Sim

open Classical in
/-- `tableC` is the model table with the compiled reps and its verdict. -/
theorem tableC_eq_modelTable {K D : Type} [Field K] [Fintype K] [DecidableEq K] (A : Model.Arith 𝔽 K)
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (S : Model.Statement) (sch : Accounting.Schedule) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) :
    Model.tableC A H E S sch hm ptLocal mPts =
      modelTable (fun pts (r : Fin 2) => Model.repC A r.val S sch hm (Model.extraClaims S ptLocal mPts pts))
        (fun cap pend o => (⟨pend.1.verdict o && pend.2.verdict o &&
          decide (Model.OpensOK H E cap 0 pend.1 o ∧ Model.OpensOK H E cap 1 pend.2 o), cap, pend, o⟩ : Model.OutC 𝔽 D)) :=
  rfl

end FlockSoundness.Refine
