import FlockSoundness.Refine.Live
import FlockSoundness.Refine.Plays
import FlockSoundness.Model.Compiled
import FlockSoundness.Protocol.Refine.LiveSim

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

theorem PlaysT.ret_inv {a b : α} {ts : List Step} (h : PlaysT (.ret a) ts b) : ts = [] ∧ b = a := by
  cases h; exact ⟨rfl, rfl⟩

theorem PlaysT.send_inv {M : Type} {k : M → Game α} {s : Step} {ts : List Step} {a : α}
    (h : PlaysT (.send M k) (s :: ts) a) : ∃ m : M, s = .inl ⟨M, m⟩ ∧ PlaysT (k m) ts a := by
  cases h with
  | send m h => exact ⟨m, rfl, h⟩

/-- A message step at a node of its type continues with that message. -/
theorem PlaysT.send_msg {M : Type} {k : M → Game α} {m : M} {ts : List Step} {a : α}
    (h : PlaysT (.send M k) (.inl ⟨M, m⟩ :: ts) a) : PlaysT (k m) ts a := by
  obtain ⟨m', hm, h⟩ := h.send_inv
  cases hm
  exact h

/-! ## Every message type inhabited -/

theorem AllInh.send_k {M : Type} {k : M → Game α} (h : AllInh (.send M k)) (m : M) : AllInh (k m) := by
  cases h with
  | send _ h => exact h m

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

variable {L : Game β} {K : A × B → Game α}

end Sim

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game

local notation "𝔽" => FlockLevel3.GF128

section Table

variable {P OutT Cp Op : Type} {mPts : ℕ}

variable (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool) (fin : Fin 2 → Array ByteArray → List Msg)

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

end Table

section Sim

variable {P OutT Cp Op : Type} [Nonempty Op] {mPts : ℕ} (out : Cp → P × P → Op → OutT) (pL : LiveOut mPts → Prop)
  (pg : OutT → Prop) (opn : Array ByteArray → Op)

theorem allInh_ib (cap : Cp) {g₀ g₁ : Game P} (h₀ : AllInh g₀) (h₁ : AllInh g₁) :
    AllInh ((interleave g₀ g₁).bind (Kg out cap)) :=
  AllInh.bind (fun _ => .send inferInstance fun _ => .ret _) (AllInh.interleave h₀ h₁)

variable {out pL pg opn}

omit [Nonempty Op] in
theorem InvG.congr {cap : Cp} {L L' : Game (LiveOut mPts)} {g₀ g₁ : Game P} {t₀ t₁ t₀' t₁' : LiveOut mPts → List Step}
    (h : InvG out pL pg opn cap L g₀ g₁ t₀ t₁) (hL : ∀ o, Out L' o → Out L o)
    (h₀ : ∀ o, Out L' o → t₀ o = t₀' o) (h₁ : ∀ o, Out L' o → t₁ o = t₁' o) :
    InvG out pL pg opn cap L' g₀ g₁ t₀' t₁' := fun o ho hp => by
  obtain ⟨pend, a, b, c⟩ := h o (hL o ho) hp
  exact ⟨pend, h₀ o ho ▸ a, h₁ o ho ▸ b, c⟩

variable (out pL pg opn)

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

variable (dg : ByteArray → ByteArray) (dec : ℕ → ByteArray → List Msg) (one : ℕ → Bool)
  (fin : Fin 2 → Array ByteArray → List Msg) (capOf : ByteArray → Cp)

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
