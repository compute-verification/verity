module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.Claims.Sampling.Path

/-!
# The circuit pin as round 2's driver checks it

Round 2's driver refuses a session unless its circuit file's SHA-512 is the committed `circuit_sha512`, the reference
statement's circuit file's (`proofs.verify`), while `pinOk` holds the circuit file to the reference's byte for byte.
`pinHash` is the driver's check: the session statement is the reference one in every field `pinOk` reads but the
circuit file, whose SHA-512 is the reference's. The two differ only at a SHA-512 collision, which `pinColl` outputs.

* `pinHash_of_pinOk`: a statement `pinOk` passes, `pinHash` passes.
* `pinOk_of_pinHash`: a statement `pinHash` passes whose circuit file exposes no collision (`pinColl = none`), `pinOk`
  passes.
* `pinColl_collides`: what `pinColl` outputs is a SHA-512 collision (`Claims.Collision sha512Msg`).

`VHash` is the two-stage verifier with round 2's pin as the driver checks it, and `V` stays the stricter one the
guarantees read. The two give one run unless the message that states the session statements (round `stage + 1`'s)
exposes a collision:

* `run_hash`: a run of `VHash` is the run of `V` on the same tape and coins when `pinCollOf` finds nothing in that
  message of the run of `V`.
* `pr_acc_hash_le`: over the prover's tape, the verifier's coins and any further coins, `VHash` accepts with an event
  at most as often as `V` does, plus the probability that `pinFinder` outputs a SHA-512 collision.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling
open Properties.NetworkCertifier.Bytes (bs bs_inj)

variable (h : Held)

/-- **Template `t`'s session statement is its reference one, its circuit file held by SHA-512**, as round 2's driver
holds it: decoded as `w`, it is `xS₀ t`'s in every field but its sample, its registered reads, its public file and its
circuit file, and its circuit file's SHA-512 is the reference's. -/
def pinHash (t : ℕ) (w : FlockDeploy.Raw) : Bool :=
  match FlockDeploy.decode (h.xS₀ t) with
  | some w₀ =>
    decide ({ w with draw := none, registered := none, publicFile := .empty, circuit := .empty } =
        { w₀ with draw := none, registered := none, publicFile := .empty, circuit := .empty }) &&
      decide (_root_.Flock.Sha512.hash w.circuit = _root_.Flock.Sha512.hash w₀.circuit)
  | none => false

/-- **The collision template `t`'s session statement exposes**: the reference's circuit file and the session's, where
the two differ and have one SHA-512 digest. -/
def pinColl (t : ℕ) (w : FlockDeploy.Raw) : Option (Msg × Msg) :=
  match FlockDeploy.decode (h.xS₀ t) with
  | some w₀ =>
    if w.circuit ≠ w₀.circuit ∧ _root_.Flock.Sha512.hash w.circuit = _root_.Flock.Sha512.hash w₀.circuit then
      some (bs w₀.circuit, bs w.circuit)
    else none
  | none => none

variable {h}

theorem pinHash_of_pinOk {t : ℕ} {w : FlockDeploy.Raw} (hp : pinOk h t w = true) : pinHash h t w = true := by
  unfold pinOk at hp
  unfold pinHash
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd, decide_eq_true_eq] at hp
    have hc : w.circuit = w₀.circuit := by simpa using congrArg FlockDeploy.Raw.circuit hp
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨congrArg (fun v : FlockDeploy.Raw => { v with circuit := .empty }) hp, by rw [hc]⟩

theorem pinOk_of_pinHash {t : ℕ} {w : FlockDeploy.Raw} (hp : pinHash h t w = true) (hn : pinColl h t w = none) :
    pinOk h t w = true := by
  unfold pinHash at hp
  unfold pinColl at hn
  unfold pinOk
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd, Bool.and_eq_true, decide_eq_true_eq] at hp hn
    obtain ⟨e, hh⟩ := hp
    have hc : w.circuit = w₀.circuit := by
      by_contra hne
      simp [hne, hh] at hn
    simp only [decide_eq_true_eq]
    obtain ⟨⟩ := w
    obtain ⟨⟩ := w₀
    simp only [FlockDeploy.Raw.mk.injEq] at e ⊢
    simp_all

theorem pinColl_collides {t : ℕ} {w : FlockDeploy.Raw} {u v : Msg} (hp : pinColl h t w = some (u, v)) :
    Claims.Collision sha512Msg u v := by
  unfold pinColl at hp
  cases hd : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hd] at hp
  | some w₀ =>
    simp only [hd] at hp
    split at hp
    · rename_i hc
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hp)
      refine ⟨fun e => hc.1 (bs_inj e).symm, ?_⟩
      simp only [sha512Msg, bytes_bs, hc.2]
    · simp at hp

theorem pin_eq {t : ℕ} {w : FlockDeploy.Raw} (hn : pinColl h t w = none) : pinHash h t w = pinOk h t w := by
  cases hp : pinOk h t w with
  | true => exact pinHash_of_pinOk hp
  | false =>
    cases hq : pinHash h t w with
    | false => rfl
    | true => simpa [hp] using pinOk_of_pinHash hq hn

/-! ## The verifier with the driver's pin -/

variable (h)

/-- `readsOk` with the circuit pin as round 2's driver checks it (`pinHash`). -/
def readsOkHash (t : ℕ) (x m₁ xS : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) : Bool :=
  match decode x, jsonMsg m₁, secondDraw h c₀ c₁, FlockDeploy.decode xS with
  | some st, some rec, some draw, some w =>
    let own := ownOf h st rec draw c₀
    Spot.SecondStage.interiorOrderOk own && ((stmtJsonOf h t w).map (codesOk h own t)).getD false && pinHash h t w
  | _, _, _, _ => false

/-- `stmtOk` with each template's reads checked by `readsOkHash`. -/
def stmtOkHash (x m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) : Bool :=
  match parse h x, stmtsOf m₂ with
  | some s, some xs =>
    (partition h s).isSome && (two h s).isSome && xs.length == h.ms.length &&
      (List.range h.ms.length).all fun t =>
        (FlockDeploy.V true).k (xs.getD t (h.xS₀ t)) == k₀ h t && sessionOk h t (xs.getD t (h.xS₀ t)) &&
          readsOkHash h t x m₁ (xs.getD t (h.xS₀ t)) c₀ c₁
  | _, _ => false

/-- `next` with round `stage + 1` checked by `stmtOkHash`. -/
def nextHash (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin) : OC Out :=
  if as.length = h.stage + 2 ∧ cs.length = h.stage + 2 then
    .ret (if stmtOkHash h x (as.getD 1 []) (as.getD (h.stage + 1) []) (cs.getD 0 default) (stageOf h cs) then .msg []
      else .rej)
  else next h x as cs

/-- **The two-stage verifier as round 2's driver pins circuits**: `V` with each session's circuit file held to the
reference's by SHA-512. -/
def VHash : Verifier FlockDeploy.Coin where
  k _ := off h h.ms.length + 1
  next := nextHash h

/-- **The collision a session statements' message exposes**: at the first template whose session statement's circuit
file differs from the reference's under one SHA-512 digest, the two files (`pinColl`). -/
def pinCollOf (m₂ : Msg) : Option (Msg × Msg) :=
  match stmtsOf m₂ with
  | some xs =>
    (List.range h.ms.length).findSome? fun t => (FlockDeploy.decode (xs.getD t (h.xS₀ t))).bind (pinColl h t)
  | none => none

/-- **The circuit-pin finder**: over the prover's tape and the verifier's coins, the collision the session statements'
message of the run of `V` exposes (`pinCollOf`). Its cost is the run's SHA-512 calls and two per template, the circuit
files'. -/
def pinFinder (H : Msg → Msg) : FinderFamily := fun Ω _ _ P x =>
  { S := Ω × (Fin ((V h).k x) → FlockDeploy.Coin)
    out := fun s => pinCollOf h ((run H P (V h) x s.1 s.2).msgs.getD (h.stage + 1) ([], .rej)).1
    cost := fun s =>
      (run H P (V h) x s.1 s.2).pLog.length + (run H P (V h) x s.1 s.2).vLog.length + 2 * h.ms.length }

variable {h}

theorem pinCollOf_collides {m₂ u v : Msg} (hc : pinCollOf h m₂ = some (u, v)) : Claims.Collision sha512Msg u v := by
  unfold pinCollOf at hc
  split at hc
  · obtain ⟨t, -, ht⟩ := List.exists_of_findSome?_eq_some hc
    obtain ⟨w, -, hw⟩ := Option.bind_eq_some_iff.1 ht
    exact pinColl_collides hw
  · simp at hc

theorem pinCollOf_none {m₂ : Msg} (hn : pinCollOf h m₂ = none) {xs : List Msg} (hx : stmtsOf m₂ = some xs) {t : ℕ}
    (ht : t < h.ms.length) : (FlockDeploy.decode (xs.getD t (h.xS₀ t))).bind (pinColl h t) = none := by
  unfold pinCollOf at hn
  simp only [hx] at hn
  exact List.findSome?_eq_none_iff.1 hn t (List.mem_range.2 ht)

theorem readsOkHash_eq {t : ℕ} {x m₁ xS : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hn : (FlockDeploy.decode xS).bind (pinColl h t) = none) :
    readsOkHash h t x m₁ xS c₀ c₁ = readsOk h t x m₁ xS c₀ c₁ := by
  unfold readsOkHash readsOk
  cases hw : FlockDeploy.decode xS with
  | none => cases decode x <;> cases jsonMsg m₁ <;> cases secondDraw h c₀ c₁ <;> rfl
  | some w =>
    have hp : pinHash h t w = pinOk h t w := pin_eq (by simpa [hw] using hn)
    cases decode x <;> cases jsonMsg m₁ <;> cases secondDraw h c₀ c₁ <;> simp only [hp]

/-- **The two statement checks agree** where the session statements' message exposes no collision. -/
theorem stmtOkHash_eq {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} (hn : pinCollOf h m₂ = none) :
    stmtOkHash h x m₁ m₂ c₀ c₁ = stmtOk h x m₁ m₂ c₀ c₁ := by
  unfold stmtOkHash stmtOk
  cases parse h x with
  | none => rfl
  | some s =>
    cases hx : stmtsOf m₂ with
    | none => rfl
    | some xs =>
      dsimp only
      congr 1
      rw [Bool.eq_iff_iff, List.all_eq_true, List.all_eq_true]
      refine forall₂_congr fun t ht => ?_
      rw [readsOkHash_eq (pinCollOf_none hn hx (List.mem_range.1 ht))]

theorem nextHash_of_ne {x : Msg} {as : List Msg} {cs : List FlockDeploy.Coin}
    (hl : ¬(as.length = h.stage + 2 ∧ cs.length = h.stage + 2)) : nextHash h x as cs = next h x as cs := by
  unfold nextHash
  rw [ite_eq_right hl]

theorem nextHash_eq {x : Msg} {as : List Msg} {cs : List FlockDeploy.Coin}
    (hn : as.length = h.stage + 2 → pinCollOf h (as.getD (h.stage + 1) []) = none) :
    nextHash h x as cs = next h x as cs := by
  by_cases hl : as.length = h.stage + 2 ∧ cs.length = h.stage + 2
  · unfold nextHash
    rw [ite_eq_left hl, next_stmt h x (hl.2.trans hl.1.symm) hl.1, stmtOkHash_eq (hn hl.1)]
  · exact nextHash_of_ne hl

/-! ## One run, unless the session statements' message exposes a collision -/

/-- **Two verifiers that answer alike off `L` prover messages give one run** wherever they answer alike at the run's
own first `L`. -/
theorem stepWith_congr_at {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (v₁ v₂ : List Msg → OC Out)
    (L : ℕ) (hv : ∀ bs : List Msg, bs.length ≠ L → v₁ bs = v₂ bs) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), as = t.msgs.map Prod.fst →
      v₁ (((stepWith H P x ω v₂ n hist as t).msgs.map Prod.fst).take L) =
        v₂ (((stepWith H P x ω v₂ n hist as t).msgs.map Prod.fst).take L) →
      stepWith H P x ω v₁ n hist as t = stepWith H P x ω v₂ n hist as t := by
  intro n
  induction n with
  | zero => intro hist as t _ _; rfl
  | succ n ih =>
    intro hist as t has hL
    have key : v₁ (as ++ [((P.next x hist ω).eval H).1]) = v₂ (as ++ [((P.next x hist ω).eval H).1]) := by
      by_cases hl : (as ++ [((P.next x hist ω).eval H).1]).length = L
      · obtain ⟨e, he⟩ := stepWith_succ_msgs H P x ω v₂ n hist as t
        rw [he] at hL
        have ht : ((t.msgs ++ (((P.next x hist ω).eval H).1,
            ((v₂ (as ++ [((P.next x hist ω).eval H).1])).eval H).1) :: e).map Prod.fst).take L =
            as ++ [((P.next x hist ω).eval H).1] := by
          rw [← hl, has]
          simp [List.take_append]
        rwa [ht] at hL
      · exact hv _ hl
    simp only [stepWith]
    rw [key]
    split
    · rename_i m hm
      refine ih _ _ _ (by simp [has]) ?_
      simpa only [stepWith, hm] using hL
    · rfl
    · rfl

theorem getD_fst_take (ms : List (Msg × Out)) {n k : ℕ} (hk : n < k) :
    ((ms.map Prod.fst).take k).getD n [] = (ms.getD n ([], .rej)).1 := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hk, List.getElem?_map]
  cases ms[n]? <;> rfl

/-- **The coupling, run by run**: on any tape and coins, the run of `VHash` is the run of `V` when the latter's
session statements' message exposes no collision. -/
theorem run_hash {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin ((V h).k x) → FlockDeploy.Coin)
    (hn : pinCollOf h ((run H P (V h) x ω r).msgs.getD (h.stage + 1) ([], .rej)).1 = none) :
    run H P (VHash h) x ω r = run H P (V h) x ω r := by
  unfold run
  refine stepWith_congr_at H P x ω _ _ (h.stage + 2) ?_ _ _ _ _ rfl ?_
  · exact fun _ hl => nextHash_of_ne fun hl' => hl hl'.1
  · exact nextHash_eq fun _ => by
      rw [getD_fst_take _ (show h.stage + 1 < h.stage + 2 by omega)]
      exact hn

/-- **The coupling**: over the prover's tape, the verifier's coins and any further coins `C`, `VHash` accepts with
event `E` at most as often as `V` does, plus the probability that the circuit-pin finder outputs a SHA-512 collision. -/
theorem pr_acc_hash_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (P : Prover Ω) (x : Msg) {C : Type}
    [Fintype C] [Nonempty C] (E : Ω × (Fin ((V h).k x) → FlockDeploy.Coin) × C → Prop) :
    Pr (fun s : Ω × (Fin ((V h).k x) → FlockDeploy.Coin) × C => (run H P (VHash h) x s.1 s.2.1).out = .acc ∧ E s) ≤
      Pr (fun s : Ω × (Fin ((V h).k x) → FlockDeploy.Coin) × C => (run H P (V h) x s.1 s.2.1).out = .acc ∧ E s) +
        (pinFinder h H Ω P x).Pr sha512Msg := by
  have hb : Pr (fun s : Ω × (Fin ((V h).k x) → FlockDeploy.Coin) × C => ∃ u v,
      pinCollOf h ((run H P (V h) x s.1 s.2.1).msgs.getD (h.stage + 1) ([], .rej)).1 = some (u, v) ∧
        Claims.Collision sha512Msg u v) = (pinFinder h H Ω P x).Pr sha512Msg :=
    Pr_drop fun ω r => ∃ u v,
      pinCollOf h ((run H P (V h) x ω r).msgs.getD (h.stage + 1) ([], .rej)).1 = some (u, v) ∧
        Claims.Collision sha512Msg u v
  refine Claims.Sampling.Pr_or_le (fun s hs => ?_) le_rfl hb.le
  cases hc : pinCollOf h ((run H P (V h) x s.1 s.2.1).msgs.getD (h.stage + 1) ([], .rej)).1 with
  | none => exact Or.inl (by rw [← run_hash H P x s.1 s.2.1 hc]; exact hs)
  | some p =>
    obtain ⟨u, v⟩ := p
    exact Or.inr ⟨u, v, rfl, pinCollOf_collides hc⟩

end SpotFlock
