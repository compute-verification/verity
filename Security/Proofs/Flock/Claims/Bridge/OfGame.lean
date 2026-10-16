import Proofs.Flock.Claims.Bridge.Coins

/-!
# The public-coin verifier of a game, and its run, path by path

`ofGame k g I d` is the `Claims.Verifier` that plays the live game `g x` on the prover's bytes: each round it reads the
prover's message `a_j` (the move, at a `send` node, through the interface's decoder) and answers with its live coin
`r_j` (the game coin, the projection of `r_j`, at the `coin` node that follows), or with the empty message at a `send`
node that follows a `send` node; at the game's end its reply is the decision `d x b` on the game's outcome `b`, an
oracle program whose `H` calls are the verifier's log. It is stateless: `next` replays the whole transcript every round
(`walk`), as `Claims.Verifier` requires.

* `opn`, `close`, `reply`, `pos`, `walk`: the verifier as a machine over game positions.
* `stratOf H P x ω`: a `Claims` prover read as a bytes strategy of the game (`RStrat`): at each `send` node the bytes it
  sends after the history the verifier's replies make.
* `run_ofGame` (**the run is the game, path by path**): within `k x` rounds, the run's decision is `d x` on
  `outcome`, the game played by `stratOf` on the run's coins, its verifier log is `d x`'s calls, and replaying the
  run's messages (`replay`) gives back that outcome.
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game

variable {β Coin : Type} [Fintype Coin]

/-- A position of the verifier: a game node and its interface. -/
abbrev Pos (Coin : Type) [Fintype Coin] (β : Type) := Σ g : Game β, Iface Coin g

/-- A round opens: at a `send` node the prover's message is the move; elsewhere the game ignores it. -/
def opn : Pos Coin β → Msg → Pos Coin β
  | ⟨.ret b, I⟩, _ => ⟨.ret b, I⟩
  | ⟨.send _ k, I⟩, a => ⟨k (I.1 a), I.2 a⟩
  | ⟨@Game.coin _ C iF iN k, I⟩, _ => ⟨@Game.coin _ C iF iN k, I⟩

/-- A round closes on its coin `u`: a `coin` node takes the game coin, the projection of `u`. -/
def close : Pos Coin β → Coin → Pos Coin β
  | ⟨.ret b, I⟩, _ => ⟨.ret b, I⟩
  | ⟨.send M k, I⟩, _ => ⟨.send M k, I⟩
  | ⟨@Game.coin _ _ _ _ k, I⟩, u => ⟨k (I.1.proj u), I.2 _⟩

/-- `OC` relabelled. -/
def ocMap {α γ : Type} (f : α → γ) : OC α → OC γ
  | .ret a => .ret (f a)
  | .query q k => .query q fun a => ocMap f (k a)

/-- The decision as a reply. -/
def toOut (b : Bool) : Out := if b then .acc else .rej

/-- The round's reply: the decision at the end, the game coin's bytes at a `coin` node, the empty message at a `send`
node. -/
def reply (d : β → OC Bool) : Pos Coin β → Coin → OC Out
  | ⟨.ret b, _⟩, _ => ocMap toOut (d b)
  | ⟨.send _ _, _⟩, _ => .ret (.msg [])
  | ⟨@Game.coin _ _ _ _ _, I⟩, u => .ret (.msg (I.1.enc (I.1.proj u)))

/-- The position after the rounds `l`. -/
def pos (P : Pos Coin β) (l : List (Msg × Coin)) : Pos Coin β := l.foldl (fun Q e => close (opn Q e.1) e.2) P

/-- The verifier's reply to the last round of `l`, from the start `P`. -/
def walk (d : β → OC Bool) (P : Pos Coin β) (l : List (Msg × Coin)) : OC Out :=
  match l.getLast? with
  | none => .ret .rej
  | some e => reply d (opn (pos P l.dropLast) e.1) e.2

/-- **The public-coin verifier of the game `g x`**, deciding by `d x` on the game's outcome, in `k x` rounds. -/
def ofGame (k : Msg → ℕ) (g : Msg → Game β) (I : ∀ x, Iface Coin (g x)) (d : Msg → β → OC Bool) :
    Verifier Coin where
  k := k
  next x as rs := walk (d x) ⟨g x, I x⟩ (as.zip rs)

/-- **A `Claims` prover as a bytes strategy of the game**, from the history `hist` (its messages and the verifier's
replies so far), with its message `pend` of the open round when that round's reply is still to come. -/
def stratOf (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) :
    (g : Game β) → (I : Iface Coin g) → List (Msg × Msg) → Option Msg → RStrat Coin g I
  | .ret _, _, _, _ => PUnit.unit
  | .send _ k, I, hist, pend =>
    let h := match pend with
      | none => hist
      | some a₀ => hist ++ [(a₀, [])]
    let a := ((P.next x h ω).eval H).1
    ⟨a, stratOf H P x ω (k (I.1 a)) (I.2 a) h (some a)⟩
  | @Game.coin _ _ _ _ k, I, hist, pend => fun c =>
    let a := match pend with
      | some a₀ => a₀
      | none => ((P.next x hist ω).eval H).1
    stratOf H P x ω (k c) (I.2 c) (hist ++ [(a, I.1.enc c)]) none

/-! ## Replaying a transcript -/

/-- A round closes on its reply: a `coin` node reads the game coin from the reply's bytes. -/
def closeR : Pos Coin β → Out → Pos Coin β
  | ⟨.ret b, I⟩, _ => ⟨.ret b, I⟩
  | ⟨.send M k, I⟩, _ => ⟨.send M k, I⟩
  | ⟨@Game.coin _ _ _ _ k, I⟩, o => ⟨k (I.1.dec (match o with | .msg m => m | _ => [])), I.2 _⟩

/-- The position after the recorded rounds `l`. -/
def replayPos (P : Pos Coin β) (l : List (Msg × Out)) : Pos Coin β := l.foldl (fun Q e => closeR (opn Q e.1) e.2) P

/-- The outcome at a position, once the game has ended. -/
def result : Pos Coin β → Option β
  | ⟨.ret b, _⟩ => some b
  | _ => none

/-- **The outcome of a transcript**: the game replayed on its messages, the coins read from the verifier's replies. -/
def replay (P : Pos Coin β) (l : List (Msg × Out)) : Option β := result (replayPos P l)

/-! ## The machine -/

theorem ocMap_eval (H : Msg → Msg) {α γ : Type} (f : α → γ) (c : OC α) :
    (ocMap f c).eval H = (f (c.eval H).1, (c.eval H).2) := by
  induction c with
  | ret a => rfl
  | query q k ih => simp only [ocMap, OC.eval, ih]

omit [Fintype Coin] in
theorem toOut_eq_acc (b : Bool) : toOut b = .acc ↔ b = true := by
  cases b <;> simp [toOut]

theorem walk_concat (d : β → OC Bool) (P : Pos Coin β) (l : List (Msg × Coin)) (e : Msg × Coin) :
    walk d P (l ++ [e]) = reply d (opn (pos P l) e.1) e.2 := by
  simp only [walk, List.getLast?_concat, List.dropLast_concat]

theorem pos_concat (P : Pos Coin β) (l : List (Msg × Coin)) (e : Msg × Coin) :
    pos P (l ++ [e]) = close (opn (pos P l) e.1) e.2 := by
  simp only [pos, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem replayPos_concat (P : Pos Coin β) (l : List (Msg × Out)) (e : Msg × Out) :
    replayPos P (l ++ [e]) = closeR (opn (replayPos P l) e.1) e.2 := by
  simp only [replayPos, List.foldl_append, List.foldl_cons, List.foldl_nil]

omit [Fintype Coin] in
theorem zip_range_concat (cs : ℕ → Coin) (as : List Msg) (a : Msg) :
    (as ++ [a]).zip ((List.range (as ++ [a]).length).map cs) =
      as.zip ((List.range as.length).map cs) ++ [(a, cs as.length)] := by
  rw [List.length_append, List.length_singleton, List.range_succ, List.map_append, List.zip_append (by simp)]
  rfl

section Run

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (d : β → OC Bool) (P0 : Pos Coin β)
  (cs : ℕ → Coin)

/-- What a run from a round boundary gives, for the induction. -/
def Concl (n : ℕ) (b : β) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) : Prop :=
  (stepWith H P x ω (fun as => walk d P0 (as.zip ((List.range as.length).map cs))) n hist as t).out =
      toOut ((d b).eval H).1 ∧
    (stepWith H P x ω (fun as => walk d P0 (as.zip ((List.range as.length).map cs))) n hist as t).vLog =
      t.vLog ++ ((d b).eval H).2 ∧
    replay P0 (stepWith H P x ω (fun as => walk d P0 (as.zip ((List.range as.length).map cs))) n hist as t).msgs =
      some b

/-- The induction hypothesis at a round boundary, `n` rounds left. -/
def Inv (n : ℕ) : Prop :=
  ∀ (g : Game β) (I : Iface Coin g) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
    pos P0 (as.zip ((List.range as.length).map cs)) = ⟨g, I⟩ → replayPos P0 t.msgs = ⟨g, I⟩ →
    RoundsLe g I (stratOf H P x ω g I hist none) false n →
    Concl H P x ω d P0 cs n (playOn g I (stratOf H P x ω g I hist none) none fun i => cs (as.length + i)) hist as t

/-- **One round, once its message is open**: the prover's message `a` has taken the verifier to the pending position
`⟨g', I'⟩`; the round's reply and the rounds after give the outcome of the game played from there. -/
theorem after_open (n : ℕ) (IH : Inv H P x ω d P0 cs n) (g' : Game β) (I' : Iface Coin g')
    (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a : Msg) (hpa : ((P.next x hist ω).eval H).1 = a)
    (hw : walk d P0 ((as ++ [a]).zip ((List.range (as ++ [a]).length).map cs)) = reply d ⟨g', I'⟩ (cs as.length))
    (hp : pos P0 ((as ++ [a]).zip ((List.range (as ++ [a]).length).map cs)) = close ⟨g', I'⟩ (cs as.length))
    (hrp : ∀ o, replayPos P0 (t.msgs ++ [(a, o)]) = closeR ⟨g', I'⟩ o)
    (hR : RoundsLe g' I' (stratOf H P x ω g' I' hist (some a)) true n) :
    Concl H P x ω d P0 cs (n + 1)
      (playOn g' I' (stratOf H P x ω g' I' hist (some a)) (some (cs as.length)) fun i => cs (as.length + 1 + i))
      hist as t := by
  cases g' with
  | ret b =>
    have hrep : reply d (⟨.ret b, I'⟩ : Pos Coin β) (cs as.length) = ocMap toOut (d b) := rfl
    have hres : ∀ o, replay P0 (t.msgs ++ [(a, o)]) = some b := fun o => by
      unfold replay
      rw [hrp]
      rfl
    have hb : playOn (.ret b) I' (stratOf H P x ω (.ret b) I' hist (some a)) (some (cs as.length))
        (fun i => cs (as.length + 1 + i)) = b := rfl
    unfold Concl
    rw [hb]
    simp only [stepWith, hpa, hw, hrep, ocMap_eval]
    cases hdb : ((d b).eval H).1 <;> simp only [toOut, Bool.false_eq_true, ↓reduceIte, hres, and_self]
  | send M k =>
    have hR' : RoundsLe (.send M k) I' (stratOf H P x ω (.send M k) I' (hist ++ [(a, [])]) none) false n := by
      cases n with
      | zero => exact hR.elim
      | succ n => exact hR
    have hp' : pos P0 ((as ++ [a]).zip ((List.range (as ++ [a]).length).map cs)) = ⟨.send M k, I'⟩ := hp
    have hrp' : replayPos P0 (t.msgs ++ [(a, .msg [])]) = ⟨.send M k, I'⟩ := hrp _
    obtain ⟨h1, h2, h3⟩ := IH (.send M k) I' (hist ++ [(a, [])]) (as ++ [a])
      ⟨t.msgs ++ [(a, .msg [])], t.pLog ++ ((P.next x hist ω).eval H).2, t.vLog ++ [], t.out⟩ hp' hrp' hR'
    have hrep : reply d (⟨.send M k, I'⟩ : Pos Coin β) (cs as.length) = .ret (.msg []) := rfl
    have hb : playOn (.send M k) I' (stratOf H P x ω (.send M k) I' hist (some a)) (some (cs as.length))
        (fun i => cs (as.length + 1 + i)) =
        playOn (.send M k) I' (stratOf H P x ω (.send M k) I' (hist ++ [(a, [])]) none) none
          (fun i => cs ((as ++ [a]).length + i)) := by
      rw [List.length_append, List.length_singleton]
      rfl
    unfold Concl
    rw [hb]
    simp only [stepWith, hpa, hw, hrep, OC.eval]
    exact ⟨h1, h2.trans (by simp), h3⟩
  | coin C k =>
    have hp' : pos P0 ((as ++ [a]).zip ((List.range (as ++ [a]).length).map cs)) =
        ⟨k (I'.1.proj (cs as.length)), I'.2 (I'.1.proj (cs as.length))⟩ := hp
    have hrp' : replayPos P0 (t.msgs ++ [(a, .msg (I'.1.enc (I'.1.proj (cs as.length))))]) =
        ⟨k (I'.1.proj (cs as.length)), I'.2 (I'.1.proj (cs as.length))⟩ := by
      rw [hrp]
      show (⟨k (I'.1.dec (I'.1.enc (I'.1.proj (cs as.length)))), I'.2 _⟩ : Pos Coin β) = _
      rw [I'.1.dec_enc]
    have hR' : RoundsLe (k (I'.1.proj (cs as.length))) (I'.2 (I'.1.proj (cs as.length)))
        (stratOf H P x ω (k (I'.1.proj (cs as.length))) (I'.2 (I'.1.proj (cs as.length)))
          (hist ++ [(a, I'.1.enc (I'.1.proj (cs as.length)))]) none) false n := hR _
    obtain ⟨h1, h2, h3⟩ := IH (k (I'.1.proj (cs as.length))) (I'.2 _)
      (hist ++ [(a, I'.1.enc (I'.1.proj (cs as.length)))]) (as ++ [a])
      ⟨t.msgs ++ [(a, .msg (I'.1.enc (I'.1.proj (cs as.length))))], t.pLog ++ ((P.next x hist ω).eval H).2,
        t.vLog ++ [], t.out⟩ hp' hrp' hR'
    have hrep : reply d (⟨.coin C k, I'⟩ : Pos Coin β) (cs as.length) =
        .ret (.msg (I'.1.enc (I'.1.proj (cs as.length)))) := rfl
    have hb : playOn (.coin C k) I' (stratOf H P x ω (.coin C k) I' hist (some a)) (some (cs as.length))
        (fun i => cs (as.length + 1 + i)) =
        playOn (k (I'.1.proj (cs as.length))) (I'.2 _)
          (stratOf H P x ω (k (I'.1.proj (cs as.length))) (I'.2 (I'.1.proj (cs as.length)))
            (hist ++ [(a, I'.1.enc (I'.1.proj (cs as.length)))]) none) none
          (fun i => cs ((as ++ [a]).length + i)) := by
      rw [List.length_append, List.length_singleton]
      rfl
    unfold Concl
    rw [hb]
    simp only [stepWith, hpa, hw, hrep, OC.eval]
    exact ⟨h1, h2.trans (by simp), h3⟩

/-- **The rounds of the run are the game's**, from any round boundary. -/
theorem inv_all : ∀ n, Inv H P x ω d P0 cs n := by
  intro n
  induction n with
  | zero =>
    intro g I hist as t _ _ hR
    cases g with
    | ret b => exact absurd hR (Nat.lt_irrefl 0)
    | send M k => exact hR.elim
    | coin C k => exact hR.elim
  | succ n ih =>
    intro g I hist as t hpos hrp hR
    have hw := walk_concat d P0 (as.zip ((List.range as.length).map cs)) (((P.next x hist ω).eval H).1, cs as.length)
    have hp := pos_concat P0 (as.zip ((List.range as.length).map cs)) (((P.next x hist ω).eval H).1, cs as.length)
    rw [← zip_range_concat, hpos] at hw hp
    have hrp' : ∀ o, replayPos P0 (t.msgs ++ [(((P.next x hist ω).eval H).1, o)]) =
        closeR (opn ⟨g, I⟩ ((P.next x hist ω).eval H).1) o := fun o => by rw [replayPos_concat, hrp]
    cases g with
    | ret b =>
      exact after_open H P x ω d P0 cs n ih (.ret b) I hist as t _ rfl hw hp hrp' trivial
    | send M k =>
      have := after_open H P x ω d P0 cs n ih _ _ hist as t _ rfl hw hp hrp' hR
      have e : (fun i => cs (as.length + 1 + i)) = fun i => cs (as.length + (i + 1)) :=
        funext fun i => congrArg cs (by omega)
      rw [e] at this
      exact this
    | coin C k =>
      have := after_open H P x ω d P0 cs n ih (.coin C k) I hist as t _ rfl hw hp hrp' hR
      have e : (fun i => cs (as.length + 1 + i)) = fun i => cs (as.length + (i + 1)) :=
        funext fun i => congrArg cs (by omega)
      rw [e] at this
      exact this

end Run

/-- **The outcome of the run**: the game played by the prover's bytes (`stratOf`) on the run's coins. -/
def outcome [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (k : Msg → ℕ) (g : Msg → Game β)
    (I : ∀ x, Iface Coin (g x)) (x : Msg) (ω : Ω) (r : Fin (k x) → Coin) : β :=
  playOn (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) none (coinSeq r)

/-- **The run is the game, path by path**: whenever the game ends within `k x` rounds, the run's decision is `d x` on
the outcome, the verifier's `H` calls are `d x`'s, and replaying the run's messages gives back the outcome. -/
theorem run_ofGame [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (k : Msg → ℕ) (g : Msg → Game β)
    (I : ∀ x, Iface Coin (g x)) (d : Msg → β → OC Bool) (x : Msg) (ω : Ω) (r : Fin (k x) → Coin)
    (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x)) :
    (run H P (ofGame k g I d) x ω r).out = toOut ((d x (outcome H P k g I x ω r)).eval H).1 ∧
      (run H P (ofGame k g I d) x ω r).vLog = ((d x (outcome H P k g I x ω r)).eval H).2 ∧
      replay ⟨g x, I x⟩ (run H P (ofGame k g I d) x ω r).msgs = some (outcome H P k g I x ω r) := by
  have := inv_all H P x ω (d x) ⟨g x, I x⟩ (coinSeq r) (k x) (g x) (I x) [] [] ⟨[], [], [], .rej⟩ rfl rfl hR
  simp only [List.length_nil, zero_add] at this
  exact this

end Claims.Bridge
