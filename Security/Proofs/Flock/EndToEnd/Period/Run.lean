import Definitions.Flock.Game.Prob

/-!
# A period of audits, run one after another (trusted)

A period runs its sampling blocks' audits in order: sampling block `i`'s game `G i`, then sampling block `i + 1`'s, and
so on, `T` sampling blocks in all (`periodG`). One prover strategy plays the whole period, so the strategy sampling
block `i` is played with is whatever the period's strategy leaves at that point, and may depend on everything the prover
saw in the sampling blocks before.

`periodRun` is the period with the prover's choices fixed by its strategy: a game of coins only, whose output lists,
for each sampling block in order, a record `rec i τ b` of the strategy `τ` that played sampling block `i` and the
sampling block's output `b`. A record may read the strategy: C-Flock's wrong units are those of the transcript the
prover is committed to at registration, which is a function of its strategy (`Flock.Guarantees.SamplingBlock.wrongSet`).
An event of the period is a predicate on that list, and its probability is `value` of the run, which has no prover
message left, so `value` is the exact probability over the verifier's coins. `compTo` is the period played up to one
sampling block, then any game at the strategy that plays it there.

`PeriodGood G good i T s` says every sampling block's strategy, on every run of the period, satisfies `good`: it is how
a per-session hypothesis about the prover (such as `cr/sha-512` for its knowledge finders) is taken at exactly the
strategies the period's prover plays. `PeriodPlays G i T s j τ` says `τ` is one of them, the strategy sampling block `j`
is played with on some run, so a statement takes such a hypothesis flat, `∀ j τ, PeriodPlays G i T s j τ → good j τ`.
-/

namespace FlockSoundness.Game

variable {α δ γ : Type}

/-- **Play `g` by `s`, then continue.** The prover's moves in `g` are fixed by `s`, and at an end `a` of `g` the run
continues with `k a s'`, where `s'` is the strategy `s` leaves for `f a`. -/
def runBind (f : α → Game δ) (k : (a : α) → Strategy (f a) → Game γ) :
    (g : Game α) → Strategy (g.bind f) → Game γ
  | .ret a, s => k a s
  | .send _ kk, s => runBind f k (kk s.1) s.2
  | @Game.coin _ C iF iN kk, s => @Game.coin _ C iF iN fun c => runBind f k (kk c) (s c)

/-- `Q a s'` holds at every end `a` of `g` played by `s`, with `s'` the strategy `s` leaves for `f a`. -/
def AtEnds (f : α → Game δ) (Q : (a : α) → Strategy (f a) → Prop) :
    (g : Game α) → Strategy (g.bind f) → Prop
  | .ret a, s => Q a s
  | .send _ kk, s => AtEnds f Q (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => ∀ c : C, AtEnds f Q (kk c) (s c)

variable {β : ℕ → Type}

/-- **A period's game**: sampling blocks `i, i + 1, …, i + T − 1`, each sampling block's game `G` played after the one
before ends. -/
def periodG (G : (i : ℕ) → Game (β i)) : ℕ → ℕ → Game Unit
  | _, 0 => .ret ()
  | i, T + 1 => (G i).bind fun _ => periodG G (i + 1) T

/-- **A period's run** at the prover strategy `s`: a game of coins only, whose output lists, for each sampling block `i`
in order, the record `rec i τ b` of the strategy `τ` that played it (what `s` leaves at that point) and its output
`b`. -/
def periodRun {ρ : Type} (G : (i : ℕ) → Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → ρ) :
    (i T : ℕ) → Strategy (periodG G i T) → Game (List ρ)
  | _, 0, _ => .ret []
  | i, T + 1, s => runBind (fun _ => periodG G (i + 1) T)
      (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·)) (G i) s

/-- The strategy `runBind f k g s` is played by: `ks a s'` at each end `a` of `g`. -/
def runBindStrat (f : α → Game δ) (k : (a : α) → Strategy (f a) → Game γ)
    (ks : (a : α) → (s' : Strategy (f a)) → Strategy (k a s')) :
    (g : Game α) → (s : Strategy (g.bind f)) → Strategy (runBind f k g s)
  | .ret a, s => ks a s
  | .send _ kk, s => runBindStrat f k ks (kk s.1) s.2
  | @Game.coin _ _ _ _ kk, s => fun c => runBindStrat f k ks (kk c) (s c)

/-- **A period up to sampling block `i + d`, then `K`**: sampling blocks `i, …, i + d − 1` played by `s`, then the game
`K (i + d) τ`, with `τ` the strategy `s` leaves for sampling block `i + d`. A finder that reads one sampling block of a
period plays the sampling blocks before it this way, on fresh coins. -/
def compTo (G : (i : ℕ) → Game (β i)) (K : (j : ℕ) → Strategy (G j) → Game γ) :
    (d i T : ℕ) → d < T → Strategy (periodG G i T) → Game γ
  | _, _, 0, h, _ => absurd h (Nat.not_lt_zero _)
  | 0, i, _ + 1, _, s => K i (Strategy.ofBind _ (G i) s)
  | d + 1, i, T + 1, h, s => runBind (fun _ => periodG G (i + 1) T)
      (fun _ s' => compTo G K d (i + 1) T (Nat.lt_of_succ_lt_succ h) s') (G i) s

/-- The strategy `compTo` is played by: `s` up to sampling block `i + d`, then `KS`. -/
def compStrat (G : (i : ℕ) → Game (β i)) (K : (j : ℕ) → Strategy (G j) → Game γ)
    (KS : (j : ℕ) → (τ : Strategy (G j)) → Strategy (K j τ)) :
    (d i T : ℕ) → (h : d < T) → (s : Strategy (periodG G i T)) → Strategy (compTo G K d i T h s)
  | _, _, 0, h, _ => absurd h (Nat.not_lt_zero _)
  | 0, i, _ + 1, _, s => KS i (Strategy.ofBind _ (G i) s)
  | d + 1, i, T + 1, h, s => runBindStrat (fun _ => periodG G (i + 1) T)
      (fun _ s' => compTo G K d (i + 1) T (Nat.lt_of_succ_lt_succ h) s')
      (fun _ s' => compStrat G K KS d (i + 1) T (Nat.lt_of_succ_lt_succ h) s') (G i) s

/-- **Every sampling block's strategy satisfies `good`**: sampling block `i`'s strategy, and on every run, every later
sampling block's. -/
def PeriodGood (G : (i : ℕ) → Game (β i)) (good : (i : ℕ) → Strategy (G i) → Prop) :
    (i T : ℕ) → Strategy (periodG G i T) → Prop
  | _, 0, _ => True
  | i, T + 1, s => good i (Strategy.ofBind _ (G i) s) ∧
      AtEnds (fun _ => periodG G (i + 1) T) (fun _ s' => PeriodGood G good (i + 1) T s') (G i) s

/-- `Q a s'` holds at some end `a` of `g` played by `s`, with `s'` the strategy `s` leaves for `f a`. -/
def AtSomeEnd (f : α → Game δ) (Q : (a : α) → Strategy (f a) → Prop) :
    (g : Game α) → Strategy (g.bind f) → Prop
  | .ret a, s => Q a s
  | .send _ kk, s => AtSomeEnd f Q (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => ∃ c : C, AtSomeEnd f Q (kk c) (s c)

/-- **The strategies a period plays sampling block `j` with**: `τ` is the strategy sampling block `j` is played with on
some run of the period's sampling blocks `i, …, i + T − 1` at `s`. A hypothesis at every such `τ` is `PeriodGood`'s,
stated flat. -/
def PeriodPlays (G : (i : ℕ) → Game (β i)) :
    (i T : ℕ) → Strategy (periodG G i T) → (j : ℕ) → Strategy (G j) → Prop
  | _, 0, _, _, _ => False
  | i, T + 1, s, j, τ => (i = j ∧ HEq (Strategy.ofBind _ (G i) s) τ) ∨
      AtSomeEnd (fun _ => periodG G (i + 1) T) (fun _ s' => PeriodPlays G (i + 1) T s' j τ) (G i) s

end FlockSoundness.Game
