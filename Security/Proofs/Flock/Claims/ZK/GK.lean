import Proofs.Flock.Claims.ZK.Statement

/-!
# A committed-coin protocol, its honest prover and its Goldreich–Kahan simulator

The deployed masked protocol commits every coin at `Hello` (`coin-tree/hm96-sha512/v2`) and opens them round by round;
the prover checks each opening before it uses the coin (`verity/ml/flock/live/PROTOCOL.md` §9). `Proto` is that
protocol's shape, as far as zero knowledge against a malicious verifier reads it:

* the prover's tape is a nonce (its `Hello`) and the rest; the verifier answers `Hello` with its commitment `ρ`;
* `K` rounds, each a prover message and the verifier's opening of one coin, which the prover checks (`chk`, its `H`
  calls logged) and reads (`dec`); from the first opening that fails it sends `abort`;
* the prover's last message, after the last coin.

`Proto.prover` is the honest prover; `Proto.dprover` runs it on the dummy witness (`dmsg`). `Proto.simulator` is the
Goldreich–Kahan simulator with `N` rewinds, and uses `V*` only through its black-box answers (`SP.ask`):

1. **Extraction.** Run the dummy-witness prover against `V*`. If its transcript does not open all `K` coins (`V*`
   stopped, or an opening failed the prover's check), output that transcript.
2. **Rewinds.** Otherwise, with `c` the coins it opened, up to `N` times: send the SHVZK simulator's messages at `c`
   (`sim`) from a fresh start on the same nonce, checking each opening and that it opens `c`'s coin (`facc`); output the
   first transcript that opens all of `c`. A rewind that opens another coin is a binding failure, and the simulator goes
   on to the next.

What the theorem needs of the protocol, each named: `PrefixHides` (the dummy-witness run is close to the real one until
the last coin), `SHVZK` (at every fixed coin vector, the honest prover's messages are close to `sim`'s), `Binding` (two
openings of one coin that pass `chk` and differ show a collision in `chk`'s calls).
-/

namespace Claims.ZKMal

open Claims

/-! ## Composing oracle and simulator programs -/

def ocBind {α β : Type} : OC α → (α → OC β) → OC β
  | .ret a, f => f a
  | .query q k, f => .query q fun r => ocBind (k r) f

def spBind {α β : Type} : SP α → (α → SP β) → SP β
  | .ret a, f => f a
  | .hash q k, f => .hash q fun r => spBind (k r) f
  | .ask as k, f => .ask as fun b => spBind (k b) f

/-- An oracle program run by a simulator: its `H` calls are the simulator's. -/
def ocSP {α : Type} : OC α → SP α
  | .ret a => .ret a
  | .query q k => .hash q fun r => ocSP (k r)

/-- The simulator's `stepWith`: `n` rounds of the prover's next message `nxt` against the black-box verifier, from the
history `hist`; the transcript. -/
def spRun (nxt : List (Msg × Msg) → OC Msg) : ℕ → List (Msg × Msg) → SP (List (Msg × Out))
  | 0, _ => .ret []
  | n + 1, hist => spBind (ocSP (nxt hist)) fun a => .ask (hist.map Prod.fst ++ [a]) fun b =>
      match b with
      | .msg m => spBind (spRun nxt n (hist ++ [(a, m)])) fun τ => .ret ((a, .msg m) :: τ)
      | .acc => .ret [(a, .acc)]
      | .rej => .ret [(a, .rej)]

/-! ## The committed-coin prover -/

/-- Whether every opening from position `j` on passes `acc`, checked in order up to the first that fails. -/
def allOk (acc : ℕ → Msg → OC Bool) : ℕ → List Msg → OC Bool
  | _, [] => .ret true
  | j, m :: ms => ocBind (acc j m) fun b => if b then allOk acc (j + 1) ms else .ret false

/-- **A committed-coin prover's next message**: `hello` first; then, if every opening so far passes `acc` against the
commitment `ρ` (the verifier's first answer), `gen` at the coins they open, else `abort`. -/
def strat {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen : List Coin → OC Msg) : List (Msg × Msg) → OC Msg
  | [] => .ret hello
  | (_, ρ) :: ops => ocBind (allOk (acc ρ) 0 (ops.map Prod.snd)) fun ok =>
      if ok then gen (ops.map fun p => dec p.2) else .ret abort

/-- Coin `j` of `c`, if `j < K`. -/
def coinAt {Coin : Type} {K : ℕ} (c : Fin K → Coin) (j : ℕ) : Option Coin := if h : j < K then some (c ⟨j, h⟩) else none

/-- **The forced check**: the opening passes `chk` and opens `c`'s coin. -/
def facc {Coin : Type} [DecidableEq Coin] {K : ℕ} (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin) (c : Fin K → Coin)
    (ρ : Msg) (j : ℕ) (m : Msg) : OC Bool :=
  ocBind (chk ρ j m) fun b => .ret (b && decide (coinAt c j = some (dec m)))

/-! ## What a transcript opens -/

/-- The verifier's `i`-th answer, if it is a message. -/
def resp (τ : List (Msg × Out)) (i : ℕ) : Option Msg :=
  match τ[i]? with
  | some (_, .msg m) => some m
  | _ => none

/-- `n` openings from coin `j` on are answered and pass `chk` against `ρ`, checked in order up to the first that fails. -/
def clsGo (chk : Msg → ℕ → Msg → OC Bool) (τ : List (Msg × Out)) (ρ : Msg) : ℕ → ℕ → OC Bool
  | _, 0 => .ret true
  | j, n + 1 => match resp τ (j + 1) with
    | none => .ret false
    | some m => ocBind (chk ρ j m) fun b => if b then clsGo chk τ ρ (j + 1) n else .ret false

/-- **The coins a transcript opens**: `some c` if the verifier answered `Hello` with a commitment and opened all `K`
coins, each passing the prover's check. -/
def clsOC {Coin : Type} (K : ℕ) (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin) (τ : List (Msg × Out)) :
    OC (Option (Fin K → Coin)) :=
  match resp τ 0 with
  | none => .ret none
  | some ρ => ocBind (clsGo chk τ ρ 0 K) fun ok =>
      .ret (if ok then some fun j => dec ((resp τ (j.val + 1)).getD []) else none)

/-- How a rewind ends: it opened all of `c`'s coins, or an opening passed the check with another coin (a binding
failure), or it stopped first. -/
inductive St where
  | succ
  | mism
  | abort
  deriving DecidableEq

/-- The forced walk: from coin `j` on, `n` openings, each answered, passing `chk` and opening `c`'s coin. -/
def fclsGo {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin) (c : ℕ → Option Coin)
    (τ : List (Msg × Out)) (ρ : Msg) : ℕ → ℕ → OC St
  | _, 0 => .ret .succ
  | j, n + 1 => match resp τ (j + 1) with
    | none => .ret .abort
    | some m => ocBind (chk ρ j m) fun b =>
        if b then (if c j = some (dec m) then fclsGo chk dec c τ ρ (j + 1) n else .ret .mism) else .ret .abort

/-- **How a rewind at `c` ended.** -/
def fclsOC {Coin : Type} [DecidableEq Coin] (K : ℕ) (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin)
    (c : Fin K → Coin) (τ : List (Msg × Out)) : OC St :=
  match resp τ 0 with
  | none => .ret .abort
  | some ρ => fclsGo chk dec (coinAt c) τ ρ 0 K

/-! ## The protocol -/

/-- **A committed-coin protocol**, as its honest prover runs it and as its simulator reads it: the coins and how an
opening is checked and read; the prover's nonce (its `Hello`) and the rest of its tape; the honest prover's message
after the coins `cs` (`msg`), the same on the dummy witness (`dmsg`), and the SHVZK simulator's messages at a fixed coin
vector (`sim`), one per round, the last after the last coin. -/
structure Proto (W : Type) where
  Coin : Type
  [decCoin : DecidableEq Coin]
  Nonce : Type
  [finNonce : Fintype Nonce]
  [neNonce : Nonempty Nonce]
  Rand : Type
  [finRand : Fintype Rand]
  [neRand : Nonempty Rand]
  SimRand : Msg → Type
  [finSim : ∀ x, Fintype (SimRand x)]
  [neSim : ∀ x, Nonempty (SimRand x)]
  K : Msg → ℕ
  hello : Msg → Nonce → Msg
  abort : Msg
  dec : Msg → Coin
  chk : Msg → Nonce → Msg → ℕ → Msg → OC Bool
  msg : Msg → W → Nonce → Rand → List Coin → OC Msg
  dmsg : Msg → Nonce → Rand → List Coin → OC Msg
  sim : (x : Msg) → Nonce → (Fin (K x) → Coin) → SimRand x → OC (List Msg)

attribute [instance] Proto.decCoin Proto.finNonce Proto.neNonce Proto.finRand Proto.neRand Proto.finSim Proto.neSim

namespace Proto

variable {W : Type}

/-- The protocol's rounds: `Hello`, the `K` coin rounds, and the last message. -/
def k (G : Proto W) (x : Msg) : ℕ := G.K x + 2

/-- **The honest prover** `P(w)`. -/
def prover (G : Proto W) (w : W) : Prover (G.Nonce × G.Rand) :=
  ⟨fun x hist ω => strat (G.hello x ω.1) G.abort G.dec (G.chk x ω.1) (G.msg x w ω.1 ω.2) hist⟩

/-- The honest prover on the dummy witness. -/
def dprover (G : Proto W) : Prover (G.Nonce × G.Rand) :=
  ⟨fun x hist ω => strat (G.hello x ω.1) G.abort G.dec (G.chk x ω.1) (G.dmsg x ω.1 ω.2) hist⟩

/-- The forced prover at `c` sending the messages `v`: `v`'s `i`-th message after `i` openings of `c`'s coins. -/
def forced (G : Proto W) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (v : List Msg) :
    List (Msg × Msg) → OC Msg :=
  strat (G.hello x ν) G.abort G.dec (facc (G.chk x ν) G.dec c) fun cs => .ret (v.getD cs.length G.abort)

/-- The coins a transcript opens. -/
def cls (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) :
    Option (Fin (G.K x) → G.Coin) :=
  ((clsOC (G.K x) (G.chk x ν) G.dec τ).eval H).1

/-- What the extraction run passes on: its transcript if it did not open every coin, else the coins. -/
def view0 (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) :
    List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin) :=
  match G.cls H x ν τ with
  | none => .inl τ
  | some c => .inr c

/-- The honest prover's messages at the fixed coin vector `c`: after `j` coins, `msg` at `c`'s first `j`. -/
def realVec (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand) (c : Fin (G.K x) → G.Coin) :
    List Msg :=
  (List.range (G.K x + 1)).map fun j => ((G.msg x w ν r ((List.ofFn c).take j)).eval H).1

/-- One rewind at the extracted coins `c`: the SHVZK simulator's messages at `c` sent by the forced prover; the rewind's
status and transcript. -/
def rewindSP (G : Proto W) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (s : G.SimRand x) :
    SP (St × List (Msg × Out)) :=
  spBind (ocSP (G.sim x ν c s)) fun v =>
  spBind (spRun (G.forced x ν c v) (G.k x) []) fun τ =>
  spBind (ocSP (fclsOC (G.K x) (G.chk x ν) G.dec c τ)) fun st => .ret (st, τ)

/-- The rewinds in order, each on its own SHVZK tape, up to the first that opens all of `c`'s coins; the empty
transcript if none does. -/
def rewindsSP (G : Proto W) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) :
    (N : ℕ) → (Fin N → G.SimRand x) → SP (List (Msg × Out))
  | 0, _ => .ret []
  | N + 1, s => spBind (G.rewindSP x ν c (s 0)) fun r =>
      if r.1 = .succ then .ret r.2 else G.rewindsSP x ν c N fun i => s i.succ

/-- **The Goldreich–Kahan simulator with `N` rewinds** (`S_GK`): its coins are the dummy run's tape (nonce and the rest)
and one SHVZK tape per rewind. It uses `V*` only through `SP.ask`. -/
def simulator (G : Proto W) (N : ℕ) : Simulator where
  C x := G.Nonce × G.Rand × (Fin N → G.SimRand x)
  prog x c :=
    spBind (spRun (fun hist => G.dprover.next x hist (c.1, c.2.1)) (G.k x) []) fun τ₀ =>
    spBind (ocSP (clsOC (G.K x) (G.chk x c.1) G.dec τ₀)) fun o =>
    match o with
    | none => .ret τ₀
    | some cs => G.rewindsSP x c.1 cs N c.2.2

/-! ## What the theorem needs of the protocol -/

/-- **(HA) The dummy-witness run hides the witness until the last coin**: at every nonce and against every verifier of
the protocol's rounds, what the extraction run passes on (its transcript if it stops early, else the coins) is within
`δ_A(x)` for the witness and for the dummy, on every event. -/
def PrefixHides (G : Proto W) (H : Msg → Msg) (R : Rel W) (δA : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (ν : G.Nonce) (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = G.k x →
    ∀ E : (List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin)) → Prop,
    |Pr (fun r : G.Rand => E (G.view0 H x ν (runStar H (G.prover w) Vs x (ν, r) θ).msgs)) -
      Pr (fun r : G.Rand => E (G.view0 H x ν (runStar H G.dprover Vs x (ν, r) θ).msgs))| ≤ δA x

/-- **(HB) Special honest-verifier zero knowledge at every fixed coin vector**: at every nonce and coin vector `c`, the
honest prover's messages at `c` and the simulator's are within `δ_B(x)` on every event. -/
def SHVZK (G : Proto W) (H : Msg → Msg) (R : Rel W) (δB : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (E : List Msg → Prop),
    |Pr (fun r : G.Rand => E (G.realVec H x w ν r c)) - Pr (fun s : G.SimRand x => E ((G.sim x ν c s).eval H).1)|
      ≤ δB x

/-- **(HC) The opening check binds, in its own calls**: two openings of one coin that pass `chk` and read as different
coins show a collision among the two checks' `H` calls. -/
def Binding (G : Proto W) (H : Msg → Msg) : Prop :=
  ∀ x ν ρ j m m', ((G.chk x ν ρ j m).eval H).1 = true → ((G.chk x ν ρ j m').eval H).1 = true → G.dec m ≠ G.dec m' →
    findColl (((G.chk x ν ρ j m).eval H).2 ++ ((G.chk x ν ρ j m').eval H).2) ≠ none

/-- The calls the simulator's cost reads: at most `cChk` per opening check, `cMsg` per dummy-witness message, and
`cSim` per SHVZK simulation. -/
def CostLe (G : Proto W) (H : Msg → Msg) (cChk cMsg cSim : Msg → ℕ) : Prop :=
  (∀ x ν ρ j m, ((G.chk x ν ρ j m).eval H).2.length ≤ cChk x) ∧
    (∀ x ν r cs, ((G.dmsg x ν r cs).eval H).2.length ≤ cMsg x) ∧
    ∀ x ν c s, ((G.sim x ν c s).eval H).2.length ≤ cSim x

/-- The simulator's bound: `N + 1` runs of `k` rounds, each round one question to `V*` and the prover's checks of every
opening so far and its message, with the checks of a run's `K` openings and a simulation per run. -/
def simBound (G : Proto W) (N : ℕ) (cChk cMsg cSim : Msg → ℕ) (x : Msg) : ℕ :=
  (N + 1) * (cSim x + G.K x * cChk x + G.k x * (1 + G.k x * cChk x + cMsg x))

end Proto

end Claims.ZKMal
