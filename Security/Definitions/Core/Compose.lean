module

public import Definitions.Core.Interaction

/-!
# Composing provers and simulators: sessions in turn, a run in parts, a public stop

The vocabulary a composition of zero knowledge is stated in (`Definitions/Core/Rec.lean`): how provers run in turn on
one tape, and how their simulators run in turn against one black-box verifier. It keeps the namespace `Claims.ZKMal`
its proofs are written in (`Proofs/Flock/Claims/ZK/`).

* Simulator programs in turn (`spBind`), an oracle program run by a simulator (`ocSP`), and the simulator's own run of a
  prover's next message against the verifier (`spRun`).
* **Two protocols in turn** (`Seq`): `A` for `kA x` rounds, then `B` at a statement read from `A`'s history. Its
  simulator runs `A`'s, then, where a test `go` of `A`'s transcript holds, `B`'s program after `A`'s messages
  (`Seq.simG`).
* **`n + 1` sessions in turn**: their tape (`SeqTape.iter`: a session adds a guess, its own tape and a second rest for
  the sessions before it, which `forkTape` reads) and their simulator (`iterSimG`).
* **A run in parts** (`Layout`): a first part, then sessions back to back at statements and witnesses the history before
  each gives (`Layout.start`), on a flat tape (`Layout.prover`); the nested tape read flat (`Layout.flatten`), and each
  session after the parts before it as `Seq` (`Layout.seq`). A session's simulator on a slot's coin type, through a
  bijection with its own coins (`progOf`).
* **A public stop**: a test of the history at the first part's rounds (`fired`); once it holds the prover sends the
  empty message (`stopP`). The simulator cuts its transcript where the stop fires (`stopIdx`, `censorSim`), and a session
  after the stop is at the empty statement and silent (`Layout.stopped`, `stopProgB`, `silentSP`).
-/

@[expose] public section

namespace Claims.ZKMal

open Claims

/-! ## Simulator programs -/

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

/-- A simulator program whose questions to the verifier come after `pre`. -/
def spUnder {α : Type} (pre : List Msg) : SP α → SP α
  | .ret a => .ret a
  | .hash q k => .hash q fun r => spUnder pre (k r)
  | .ask as k => .ask (pre ++ as) fun b => spUnder pre (k b)

/-! ## Two protocols in turn -/

/-- The answer is a message, not a decision. -/
def isMsg : Out → Bool
  | .msg _ => true
  | _ => false

/-- **The run goes on after `k` rounds**: `k` rounds, each answered with a message. -/
def goesOn (k : ℕ) (τ : List (Msg × Out)) : Bool := τ.length == k && τ.all fun p => isMsg p.2

/-- **Two protocols in turn**: `A` for `kA x` rounds at `x`, then `B` at the statement `xB x u` with witness
`wB w ωA x u`, where the unit `u` is read from `A`'s history (`sel`), and `B`'s witness may read `A`'s tape `ωA` (the
salts of what `A` committed, which `B` opens). -/
structure Seq (WA WB TA : Type) where
  kA : Msg → ℕ
  U : Msg → Type
  sel : (x : Msg) → List (Msg × Msg) → U x
  xB : (x : Msg) → U x → Msg
  wB : WA → TA → (x : Msg) → U x → WB

/-- **The composed simulator, continuing where `go` holds**: `A`'s, then, where `go` holds at its transcript, `B`'s
program at the unit that transcript selects (`progB`), asking `V*` after `A`'s messages. `B`'s coins are one type `CB x`
for every unit. `go` need agree with `goesOn` only at `A`'s real transcripts (`zkmalAt_seqG`). -/
def Seq.simG {WA WB TA : Type} (Q : Seq WA WB TA) (SA : Simulator) (CB : Msg → Type) [∀ x, Fintype (CB x)]
    [∀ x, Nonempty (CB x)] (progB : (x : Msg) → Q.U x → CB x → SP (List (Msg × Out)))
    (go : Msg → List (Msg × Out) → Bool) : Simulator where
  C x := SA.C x × CB x
  prog x c := spBind (SA.prog x c.1) fun τA =>
    if go x τA then
      spBind (spUnder (τA.map Prod.fst) (progB x (Q.sel x (histOf τA)) c.2)) fun τB => .ret (τA ++ τB)
    else .ret τA

/-! ## `n` sessions in turn -/

/-- **The composed tape**: the guess `j`, `A`'s and `B`'s nonces, and `A`'s first rest, with `A`'s second rest and
`B`'s rest as the rest. `A` reads its second rest when `j < a`, so `Fork`'s two runs differ in `A`, and its first when
`j = a`, so they share `A`'s run and differ in `B`. -/
def forkTape (a : ℕ) {NA ΩA NB ΩB : Type} (s : (Fin (a + 1) × NA × NB × ΩA) × (ΩA × ΩB)) :
    (NA × ΩA) × (NB × ΩB) :=
  ((s.1.2.1, if (s.1.1 : ℕ) < a then s.2.1 else s.1.2.2.2), (s.1.2.2.1, s.2.2))

/-- A tape: a nonce and the rest, each finite and inhabited. -/
structure SeqTape where
  N : Type
  Ω : Type
  [fN : Fintype N]
  [nN : Nonempty N]
  [fΩ : Fintype Ω]
  [nΩ : Nonempty Ω]

attribute [instance] SeqTape.fN SeqTape.nN SeqTape.fΩ SeqTape.nΩ

/-- The tape of `n + 1` sessions in turn: the first's is `T₀`, and session `n + 1` adds `forkTape (c + n)`'s guess, its
own `NB × ΩB`, and a second rest for the sessions before it. -/
def SeqTape.iter (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] :
    ℕ → SeqTape
  | 0 => T₀
  | n + 1 => ⟨Fin (c + n + 1) × (SeqTape.iter T₀ c NB ΩB n).N × NB × (SeqTape.iter T₀ c NB ΩB n).Ω,
      (SeqTape.iter T₀ c NB ΩB n).Ω × ΩB⟩

/-- The first `n + 1` sessions' tape as a type. -/
abbrev SeqTape.at (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (n : ℕ) : Type :=
  (T₀.iter c NB ΩB n).N × (T₀.iter c NB ΩB n).Ω

/-- The simulator of `n + 1` sessions in turn, continuing where `go` holds: `S₀`, then, where `go j` holds at the
simulated history, session `j + 1`'s program at the unit that history selects (`Seq.simG`), on coins `CB j`. -/
def iterSimG {WA WB : Type} (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] (Q : (n : ℕ) → Seq WA WB (T₀.at c NB ΩB n)) (S₀ : Simulator) (CB : ℕ → Msg → Type)
    [∀ n x, Fintype (CB n x)] [∀ n x, Nonempty (CB n x)]
    (progB : (n : ℕ) → (x : Msg) → (Q n).U x → CB n x → SP (List (Msg × Out)))
    (go : ℕ → Msg → List (Msg × Out) → Bool) : ℕ → Simulator
  | 0 => S₀
  | n + 1 => (Q n).simG (iterSimG T₀ c NB ΩB Q S₀ CB progB go n) (CB n) (progB n) (go n)

/-! ## A run in parts -/

/-- **A run in parts** at each statement: a first part of `k₀ x` rounds, then sessions back to back, session `j` of
`n x j` rounds at the statement `xs x j h`, with the witness `ws j w t₀ x h`, from the history `h` before it and the
first part's tape `t₀`. -/
structure Layout (W WS T : Type) where
  k₀ : Msg → ℕ
  n : Msg → ℕ → ℕ
  xs : Msg → ℕ → List (Msg × Msg) → Msg
  ws : ℕ → W → T → Msg → List (Msg × Msg) → WS

namespace Layout

variable {W WS T TS : Type}

/-- Where session `j` starts: after the first part and the sessions before `j`. -/
def start (L : Layout W WS T) (x : Msg) : ℕ → ℕ
  | 0 => L.k₀ x
  | j + 1 => L.start x j + L.n x j

/-- **The parts on a flat tape**: the first part's prover on its tape, then session `j` on slot `j`'s tape, at the
statement and witness the history before it gives, for the first `m` sessions. -/
def prover (L : Layout W WS T) (P₀ : W → Prover T) (PS : WS → Prover TS) : ℕ → W → Prover (T × (ℕ → TS))
  | 0 => fun w => (P₀ w).comap Prod.fst
  | j + 1 => fun w => ⟨fun x hist ω => if hist.length < L.start x j then (L.prover P₀ PS j w).next x hist ω else
      (PS (L.ws j w ω.1 x (hist.take (L.start x j)))).next (L.xs x j (hist.take (L.start x j)))
        (hist.drop (L.start x j)) (ω.2 j)⟩

/-- **The nested tape read flat**: the first part's tape, and session `j`'s at slot `j` for `j < m`; the other slots
are arbitrary. -/
noncomputable def flatten (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB]
    [Nonempty ΩB] : (m : ℕ) → T₀.at c NB ΩB m → (T₀.N × T₀.Ω) × (ℕ → NB × ΩB)
  | 0 => fun s => (s, fun _ => Classical.arbitrary _)
  | m + 1 => fun s => ((flatten T₀ c NB ΩB m (forkTape (c + m) s).1).1, fun i =>
      if i < m then (flatten T₀ c NB ΩB m (forkTape (c + m) s).1).2 i else (forkTape (c + m) s).2)

/-- **Session `j` after the parts before it**, as `Seq`: the first part and the sessions before `j` as `A`, on the
nested tape, then session `j` at the history before it, with the witness from the first part's tape `flatten` reads. -/
noncomputable def seq (T₀ : SeqTape) (c : ℕ) (NB ΩB : Type) [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (L : Layout W WS (T₀.N × T₀.Ω)) (j : ℕ) : Seq W WS (T₀.at c NB ΩB j) where
  kA x := L.start x j
  U _ := List (Msg × Msg)
  sel _ h := h
  xB x u := L.xs x j u
  wB w t x u := L.ws j w (flatten T₀ c NB ΩB j t).1 x u

end Layout

open Classical in
/-- **A simulator's program on coins of another type**: `S`'s at `xS` through a bijection, where there is one. -/
noncomputable def progOf (S : Simulator) (C : Type) (xS : Msg) (c : C) : SP (List (Msg × Out)) :=
  if h : Nonempty (C ≃ S.C xS) then S.prog xS (h.some c) else .ret []

/-! ## A public stop -/

/-- **The stop has fired** at the history `h`: `s` holds at one of `h`'s prefixes of at most `k` rounds. -/
def fired (s : List (Msg × Msg) → Bool) (k : ℕ) (h : List (Msg × Msg)) : Bool :=
  (List.range (min h.length k + 1)).any fun i => s (h.take i)

/-- **The prover with the stop**: `P`, except that it sends the empty message once `s x` has fired within the first
`k x` rounds. -/
def stopP {Ω : Type} (P : Prover Ω) (s : Msg → List (Msg × Msg) → Bool) (k : Msg → ℕ) : Prover Ω :=
  ⟨fun x h ω => if fired (s x) (k x) h then .ret [] else P.next x h ω⟩

open Classical in
/-- **Where the stop fires on the run `τ`**: the first round `i ≤ k` at whose history `s` holds, else `n`. -/
noncomputable def stopIdx (s : List (Msg × Msg) → Bool) (k n : ℕ) (τ : List (Msg × Out)) : ℕ :=
  if h : ∃ i, i ≤ k ∧ s (histOf (τ.take i)) = true then Nat.find h else n

/-- **A simulator with the stop**: `S`'s transcript at `x`, cut where the stop fires within `k x` rounds, the rounds
from there asked of the verifier with the empty message. -/
noncomputable def censorSim (S : Simulator) (s : Msg → List (Msg × Msg) → Bool) (k : Msg → ℕ) : Simulator where
  C := S.C
  prog x c := spBind (S.prog x c) fun τ =>
    spRun (fun h => .ret (cutMsg [] (stopIdx (s x) (k x) (k x) τ) τ h)) (k x) []

/-- The silent simulator of `n` rounds: the empty message every round. -/
def silentSP (n : ℕ) : SP (List (Msg × Out)) := spRun (fun _ => .ret []) n []

/-- **The parts with the stop `s`**: session `j` at a history where the stop has fired within the first part is at
the empty statement with no witness; elsewhere at its statement tagged `1 ::`, with its witness. -/
def Layout.stopped {W WS T : Type} (L : Layout W WS T) (s : Msg → List (Msg × Msg) → Bool) :
    Layout W (Option WS) T where
  k₀ := L.k₀
  n := L.n
  xs x j u := if fired (s x) (L.k₀ x) u then [] else 1 :: L.xs x j u
  ws j w t x u := if fired (s x) (L.k₀ x) u then none else some (L.ws j w t x u)

/-- The sessions' programs with the stop: silent where the stop has fired, else `progB`. -/
def stopProgB (s : Msg → List (Msg × Msg) → Bool) (k : Msg → ℕ) (n : Msg → ℕ → ℕ) {CB : ℕ → Msg → Type}
    (progB : (j : ℕ) → (x : Msg) → List (Msg × Msg) → CB j x → SP (List (Msg × Out))) :
    (j : ℕ) → (x : Msg) → List (Msg × Msg) → CB j x → SP (List (Msg × Out)) :=
  fun j x u c => if fired (s x) (k x) u then silentSP (n x j) else progB j x u c

end Claims.ZKMal
