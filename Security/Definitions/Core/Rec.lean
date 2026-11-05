module

public import Definitions.Core.Compose
public import Definitions.Core.Hm96

/-!
# A recursion: a proof system's sessions after a committed prefix, and its zero knowledge

The objects `RecZK` (`Definitions/Core/Guarantees/Rec.lean`) is stated over. A recursion `Rec.system O G` runs the
compiled prover `G` and then, one at a time, sessions of an outer proof system `O` (`ZkSystem`) at statements built
from the history before each.

* **The prefix** (`Rec.Compiled`): `kI x` inner rounds, each one hm96 commitment string, then one stage round of
  `nC x` commitment strings (`stg`). Its tape is the inner prover's coins `Rnd x` and a salt per string. How many
  strings each round sends is a function of the statement, so it is public.
* **The sessions**: session `j < nS x` runs `O`'s prover for `n x j` rounds at the statement `xs x j u` with the
  witness `ws j w x t₀ u`, read from the history `u` before it and the prefix's tape `t₀` (`Rec.Compiled.lay`). Each
  runs on its own slot of the tape, on coins of one type `NB x × ΩB x` for every session, through a bijection with the
  coins of `O` at its statement (`Rec.slotP`).
* **The public stop** (`stop`): once it holds at a history within the prefix, the prover sends the empty message, and
  every session after it is silent.
* **The simulator** (`Rec.sim`): zero messages under fresh salts for the prefix's strings (`Rec.Compiled.S₀`), cut where
  the stop fires, then `O`'s simulator on each session's statement after the history it simulated, where the
  continuation test `go` holds at it.

What the recursion rests on is stated here as conditions on `G` and `O`: zero knowledge of `O` (`ZkSystem.ZK`), the
prefix's strings are hm96 commitments (`Rec.Compiled.Commits`), each session's shape is public (`Rec.Compiled.ByShape`),
each session's witness is in `O`'s relation unless the stop fired (`Rec.Compiled.WitnessIn`), and the continuation test
is `goesOn` at every real run (`Rec.Compiled.GoesOnAt`).
-/

@[expose] public section

namespace Verity

open Claims Claims.ZKMal
open scoped NNReal

/-- **A zero-knowledge proof system** on statement-indexed tapes: `k x` rounds at `x`, the prover `P` on a nonce `N x`
and a rest `Ω x`, the relation `R`, the simulator `S`, and the bound's terms at the hiding parameter `δ₁`. -/
structure ZkSystem (W : Type) (N Ω : Msg → Type) where
  k : Msg → ℕ
  P : W → (x : Msg) → Prover (N x × Ω x)
  R : Rel W
  S : Simulator
  εzk : ℝ≥0 → Msg → ℝ
  γzk : Msg → ℝ

/-- **`O` is zero knowledge** at `δ₁` against a malicious verifier with a black-box simulator, on any auxiliary tape
(`ZKmalRx`). -/
def ZkSystem.ZK {W : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (O : ZkSystem W N Ω) (δ₁ : ℝ≥0) : Prop :=
  ZKmalRx N Ω H O.k O.P O.R O.S (O.εzk δ₁) O.γzk

/-- **A compiled recursion's prefix and sessions.** The prefix: `kI x` inner rounds, each the string
`str (d + mask y, H (saltMsg y))` of a salt `y`, then a stage round `stg x h s` of `nC x` such strings; the inner
prover `P₀` on its coins `Rnd x` and the salts. The sessions: `nS x` of them, session `j` of `n x j` rounds at the
statement `xs x j u` with the witness `ws j w x t₀ u`, on coins `NB x × ΩB x`, its simulator's coins `CS x j` and its
bound's terms `εS δ₁ x j` and `γS x`. The public stop is `stop x` of the history. -/
structure Rec.Compiled (W WS Sa Gp : Type) where
  kI : Msg → ℕ
  nC : Msg → ℕ
  mask : Sa → Gp
  saltMsg : Sa → Msg
  str : Gp × Msg → Msg
  stg : (x : Msg) → List (Msg × Msg) → (Fin (nC x) → Msg) → Msg
  Rnd : Msg → Type
  [finRnd : ∀ x, Fintype (Rnd x)]
  [neRnd : ∀ x, Nonempty (Rnd x)]
  P₀ : W → (x : Msg) → Prover ((Rnd x × (Fin (kI x) → Sa)) × (Fin (nC x) → Sa))
  R : Rel W
  stop : Msg → List (Msg × Msg) → Bool
  nS : Msg → ℕ
  n : Msg → ℕ → ℕ
  xs : Msg → ℕ → List (Msg × Msg) → Msg
  ws : ℕ → W → (x : Msg) → (Rnd x × (Fin (kI x) → Sa)) × (Fin (nC x) → Sa) → List (Msg × Msg) → WS
  NB : Msg → Type
  ΩB : Msg → Type
  [finNB : ∀ x, Fintype (NB x)]
  [neNB : ∀ x, Nonempty (NB x)]
  [finΩB : ∀ x, Fintype (ΩB x)]
  [neΩB : ∀ x, Nonempty (ΩB x)]
  CS : Msg → ℕ → Type
  [finCS : ∀ x j, Fintype (CS x j)]
  [neCS : ∀ x j, Nonempty (CS x j)]
  εS : ℝ≥0 → Msg → ℕ → ℝ
  γS : Msg → ℝ

attribute [instance] Rec.Compiled.finRnd Rec.Compiled.neRnd Rec.Compiled.finNB Rec.Compiled.neNB
  Rec.Compiled.finΩB Rec.Compiled.neΩB Rec.Compiled.finCS Rec.Compiled.neCS

/-- `n` oracle queries `q 0, …, q (n - 1)` in turn, then `k` of their answers. -/
def Rec.queries {α : Type} : (n : ℕ) → (Fin n → Msg) → ((Fin n → Msg) → OC α) → OC α
  | 0, _, k => k Fin.elim0
  | n + 1, q, k => .query (q 0) fun a => Rec.queries n (fun i => q i.succ) fun r => k (Fin.cons a r)

open Classical in
/-- **A session on a slot's coins**: `O`'s prover at the statement it is asked at, on coins `NB × ΩB` through a
bijection with `O`'s coins there, where there is one; else silent. -/
noncomputable def Rec.slotP {WS : Type} {NO ΩO : Msg → Type} (O : ZkSystem WS NO ΩO) (NB ΩB : Type) :
    WS → Prover (NB × ΩB) :=
  fun w => ⟨fun xs h ω => if e : Nonempty ((NB ≃ NO xs) × (ΩB ≃ ΩO xs)) then
    (O.P w xs).next xs h (e.some.1 ω.1, e.some.2 ω.2) else .ret []⟩

namespace Rec.Compiled

variable {W WS Sa Gp : Type} (G : Rec.Compiled W WS Sa Gp)

/-- The prefix's tape at `x`: the inner prover's coins and a salt per inner round as the nonce, the stage round's
salts as the rest. -/
def T₀ [Fintype Sa] [Nonempty Sa] (x : Msg) : SeqTape where
  N := G.Rnd x × (Fin (G.kI x) → Sa)
  Ω := Fin (G.nC x) → Sa

/-- Where session `j` starts: after the prefix's `kI x + 1` rounds and the sessions before `j`. -/
def start (x : Msg) : ℕ → ℕ
  | 0 => G.kI x + 1
  | j + 1 => start x j + G.n x j

/-- The recursion as a run in parts: the prefix, then the sessions. -/
def lay [Fintype Sa] [Nonempty Sa] (x : Msg) : Layout W WS ((G.T₀ x).N × (G.T₀ x).Ω) where
  k₀ x' := G.kI x' + 1
  n := G.n
  xs := G.xs
  ws j w t _ u := G.ws j w x t u

/-- **The prefix's zero messages**: at an inner round, the string of the zero message under that round's salt; at the
stage round, the stage message of zero strings under its salts. -/
def zeroMsg (x : Msg) (c : (Fin (G.kI x) → Sa) × (Fin (G.nC x) → Sa)) (h : List (Msg × Msg)) : OC Msg :=
  if hh : h.length < G.kI x then
    .query (G.saltMsg (c.1 ⟨h.length, hh⟩)) fun s => .ret (G.str (G.mask (c.1 ⟨h.length, hh⟩), s))
  else Rec.queries (G.nC x) (fun j => G.saltMsg (c.2 j)) fun s =>
    .ret (G.stg x h fun j => G.str (G.mask (c.2 j), s j))

/-- **The prefix's simulator**: the zero messages under fresh salts, against the verifier, for the prefix's rounds. -/
def S₀ [Fintype Sa] [Nonempty Sa] : Simulator where
  C x := (Fin (G.kI x) → Sa) × (Fin (G.nC x) → Sa)
  prog x c := spRun (G.zeroMsg x c) (G.kI x + 1) []

/-- **The prefix commits**: for `(x, w) ∈ R`, the prover's inner round is the string of some message `dI` under that
round's salt, and its stage round is the stage message of strings of some messages `dS` under the stage salts, where
`dI` reads no salt and `dS` no stage salt. -/
def Commits [AddCommGroup Gp] (H : Msg → Msg) : Prop :=
  ∀ x w, G.R x w → ∃ (dI : G.Rnd x → List (Msg × Msg) → Gp)
    (dS : G.Rnd x × (Fin (G.kI x) → Sa) → List (Msg × Msg) → Fin (G.nC x) → Gp),
    ∀ (t : (G.Rnd x × (Fin (G.kI x) → Sa)) × (Fin (G.nC x) → Sa)) (h : List (Msg × Msg)),
      (∀ hh : h.length < G.kI x, (((G.P₀ w x).next x h t).eval H).1 =
        G.str (dI t.1.1 h + G.mask (t.1.2 ⟨h.length, hh⟩), H (G.saltMsg (t.1.2 ⟨h.length, hh⟩)))) ∧
      (h.length = G.kI x → (((G.P₀ w x).next x h t).eval H).1 =
        G.stg x h fun j => G.str (dS t.1 h j + G.mask (t.2 j), H (G.saltMsg (t.2 j))))

/-- **Each session's shape is public**: at every history `u` of its start's length where the stop has not fired,
session `j`'s statement has `O`'s rounds `n x j`, `O`'s bound's terms at most `εS δ₁ x j` and `γS x`, and simulator
coins and prover coins in bijection with the slot's. -/
def ByShape {NO ΩO : Msg → Type} (O : ZkSystem WS NO ΩO) (δ₁ : ℝ≥0) : Prop :=
  (∀ x, 0 ≤ G.γS x) ∧ ∀ x, ∀ j < G.nS x, 0 ≤ G.εS δ₁ x j ∧ ∀ u : List (Msg × Msg), u.length = G.start x j →
    fired (G.stop x) (G.kI x + 1) u = false →
      O.k (G.xs x j u) = G.n x j ∧ O.εzk δ₁ (G.xs x j u) ≤ G.εS δ₁ x j ∧ O.γzk (G.xs x j u) ≤ G.γS x ∧
        Nonempty (G.CS x j ≃ O.S.C (G.xs x j u)) ∧
        Nonempty ((G.NB x ≃ NO (G.xs x j u)) × (G.ΩB x ≃ ΩO (G.xs x j u)))

/-- **Each session's witness is in `O`'s relation unless the stop fired**: for `(x, w) ∈ R`, every prefix tape `t₀` and
every history `u` of session `j`'s start's length whose prefix rounds are the honest prover's on `t₀` (whatever the
verifier answered), where the stop has not fired, `(xs x j u, ws j w x t₀ u) ∈ O.R`. -/
def WitnessIn {NO ΩO : Msg → Type} (H : Msg → Msg) (O : ZkSystem WS NO ΩO) : Prop :=
  ∀ x, ∀ j < G.nS x, ∀ w, G.R x w → ∀ (t₀ : (G.Rnd x × (Fin (G.kI x) → Sa)) × (Fin (G.nC x) → Sa))
    (u : List (Msg × Msg)), u.length = G.start x j →
    (∀ i (hi : i < u.length), i < G.kI x + 1 → u[i].1 = (((G.P₀ w x).next x (u.take i) t₀).eval H).1) →
    fired (G.stop x) (G.kI x + 1) u = false → O.R (G.xs x j u) (G.ws j w x t₀ u)

/-- **The continuation test is `goesOn` at every real run**: at each session `j < nS x`, `go j x` agrees with
`goesOn (start x j)` at every run of `start x j` rounds against any verifier, so the simulator may go on at a simulated
history by any test the real runs can't tell from `goesOn`: `goesOn` itself, or that the last answer is a message. -/
def GoesOnAt (go : ℕ → Msg → List (Msg × Out) → Bool) : Prop :=
  ∀ x, ∀ j < G.nS x, ∀ (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg),
    go j x (interact ask pm (G.start x j) []) = goesOn (G.start x j) (interact ask pm (G.start x j) [])

end Rec.Compiled

section System

variable {W WS Sa Gp : Type} [Fintype Sa] [Nonempty Sa] {NO ΩO : Msg → Type}

/-- **The recursion's prover**: the prefix, then the sessions on their slots, with the stop, on the nested tape of
`nS x + 1` parts read flat. -/
noncomputable def Rec.prover (O : ZkSystem WS NO ΩO) (G : Rec.Compiled W WS Sa Gp) (w : W) (x : Msg) :
    Prover ((G.T₀ x).at 0 (G.NB x) (G.ΩB x) (G.nS x)) :=
  (stopP ((G.lay x).prover (fun w => G.P₀ w x) (Rec.slotP O (G.NB x) (G.ΩB x)) (G.nS x) w) G.stop
    (G.lay x).k₀).comap (Layout.flatten (G.T₀ x) 0 (G.NB x) (G.ΩB x) (G.nS x))

/-- The recursion's simulator at `x`: the prefix's, cut where the stop fires, then each session's where `go` holds at
the history it simulated, silent once the stop has fired. -/
noncomputable def Rec.simAt (O : ZkSystem WS NO ΩO) (G : Rec.Compiled W WS Sa Gp)
    (go : ℕ → Msg → List (Msg × Out) → Bool) (x : Msg) : Simulator :=
  iterSimG (G.T₀ x) 0 (G.NB x) (G.ΩB x) (Layout.seq (G.T₀ x) 0 (G.NB x) (G.ΩB x) ((G.lay x).stopped G.stop))
    (censorSim G.S₀ G.stop (G.lay x).k₀) (fun j x' => G.CS x' j)
    (stopProgB G.stop (G.lay x).k₀ (G.lay x).n fun j x' u => progOf O.S (G.CS x' j) (G.xs x' j u)) go (G.nS x)

/-- **The recursion's simulator** with the continuation test `go`: at each statement, `Rec.simAt`'s there. -/
noncomputable def Rec.sim (O : ZkSystem WS NO ΩO) (G : Rec.Compiled W WS Sa Gp)
    (go : ℕ → Msg → List (Msg × Out) → Bool) : Simulator where
  C x := (Rec.simAt O G go x).C x
  prog x := (Rec.simAt O G go x).prog x

/-- **The recursion as a proof system**: `start x (nS x)` rounds, its prover and simulator (with the continuation test
`go`), `G`'s relation, and the bound `2 (kI + nC) δ₁ + Σ_j εS δ₁ x j` and `nS x · γS x`. -/
noncomputable def Rec.system (O : ZkSystem WS NO ΩO) (G : Rec.Compiled W WS Sa Gp)
    (go : ℕ → Msg → List (Msg × Out) → Bool) :
    ZkSystem W (fun x => ((G.T₀ x).iter 0 (G.NB x) (G.ΩB x) (G.nS x)).N)
      (fun x => ((G.T₀ x).iter 0 (G.NB x) (G.ΩB x) (G.nS x)).Ω) where
  k x := G.start x (G.nS x)
  P := Rec.prover O G
  R := G.R
  S := Rec.sim O G go
  εzk δ₁ x := 2 * ((G.kI x : ℝ) + G.nC x) * δ₁ + ∑ j ∈ Finset.range (G.nS x), G.εS δ₁ x j
  γzk x := G.nS x * G.γS x

end System

end Verity
