module

public import Definitions.Claims.RecursionGeneric.Proto
public import Definitions.Claims.RecursionGeneric.Plans
public import Definitions.Claims.Sampling

/-!
# The composition: a compiled verifier with any outer extraction

`Comp Coin` is what the recursion's knowledge soundness reads of a compiled verifier and the outer proof system: the
compiled verifier `V`; the inner statement it carries (`xI`) and whether a statement parses into the relation's form
(`parses`); the outer extraction `E_O` over `V`'s runs with its witnesses `WO`, the relation they are extracted for
(`RO`) and its errors `κ_O`, `γ_O`; the inner messages a rerun's witnesses give (`msgs`); the SHA-512 evaluations a finder
searches at a rerun (`calls`: those the witnesses record and those the finder evaluates itself, the inner setup's and the
inner verifier's on the witnesses' messages) and the compiled verifier's own at a run's staged statements (`own`).

Nothing here fixes the outer proof system: `E_O` is any extractor meeting `Claims.KS` at `RO`. An outer argument embedded
session by session (`Definitions.Claims.RecursionGeneric.Outer`, `Compiled.toComp`) is one; Flock's `E_O`
(`Claims.Recursion.EO`) at today's compiled record is another (`Proofs.Flock.RecGeneric.AtFlock`).

`E*` (`Comp.E`) is the inner protocol's `E^rep` at the inner statement, each rerun served by `E_O` at that rerun, the
inner view read off `E_O`'s witnesses (`Comp.decProg`) as the run of `Open(Π, S)` on the messages they give. `E*` is a
rerun program that reads no `H`: the inner view's run is an oracle program (`Comp.replayOC`) run through `ofOC`, so
each of its SHA-512 calls is a `.hash` node, which `RP.eval` charges. Its knowledge error is the inner protocol's,
`N_I = 1 + m_Π·k_Π` times `E_O`'s, and the inner terms' finders run on the composite's reruns (`Comp.innerModel`,
`Comp.verModel`), each charged the rerun's calls with the inner view's (`Comp.rerunCost`), plus `E_O`'s witnesses
outside `R_O` at each rerun the terms read (`Comp.PrOut`, `Comp.reads`).

`Comp.Contract H K IV x` is what the compiled verifier must give at the statement `x`, each a code fact proved per
compiler (and, for `forks` and `accepts`, per circuit V* proves, where some run accepts `x`); `Comp.OuterKSAt` is
`E_O`'s knowledge soundness at `x`. The composite's bound at `x` reads both at `x` alone.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **A compiled verifier with its outer extraction**: see the module docstring. -/
structure Comp (Coin : Type) where
  V : Verifier Coin
  xI : Msg → Msg
  parses : Msg → Prop
  WO : Type
  RO : Rel WO
  EO : Extractor Coin WO V
  msgs : Msg → WO → List Msg
  calls : (Msg → Msg) → (x : Msg) → (Fin (V.k x) → Coin) → WO → List (Msg × Msg)
  own : (Msg → Msg) → Msg → List (Msg × Out) → List (Msg × Msg)
  κO : Msg → ℝ
  γO : Msg → ℝ

/-- **Some run of `V` accepts `x`**: some prover, tape and coins. Where none does, `Pr[run = acc ∧ …] = 0`. -/
def AccAt {Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (x : Msg) : Prop :=
  ∃ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin), (run H P V x ω r).out = .acc

namespace Comp

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The inner view a rerun's witnesses give**, at the inner verifier `IV`: `IV`'s run, at the compiled verifier's
first coins, of the inner messages the witnesses give. -/
def view (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) (x : Msg) (r : Fin (K.V.k x) → Coin) (w : K.WO) :
    Seen :=
  (run H (replay (K.msgs x w)) IV (K.xI x) () fun i => coinSeq r i.val).seen

/-- The inner view at a rerun, read off `E_O`'s output; a rejected view where it gave up. -/
def dec (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) (x : Msg) (r : Fin (K.V.k x) → Coin) :
    Option K.WO → Seen
  | some w => K.view IV H x r w
  | none => ⟨[], .rej⟩

/-- **`IV`'s run on the prover messages `as`, as an oracle program**: `n` rounds after the prover messages `sent` and
the messages `ms`, each round putting `as`'s next to `IV`'s next step at the coins `cs`, as `run` does against
`replay as`; a verifier that has not decided when the rounds run out rejects. Its queries are `IV`'s calls. -/
def replayOC (IV : Verifier Coin) (xI : Msg) (as : List Msg) (cs : ℕ → Coin) :
    ℕ → List Msg → List (Msg × Out) → OC Seen
  | 0, _, ms => .ret ⟨ms, .rej⟩
  | n + 1, sent, ms =>
    (IV.next xI (sent ++ [as.getD sent.length []]) ((List.range (sent.length + 1)).map cs)).bind fun b =>
      match b with
      | .msg m => replayOC IV xI as cs n (sent ++ [as.getD sent.length []]) (ms ++ [(as.getD sent.length [], .msg m)])
      | .acc => .ret ⟨ms ++ [(as.getD sent.length [], .acc)], .acc⟩
      | .rej => .ret ⟨ms ++ [(as.getD sent.length [], .rej)], .rej⟩

/-- **The inner view `E_O`'s output gives, as a rerun program**: `IV`'s run, at the compiled verifier's first coins, of
the inner messages the witnesses give (`replayOC`, `dec`'s view at every `H`), through `ofOC`, so each SHA-512 call `IV`
makes is a `.hash` node, which `RP.eval` charges; a rejected view where `E_O` gave up. -/
def decProg (K : Comp Coin) (IV : Verifier Coin) (x : Msg) (r : Fin (K.V.k x) → Coin) :
    Option K.WO → RP (Fin (K.V.k x) → Coin) Seen
  | some w => Claims.Sampling.ofOC (replayOC IV (K.xI x) (K.msgs x w)
      (coinSeq fun i : Fin (IV.k (K.xI x)) => coinSeq r i.val) (IV.k (K.xI x)) [] [])
  | none => .ret ⟨[], .rej⟩

/-- **The outer stage at a sample**: `E_O` at the sample's run coins and its own coins, then the inner view its output gives
(`decProg`). It is `outerStage` with the decoding a rerun program. -/
def stage (K : Comp Coin) (IV : Verifier Coin) (x : Msg) (d : (Fin (K.V.k x) → Coin) × K.EO.C x) :
    RP (Fin (K.V.k x) → Coin) Seen :=
  (K.EO.ext x d.1 d.2).bind (K.decProg IV x d.1)

/-- **`E*`, the composite extractor**: the inner protocol's `E^rep` at the inner statement, over reruns of the compiled
verifier's coins and `E_O`'s, each rerun served by `E_O` there and seen as the inner view of `Open(Π, S)` its witnesses
give (`stage`), opened (`Server.openView`) for the inner protocol's `opens`. It is `served` with each stage a rerun
program, and reads no `H`: every SHA-512 call it makes, `E_O`'s and the inner view's, is one of its `.hash` nodes. -/
def E (K : Comp Coin) (I : Proto Coin W) (S : Server) : Extractor Coin W K.V where
  C x := K.EO.C x × (Fin (I.waits (K.xI x)) → Fin (I.cap (K.xI x)) → (Fin (K.V.k x) → Coin) × K.EO.C x)
  ext x r₀ c := serve (K.stage (S.IVStar I.V I.ans) x)
    (erep (I.waits (K.xI x)) (I.k (K.xI x)) (I.cap (K.xI x)) (fun v => I.opens (S.openView v))
      (I.assemble (K.xI x)) (I.check (K.xI x)) (r₀, c.1) c.2)

/-- **The relation, inherited**: `x` parses into the relation's form and the inner relation holds at its inner
statement. -/
def R (K : Comp Coin) (I : Proto Coin W) : Rel W := fun x w => K.parses x ∧ I.R (K.xI x) w

/-- The inner `E^rep`'s expected reruns, `N_I = 1 + m_Π·k_Π`, each served by `E_O`. -/
def NI (K : Comp Coin) (I : Proto Coin W) (x : Msg) : ℕ := 1 + I.waits (K.xI x) * I.k (K.xI x)

/-- **The composite's knowledge error**: `κ = κ_Π(x_I) + N_I·κ_O`. -/
noncomputable def κ (K : Comp Coin) (I : Proto Coin W) (x : Msg) : ℝ := I.κ (K.xI x) + K.NI I x * K.κO x

/-- **The composite's `B` weight**: `γ = N_I²·γ_O`. -/
noncomputable def γ (K : Comp Coin) (I : Proto Coin W) (x : Msg) : ℝ := (K.NI I x : ℝ) ^ 2 * K.γO x

/-- **The terms' reads**: `Σ_i w_i·d_i` at the inner statement, the reruns the terms' plans read, each term's weighted
(`RunFinder.ReadsLe`). -/
noncomputable def reads (K : Comp Coin) (I : Proto Coin W) (x : Msg) : ℝ :=
  ∑ i, (I.terms i).w (K.xI x) * (I.terms i).d (K.xI x)

/-- **The deviation finder's weight**: `β = (N_I + Σ_i w_i·d_i)·k_Π`, at the inner statement: a guess among the inner
rounds at each rerun the inner `E^rep` makes and each rerun a term's plan reads (`reads`). -/
noncomputable def β (K : Comp Coin) (I : Proto Coin W) (x : Msg) : ℝ :=
  ((K.NI I x : ℝ) + K.reads I x) * I.V.k (K.xI x)

/-! ## The composite's reruns as runs -/

/-- `E_O`'s witnesses at a rerun (the compiled verifier's coins and `E_O`'s). -/
def wits (K : Comp Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (u : (Fin (K.V.k x) → Coin) × K.EO.C x) : Option K.WO :=
  (K.EO.eval H P x ω u.1 u.2).1

/-- **`Pr[E_O gives witnesses outside R_O]`** at one rerun, over the tape and the rerun's coins. A term's plan reads
such a rerun as its inner view accepts, while the inner prover the composite's bound is built on runs only on witnesses
in `R_O`, so the composite pays it at each rerun the terms read (`reads`). Where `E_O` replays the judged run first, it
is at most `E_O`'s failure, `κ_O + √(γ_O·Pr[B])`; where `R_O` holds at every witness, as at Flock's, it is `0`. -/
noncomputable def PrOut (K : Comp Coin) (H : Msg → Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) : ℝ :=
  Pr fun t : Ω × ((Fin (K.V.k x) → Coin) × K.EO.C x) => ∃ w, K.wits H P x t.1 t.2 = some w ∧ ¬ K.RO x w

/-- The evaluations a finder searches at a rerun: those of `calls` at its witnesses, none where `E_O` gave up. -/
def rerunCalls (K : Comp Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (u : (Fin (K.V.k x) → Coin) × K.EO.C x) : List (Msg × Msg) :=
  ((K.wits H P x ω u).map (K.calls H x u.1)).getD []

/-- The compiled verifier's own evaluations in a rerun's compiled run (`own`). -/
def ownAt (K : Comp Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (u : (Fin (K.V.k x) → Coin) × K.EO.C x) : List (Msg × Msg) :=
  K.own H x (run H P K.V x ω u.1).msgs

/-- The inner verifier's calls in the inner view's run at a rerun's witnesses. -/
def innerLog (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) (x : Msg) (r : Fin (K.V.k x) → Coin) (w : K.WO) :
    List (Msg × Msg) :=
  (run H (replay (K.msgs x w)) IV (K.xI x) () fun i => coinSeq r i.val).vLog

/-- The inner verifier's calls in the inner view's run at a rerun (`innerLog` at its witnesses), none where `E_O` gave
up: what deciding whether the inner view accepts evaluates. -/
def viewCalls (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (u : (Fin (K.V.k x) → Coin) × K.EO.C x) : List (Msg × Msg) :=
  ((K.wits H P x ω u).map (K.innerLog IV H x u.1)).getD []

/-- **A rerun's SHA-512 calls**: the compiled run's, both parties', `E_O`'s, the inner view's run's (`viewCalls`, which
deciding the rerun's acceptance in `innerModel` evaluates), and one per evaluation a finder searches there
(`rerunCalls`, `ownAt`), which it evaluates itself to search them. A run finder's cost is these over the reruns it
names (`RunFinder.finder`), so its declared cost is every call its search makes. -/
def rerunCost (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (u : (Fin (K.V.k x) → Coin) × K.EO.C x) : ℕ :=
  (run H P K.V x ω u.1).pLog.length + (run H P K.V x ω u.1).vLog.length + (K.EO.eval H P x ω u.1 u.2).2.1 +
    (K.viewCalls IV H P x ω u).length + (K.rerunCalls H P x ω u).length + (K.ownAt H P x ω u).length

/-- **The composite's reruns, as the inner terms' finders read them**: each run of the model is a rerun; a fork at the
inner round `j` keeps the compiled coins before `min j k_Π`; a rerun is accepted where the inner view `E_O`'s witnesses
give accepts; its coins are the inner session's, the compiled verifier's first `k_Π`, as the inner run shows them
(`default` past them); its evaluations are `rerunCalls`. -/
def innerModel (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) :
    RunModel Coin Ω (K.V.k x) (K.EO.C x) where
  cut j := min j (IV.k (K.xI x))
  acc ω u := (K.dec IV H x u.1 (K.wits H P x ω u)).out = .acc
  coins u := coinSeq fun i : Fin (IV.k (K.xI x)) => coinSeq u.1 i.val
  log := K.rerunCalls H P x
  cost := K.rerunCost IV H P x

/-- **The composite's reruns, as `B_I` and the link read them**: as `innerModel`, but a rerun is accepted where `E_O`
gives witnesses, and its evaluations are the compiled verifier's own as well. -/
def verModel (K : Comp Coin) (IV : Verifier Coin) (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) :
    RunModel Coin Ω (K.V.k x) (K.EO.C x) where
  cut j := min j (IV.k (K.xI x))
  acc ω u := (K.wits H P x ω u).isSome = true
  coins u := coinSeq fun i : Fin (IV.k (K.xI x)) => coinSeq u.1 i.val
  log ω u := K.rerunCalls H P x ω u ++ K.ownAt H P x ω u
  cost := K.rerunCost IV H P x

/-! ## The compiler's contract, at a statement -/

/-- `E_O` gives witnesses at `x` only on accepted runs (it replays the judged run first). -/
def Replays (H : Msg → Msg) (K : Comp Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin) (c : K.EO.C x) (w : K.WO),
    (K.EO.eval H P x ω r c).1 = some w → (run H P K.V x ω r).out = .acc

/-- An accepted run of `x` shows that `x` parses into the relation's form. -/
def Refuses (H : Msg → Msg) (K : Comp Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin), (run H P K.V x ω r).out = .acc → K.parses x

/-- The inner rounds at `x` are among the compiled verifier's, its first. -/
def InnerRounds (K : Comp Coin) (IV : Verifier Coin) (x : Msg) : Prop := IV.k (K.xI x) ≤ K.V.k x

/-- On an accepted run of `x` the inner protocol takes a round and the compiled verifier at least three more. -/
def RoundsMargin (H : Msg → Msg) (K : Comp Coin) (IV : Verifier Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin), (run H P K.V x ω r).out = .acc →
    0 < IV.k (K.xI x) ∧ IV.k (K.xI x) + 3 ≤ K.V.k x

/-- **`E_O`'s witnesses on an accepted run**: `E_O` gives `w` at `(r, c)` and the run at `r` accepts. -/
def Gives (H : Msg → Msg) (K : Comp Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r : Fin (K.V.k x) → Coin)
    (c : K.EO.C x) (w : K.WO) : Prop :=
  (run H P K.V x ω r).out = .acc ∧ (K.EO.eval H P x ω r c).1 = some w

/-- Every evaluation `calls` holds at given witnesses at `x` is a true one. -/
def CallsTrue (H : Msg → Msg) (K : Comp Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin) (c : K.EO.C x) (w : K.WO),
    K.Gives H P x ω r c w → ∀ e ∈ K.calls H x r w, H e.1 = e.2

/-- Every evaluation `own` holds at a run of `x` is a true one, so a collision among a rerun's `calls` and `own` is
one of `H`'s. -/
def OwnTrue (H : Msg → Msg) (K : Comp Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin),
    ∀ e ∈ K.own H x (run H P K.V x ω r).msgs, H e.1 = e.2

/-- **The inner verifier's calls are searched**: at given witnesses at `x` in the outer relation, the inner view's run's
calls are among `calls`, or two of those collide. The inner terms' finders search `calls` alone at a rerun
(`innerModel`), so the inner prover's calls are among what they search. -/
def CallsLogColl (H : Msg → Msg) (K : Comp Coin) (IV : Verifier Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin) (c : K.EO.C x) (w : K.WO),
    K.Gives H P x ω r c w → K.RO x w →
      (∀ e ∈ K.innerLog IV H x r w, e ∈ K.calls H x r w) ∨
      ∃ u v, findColl (K.calls H x r w) = some (u, v) ∧ Collision H u v

/-- **The commitments bind across forks** (from the server's `Binds` and V\*'s reading of the run's commit strings),
modulo what the inner extractor reads at the last inner round: at `x`, two accepted reruns forked at an inner round `j`
whose witnesses are in the outer relation and give inner views that differ at `j` show a collision among the two
reruns' evaluations, where `j` is before the last inner round, or the two views differ in what the inner extractor
reads of them (`rd`) or in the set of the inner verifier's calls. Two last messages read alike (as the deployed reader
reads a trailing space) are one view to `E*`, so they need no collision. A rerun whose witnesses fail the outer relation
is `E_O`'s failure, charged to `N_I·κ_O` (`Comp.κ`). -/
def ForksBind {β : Type} (H : Msg → Msg) (K : Comp Coin) (IV : Verifier Coin) (rd : Seen → β) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (j : ℕ) (ω : Ω) (r r' : Fin (K.V.k x) → Coin) (c c' : K.EO.C x) (w w' : K.WO),
    j < IV.k (K.xI x) → K.Gives H P x ω r c w → K.Gives H P x ω (fork j r r') c' w' → K.RO x w → K.RO x w' →
    ((K.view IV H x r w).msgs.map Prod.fst).getD j [] ≠
      ((K.view IV H x (fork j r r') w').msgs.map Prod.fst).getD j [] →
    (j + 1 < IV.k (K.xI x) ∨ rd (K.view IV H x r w) ≠ rd (K.view IV H x (fork j r r') w') ∨
      (K.innerLog IV H x r w).toFinset ≠ (K.innerLog IV H x (fork j r r') w').toFinset) →
    ∃ u v, findColl (K.calls H x r w ++ K.own H x (run H P K.V x ω r).msgs ++
      (K.calls H x (fork j r r') w' ++ K.own H x (run H P K.V x ω (fork j r r')).msgs)) = some (u, v) ∧
      Collision H u v

/-- **V\* proves the inner check** (soundness of V\*'s circuits for `Open(Π, S)`): at `x`, witnesses `E_O` gives on an
accepted run, in the outer relation, give an accepting inner view, or a collision among the rerun's evaluations. -/
def InnerAcceptsColl (H : Msg → Msg) (K : Comp Coin) (IV : Verifier Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (K.V.k x) → Coin) (c : K.EO.C x) (w : K.WO),
    K.Gives H P x ω r c w → K.RO x w →
    (K.view IV H x r w).out = .acc ∨
      ∃ u v, findColl (K.calls H x r w ++ K.own H x (run H P K.V x ω r).msgs) = some (u, v) ∧ Collision H u v

/-- **The compiler's contract** at the inner verifier `IV`, what the inner extractor reads of a view (`rd`) and the
statement `x`: the code facts about the compiled verifier and `E_O` there, and V\*'s two where some run accepts `x`.
The composite's bound at `x` reads it at `x` alone, so a compiler whose V\* is shown at some statements' forms gets the
bound at those statements. -/
structure Contract {β : Type} (H : Msg → Msg) (K : Comp Coin) (IV : Verifier Coin) (rd : Seen → β) (x : Msg) :
    Prop where
  replays : K.Replays H x
  refuses : K.Refuses H x
  rounds : K.InnerRounds IV x
  margin : K.RoundsMargin H IV x
  callsTrue : K.CallsTrue H x
  ownTrue : K.OwnTrue H x
  callsLog : K.CallsLogColl H IV x
  forks : AccAt H K.V x → K.ForksBind H IV rd x
  accepts : AccAt H K.V x → K.InnerAcceptsColl H IV x

/-- **`E_O` is knowledge sound at `x`** (`Claims.KS` at one statement): every prover's
`Pr[run = acc ∧ ¬R_O(x, E_O(x))] ≤ κ_O(x) + √(γ_O(x)·Pr[B])`. -/
def OuterKSAt (H : Msg → Msg) (K : Comp Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
    PrBad H K.V K.RO K.EO x P ≤ K.κO x + Real.sqrt (K.γO x * PrB H K.V P x)

end Comp

end Claims.RecGeneric
