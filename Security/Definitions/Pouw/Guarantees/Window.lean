import Definitions.Pouw.Window

/-!
# A window's ε from its calls' (M2), with a draw that depends on the registration (C1) (trusted)

C1, a draw that depends on the registration, is already expressed on main: `FlockSoundness.Audit.auditReg`
(`Proofs/Flock/Soundness/Audit/RegDraw.lean`) and the keyed window's record theorem over it,
`keyedWindowReg_extraction_audit_of_record` (`Proofs/Flock/Soundness/KeyedDraw.lean`), which takes one
`ExtractionAnalysis` of the whole window with state-dependent `ε_ks(R, τ)` and `δ_link(R, τ)`. This module's own content
is M2: what the window's `κ`, `coll(R, τ)` and `δ_link(R, τ)` are in terms of its calls', for a window of `C` calls,
registered at once, drawn at once by a law that depends on the registration, and proved call by call (`Pouw.Window`).
Stated, not proved: the order is statement, review, card, then proof, so `lean-audit.json` lists
`WindowEps` and `ServiceEscapes` under `shape.frozen` as `unstated`.

* `WindowEps`: `Pr[accept ∧ wrong(X) ∈ 𝓑] ≤ sup_{B ∈ 𝓑} escape_{∏ L_c(R)}(B) + η + max_c κ_c + coll(R, τ) +
  ∑_c δlink_c(R, τ)`, where `X` is the committed transcript (fixed at registration), `∏ L_c(R)` the calls' laws at the
  prover's registration drawn independently, `η` the draw's distance from that product on `𝓑`, `κ_c + g_c(state)`
  call `c`'s knowledge bound in a compiled table's shape, `coll(R, τ)` the calls' `g_c` summed along the window's run
  (`Pouw.Window.coll`), and `δlink_c(R, τ)` call `c`'s link term in the window at the window's state.
* `ServiceEscapes`: the deployed draw, sampled proofs' service, is within `η = 0` of its sessions' own laws on the
  copy-closed sets, so `WindowEps` reads it with each session a call.

Knowledge soundness enters at its largest call's constant: whatever the draw, the rule picks one unit, of one call,
and that call's extraction fails from the state the window brings with probability at most `κ_c + g_c(state)`. The
state terms are one window-level collision term, in expectation over the window's run. The link terms add: each bounds
the picked unit's disagreement with the window's committed transcript on the draws where the unit is its call's. So
`WindowEps` gives gap (4)'s premises (`WindowBoundOfService` on `cursor/pouw-work-bound-stmt2-c7a9`, `KnowledgeSound εks`
and `LinkSound δlink`) at `εks R τ := max_c κ_c + coll R τ` with `coll := Pouw.Window.coll`, and `δlink := ∑_c δlink_c`.

**The deployed draw** (decision 102, `verity/core/protocols/pouw/window.py`). The window's calls are the service's
sessions and its units the `N` tiles in each (`serviceLaw`): after the receipt the verifier draws one uniform
`K′`-subset, and session `σ` proves the drawn tiles of rank `σ·cap` to `σ·cap + cap − 1`. The sessions' laws `L σ` are
uniform `share σ`-subsets on coins of their own (`sessionLaw`), so `η = 0` (`ServiceEscapes`). PoUW reads `WindowEps` at
`𝓑 = harmSets Wref ε ∧ CopyClosed`, where `Wref ⟨σ, t⟩` is tile `t`'s work: a copy-closed set holding the share `ε` of
the work is every session's copy of a tile set holding the share `ε` of the window's, at least `ε·W/w_max` tiles, so
the first term is at most `∏_σ (1 − ε·W/(N·w_max))^{share σ} = (1 − ε·W/(N·w_max))^{K′}`, which is at most `δ` where
`K′ < N`: that is how `k_prime` sizes `K′`. Where `K′ = N` (`k_prime`'s answer when no `K < N` reaches `δ`) the product
law still misses, while the service's draw misses nothing (the discharge table's last row). The service doesn't fix the
order of the sessions; the window runs them in a fixed order
(`List.finRange`), and a relabelling of the sessions covers any other fixed order. The circuit path (`anchors.py`,
decision 101's keyed draw) is retired.

**Discharge table**: which deployed piece discharges each premise.
* `EscapesWithin Lw L 𝓑 0`: the service's uniform subset after the receipt (`subset:K′`, `service.py`), by
  `ServiceEscapes`. On the retired keyed draw it was A4, A5 and M1 (`keyedWindow_escape_le`).
* `CallAnalysis.cover`: each session's tile instances at the public input `t` from the verifier's own draw, the window
  root and domains from the registration (`PUBLIC_PORTS`, `public_inputs`), and the tile's place, rows and roots from
  `X`'s entries under that root (`tile_at`), so a unit's seed inputs are `UnitStmts.seedIn X ⟨σ, t⟩`.
* `CallAnalysis.KnowledgeSound (κ σ) (g σ)`: each drawn unit's compiled `--zk` session over hm96 hiding leaves
  (`analysisZC` at `ksBoundAccZ`, its joint inequality `jointZC_ksBound`), pointwise in the state, with `κ = 0` and
  `g = ⨆_j ksBoundAccZ_j/(1 − rateZ_j)` at the session's strategy, `ksBoundAccZ = ε_c⁻ + √(k·adv₀)` and
  `rateZ = N₀/(ek)` (`Pouw.Window.flockKnowledgeSound_analysisZC`, proved, read through the bridge below).
  `ksJoint_le` (`κ = 2^logLen/(e·Kr)`) is the non-ZK plain-leaf table's, which main's window doesn't run.
* `LinkSoundIn … σ (δlink σ)`: C-Flock's link finder at the window's state, over the window's run.
* `coll`: the random-oracle claim's collision bound at the sessions' counted queries (`CollCounted`, premise, open), on
  average over the oracle; `WindowEps`'s bound at each oracle reads it through `Pouw.Window.avg_counted_le` (proved).
  It holds only where the plays are one oracle program's run at each oracle, counted by that program's calls. As
  stated, over a separate play at each oracle with a free count, it is false at a true count (a play holding a
  hardcoded collision of each oracle makes 0 calls and has `coll` about 1) and vacuous at a made-up one. Gap (4)'s
  `ProverModel.sessCalls`, the count it is meant to read, isn't tied to an oracle program either. Before the card
  discharges it, it is restated over an oracle strategy, one `QTree` per prover move with the oracle only in the
  verdict, with gap (4)'s session-query tie.
* The harm family's first term: `k_prime`'s `K′`, at most `δ` where `K′ < N`. Where `K′ = N` (levels 87 to 100 at
  `(ε, δ) = (0.1%, 2^-40)` and level 87 at `(1%, 2^-128)`, the
  recheck's probe 2), the product law, one uniform pick per session, misses up to `2^-13.0` (`2^-118.8`), while the service's draw
  misses nothing. PoUW's reading there takes the service's own escape (gap (4)'s `WindowBoundOfService`, the subset
  law's miss, `0` at `K′ = N`).

**PoUW's reading: three links.** `WindowEps` bounds the committed transcript's wrong units. PoUW's reading of it needs
three more links, each stated:
* **The deployed wrong sets are copy-closed** (`WrongCopyClosed`), so `ServiceEscapes`'s `η = 0` applies to them. The
  service's units' statements are the tiles' read in every session (`UnitStmts.copies`: an instance's public inputs are
  its tile and the window's common words, never the session), and their wrong sets are copy-closed, proved
  (`Pouw.Window.wrongCopyClosed_copies`). On them `𝓑 = harmSets Wref ε ∧ CopyClosed` holds of a wrong set exactly when
  `harmSets` does (`Pouw.Window.harmSets_copyClosed_iff`).
* **The tie to what's served** (`ServedTie`, premise `δ_in`, open on main): the window accepts while its committed
  transcript isn't anchored (`Anchored`: its values at every unit's input wires are the served pass's anchors at the
  unit's position) and served (`Served`: its window entries hold the pass's `-h3` root, and its outputs at every unit
  are the words `tile_read` reads from the `-h3` leaf at the unit's position) with probability at most `δ_in(R, τ)`.
  The units' statements are the verifier's own (`Check.program`) and the served pass is fixed before the window, so
  the prover picks neither. On main no Program reads the `-h3` leaf (`Pc8TileSide` passes neither `ia` nor `jb` on, its
  side ports' regions are empty, nothing outside `window.py` calls `tile_read`, and `window_problem` is Python only), so
  `δ_in = 1`. It is discharged when the window's Program reads the `-h3` leaf in gates and `window_problem` is a
  Program: the outputs' half is that Program's opening of the `-h3` root, the inputs' half C-Flock's `AnchorsSound`
  (`Proofs/Flock/Soundness/Audit/OneStage.lean`). PoUW's event, the window accepting while its committed transcript
  isn't anchored, served and outside `𝓑`, is then at most `WindowEps`'s bound plus `δ_in`
  (`Pouw.Window.window_served_le`, proved).
* **The bridge to C-Flock** (`Proofs/Pouw/Window/FlockBridge.lean`): C-Flock's sessions read in this game
  (`Pouw.Window.ofFlock`) win with the same probabilities (`prob_ofFlock`, proved), a C-Flock `ExtractionAnalysis` of a
  session is a `CallAnalysis` of its call (`CallAnalysis.ofFlock`), and C-Flock's pointwise knowledge bound
  (`FlockKnowledgeSound`) is the call's `KnowledgeSound` (`ofFlock_knowledgeSound`, proved). The bound reads the
  session's own coins and the extractor's `k` accepted reruns, at the state the registration, the draw and the earlier
  sessions left, never the draw's coin. At main's `--zk` sessions it is proved (`flockKnowledgeSound_analysisZC`, from
  `jointZC_ksBound`), with `κ_c = 0` and `g_c = ⨆_j ksBoundAccZ_j/(1 − rateZ_j)`; `δlink_c` is C-Flock's link finder at
  the window's state.

Each unit's statement carries its position: its seed inputs (tag, salt, position, root(A_j), root(B_j)) are
`UnitStmts.seedIn X v`, the tag and salt public, and the position and roots functions of the place `v` and `X`, bound
through the opening path to the window root. So a prover can't pick a unit's position, or with it its seed, inside the
unit.

Two things are outside this statement by design:
* **Luck.** Wrong means committed outputs that aren't the unit's function of its inputs (`UnitStmts.wrong`), so a
  deficient unit whose receipt came out right by luck isn't wrong. PoUW adds luck through gap (4), TT or the TS model's
  `ε_luck`, composed with the window audit, which reads `WindowEps`'s bound.
* **Fiat–Shamir grinding** of the per-matmul seeds (decision 103) enters PoUW's `δ` through the seeded pair
  (`Accounting.Guarantees.Compose.CircuitWorkSeeded`, `CircuitCreditSeeded`: GRIND's premium), not here.
-/

namespace Pouw.Guarantees.Window

open Pouw.Window
open scoped ENNReal

/-- **M2 with C1: the window's ε from its calls'.** For a window of `C` calls, call `c` with `n c` units, a registration
`R` (the receipt), the calls' laws `L R c` at each registration and the law `Lw R` the verifier runs there, each call's
session `sess c` on its share of the draw, the window's transcripts and its units' statements `U` (each unit's seed
inputs, with its position and roots read at its place in the transcript, and what it computes), a committed transcript
that is a function of the prover's state at registration, an analysis of each call, and a family `𝓑` of wrong-unit
sets: if the draw's escapes on `𝓑` are within `η` of the calls' product law at every registration (`EscapesWithin`),
every call's extraction fails from a state `τ` of its session with probability at most `κ c + g c τ`
(`CallAnalysis.KnowledgeSound`), and every call's link term in the window is at most `δlink c (R, τ)` at the window's
state (`LinkSoundIn`), then for every prover the window accepts with its committed transcript's wrong units in `𝓑` with
probability at most the largest escape over `𝓑` of the calls' product law at the prover's registration, plus `η`, plus
the largest `κ c`, plus the window's collision term at the prover's state (`coll`), plus the sum of the
`δlink c (R, τ)`.

A unit is wrong when its committed outputs aren't its function of its inputs, so luck is not in the event: a deficient
unit whose receipt came out right isn't wrong, and gap (4) adds it. Fiat–Shamir grinding of the per-matmul seeds enters
PoUW's `δ` through the seeded pair (`CircuitWorkSeeded`, `CircuitCreditSeeded`). -/
def WindowEps : Prop :=
  ∀ {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ} (L : Reg → (c : Fin C) → Law (Fin (n c))) (Lw : Reg → Law (WUnit n))
    (η : ℝ≥0∞) (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool) (U : UnitStmts Tr n)
    (committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr) (A : (c : Fin C) → CallAnalysis U c (sess c))
    (κ : Fin C → ℝ≥0∞) (g : (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess c S R) → ℝ≥0∞)
    (δlink : Fin C → (R : Reg) → Cont Lw (windowSession sess) R → ℝ≥0∞) (𝓑 : Finset (WUnit n) → Prop),
    EscapesWithin Lw L 𝓑 η →
    (∀ c, (A c).KnowledgeSound (κ c) (g c)) →
    (∀ c, LinkSoundIn Lw sess committed A c (δlink c)) →
    ∀ σ : Strategy (auditReg Lw (windowSession sess)),
      prob (fun o => o.1 = true ∧ 𝓑 (U.wrong (committed (reg σ) (cont σ)))) (auditReg Lw (windowSession sess)) σ ≤
        (⨆ (B : Finset (WUnit n)) (_ : 𝓑 B), (Law.prod (L (reg σ))).escape B) + η + (⨆ c, κ c) +
          coll Lw sess g (reg σ) (cont σ) + ∑ c, δlink c (reg σ) (cont σ)

/-- **The service's draw is its sessions' laws on copy-closed sets.** For `N` tiles, `K ≤ N` drawn and sessions of
`cap > 0` instances, at every registration: a copy-closed set of units, every session's copy of a tile set `B₀`, escapes
the service's draw (`serviceLaw`, one uniform `K`-subset in rank-ordered sessions) with probability at most its escape
under the sessions' own uniform `share σ`-subsets drawn independently (`sessionLaw`). Both are hypergeometric: the
service's is `∏_{i<K} (N − |B₀| − i)/(N − i)`, and the product's is the same factors with `i` restarting at `0` in every
session, each at least the service's. So `WindowEps` reads the service with each session a call at `η = 0` on every
family of copy-closed sets. -/
def ServiceEscapes : Prop :=
  ∀ {Reg : Type} (N K cap : ℕ), 0 < cap → K ≤ N →
    EscapesWithin (fun _ : Reg => serviceLaw N K cap) (fun _ σ => sessionLaw N K cap σ.val) CopyClosed 0

end Pouw.Guarantees.Window
