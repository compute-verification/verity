import Mathlib.Data.Fin.VecNotation
import Proofs.Flock.Soundness.Assumptions.WindowDraws
import Proofs.Flock.Soundness.Audit.Stratified

/-!
# A window's stratified draw, restricted to a member, is the member's `subset` draw

The sampled-proofs service draws a sampling window's units once, by the window's law (`stratified`: a uniform
`k_s`-subset of each stratum `s`, the strata independent). Each member of the window, the units `[base, base + size)`,
then runs its own session, `verify --zk` on the member's own draw file: a `subset` draw over the member's `size` units,
whose units are the window's drawn units in the member's range, shifted to the member's own indices, and no session when
there are none (`benchmarks/one_stage/a5.py`'s split, `Member.sessionFile`).

* `WindowDraw`, the decomposition: the service's split of a draw file is the member's share of the draw, and under the
  window's stratified law a member inside one stratum `s` has a share of at most `k_s` units, of hypergeometric size,
  which given its size `κ` is the member's own uniform `κ`-subset (`Law.subset size κ`), independent of the draw outside
  the member. At `k_s = 1` a member's session, when it runs, draws `k = 1` unit.
* `WindowBound`, the composed bound: a bound on each member's event in its own `subset` session game, at each share
  size, bounds the window's, by the union bound over the members: at most `∑_j ∑_κ Pr[|share_j| = κ] · β_j(κ)`, which
  is `∑_j size_j / n_{s_j} · β_j(1)` at `k = 1`.
* `WindowBound_honest`: the deployed window, its draw files pinned (`FlockSoundness.Assumptions.PinnedWindowDraws`).
* `WindowBound_nontrivial`: with a member across two strata, the composed bound fails.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.Window
open scoped ENNReal

/-- **A window's stratified draw, restricted to a member, is the member's `subset` draw.** For every member `mb` of a
window of `N` units, inside the window (`base + size ≤ N`):

1. *The service's split is the share.* For every unit list, the member's local units (`Member.localUnits`, as the
   service computes them) are its share (`Member.share`) of the draw the list names. For a list without repeats, its
   session file (`Member.sessionFile`) is none exactly when the share is empty, and is otherwise a `subset` draw over
   its `size` units of `|share|` units, its local units.
2. *Restricted to the member, the stratified law is the member's `subset` law.* For every stratified law
   (`Law.stratified σ k`) whose stratum `s₀` holds all the member's units: every draw's share has at most `k s₀` units;
   it has `κ` of them with probability `C(size, κ) · C(n_{s₀} − size, k s₀ − κ) / C(n_{s₀}, k s₀)`; and given that, the
   share is the member's own uniform `κ`-subset, independent of the draw outside the member: for every function `f` of
   the draw, `f`'s average over the draws whose share has `κ` units is, over the same draws, the average of `f` at the
   draw outside the member with the member's units drawn by `Law.subset size κ`.
3. *At `k s₀ = 1`* the share has one unit with probability `size / n_{s₀}`, and at every draw of the law whose units a
   draw file lists, the member's session file, when there is one, draws `k = 1` unit. -/
def WindowDraw : Prop :=
  (∀ (N : ℕ) (mb : Member), mb.base + mb.size ≤ N →
    (∀ (units : List ℕ) (v : Fin mb.size),
      v ∈ mb.share (Finset.univ.filter fun u : Fin N => u.val ∈ units) ↔ v.val ∈ mb.localUnits units) ∧
    ∀ W : DrawFile, W.units.Nodup →
      (mb.sessionFile W = none ↔ mb.share (Finset.univ.filter fun u : Fin N => u.val ∈ W.units) = ∅) ∧
      ∀ F, mb.sessionFile W = some F → F.law = "subset" ∧ F.population = mb.size ∧
        F.k = (mb.share (Finset.univ.filter fun u : Fin N => u.val ∈ W.units)).card ∧
        F.units = mb.localUnits W.units) ∧
  ∀ (N m : ℕ) (σ : Fin N → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card) (mb : Member)
    (s₀ : Fin m), mb.base + mb.size ≤ N →
    (∀ u : Fin N, mb.base ≤ u.val → u.val < mb.base + mb.size → σ u = s₀) →
    (∀ ω, (mb.share ((Law.stratified σ k hk).draw ω)).card ≤ k s₀) ∧
    (∀ κ ≤ k s₀, prCoin (fun ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = κ) =
      ((mb.size.choose κ * ((Law.stratum σ s₀).card - mb.size).choose (k s₀ - κ) : ℕ) : ℝ≥0∞) /
        ((Law.stratum σ s₀).card.choose (k s₀) : ℝ≥0∞)) ∧
    (∀ κ (hκ : κ ≤ mb.size) (f : Finset (Fin N) → ℝ≥0∞),
      avg (fun ω => if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
        f ((Law.stratified σ k hk).draw ω) else 0) =
      avg (fun ω => if (mb.share ((Law.stratified σ k hk).draw ω)).card = κ then
        avg (fun S : (Law.subset mb.size κ hκ).Ω =>
          f (((Law.stratified σ k hk).draw ω \ mb.range) ∪ mb.unshift ((Law.subset mb.size κ hκ).draw S))) else 0)) ∧
    (k s₀ = 1 →
      prCoin (fun ω => (mb.share ((Law.stratified σ k hk).draw ω)).card = 1) =
        (mb.size : ℝ≥0∞) / ((Law.stratum σ s₀).card : ℝ≥0∞) ∧
      ∀ ω (W : DrawFile), W.units.Nodup →
        (Finset.univ.filter fun u : Fin N => u.val ∈ W.units) = (Law.stratified σ k hk).draw ω →
        ∀ F, mb.sessionFile W = some F → F.k = 1)

/-- **The window's bound from its members' session bounds.** For every stratified window law (`Law.stratified σ k`)
over `N` units, every window body `body` (what the window's sessions run on its draw, after the registration), and every
finite family `ms` of members `mem j` with events `E j` of the window's output, where each member is inside the window,
its units all in its stratum `str j`, and its event happens only when its share is nonempty, that is when its session
runs: if each member's event has probability at most `β j κ` in the member's own `subset` session game, at every share
size `κ` from `1` to `k (str j)`, against every prover strategy (the prover registers, the session draws `κ` of the
member's units by `Law.subset`, and the window's body runs with the rest of the window's draw fixed at any `ctx` outside
the member), then against every prover strategy in the window's audit (`audit`), some member's event happens with
probability at most `∑_j ∑_{κ = 1}^{k (str j)} Pr[|share_j| = κ] · β j κ`. When every member's stratum draws one unit,
that sum is `∑_j size_j / n_{str j} · β j 1`. -/
def WindowBound : Prop :=
  (∀ (α Reg ι : Type) (N m : ℕ) (σ : Fin N → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card)
    (body : Finset (Fin N) → Reg → Game α) (ms : Finset ι) (mem : ι → Member) (str : ι → Fin m)
    (E : ι → Reg × Finset (Fin N) × α → Prop) (β : ι → ℕ → ℝ≥0∞),
    (∀ j ∈ ms, (mem j).base + (mem j).size ≤ N) →
    (∀ j ∈ ms, ∀ u : Fin N, (mem j).base ≤ u.val → u.val < (mem j).base + (mem j).size → σ u = str j) →
    (∀ j ∈ ms, ∀ o, E j o → ((mem j).share o.2.1).Nonempty) →
    (∀ j ∈ ms, ∀ κ (hκ : κ ≤ (mem j).size), 1 ≤ κ → κ ≤ k (str j) →
      ∀ ctx : Finset (Fin N), Disjoint ctx (mem j).range →
      ∀ τ : Strategy (audit (Law.subset (mem j).size κ hκ) Reg fun S R => body (ctx ∪ (mem j).unshift S) R),
        prob (fun o => E j (o.1, ctx ∪ (mem j).unshift o.2.1, o.2.2))
          (audit (Law.subset (mem j).size κ hκ) Reg fun S R => body (ctx ∪ (mem j).unshift S) R) τ ≤ β j κ) →
    ∀ τ : Strategy (audit (Law.stratified σ k hk) Reg body),
      prob (fun o => ∃ j ∈ ms, E j o) (audit (Law.stratified σ k hk) Reg body) τ ≤
        ∑ j ∈ ms, ∑ κ ∈ Finset.Icc 1 (k (str j)),
          prCoin (fun ω => ((mem j).share ((Law.stratified σ k hk).draw ω)).card = κ) * β j κ) ∧
  ∀ (ι : Type) (N m : ℕ) (σ : Fin N → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (Law.stratum σ s).card)
    (ms : Finset ι) (mem : ι → Member) (str : ι → Fin m) (β : ι → ℕ → ℝ≥0∞),
    (∀ j ∈ ms, (mem j).base + (mem j).size ≤ N) →
    (∀ j ∈ ms, ∀ u : Fin N, (mem j).base ≤ u.val → u.val < (mem j).base + (mem j).size → σ u = str j) →
    (∀ j ∈ ms, k (str j) = 1) →
    ∑ j ∈ ms, ∑ κ ∈ Finset.Icc 1 (k (str j)),
        prCoin (fun ω => ((mem j).share ((Law.stratified σ k hk).draw ω)).card = κ) * β j κ =
      ∑ j ∈ ms, ((mem j).size : ℝ≥0∞) / ((Law.stratum σ (str j)).card : ℝ≥0∞) * β j 1

namespace WindowDeployed

/-- The SHA-512 digest of the deployed window's draw file (`rank0/draw.json`). -/
def drawDigest : String :=
  "ca5031c6ca161ee6e3d542e280ebf3a5ec0eacb4af56db5061d4bd04d41102949c53d5cab3d9a2f200d77a851ecabcbc14205adcfdb7e1119abae3c87eedf220"

/-- The SHA-512 digest of member 1's session draw file (`session-m1/draw.json`). -/
def m1Digest : String :=
  "20495de9fbad861572ade8c086ffbc8d5cd52c5659a805a3e46104c267f9233e6b516da7633dbda18c1565b7efd7ecdced5209d0ee7b0c63fd9a2156bcc7de95"

/-- The SHA-512 digest of member 10's session draw file (`session-m10/draw.json`). -/
def m10Digest : String :=
  "d047a561298f60ea07a72e7f7915650db0661441aa5bcd5d8967f0989f3e4ce7e4f018eb066cb1f663319ee6e3440180e33b1d80ac7f7e18290626c72eb18759"

/-- The SHA-512 digest of member 12's session draw file (`session-m12/draw.json`). -/
def m12Digest : String :=
  "bb1690fe0ceed558c068e1d2ced851113d4a46a8d9067760fde29bdbc2abcbd9b9aa110142b079fc8afa59f2070f7caba3eaf3ffcf3831992a054a2c06a37763"

/-- The deployed window's thirteen members, its summary's `members` in order (`base`, `n`): nine `T=64` split parts of
256 units and one of 192, two `T=16` split parts of 512 and 112 units, and the combine's 624. -/
def member : Fin 13 → Member :=
  ![⟨0, 256⟩, ⟨256, 256⟩, ⟨512, 256⟩, ⟨768, 256⟩, ⟨1024, 256⟩, ⟨1280, 256⟩, ⟨1536, 256⟩, ⟨1792, 256⟩, ⟨2048, 256⟩,
    ⟨2304, 192⟩, ⟨2496, 512⟩, ⟨3008, 112⟩, ⟨3120, 624⟩]

/-- Each member's stratum: the index of the draw file's stratum whose template is the member's. -/
def str : Fin 13 → Fin 3 := ![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]

/-- The window's strata as a map of its 3744 units: `[0, 2496)`, `[2496, 3120)`, `[3120, 3744)`. -/
def strata (u : Fin 3744) : Fin 3 := if u.val < 2496 then 0 else if u.val < 3120 then 1 else 2

/-- Each stratum's `k`. -/
def k : Fin 3 → ℕ := fun _ => 1

end WindowDeployed

/-- **`WindowBound`'s honest lemma: the deployed window.** For every reading `files` of draw files by their SHA-512
digests at which the deployed window's four draw files read as recorded
(`FlockSoundness.Assumptions.PinnedWindowDraws`), the window's draw file `W` gives `WindowDraw` and `WindowBound` their
hypotheses, and the service's split of it is the three pinned session files:

* `W` is a `stratified` draw over 3744 units whose strata are `WindowDeployed.strata`'s (`DrawFile.stratumOf`), of 2496,
  624 and 624 units, each drawing `WindowDeployed.k = 1`, as `W`'s strata say; and `W`'s units are a draw of that law;
* the thirteen members (`WindowDeployed.member`, the run's summary's) tile the window, each inside its stratum
  (`WindowDeployed.str`, its template's);
* the service's split of `W` (`Member.sessionFile`) is the three pinned session files, members 1, 10 and 12, and no
  session for the other ten; at every draw of the law every member's share has at most one unit, so a session's `k` is
  `1`, as the pinned files' is;
* for every window body, members' events and session bounds as in `WindowBound`, the deployed window's event has
  probability at most `∑_j size_j / n_{str j} · β j 1`, and at most `3 · b` when every `β j 1 ≤ b`. -/
def WindowBound_honest : Prop :=
  ∀ files : String → Option DrawFile, FlockSoundness.Assumptions.PinnedWindowDraws files →
    ∃ W : DrawFile, files WindowDeployed.drawDigest = some W ∧
      W.law = "stratified" ∧ W.population = 3744 ∧ W.units.Nodup ∧
      (∀ u : Fin 3744, (WindowDeployed.strata u).val = W.stratumOf u.val) ∧
      (∀ s : Fin 3, W.strata[s.val]?.map StratumFile.k = some (WindowDeployed.k s)) ∧
      (Law.stratum WindowDeployed.strata 0).card = 2496 ∧ (Law.stratum WindowDeployed.strata 1).card = 624 ∧
      (Law.stratum WindowDeployed.strata 2).card = 624 ∧
      (∀ s, WindowDeployed.k s ≤ (Law.stratum WindowDeployed.strata s).card) ∧
      (∀ j, (WindowDeployed.member j).base + (WindowDeployed.member j).size ≤ 3744) ∧
      (∀ j, ∀ u : Fin 3744, (WindowDeployed.member j).base ≤ u.val →
        u.val < (WindowDeployed.member j).base + (WindowDeployed.member j).size →
        WindowDeployed.strata u = WindowDeployed.str j) ∧
      (∀ u : Fin 3744, ∃! j, (WindowDeployed.member j).base ≤ u.val ∧
        u.val < (WindowDeployed.member j).base + (WindowDeployed.member j).size) ∧
      (∀ j, (WindowDeployed.member j).sessionFile W =
        if j = 1 then files WindowDeployed.m1Digest else if j = 10 then files WindowDeployed.m10Digest
        else if j = 12 then files WindowDeployed.m12Digest else none) ∧
      ∀ hk : (∀ s, WindowDeployed.k s ≤ (Law.stratum WindowDeployed.strata s).card),
        (∃ ω, (Law.stratified WindowDeployed.strata WindowDeployed.k hk).draw ω =
          Finset.univ.filter fun u => u.val ∈ W.units) ∧
        (∀ ω j, ((WindowDeployed.member j).share
          ((Law.stratified WindowDeployed.strata WindowDeployed.k hk).draw ω)).card ≤ 1) ∧
        ∀ (α Reg : Type) (body : Finset (Fin 3744) → Reg → Game α) (E : Fin 13 → Reg × Finset (Fin 3744) × α → Prop)
          (β : Fin 13 → ℕ → ℝ≥0∞),
          (∀ j, ∀ o, E j o → ((WindowDeployed.member j).share o.2.1).Nonempty) →
          (∀ j κ (hκ : κ ≤ (WindowDeployed.member j).size), 1 ≤ κ → κ ≤ 1 →
            ∀ ctx : Finset (Fin 3744), Disjoint ctx (WindowDeployed.member j).range →
            ∀ τ : Strategy (audit (Law.subset (WindowDeployed.member j).size κ hκ) Reg
                fun S R => body (ctx ∪ (WindowDeployed.member j).unshift S) R),
              prob (fun o => E j (o.1, ctx ∪ (WindowDeployed.member j).unshift o.2.1, o.2.2))
                (audit (Law.subset (WindowDeployed.member j).size κ hκ) Reg
                  fun S R => body (ctx ∪ (WindowDeployed.member j).unshift S) R) τ ≤ β j κ) →
          ∀ τ : Strategy (audit (Law.stratified WindowDeployed.strata WindowDeployed.k hk) Reg body),
            prob (fun o => ∃ j, E j o) (audit (Law.stratified WindowDeployed.strata WindowDeployed.k hk) Reg body) τ ≤
              ∑ j, ((WindowDeployed.member j).size : ℝ≥0∞) /
                ((Law.stratum WindowDeployed.strata (WindowDeployed.str j)).card : ℝ≥0∞) * β j 1 ∧
            ∀ b : ℝ≥0∞, (∀ j, β j 1 ≤ b) →
              prob (fun o => ∃ j, E j o) (audit (Law.stratified WindowDeployed.strata WindowDeployed.k hk) Reg body) τ ≤
                3 * b

namespace WindowToy

/-- The nontrivial instance's strata over 3 units: unit 0 alone, units 1 and 2 together. -/
def strata : Fin 3 → Fin 2 := ![0, 1, 1]

/-- Each stratum draws one unit. -/
def k : Fin 2 → ℕ := fun _ => 1

/-- Each stratum has at least its one drawn unit, so `Law.stratified strata k` is a law. -/
theorem hk : ∀ s, k s ≤ (Law.stratum strata s).card := by
  decide

/-- The instance's one member, the units `[0, 2)`, across both strata. -/
def member : Member := ⟨0, 2⟩

/-- The window's body: nothing runs. -/
def body : Finset (Fin 3) → Unit → Game Unit := fun _ _ => .ret ()

end WindowToy

/-- **`WindowBound`'s nontrivial lemma**: its hypothesis that each member's units are in its stratum can't be dropped.
At a window of 3 units in two strata (`WindowToy.strata`: unit 0, then units 1 and 2), each drawing one unit, with one
member, the units `[0, 2)` taken in stratum 0, whose event is that its share has two units, and session bound `β = 0`,
the composed bound's conclusion fails: the window's event, probability `1/2`, is above the composed bound `0`.
The premise dropped is the member's units in its stratum (`SecurityProofs.WindowToy.in_stratum_fails`); `WindowBound`'s
others hold at the toy (`WindowToy.hk`, `SecurityProofs.WindowToy.fit_holds`, `event_nonempty_holds` and
`session_bound_holds`). -/
def WindowBound_nontrivial : Prop :=
  ¬ ∀ τ : Strategy (audit (Law.stratified WindowToy.strata WindowToy.k WindowToy.hk) Unit WindowToy.body),
    prob (fun o => ∃ _ ∈ (Finset.univ : Finset Unit), (WindowToy.member.share o.2.1).card = 2)
        (audit (Law.stratified WindowToy.strata WindowToy.k WindowToy.hk) Unit WindowToy.body) τ ≤
      ∑ _j ∈ (Finset.univ : Finset Unit), ∑ κ ∈ Finset.Icc 1 (WindowToy.k 0),
        prCoin (fun ω =>
          (WindowToy.member.share ((Law.stratified WindowToy.strata WindowToy.k WindowToy.hk).draw ω)).card = κ) * 0

end Flock.Guarantees
