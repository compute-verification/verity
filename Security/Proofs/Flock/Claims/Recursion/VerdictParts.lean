import Proofs.Flock.Claims.Recursion.NextR
import Proofs.Flock.Claims.Recursion.CoinPos

/-!
# `lastSplitR`'s verdict, split into the parts' pieces

`lastSplitR` asks the verdict on the decoded proofs (`VerdictR.verdict`: `verifyJPWith` at `jAt`) as one check, and
`decompB_nextR`'s inventory (`ownR`) gives a check to one part. No V\* part's circuit computes the whole verdict: a
`rec-residuals` part checks its group of `GfResiduals`' residuals, a `rec-open` part its openings, and so on. So at
`lastSplitR`, the lowering fact `hlow` would ask one part's circuit for the verdict of all of them, and a check given to
no part would have to hold off any witnesses (`hrest`). Neither holds.

`lastSplitP` is the split with the verdict replaced by pieces of the code side's choice (`pieces`, `pholds`), each one a
part can own, and one obligation, `hpieces`: wherever the reading holds, the other four checks of `VerdictR` do and
every piece does, the verdict holds. Those four stay as they are (`VerdictP.base`). `ownP` gives the base checks and
the pieces their owners, and `decompB_lastSplitP` is `decompB_nextR` at it: the lowering facts are then one per piece.

`staged_rep_drawn` is the coin side of a `rec-residuals` piece: at a staged part whose statement shows coins `c`,
the rep that check 5 builds (`RecCoins.Inner.rep`, whose `verifierRows` give the `v` it pins) has, at each round `m`,
the values the coin drew at the one inner round `g + 1` of its stream at round `sp m` (`staged_coinTie_drawn_unique`).
-/

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalP nextR nextR_coin ReadOkR VerdictR holdsR)
open Claims.Recursion.VStarB

noncomputable section

/-- **A check of the split verdict**: one of `VerdictR`'s (`base`; `base .verdict` is never among the checks), or a
piece of the verdict (`piece`). -/
inductive VerdictP (Q : Type) where
  | base (q : VerdictR)
  | piece (q : Q)

/-- `VerdictR`'s checks other than the verdict. -/
def VerdictP.bases : List VerdictR := [.setup, .shaped, .names, .after]

theorem VerdictP.mem_bases {q : VerdictR} (h : q ≠ .verdict) : q ∈ VerdictP.bases := by
  cases q <;> simp_all [VerdictP.bases]

/-- **The split with the verdict in pieces**: `VerdictR`'s checks but the verdict, then the pieces at the inner
statement. Where the reading, the base checks and every piece hold, the verdict holds (`hpieces`), so `nextR_sound`
makes the last step accept. -/
def lastSplitP (H : Msg → Msg) (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
    (pholds : Q → Msg → List Msg → List Coin → Prop)
    (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
      (∀ b ∈ VerdictP.bases, holdsR H rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
        holdsR H rd .verdict x ms cs) :
    LastSplitR H rd (VerdictP Q) where
  checks x := VerdictP.bases.map .base ++ (pieces x).map .piece
  holds q x ms cs := match q with
    | .base q => holdsR H rd q x ms cs
    | .piece q => pholds q x ms cs
  sound x ms cs _ _ hr hc := by
    refine FlockDeploy.nextR_sound H rd x ms cs hr fun q => ?_
    by_cases hq : q = .verdict
    · subst hq
      exact hpieces x ms cs hr
        (fun b hb => hc (.base b) (List.mem_append_left _ (List.mem_map.2 ⟨b, hb, rfl⟩))) fun p hp =>
          hc (.piece p) (List.mem_append_right _ (List.mem_map.2 ⟨p, hp, rfl⟩))
    · exact hc (.base q) (List.mem_append_left _ (List.mem_map.2 ⟨q, VerdictP.mem_bases hq, rfl⟩))

/-- **The split verdict's inventory**: the base checks as `ownB` gives them, each piece as `ownQ` does. -/
def ownP {Q : Type} (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ) (x : Msg) :
    VerdictP Q → Option ℕ
  | .base q => ownB x q
  | .piece q => ownQ x q

/-- **The composition at the split verdict**: `decompB_nextR` at `lastSplitP`, rows joined by `joinR`, its inventory
`ownP`. What it leaves: the owners among the statement's parts (`hR`), the checks no part holds holding off any
witnesses (`hrest`), and one lowering fact per part (`hlow`), each part's circuit accepting implying the checks `ownP`
gives it, a piece of the verdict among them. -/
theorem decompB_lastSplitP (H : Msg → Msg) (rd : Msg → Option FinalP) {Q : Type} (pieces : Msg → List Q)
    (pholds : Q → Msg → List Msg → List Coin → Prop)
    (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ReadOkR rd x ms cs →
      (∀ b ∈ VerdictP.bases, holdsR H rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
        holdsR H rd .verdict x ms cs)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (ownB : Msg → VerdictR → Option ℕ) (ownQ : Msg → Q → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ (lastSplitP H rd pieces pholds hpieces).checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s,
        ownP ownB ownQ x r = some s → s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
          (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
            (lastSplitP H rd pieces pholds hpieces).toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ))) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
            (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
              (lastSplitP H rd pieces pholds hpieces).toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w)
        (deploySplit H rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w (nextR_coin rd H)
          (lastSplitP H rd pieces pholds hpieces).toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L (ownP ownB ownQ)))) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextR rd) colsE wE FlockDeploy.joinR rowIn final w).IVStar L
      shape (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_nextR H rd rel colsE wE FlockDeploy.joinR_head rowIn final w (lastSplitP H rd pieces pholds hpieces)
    (ownP ownB ownQ) rel' b L IV shape DO shows cmt hR hrest hlow

end

/-! ## The coins of a `rec-residuals` piece -/

open RecDeploy.CoinTie RecDeploy.CoinPos

theorem rep_ok {inn : RecCoins.Inner} {sh : Flock.Shape} {extras : Array Flock.Claim} {r : ℕ} {rep : RecCoins.Rep}
    (h : inn.rep sh extras r = .ok rep) :
    inn.rounds r = .ok rep.rounds ∧
      RecAlgebra.verifierRows sh rep.rounds extras RecCoins.ZERO_COMB = .ok (rep.v, rep.info, rep.kappa) := by
  unfold RecCoins.Inner.rep at h
  cases hr : inn.rounds r with
  | error e => rw [hr] at h; cases h
  | ok rounds =>
    rw [hr] at h
    cases hv : RecAlgebra.verifierRows sh rounds extras RecCoins.ZERO_COMB with
    | error e => simp [hv, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    | ok t =>
      obtain ⟨v, info, kappa⟩ := t
      simp only [hv, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, hv⟩

/-- **The rep check 5 builds is at the coins the transcript shows**: at a staged part whose statement shows coins `c`,
wherever check 5 loads the inner statement from its source and builds rep `r` (`Inner.rep`, the rep whose
`verifierRows` give the `v` a `rec-residuals` part's port must hold), each of the rep's rounds `m` is what the coin
drew at the inner round `g + 1`, the one round of `R` of rep `r`'s stream at round `sp m`, and the rep's `v` is
`verifierRows` at those rounds. -/
theorem staged_rep_drawn {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {c : ℕ → Coin}
    (hc : showsTr recRead s x xO c) :
    ∃ (i : RecIns) (R : Flock.RecRecord.Record), insO xO = some i ∧
      shownOf recRead x xO = some (R.msgs.map fun mc => (mc.1, mc.2.toArray)) ∧
      ∀ f : RecCoins.Files, RecCoins.Source.atC3 i.src i.inner i.commits = some (.files f) →
      ∀ (unit : String) (inn : RecCoins.Inner), RecCoins.Inner.load unit f = .ok inn →
      ∀ (sh : Flock.Shape) (extras : Array Flock.Claim) (r : ℕ) (rep : RecCoins.Rep), inn.rep sh extras r = .ok rep →
        RecAlgebra.verifierRows sh rep.rounds extras RecCoins.ZERO_COMB = .ok (rep.v, rep.info, rep.kappa) ∧
        ∃ t : String, ∃ sp : ℕ → ℕ, (sp = id ∨ ∃ j, sp = fun m => if m < j then m else m + 1) ∧
          ∀ m (hm : m < rep.rounds.size), ∃ g, ∃ hg : g < R.rounds.length,
            R.rounds[g].stream = (s!"{t}/rep{r}").toUTF8.toList ∧ R.rounds[g].k = sp m ∧
            Drawn (c (g + 1)) rep.rounds[m] ∧
            ∀ g' (hg' : g' < R.rounds.length), R.rounds[g'].stream = R.rounds[g].stream →
              R.rounds[g'].k = R.rounds[g].k → g' = g := by
  obtain ⟨i, R, hins, hshow, hall⟩ := staged_coinTie_drawn_unique hS hc
  refine ⟨i, R, hins, hshow, fun f hf unit inn hload sh extras r rep hrep => ?_⟩
  obtain ⟨hr, hv⟩ := rep_ok hrep
  exact ⟨hv, hall f hf unit inn hload r rep.rounds hr⟩

end RecDeploy
