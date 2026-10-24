import Proofs.Flock.Claims.Recursion.TopsOpens

/-!
# `PartChecksB` at what each part's circuit accepts (`PartAccepts`), and the corollaries without it

Every route that reads `VStarB.PartChecksB` takes it with `VStarB.DecompB` at one inner check it chooses:
`∃ chk, (∀ s, PartChecksB … chk s) ∧ DecompB … chk`. `PartChecksB` bounds `chk` from below, and `DecompB` reads `chk`
only as a hypothesis on the readers' messages, so it is antitone in `chk` (`decompB_mono`). So there is a least `chk`
that `PartChecksB` admits, what part `s`'s circuit accepts, read at the messages (`PartAccepts`): at a compiled
statement of the shape, some V\* statement showing the coins and some outer witness passing `Flock.check` there, with
its recorded calls true and none colliding, that agrees with the messages at the rounds it holds whole and at its
pieces.

* `PartChecksB` holds at `PartAccepts`, at every deployment, layout, shape and staging (`partChecksB_partAccepts`).
* `PartAccepts` implies every inner check `PartChecksB` admits (`partAccepts_chk`).
* So the pair is `DecompB` at `PartAccepts` (`exists_chk_iff`).

`PartChecksB` therefore has no content apart from `DecompB`. What V\*'s units must compute, each its share of
`IVStar`'s checks, is all in `DecompB` at `PartAccepts`: wherever every part's circuit accepts at a witness that agrees
with the readers' messages, `IVStar` accepts those messages, or two of the parts' recorded calls collide. An inner check
of the code side's choice (one stated against the element-level verdict's steps, say) still gives it: any `chk` with
both facts does.

`ksLin2_deployed_partAccepts` and `bitsLin2_deployed_partAccepts` are `ksLin2_deployed_recRead` and
`bitsLin2_deployed_recRead` with the pair replaced by `DecompB` at `PartAccepts`, the same premise. What they take:
`DecompB` there, the readers' `ReadersTrue`, and `recRead` reading the coin server's transcript back (`ReadsBackB`).
-/

namespace Claims.Recursion.VStarB

open Claims

section Facts

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **What V\* part `s`'s circuit accepts, read at the inner messages `ms`**, at the coins `c`: `x` is a compiled
statement of the shape, and some V\* statement `xO` showing `c` for the part (`shows`) has an outer witness `w` passing
`Flock.check`, its recorded calls true evaluations and none colliding, that agrees with `ms` at each round the part
holds whole and at each of its pieces, at its offset. It is the least inner check `PartChecksB` admits
(`partChecksB_partAccepts`, `partAccepts_chk`). -/
def PartAccepts (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (s : ℕ) (x : Msg) (c : ℕ → Coin)
    (ms : List Msg) : Prop :=
  (∃ st : Stmt, D.parse x = some st ∧ shape st.form) ∧
    ∃ (xO : Msg) (w : Flock.Wit), Flock.check DO.parse xO w = true ∧ (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧
      (findColl (L.calls x s w)).isSome = false ∧ shows s x xO c ∧
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) ∧
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`PartChecksB` at what each part's circuit accepts**, at every deployment, layout, shape and staging: a witness
whose calls collide takes the escape, and any other is its own `PartAccepts`. -/
theorem partChecksB_partAccepts (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (s : ℕ) :
    PartChecksB H D DO L shape shows (PartAccepts H D DO L shape shows) s := by
  intro x st hx hf xO w hck hT
  cases hc : (findColl (L.calls x s w)).isSome
  · exact Or.inl fun c hs ms hh hp => ⟨⟨st, hx, hf⟩, xO, w, hck, hT, hc, hs, hh, hp⟩
  · exact Or.inr rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`PartAccepts` is the least inner check `PartChecksB` admits**: at every part, it implies any `chk` at which
`PartChecksB` holds. -/
theorem partAccepts_chk {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hP : ∀ s, PartChecksB H D DO L shape shows chk s)
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg} (h : PartAccepts H D DO L shape shows s x c ms) :
    chk s x c ms := by
  obtain ⟨⟨st, hx, hf⟩, xO, w, hck, hT, hc, hs, hh, hp⟩ := h
  rcases hP s x st hx hf xO w hck hT with h | h
  · exact h c hs ms hh hp
  · rw [hc] at h
    exact absurd h Bool.false_ne_true

omit [Fintype Coin] [Nonempty Coin] in
/-- **`DecompB` is antitone in the inner check**: it holds at any check that implies one where it holds. -/
theorem decompB_mono {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {L : LayoutB} {shape : Form → Prop}
    {chk chk' : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hle : ∀ s x c ms, chk' s x c ms → chk s x c ms)
    (hD : DecompB H D IV L shape chk) : DecompB H D IV L shape chk' :=
  fun x st hx hf c ws hl hT hC => hD x st hx hf c ws hl hT fun s hs => hle _ _ _ _ (hC s hs)

/-- **Some inner check gives both `PartChecksB` at every part and `DecompB` exactly when `DecompB` holds at
`PartAccepts`.** So the pair the routes take is one premise, `DecompB` at what each part's circuit accepts. -/
theorem exists_chk_iff (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (IV : Verifier Coin)
    (L : LayoutB) (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) :
    (∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, PartChecksB H D DO L shape shows chk s) ∧ DecompB H D IV L shape chk) ↔
      DecompB H D IV L shape (PartAccepts H D DO L shape shows) :=
  ⟨fun ⟨_, hP, hD⟩ => decompB_mono (fun _ _ _ _ => partAccepts_chk hP) hD,
    fun hD => ⟨_, partChecksB_partAccepts H D DO L shape shows, hD⟩⟩

end Facts

end Claims.Recursion.VStarB

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)` and the coin server's record,
without `PartChecksB`**: `ksLin2_deployed_recRead`, with its inner check `PartAccepts` (`partChecksB_partAccepts`).
What stays open at the dense form is `DecompB` at what each part's circuit accepts, the same premise as the pair
(`exists_chk_iff`); then the readers' `ReadersTrue` and `recRead` reading the coin server's transcript back
(`ReadsBackB`). This bounds no number by itself: those premises are open. -/
theorem ksLin2_deployed_partAccepts (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead b))) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hD hRT
  exact ksLin2_deployed_recRead b rest wE join final w hrb
    (fun shape hsh => (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _).2 (hD shape hsh)) hRT

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)` and the coin server's record,
without `PartChecksB`**: `bitsLin2_deployed_recRead`, with its inner check `PartAccepts`. It reads `StrictBirthday`
and `ExpectedCR`, the named SHA-512 collision assumptions, and `ksLin2_deployed_partAccepts`'s open premises: a bound
on a number under named premises, not a proved security level. -/
theorem bitsLin2_deployed_partAccepts (b : Brings)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation b L Bd.IVStar
    let bL : Brings := { b with rd := L.readers }
    ReadsBackB FlockDeploy.sha512 Bd.IVStar bL (fun _ => True) recRead →
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead b))) →
    ReadersTrue FlockDeploy.sha512 Bd.IVStar bL (deployZkRec _root_.Flock.Readable.relation) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation b L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro L Bd D bL hrb hD hRT
  exact bitsLin2_deployed_recRead b rest wE join final w hrb
    (fun shape hsh => (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _).2 (hD shape hsh)) hRT

end RecDeploy
