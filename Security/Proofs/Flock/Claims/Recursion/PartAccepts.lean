import Proofs.Flock.Claims.Recursion.TopsOpens

/-!
# `PartChecksBV` at what each part's circuit accepts (`PartAccepts`), and the corollaries without it

Every route 2C reads `VStarB.PartChecksBV` with `VStarB.DecompB` at one inner check it chooses:
`∃ chk, (∀ s, PartChecksBV … chk s) ∧ DecompB … chk`. `PartChecksBV` bounds `chk` from below, and `DecompB` reads `chk`
only as a hypothesis on the readers' messages, so it is antitone in `chk` (`decompB_mono`). So there is a least `chk`
that `PartChecksBV` admits, what part `s`'s circuit accepts, read at the messages (`PartAccepts`): at a compiled
statement of the shape, some V\* statement showing the coins and some outer witness passing `Flock.check` there, with
its recorded calls true and none colliding with each other or with the compiled verifier's own evaluations at that
statement (`vcallsAt`: check 5's row commitments among them), that agrees with the messages at the rounds it holds whole
and at its pieces, whose commit strings are the messages at every round the statement doesn't fix before the last, on
the bytes the readers read (`LayoutB.CmtAgrees`), and whose shared rows are the messages' where they carry them
(`LayoutB.RowsRead`). The commit strings tie every part to the readers' rounds, those it reads without holding them
included, so no two parts' checks can pass on different values of one round; the shared rows do the same for a row one
part writes and another reads. On the run, `VstarShowsB` supplies the commit strings and `LayoutB.ReadsRows` the shared
rows (`RecDeploy.vstarAccepts_bound`). `ReadsRows` holds outright at a layout whose messages carry no shared row
(`LayoutB.readsRows_of_rowIn`), as at every deployed layout, so no deployed route gains a premise.

* `PartChecksBV` holds at `PartAccepts`, at every deployment, layout, shape and staging (`partChecksBV_partAccepts`).
* `PartAccepts` implies every inner check `PartChecksBV` admits (`partAccepts_chk`).
* So the pair is `DecompB` at `PartAccepts` (`exists_chk_iff`).
* A row a part commits is the one the compiled verifier recomputes the commitment of (`partAccepts_tie`, check 5).

`PartChecksBV` therefore has no content apart from `DecompB`. What V\*'s units must compute, each its share of
`IVStar`'s checks, is all in `DecompB` at `PartAccepts`: wherever every part's circuit accepts at a witness that agrees
with the readers' messages, `IVStar` accepts those messages, or two of the parts' recorded calls collide. An inner check
of the code side's choice (one stated against the element-level verdict's steps, say) still gives it: any `chk` with
both facts does.

`ksLin2_deployed_partAccepts` and `bitsLin2_deployed_partAccepts` are `ksLin2_deployed_recRead` and
`bitsLin2_deployed_recRead` at the coin server's record (`recTr`), with the pair replaced by `DecompB` at
`PartAccepts`, the same premise, `ReadsBackB` from `readsBack_recTr` (`recRead` reads that record back at every
accepted run) and `ReadersTrue` from `readersTrue_layoutBTops` (every hm96 slot `layoutBTops` reads outputs its commit
string at the outer relation, `hmSlotsCommit_of_full`). What they take: `DecompB` there, and nothing else.
-/

namespace Claims.Recursion.VStarB

open Claims

section Facts

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **What V\* part `s`'s circuit accepts, read at the inner messages `ms`**, at the coins `c`: `x` is a compiled
statement of the shape, and some V\* statement `xO` showing `c` for the part (`shows`) has an outer witness `w` passing
`Flock.check`, its recorded calls true evaluations and none colliding with another or with the compiled verifier's at
`xO` (`vcallsAt`), that agrees with `ms` at each round the part holds whole and at each of its pieces, at its offset,
whose commit strings (`cmt`) are `ms`'s at every round the statement doesn't fix before the last, on the bytes the
readers read (`LayoutB.CmtAgrees`), and whose shared rows are `ms`'s where `ms` carries them (`LayoutB.RowsRead`). It is
the least inner check `PartChecksBV` admits (`partChecksBV_partAccepts`, `partAccepts_chk`). Refusing a collision with
the verifier's evaluations ties the rows the part commits to the ones the verifier recomputes the commitments of (check
5, `partAccepts_tie`). -/
def PartAccepts (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) (x : Msg)
    (c : ℕ → Coin) (ms : List Msg) : Prop :=
  (∃ st : CompiledStatement, D.parse x = some st ∧ shape st.form) ∧
    ∃ (xO : Msg) (w : Flock.Wit), Flock.check DO.parse xO w = true ∧ (∀ e ∈ L.calls x s w, H e.1 = e.2) ∧
      (findColl (L.calls x s w ++ vcallsAt H D xO)).isSome = false ∧ shows s x xO c ∧
      (∀ i < L.rounds x, ∀ m, L.held x s w i = some m → ms.getD i [] = m) ∧
      (∀ i < L.rounds x, ∀ o m, (o, m) ∈ L.piecesOf x s w i → ((ms.getD i []).drop o).take m.length = m) ∧
      L.CmtAgrees cmt x xO ms ∧ L.RowsRead x s w ms

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`PartChecksBV` at what each part's circuit accepts**, at every deployment, layout, shape and staging: a witness
whose calls collide, or collide with the verifier's, takes the escape, and any other is its own `PartAccepts`. -/
theorem partChecksBV_partAccepts (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (L : LayoutB)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) (s : ℕ) :
    PartChecksBV H D DO L shape shows cmt (PartAccepts H D DO L shape shows cmt) s := by
  intro x st hx hf xO w hck hT
  cases hc : (findColl (L.calls x s w ++ vcallsAt H D xO)).isSome
  · exact Or.inl fun c hs ms hh hp hm hr => ⟨⟨st, hx, hf⟩, xO, w, hck, hT, hc, hs, hh, hp, hm, hr⟩
  · exact Or.inr rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **`PartAccepts` is the least inner check `PartChecksBV` admits**: at every part, it implies any `chk` at which
`PartChecksBV` holds. -/
theorem partAccepts_chk {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hP : ∀ s, PartChecksBV H D DO L shape shows cmt chk s)
    {s : ℕ} {x : Msg} {c : ℕ → Coin} {ms : List Msg} (h : PartAccepts H D DO L shape shows cmt s x c ms) :
    chk s x c ms := by
  obtain ⟨⟨st, hx, hf⟩, xO, w, hck, hT, hc, hs, hh, hp, hm, hr⟩ := h
  rcases hP s x st hx hf xO w hck hT with h | h
  · exact h c hs ms hh hp hm hr
  · rw [hc] at h
    exact absurd h Bool.false_ne_true

omit [Fintype Coin] [Nonempty Coin] in
/-- **`DecompB` is antitone in the inner check**: it holds at any check that implies one where it holds. -/
theorem decompB_mono {H : Msg → Msg} {D : Deploy Coin} {IV : Verifier Coin} {L : LayoutB} {shape : Form → Prop}
    {chk chk' : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} (hle : ∀ s x c ms, chk' s x c ms → chk s x c ms)
    (hD : DecompB H D IV L shape chk) : DecompB H D IV L shape chk' :=
  fun x st hx hf c ws hl hT hC => hD x st hx hf c ws hl hT fun s hs => hle _ _ _ _ (hC s hs)

/-- **Some inner check gives both `PartChecksBV` at every part and `DecompB` exactly when `DecompB` holds at
`PartAccepts`.** So the pair the routes take is one premise, `DecompB` at what each part's circuit accepts. -/
theorem exists_chk_iff (H : Msg → Msg) (D : Deploy Coin) (DO : Flock.Deploy Coin Col T) (IV : Verifier Coin)
    (L : LayoutB) (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) :
    (∃ chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop,
        (∀ s, PartChecksBV H D DO L shape shows cmt chk s) ∧ DecompB H D IV L shape chk) ↔
      DecompB H D IV L shape (PartAccepts H D DO L shape shows cmt) :=
  ⟨fun ⟨_, hP, hD⟩ => decompB_mono (fun _ _ _ _ => partAccepts_chk hP) hD,
    fun hD => ⟨_, partChecksBV_partAccepts H D DO L shape shows cmt, hD⟩⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Check 5's tie at what a part's circuit accepts**: `PartAccepts` names a V\* statement `xO` showing the coins and
an outer witness `w` there, and wherever `w`'s recorded calls hold `frame v σ` with the output `d` to which the compiled
verifier at `xO` evaluates `frame v₀ σ₀` (`Deploy.setup`), `v = v₀` (`row_tie`): the part's row is the verifier's. -/
theorem partAccepts_tie {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : LayoutB}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {ms : List Msg} (h : PartAccepts H D DO L shape shows cmt s x c ms) :
    ∃ (xO : Msg) (w : Flock.Wit), Flock.check DO.parse xO w = true ∧ shows s x xO c ∧
      ∀ (frame : Msg → Msg → Msg), (∀ v σ v' σ', frame v σ = frame v' σ' → v = v') → ∀ v σ v₀ σ₀ d : Msg,
        (frame v σ, d) ∈ L.calls x s w → frame v₀ σ₀ ∈ D.setup xO → H (frame v₀ σ₀) = d → v = v₀ := by
  obtain ⟨-, xO, w, hck, hT, hc, hs, -⟩ := h
  exact ⟨xO, w, hck, hs, fun _ hinj _ _ _ _ _ hm hv hd => row_tie hinj hm hT hv hd hc⟩

end Facts

end Claims.Recursion.VStarB

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin)

/-- **`KSLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)` and the coin server's record
(`recTr`), from `DecompB` alone**: `ksLin2_deployed_recRead` at the brings whose transcript is
`recTr aux nOf hd recRest`, with its inner check `PartAccepts` (`partChecksBV_partAccepts`), `recRead` reading that
transcript back (`readsBack_recTr`) and the readers' calls true (`readersTrue_layoutBTops`,
`hmSlotsCommit_of_full`). What stays open at the dense form is `DecompB` at what each part's circuit accepts, the same
premise as the pair (`exists_chk_iff`). This bounds no number by itself: that premise is open. -/
theorem ksLin2_deployed_partAccepts (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  exact ksLin2_deployed_recRead bR rest wE join final w
    (readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl)
    (fun shape hsh => (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _ _).2 (hD shape hsh))
    (readersTrue_layoutBTops (finOpen rest) Bd.IVStar bR _ hmSlotsCommit_of_full)

/-- **`BitsLin2` at the deployment's dense statements, at `layoutBTops (finOpen rest)` and the coin server's record
(`recTr`), from `DecompB` alone**: `bitsLin2_deployed_recRead` at `recTr aux nOf hd recRest`, with its inner check
`PartAccepts`, `ReadsBackB` from `readsBack_recTr` and `ReadersTrue` from `readersTrue_layoutBTops`. It reads
`StrictBirthday` and `ExpectedCR`, the named SHA-512 collision assumptions, and `ksLin2_deployed_partAccepts`'s open
premise: a bound on a number under named premises, not a proved security level. -/
theorem bitsLin2_deployed_partAccepts (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (rest : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE : Msg → ℝ)
    (join : List Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    let bR : Brings := { b with tr := recTr aux nOf hd recRest }
    let L := layoutBTops (finOpen rest)
    let Bd := bdDeployed _root_.Flock.Readable.relation (FlockDeploy.V false).next
      (FlockDeploy.deployPlainTrim _root_.Flock.Readable.relation).cols wE join rowInDeployed final w
    let D := recBound _root_.Flock.Readable.relation bR L Bd.IVStar
    (∀ shape ∈ [Claims.Recursion.Form.Dense],
      Claims.Recursion.VStarB.DecompB FlockDeploy.sha512 D Bd.IVStar L shape
        (Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 D (deployZkRec _root_.Flock.Readable.relation) L
          shape (showsPart recRead bR) (cmtTr recRead))) →
    Claims.Recursion.BitsLin2On FlockDeploy.sha512 (recBoundAt _root_.Flock.Readable.relation bR L) Bd
      (deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep DenseAt := by
  intro bR L Bd D hD
  exact bitsLin2_deployed_recRead bR rest wE join final w
    (readsBack_recTr _ _ { bR with rd := L.readers } _ aux nOf hd recRest rfl)
    (fun shape hsh => (Claims.Recursion.VStarB.exists_chk_iff _ _ _ _ _ _ _ _).2 (hD shape hsh))
    (readersTrue_layoutBTops (finOpen rest) Bd.IVStar bR _ hmSlotsCommit_of_full)

end RecDeploy
