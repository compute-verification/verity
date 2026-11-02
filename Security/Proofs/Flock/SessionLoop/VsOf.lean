import Proofs.Flock.SessionLoop.Phase
import Proofs.Flock.SessionLoop.FinalC

/-!
# The values a pad-blind worker supplies

`Loop.TakesVals vs L wk x w` ties a worker to values: each word the frame frees in the worker's round is `vs`'s value at
that pad index. A worker of the algebraic phase (`Loop.AlgWorker`, run as `Loop.phased`) reads the nonce, what the
workers see of the tape (`RefSession.seenOf`) and the answers read as coins, so the words it frees are a function of
what `RefSession.SwVals` reads, once each pad index has one round that frees it. This file builds that function:

* `Loop.FreedAt L wk x w ν s cs j r i a`: after a history `TakesVals` reads, whose answers after `Hello` open the coins
  `cs` and whose tape the workers see as `s`, the frame frees pad `i` of rep stream `2j + r` in `wk`'s round as the
  word `a`.
* `Loop.freeWord L bx`: the word `bx`, as a pad-blind worker (`blind (bwOf bx)`), frees at pad `i`, at the history
  whose answers open the coins' first `n` (`FreedAt`), where there is one.
* **`Loop.vsOf L bx`**: at each pad index, `freeWord`'s word where the frame frees one there; at `final_c`'s index the
  c_eval of the round-1c values at the coins' zerocheck point (`zcE`), as the firewall's finish computes it; at every
  other index the table's own value (`Table.vals`).
* **`Loop.takesVals_vsOf`**: at `SessionLoopZKDeployed`'s release with the loop's own inner rounds (`withInner`), at
  its premises on the statement and the loop's inputs, `phased bx post` meets `TakesVals` at `vsOf bx`, whatever `post`
  reads, at every `bx` that never reads `Hello`'s answer (`Loop.HelloBlind`).
* **`Loop.finalCFixed_vsOf`**: `vsOf bx`'s `final_c` is its round-1c values' c_eval (`Loop.FinalCFixed`), at every
  coins, by construction: no round frees `final_c`'s pad.

`Loop.HelloBlind` is not yet a premise of any guarantee. `Loop.blind` hands the worker `Hello`'s answer read as a coin
(`ZkServer.lift` maps `CoinLayer.rd` over the whole history), while `TakesVals` reads `vs` at the coins the later answers
open, so at a worker that reads `Hello`'s answer no `vs` meets it: the verifier chooses that answer, and two answers
whose roots agree validate the same openings.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace Loop

section VsOf

variable (L : LeanRelease Flock.Wit
  ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))

/-- **The frame frees pad `i` of rep stream `2j + r` as the word `a`** in `wk`'s round after a history `TakesVals`
reads (`0 < ops.length < refK x` openings, live, whose openings pass under the nonce `ν`), whose answers after `Hello`
open the coins `cs` and whose tape the workers see as `s`: the frame takes the stream's coins (`answered`), frees words
in `wk`'s round there (`Frame.roundFree`), and its `k`-th, at the absorbed order's place `m.pos + k`, is pad `i`, as
`a`. -/
def FreedAt (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z) (cs : List FlockDeploy.Coin)
    (j : Fin (jOf x)) (r : Fin 2) (i : ℕ) (a : F128) : Prop :=
  ∃ (t : (CoinLayer.session loopNr Masked.dataE).Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker) (out : ByteArray) (s' : _root_.Flock.Firewall.Frame.Masker)
    (fr : Array ℕ) (k : ℕ),
    (CoinLayer.session loopNr Masked.dataE).seenOf x t = s ∧
    ops.map (fun p => (CoinLayer.session loopNr Masked.dataE).decOf p.2) = cs ∧
    0 < ops.length ∧ ops.length < refK x ∧ L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true ∧
    ((allOk ((CoinLayer.session loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true ∧
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) ∧
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m ∧
    _root_.Flock.Firewall.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
      ⟨v.toArray⟩ (L.coins x (ops.length + 1)) = .ok (out, s', fr) ∧
    L.rep x (ops.length + 1) = 2 * j.val + r.val ∧ k < fr.size ∧
    (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]! = i ∧
    (_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! = a

open Classical in
/-- **The word `bx` frees at pad `i` of rep stream `2j + r`**, at the coins `c` and what the workers see `s`: where for
some `n < refK x` the frame frees pad `i` in the pad-blind worker `blind (bwOf bx)`'s round after a history whose
answers after `Hello` open `c`'s first `n` coins (`FreedAt`), a word it frees there; none where no round frees it. -/
noncomputable def freeWord (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (c : Fin (refK x) → FlockDeploy.Coin) (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (j : Fin (jOf x)) (r : Fin 2) (i : ℕ) : Option F128 :=
  if h : ∃ n a, n < refK x ∧ FreedAt L (blind L (bwOf L bx)) x w ν s ((List.ofFn c).take n) j r i a then
    some (Classical.choose (Classical.choose_spec h))
  else none

/-- **`vsOf`'s value at pad `i` other than `final_c`'s**: `freeWord`'s where a round frees pad `i`, else the table's own
value at the table witness `wt` and the base `(u, V)`; `0` from `n_E` on. -/
noncomputable def vsOfWord (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (c : Fin (refK x) → FlockDeploy.Coin) (s : ∀ j, ZkView.Seen ((CoinLayer.session loopNr Masked.dataE).C x j).z)
    (j : Fin (jOf x)) (r : Fin 2) (wt : FlockSoundness.Model.Witness ((CoinLayer.session loopNr Masked.dataE).C x j).S.m)
    (u : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Mask)
    (V : (((CoinLayer.session loopNr Masked.dataE).C x j).T
      ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.Vis) (i : ℕ) : F128 :=
  match freeWord L bx x w ν c s j r i with
  | some a => a
  | none =>
    if hi : i < (((CoinLayer.session loopNr Masked.dataE).C x j).T
        ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.nE then
      FlockLevel3.GF128.val ((((CoinLayer.session loopNr Masked.dataE).C x j).T
        ((CoinLayer.session loopNr Masked.dataE).cz x c j)).tab.vals r wt u V ⟨i, hi⟩)
    else 0

/-- **The values a pad-blind worker of the algebraic phase supplies**: at each pad index the frame frees, the word `bx`
frees there at the history whose answers open the coins' prefix (`freeWord`); at `final_c`'s index the c_eval
`Σ_i λ_i(z)·r1c_i` of the round-1c values at the rep's zerocheck point (`zcE`), as the firewall's finish computes it
(`Flock.dot`, `Flock.lagrange`); at every other index the table's own value (`vsOfWord`). -/
noncomputable def vsOf (bx : AlgWorker L) : (CoinLayer.session loopNr Masked.dataE).SwVals :=
  fun x w ν c s j r wt u V i =>
    if i.val = (layE x j).fc then
      (_root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
        ((Array.range 64).map fun k => vsOfWord L bx x w ν c s j r wt u V ((layE x j).r1c k)) : FlockLevel3.GF128)
    else (vsOfWord L bx x w ν c s j r wt u V i.val : FlockLevel3.GF128)

/-- **The algebraic phase's worker never reads `Hello`'s answer**: its value is the same at two answer lists that differ
in their first, `Hello`'s, answer read as a coin. The deployed worker never receives `Hello`'s answer. -/
def HelloBlind (bx : AlgWorker L) : Prop :=
  ∀ x w ν s (a a' : FlockDeploy.Coin) (cs : List FlockDeploy.Coin) (n : ℕ), bx x w ν s (a :: cs) n = bx x w ν s (a' :: cs) n

/-- **`vsOf`'s word below `n_E`**: the c_eval of its round-1c words at `final_c`'s index, `vsOfWord` elsewhere, at the
tape's base and what the workers see of the tape. -/
theorem vsWord_vsOf (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce)
    (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {i : ℕ}
    (hi : i < (tabC PE x j).z.nE) :
    vsWord (vsOf L bx) x w ν c τ j r i =
      if i = (layE x j).fc then
        _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
          ((Array.range 64).map fun k => vsOfWord L bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r
            (wtOf PE x w j) (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 ((layE x j).r1c k))
      else vsOfWord L bx x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ) j r (wtOf PE x w j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1 i := by
  unfold vsWord
  rw [dif_pos hi]
  rfl

/-- **`vsOf`'s `final_c` is its round-1c values' c_eval**, at every coins: `vsOf` sets it so, and the round-1c indices
are below `final_c`'s. -/
theorem finalCFixed_vsOf (bx : AlgWorker L) (x : Msg) (w : Flock.Wit) : FinalCFixed (vsOf L bx) x w := by
  intro ν c τ j _ r
  have hE : (tabC PE x j).z.nE = (layE x j).hab := by
    show (layE x j).len - 4 = (layE x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlt : ∀ i < 64, (layE x j).r1c i < (layE x j).fc := fun i _ => by
    simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1c, _root_.Flock.Zk.PadLayout.fc,
      _root_.Flock.Zk.PadLayout.fa]
    omega
  have hfc : (layE x j).fc < (tabC PE x j).z.nE := by rw [hE]; exact lt_hab (layE x j) le_rfl
  have hr1c : ∀ i < 64, (layE x j).r1c i < (tabC PE x j).z.nE := fun i hi => by
    rw [hE]; exact lt_hab (layE x j) (hlt i hi).le
  rw [vsWord_vsOf L bx x w ν c τ j r hfc, if_pos rfl]
  refine congrArg₂ _root_.Flock.dot rfl (Array.ext (by simp) fun i h1 _ => ?_)
  have hi : i < 64 := by simpa using h1
  simp only [Array.getElem_map, Array.getElem_range]
  rw [vsWord_vsOf L bx x w ν c τ j r (hr1c i hi), if_neg (Nat.ne_of_lt (hlt i hi))]

end VsOf

/-- **A pad-blind worker of the algebraic phase meets `TakesVals` at its own values**: at `SessionLoopZKDeployed`'s
release with the loop's own inner rounds (`withInner`), at its premises on the statement and the loop's inputs, at
every algebraic phase's worker `bx` that never reads `Hello`'s answer (`HelloBlind`) and any worker `post` after the
phase, `phased bx post` meets `TakesVals` at `vsOf bx`. Each pad index has one round that frees it: an earlier round of
a live history passed the frame's check, so it frees the scheduled round's words (`passes_frames`,
`roundFree_alike_ok`), and the frame's state before a round reads only the coins (`Loop.progOf`, `Loop.freshOf`); and
there the word is `bx`'s at the nonce, what the workers see and the coins after `Hello` (`phased_eq`). -/
theorem takesVals_vsOf
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (hreal : real = Loop.realOf src) (x : Msg) (w : Flock.Wit)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Serves x)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hSrc : Loop.Src (LeanRelease.withInner Masked.dataE L₁ real fin) src x w)
    (bx : AlgWorker (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real))
    (hH : HelloBlind _ bx)
    (post : (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real).toSrv.Worker) :
    TakesVals (vsOf _ bx) (LeanRelease.ofLoopLink (LeanRelease.withInner Masked.dataE L₁ real fin) src real)
      (phased _ bx post) x w := by
  sorry

end Loop

end Claims.ZKServer
