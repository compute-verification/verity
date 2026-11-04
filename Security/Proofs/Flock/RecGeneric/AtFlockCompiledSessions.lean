module

public import Proofs.Flock.RecGeneric.AtFlockCompiled
public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Flock.Claims.Recursion.BoundDeploy
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Assembled
public import Proofs.Flock.SamplingGeneric

/-!
# L2 at Flock: the compiler's contract for `E_O`, and Flock's KS as the outer argument's (A1b, A1c)

`Compiled.Sessions` at `Compiled.ofRec`. At every compiled record the staged statements read only the rounds before
the sessions (`ofRec_stages`); where each session's V\* statement has the session's numbers (`SessionNums`), it has
the session's public shape (`ofRec_shapes`) and its errors are the numbers' `κ`, `γ` (`ofRec_bounded`, at nonnegative
table errors); an accepted run shows each session accepting where the record's reader does (`ofRec_accepts`). At the
deployment a session's first round in a run (`Claims.Recursion.offO`) is `Compiled.off`'s, read off the numbers alone
(`offO_deployed`), so the sessions end at the compiled verifier's last round (`ofRec_fits_deployed`) and each fork at
a session's first round runs the outer verifier there (`ofRec_runs_deployed`, from `RecDeploy.sessionRuns_sha512`):
`sessionsOfRec_deployed`.

Flock `--zk`'s KS at the deployed outer record is `Arg.ofFlock`'s (`flockArgKS_deployed`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

section Generic

variable {Coin : Type} [Inhabited Coin] {Col' T' : Type}
  (D : RecAt Coin) (DO : Claims.Flock.Deploy Coin Col' T') (IV : Verifier Coin)

/-- The staged statements read only the inner and staging rounds' messages. -/
theorem ofRec_stages : (Compiled.ofRec D DO IV).Stages := by
  intro x ms
  show (D.record IV).vstar x (ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x)) =
    (D.record IV).vstar x ((ms.take (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x)).take
      (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))
  rw [List.take_take, min_self]

/-- Each session's staged statement has the session's waits, truncation, rounds and `N₀`, so its public shape. -/
theorem ofRec_shapes (H : Msg → Msg) (hN : SessionNums (D.record IV) DO) : (Compiled.ofRec D DO IV).Shapes H := by
  intro x Ω P ω r _ s
  obtain ⟨hk, hW, hK, hN₀, -, -⟩ := hN x s ((run H P (D.record IV).V x ω r).msgs.take
    (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))
  exact congrArg₂ Prod.mk hW (congrArg₂ Prod.mk hK (congrArg₂ Prod.mk hk hN₀))

/-- Each session's errors at its staged statement are its numbers' `κ` and `γ`, which are nonnegative where its table
error is. -/
theorem ofRec_bounded (H : Msg → Msg) (hN : SessionNums (D.record IV) DO) (hε : TableErrNonneg (D.record IV)) :
    (Compiled.ofRec D DO IV).Bounded H := by
  intro x s
  refine ⟨Nums.κ_nonneg _ (hε x s), Nums.γ_nonneg _, fun Ω P ω r _ => ?_⟩
  obtain ⟨hk, hW, hK, hN₀, hεt, hwt⟩ := hN x s ((run H P (D.record IV).V x ω r).msgs.take
    (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))
  have hkk : DO.k (((D.record IV).vstar x ((run H P (D.record IV).V x ω r).msgs.take
      (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))).getD s []) =
      ((D.record IV).nums x s).k := by
    unfold Claims.Flock.Deploy.k Nums.k
    rw [hN₀]
  refine ⟨le_of_eq ?_, le_of_eq ?_⟩
  · show Claims.Flock.κ DO (((D.record IV).vstar x ((run H P (D.record IV).V x ω r).msgs.take
      (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))).getD s []) = ((D.record IV).nums x s).κ
    unfold Claims.Flock.κ Nums.κ
    rw [hεt, hW, hK, hkk]
  · show Claims.Flock.γ DO (((D.record IV).vstar x ((run H P (D.record IV).V x ω r).msgs.take
      (IV.k (Claims.Recursion.xI (D.record IV) x) + (D.record IV).kS x))).getD s []) = ((D.record IV).nums x s).γ
    unfold Claims.Flock.γ Nums.γ
    rw [hk, hwt]

/-- An accepted run shows each session accepting where the record's reader does. -/
theorem ofRec_accepts (hS : SessionAccepts (D.record IV)) : (Compiled.ofRec D DO IV).Accepts :=
  fun x s v hs hv => hS x s v hs hv

end Generic

section Deployed

open FlockDeploy (Coin)

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : RecDeploy.Brings)
  (Ly : Claims.Recursion.VStarB.LayoutB)

/-- An inner protocol whose verifier is `IV`: the deployed facts about the sessions read an inner protocol only
through its verifier. -/
noncomputable def innerAt (IV : Verifier Coin) : Claims.Recursion.Inner Coin Claims.Flock.Wit :=
  { Claims.Recursion.Inner.ofFlock (RecDeploy.deployZkRec rel) (RecDeploy.deployZkRec rel).decode with V := IV }

/-- **A session's first round in a run is `Compiled.off`'s** at the deployment: the earlier sessions' V\* statements
have their numbers' rounds whatever the run (`RecDeploy.vstar_map_k`). -/
theorem offO_deployed (IV : Verifier Coin) {W : Type} (I : Claims.Recursion.Inner Coin W) (hI : I.V = IV)
    (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (s : ℕ) (hs : s ≤ RecDeploy.nS x)
    (r : Fin (((RecDeploy.recBoundAt rel b Ly).record IV).V.k x) → Coin) :
    offO H ((RecDeploy.recBoundAt rel b Ly).record IV) I (RecDeploy.deployZkRec rel) P x ω s r =
      (Compiled.ofRec (RecDeploy.recBoundAt rel b Ly) (RecDeploy.deployZkRec rel) IV).off x s := by
  subst hI
  show I.V.k (RecDeploy.xIOf x) + RecDeploy.kS x +
      (((RecDeploy.vstar b.st x ((run H P ((RecDeploy.recBoundAt rel b Ly).record I.V).V x ω r).msgs.take
        (I.V.k (RecDeploy.xIOf x) + RecDeploy.kS x))).take s).map RecDeploy.VO.k).sum =
    I.V.k (RecDeploy.xIOf x) + RecDeploy.kS x +
      ((List.range s).map fun t => (RecDeploy.nums b.st x t).rounds).sum
  rw [List.map_take, RecDeploy.vstar_map_k, ← List.map_take, List.take_range, Nat.min_eq_left hs]

/-- **The sessions end at the compiled verifier's last round** at the deployment: its rounds are the inner and staging
rounds and every session's. -/
theorem ofRec_fits_deployed (IV : Verifier Coin) :
    (Compiled.ofRec (RecDeploy.recBoundAt rel b Ly) (RecDeploy.deployZkRec rel) IV).Fits :=
  fun _ => le_of_eq rfl

/-- **Each fork at a session's first round runs the outer verifier there** at the deployment, at SHA-512:
`RecDeploy.sessionRuns_sha512` at `Compiled.off` (`offO_deployed`). -/
theorem ofRec_runs_deployed (IV : Verifier Coin) :
    (Compiled.ofRec (RecDeploy.recBoundAt rel b Ly) (RecDeploy.deployZkRec rel) IV).Runs FlockDeploy.sha512 := by
  intro x s Ω P ω r d hacc
  rw [← offO_deployed rel b Ly IV (innerAt rel IV) rfl FlockDeploy.sha512 P x ω s s.2.le r]
  exact RecDeploy.sessionRuns_sha512 (innerAt rel IV) (FlockDeploy.deployPlain rel).scopeCheck
    { b with rd := Ly.readers } rel x s Ω P ω r d hacc

/-- **A1b: the compiler meets its contract for `E_O` at the deployment**, at every inner verifier. -/
theorem sessionsOfRec_deployed (IV : Verifier Coin) :
    SessionsOfRec FlockDeploy.sha512 (RecDeploy.recBoundAt rel b Ly) (RecDeploy.deployZkRec rel) IV := by
  show (Compiled.ofRec (RecDeploy.recBoundAt rel b Ly) (RecDeploy.deployZkRec rel) IV).Sessions FlockDeploy.sha512
  exact
    { stages := ofRec_stages _ _ _
      shapes := ofRec_shapes _ _ _ _ (RecDeploy.sessionNums _ _ _ rel)
      bounded := ofRec_bounded _ _ _ _ (RecDeploy.sessionNums _ _ _ rel) (RecDeploy.tableErrNonneg _ _ _)
      fits := ofRec_fits_deployed rel b Ly IV
      accepts := ofRec_accepts _ _ _ (RecDeploy.sessionAccepts _ _ _)
      runs := ofRec_runs_deployed rel b Ly IV }

end Deployed

/-- **A1c: Flock `--zk`'s KS at the deployed outer record is `Arg.ofFlock`'s**, at its deployed decoder, which lists
within its radius. -/
theorem flockArgKS_deployed :
    (Arg.ofFlock (RecDeploy.deployZkRec _root_.Flock.Readable.relation)).KS FlockDeploy.sha512 :=
  Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ FlockDeploy.KS.zk_KS_outer
    (Claims.Sampling.listWithin_deployZk _root_.Flock.Readable.relation)

end Claims.RecGeneric
