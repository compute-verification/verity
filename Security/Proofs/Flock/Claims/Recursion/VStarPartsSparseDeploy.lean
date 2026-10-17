import Proofs.Flock.Claims.Recursion.DeployReg
import Proofs.Flock.Claims.Recursion.VStarPartsSparse

/-!
# V*'s parts at the sparse shape, at the compiled verifier

At the readers of a layout (`RecDeploy.readersOf L`), `deployRec` reads through them by its definition
(`RecDeploy.readsThrough_deployRec`) and stages one V* statement per session (`RecDeploy.vstarLen_deployRec`). There
`VStarAcceptsR` at the sparse shape holds from the parts' links (`PartLinks`), their openings (`PartOpens`), the staging
(`Staged`) and the inner verifier's decomposition (`Decomp`): `vstarAccepts_sparse_deployRec`. At the outer record
`deployZkRec relation`, the parser is the deployed one by definition (`vstarAccepts_sparse_deployZkRec`), and V*'s
parts at the sparse shape there (`vstarParts_sparse_deployZkRec`) are the sparse premise of
`RecDeploy.ksA_deployed_parts`.
-/

namespace Claims.Recursion.VStar

variable {W Col T : Type}

/-- **`VStarAcceptsR` at the sparse shape, at the compiled verifier**: at the readers of a layout
(`RecDeploy.readsThrough_deployRec`) and the deployed outer parser, every part's links, its openings, the staging and the
inner verifier's decomposition give `VStarAcceptsR` at `Form.Sparse` (`vstarAccepts_sparse_of_links`, with
`RecDeploy.vstarLen_deployRec`). -/
theorem vstarAccepts_sparse_deployRec {H : Msg → Msg} {IV : Verifier FlockDeploy.Coin} {ivScope : Msg → Bool}
    {b : RecDeploy.Brings} {I : Inner FlockDeploy.Coin W} {DO : Flock.Deploy FlockDeploy.Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read} (hb : b.rd = RecDeploy.readersOf L)
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hC : ∀ s, PartLinks H (RecDeploy.deployRec IV ivScope b) decode L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens H (RecDeploy.deployRec IV ivScope b) DO L Form.Sparse opening cmt s)
    (hS : Staged H (RecDeploy.deployRec IV ivScope b) I L Form.Sparse shows cmt)
    (hD : Decomp H (RecDeploy.deployRec IV ivScope b) IV Form.Sparse chk) :
    VStarAcceptsR H (RecDeploy.deployRec IV ivScope b) I DO Form.Sparse :=
  vstarAccepts_sparse_of_links hDO (RecDeploy.readsThrough_deployRec H IV ivScope b L hb)
    (RecDeploy.vstarLen_deployRec IV ivScope b) hC hO hS hD

/-- **`VStarAcceptsR` at the sparse shape, at the compiled verifier and its outer record** `deployZkRec relation`,
whose parser is `parseOf relation readOfO` by its definition (`vstarAccepts_sparse_deployRec`). -/
theorem vstarAccepts_sparse_deployZkRec {H : Msg → Msg} {IV : Verifier FlockDeploy.Coin} {ivScope : Msg → Bool}
    {b : RecDeploy.Brings} {I : Inner FlockDeploy.Coin W} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hb : b.rd = RecDeploy.readersOf L)
    (hC : ∀ s, PartLinks H (RecDeploy.deployRec IV ivScope b) RecDeploy.readOfO L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens H (RecDeploy.deployRec IV ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Sparse opening cmt s)
    (hS : Staged H (RecDeploy.deployRec IV ivScope b) I L Form.Sparse shows cmt)
    (hD : Decomp H (RecDeploy.deployRec IV ivScope b) IV Form.Sparse chk) :
    VStarAcceptsR H (RecDeploy.deployRec IV ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation)
      Form.Sparse :=
  vstarAccepts_sparse_deployRec hb rfl hC hO hS hD

/-- **V\*'s parts at the sparse shape, at the compiled verifier and its outer record** `deployZkRec relation`: the
parts' links, openings and logs, the staging, the decompositions and the held rounds give `VStarParts` at `Form.Sparse`
(`vstarParts_sparse_of_links`, the parser by definition), which is `RecDeploy.ksA_deployed_parts`' sparse premise at
`relO := relation`. -/
theorem vstarParts_sparse_deployZkRec {H : Msg → Msg} {IV : Verifier FlockDeploy.Coin} {ivScope : Msg → Bool}
    {b : RecDeploy.Brings} {I : Inner FlockDeploy.Coin W} {L : Layout} {opening : Opening}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {cmt : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hC : ∀ s, PartLinks H (RecDeploy.deployRec IV ivScope b) RecDeploy.readOfO L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens H (RecDeploy.deployRec IV ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Sparse opening cmt s)
    (hL : ∀ s, PartLogs H (RecDeploy.deployRec IV ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Sparse shows logs s)
    (hS : Staged H (RecDeploy.deployRec IV ivScope b) I L Form.Sparse shows cmt)
    (hD : Decomp H (RecDeploy.deployRec IV ivScope b) I.V Form.Sparse chk)
    (hLD : LogDecomp H (RecDeploy.deployRec IV ivScope b) I.V Form.Sparse logs)
    (hH : Held (RecDeploy.deployRec IV ivScope b) I L Form.Sparse) :
    VStarParts H (RecDeploy.deployRec IV ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) L opening
      Form.Sparse :=
  vstarParts_sparse_of_links rfl hC hO hL hS hD hLD hH

end Claims.Recursion.VStar
