import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Claims.Recursion.VStarPartsSparse

/-!
# V*'s parts at the sparse shape, at the compiled verifier

`deployRec` reads its inner view and its recorded calls through the readers it brings (`Brings.rd`), so at the readers
`⟨L.msgs, L.calls⟩` it reads through them by its definition (`readsThrough_deployRec`), and it stages one V* statement
per session (`vstarLen_deployRec`). There `VStarAcceptsR` at the sparse shape holds from the parts' links
(`PartLinks`), their openings (`PartOpens`), the staging (`Staged`) and the inner verifier's decomposition (`Decomp`):
`vstarAccepts_sparse_deployRec`. At the outer record
`deployZkRec relation`, the parser is the deployed one by definition (`vstarAccepts_sparse_deployZkRec`).
-/

namespace Claims.Recursion.VStar

/-- **`deployRec` reads through the readers it brings**: at `b.rd = ⟨L.msgs, L.calls⟩`, its inner view is `IV`'s run
of `L.msgs` and its recorded calls are each part's `L.calls`, by its definition. -/
theorem readsThrough_deployRec (H : Msg → Msg) (IV : Verifier FlockDeploy.Coin) (ivScope : Msg → Bool)
    (b : RecDeploy.Brings) (L : Layout) (hrd : b.rd = ⟨L.msgs, L.calls⟩) :
    ReadsThrough H (RecDeploy.deployRec IV ivScope b) IV L.msgs L.calls := by
  refine ⟨fun x r ws => ?_, fun x ws => ?_⟩
  · show RecDeploy.innerView H IV b.rd x r ws = _
    rw [hrd]
    rfl
  · show (ws.mapIdx fun s w => b.rd.calls x s w).flatten = _
    rw [hrd]

/-- **`deployRec`'s staging gives every session a V\* statement**: its V* statements are one per session
(`RecDeploy.vstar`). -/
theorem vstarLen_deployRec (IV : Verifier FlockDeploy.Coin) (ivScope : Msg → Bool) (b : RecDeploy.Brings) :
    VstarLen (RecDeploy.deployRec IV ivScope b) := fun x ms => by
  show RecDeploy.nS x ≤ (RecDeploy.vstar b.st x ms).length
  simp [RecDeploy.vstar]

variable {W Col T : Type}

/-- **`VStarAcceptsR` at the sparse shape, at the compiled verifier**: at readers `⟨L.msgs, L.calls⟩`
(`readsThrough_deployRec`) and the deployed outer parser, every part's links, its openings, the staging and the inner
verifier's decomposition give `VStarAcceptsR` at `Form.Sparse` (`vstarAccepts_sparse_of_links`, with
`vstarLen_deployRec`). -/
theorem vstarAccepts_sparse_deployRec {H : Msg → Msg} {IV : Verifier FlockDeploy.Coin} {ivScope : Msg → Bool}
    {b : RecDeploy.Brings} {I : Inner FlockDeploy.Coin W} {DO : Flock.Deploy FlockDeploy.Coin Col T} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read} (hrd : b.rd = ⟨L.msgs, L.calls⟩)
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hC : ∀ s, PartLinks H (RecDeploy.deployRec IV ivScope b) decode L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens H (RecDeploy.deployRec IV ivScope b) DO L Form.Sparse opening cmt s)
    (hS : Staged H (RecDeploy.deployRec IV ivScope b) I L Form.Sparse shows cmt)
    (hD : Decomp H (RecDeploy.deployRec IV ivScope b) IV Form.Sparse chk) :
    VStarAcceptsR H (RecDeploy.deployRec IV ivScope b) I DO Form.Sparse :=
  vstarAccepts_sparse_of_links hDO (readsThrough_deployRec H IV ivScope b L hrd) (vstarLen_deployRec IV ivScope b) hC
    hO hS hD

/-- **`VStarAcceptsR` at the sparse shape, at the compiled verifier and its outer record** `deployZkRec relation`,
whose parser is `parseOf relation readOfO` by its definition (`vstarAccepts_sparse_deployRec`). -/
theorem vstarAccepts_sparse_deployZkRec {H : Msg → Msg} {IV : Verifier FlockDeploy.Coin} {ivScope : Msg → Bool}
    {b : RecDeploy.Brings} {I : Inner FlockDeploy.Coin W} {L : Layout}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {opening : Opening} {cmt : Msg → ℕ → Msg → Msg}
    (hrd : b.rd = ⟨L.msgs, L.calls⟩)
    (hC : ∀ s, PartLinks H (RecDeploy.deployRec IV ivScope b) RecDeploy.readOfO L Form.Sparse shows chk s)
    (hO : ∀ s, PartOpens H (RecDeploy.deployRec IV ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      Form.Sparse opening cmt s)
    (hS : Staged H (RecDeploy.deployRec IV ivScope b) I L Form.Sparse shows cmt)
    (hD : Decomp H (RecDeploy.deployRec IV ivScope b) IV Form.Sparse chk) :
    VStarAcceptsR H (RecDeploy.deployRec IV ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation)
      Form.Sparse :=
  vstarAccepts_sparse_deployRec hrd rfl hC hO hS hD

end Claims.Recursion.VStar
