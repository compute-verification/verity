module

public import Proofs.Poub.BoundReceipt.Witnesses
public import Proofs.Poub.BoundReceipt.Audit
public import Proofs.Poub.BoundReceipt.Interactive
public import Proofs.Poub.BoundReceipt.Tile
public import Proofs.Poub.BoundReceipt.Sketch

/-! The proofs for receipts bound to the matmul: co-location (`ResourceAccounting.SecurityProofs.Colocation`), the link relay
(`ResourceAccounting.SecurityProofs.BoundReceiptLink`), the same at `e > 0`, where the answers between the locations may depend
on the mix (`ResourceAccounting.SecurityProofs.BoundReceiptLinkInteractive`), and its share
(`ResourceAccounting.SecurityProofs.BoundReceiptShare`), and the tile's unpredictability below its work from PoUW's named
assumptions (`ResourceAccounting.SecurityProofs.TileUnpredictableOfTT`, `ResourceAccounting.SecurityProofs.TileUnpredictableOfNCPU`,
`ResourceAccounting.SecurityProofs.TileUnpredictablePearlCOfTTOut`) and over fewer bound entries
(`ResourceAccounting.SecurityProofs.TileUnpredictableMono`), and the deployed sketch's relay bound by Lindsey's lemma
(`ResourceAccounting.SecurityProofs.InnerProductRelay`, `ResourceAccounting.SecurityProofs.BoundSketchRelay`). -/

@[expose] public section
