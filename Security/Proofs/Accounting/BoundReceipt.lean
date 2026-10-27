import Proofs.Accounting.BoundReceipt.Audit
import Proofs.Accounting.BoundReceipt.Interactive
import Proofs.Accounting.BoundReceipt.Tile
import Proofs.Accounting.BoundReceipt.Sketch

/-! The proofs for receipts bound to the matmul: co-location (`Accounting.SecurityProofs.Colocation`), the link relay
(`Accounting.SecurityProofs.BoundReceiptLink`), the same at `e > 0`, where the answers between the locations may depend
on the mix (`Accounting.SecurityProofs.BoundReceiptLinkInteractive`), and its share
(`Accounting.SecurityProofs.BoundReceiptShare`), and the tile's unpredictability below its work from PoUW's named
assumptions (`Accounting.SecurityProofs.TileUnpredictableOfTT`, `Accounting.SecurityProofs.TileUnpredictableOfNCPU`,
`Accounting.SecurityProofs.TileUnpredictablePearlCOfTTOut`) and over fewer bound entries
(`Accounting.SecurityProofs.TileUnpredictableMono`), and the deployed sketch's relay bound by Lindsey's lemma
(`Accounting.SecurityProofs.InnerProductRelay`, `Accounting.SecurityProofs.BoundSketchRelay`). -/
