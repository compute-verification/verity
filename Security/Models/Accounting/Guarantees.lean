import Models.Accounting.Guarantees.Actual
import Models.Accounting.Guarantees.Bandwidth
import Models.Accounting.Guarantees.Bandwidth.Causal
import Models.Accounting.Guarantees.BandwidthAudit
import Models.Accounting.Guarantees.BoundReceipt
import Models.Accounting.Guarantees.BoundSketch
import Models.Accounting.Guarantees.Bridges
import Models.Accounting.Guarantees.Capacity
import Models.Accounting.Guarantees.Confidence
import Models.Accounting.Guarantees.Joint
import Models.Accounting.Guarantees.Localization
import Models.Accounting.Guarantees.Memory
import Models.Accounting.Guarantees.PerRun
import Models.Accounting.Guarantees.Pouw
import Models.Accounting.Guarantees.RateAudit
import Models.Accounting.Guarantees.RateAuditDeployed
import Models.Accounting.Guarantees.RateAuditHorizon
import Models.Accounting.Guarantees.RateAuditNamed
import Models.Accounting.Guarantees.RateAuditPartial
import Models.Accounting.Guarantees.RateAuditRemote
import Models.Accounting.Guarantees.RateAuditSketch
import Models.Accounting.Guarantees.Responder
import Models.Accounting.Guarantees.Remote
import Models.Accounting.Guarantees.Sketch
import Models.Accounting.Guarantees.Tag

/-! Resource accounting's guarantee statements: the joint guarantee and its covert residual, PoUW's and the bandwidth
audit's guarantees as a joint run's compute and bandwidth guarantees, a state-bounded game's soundness as its capacity
guarantee, derived state's share of the capacity residual, the rate audit's residency and bandwidth with the rebuild
charged as compute, with partial completion and with a sketch-then-hash tag, the remote rate audit's conservation over
tiers and chained passes, the causal horizon bound on its samples and its conservation with named words, the one
composed guarantee for the remote bandwidth audit as deployed and its deployed instance, the remote
bandwidth audit's conservation and node-level guarantees on the actual-circuit model, the actual circuit's capacity
theorem, runtime corollary, localization, composition and fit, the obligations B1, B2 and B4, the tag and sketch
guarantees, co-location and the link relay for receipts bound to the matmul, what a relay of the deployed bound
receipt's sketch must exchange, the bandwidth audit's guarantee, and their per-run forms. -/
